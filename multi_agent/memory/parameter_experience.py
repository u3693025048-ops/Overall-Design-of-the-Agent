#!/usr/bin/env python3
"""
Dynamic Memory Buffer (Parameter Experience)
Implements trial-error-learning-reuse cycle
Supports short-term (current task) and long-term (validated best parameters) memory
"""

import logging
import shutil
import time
import json
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import OrderedDict

from multi_agent.metric_contract import (
    BW_MAX,
    BW_MIN,
    HIT_RATE_MIN,
    PEAK_NY_MAX,
    PM_MAX,
    PM_MIN,
    SEP_MAX,
)

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"


@dataclass
class MemoryEntry:
    memory_id: str
    memory_type: MemoryType
    task_context: Dict[str, Any]
    parameters: Dict[str, float]
    objectives: Dict[str, float]
    fitness: float
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_access: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ParameterExperience:
    def __init__(
        self,
        max_short_term_size: int = 100,
        max_long_term_size: int = 1000,
        max_episodic_size: int = 500,
        similarity_threshold: float = 0.7,
        decay_factor: float = 0.95,
        persist_path: str = "./parameter_experience_memory.json",
        experience_base_dir: str = "./parameter_experience_base",
    ):
        self.max_short_term_size = max_short_term_size
        self.max_long_term_size = max_long_term_size
        self.max_episodic_size = max_episodic_size
        self.similarity_threshold = similarity_threshold
        self.decay_factor = decay_factor
        self.persist_path = Path(persist_path)
        self.experience_base_dir = Path(experience_base_dir)

        # Create persistent base directory structure (tiered: short_term / long_term)
        for _tier in ("short_term", "long_term"):
            (self.experience_base_dir / "params" / _tier).mkdir(parents=True, exist_ok=True)
            (self.experience_base_dir / "models" / _tier).mkdir(parents=True, exist_ok=True)
        # Keep flat dirs for backward-compatibility with pre-tiering files
        (self.experience_base_dir / "params").mkdir(parents=True, exist_ok=True)
        (self.experience_base_dir / "models").mkdir(parents=True, exist_ok=True)

        self._short_term_memory: OrderedDict[str, MemoryEntry] = OrderedDict()
        self._long_term_memory: OrderedDict[str, MemoryEntry] = OrderedDict()
        self._episodic_memory: List[MemoryEntry] = []

        self._access_counts: Dict[str, int] = {}
        
        # Load persisted memory on initialization
        self.load()

    async def store(
        self,
        task_context: Dict[str, Any],
        parameters: Dict[str, float],
        objectives: Dict[str, float],
        fitness: float,
        metadata: Dict[str, Any] = None,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
    ) -> str:
        """Store a new experience in memory"""
        memory_id = self._generate_memory_id(task_context, parameters)

        entry = MemoryEntry(
            memory_id=memory_id,
            memory_type=memory_type,
            task_context=task_context,
            parameters=parameters,
            objectives=objectives,
            fitness=fitness,
            metadata=metadata or {},
        )

        if memory_type == MemoryType.SHORT_TERM:
            self._short_term_memory[memory_id] = entry
            self._evict_short_term_if_needed()
        elif memory_type == MemoryType.LONG_TERM:
            self._long_term_memory[memory_id] = entry
            self._evict_long_term_if_needed()
        else:
            self._episodic_memory.append(entry)

        logger.info(f"Stored memory {memory_id} with fitness {fitness}")
        
        # Save memory state and write to experience base directory
        self.save()
        self._write_experience_file(entry)
        return memory_id

    async def store_event(
        self,
        event_type: str,
        description: str,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Store a lightweight workflow event as an EPISODIC entry.

        Unlike ``store()``, no parameters / objectives / fitness are
        required.  Events are appended to ``_episodic_memory`` and used
        by ``get_recent_events()`` to build session-history context for
        the Hermes Step 0B system prompt.
        """
        ts_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        task_context = {
            "event_type":    event_type,
            "description":   description[:300],
            "timestamp_iso": ts_iso,
        }
        memory_id = self._generate_memory_id(
            task_context, {"_event": event_type, "_ts": ts_iso}
        )
        entry = MemoryEntry(
            memory_id=memory_id,
            memory_type=MemoryType.EPISODIC,
            task_context=task_context,
            parameters={},
            objectives={},
            fitness=0.0,
            metadata=metadata or {},
        )
        self._episodic_memory.append(entry)
        self._evict_episodic_if_needed()
        self.save()
        return memory_id

    def _tier_subdir(self, memory_type: MemoryType) -> str:
        """Return the subdirectory name (``short_term`` or ``long_term``) for a tier."""
        return "long_term" if memory_type == MemoryType.LONG_TERM else "short_term"

    def _move_entry_files(self, memory_id: str, from_tier: str, to_tier: str) -> None:
        """Move params and model files between tier subdirectories on promotion.

        Moves:
          - ``params/<from_tier>/<id[:16]>.json``  →  ``params/<to_tier>/``
          - ``models/<from_tier>/<id[:16]>_*``     →  ``models/<to_tier>/``
        """
        prefix = memory_id[:16]
        # params file
        src_p = self.experience_base_dir / "params" / from_tier / f"{prefix}.json"
        dst_p = self.experience_base_dir / "params" / to_tier   / f"{prefix}.json"
        if src_p.exists():
            try:
                dst_p.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_p), str(dst_p))
                logger.debug(f"Moved params: {src_p.name} → {to_tier}/")
            except Exception as exc:
                logger.warning(f"Could not move params file {src_p}: {exc}")
        # model files
        src_mdir = self.experience_base_dir / "models" / from_tier
        dst_mdir = self.experience_base_dir / "models" / to_tier
        if src_mdir.exists():
            for mf in src_mdir.glob(f"{prefix}_*"):
                try:
                    dst_mdir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(mf), str(dst_mdir / mf.name))
                    logger.debug(f"Moved model: {mf.name} → {to_tier}/")
                except Exception as exc:
                    logger.warning(f"Could not move model file {mf}: {exc}")

    def _write_experience_file(self, entry: MemoryEntry) -> None:
        """Write experience entry as a JSON file to the appropriate tier sub-directory.

        Path: ``params/<short_term|long_term>/<memory_id[:16]>.json``
        """
        try:
            tier      = self._tier_subdir(entry.memory_type)
            params_dir = self.experience_base_dir / "params" / tier
            params_dir.mkdir(parents=True, exist_ok=True)
            file_path = params_dir / f"{entry.memory_id[:16]}.json"
            def _numeric_fields(d: Dict[str, Any]) -> Dict[str, float]:
                out: Dict[str, float] = {}
                for k, v in (d or {}).items():
                    if str(k).startswith("_"):
                        continue
                    try:
                        out[k] = float(v)
                    except (TypeError, ValueError):
                        continue
                return out

            data = {
                "memory_id":   entry.memory_id,
                "memory_type": entry.memory_type.value,
                "task_context": entry.task_context,
                "parameters":  _numeric_fields(entry.parameters),
                "objectives":  _numeric_fields(entry.objectives),
                "fitness":     float(entry.fitness),
                "timestamp":   entry.timestamp,
                "metadata":    entry.metadata,
            }
            with open(file_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning(f"Failed to write experience file: {exc}")

    # Unicode characters that mark LLM annotation / verification lines
    _ANNOTATION_CHARS = frozenset((
        '\u2705', '\u274c', '\u26a0', '\u2714', '\u2716',
        '\U0001f50d', '\U0001f4dd',
    ))

    def save_successful_model(
        self, src_path: str, memory_id: str, suffix: str = ".m"
    ) -> Optional[str]:
        """Copy a successful model file to the appropriate tier sub-directory.

        Path: ``models/<short_term|long_term>/<memory_id[:16]>_<filename>``

        The tier is inferred from the in-memory dicts; defaults to
        ``short_term`` when the entry is not yet in either dict.

        For ``.m`` files: LLM verification/annotation lines (those whose first
        non-space character is one of ✅❌⚠️✔✖🔍📝) are stripped before saving.
        If the remaining content is empty the file is rejected and ``None`` is
        returned, preventing a corrupted all-annotation file from polluting the
        PE model store.
        """
        src = Path(src_path)
        if not src.exists():
            logger.warning(f"Model file not found: {src_path}")
            return None
        tier = (
            "long_term"
            if memory_id in self._long_term_memory
            else "short_term"
        )
        models_dir = self.experience_base_dir / "models" / tier
        models_dir.mkdir(parents=True, exist_ok=True)
        dst = models_dir / f"{memory_id[:16]}_{src.name}"
        try:
            if src.suffix.lower() == ".m":
                # Strip annotation lines from .m content before persisting
                raw = src.read_text(encoding="utf-8", errors="ignore")
                clean_lines = [
                    ln for ln in raw.splitlines()
                    if not (ln.lstrip() and ln.lstrip()[0] in self._ANNOTATION_CHARS)
                ]
                clean = "\n".join(clean_lines).strip()
                if not clean:
                    logger.warning(
                        f"Refusing to save '{src.name}' to PE models: content is "
                        f"empty after stripping annotation lines (all-annotation file)."
                    )
                    return None
                # Additional structural validation using the same heuristic as RL
                try:
                    from multi_agent.rl.matlab_rl_optimizer import _looks_like_matlab_script
                    _ok, _why = _looks_like_matlab_script(clean)
                    if not _ok:
                        logger.warning(
                            f"Refusing to save '{src.name}' to PE models: "
                            f"does not look like MATLAB ({_why})."
                        )
                        return None
                except ImportError:
                    pass  # validator not available; proceed without structural check
                dst.write_text(clean, encoding="utf-8")
            else:
                shutil.copy2(str(src), str(dst))
            logger.info(f"Saved successful model to: {dst}")
            # ── Update params JSON to record model_filename ──────────────────
            # This enables param → model lookup without scanning the models dir.
            try:
                tier      = "long_term" if memory_id in self._long_term_memory else "short_term"
                prefix    = memory_id[:16]
                json_paths = [
                    self.experience_base_dir / "params" / tier / f"{prefix}.json",
                    self.experience_base_dir / "params" / f"{prefix}.json",  # legacy flat
                ]
                for jp in json_paths:
                    if jp.exists():
                        try:
                            rec = json.loads(jp.read_text(encoding="utf-8"))
                            rec["model_filename"] = dst.name
                            rec["model_path"]     = str(dst)
                            jp.write_text(
                                json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
                            )
                            logger.debug(f"Updated params JSON with model_filename={dst.name}")
                        except Exception as je:
                            logger.debug(f"Could not update params JSON {jp}: {je}")
                        break
            except Exception as link_exc:
                logger.debug(f"model_filename link update failed: {link_exc}")
            return str(dst)
        except Exception as exc:
            logger.warning(f"Failed to copy model file: {exc}")
            return None

    @staticmethod
    def compute_fitness_from_objectives(
        objectives: Dict[str, float],
        hit_min:    float = HIT_RATE_MIN,
        sep_max:    float = SEP_MAX,
        peak_ny_max: float = PEAK_NY_MAX,
        pm_min:     float = PM_MIN,
        pm_max:     float = PM_MAX,
        bw_min:     float = BW_MIN,
        bw_max:     float = BW_MAX,
    ) -> float:
        """Compute a normalised fitness \u2208 [0, 1] from simulation objectives.

        Scoring per metric:
          hit_rate : (hit_rate - hit_min) / (100 - hit_min), clamped to [0,1]
          SEP      : max(0, 1 - sep / sep_max)
          peak_ny  : 1.0 if \u2264 peak_ny_max, else max(0, 1 - (ny - max)/max)
          PM       : 1.0 if in [pm_min, pm_max], else 0.5 * max(0, 1 - dist/pm_min)
          BW       : 1.0 if in [bw_min, bw_max], else 0.5 * max(0, 1 - dist/bw_min)
        Final fitness = weighted average.
        """
        o = objectives or {}
        hit  = float(o.get("hit_rate",    o.get("HitRate",   0.0)))
        sep  = float(o.get("SEP",         o.get("MissMean",  sep_max)))
        ny   = float(o.get("peak_ny",     o.get("PeakN",     0.0)))
        pm   = float(o.get("pitch_PM",    o.get("PM",        0.0)))
        bw   = float(o.get("pitch_BW",    o.get("BW",        0.0)))

        # hit_rate score
        rng_hit = max(100.0 - hit_min, 1.0)
        s_hit = max(0.0, min(1.0, (hit - hit_min) / rng_hit))

        # SEP score (lower = better)
        s_sep = max(0.0, 1.0 - sep / max(sep_max, 1.0))

        # PeakNy score (hard upper limit)
        if ny <= 0.0 or peak_ny_max <= 0.0:
            s_ny = 0.5
        elif ny <= peak_ny_max:
            s_ny = 1.0
        else:
            overshoot = (ny - peak_ny_max) / max(peak_ny_max, 1.0)
            s_ny = max(0.0, 1.0 - overshoot)

        # PM score (target range)
        if pm_min <= pm <= pm_max:
            s_pm = 1.0
        else:
            dist = max(pm_min - pm, pm - pm_max, 0.0)
            s_pm = max(0.0, 0.5 * (1.0 - dist / max(pm_min, 1.0)))

        # BW score (target range)
        if bw_min <= bw <= bw_max:
            s_bw = 1.0
        else:
            dist = max(bw_min - bw, bw - bw_max, 0.0)
            s_bw = max(0.0, 0.5 * (1.0 - dist / max(bw_min, 1.0)))

        # Weighted average: hit_rate dominates, SEP secondary, others equal
        fitness = (3.0*s_hit + 2.0*s_sep + 1.0*s_ny + 1.0*s_pm + 1.0*s_bw) / 8.0
        return round(float(fitness), 6)

    def get_recent_events(self, n: int = 5) -> str:
        """Return a formatted summary covering the 3 key event types.

        Always retrieves the **most recent** entry for each of the three
        event types injected into the Hermes Step 0B system prompt:
          • plan        — task-mode decision and reasoning
          • hermes_sim  — Hermes-internal simulation metrics
          • reflection  — reflection-agent judgement and suggestion

        Events are returned in chronological order (oldest first).
        The ``n`` parameter is kept for API compatibility but is no
        longer used; coverage is type-based, not count-based.
        """
        if not self._episodic_memory:
            return ""

        _target_types = ("plan", "hermes_sim", "reflection")
        latest: dict = {}
        for ev in self._episodic_memory:
            et = ev.task_context.get("event_type", "")
            if et in _target_types:
                latest[et] = ev

        ordered = sorted(latest.values(), key=lambda e: e.timestamp)
        lines = []
        for ev in ordered:
            ts   = ev.task_context.get("timestamp_iso", "")
            et   = ev.task_context.get("event_type", "event")
            desc = ev.task_context.get("description", "")
            lines.append(f"  [{ts}] {et}: {desc}")
        return "\n".join(lines)

    async def retrieve_similar(
        self,
        query: Dict[str, Any],
        top_k: int = 5,
        memory_type: Optional[MemoryType] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve similar experiences based on task context"""
        results = []

        search_sources = []
        if memory_type is None:
            search_sources = [self._short_term_memory, self._long_term_memory]
        elif memory_type == MemoryType.SHORT_TERM:
            search_sources = [self._short_term_memory]
        elif memory_type == MemoryType.LONG_TERM:
            search_sources = [self._long_term_memory]
        elif memory_type == MemoryType.EPISODIC:
            search_sources = [self._episodic_memory]

        for memory_dict in search_sources:
            if isinstance(memory_dict, list):
                entries = memory_dict
            else:
                entries = list(memory_dict.values())

            scored_entries = []
            for entry in entries:
                similarity = self._compute_similarity(query, entry.task_context)
                if similarity >= self.similarity_threshold:
                    decayed_fitness = entry.fitness * (
                        self.decay_factor**entry.access_count
                    )
                    scored_entries.append((entry, similarity, decayed_fitness))

            scored_entries.sort(key=lambda x: (x[1] * x[2]), reverse=True)

            for entry, similarity, decayed_fitness in scored_entries[:top_k]:
                entry.access_count += 1
                entry.last_access = time.time()
                quality = self.compute_retrieval_quality(
                    similarity, entry.fitness,
                    entry.memory_type.value, entry.access_count
                )
                results.append(
                    {
                        "memory_id":         entry.memory_id,
                        "parameters":        entry.parameters,
                        "objectives":        entry.objectives,
                        "fitness":           decayed_fitness,
                        "original_fitness":  entry.fitness,
                        "similarity":        similarity,
                        "quality_score":     quality["quality_score"],
                        "retrieval_quality": quality,
                        "timestamp":         entry.timestamp,
                        "memory_type":       entry.memory_type.value,
                    }
                )

        return results

    async def retrieve_best(
        self,
        task_context: Dict[str, Any],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve best performing experiences.

        Records with empty parameters are excluded — they have no
        reusable data for REUSE_HISTORY mode.
        """
        all_entries = [
            e for e in self._long_term_memory.values()
            if e.parameters and any(e.parameters.values())
        ]

        scored_entries = []
        for entry in all_entries:
            similarity = self._compute_similarity(task_context, entry.task_context)
            score = entry.fitness * similarity
            scored_entries.append((entry, similarity, score))

        scored_entries.sort(key=lambda x: x[2], reverse=True)

        results = []
        for entry, similarity, score in scored_entries[:top_k]:
            entry.access_count += 1
            entry.last_access = time.time()
            quality = self.compute_retrieval_quality(
                similarity, entry.fitness,
                entry.memory_type.value, entry.access_count
            )
            results.append(
                {
                    "memory_id":        entry.memory_id,
                    "parameters":       entry.parameters,
                    "objectives":       entry.objectives,
                    "fitness":          entry.fitness,
                    "similarity":       similarity,
                    "combined_score":   score,
                    "quality_score":    quality["quality_score"],
                    "retrieval_quality": quality,
                    "timestamp":        entry.timestamp,
                    "memory_type":      entry.memory_type.value,
                }
            )

        return results

    async def promote_to_long_term(self, memory_id: str) -> bool:
        """Promote short-term memory to long-term.

        Also moves associated params and model files from
        ``params/short_term/`` and ``models/short_term/`` to the
        corresponding ``long_term/`` sub-directories.
        """
        if memory_id in self._short_term_memory:
            entry = self._short_term_memory.pop(memory_id)
            entry.memory_type = MemoryType.LONG_TERM
            self._long_term_memory[memory_id] = entry
            self._evict_long_term_if_needed()
            self._move_entry_files(memory_id, "short_term", "long_term")
            logger.info(f"Promoted memory {memory_id} to long-term")
            self.save()
            return True
        return False

    async def auto_promote(self, min_fitness: float = 0.4) -> int:
        """Promote SHORT_TERM entries to LONG_TERM using a composite score.

        Composite promote score:
            score = fitness + 0.05 × min(access_count, 5)

        Frequently-accessed entries receive a small bonus (max +0.25) so
        records referenced multiple times during the session are more
        likely to survive into long-term memory even when their raw
        fitness is just below the threshold.

        Returns the number of entries promoted.
        """
        to_promote = [
            mid for mid, entry in list(self._short_term_memory.items())
            if (entry.fitness + 0.05 * min(entry.access_count, 5)) >= min_fitness
        ]
        for mid in to_promote:
            await self.promote_to_long_term(mid)
        if to_promote:
            logger.info(
                f"auto_promote: {len(to_promote)} SHORT_TERM entries promoted to "
                f"LONG_TERM (composite_score >= {min_fitness:.3f})"
            )
        return len(to_promote)

    async def promote_top_k(self, k: int = 3) -> int:
        """Promote the top-*k* highest-fitness SHORT_TERM entries to LONG_TERM.

        Unlike ``auto_promote()``, no fitness threshold is applied — the
        best *k* short-term entries are always promoted regardless of
        their raw fitness.  Useful for manual promotion when the user
        wants to retain the session’s best results even if they fall
        below the auto-promote threshold.

        Returns the number of entries actually promoted.
        """
        if not self._short_term_memory:
            return 0
        sorted_ids = sorted(
            self._short_term_memory.keys(),
            key=lambda mid: self._short_term_memory[mid].fitness,
            reverse=True,
        )[:k]
        for mid in sorted_ids:
            await self.promote_to_long_term(mid)
        if sorted_ids:
            logger.info(
                f"promote_top_k: promoted top-{len(sorted_ids)} SHORT_TERM "
                "entries to LONG_TERM"
            )
        return len(sorted_ids)

    def get_short_term_summary(self) -> List[Dict[str, Any]]:
        """Return a summary of all SHORT_TERM entries, sorted by fitness descending.

        Used by the manual-promotion display in the human-interrupt
        handler so the user can see which short-term records are
        available to promote to long-term.
        """
        entries = sorted(
            self._short_term_memory.values(),
            key=lambda e: e.fitness,
            reverse=True,
        )
        return [
            {
                "memory_id":    e.memory_id[:12],
                "fitness":      round(e.fitness, 4),
                "access_count": e.access_count,
                "timestamp_iso": time.strftime(
                    "%m-%d %H:%M", time.localtime(e.timestamp)
                ),
                "objectives": {
                    k: round(float(v), 3)
                    for k, v in list(e.objectives.items())[:4]
                },
            }
            for e in entries
        ]

    async def get_statistics(self) -> Dict[str, Any]:
        """Get memory buffer statistics"""
        short_term_avg_fitness = 0.0
        if self._short_term_memory:
            short_term_avg_fitness = sum(
                e.fitness for e in self._short_term_memory.values()
            ) / len(self._short_term_memory)

        long_term_avg_fitness = 0.0
        if self._long_term_memory:
            long_term_avg_fitness = sum(
                e.fitness for e in self._long_term_memory.values()
            ) / len(self._long_term_memory)

        return {
            "short_term_count": len(self._short_term_memory),
            "long_term_count": len(self._long_term_memory),
            "episodic_count": len(self._episodic_memory),
            "short_term_avg_fitness": short_term_avg_fitness,
            "long_term_avg_fitness": long_term_avg_fitness,
            "total_accesses": sum(self._access_counts.values()),
        }

    def _generate_memory_id(
        self, task_context: Dict[str, Any], parameters: Dict[str, float]
    ) -> str:
        """Generate unique memory ID"""
        content = json.dumps(
            {
                "context": task_context,
                "params": parameters,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _compute_similarity(
        self, query: Dict[str, Any], stored: Dict[str, Any]
    ) -> float:
        """Compute similarity between query and stored context"""
        if not query or not stored:
            return 0.0

        common_keys = set(query.keys()) & set(stored.keys())
        if not common_keys:
            return 0.0

        similarities = []
        for key in common_keys:
            q_val = query[key]
            s_val = stored[key]

            if isinstance(q_val, (int, float)) and isinstance(s_val, (int, float)):
                max_val = max(abs(q_val), abs(s_val), 1e-6)
                similarity = 1.0 - abs(q_val - s_val) / max_val
            elif isinstance(q_val, str) and isinstance(s_val, str):
                similarity = 1.0 if q_val == s_val else 0.0
            elif q_val == s_val:
                similarity = 1.0
            else:
                similarity = 0.0

            similarities.append(similarity)

        return sum(similarities) / len(similarities) if similarities else 0.0

    @staticmethod
    def compute_retrieval_quality(
        similarity: float,
        fitness: float,
        memory_type_str: str,
        access_count: int = 0,
    ) -> Dict[str, Any]:
        """
        检索质量评分 (Retrieval Quality Score).

        Q = 0.40 × similarity + 0.40 × fitness + 0.20 × reliability

        - similarity    : 上下文匹配度 [0, 1]
        - fitness       : 存储的四指标适应度 [0, 1]
        - reliability   : LONG_TERM = 1.0, SHORT_TERM = 0.6
        - access_count  : 已被检索次数（越大说明该条经验越常被引用）

        Returns dict with component scores and composite Q ∈ [0, 1].
        Grade: Q ≥ 0.75 高可信 | 0.50–0.75 中等 | < 0.50 低可信
        """
        reliability = 1.0 if memory_type_str == "long_term" else 0.6
        Q = 0.40 * float(similarity) + 0.40 * float(fitness) + 0.20 * reliability
        Q = max(0.0, min(1.0, Q))
        if Q >= 0.75:
            grade = "高可信"
        elif Q >= 0.50:
            grade = "中等"
        else:
            grade = "低可信"
        return {
            "quality_score":  round(Q, 4),
            "sim_score":      round(float(similarity), 4),
            "fitness_score":  round(float(fitness), 4),
            "reliability":    round(reliability, 2),
            "memory_type":    memory_type_str,
            "access_count":   int(access_count),
            "grade":          grade,
        }

    def _cleanup_entry_files(self, memory_id: str) -> None:
        """Delete all on-disk files associated with a memory entry.

        Searches every possible location so that both pre-tiering (flat)
        and tiered files are handled correctly:
          - ``params/<id[:16]>.json``                (legacy flat)
          - ``params/short_term/<id[:16]>.json``
          - ``params/long_term/<id[:16]>.json``
          - ``models/<id[:16]>_*``                   (legacy flat)
          - ``models/short_term/<id[:16]>_*``
          - ``models/long_term/<id[:16]>_*``
        """
        prefix = memory_id[:16]
        # --- params files ---
        params_candidates = [
            self.experience_base_dir / "params" / f"{prefix}.json",
            self.experience_base_dir / "params" / "short_term" / f"{prefix}.json",
            self.experience_base_dir / "params" / "long_term"  / f"{prefix}.json",
        ]
        for pf in params_candidates:
            if pf.exists():
                try:
                    pf.unlink()
                    logger.debug(f"Deleted params file: {pf}")
                except Exception as exc:
                    logger.warning(f"Could not delete params file {pf}: {exc}")
        # --- model files ---
        models_search_dirs = [
            self.experience_base_dir / "models",
            self.experience_base_dir / "models" / "short_term",
            self.experience_base_dir / "models" / "long_term",
        ]
        for mdir in models_search_dirs:
            if mdir.exists():
                for mf in mdir.glob(f"{prefix}_*"):
                    try:
                        mf.unlink()
                        logger.debug(f"Deleted model file: {mf}")
                    except Exception as exc:
                        logger.warning(f"Could not delete model file {mf}: {exc}")

    def _evict_short_term_if_needed(self) -> None:
        """Evict the lowest-fitness SHORT_TERM entry when capacity is exceeded.

        Replaced FIFO (popitem) with fitness-based eviction so that
        high-quality older entries are not displaced by low-quality new
        arrivals.  Also cleans up the associated on-disk files.
        """
        while len(self._short_term_memory) > self.max_short_term_size:
            worst_id = min(
                self._short_term_memory.keys(),
                key=lambda mid: self._short_term_memory[mid].fitness,
            )
            self._short_term_memory.pop(worst_id)
            self._cleanup_entry_files(worst_id)
            logger.debug(f"Evicted SHORT_TERM entry {worst_id} (lowest fitness)")

    def _evict_long_term_if_needed(self) -> None:
        """Evict the lowest-scored LONG_TERM entry when capacity is exceeded.

        Eviction score = 0.7 × fitness + 0.3 × recency, where:
            recency = 1 / (1 + 0.05 × days_idle)
        days_idle is measured from last_access (falls back to creation
        timestamp when never accessed).  This prevents stale entries that
        have never been re-used from permanently occupying long-term slots.
        """
        now = time.time()
        while len(self._long_term_memory) > self.max_long_term_size:
            if not self._long_term_memory:
                break

            def _score(mid: str) -> float:
                e = self._long_term_memory[mid]
                ref_time = e.last_access if e.last_access > 0 else e.timestamp
                days_idle = max(0.0, (now - ref_time) / 86400)
                recency = 1.0 / (1.0 + 0.05 * days_idle)
                return 0.7 * e.fitness + 0.3 * recency

            worst_id = min(self._long_term_memory.keys(), key=_score)
            self._long_term_memory.pop(worst_id)
            self._cleanup_entry_files(worst_id)
            logger.debug(f"Evicted LONG_TERM entry {worst_id} (lowest fitness+recency)")

    def _evict_episodic_if_needed(self) -> None:
        """Evict the oldest EPISODIC events (FIFO) when capacity exceeds max_episodic_size."""
        while len(self._episodic_memory) > self.max_episodic_size:
            self._episodic_memory.pop(0)

    def clear(self, memory_type: Optional[MemoryType] = None) -> None:
        """Clear memory of specified type or all memory"""
        if memory_type is None:
            self._short_term_memory.clear()
            self._long_term_memory.clear()
            self._episodic_memory.clear()
            self._access_counts.clear()
        elif memory_type == MemoryType.SHORT_TERM:
            self._short_term_memory.clear()
        elif memory_type == MemoryType.LONG_TERM:
            self._long_term_memory.clear()
        elif memory_type == MemoryType.EPISODIC:
            self._episodic_memory.clear()
            
        # Always persist clear operations
        self.save()

    def save(self) -> bool:
        """Persist memory to disk"""
        try:
            # Ensure directory exists
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            
            def serialize_entry(entry: MemoryEntry) -> Dict:
                data = asdict(entry)
                data["memory_type"] = entry.memory_type.value
                return data
                
            data = {
                "short_term": [serialize_entry(e) for e in self._short_term_memory.values()],
                "long_term": [serialize_entry(e) for e in self._long_term_memory.values()],
                "episodic": [serialize_entry(e) for e in self._episodic_memory],
                "access_counts": self._access_counts
            }
            
            with open(self.persist_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Successfully saved Parameter Experience memory to {self.persist_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save Parameter Experience memory to disk: {e}")
            return False

    def _rebuild_from_base_dir(self) -> int:
        """Rebuild in-memory entries from individual JSON files in experience_base_dir.

        Scans ``params/long_term/*.json`` and ``params/short_term/*.json``,
        deserialises each file (same format as ``_write_experience_file``),
        and inserts the entry into the corresponding in-memory dict.

        Returns the number of entries recovered.
        """
        recovered = 0
        for tier, mem_dict, mem_type in [
            ("long_term",  self._long_term_memory,  MemoryType.LONG_TERM),
            ("short_term", self._short_term_memory, MemoryType.SHORT_TERM),
        ]:
            tier_dir = self.experience_base_dir / "params" / tier
            if not tier_dir.exists():
                continue
            for jf in sorted(tier_dir.glob("*.json")):
                try:
                    raw = json.loads(jf.read_text(encoding="utf-8"))
                    mid = raw.get("memory_id", "")
                    if not mid or mid in mem_dict:
                        continue
                    entry = MemoryEntry(
                        memory_id=mid,
                        memory_type=mem_type,
                        task_context=raw.get("task_context", {}),
                        parameters=raw.get("parameters", {}),
                        objectives=raw.get("objectives", {}),
                        fitness=float(raw.get("fitness", 0.0)),
                        timestamp=float(raw.get("timestamp", 0.0)),
                        access_count=int(raw.get("access_count", 0)),
                        last_access=float(raw.get("last_access", 0.0)),
                        metadata=raw.get("metadata", {}),
                    )
                    mem_dict[mid] = entry
                    recovered += 1
                except Exception as exc:
                    logger.debug(f"Skipping {jf.name}: {exc}")
        if recovered:
            logger.info(
                f"[PE] Rebuilt {recovered} entries from {self.experience_base_dir}/params/ "
                f"(long_term={len(self._long_term_memory)}, "
                f"short_term={len(self._short_term_memory)})"
            )
            self.save()
        return recovered

    def load(self) -> bool:
        """Load memory from disk"""
        if not self.persist_path.exists():
            return False
            
        try:
            with open(self.persist_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            def deserialize_entry(entry_dict: Dict) -> MemoryEntry:
                entry_dict["memory_type"] = MemoryType(entry_dict["memory_type"])
                return MemoryEntry(**entry_dict)
                
            # Clear existing memory before loading
            self.clear()
            
            # Load short-term
            for item in data.get("short_term", []):
                entry = deserialize_entry(item)
                self._short_term_memory[entry.memory_id] = entry
                
            # Load long-term
            for item in data.get("long_term", []):
                entry = deserialize_entry(item)
                self._long_term_memory[entry.memory_id] = entry
                
            # Load episodic
            self._episodic_memory = [deserialize_entry(item) for item in data.get("episodic", [])]
            
            # Load access counts
            self._access_counts = data.get("access_counts", {})

            # ── Auto-recovery ────────────────────────────────────────────────
            # If the main JSON has no long_term/short_term entries but the
            # base directory has individual JSON files, rebuild from them.
            # This handles the case where the main JSON was cleared/corrupted
            # but the individual experience files are still intact.
            if not self._long_term_memory and not self._short_term_memory:
                n = self._rebuild_from_base_dir()
                if n:
                    logger.info(f"[PE] Auto-recovered {n} entries from base directory")
            
            logger.info(f"Successfully loaded Parameter Experience memory from {self.persist_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load Parameter Experience memory from disk: {e}")
            return False
