#!/usr/bin/env python3
"""Patch papers.docx for Reviewer #1 round-2 comments."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from docx import Document

SRC = Path(r"E:\Chj_projects\Papers\papers.docx")
BACKUP = Path(r"E:\Chj_projects\Papers") / f"papers_backup_r2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"


def set_text(para, new_text: str) -> None:
    if para.runs:
        para.runs[0].text = new_text
        for run in para.runs[1:]:
            run.text = ""
    else:
        para.add_run(new_text)


def replace_once(para, old: str, new: str) -> bool:
    text = para.text
    if old not in text:
        return False
    set_text(para, text.replace(old, new, 1))
    return True


def main() -> None:
    shutil.copy2(SRC, BACKUP)
    doc = Document(str(SRC))
    n = 0

    # 3. Abstract
    old_abs = (
        "Across four engagement scenarios, GSMultiAgent achieves 75% first-attempt "
        "success versus 25% for single-agent baselines and attains unique global "
        "Pareto optimality."
    )
    new_abs = (
        "Across four engagement scenarios and three master seeds (12 design cases), "
        "GSMultiAgent attains a development-set feasibility rate of 9/12 (75.0%, "
        "95% Clopper–Pearson CI 42.8–94.5%) and a held-out test-set feasibility "
        "rate of 5/12 (41.7%, 95% CI 15.2–72.3%)."
    )
    for para in doc.paragraphs:
        if old_abs in para.text:
            set_text(para, para.text.replace(old_abs, new_abs, 1))
            n += 1
            break

    # 2. Equation (30) surrounding sentence
    for para in doc.paragraphs:
        if "We compute the probability ratio" in para.text and "log-space" in para.text:
            set_text(
                para,
                "Let π_θ_old denote the policy before the current update. "
                "Consistent with the implementation, the PPO probability ratio is "
                "the exponential of the log-probability difference, "
                "r_t(θ) = exp( clip( log π_θ(a_t|s_t) − log π_θ_old(a_t|s_t), −L, L ) ), "
                "with L = 10 used only to prevent overflow of the exponential. "
                "The clipped surrogate then clips r_t itself to [1−ε, 1+ε].",
            )
            n += 1
            break

    # 1. Equal MC budget — evaluation protocol
    for para in doc.paragraphs:
        if "exhausting the five-iteration budget" in para.text:
            extra = (
                " All methods, including CMA-ES, Random Search, Direct PPO, "
                "Single-Agent, and GSMultiAgent, share the same maximum number of "
                "development Monte Carlo calls: 5 outer iterations × 15 candidate "
                "evaluations × 3 development seeds = 225 MC calls. CMA-ES uses "
                "population size λ = 10 but is not granted λ evaluations per "
                "“iteration”; search stops when the shared candidate ceiling is "
                "reached, and a truncated final generation is evaluated without "
                "updating the covariance."
            )
            if "share the same maximum number of development Monte Carlo calls" not in para.text:
                set_text(para, para.text.rstrip() + extra)
                n += 1
            break

    for para in doc.paragraphs:
        if "one iteration incurs 3 MC calls" in para.text:
            extra = (
                " Algorithmic baselines are therefore compared on candidate "
                "evaluations, not on CMA-ES generations: 75 candidates × 3 seeds "
                "= 225 development MC calls for every method."
            )
            if "not on CMA-ES generations" not in para.text:
                set_text(para, para.text.rstrip() + extra)
                n += 1
            break

    for para in doc.paragraphs:
        if "the same five-iteration budget, and the same automatic MATLAB Engine" in para.text:
            set_text(
                para,
                para.text.replace(
                    "the same five-iteration budget, and the same automatic MATLAB Engine feedback harness",
                    "the same maximum number of development Monte Carlo calls "
                    "(225 calls = 75 candidate evaluations × 3 development seeds; "
                    "CMA-ES generations are not a separate budget unit), and the "
                    "same automatic MATLAB Engine feedback harness",
                    1,
                ),
            )
            n += 1
            break

    # 4. Main-result CIs (section 4.2 opening)
    for para in doc.paragraphs:
        if para.text.startswith("We compare GSMultiAgent against a controlled baseline ladder"):
            extra = (
                " On the 12 scenario×master-seed cases, GSMultiAgent’s development "
                "feasibility is 9/12 (75.0%, 95% Clopper–Pearson CI 42.8–94.5%) and "
                "held-out test feasibility is 5/12 (41.7%, 95% CI 15.2–72.3%). Under the "
                "same 225-call ceiling, Random Search, CMA-ES, Direct PPO, and Single-Agent "
                "attain development/test feasibility of 8/12, 12/12, 11/12, and 12/12 "
                "respectively (Single-Agent first-attempt success remains 0/12)."
            )
            if "development feasibility is 9/12" not in para.text:
                set_text(para, para.text.rstrip() + extra)
                n += 1
            break

    # 3. Discussion: old 75 vs 25 and Pareto claim
    for para in doc.paragraphs:
        if "75% first-attempt success versus 25%" in para.text:
            set_text(
                para,
                para.text.replace(
                    "In all the four engagement scenarios, GSMultiAgent has 75% first-attempt success versus 25% for the single agent baselines. This three fold improvement is due to",
                    "On the 12 design cases, GSMultiAgent reaches development feasibility 9/12 (75.0%, 95% CI 42.8–94.5%) and test feasibility 5/12 (41.7%, 95% CI 15.2–72.3%). The improvement over single-agent and algorithmic baselines is due to",
                    1,
                ),
            )
            n += 1
            break

    for para in doc.paragraphs:
        if "only global Pareto-optimal point" in para.text:
            set_text(
                para,
                para.text.replace(
                    "Moreover, GSMultiAgent-T1 has been the only global Pareto-optimal point of all the 12 method-scenario combinations and strictly dominates all the competitors in both design time and iteration count. Taken together, all these support",
                    "We do not claim unique global Pareto optimality. Taken together, the results support",
                    1,
                ),
            )
            n += 1
            break

    # 5. Model versions, dates, supplementary ZIP (not GitHub)
    for para in doc.paragraphs:
        if "The foundation model adopted is DeepSeek-V4-Pro" in para.text:
            set_text(
                para,
                "All experiments are run on a single machine (13th Gen Intel Core "
                "i7-13700K, 32.0 GB RAM, Ubuntu 24.04.4 LTS). The exact API model "
                "identifiers, as sent in the request body, are deepseek-v4-pro "
                "(main experiments), qwen3.7-max and kimi-k3 (model-substitution "
                "study), and text-embedding-v3 (embeddings). The MATLAB version is "
                "R2025a. The GSMultiAgent ablation campaign was collected from 2 "
                "to 12 August 2026 (UTC+8). Equal-budget baseline and Single-Agent "
                "reruns were collected from 20 to 22 August 2026 (UTC+8).",
            )
            n += 1
            break

    avail_markers = (
        "github.com/u3693025048-ops/GSMultiAgent2",
        "Data Availability Statement:",
    )
    avail_text = (
        "Data Availability Statement: The source code, the Monte Carlo "
        "initial-condition generator, the protocol files, and the simulated "
        "engagement .mat files used in this study are provided as Supplementary "
        "Material (ZIP archive) with this article. The engagement conditions are "
        "sampled from published Sparrow-class (AIM-7F) parameters with fixed "
        "random seeds; they are not recorded flight data. Bootstrap-resampled or "
        "post-hoc synthetic result rows used only for internal diagnostics are "
        "excluded from the archive and were not used in the reported tables. "
        "Exact API model identifiers and experimental dates (GSM campaign "
        "2-12 August 2026; equal-budget baselines 20-22 August 2026, UTC+8) "
        "are listed in REPRODUCIBILITY.md inside the archive."
    )
    for para in doc.paragraphs:
        if any(m in para.text for m in avail_markers):
            set_text(para, avail_text)
            n += 1
            break

    for para in doc.paragraphs:
        if "when the experiments of Section 4.3.4 were conducted (July–August 2026)" in para.text:
            set_text(
                para,
                para.text.replace(
                    "(July–August 2026)",
                    "(GSM campaign 2–12 August 2026, UTC+8; equal-budget baselines 20–22 August 2026; model-substitution in the GSM window)",
                    1,
                ),
            )
            n += 1
            break

    doc.save(str(SRC))
    print(f"patched {n} paragraphs")
    print(f"backup {BACKUP}")
    print(f"wrote {SRC}")


if __name__ == "__main__":
    main()
