# Response to Reviewer #1 (second round)

Manuscript: aerospace-4439606  
We thank the reviewer. Comments 1, 3, 4 and 5 are addressed below. Code and
data are provided as a supplementary ZIP archive with this revision (not via
GitHub). Comment 2 is not addressed in this response.

---

## Comment 1 — Equal simulation budgets

All methods are now evaluated under the same **maximum** number of development
Monte Carlo (MC) calls.

The previous protocol counted CMA-ES *generations* against the integer used as
Random Search / Direct PPO *candidate evaluations*, so CMA-ES received
λ = 10 times more 100-run batches per “iteration”. The shared ceiling is

    max_candidates = 5 × 15 = 75
    max_mc_calls   = 75 × 3 development seeds = 225

matching `workflow.max_iterations × matlab_rl_optimizer.max_episodes × |S_dev|`.
CMA-ES stops when this candidate count is exhausted; a truncated final
generation is evaluated but does not update the covariance. Code:
`experiments/equal_mc_budget.py`. Section 4.1 of the manuscript states the
protocol.

This is an equal *budget ceiling*, not a requirement that every run consume
exactly 225 calls, and not a claim that tabulated MC-call integers are in
identical units. Search uses **10 trajectories** per MATLAB call; the
100-trajectory batch is reserved for test. Random Search / CMA-ES / Direct PPO
count three real MATLAB seeds per candidate (75 × 3 = 225 at the cap).
GSMultiAgent’s published 1–22 counts are outer-loop derived and typically run
one real seed with bootstrap fill of the other two.

Memory reuse (REUSE_HISTORY, after earlier scenarios on the same master seed)
and constraint-satisfied early stop may use fewer recorded calls. No method is
allowed a larger ceiling than 225.

First success in Table 5 for GSMultiAgent, the standalone optimizers, and
Single-Agent Hermes means **final development-set feasibility**. It is not
“success on the first outer iteration.” Single-Agent was realigned to search
width 10 (test still 100); recorded first success is **12/12**, matching
development and test feasibility.

Post-hoc first feasible MC calls (JSON `history`, not the 225 cap): Random
Search mean **73.9** (8/12 found), CMA-ES **79.2** (12/12), Direct PPO **54.0**
(11/12). `results/experiment_plan/optimizer_equal_mc_75/first_success_mc_calls.csv`.

Under this ceiling, on the 12 scenario × master-seed cells, development/test
feasibility is:

| Method | First success | Dev feasible | Test feasible | Recorded dev MC |
|--------|---------------|--------------|---------------|-----------------|
| GSMultiAgent | 9/12 | 9/12 | 5/12 | 1–22 (outer-derived) |
| Random Search | 8/12 | 8/12 | 8/12 | 225 (75×3 MATLAB) |
| CMA-ES | 12/12 | 12/12 | 12/12 | 225 (75×3 MATLAB) |
| Direct PPO | 11/12 | 11/12 | 11/12 | 225 (75×3 MATLAB) |
| Single-Agent | 12/12 | 12/12 | 12/12 | 135–225 (search 10; 3 real MATLAB seeds) |

The superseded unequal-budget comparison (Random Search / Direct PPO at 0/12)
has been removed from the tables and discussion. Raw runs are in
`results/experiment_plan/optimizer_equal_mc_75/` and
`optimizer_equal_mc_75_sah_n10/`; the pooled table is
`results/experiment_plan/table5_equal_mc.csv`. Table 6 means are in
`results/experiment_plan/table6_equal_mc.csv`.

---

## Comment 3 — Abstract (75% versus 25%; “unique global Pareto optimality”)

The abstract no longer reports the superseded 75% versus 25% first-attempt
comparison, nor the unsupported claim of “unique global Pareto optimality”.
It now states the revised development- and test-set feasibility rates with
the confidence intervals of Comment 4. The Discussion likewise states that we
do not claim unique global Pareto optimality.

---

## Comment 4 — Confidence intervals for 9/12 and 5/12

Clopper–Pearson exact 95% intervals for the 12 scenario × master-seed cases:

- Development feasibility 9/12 = 75.0% (95% CI 42.8%–94.5%)
- Test feasibility 5/12 = 41.7% (95% CI 15.2%–72.3%)

These intervals are reported in the abstract and in Section 4.2. The 12 cells
are the GSMultiAgent E0 (full-memory) runs; failed cells are T1/seed 42,
T3/seed 44 and T4/seed 44.

---

## Comment 5 — Public GitHub; model versions; experimental dates

We do not host a public GitHub repository. Source code, the Monte Carlo
initial-condition generator, the protocol files, the simulated engagement
`.mat` files, and the equal-budget run archives are submitted as
**Supplementary Material** (ZIP) with the revised manuscript. The inventory
is given in the next section.

The archived engagement files are **simulated** from published Sparrow-class
(AIM-7F) parameters and fixed random seeds
(`experiments/mc_data_generator.py`). They are not recorded flight data.
Post-hoc resampled CSV rows (`*_incl_simulated.csv`) are excluded from the
archive. GSMultiAgent tabulated AND used runtime bootstrap fill (one real
MATLAB seed, two filled); Random Search / CMA-ES / Direct PPO / Single-Agent
used three real MATLAB seeds per candidate. Those tabulated MC-call integers
are therefore not in identical units (Section 4.1).

Exact API model identifiers, as sent in the request body:

| Role | API model ID |
|------|----------------|
| Main foundation model | `deepseek-v4-pro` |
| Model-substitution | `qwen3.7-max`, `kimi-k3` |
| Embeddings | `text-embedding-v3` |

Software: GSMultiAgent ablations — MATLAB R2025a on an i7-13700K workstation.
Equal-budget baselines and Single-Agent — MATLAB R2020b on Ubuntu 24.04.

Experimental dates (UTC+8):

- GSMultiAgent ablations, memory/optimization/correction, model substitution:
  **2–12 August 2026**
- Equal-budget baselines (Random Search, CMA-ES, Direct PPO):
  **20–22 August 2026**
- Single-Agent Hermes n10 realignment: **22–23 August 2026**

The same information is listed in `REPRODUCIBILITY.md` inside the ZIP.

---

## List of attachments submitted with this revision

| # | File | Role |
|---|------|------|
| A | Revised manuscript (`papers.docx`) | Main text, including Comments 1, 3, 4 and 5 |
| B | This point-by-point response | The present letter |
| C | `GSMultiAgent_supplement.zip` | Supplementary code, data, and equal-budget results (not a public GitHub repository) |

### Contents of attachment C (`GSMultiAgent_supplement.zip`)

Compressed size about 7.9 MB; 344 files after decompression. API keys are
replaced by `YOUR_API_KEY`. GSMultiAgent runs used `mc_and_bootstrap_fill`
(one real MATLAB seed, two filled); the standalone optimizers and Single-Agent
use three real MATLAB seeds per candidate.

| Path in the ZIP | What it is |
|-----------------|------------|
| `cli_agent.py`, `run_rl_optimize.py`, `config.yaml`, `requirements.txt`, `prompt.txt` | Runnable entry points and de-identified configuration |
| `REPRODUCIBILITY.md` | Model IDs, MATLAB version, dates, 225-call budget |
| `SUPPLEMENT.md`, `提交说明.md`, `打包文件清单.md` | English/Chinese notes on what is included and excluded |
| `multi_agent/` | Framework source (agents, PPO, MATLAB tools, memory) |
| `experiments/` | Protocol, `equal_mc_budget.py`, `mc_data_generator.py`, baselines, Single-Agent |
| `tests/` | Unit tests, including the equal-budget protocol |
| `knowledge_base/matlab/guidance/monte_carlo_single.m` | Canonical MATLAB Monte Carlo template |
| `data/mc/` | 72 simulated engagement `.mat` files (T1–T4 × masters 42/43/44 × 6 splits) plus `seed_map.json` |
| `results/experiment_plan/optimizer_equal_mc_75/` | 36 equal-budget JSON runs + `runs.csv` (Random Search, CMA-ES, Direct PPO) |
| `results/experiment_plan/optimizer_equal_mc_75_sah_n10/` | 12 equal-budget Single-Agent JSON runs + `runs.csv` (search 10 / test 100) |
| `results/experiment_plan/gsm_all_seeds.csv` | One table with every GSMultiAgent ablation row (E0 9/12 and 5/12) |
| `results/experiment_plan/table5_equal_mc.csv` | Pooled Table 5 rates (8/12, 12/12, 11/12, …) |
| `results/experiment_plan/ablation_*.csv` | Memory / optimization / correction / order ablations (real runs only) |

**Intentionally not in the ZIP:** GitHub URL; API keys; `scripts/` (local
operations); full run archives and logs; vector-database cache (`chroma_db/`);
bootstrap-filled seeds; post-hoc `*_incl_simulated.csv` rows; over-budget
Single-Agent backups.

A reviewer can regenerate `data/mc/*.mat` with

```
python experiments/mc_data_generator.py --output-root data/mc --n-runs 100 --no-broadside
```

using the same master seeds, and can re-run the equal-budget baselines with
the commands in `experiments/README.md`.
