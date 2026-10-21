# Round-1 fresh-agent audit — embodied-offpolicy-study (2026-10-22)

Independent read-then-fix review on HEAD `6b0fac0`. Environment: Windows 11,
Python 3.13.16, no C compiler; pure-wheel venv. Full pytest output:
`AUDIT_ROUND1_1022_pytest.txt`. This file is an internal audit note and is not
part of the published `docs/` site.

## Final test counts (local, Windows py3.13)

- Baseline (pristine `6b0fac0`): **21 passed, 0 skipped**.
- After this round (docs-only change; no source edit): **21 passed, 0 skipped**.
- CI-parity gates: `black --check` (10 files unchanged), `isort --check-only`
  (clean), `flake8` (clean), `compileall` (clean).
- Pendulum learn gate (`python ci_check.py`) run locally: **PASS**,
  Δ = +1039.3 ≥ 150 (first ≈ −1600 → tail ≈ −85). Output under the
  gitignored `logs/ci_pendulum/`.

## Independent numeric recomputation (not reusing plot.py / recompute_stats.py)

A fresh script read the nine committed raw CSVs
(`logs/hopper_{baseline,rewardscale,fixedalpha}_s{0,1,2}/progress.csv`), filtered
`phase == "eval"`, took the last 4 points per seed, and recomputed everything
with numpy/scipy. Every figure on the study page and in `docs/assets/results.json`
MATCHED:

- Per-seed tail means: baseline 547.4 / 1873.2 / 833.1; rewardscale
  644.6 / 886.1 / 1823.8; fixedalpha 2133.1 / 1038.7 / 1039.2.
- Means / SD / t-95% half-width (df=2, t=4.303): baseline 1084.6 ± 697.7
  (±1733.2); rewardscale 1118.2 ± 622.9 (±1547.3); fixedalpha 1403.6 ± 631.7
  (±1569.2).
- Welch: BvsR t=−0.062 p=0.953 diff=−33.6 CI[−1540.4, 1473.3];
  BvsFA t=−0.587 p=0.589 diff=−319.1 CI[−1833.6, 1195.5];
  FAvsR t=+0.557 p=0.607 diff=+285.5 CI[−1136.7, 1707.7].
- ANOVA F=0.217 p=0.811; Cohen's d=−0.051, Hedges' g=−0.041, pooled SD 661.4.
- Descriptive final-train dynamics: baseline α 0.066/0.106/0.068 (mean ≈0.080),
  Q₁ ≈170; rewardscale α 0.0037/0.0066/0.0063, Q₁ ≈19; fixedalpha α=0.2, Q₁ ≈36.

No fabricated or drifted number found; the null-result wording ("not detectable",
never "proves no effect") was preserved everywhere.

## Findings

### P0 / P1 / P2 (source code)

None. SAC critic target, actor loss (`alpha.detach()`), alpha entropy loss,
tanh + rescale Jacobian, circular replay, done/bootstrap semantics (bootstrap on
truncation only, `done=terminated`), and the learn gate were re-derived and match
the committed tests. No source file was modified this round.

### Docs polish (published `docs/` site) — P2 factual corrections in `docs/deploy-bench.html`

1. **CI overstated:** the note claimed CI runs `mypy`; `.github/workflows/ci.yml`
   runs black/isort/flake8, byte-compile, and pytest only — mypy is local-only.
   Corrected.
2. **Fictional CLI walkthrough:** `export model.pt --env Pendulum-v1` (no
   `--env` option; export takes an SB3 zip + `--algo`) and a nonexistent
   `convert --format fp16` subcommand were replaced with real commands
   (`export ... --algo PPO`, `quantize --mode dynamic`, `benchmark`, `compare`);
   FP16 is noted as a Python-API call only.
3. **E5 overclaim:** the note said the ASCII temp-copy happens "before
   `torch.jit.save`"; the hardening actually sits in `_safe_load_torchscript` on
   the **load** side; corrected to match the code.
4. **Missing blog cross-link:** `docs/index.html` footer linked the personal site
   but not the blog; added an absolute `https://dafahaha.github.io/blog/` link
   (verified live) between Personal site and TransitTruth.

No research conclusion, number, or arm definition was altered.

### Site / Pages verification (2026-10-22)

- All internal nav (Home / SAC null / deploy-bench pitfalls / Methodology) and
  every relative asset (`assets/style.css`, `learning_curve.png`,
  `asymptotic.png`, `recompute_stats.py`, `results.json`) resolve on disk; the
  two figures are tracked blobs identical to `figures/`.
- The expected URL `https://dafahaha.github.io/embodied-offpolicy-study/`
  (and `/index.html`, `/docs/`, `/sac-reward-scaling.html`) returns 404:
  GitHub Pages is **not enabled / not published** for this repo. The personal
  homepage (`dafahaha.github.io`) "Write-up" button points to the
  `RESEARCH_RETRO.md` blob, not to this static site. Anonymous API/Settings are
  blocked (403), so enabling Pages is a maintainer Settings action this audit
  could not perform; the site is ready to publish if the maintainer selects
  main `/docs` as the Pages source. No Pages URL can be claimed as live.

### Checked and already correct (no change)

- Study-page figures (`assets/learning_curve.png`, `assets/asymptotic.png`) are
  tracked blobs identical to `figures/`; absent only from this local checkout,
  so no image fix was needed or committed.
- All internal page links and assets resolve; external repo/blob links target
  files that exist; accent is Apple `#0071e3`, no purple/indigo; sidebar collapses
  ≤860px, grid2 ≤900px, pitfall rows ≤700px (mobile readable, zero build).
- `README.md`, `RESEARCH_RETRO.md`, `PRE_REGISTRATION.md`, `methodology.html`,
  `index.html` numbers and honest-null narrative cross-check against the
  recomputation above.

## Mutation experiments

| ID | Mutation | Targeted `-k` | Result |
|----|----------|---------------|--------|
| E-M1 | drop `reward_scale` from Bellman target | reward_scale | CAUGHT (red) |
| E-M2 | remove `alpha.detach()` in actor loss | sac_update / reward_scale | NOT-CAUGHT (gap; harmless here — actor_opt never steps log_alpha) |
| E-M3 | drop rescale Jacobian in logprob | logprob | CAUGHT (red) |
| E-M4 | center Gaussian logprob at zero instead of μ | logprob | CAUGHT (red) |
| E-M5 | learn-gate `>= 150` → `> 150` (exact-boundary edge) | learn_gate | NOT-CAUGHT (gap; no test at Δ=150 exactly) |
| E-M6 | disable nonfinite-logp abort guard | guard | CAUGHT (red) |
| E-M7 | remove circular-replay size cap | replay_buffer | CAUGHT (red) |
| E-M8 | Welch `equal_var=False` → `True` | plot_stats | NOT-CAUGHT (gap; toy arrays have equal variance so Welch==Student) |

The three NOT-CAUGHT items are coverage gaps, not defects: E-M2 is functionally
neutral in this optimizer wiring; E-M5 is an exact-threshold convention; E-M8
only fails to distinguish Welch/Student on equal-variance toy data.

## Platform notes (Windows py3.13, no compiler)

- `mujoco` (pinned 3.15.0) was skipped locally; no unit test imports it, and the
  CI learn gate uses Pendulum-v1 (classic control, no MuJoCo). This is a
  dependency skip, not a platform failure; Hopper runs are not re-executed (CPU
  study, main experiments untouched).
- The configured pip mirror (huaweicloud) was unreachable; installs used
  `--index-url https://pypi.org/simple/`. No pristine-`stash` Windows failure
  reproduction was needed because none occurred.

## Attack surface for the next adversarial reviewer

1. **E-M2 gap:** actor loss alpha detach untested — check whether stepping
   `log_alpha` elsewhere could make a missing detach matter.
2. **E-M5 exact-boundary learn gate:** add a case at Δ=150 to lock `>=`.
3. **E-M8 Welch/Student:** extend `plot.py` stats test with unequal-variance
   toy arrays so `equal_var=False` is actually exercised.
4. `train.py` opens the metrics CSV without an explicit close on the
   `_guard_metrics` abort path; buffered rows could be lost on abort (OS closes
   on exit).
5. No checkpoint/resume (noted as future work in RETRO); a killed 120k run
   restarts from step 0.
6. Multi-threaded CPU torch reductions are not bit-deterministic; only the loose
   learn gate, not exact floats, is reproducible by design.
