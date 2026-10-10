# Reproduction for From Transfer Probes to Expert Grouping

This repository is a joint work of Claude (Anthropic) and Codex (OpenAI), under the authors' direction.

It holds the evidence code and result files for the technical report
**"From Transfer Probes to Expert Grouping: Evidence from LoRA Post-Training"**.
The report is not part of this repository (report link: to be added).

The report asks whether single-source transfer probes can tell which sources should share a LoRA
expert. Seven public sources stand in for business lines: five mathematics subfields from
DeepMath-103K (Integral Calculus, Differential Calculus, Abstract Algebra, Linear Algebra,
Statistics) and two programming domains from OpenCodeInstruct (algorithmic, generic). Four LoRA
experts on Qwen3.5-2B-Base are trained under fixed per-source exposure and per-expert budgets.
Three groupings are compared: the label-tree reference C0, the uniform default C1, and the
probe-derived grouping C2. A fourth executed candidate, C3, is reported in an appendix. Each
candidate is trained once for one interval from a shared warm-up state W, then scored on 400 test
problems per source with two samples each.

What this repository supports:

- **CPU only, about 5 minutes.** Recompute the report's statistical analyses from the shipped
  per-item scores and probe flip counts, and run the test suite. The tests include a check that
  this library reproduces the training trajectories of the unreduced code that produced the
  results.
  Two inputs are summaries only, because their raw inputs are not shipped: the decoding-variance
  design inputs and the parameter distances (see the evidence map below).
- **GPU.** Rerun the pipeline: data manifest, probes, warm-up W, candidate training, evaluation.
  Model checkpoints and the data manifest are not shipped.

## Contents

```
lwts/          experiment library: data manifest, LoRA backend, uniform and fixed allocation,
               transfer probes, answer verifiers
pipeline/      GPU pipeline: Python entry points and launch scripts (run_*.sh)
analysis/      CPU analyses; run_all.sh regenerates results/analysis/
results/
  probes/      probe accuracy changes and flip counts (s17: 7 rows; s23: 6 rows)
  candidates/  per-item 0/1 test scores of C0-C3 (two evaluation seeds each, hashed item ids)
               and training_summary.json (block matrices, routes, token counts, wall-clock seconds)
  protocol/    manifest summary; decoding-variance summary used to choose the number of samples
  analysis/    analysis outputs (*.json is the machine-readable record; *.txt are console reports)
tests/         library tests, analysis reproduction tests, equivalence check with fixtures
pyproject.toml, uv.lock, LICENSE
```

Code comments, docstrings and the console reports (`results/analysis/*.txt`) are partly in
Chinese. JSON keys and file names are in English.

## Reproduce the analyses on a CPU

Run every command from the repository root. Nothing here uses a GPU.

### 1. Environment

The analysis scripts need only the Python standard library and run on Python 3.10 or newer, as
declared in `pyproject.toml`. Reproducing `results/` byte for byte needs **Python 3.12 or newer**.
From 3.12 on, the built-in `sum()` adds floats with compensated summation. On 3.10 and 3.11, some
JSON values therefore differ in the last floating-point digits (by at most 6.2e-15), and four tests
in `tests/test_analysis_reproduce.py` fail. In the console reports, a printed number changes only
where its exact value is a tie at the printed precision. On Python 3.10 the console reports changed
in these places only:

- U_other5 of C0 (0.5042 becomes 0.5043) and of C2 (0.4747 becomes 0.4748) in
  `CANDIDATES-analysis.txt` and `LABELREF-analysis.txt`; the exact values are 2017/4000 and
  1899/4000;
- in `PROXY-posthoc.txt`, the predicted ΔU_other5 of C3 under block-matrix routing on s17
  (−0.13 becomes −0.12 pp, an exact −0.125), and the self-check line (deviation 1.1e-16 becomes
  0.0e+00).

For the analyses alone, no installation is needed: skip to step 2. If your
interpreter is called `python3`, use it wherever the commands say `python`, and run
`PYTHON=python3 bash analysis/run_all.sh` in step 3. The full test suite and the GPU pipeline need
the package dependencies. Two ways to install them:

**uv** (exact versions from `uv.lock`):

```bash
uv sync --frozen --extra train
uv pip install pytest
source .venv/bin/activate
```

The lock takes `torch` from the CUDA 12.6 wheel index. That build also runs on a CPU. Notes on
the lock:

- The locked `torch` 2.14.0+cu126 has wheels only for Linux (x86_64 and aarch64) and Windows
  (x86_64). On macOS, use the pip route below.
- The environment pulls about 5 GB of CUDA libraries even if you only use a CPU (`.venv` of about
  6.6 GB, against about 1.4 GB for the pip route with the CPU wheel).
- `pyproject.toml` declares the PyTorch index without restricting it to `torch`, so the lock also
  takes 37 other packages from it. Besides the CUDA libraries, these include general packages at
  older versions, for example `requests` 2.28.1, `urllib3` 1.26.13 and `certifi` 2022.12.7.

**pip** (newest compatible versions instead of the locked ones):

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: smaller CPU-only wheel
pip install -e ".[train]" pytest
```

The `train` extra (`torch`, `peft`) is needed by the LoRA tests and by the GPU pipeline.

### 2. Make sure Python imports this copy of `lwts`

```bash
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1
python -c "import lwts, os; print(os.path.realpath(lwts.__file__))"   # must print a path inside this directory
```

If another `lwts` is installed in the same environment, Python could silently import it instead,
and the checks below would then test the wrong code. Setting `PYTHONPATH` gives this copy
precedence. `analysis/run_all.sh`, `tests/conftest.py` and the analysis tests check the import
location and stop if it is wrong.

### 3. Recompute all analyses (about 2.5 minutes)

```bash
bash analysis/run_all.sh
git diff --exit-code -- results/ && echo "results/ reproduced byte for byte"
```

The `git diff` check works only in a git clone. In a copy without git history, such as a
downloaded ZIP archive, use one of these instead:

- run the analysis tests of step 4 (`tests/test_analysis_reproduce.py` and
  `tests/test_analysis_probe_stats.py`); they rerun each analysis in a temporary directory and
  compare the outputs with `results/analysis/`;
- write the outputs elsewhere with `--out` and compare them with `cmp`, as shown below.

`run_all.sh` regenerates `results/analysis/` in place from `results/probes/` and
`results/candidates/`. It runs `analyze_candidates.py`, `analyze_label_reference.py`,
`proxy_posthoc.py`, `allocation_sweep.py --experts 2 3 4 5`, `probe_stats.py`,
`c0_construction.py` and `control_agreement.py`. It does not run two scripts whose inputs are not
shipped: `param_distance.py` (model snapshots) and `prepare_results.py` (raw run directories).

To write somewhere else, pass `--out` to each script except `allocation_sweep.py`, which prints
its report to stdout (redirect it yourself, as `run_all.sh` does). The JSON outputs are identical.
The `.txt` reports differ only in the line that names the output path. For example:

```bash
mkdir -p /tmp/check
python analysis/probe_stats.py --out /tmp/check
cmp /tmp/check/PROBE-stats.json results/analysis/PROBE-stats.json && echo identical
python analysis/allocation_sweep.py --experts 2 3 4 5 > /tmp/check/ALLOC-sweep.txt
cmp /tmp/check/ALLOC-sweep.txt results/analysis/ALLOC-sweep.txt && echo identical
```

### 4. Run the tests (about 3 minutes)

```bash
python -m pytest -q tests                                       # 224 tests
python -m pytest -q tests --ignore-glob='tests/test_analysis_*'  # 194 tests, about 35 s; skips the analysis reruns
```

Without the package dependencies, the 30 analysis tests also run under the standard library:
`python -m unittest discover -s tests -p 'test_analysis_*.py'`.

`tests/test_lora.py` loads the `Qwen/Qwen3.5-2B-Base` tokenizer. With `HF_HUB_OFFLINE=1` and no
cached copy, its 19 tests are skipped. Unset `HF_HUB_OFFLINE` once to download the tokenizer
files; model weights are not downloaded.

### Verified environments

The commands above were run on fresh copies of this repository, on Linux and on CPU only:

| Environment                                                                                        | `run_all.sh`                                                                                               | Tests                                                                                                          |
| -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| uv, locked versions (Python 3.13; torch 2.14.0, transformers 5.16.1, peft 0.20.0, datasets 2.19.1) | byte-identical`results/`                                                                                   | 224 passed; 194 passed with the analysis tests excluded                                                        |
| pip (Python 3.13; torch 2.14.1 CPU wheel, transformers 5.19.0, peft 0.21.2, datasets 5.1.0)        | byte-identical`results/`                                                                                   | 224 passed                                                                                                     |
| No third-party packages (Python 3.13)                                                              | byte-identical`results/`                                                                                   | 30 analysis tests passed (unittest)                                                                            |
| No third-party packages (Python 3.12)                                                              | byte-identical`results/`                                                                                   | not run                                                                                                        |
| uv, locked versions (Python 3.10)                                                                  | not byte-identical: 7 files under`results/analysis/` differ in the last floating-point digits (see step 1) | 220 passed, 4 failed (`tests/test_analysis_reproduce.py`: Candidates, LabelReference, Proxy, C0Construction) |
| No third-party packages (Python 3.10)                                                              | as above                                                                                                     | 26 of 30 analysis tests passed (unittest); the same 4 failed                                                   |
| No third-party packages (Python 3.11)                                                              | as above (the same 7 files)                                                                                  | not run                                                                                                        |

## Report → evidence map

Paths are relative to the repository root. `a.b` names a key inside a JSON file, and `*` stands for
each candidate or shuffle.

| Report table or claim                                                                                                                                                                                                                                                                              | Produced by                                                                                            | Result file and key                                                                                                                                                                            |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Data splits (66/12/6/10/6%); test-bank sizes (smallest 421); the 400 algorithmic test problems are 376 MBPP+ plus 24 OpenCodeInstruct; base model                                                                                                                                                  | `pipeline/build_manifest_7src.py` (`SPLIT_RATIOS`)                                                 | `results/protocol/manifest_summary.json`: `split_fraction_all_records`, `test_by_dataset`, `first_400_test_sorted_by_id_by_dataset`, `min_test_count`, `header`                    |
| Interval geometry: M=4, H=56, 14 blocks per expert, 8 per source; at least 32,768 completion tokens per block, 35,144 on average; W gives each expert 2 blocks of each source                                                                                                                      | `pipeline/run_warmup.sh`, `pipeline/run_candidates.sh`                                             | `results/candidates/training_summary.json`: `candidates.*.blocks`, `candidates.*.block_tokens_realized`, `warmup_W_round0`                                                             |
| Candidates table: block matrices and block-matrix routes of C0–C3, including C1's tie that routes every source to expert 0                                                                                                                                                                        | `pipeline/run_candidates.sh`, `pipeline/run_eval.sh`; route check in `analysis/proxy_posthoc.py` | `training_summary.json`: `candidates.*.blocks`, `candidates.*.routing`; `results/analysis/PROXY-posthoc.json`: `block_routing`, `self_checks.block_routing_matches_eval`           |
| Probe design: 16 blocks per probe; 80 audit problems per source; s17 has 7 rows, s23 has 6 (generic missing)                                                                                                                                                                                       | `pipeline/probe_transfer.py`, `pipeline/salvage_partial_tensor.py`                                 | `results/probes/s17_transfer.json` (`blocks`, `step` = 1/80); `results/probes/s23_transfer_partial.json` (`rows_complete`, `rows_missing`)                                         |
| Probe cell statistics and the probe-cell table: S0 accuracies; exact two-sided McNemar per cell; nominal cells (5 of 49 in s17, 4 of 42 in s23); Bonferroni over 49 cells (Differential→Statistics in s23, p_Bonf = 0.00103; none in s17; no edge in both); expected nominal counts 2.45 and 1.35 | `analysis/probe_stats.py`                                                                            | `results/analysis/PROBE-stats.json`: `s0_accuracy`, `shuffles.*`, `table_cells`, `bonferroni_in_both_shuffles`                                                                       |
| Statistics column +111/−36 (per donor, strongest donor), generic control +55/−60; label-pair counts (+71/−23, +66/−42, +60/−41, +44/−22) and Fisher exact p (0.035, 0.022, 0.284)                                                                                                            | `analysis/probe_stats.py`                                                                            | `PROBE-stats.json`: `statistics_column`, `generic_column`, `pairs`, `fisher_vs_linear_statistics`                                                                                    |
| C0 is fixed by the tree and the margins: one four-group cut at M=4 (1, 1, 1, 3, 3, 1 cuts for M=2…7); Sinkhorn balancing of the label map plus integer rounding gives C0; C0 ties for optimum in the tree-distance transportation problem                                                         | `analysis/c0_construction.py`, `tests/test_c0_derivation.py`                                       | `results/analysis/C0-construction.json`: `covering_cut_counts_M2_to_M7`, `derived_equals_C0`, `tree_distance_cost_times_56`                                                            |
| Development proxy at M=4: 0.4426 (best) against 0.4348 (tree-distance fill of the tree grouping) and 0.4336 (C0); C2 is the transportation-LP vertex and is already integral                                                                                                                       | `analysis/allocation_sweep.py`, `analysis/proxy_posthoc.py`                                        | `results/analysis/ALLOC-sweep.txt` (row M = 4); `PROXY-posthoc.json`: `self_checks`, `results.argmax.s17`                                                                              |
| Candidate comparison table, comparisons specified before the run (C1−C0, C2−C0 for U_all, U_LS, U_other5 with 95% joint paired bootstrap intervals); disagreement between two samples (20–34% per source)                                                                                       | `analysis/analyze_candidates.py`                                                                     | `results/analysis/CANDIDATES-analysis.json`: `point`, `contrasts`, `meta.*.seed_flip_rate`                                                                                             |
| Candidate comparison table, post hoc row C2−C1, and its stability over 22 bootstrap seeds                                                                                                                                                                                                         | `analysis/analyze_label_reference.py`                                                                | `results/analysis/LABELREF-analysis.json`: `pairs.C2-C1`, `stability`                                                                                                                    |
| Traffic-mix table: per-source Δ_j, range over traffic weights, simultaneous 95% intervals (q = 3.025), footnotes on bootstrap-seed stability                                                                                                                                                      | `analysis/analyze_label_reference.py`                                                                | `LABELREF-analysis.json`: `flow_sensitivity`, `simultaneous`, `stability`                                                                                                              |
| Proxy diagnostic, both proxy tables: predictions under block-matrix routing, competence routing and zero transfer for s17 and s23*; effect of the routing rule on C0 (1.66 and 1.75 pp)                                                                                                            | `analysis/proxy_posthoc.py`                                                                          | `PROXY-posthoc.json`: `results.{block,argmax,zero_transfer}.*.delta_vs_C0`, `C0_argmax_minus_block`                                                                                      |
| Decoding-variance table; SE 0.98 pp and MDE 2.74 / 1.93 / 1.36 pp; choice k = 2                                                                                                                                                                                                                    | summary only; see Known limitations                                                                    | `results/protocol/decoding_variance_summary.json`                                                                                                                                            |
| Cost: about 4,570 s per training interval; about 32,300 s and 2,800 generations per evaluation                                                                                                                                                                                                     | run ledger and evaluation files                                                                        | `training_summary.json`: `candidates.*.train_wall_seconds`, `eval_wall_seconds`, `eval_generations`; `results/candidates/C*-eval-s*.json`: `seconds`, `generated`                |
| Interval-control diagnostic: matched experts differ by about 0.027 against a displacement of 1.4–1.6 from W (about 2%); unmatched experts differ by 0.88–0.94                                                                                                                                    | `analysis/param_distance.py` (inputs not shipped)                                                    | `results/analysis/PARAM-distance.json`                                                                                                                                                       |
| Interval-control diagnostic: differences on the control sources; per-item agreement between C0 and C3                                                                                                                                                                                              | `analysis/analyze_candidates.py`, `analysis/control_agreement.py`                                  | `CANDIDATES-analysis.json`: `contrasts.C3-C0.per_source`; `results/analysis/CONTROL-agreement.json`: `sets.*.range`                                                                    |
| C3 appendix: both tables, range over traffic weights, simultaneous intervals, proxy predictions                                                                                                                                                                                                    | `analyze_candidates.py`, `analyze_label_reference.py`, `proxy_posthoc.py`                        | `CANDIDATES-analysis.json`: `point.C3`, `contrasts.C3-C0`; `LABELREF-analysis.json`: `pairs.C3-C1`, `flow_sensitivity.C3`, `simultaneous`, `stability`; `PROXY-posthoc.json` |
| Cost estimates for open questions                                                                                                                                                                                                                                                                  | unit costs from the ledger                                                                             | `training_summary.json`                                                                                                                                                                      |

The two console reports `CANDIDATES-analysis.txt` and `LABELREF-analysis.txt` print the same
numbers in table form.

## GPU pipeline

Requirements:

- One CUDA GPU with 24 GB of memory.
- The `train` extra installed.
- Network access to Hugging Face for the datasets and the model. If you exported
  `HF_HUB_OFFLINE=1` for the CPU steps above, run `unset HF_HUB_OFFLINE` first or use a new shell.
  Step 1 fails in offline mode even when the datasets are already cached, because it resolves the
  OpenCodeInstruct shards through the Hugging Face API.
- About 16 GB of RAM for step 1 (peak on a fresh copy).
- Disk space: the dataset cache under `data/raw/hf` grows to about 8 GB, and the manifest is about
  1.4 GB. Each training run directory also stores a copy of the manifest with its banks capped
  (about 1.1 GB) and about 0.7 GB per backend snapshot.

> **Warning.** Grading executes model-generated code. The verifiers run it in a subprocess with a
> timeout and resource limits; that isolation is for robustness and is not a security sandbox. Run
> the pipeline only on a machine you can afford to lose.

The launch scripts `cd` to the repository root and put it first on `PYTHONPATH`. They read these
environment variables:

| Variable                     | Meaning                                      | Default                               |
| ---------------------------- | -------------------------------------------- | ------------------------------------- |
| `GPU`                      | device index                                 | `0`                                 |
| `MANIFEST`                 | manifest path                                | `runs/manifest-7src/manifest.jsonl` |
| `OUT`, `OUT_ROOT`, `W` | output directories and the W checkpoint path | see each script                       |
| `PY`                       | Python interpreter                           | `python`                            |

| Step                  | Command                                                  | Output                                                    | Time on one 24 GB GPU                               | Command provenance                                                                                                                                        |
| --------------------- | -------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Data manifest      | `python pipeline/build_manifest_7src.py --code-pool 0` | `runs/manifest-7src/manifest.jsonl`                     | CPU only, CPU-bound; about 15 min on a fresh copy   | the original build used all rows of the three OpenCodeInstruct shards (inferred from its build log); rebuilt this way on a fresh copy, the sha256 matches |
| 2. Probes s17 and s23 | `bash pipeline/run_probes.sh 17 23`                    | `runs/probe-s{17,23}/transfer.json`                     | roughly 16–17 GPU-h per seed (estimate, see below) | **inferred**                                                                                                                                        |
| 3. Warm-up W          | `bash pipeline/run_warmup.sh`                          | `runs/W/checkpoint.json`                                | about 1.3 GPU-h (estimate, see below)               | executed command without two measurement-only options                                                                                                     |
| 4. Candidates C0–C3  | `bash pipeline/run_candidates.sh C0 C1 C2 C3`          | `runs/candidates/C*/checkpoint-r2.json`                 | 4,545–4,583 s each (recorded)                      | same arguments as executed                                                                                                                                |
| 5. Evaluation         | `K=2 bash pipeline/run_eval.sh C0 C1 C2 C3`            | `runs/candidates/C*-eval-s200{1,2}/test_extension.json` | 32,218–32,444 s each, 8 evaluations (recorded)     | same arguments as executed                                                                                                                                |

The recorded steps take about 77 GPU-hours: 18,260 s of training plus 258,659 s of evaluation,
from `training_summary.json`. Training one candidate and evaluating it twice takes about
4,570 + 2 × 32,300 ≈ 69,200 s ≈ 19.2 GPU-hours. With the estimates for steps 2 and 3, the whole
pipeline takes about 110 GPU-hours.

The step 2 and step 3 times were not recorded for the commands given here, so they are estimated
from the recorded unit costs:

- **Probes.** About 82 s per training block (4,570 s / 56 blocks) and about 11.5 s per generation
  (32,300 s / 2,800). One seed trains 7 rows × 16 blocks = 112 blocks (≈ 2.5 h) and generates
  (1 baseline + 7 rows) × 7 sources × 80 = 4,480 audit answers (≈ 14.4 h).
- **W.** Trains the same 56 blocks as one candidate interval (2.52 M against 2.48 M processed
  tokens).

**1. Data manifest.** Pass `--code-pool 0`. It keeps all rows of the three OpenCodeInstruct
shards, as the reported manifest did. The script's default (`--code-pool 120000`) truncates the code
pool and gives a different manifest with entirely different splits; every later step would then run
on other data without any error.

```bash
unset HF_HUB_OFFLINE                       # this step needs the network
python pipeline/build_manifest_7src.py --code-pool 0
sha256sum runs/manifest-7src/manifest.jsonl   # f38f2fb97f2171f22b4756bd2f85e5629fda3403c6470ccd1c7ae5188186e8c3
```

This step uses only the CPU and is limited by it, not by the network. On a fresh copy it took about
15 minutes, most of it tokenizing DeepMath-103K, with a peak of about 16 GB of RAM. It downloads
DeepMath-103K, OpenCodeInstruct shards 45, 48 and 49 and MBPP+ into `data/raw/hf` (about 8 GB
after the build). The Qwen tokenizer goes to the default Hugging Face cache (`HF_HOME`), not to
`data/raw/hf`. The default `--salt` fixes the split. The console output should report
`OpenCodeInstruct   train 原始 298043 → 保留 106912` and `记录 158208`.

The expected hash is `sha256` in `results/protocol/manifest_summary.json`. If your hash differs,
first check that you passed `--code-pool 0` and kept the defaults of `--pool`, `--code-shards`,
`--seq-len` and `--salt`. Only then suspect the upstream data: dataset revisions are not pinned
(see Known limitations).

**2. Probes.** Each probe starts at S0, the base model with a freshly initialized adapter. It
trains on 16 consecutive blocks of one source, scores 80 audit problems per source, and rolls the
state back before the next source. The command lines of the two probe runs were not recorded.
`run_probes.sh` reconstructs them from the run outputs and uses `probe_transfer.py` defaults for
everything else. The reconstructed settings are:

- the 7-source manifest and an audit bank of 80 per source;
- 16 blocks per cell;
- generation caps of 8,192 tokens for the five mathematics sources and 1,024 for the two code
  sources;
- seeds 17 and 23.

The s23 run behind the shipped file stopped before its last row (generic). `results/probes/s23_transfer_partial.json`
holds the six completed rows, recovered from the run log with `pipeline/salvage_partial_tensor.py`.
The script runs all seven rows.

**3. Warm-up W.** This step runs `pipeline/run_pilot.py --method uniform --rounds 1 --warmup 1`
with seed 101. It starts from the base model with freshly initialized adapters and trains one
uniform interval of 2 blocks per expert per source. The run that produced W also carried out a
read-only diagnostic during the interval: it generated and scored answers on held-out banks and
computed measurements from them. That code is not part of this package, and the command here drops
the two options that enabled it. Known limitations explains why this does not change the training
state.

`run_pilot.py` keeps its full settings table, so the `--set` options of the executed commands
replay verbatim. The `uniform` and `fixed` methods read only `block_tokens` and
`tokens_per_interval`.

**4. Candidates.** Each candidate forks from W (`--fork-from`) and trains one interval with
`--method fixed` and its block matrix. The fork checks that adapters, optimizer and scheduler
states, the training RNG, data cursors and step counts equal those of W.

**5. Evaluation.** `pipeline/score_test_extension.py` loads each `checkpoint-r2.json` and serves
each source from the expert that received most of its blocks, with ties going to the lower index.
It samples answers (temperature 0.8) for the first 400 test problems per source in sorted-id order,
with CUDA seeds 2001 and 2002 shared across candidates. `run_candidates.sh` and `run_eval.sh`
append wall-clock lines to `$OUT/ledger.txt`.

Checkpoint files point to their backend snapshot files by the path the snapshots were written
under. With the default relative `OUT`, that path is relative to the repository root. Keep run
directories in place, and run every step from the repository root; the scripts do this.

### Analysing your own runs

The analysis scripts accept raw evaluation files, which have the same layout as the shipped ones
except for unhashed item ids. To analyse your own runs:

```bash
mkdir -p runs/analysis
for c in C0 C1 C2 C3; do for s in 2001 2002; do
  ln -sf "$PWD/runs/candidates/$c-eval-s$s/test_extension.json" "runs/analysis/$c-eval-s$s.json"
done; done
python analysis/analyze_candidates.py --run runs/analysis --out runs/analysis > runs/analysis/CANDIDATES-analysis.txt
python analysis/analyze_label_reference.py --run runs/analysis --out runs/analysis
# probe_stats.py and proxy_posthoc.py read the second shuffle as {"rows": [...]}
python -c "import json,sys; json.dump({'rows': json.load(open(sys.argv[1]))}, open(sys.argv[2], 'w'))" \
  runs/probe-s23/transfer.json runs/analysis/s23_rows.json
python analysis/probe_stats.py --s17 runs/probe-s17/transfer.json --s23 runs/analysis/s23_rows.json --out runs/analysis
python analysis/proxy_posthoc.py --s17 runs/probe-s17/transfer.json --s23 runs/analysis/s23_rows.json \
  --run runs/analysis --out runs/analysis
```

`analyze_label_reference.py` and `proxy_posthoc.py` read `CANDIDATES-analysis.json` from their
`--out` directory, so run `analyze_candidates.py` first.

`analysis/prepare_results.py` turned the raw run directories into `results/`. It records each
transformation as code:

- it copies s17 unchanged;
- it replaces two descriptive metadata fields in s23;
- it hashes test ids and drops the absolute checkpoint path from the evaluation files;
- it writes the summaries.

Its inputs are not shipped. `analysis/param_distance.py --w <W.pt> --c0 <C0.pt> --c3 <C3.pt>`
needs the backend snapshots of W and of the C0 and C3 training intervals.

## Code written for this report

The rest of the code is the code that produced the results, reduced as described in the next
section, with comments and some file names edited. The four analysis scripts carried over
(`analyze_candidates.py`, `analyze_label_reference.py`, `proxy_posthoc.py`, `allocation_sweep.py`)
changed only in comments and in their default paths, which now point to `results/`. The first
three read evaluation files named `{candidate}-eval-s{seed}.json` and gained an `--out` option.
`allocation_sweep.py` has no `--out` option: it prints its report to stdout, and it now reads only
the s17 tensor by default. The files below were written for this report:

| File                                                                                                                                            | Purpose                                                                                                                                                                                          |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `analysis/probe_stats.py`                                                                                                                     | Per-cell exact McNemar from flip counts, Bonferroni over the 49-cell family, expected nominal counts under the null, column and label-pair aggregates, exact Fisher tests in rational arithmetic |
| `analysis/param_distance.py`                                                                                                                  | L2 distances between expert LoRA parameters of C0, C3 and W, with input sha256                                                                                                                   |
| `analysis/c0_construction.py`                                                                                                                 | Tree cuts per number of experts, derivation of C0, tie in the tree-distance transportation problem                                                                                               |
| `analysis/control_agreement.py`                                                                                                               | Per-item agreement on the control sources of C0 and C3                                                                                                                                           |
| `analysis/prepare_results.py`                                                                                                                 | Cleaning tool that produced`results/` from the raw run directories                                                                                                                             |
| `analysis/run_all.sh`                                                                                                                         | Regenerates`results/analysis/` on a CPU                                                                                                                                                        |
| `tests/test_strip_equivalence.py`, `tests/_strip_scenarios.py`, `tests/fixtures/strip_equivalence_{synthetic,lora}.json`                  | Equivalence check between this library and the code that produced the results (next section)                                                                                                     |
| `tests/test_c0_derivation.py`                                                                                                                 | C0 equals Sinkhorn balancing of the label map followed by integer rounding, and the tree has a single four-group cut                                                                             |
| `tests/conftest.py`                                                                                                                           | Stops the test run if`lwts` is not imported from this directory                                                                                                                                |
| `tests/_analysis_common.py`, `tests/test_analysis_reproduce.py`, `tests/test_analysis_probe_stats.py`, `tests/test_analysis_results.py` | Rerun the analyses against`results/analysis/`; check key numbers and the consistency of `results/`                                                                                           |
| `pipeline/run_probes.sh`, `pipeline/run_warmup.sh`, `pipeline/run_eval.sh`                                                                | Launch scripts with relative paths and environment variables                                                                                                                                     |
| `pipeline/run_candidates.sh`                                                                                                                  | Executed launch script with its paths turned into parameters; evaluation moved to`run_eval.sh`                                                                                                 |

Their outputs are `results/analysis/PROBE-stats.*`, `PARAM-distance.json`, `C0-construction.*`
and `CONTROL-agreement.json`, and everything under `results/candidates/` and `results/protocol/`.

## How the library relates to the code that produced the results

`lwts/` and `pipeline/` are the code that produced the results with these code paths deleted:

- allocation methods other than `uniform` and `fixed`, with the estimators and diagnostics that
  served them;
- the read-only diagnostic attached to the W run;
- datasets and verifiers that the evidence does not use.

The logic of the remaining paths is unchanged. Three consequences:

- `lwts.allocation.mirror` is kept only as a general Sinkhorn balancing utility, which the C0
  derivation uses with η = 0.
- The checkpoint format (`lwts-experiment-v2`) is unchanged. Checkpoints written by the reported
  runs carry an extra array file of about 2 GB that holds diagnostic state. The library checks that
  file's presence, hash and key set, but does not load it. New checkpoints write an empty array file.
- The `portfolio_return` field in `results/probes/s17_transfer.json` summarizes the probed expert's
  own before/after change. The other experts' rows are zero. No analysis or claim in the report uses
  this field.

`tests/test_strip_equivalence.py` replays these scenarios with this package's `lwts`:

| Scenario | What it covers                                                                                       |
| -------- | ---------------------------------------------------------------------------------------------------- |
| S1       | warm-up W                                                                                            |
| S2       | C0–C3 forked from W                                                                                 |
| S3       | the same with a checkpoint in the older layout, including the extra state and a non-empty array file |
| S4       | transfer probes                                                                                      |
| S5       | test evaluation after restoring a candidate checkpoint                                               |

The test compares each replay with fixtures recorded by the code before the reduction, on the same
scenarios. It covers S1–S5 on a synthetic backend and S1–S3 on a tiny CPU LoRA model. The compared
quantities are:

- the training block sequence (expert, source, record ids);
- the data cursors and allocations;
- the logged block and token counts;
- the full training state: adapters, optimizer and scheduler states, step counts, training and
  Python RNG states;
- checkpoint contents;
- probe outputs;
- per-item evaluation scores.

The fixtures hold only record ids and numbers.

The fixtures were recorded with the unreduced code that trained and evaluated the candidates. The
probes and W were run with earlier revisions of that code. Their differences from it lie outside
the probe path and do not change the blocks or the training of the uniform warm-up interval. They
concern, for example, logging, options whose defaults keep the earlier behaviour, code used only by
the removed diagnostic, and an integer rounding routine that gives the same blocks for the uniform
interval.

## Known limitations

- **One training run per candidate.** Every interval in the report covers test-item and decoding
  sampling only, conditional on that single run. No candidate was retrained.
- **W can be reproduced semantically, not bitwise.** The commands give the same data order,
  allocation and optimizer settings. GPU kernels are not deterministic, however. In C0 and C3, two
  experts that received identical blocks from the same W ended about 2% of their training
  displacement apart (`results/analysis/PARAM-distance.json`). Expect differences of that order in
  any rerun.
- **The W run also carried a read-only diagnostic process, which this package removes.** The
  diagnostic generated and scored answers on held-out banks during the warm-up interval. Its code
  isolated its random numbers and asserted at run time that it left the training state unchanged.
  The equivalence test's fixtures for W were recorded with the diagnostic switched on. This package
  reproduces them without it: the same block sequence, cursors and training state. The only
  difference is the stored CPU evaluation-RNG state. GPU sampling does not use that state, and
  candidate evaluations set the CUDA seed explicitly.
- **Probe before/after samples are not common-random-number pairs on the GPU.** The code resets
  the CPU generator before the "after" evaluation, but GPU sampling uses the CUDA generator, which
  is not reset. The per-item paired McNemar tests do not rely on common random numbers and remain
  valid.
- **The probe command lines are inferred** (see step 2), and the s23 probe has six of seven rows.
- **The algorithmic test set is mostly MBPP+.** The 400 algorithmic test problems are 376 MBPP+
  problems and 24 OpenCodeInstruct problems. That source was trained and probed on
  OpenCodeInstruct problems only.
- **Dataset and model revisions are not pinned.** The manifest records `main` for DeepMath-103K,
  OpenCodeInstruct, MBPP+ and the tokenizer. The dataset caches used for the build correspond to
  DeepMath-103K commit `5cf055d1` and MBPP+ commit `b2d74c91`; the OpenCodeInstruct parquet shards
  carry no recorded commit. As of October 2026, `main` of the three dataset repositories points to
  commits last modified on 2025-05-29 (DeepMath-103K, `5cf055d1`), 2025-04-28 (OpenCodeInstruct,
  `8f3ba5ba`) and 2024-04-17 (MBPP+, `b2d74c91`). The manifest sha256 is the final check.
- **The decoding-variance inputs are not shipped.** The numbers used to choose k = 2 are design
  inputs measured on four-expert portfolios of an earlier comparison, each trained from W for one
  interval. The split of decoding variance used one of these portfolios; the paired standard error
  and the minimum detectable effects used the per-item paired difference between two of them (only
  its variance, not the difference itself). Neither portfolio is part of this report.
  `results/protocol/decoding_variance_summary.json` holds the summary numbers only; the per-item
  scores and the script that computed them are not included.
- **Checkpoints and the manifest are not shipped.** Model checkpoints (about 0.7 GB per backend
  snapshot), the manifest (about 1.4 GB, containing full problem texts and reference solutions) and
  the raw run logs are not included. The training and evaluation steps can therefore be rerun but
  not resumed from the reported states, and `param_distance.py` and `prepare_results.py` cannot be
  rerun from this repository.

## Data sources and licenses

| Source                                                                                                                                  | Used for                                                              | License (Hugging Face metadata, checked October 2026)                                                   |
| --------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K) (`zwhe99/DeepMath-103K`)                                         | the five mathematics sources                                          | MIT                                                                                                     |
| [OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) (`nvidia/OpenCodeInstruct`)                                | generic programming; algorithmic training, audit and 24 test problems | CC BY 4.0                                                                                               |
| [MBPP+](https://huggingface.co/datasets/evalplus/mbppplus) (`evalplus/mbppplus`, from [EvalPlus](https://github.com/evalplus/evalplus)) | 376 algorithmic test problems, scored with the EvalPlus tests         | Apache-2.0; derived from[MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp), CC BY 4.0 |
| [Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) (`Qwen/Qwen3.5-2B-Base`)                                                | base model and tokenizer                                              | Apache-2.0                                                                                              |

DeepMath-103K collects problems from several upstream sources. Their individual terms were not
reviewed here. Check each dataset's terms for your intended use.

**No problem text is redistributed.** The shipped evaluation files store, for every test problem,
the first 16 hexadecimal characters of the SHA-256 of its manifest record id instead of the id
itself, because the id contains the problem statement. The probe files contain only aggregate
accuracies and flip counts. To map hashes back to problems:

1. Rebuild the manifest (step 1).
2. Take each source's test records sorted by id. The first 400 are the scored problems, in file
   order.
3. Compute `hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:16]` for each record.

The hashes are unique within each source, and `manifest_summary.json`
(`eval_test_ids_equal_hashed_first_400_sorted`) records that they match the manifest.

## License

The code and result files in this repository are released under the MIT License; see
[LICENSE](LICENSE). The datasets and the model listed above keep their own licenses.
