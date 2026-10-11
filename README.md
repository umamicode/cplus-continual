# CPLUS: Continual Learning via Self-Probe Gradients

[Paper](https://arxiv.org/abs/2609.32771) · [TPU implementation](#environment) · [Reproduction protocol](#released-csqa-protocol)

This repository contains a JAX/Google TPU implementation for CPLUS. CPLUS preserves previously acquired behavior by using a compact set of real anchors together with self-probes generated and labeled by the frozen base model. Their gradients define an optimizer-update mask; supervised fine-tuning on initially incorrect examples supplies the learning signal.

The public reproduction workflow runs Qwen3-0.6B on CommonsenseQA with anchor budgets `8` and `16`.

## Quick start

Run the released comparison from a Google TPU environment:

```bash
git clone https://github.com/umamicode/cplus-continual.git
cd cplus-continual

python -m venv .venv-tpu
source .venv-tpu/bin/activate
pip install -r requirements-tpu.txt
python scripts/download_models.py --output-root weights --model-size 0.6B

python scripts/validate_csqa_input.py data/inputs/csqa.jsonl
bash experiments/run_csqa_main.sh data/inputs/csqa.jsonl weights artifacts/csqa-main
```

The launcher validates the prepared input, writes the exact nine-command plan to the output directory, and executes those commands serially on the configured TPU allocation. It never downloads benchmark data or enables experiment tracking.

## Released CSQA protocol

| Item | Setting |
| --- | --- |
| Model | Qwen3-0.6B, pinned revision in `configs/reproduction_matrix.json` |
| Benchmark | CommonsenseQA (CSQA) |
| Anchor budgets | `A = 8, 16` |
| Seed | `0` |
| CPLUS self-probes | `N = 512` |
| Self-probe prompts | 2 in-context anchors; temperature 0.8; frozen-model argmax labels |
| Training | AdamW, learning rate `1e-6`, weight decay `0.1`, 25 epochs |
| Backend | JAX on Google TPU with BF16 model computation; default tensor parallelism 4 |

The plan evaluates four methods under the same model-specific initial split:

| Method | Update/preservation rule |
| --- | --- |
| SFT | Supervised fine-tuning on initially incorrect examples. |
| Replay | Standard supervised replay of the selected initially-correct anchors alongside new-data updates. |
| CPL | Optimizer-update masking from real-anchor gradients. |
| CPLUS | Optimizer-update masking from the equal-weight blend of real-anchor and self-probe gradients, with bias-corrected mask EMA (`beta=0.9`). |

Retention is the fraction of initially correct examples that remain correct after training. Acquisition is the fraction of initially incorrect examples that become correct. The `aggregate_results.py` collector writes the per-run metric summary to a CSV.

## Repository layout

```text
configs/reproduction_matrix.json  locked released-protocol settings
data/README.md                    prepared input contract and provenance record
experiments/run_csqa_main.sh      run the released nine-job CSQA protocol
experiments/README.md             experiment entry-point documentation
jax_sft/                          JAX/TPU model, training, replay, and masking code
scripts/download_models.py        download the pinned public Qwen3 revision
scripts/validate_csqa_input.py    validate the CSQA protocol input and report its SHA-256
scripts/reproduce.py              print or run one protocol cell
scripts/print_csqa_main_grid.py   print the 9-job main CSQA plan
scripts/aggregate_results.py      collect final summaries into a CSV
scripts/check_release.py          offline source, manifest, and privacy checks
requirements-tpu.txt              TPU installation requirements
```

## Environment

Create a TPU-compatible Python environment and install the dependencies:

```bash
python -m venv .venv-tpu
source .venv-tpu/bin/activate
pip install -r requirements-tpu.txt
python scripts/download_models.py --output-root weights
```

The requirements file expresses installation constraints rather than a frozen runtime image. Record the TPU type, Python version, JAX, jaxlib, libtpu, and resolved dependency versions with each reproduction. The model revision is pinned in `configs/reproduction_matrix.json`; use the downloaded local checkpoint so the run does not depend on a moving model default.

## Data

CommonsenseQA is publicly available. Place the task-formatted input at `data/inputs/csqa.jsonl`; the driver uses the same file for all four methods, evaluates the frozen checkpoint, and deterministically produces model-specific initially-correct and initially-incorrect splits.

Validate the input before running:

```bash
python scripts/validate_csqa_input.py data/inputs/csqa.jsonl
```

See [`data/README.md`](data/README.md) for the task input format.

## Reproduce the released comparison

The quickstart command above is the recommended entry point. It writes `csqa-main-commands.sh` into the chosen output directory before executing it, so the exact run plan is retained with the artifacts.

To inspect the fixed plan without executing it:

```bash
python scripts/print_csqa_main_grid.py \
  --source-jsonl data/inputs/csqa.jsonl \
  --model-root weights \
  --output-root artifacts/csqa-main \
  > csqa-main-commands.sh

bash csqa-main-commands.sh
```

The plan contains nine jobs: one anchor-independent SFT run, plus Replay, CPL, self-probe generation, and CPLUS for each of two anchor budgets. It requests one TPU execution per command and uses non-overlapping output paths. Run a command only once; reuse its completed output rather than launching a duplicate backend attempt.

For an individual cell, `scripts/reproduce.py` prints a command by default. Add `--execute` after confirming the paths and TPU allocation:

```bash
python scripts/reproduce.py probes \
  --model-size 0.6B --task csqa --anchors 8 --seed 0 \
  --source-jsonl data/inputs/csqa.jsonl \
  --model-path weights/Qwen/Qwen3-0.6B \
  --output-root artifacts/csqa-main

python scripts/reproduce.py train \
  --model-size 0.6B --task csqa --method cplus --anchors 8 --seed 0 \
  --source-jsonl data/inputs/csqa.jsonl \
  --model-path weights/Qwen/Qwen3-0.6B \
  --output-root artifacts/csqa-main
```

CPLUS consumes the probe bank generated by the first command. Use `--method sft`, `--method replay`, or `--method cpl` for the baselines. A `--smoke` invocation checks the short pipeline only and is not a reported result.

## Collect results

After all nine jobs complete successfully:

```bash
python scripts/aggregate_results.py \
  --input-root artifacts/csqa-main \
  --output-csv outputs/csqa-main-summary.csv
```

The collector expects one completed SFT run and one completed Replay, CPL, and CPLUS run for each anchor budget.

## Release checks

Before making an archive or tag, run:

```bash
python -B scripts/check_release.py
```

This performs offline Python and shell syntax checks, validates the checksum manifest, and scans for credentials, private paths, account identifiers, and machine metadata. It does not launch a TPU experiment. To make a deterministic source-only archive:

```bash
python -B scripts/package_release.py --output ../cplus-continual-source.zip
```

Model weights, benchmark inputs, generated probes, experiment artifacts, logs, and dashboard exports are excluded. Experiment tracking is disabled by the launchers.

## License

Code in this repository is released under the [Apache License 2.0](LICENSE). Models and datasets remain subject to their own licenses and terms.

## Citation

```bibtex
@misc{cho2026continuallearningselfprobegradients,
  title={Continual Learning via Self-Probe Gradients},
  author={Dongkyu Cho and Rumi Chunara and Sungmin Cha},
  year={2026},
  eprint={2609.32771},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2609.32771},
}
```
