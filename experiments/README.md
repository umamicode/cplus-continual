# Experiments

`run_csqa_main.sh` is the public entry point for the released JAX/TPU reproduction protocol. It runs the fixed Qwen3-0.6B CommonsenseQA comparison with seed `0`, `N=512` self-probes, anchor budgets `8` and `16`, and SFT, Replay, CPL, and CPLUS.

```bash
bash experiments/run_csqa_main.sh \
  data/inputs/csqa.jsonl \
  weights \
  artifacts/csqa-main
```

Pass a fourth positional argument to change the tensor-parallel degree, for example `4`. The launcher validates the input, records the exact commands as `csqa-main-commands.sh` in the output root, and then executes the nine TPU jobs serially.

The output root is a protocol boundary. Start with a fresh directory if the input file, downloaded model revision, or TPU execution settings change.
