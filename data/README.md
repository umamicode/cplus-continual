# CSQA input

CommonsenseQA is a public benchmark. The JAX/TPU runner consumes a task-formatted, four-option JSONL input.

## File location and validation

Place the protocol input at:

```text
data/inputs/csqa.jsonl
```

Validate the file before running:

```bash
python scripts/validate_csqa_input.py data/inputs/csqa.jsonl
```

## Row format

Each non-empty line is one JSON object:

```json
{
  "question": "Question: ...\nOptions:\nA: ...\nB: ...\nC: ...\nD: ...\n\nPlease reply with only A, B, C, or D.",
  "label": "B"
}
```

Requirements:

- `question` contains exactly one option line for each of `A`, `B`, `C`, and `D`, and no additional answer-option lines.
- `label` is exactly one of `A`, `B`, `C`, or `D`.
- Preserve the protocol row order. Use the same prepared file for SFT, Replay, CPL, and CPLUS.
- Do not include answer explanations, private annotations, model predictions, or completion metadata.
