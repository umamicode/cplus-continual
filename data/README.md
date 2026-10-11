# CSQA protocol input

The released CSQA comparison consumes a prepared, four-option JSONL file. Benchmark data is not redistributed with the source code. Obtain it under the source dataset's terms and retain a provenance record with the prepared file.

## File location and validation

Place the protocol input at:

```text
data/inputs/csqa.jsonl
```

It must contain 1,096 examples. Validate the file and record the reported SHA-256 in the experiment metadata:

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

## Provenance record

Keep a small, versioned record alongside the input that identifies the dataset release/configuration and split, selected source rows and their order, four-choice conversion rule, prompt template, and the prepared file's SHA-256. This makes the benchmark instance unambiguous without redistributing it.

The raw CommonsenseQA release has five answer choices; it is not automatically interchangeable with this fixed four-option protocol. Use the documented protocol input or its corresponding preparation record rather than applying an ad hoc choice-removal rule.
