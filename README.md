# Assignment 3: Few-shot OpenQA with ColBERT Retrieval

## Repository Contents

- `hw-openqa.ipynb`: completed notebook with outputs
- `cs224u-openqa-bakeoff-entry.json`: bakeoff submission file
- `README.md`: summary of results and checks

## Completion Check

Both required deliverables are present and complete.

- The notebook contains `132` cells and has **no stored error outputs**.
- The implemented homework helper tests that were executed in the notebook passed:
  - `build_few_shot_no_context_prompt`
  - `build_few_shot_open_qa_prompt`
  - `get_passages_with_scores`
  - `answer_scoring`
- The bakeoff JSON file exists and contains **400 / 400** answered questions.
- Every JSON entry includes the expected generation fields:
  - `prompt`
  - `generated_text`
  - `generated_tokens`
  - `generated_probs`
  - `generated_answer`
  - `generated_answer_tokens`
  - `generated_answer_probs`

## Main Experimental Results

The notebook evaluates multiple OpenQA settings on `dev_exs` from SQuAD:

| System | Description | Macro-F1 |
|---|---|---:|
| No-context QA | question only, no retrieved passage | `0.02999` |
| Few-shot QA | gold-context prompt with SQuAD examples | `0.07062` |
| Zero-shot OpenQA | retrieved passage + no few-shot examples | `0.06212` |
| Original system | BM25-selected few-shot examples + retrieval score weighting + length-normalized answer scoring + beam search | `0.143387` |

## Original System Summary

The original system improves on the baseline pipeline in four main ways:

1. It retrieves the top passages with ColBERT and converts retrieval scores into pseudo-probabilities.
2. It selects the most relevant SQuAD training examples using BM25 rather than random sampling.
3. It scores candidate answers using both passage relevance and answer likelihood.
4. It applies length normalization and uses beam search for final decoding.

Best reported configuration from the notebook:

- `num_beams = 4`
- peak `macro_f1 = 0.143387`

Comparison table recorded in the notebook:

| num_beams | temperature | macro_f1 |
|---:|---:|---:|
| 2 | n/a | `0.139352` |
| 3 | n/a | `0.138919` |
| 4 | n/a | `0.143387` |
| n/a | 0.1 | `0.124275` |
| n/a | 0.2 | `0.118013` |
| n/a | 0.3 | `0.128947` |
| n/a | 0.4 | `0.116393` |
| n/a | 0.5 | `0.132991` |
| n/a | 0.6 | `0.135449` |
| n/a | 0.7 | `0.121961` |
| n/a | 0.8 | `0.101458` |

## Output File Analysis

The submission file `cs224u-openqa-bakeoff-entry.json` looks structurally correct:

- total questions: `400`
- non-empty generated answers: `400`
- average answer length: `2.48` words
- median answer length: `2`
- min answer length: `1`
- max answer length: `15`

This suggests the system generally generates short answer-like outputs rather than long passages, which is appropriate for the assignment format.

## Notes on Kaggle Execution

This notebook was successfully completed on Kaggle, but the ColBERT `cpu_inference` branch required a few runtime compatibility patches because of newer Kaggle package versions:

- Python 3.12 dataclass compatibility
- newer `transformers` compatibility with ColBERT
- ColBERT CPU/GPU tensor indexing mismatches on Kaggle

These adjustments were applied during execution so the notebook could complete and produce the final bakeoff JSON.# Assignment-3-Few-shot-OpenQA-with-ColBERT-retrieval
