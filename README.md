# RAG over the EU AI Act: Baseline Retrieval vs. Cross-Encoder Re-Ranking

LMA assignment (CL3.410, Language Models and Agents). Two retrieval pipelines run over the same chunked document and are compared. A third, improved pipeline and a no-retrieval control are added as extras.

| | Baseline | Re-ranking |
|---|---|---|
| Retrieve | bi-encoder `sentence-transformers/all-MiniLM-L6-v2`, FAISS exact inner product, Top 10 | same Top 10 |
| Re-rank | none | cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` scores all 10 (query, chunk) pairs |
| Context | Top 3 by cosine similarity | Top 3 by cross-encoder score |
| Generate | `qwen2.5:7b` via Ollama, temperature 0, seed 42 | identical prompt and settings |

**Document:** [Regulation (EU) 2024/1689 (Artificial Intelligence Act)](https://eur-lex.europa.eu/eli/reg/2024/1689/oj), the official 144-page Official Journal PDF (`data/eu_ai_act_2024_1689.pdf`).

## Repository contents

| Path | What it is |
|---|---|
| `code.ipynb` | The whole assignment, executed with outputs saved |
| `report_final.pdf` | The report (7 pages) |
| `data/` | The source PDF and the chunked corpus (`chunks.jsonl`: 747 chunks, 289 recital, 387 article, 71 annex, with metadata) |
| `results/retrieval_runs.csv` | For every query, all 10 candidates with bi-encoder and cross-encoder scores and ranks, plus a flag for the right clause |
| `results/rank_shifts.csv` | Rank of the right clause before and after re-ranking for every query |
| `results/answers.csv`, `answers.json` | Answer 1 (baseline) and Answer 2 (re-ranking), key-fact coverage |
| `results/answers_norag.csv`, `answers_norag.json` | Answers from the LLM with no retrieved chunks (control) |
| `results/grades.csv`, `grades_by_group.csv`, `grades_cache.json` | Every answer graded correct, partial or wrong by two judge models, with the kind of mistake |
| `results/latency.csv`, `batched_latency.csv`, `scaling.csv` | Latency per stage, batched re-ranking, and the full-corpus cross-encoder scaling experiment |
| `results/ablations.csv`, `ablation_summary.csv`, `pipeline_comparison.csv`, `answers_final_pipeline.csv` | Extra experiments and the final pipeline |
| `results/figures/` | All figures made by the notebook |

## Notebook structure

| Section | Content |
|---|---|
| 1 | Configuration |
| 2 | PDF cleaning with PyMuPDF: drops headers, footers and footnotes; keeps recital numbers; restores the superscript in 10^25 |
| 3 | Structure-aware chunking into recitals, articles (by paragraph) and annexes (by point); each chunk is at most 256 word-pieces (the bi-encoder's limit) |
| 4 to 6 | FAISS index, the baseline pipeline, the cross-encoder re-ranking pipeline |
| 7 | Shared prompt and LLM call, with an answer cache |
| 8 | 49 queries, each with its right clause marked (anchored to the binding clause) |
| 9 to 11 | Retrieval metrics (hit@3, all@3, MRR@10, recall), answers with key-fact coverage, deep dives on failure cases |
| 12 to 13 | Latency profiling and the scaling experiment |
| 14 to 16 | Rank-shift figure, reproducibility check, more figures |
| 17 | Extra experiments: one-change-at-a-time ablations with a dev/test split, and the final pipeline |
| 18 | Failure-mode analysis: keyword trap, exceptions, negation, role limits, clauses buried at rank 4 to 10 |
| 19 | No-retrieval control and graded answers (LLM judges) |
| 20 | Scoring formulas, a worked example of both scores, rank figures, batched re-ranking latency |

Every function is in its own cell, with a markdown explanation above it.

## Results at a glance (49 queries)

| | No retrieval | Baseline | Re-ranking | Final pipeline (extra) |
|---|---|---|---|---|
| hit@3 (a right clause in the context) | n/a | 0.633 | 0.653 | 0.837 |
| all@3 (every needed clause in the context) | n/a | 0.612 | 0.633 | 0.796 |
| MRR@10 | n/a | 0.474 | 0.502 | 0.818 |
| Key-fact coverage of the answer | n/a | 0.673 | 0.769 | 0.810 |
| Mean answer grade, gpt-4.1 judge | 0.265 | 0.735 | 0.724 | 0.694 |
| Mean answer grade, gpt-4.1-mini judge | 0.337 | 0.684 | 0.694 | 0.735 |
| Median retrieval time | n/a | 4.6 ms | 21.8 ms | 84 ms |

- Re-ranking moved four right clauses from rank 4 or 5 into the top 3 (Q16 #4 to #1, Q02 #4 to #2, Q27 #5 to #2, Q47 #5 to #2). It pushed three out of the top 3 (Q20, Q46, Q44), and it cannot help the 11 questions whose right clause is not in the baseline top 10.
- Retrieval clearly matters: the mean grade is 0.27 without it and 0.73 with it. Better retrieval did not clearly give better graded answers. The two judges even disagree on the order of the three setups (they agree exactly on 84% of answers and within one step on 100%).
- The final pipeline indexes only the binding text (no recitals), merges dense and BM25 retrieval, takes the Top 30 and re-ranks them with `ms-marco-MiniLM-L12-v2`. It was chosen on Q01 to Q24. On the held-out Q25 to Q49 its MRR@10 is 0.825.

Median latency per query (RTX 4060 Laptop GPU):

| Stage | Median |
|---|---|
| Baseline retrieval (embed + search) | 4.6 ms |
| Re-ranking retrieval, cross-encoder on GPU | 21.8 ms |
| Re-ranking retrieval, cross-encoder on CPU | 134 ms |
| Cross-encoder over all 747 chunks | 847 ms |
| LLM answer | about 5.3 s |

## How to run

Requires Python 3.11 or newer, an NVIDIA GPU (optional, CPU works but is slower), and [Ollama](https://ollama.com).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name lma-rag
ollama pull qwen2.5:7b
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=lma-rag code.ipynb
```

- LLM answers are cached in `results/answers.json`, so a re-run only calls the LLM for prompts it has not seen. Set `FORCE_REGENERATE = True` in the configuration cell to regenerate everything.
- Section 19 grades answers with OpenAI models (`gpt-4.1` and `gpt-4.1-mini`). It needs a file named `.env` in the project folder with a line `OPENAI_KEY=...`. This file is listed in `.gitignore` and must never be committed. Grades are cached in `results/grades_cache.json`, so a re-run makes no new calls.
- The bi-encoder and cross-encoder download from Hugging Face on first use.

## References

- Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP.
- Nogueira & Cho (2019). *Passage Re-ranking with BERT.* arXiv:1901.04085.
- Nguyen et al. (2016). *MS MARCO: A Human Generated MAchine Reading COmprehension Dataset.*
- Wang et al. (2020). *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers.* NeurIPS.
- Karpukhin et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP.
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
- Robertson & Zaragoza (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.*
- Cormack, Clarke & Büttcher (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods.* SIGIR.
