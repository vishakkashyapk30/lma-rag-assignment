# RAG over the EU AI Act: Baseline Retrieval vs. Cross-Encoder Re-Ranking

LMA assignment. Two retrieval pipelines run over the same chunked document and are compared:

| | Pipeline 1 (baseline) | Pipeline 2 (re-ranking) |
|---|---|---|
| Retrieve | bi-encoder `sentence-transformers/all-MiniLM-L6-v2`, FAISS exact inner product, Top 10 | same Top 10 |
| Re-rank | none | cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` scores all 10 (query, chunk) pairs |
| Context | Top 3 by cosine similarity | Top 3 by cross-encoder score |
| Generate | `qwen2.5:7b` via Ollama, temperature 0, seed 42 | identical prompt and settings |

**Document:** [Regulation (EU) 2024/1689 (Artificial Intelligence Act)](https://eur-lex.europa.eu/eli/reg/2024/1689/oj), the official 144-page Official Journal PDF (`data/eu_ai_act_2024_1689.pdf`).

## Repository contents

| Path | What it is |
|---|---|
| `rag_rerank.ipynb` | The whole assignment, executed with outputs saved |
| `data/` | The source PDF and the chunked corpus (`chunks.jsonl`: 747 chunks, 289 recital, 387 article, 71 annex, with metadata) |
| `results/retrieval_runs.csv` | For every query, all 10 candidates with bi-encoder and cross-encoder scores and ranks, plus a gold flag |
| `results/rank_shifts.csv` | Best gold rank before and after re-ranking for every query |
| `results/answers.csv`, `answers.json` | Answer 1 vs Answer 2, key-fact coverage, and empty manual-label columns |
| `results/latency.csv`, `scaling.csv` | Latency per stage, and the full-corpus cross-encoder scaling experiment |
| `results/ablations.csv`, `ablation_summary.csv`, `pipeline_comparison.csv`, `answers_pipeline3.csv` | Extra experiments (§17) |
| `results/figures/` | Chunk-length histogram, rank-shift chart, latency chart |
| `report.pdf` | The report |

## Notebook structure

| § | Content |
|---|---|
| 0–1 | Overview, references, config |
| 2 | PDF cleaning with PyMuPDF: drops headers, footers and footnotes; keeps recital numbers; restores the superscript in 10^25 |
| 3 | Structure-aware chunking into recitals, articles (by paragraph) and annexes (by point); each chunk is at most 256 word-pieces (the bi-encoder's limit) |
| 4–7 | FAISS index, both pipelines, shared prompt, LLM wrapper with answer cache |
| 8 | 49 queries with gold labels anchored to the binding clause |
| 9 | Retrieval metrics: hit@3, all@3, MRR@10, recall@10, rank shifts |
| 10 | Answer generation and key-fact coverage |
| 11 | Deep dives on failure cases |
| 12 | Latency profiling |
| 13 | Scaling experiment |
| 14 | Rank-shift figure |
| 15 | Reproducibility check |
| 16 | More figures: score scatter, rank transitions, per-category metrics, context composition, answer coverage, latency distribution, scaling, full-corpus ranks |
| 17 | Extra experiments: one-factor-at-a-time ablations with a dev/test split, and Pipeline 3 |

## Results at a glance (49 queries)

| | Pipeline 1 (bi-encoder) | Pipeline 2 (+ re-rank) | Pipeline 3 (extra, §17) |
|---|---|---|---|
| hit@3 (any gold chunk in the context) | 0.633 | 0.653 | 0.837 |
| all@3 (every required clause in the context) | 0.612 | 0.633 | 0.796 |
| MRR@10 | 0.474 | 0.502 | 0.818 |
| Answer key-fact coverage | 0.673 | 0.769 | 0.810 |
| Median retrieval latency | 4.6 ms | 22.3 ms | 88 ms |

Pipeline 3 is not part of the assignment spec. It indexes only the binding text (no recitals), merges dense and BM25 retrieval, takes the Top 30 and re-ranks them with `ms-marco-MiniLM-L12-v2`. It was chosen on Q01–Q24; on the held-out Q25–Q49 its MRR@10 is 0.825.

Showcase cases, where the gold chunk sat at baseline rank 4–10 and re-ranking moved it into the Top 3:

| Query | Gold clause | Rank |
|---|---|---|
| Q02 | Art 5(1)(e), scraping | #4 → #2 |
| Q16 | Art 23(1), importer | #4 → #1 |
| Q27 | Art 5(1)(g), biometric categorisation | #5 → #2 |
| Q47 | Annex III 5(b), fraud-detection carve-out | #5 → #2 |

The notebook also lists queries where re-ranking made the gold rank worse, and queries whose gold clause was never in the bi-encoder Top 10.

Median latency per query (RTX 4060 Laptop GPU):

| Stage | Median |
|---|---|
| Pipeline 1 retrieval | 4.6 ms |
| Pipeline 2 retrieval, re-rank on GPU | 22.3 ms |
| Pipeline 2 retrieval, re-rank on CPU | 147 ms |
| LLM generation | ~5.3 s (median) |

Figures are in `results/figures/`.

## How to run

Requires Python ≥ 3.11, an NVIDIA GPU (optional; CPU works, only slower), and [Ollama](https://ollama.com).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name lma-rag
ollama pull qwen2.5:7b
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=lma-rag rag_rerank.ipynb
```

LLM answers are cached in `results/answers.json`, so a re-run only calls the LLM for prompts it has not seen. Set `FORCE_REGENERATE = True` in the config cell to regenerate everything. The bi-encoder and cross-encoder download from Hugging Face on first use.

## References

- Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP.
- Nogueira & Cho (2019). *Passage Re-ranking with BERT.* arXiv:1901.04085.
- Nguyen et al. (2016). *MS MARCO: A Human Generated MAchine Reading COmprehension Dataset.*
- Wang et al. (2020). *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers.* NeurIPS.
- Karpukhin et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP.
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
