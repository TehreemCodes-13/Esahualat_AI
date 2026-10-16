# Multilingual E-Sahulat AI: report

**Result in one line:** On the organizers' 60-query test split (tuned on train_dev only), hybrid
retrieval with full normalization reaches Recall@5 = [x] (raw BM25 baseline [y]); behaviour accuracy
across the validation categories is [z]; citation support rate [c]; median latency [m] ms on a lab CPU.

| Rubric criterion (points) | Our evidence |
|---|---|
| NLP innovation (20) | Phonetic-skeleton normalization, cross-lingual gloss, hybrid RRF, ordered guards |
| Retrieval quality (20) | Recall@5/MRR@5, 3 methods x 3 normalization modes, per language (outputs/retrieval.csv) |
| Answer reliability (20) | Extractive answers, citation support, abstention, clarification, conflict notice; optional generation gated by our verifier |
| Validation rigor (15) | Organizers' splits respected, baselines, ablation, failures (outputs/failures.csv) |
| Prototype usability (10) | Language selection, evidence inspection, benchmark explorer, live URL |
| Practical impact (10) | Roman Urdu/Urdu users, CPU-only, offline-capable, privacy |
| Presentation (5) | Live demo against the organizers' own queries |

## 1. Problem and users  ## 2. Documents and provenance  ## 3. Preprocessing and normalization
## 4. System design  ## 5. Validation protocol  ## 6. Results (actual measured values only)
## 7. Error analysis (top failure types with examples)  ## 8. Limitations (see DEFENSE.md)
## 9. Models and licences  ## 10. AI-use statement
As the organizers allowed, we used Claude to help write and debug code. Design decisions, the
evaluation and the findings are ours. The prototype calls no AI service.
