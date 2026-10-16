# Every design decision, and why (P1, P2, P3: learn your rows)

| # | Decision | Why we chose it | Alternative we rejected, and why | Evidence to show |
|---|---|---|---|---|
| 1 | Retrieval first, extractive answers | Brief: "not merely a chatbot that produces fluent text"; a copied official sentence cannot be invented | Generative LLM: can hallucinate fees, slow on CPU | Citation support rate; verified badge |
| 2 | Hybrid BM25 + e5, fused by RRF | FR-3 needs two approaches; keywords catch "72 pages", meaning catches paraphrase and Urdu | Weighted score sum: scales differ (BM25 0-20, cosine 0-1) and needs tuning data | Retrieval table, 3 methods |
| 3 | multilingual-e5-small | MIT, 100+ languages incl. Urdu, fast on a lab CPU | Larger embedders: slower, no CPU budget | Latency median/p90 |
| 4 | Three-layer normalization | FR-2 names Unicode, whitespace and Roman spelling | Translating every query: extra model, latency, new errors | Ablation off / lexicon / full |
| 5 | Phonetic skeleton matching (our method) | Roman Urdu has no standard spelling; variation is mostly vowels (sakta/skta, pehle/pehlay) | Edit distance on every word: changes correct words; we only touch words NOT in our vocabulary, so meaning is preserved (FR-2 wording) | Normalization trace in the app; ablation |
| 6 | Hand-written domain gloss (~100 terms) | Lets BM25 match English documents from Urdu/Roman words; transparent | Machine translation: heavier, opaque | Ablation, train_dev by language |
| 7 | Rule-based guards, fixed order | The dataset has no negative examples outside test, so a classifier cannot be trained honestly; rules are explainable | LLM judge: an API or a big model; untestable | Behaviour per category; 0 false blocks on 120 answerable queries |
| 8 | Threshold tuned on train_dev + our own negatives | Section 7: keep test separate from tuning | Tuning on test: inflated scores | outputs/threshold.txt, summary |
| 9 | Gold evidence never indexed | Indexing it would leak the test answers | "Easy 100%": dishonest | We say so in the report |
| 10 | Relevance = 60% of gold content words | Gold evidence is a summarized excerpt, exact match is impossible | Exact string match: marks correct passages wrong | Rule stated in summary.json |
| 11 | Urdu mode frames the verbatim English evidence | Translating official text could create unsupported claims | Showing the dataset's Urdu answers: leakage | Reference-answer F1 reported separately |
| 12 | Conflict detection for processing days | Case 10; DGIP pages state different durations | Silently picking one | Conflict notice in the app |
| 14 | Generation is optional and verified, never the default | Brief Section 10: unsupported fluent answers score lower; a 0.6B model can still mix facts | LLM as main engine: fluent but unverifiable | Verifier acceptance rate; answer F1 extractive vs generated (evaluate.py --generate) |
| 15 | Our verifier: copied numbers + 70% content-word support | Fees and days are the costliest errors; numbers must come from the evidence or the question | Trusting the LLM, or an LLM-as-judge (another model, untestable) | Rejections logged with reasons (outputs/generation.csv) |
| 16 | Urdu answers by translating the shown English answer (opus-mt-en-ur) | FR-8 response language; a dedicated MT model is faster and more faithful than a 0.6B LLM writing Urdu | Generating Urdu with Qwen: weak Urdu, slow on CPU | chrF vs the organizers' expected_answer_ur |
| 13 | LITE mode for the cloud | Free hosting has ~1-2.7 GB RAM | Paid hosting | Live URL |

| 17 | NO_TORCH fallback mode | Lab PCs with an outdated Visual C++ runtime crash in native torch/tokenizer libraries; a char n-gram retriever keeps the full pipeline running on scikit-learn | Giving up, or requiring admin installs | Mode printed by evaluate.py |

| 18 | Answer-source rule chosen on train_dev (best-of-3 by QA score vs top-1 retrieved) | Citation support is a required metric; the choice is made on dev data, never on test | Picking by intuition | outputs/config.json, printed by evaluate.py |
| 19 | Skeleton matching guarded (skeleton >= 3 letters, similar length, >= 70% spelling similarity) | We found 'gyta' wrongly mapped to 'gate'; the guard keeps FR-2's 'without changing meaning' | Unguarded fuzzy matching | Normalization trace in the app |

## Honest limitations (say these before the judges ask)
- Guard rules were written from the brief's categories, but we had seen the test hard cases, so
  behaviour accuracy on N001-N030 is optimistic; the 120 answerable queries show 0 false blocks.
- Information inside images on DGIP pages is invisible to text retrieval.
- The relevance rule is automatic; we spot-checked a sample by hand.
- Small test set: one query is about 1.7 percentage points; we only claim large differences.
