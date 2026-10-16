# Multilingual E-Sahulat AI: evidence-grounded passport information

Answers Urdu-script, Roman Urdu, English and code-mixed questions about Pakistani passport procedures
from official DGIP pages only. Every answer is an extracted sentence with its document, passage ID,
URL and date. The system asks for clarification on ambiguous questions; declines personal-status,
guarantee, booking, unsafe, time-sensitive and prompt-injection requests; and abstains out of domain.
All models run in our own code: no AI APIs.

## Install (Windows, Command Prompt)
    python -m venv .venv
    .venv\Scripts\activate.bat
    pip install -r requirements.txt

## Optional models (download once)
    python -c "from transformers import MarianMTModel, MarianTokenizer; MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-ur'); MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-ur'); print('mt ok')"
    python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B'); print('qwen ok')"

## Modes (pick automatically by creating an empty file in this folder)
- Full (default): e5 semantic retrieval + extractive QA model.
- LITE_MODE: e5 retrieval, best-sentence extraction (no QA model); for small cloud machines.
- NO_TORCH: for PCs whose native PyTorch/tokenizer libraries crash (outdated Visual C++ runtime):
  semantic retrieval becomes character n-gram TF-IDF (scikit-learn); answers use best-sentence extraction.
Report which mode produced your numbers (evaluate.py prints it on its first line).

## Reproduce everything (three commands; add --generate to compare answer styles)
    python build_corpus.py     # official pages -> data/chunks.jsonl, data/sources.csv
    python evaluate.py         # organizers' dataset -> outputs/*.csv, outputs/summary.json, figures/
    streamlit run app.py       # tabs: Ask · Benchmark explorer · Evaluation · Dataset and sources

## Architecture
query -> normalize.py: Unicode NFKC, Arabic->Urdu letters, digits, whitespace -> Roman Urdu lexicon
-> phonetic-skeleton spelling matching (unknown words only) -> English keyword gloss
-> guards (injection, unsafe, booking, time-sensitive, personal/guarantee, ambiguous fee, domain gate)
-> BM25 + e5 semantic retrieval fused by Reciprocal Rank Fusion -> confidence threshold (tuned on train_dev)
-> extractive answer (xlm-roberta SQuAD2) -> citation check + conflict detection -> cited answer or safe response

## Files
| File | Owner | Role |
|---|---|---|
| build_corpus.py | P2 | Collect and clean official pages; doc IDs, passage IDs, URLs, dates (FR-6) |
| normalize.py | P2 | FR-2 normalization: Unicode, lexicon, phonetic skeleton, gloss |
| engine.py | P1 | FR-3 retrieval, FR-4 grounded answers, FR-5 abstention, guards |
| evaluate.py | P1 | FR-7 validation on the organizers' dataset (Section 7 metrics) |
| generate.py | P1 | Optional generate-then-verify (Qwen3-0.6B) and Urdu translation (opus-mt) |
| app.py | P3 | FR-8 interface: language selection, evidence inspection, benchmark explorer |
| data/esahulat_dataset.jsonl | - | Organizers' 150 records, unchanged |

## Models and data (open-source, loaded locally)
| Component | Exact name | Licence | Purpose |
|---|---|---|---|
| Semantic retrieval | intfloat/multilingual-e5-small | MIT | Cross-lingual embeddings |
| Answer extraction | deepset/xlm-roberta-base-squad2 | CC-BY-4.0 | Span extraction (CPU) |
| Keyword retrieval | rank_bm25 (BM25Okapi) | Apache-2.0 | Lexical baseline |
| Optional: Urdu response | Helsinki-NLP/opus-mt-en-ur | Apache-2.0 | Translates the shown answer (FR-8) |
| Optional: plain-language answer | Qwen/Qwen3-0.6B (CPU, float32, not quantized) | Apache-2.0 | Generate, then verify against the cited passage |
| Interface | Streamlit | Apache-2.0 | Application |
| Documents | DGIP official pages (data/sources.csv) | Public government information | Knowledge base |
| Benchmark | Organizers' Multilingual E-Sahulat dataset | Provided by organizers | Evaluation only |

No transformer was trained. The dataset's gold evidence is NOT indexed (that would leak test answers).
