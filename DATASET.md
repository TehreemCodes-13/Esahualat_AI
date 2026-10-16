# Data documentation

## Knowledge base (our collection, built by build_corpus.py)
- data/sources.csv: doc_id, title, url, retrieved date, number of passages
- data/chunks.jsonl: id, doc_id, passage (e.g. D01-P004), title, section, url, retrieved, text
Preprocessing: scripts/menus removed; table rows flattened with their headers; merged cells filled;
consecutive blocks under one heading grouped into ~120-word passages; heading-only passages removed;
Unicode normalized.

## Benchmark (organizers): data/esahulat_dataset.jsonl, 150 records, unchanged
- 30 intents (F01-F30) x English, Urdu, Roman Urdu in train_dev (90) + one Roman Urdu spelling
  variant per intent in test (SV001-SV030)
- 30 hard cases in test (N001-N030): ambiguous, personal/guarantee, out-of-domain, prompt injection,
  time-sensitive, booking, conflicting evidence, unsafe, partially answerable
How we use it: train_dev builds the normalization vocabulary and tunes the confidence threshold
(with our own 10 negative queries). test is used only for measurement. The gold evidence is used
only to judge relevance (a passage is relevant if it contains >= 60% of the gold evidence's content
words); it is never indexed or shown to users.
