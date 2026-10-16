"""engine.py (owner: P1) - retrieval, safety guards and evidence-grounded answers.
Retrieval: BM25 keyword, e5 semantic, hybrid (Reciprocal Rank Fusion), each with normalization
mode off / lexicon / full (for the ablation). Answers are extracted from official passages only.
All models are loaded locally; no API calls. LITE mode (file LITE_MODE or ESAHULAT_LITE=1) skips the
QA model for small cloud machines and picks the best evidence sentence instead."""
import json, os, re, time
import numpy as np
from rank_bm25 import BM25Okapi
from normalize import normalize_query_mode, normalize_text, set_vocab, tokens

EMBED = "intfloat/multilingual-e5-small"   # MIT, 100+ languages incl. Urdu
QA = "deepset/xlm-roberta-base-squad2"     # CC-BY-4.0, extractive question answering
# NO_TORCH mode (file NO_TORCH): for PCs whose native PyTorch/tokenizer libraries crash. The semantic
# retriever becomes a character n-gram TF-IDF model (scikit-learn); answers use best-sentence extraction.
NO_TORCH = os.path.exists("NO_TORCH") or os.environ.get("ESAHULAT_NO_TORCH") == "1"
LITE = NO_TORCH or os.path.exists("LITE_MODE") or os.environ.get("ESAHULAT_LITE") == "1"
SEMANTIC_NAME = "char n-gram TF-IDF (no-torch fallback)" if NO_TORCH else EMBED

# ---------- guards, written from the brief's validation-case categories (rule-based, transparent) ----------
G = lambda p: re.compile(p, re.I)
INJECTION = G(r"ignore (all |your |the )?(previous |above |prior )?(rules|instructions)|disregard (the |your )?(rules|instructions)|"
              r"system prompt|you are now|pretend|jailbreak|\binvent\b|make up|fabricate|"
              r"(don'?t|do not|without) (cite|citing|sources?)|cite na karo|jawab bana do|bana do jawab|"
              r"\u062d\u0648\u0627\u0644\u06c1 \u0645\u062a|\u062c\u0648\u0627\u0628 \u0628\u0646\u0627")  # hawala mat, jawab bana
UNSAFE = G(r"bypass|forge|forged|fake|counterfeit|cheat|trick the|jaali|jali|\u062c\u0639\u0644\u06cc|\u062f\u06be\u0648\u06a9")
ACTION = G(r"book (an |my |the )?appointment|appointment .*(confirm|book)|(confirm|book|schedule) kar|kar do\b|"
           r"submit (my|the) (form|application) for me|apply for me|\u0628\u06a9 \u06a9\u0631")
TIME_SENSITIVE = G(r"still (current|valid|applicable|the same)|currently applicable|definitely|\btoday\b|abhi bhi|ab bhi|aaj bhi|aj bhi|"
                   r"latest fee|up to date|\u0622\u062c \u0628\u06be\u06cc|\u0627\u0628 \u0628\u06be\u06cc")
PERSONAL = G(r"guarantee|guaranteed|promise|my (current )?(passport )?application status|\bstatus\b|track my|"
             r"(will|would) (my )?(passport|application) (arrive|be approved|be ready)|be approved|confirm (that )?my|"
             r"kab .{0,25}(pohanch|milega|mile ga|milay ga|aaye|aa jaye)|pakka|yaqeeni|exact date|"
             r"\u0636\u0645\u0627\u0646\u062a|\u0644\u0627\u0632\u0645\u0627|\u0627\u0633\u0679\u06cc\u0679\u0633|\u06cc\u0642\u06cc\u0646\u06cc")
FEE = {"fee", "cost", "price", "charges", "charge", "paise", "rupay", "rupees"}
HOWMUCH = {"much", "kitni", "kitna", "kitne", "cost", "price", "\u06a9\u062a\u0646\u06cc", "\u06a9\u062a\u0646\u0627"}
SPECIFIC = {"pages", "36", "72", "100", "years", "year", "saal", "5", "10", "urgent", "normal", "fast", "track", "epassport",
            "mrp", "lost", "second", "quadruple", "double", "online", "refund", "pay", "payment", "renewal", "child", "minor"}
DOMAIN = {"passport", "passports", "epassport", "cnic", "nicop", "crc", "frc", "fee", "renewal", "renew", "documents",
          "document", "child", "minor", "parents", "parent", "police", "lost", "office", "dgip", "nadra", "validity",
          "urgent", "normal", "fast", "processing", "biometrics", "photo", "interview", "refund", "payment", "online",
          "pages", "name", "nationality", "noc", "custody", "divorce", "deceased", "apply", "application", "token",
          "delivery", "courier", "expiry", "expired", "expire", "employee", "government", "ordinary", "adult", "cost",
          "separate", "exhausted", "visa", "travel", "abroad", "citizens", "eligible", "dual"}
TIME_WORDS = {"days", "din", "long", "time", "processing", "duration", "kitne"}
CONFLICT_Q = G(r"another page|other page|different|conflict|contradict|which (one )?is correct|which is right|sources disagree")
try:
    SELECT = json.load(open("outputs/config.json")).get("select", "best3")
except Exception:
    SELECT = "best3"
CATEGORIES = {"urgent": "urgent", "normal": "normal", "fast track": "fast track"}


def body(chunk):
    head = chunk["section"] + ". "
    return chunk["text"][len(head):] if chunk["text"].startswith(head) else chunk["text"]


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+", text) if len(s.split()) >= 3]


def sentence_with(text, span):
    span = span.strip()
    return next((s for s in sentences(text) if span and span in s), span)


def durations(text):
    found = []
    for seg in re.split(r"(?<=\.)\s+", text.lower()):
        nums = [(m.start(), int(m.group(1))) for m in re.finditer(r"(\d+)\s*working days", seg)]
        cats = [(m.start(), c) for name, c in CATEGORIES.items() for m in re.finditer(name, seg)]
        for pos, days in nums:
            if cats:
                found.append((min(cats, key=lambda c: abs(c[0] - pos))[1], days))
    return found


class Engine:
    def __init__(self, path="data/chunks.jsonl", dataset="data/esahulat_dataset.jsonl"):
        raw = [json.loads(line) for line in open(path, encoding="utf-8")]
        self.chunks = [c for c in raw if len(body(c).split()) >= 8]
        texts = [normalize_text(c["text"]) for c in self.chunks]
        vocab = [w for t in texts for w in tokens(t)]
        if os.path.exists(dataset):                                   # train_dev queries only, never test
            vocab += [w for line in open(dataset, encoding="utf-8") for r in [json.loads(line)]
                      if r["split"] == "train_dev" for w in tokens(r["query"])]
        set_vocab(vocab)
        self.index = BM25Okapi([tokens(t) for t in texts])
        if NO_TORCH:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.cvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
            self.vectors = self.cvec.fit_transform(texts)               # rows are L2-normalised
            self.embedder = None
        else:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(EMBED)
            self.vectors = self.embedder.encode(["passage: " + t for t in texts], normalize_embeddings=True)
        self.qa = None

    # ---------- retrieval ----------
    def search_bm25(self, q, k=5, mode="full"):
        return [int(i) for i in np.argsort(-self.index.get_scores(tokens(normalize_query_mode(q, mode)[0])))[:k]]

    def dense_scores(self, q, mode="full"):
        if NO_TORCH:
            qv = self.cvec.transform([normalize_query_mode(q, mode)[0]])
            return (self.vectors @ qv.T).toarray().ravel()
        qv = self.embedder.encode(["query: " + normalize_query_mode(q, mode)[0]], normalize_embeddings=True)[0]
        return self.vectors @ qv

    def search_dense(self, q, k=5, mode="full"):
        return [int(i) for i in np.argsort(-self.dense_scores(q, mode))[:k]]

    def search_hybrid(self, q, k=5, mode="full", pool=20):
        fused = {}
        for ranking in (self.search_bm25(q, pool, mode), self.search_dense(q, pool, mode)):
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0) + 1 / (60 + rank)
        return sorted(fused, key=fused.get, reverse=True)[:k]

    def search(self, q, method="hybrid", k=5, mode="full"):
        return {"bm25": self.search_bm25, "dense": self.search_dense, "hybrid": self.search_hybrid}[method](q, k, mode)

    # ---------- guards ----------
    @staticmethod
    def guard(q, norm_q):
        words = set(tokens(norm_q))
        if INJECTION.search(q):
            return "refused"
        if UNSAFE.search(q):
            return "unsafe"
        if ACTION.search(q):
            return "action"
        if TIME_SENSITIVE.search(q):
            return "time_sensitive"
        if PERSONAL.search(q):
            return "unsupported"
        if words & (FEE | {"cost", "price"}) and words & HOWMUCH and not (words & SPECIFIC) and len(tokens(q)) <= 10:
            return "clarify"
        if not (words & DOMAIN):
            return "abstain"                                           # domain gate: nothing passport-related
        return None

    def conflicts(self, q, sources=None):
        """Case 10: same category with different working-day values. Looks at the top 10 passages."""
        if not (set(tokens(normalize_query_mode(q)[0])) & TIME_WORDS or CONFLICT_Q.search(q)):
            return []
        seen = {}
        for s in [self.chunks[i] for i in self.search(q, "hybrid", 10)]:
            for cat, days in durations(body(s)):
                seen.setdefault(cat, {}).setdefault(days, s)
        return [{"category": c, "values": {d: f"{s['section']} ({s['passage']})" for d, s in v.items()}}
                for c, v in seen.items() if len(v) > 1]

    def best_sentence(self, q, sources):
        """LITE / fallback answer: the evidence sentence sharing most words with the normalized query."""
        qw = set(tokens(normalize_query_mode(q)[0]))
        cands = [(len(qw & set(tokens(s))) - 0.1 * rank, s, src) for rank, src in enumerate(sources[:3])
                 for s in sentences(body(src))]
        if not cands:
            return body(sources[0])[:300], sources[0]
        _, s, src = max(cands, key=lambda c: c[0])
        return s, src

    def answer(self, q, method="hybrid", k=5, threshold=0.80, mode="full", style="extractive", select=None):
        t0 = time.perf_counter()
        norm_q, steps = normalize_query_mode(q, mode)
        out = {"query": q, "normalized": norm_q, "steps": steps, "status": "answer", "answer": None, "source": None,
               "sources": [], "confidence": 0.0, "verified": False, "conflicts": []}
        status = self.guard(q, norm_q)
        if status in ("unsupported", "time_sensitive", None):             # show related official evidence too
            out["sources"] = [self.chunks[i] for i in self.search(q, method, k, mode)]
            out["confidence"] = float(self.dense_scores(q, mode).max())
        if status is None and out["confidence"] < threshold:
            status = "abstain"
        if status:
            out["status"] = status
        else:
            cands = out["sources"][:1] if (select or SELECT) == "top1" else out["sources"][:3]
            try:
                if LITE:
                    raise RuntimeError("lite mode")
                if self.qa is None:
                    from transformers import pipeline
                    self.qa = pipeline("question-answering", model=QA, device=-1)
                best = max((dict(self.qa(question=norm_q, context=body(s)), src=s) for s in cands),
                           key=lambda r: r["score"])
                ans, src = sentence_with(body(best["src"]), best["answer"]), best["src"]
                if len(ans.split()) < 3:
                    ans, src = self.best_sentence(q, cands)
            except Exception:
                ans, src = self.best_sentence(q, cands)
            out.update(answer=ans.strip(), source=src, conflicts=self.conflicts(q, out["sources"]))
            out["verified"] = normalize_text(ans) in normalize_text(src["text"]) and src["url"].startswith("http")
            if style == "generate" and not LITE:                          # optional: generate, then verify
                from generate import llm_answer, verify
                g0 = time.perf_counter()
                gen = llm_answer(q, body(src))
                ok, why = verify(gen, body(src), q)
                out.update(generated=gen if ok else None, gen_candidate=gen, gen_check=why,
                           gen_ms=(time.perf_counter() - g0) * 1000)
        out["latency_ms"] = (time.perf_counter() - t0) * 1000
        return out
