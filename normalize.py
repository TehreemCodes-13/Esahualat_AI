"""normalize.py (owner: P2) - FR-2 query normalization.
1) Unicode fixes: NFKC, Arabic -> Urdu letter forms, remove diacritics and zero-width marks,
   Urdu/Arabic digits -> 0-9, collapse whitespace.
2) Roman Urdu spelling variants -> one canonical spelling (pasport, renu -> passport, renew).
3) Cross-lingual gloss: append English keywords for Urdu-script and Roman Urdu words, so keyword
   search can match English documents. The original words are kept, so meaning is not changed."""
import re
import unicodedata
from difflib import SequenceMatcher

CHAR_MAP = {"\u064a": "\u06cc", "\u0649": "\u06cc",   # Arabic yeh / alef maksura -> Urdu yeh
            "\u0643": "\u06a9",                          # Arabic kaf -> Urdu kaf
            "\u0647": "\u06c1",                          # Arabic heh -> Urdu heh goal
            "\u06d4": ".", "\u060c": ",", "\u061f": "?"}  # Urdu full stop, comma, question mark
for i in range(10):
    CHAR_MAP[chr(0x0660 + i)] = str(i)                  # Arabic-Indic digits
    CHAR_MAP[chr(0x06F0 + i)] = str(i)                  # Extended (Urdu) digits
DIACRITICS = re.compile(r"[\u064b-\u0652\u0670\u0654\u0655]")
ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]")

# Roman Urdu spelling variant -> canonical form (built from the DEV split + common spellings)
VARIANTS = {
    "passport": ["pasport", "paspot", "passprt", "pasprt", "passpord", "passpot", "paasport"],
    "renew": ["renu", "renue", "rinew", "renw", "reneu", "renwe", "rinue"],
    "documents": ["dakuments", "documnts", "docs", "dokument", "dokuments", "document", "dacuments"],
    "fee": ["fees", "fess", "fis", "feez", "fiss", "fis"],
    "karwana": ["karwanay", "karana", "krwana", "karwane", "krwane", "karwaun", "krwani", "karwani"],
    "chahiye": ["chahye", "chaiye", "chahiay", "chahie", "chahiyay", "chaheye"],
    "bacha": ["bachay", "bachy", "bachey", "bachon", "bache", "bachchay", "bachey", "bacho"],
    "gum": ["gm", "gumm", "gumshuda"],
    "chori": ["chory", "chouri"],
    "hai": ["hy", "he", "hay", "hae"],
    "din": ["dino", "dinon", "dn"],
    "urgent": ["urjent", "arjent", "urgnt", "arjant", "urjant"],
    "pages": ["pej", "page", "peges", "safhay", "safhe", "safat", "safhat"],
    "saal": ["sal", "saala"],
    "office": ["daftar", "dafter", "offis", "ofis", "aafis"],
    "naam": ["nam"],
    "shehar": ["shahar", "sheher", "shehr", "shaher"],
    "online": ["onlain", "onlin", "aanlain"],
    "tajdeed": ["tajdid", "tjdeed"],
}
CANON = {v: k for k, vs in VARIANTS.items() for v in vs}

# Canonical Roman Urdu / Urdu word -> English keywords appended for search
GLOSS = {
    "renew": "renewal renew", "tajdeed": "renewal renew", "gum": "lost", "chori": "stolen lost",
    "bacha": "child minor", "din": "days working", "documents": "documents required",
    "kaghzat": "documents required", "kaghaz": "documents required", "kitni": "how much",
    "saal": "years validity", "naam": "name change modify", "shehar": "city office any",
    "office": "office", "pages": "pages", "urgent": "urgent", "online": "online",
    "\u067e\u0627\u0633\u067e\u0648\u0631\u0679": "passport",          # paasport
    "\u062a\u062c\u062f\u06cc\u062f": "renewal renew",                    # tajdeed
    "\u062f\u0633\u062a\u0627\u0648\u06cc\u0632\u0627\u062a": "documents required",  # dastavezat
    "\u06a9\u0627\u063a\u0630\u0627\u062a": "documents required",        # kaghzat
    "\u0641\u06cc\u0633": "fee",                                          # fees
    "\u06af\u0645": "lost",                                               # gum
    "\u0686\u0648\u0631\u06cc": "stolen lost",                            # chori
    "\u0628\u0686\u06d2": "child minor", "\u0628\u0686\u06c1": "child minor", "\u0628\u0686\u0648\u06ba": "child minor",
    "\u0627\u0648\u0642\u0627\u062a": "timings",                          # auqaat
    "\u062f\u0641\u062a\u0631": "office", "\u0622\u0641\u0633": "office",
    "\u062f\u0646": "days working",                                       # din
    "\u0627\u0631\u062c\u0646\u0679": "urgent", "\u0646\u0627\u0631\u0645\u0644": "normal",
    "\u0641\u0627\u0633\u0679": "fast", "\u0679\u0631\u06cc\u06a9": "track",
    "\u0635\u0641\u062d\u0627\u062a": "pages", "\u0633\u0627\u0644": "years",
    "\u0646\u0627\u0645": "name", "\u0634\u06c1\u0631": "city",
    "\u062f\u0631\u062e\u0648\u0627\u0633\u062a": "apply application",
    "\u067e\u0648\u0644\u06cc\u0633": "police", "\u0644\u0627\u0626\u0646": "online",
}


def normalize_text(s):
    """Unicode and whitespace normalization (safe for documents and queries)."""
    s = unicodedata.normalize("NFKC", str(s))
    s = ZERO_WIDTH.sub("", s)
    s = "".join(CHAR_MAP.get(ch, ch) for ch in s)
    s = DIACRITICS.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


GLOSS = {normalize_text(k): v for k, v in GLOSS.items()}


def tokens(text):
    return re.findall(r"\w+", text.lower())


def normalize_query(q, expand=True):
    """Return (query used for search, list of applied steps) for display and evaluation."""
    steps = []
    clean = normalize_text(q)
    if clean != q.strip():
        steps.append("unicode/whitespace fixed")
    words = []
    for w in tokens(clean):
        c = CANON.get(w, w)
        if c != w:
            steps.append(f"{w} -> {c}")
        words.append(c)
    out = " ".join(words)
    if expand:
        extra = []
        for w in words:
            if w in GLOSS and GLOSS[w] not in extra:
                extra.append(GLOSS[w])
        if extra:
            steps.append("added English keywords: " + ", ".join(extra))
            out += " " + " ".join(extra)
    return out, steps


# ---------------- extra domain lexicon (written by the team for the 30 service intents) ----------------
MORE_VARIANTS = {
    "sakta": ["skta", "sakte", "skte", "sakti", "skti"], "pehlay": ["pehle", "pehley", "phle", "pahle", "pehly"],
    "walidain": ["waldain", "walden", "waldein"], "tasveer": ["tasvir", "tasweer", "tasvier"],
    "miyad": ["miyaad", "mayad", "meyad", "miad"], "tabdeel": ["tabdil", "tabdeeli", "tabdili"],
    "alag": ["alug", "allag", "alehda", "alaihda"], "khatam": ["khtm", "khatm", "khattam"],
    "sarkari": ["srkari", "sarkaari"], "mulazim": ["mulazm", "mulazmeen", "mulazmat"],
    "wapsi": ["wapis", "wapas"], "adaigi": ["adaegi", "adaygi", "adayegi"], "talaq": ["talak", "tlaq"],
    "wafat": ["wafaat", "inteqal", "intqal", "intiqal"], "dohri": ["dohra", "dohari", "duhri"],
    "shehriat": ["shehriyat", "shahriyat", "shehryat"], "ijazat": ["ijazt", "ejazat"],
}
for k, vs in MORE_VARIANTS.items():
    for v in vs:
        CANON.setdefault(v, k)
MORE_GLOSS = {
    "sakta": "", "pehlay": "before", "walidain": "parents parent", "tasveer": "photo photograph",
    "miyad": "validity years", "tabdeel": "change modify", "alag": "separate", "khatam": "exhausted finished",
    "sarkari": "government", "mulazim": "employee employees", "wapsi": "refund", "adaigi": "payment pay",
    "talaq": "divorce custody", "wafat": "deceased death", "dohri": "dual", "shehriat": "nationality citizenship",
    "ijazat": "permission noc", "hasil": "get obtain issued", "bacha": "child minor", "expire": "expiry expire",
    "ungliyon": "fingerprints biometrics", "nishan": "fingerprints biometrics", "interview": "interview",
    "parents": "parents", "jama": "submit", "bayan": "statement",
    "\u0648\u0627\u0644\u062f\u06cc\u0646": "parents parent", "\u062a\u0635\u0648\u06cc\u0631": "photo photograph",
    "\u0645\u06cc\u0639\u0627\u062f": "validity years expiry", "\u062a\u0628\u062f\u06cc\u0644\u06cc": "change modify",
    "\u0627\u0644\u06af": "separate", "\u0639\u0644\u06cc\u062d\u062f\u06c1": "separate", "\u062e\u062a\u0645": "exhausted finished",
    "\u0633\u0631\u06a9\u0627\u0631\u06cc": "government", "\u0645\u0644\u0627\u0632\u0645": "employee",
    "\u0648\u0627\u067e\u0633\u06cc": "refund", "\u0627\u062f\u0627\u0626\u06cc\u06af\u06cc": "payment pay",
    "\u0637\u0644\u0627\u0642": "divorce custody", "\u0648\u0641\u0627\u062a": "deceased death",
    "\u0627\u0646\u062a\u0642\u0627\u0644": "deceased death", "\u062f\u0648\u06c1\u0631\u06cc": "dual",
    "\u0634\u06c1\u0631\u06cc\u062a": "nationality citizenship", "\u0627\u062c\u0627\u0632\u062a": "permission noc",
    "\u0628\u0627\u0644\u063a": "adult 18", "\u0639\u0627\u0645": "ordinary", "\u0634\u0646\u0627\u062e\u062a\u06cc": "cnic identity",
    "\u06a9\u0627\u0631\u0688": "card cnic", "\u0627\u0646\u0679\u0631\u0648\u06cc\u0648": "interview",
    "\u0627\u0646\u06af\u0644\u06cc\u0648\u06ba": "fingerprints biometrics", "\u0646\u0634\u0627\u0646\u0627\u062a": "fingerprints biometrics",
    "\u06a9\u0648\u0631\u06cc\u0626\u0631": "courier delivery", "\u062a\u0631\u0633\u06cc\u0644": "delivery courier",
    "\u0627\u06c1\u0644": "eligible", "\u062d\u0627\u0635\u0644": "get obtain issued", "\u0633\u0627\u0628\u0642\u06c1": "previous",
    "\u067e\u0631\u0627\u0646\u0627": "previous old", "\u0631\u067e\u0648\u0631\u0679": "report",
}
for k, v in MORE_GLOSS.items():
    if v:
        GLOSS[normalize_text(k)] = v

# ---------------- phonetic-skeleton spelling normalization (our method) ----------------
VOWELS = set("aeiouy")
VOCAB, SKELETON = set(), {}


def skeleton(w):
    """Consonant skeleton: collapse doubled letters, keep the first letter, drop later vowels.
    sakta and skta -> skt; pehle and pehlay -> phl; rinew and renew -> rnw; paasport -> psprt."""
    w = re.sub(r"(.)\1+", r"\1", w)
    return w[:1] + "".join(c for c in w[1:] if c not in VOWELS)


def set_vocab(words):
    """Build the skeleton -> known word map from words we trust (document text, train_dev queries, lexicon)."""
    from collections import Counter
    counts = Counter(w for w in words if w.isascii() and w.isalpha() and len(w) >= 3)
    for k in list(CANON.values()) + list(GLOSS.keys()):
        if k.isascii():
            counts[k] += 50                                   # lexicon words win ties
    VOCAB.clear(); VOCAB.update(counts)
    SKELETON.clear()
    for w, _ in counts.most_common():
        sk = skeleton(w)
        if len(sk) >= 2:
            SKELETON.setdefault(sk, w)


def normalize_query_mode(q, mode="full"):
    """mode: 'off' (raw), 'lexicon' (Unicode + variants + gloss), 'full' (lexicon + skeleton matching)."""
    if mode == "off":
        return q, []
    steps = []
    clean = normalize_text(q)
    if clean != q.strip():
        steps.append("unicode/whitespace fixed")
    words = []
    for w in tokens(clean):
        c = CANON.get(w, w)
        if c != w:
            steps.append(f"{w} -> {c} (lexicon)")
        elif mode == "full" and w.isascii() and w.isalpha() and len(w) >= 3 and w not in VOCAB:
            sk, cand = skeleton(w), SKELETON.get(skeleton(w))
            if (cand and cand != w and len(sk) >= 3 and abs(len(cand) - len(w)) <= 2
                    and SequenceMatcher(None, w, cand).ratio() >= 0.7):     # guard: no 'gyta' -> 'gate'
                c = CANON.get(cand, cand)
                steps.append(f"{w} ~ {c} (spelling skeleton '{sk}')")
        words.append(c)
    out, extra = " ".join(words), []
    for w in words:
        if w in GLOSS and GLOSS[w] not in extra:
            extra.append(GLOSS[w])
    if extra:
        steps.append("added English keywords: " + ", ".join(extra))
        out += " " + " ".join(extra)
    return out, steps
