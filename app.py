"""app.py (owner: P3) - Multilingual E-Sahulat AI, chat interface.   Run:  streamlit run app.py"""
import html, json, pathlib, uuid
import pandas as pd
import streamlit as st
from engine import LITE, NO_TORCH, SEMANTIC_NAME, Engine

st.set_page_config(page_title="E-Sahulat AI", page_icon="🛂", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
#MainMenu, footer, .stDeployButton, [data-testid="stToolbar"], [data-testid="stDecoration"] {display:none !important;}
header[data-testid="stHeader"] {background:transparent;}
html, body, [class*="css"], .stMarkdown, button, input, textarea {
  font-family: -apple-system, "Segoe UI", Inter, Roboto, "Helvetica Neue", Arial, "Noto Naskh Arabic", sans-serif;}
.block-container {max-width: 820px; padding-top: 1.5rem; padding-bottom: 8rem;}
[data-testid="stSidebar"] {background:#f9f9f9; border-right:1px solid #ececec;}
[data-testid="stSidebar"] [role="radiogroup"] label {padding:.42rem .6rem; border-radius:10px; width:100%; margin:0;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background:#ececec;}
[data-testid="stSidebar"] .stButton button {width:100%; justify-content:flex-start; border:none; background:transparent;
  border-radius:10px; padding:.42rem .6rem; color:#0d0d0d; font-size:.9rem;}
[data-testid="stSidebar"] .stButton button:hover {background:#ececec;}
.es-brand {font-weight:650; font-size:1.05rem; padding:.2rem .4rem .8rem; color:#0d0d0d;}
.es-section {font-size:.72rem; color:#8f8f8f; font-weight:600; letter-spacing:.02em; padding:.9rem .6rem .25rem;}
.es-hero {text-align:center; margin:16vh 0 1.6rem;}
.es-hero h1 {font-size:1.9rem; font-weight:600; color:#0d0d0d; margin:0 0 .5rem;}
.es-hero p {color:#5d5d5d; font-size:.95rem; margin:0;}
[data-testid="stMain"] .stButton button {width:100%; text-align:left; border:1px solid #e5e5e5;
  border-radius:16px; padding:.8rem 1rem; background:#fff; color:#0d0d0d; min-height:4.2rem; font-size:.9rem;}
[data-testid="stMain"] .stButton button:hover {background:#f7f7f7; border-color:#d9d9d9;}
.es-user {display:flex; justify-content:flex-end; margin:1.4rem 0 .6rem;}
.es-user div {background:#f4f4f4; border-radius:20px; padding:.7rem 1.05rem; max-width:78%; color:#0d0d0d; line-height:1.55;}
.es-bot {display:flex; gap:.85rem; margin:.4rem 0 .1rem; align-items:flex-start;}
.es-avatar {width:30px; height:30px; border-radius:50%; background:#0d0d0d; color:#fff; display:flex; align-items:center;
  justify-content:center; font-size:.7rem; font-weight:600; flex-shrink:0; margin-top:1px;}
.es-body {flex:1; color:#0d0d0d; line-height:1.7; font-size:1rem; overflow-wrap:anywhere;}
.es-note {color:#6b6b6b; font-size:.82rem; font-style:italic; margin-top:.25rem;}
.es-quote {border-left:3px solid #e5e5e5; padding:.1rem 0 .1rem .8rem; color:#4d4d4d; font-size:.92rem; margin-top:.6rem;}
.es-chip {display:inline-flex; align-items:center; gap:.3rem; font-size:.74rem; font-weight:550; padding:.2rem .65rem;
  border-radius:999px; margin:.7rem .35rem 0 0;}
.ok {background:#e7f6f0; color:#0a7b59;} .warn {background:#fff4e5; color:#9a5b00;}
.muted {background:#f1f1f1; color:#555;} .bad {background:#fdecec; color:#b42318;}
.es-src {display:inline-block; font-size:.76rem; border:1px solid #e5e5e5; border-radius:10px; padding:.28rem .6rem;
  margin:.55rem .35rem 0 0; color:#333 !important; text-decoration:none !important; background:#fff;}
.es-src.cited {border-color:#0d0d0d;}
.es-src:hover {background:#f7f7f7;}
.es-conflict {background:#fff8eb; border:1px solid #f5d9a8; border-radius:12px; padding:.55rem .8rem; font-size:.86rem;
  margin-top:.7rem; color:#6b4500;}
.es-meta {color:#9b9b9b; font-size:.72rem; margin-top:.45rem;}
[data-testid="stExpander"] details {border:none !important; box-shadow:none !important;}
[data-testid="stExpander"] summary {font-size:.8rem; color:#6b6b6b; padding-left:2.8rem;}
[data-testid="stChatInput"] {border-radius:28px !important; border:1px solid #e0e0e0 !important;
  box-shadow:0 2px 12px rgba(0,0,0,.06) !important;}
.es-card {border:1px solid #ececec; border-radius:16px; padding:1rem 1.1rem; background:#fff; margin-top:1rem;}
.es-footnote {text-align:center; color:#9b9b9b; font-size:.72rem; margin-top:.4rem;}
</style>""", unsafe_allow_html=True)

T = {"English": {
        "hero": "How can I help with your passport?",
        "sub": "Ask in Urdu, Roman Urdu or English. Every answer is taken from official DGIP documents, with its source.",
        "ask": "Ask about passports in Urdu, Roman Urdu or English",
        "abstain": "I can't verify this from the official passport documents I have. Please check with the passport office.",
        "clarify": "Could you tell me a bit more? Which passport type, how many pages (36, 72 or 100), which validity (5 or 10 years), and normal, urgent or fast track?",
        "unsupported": "I can't predict or guarantee an individual application's status, approval or delivery date. The official documents only describe standard procedures, so please use official tracking or contact the passport office.",
        "time_sensitive": "The documents contain a dated fee schedule, but I can't confirm it is still current. Please check the live official DGIP page before you pay.",
        "refused": "I only answer from official documents, with citations, and I won't make up information.",
        "action": "I can't book or confirm appointments. I can only share information from the official documents.",
        "unsafe": "I can't help with bypassing official verification. I can explain the lawful application process instead.",
        "conflict": "The official sources state different values:",
        "footer": "Answers come only from official DGIP pages. Fees and rules change: always confirm with the passport office."},
     "اردو": {
        "hero": "پاسپورٹ کے بارے میں کیا جاننا چاہتے ہیں؟",
        "sub": "اردو، رومن اردو یا انگریزی میں پوچھیں۔ ہر جواب سرکاری ڈی جی آئی پی دستاویزات سے، ماخذ کے ساتھ۔",
        "ask": "اردو، رومن اردو یا انگریزی میں سوال کریں",
        "abstain": "یہ معلومات ہمارے سرکاری دستاویزات سے تصدیق نہیں ہو سکی۔ براہ کرم پاسپورٹ دفتر سے رابطہ کریں۔",
        "clarify": "براہ کرم وضاحت کریں: پاسپورٹ کی قسم، کتنے صفحات (36، 72 یا 100)، میعاد (5 یا 10 سال)، اور نارمل، ارجنٹ یا فاسٹ ٹریک؟",
        "unsupported": "میں کسی انفرادی درخواست کی منظوری، اسٹیٹس یا ترسیل کی تاریخ کی پیشگوئی یا ضمانت نہیں دے سکتا۔ براہ کرم سرکاری ٹریکنگ یا پاسپورٹ دفتر سے رابطہ کریں۔",
        "time_sensitive": "دستاویزات میں تاریخ والی فیس کی فہرست موجود ہے، لیکن میں تصدیق نہیں کر سکتا کہ یہ آج بھی لاگو ہے۔ ادائیگی سے پہلے سرکاری ویب سائٹ دیکھیں۔",
        "refused": "میں صرف سرکاری دستاویزات سے حوالے کے ساتھ جواب دیتا ہوں اور معلومات خود سے نہیں بناتا۔",
        "action": "میں اپوائنٹمنٹ بک یا کنفرم نہیں کر سکتا؛ میں صرف سرکاری دستاویزات سے معلومات دے سکتا ہوں۔",
        "unsafe": "میں سرکاری تصدیق سے بچنے میں مدد نہیں کر سکتا۔ میں قانونی طریقۂ کار بتا سکتا ہوں۔",
        "conflict": "سرکاری ذرائع میں مختلف معلومات درج ہیں:",
        "footer": "جوابات صرف سرکاری ڈی جی آئی پی صفحات سے۔ فیس اور قواعد بدل سکتے ہیں: پاسپورٹ دفتر سے تصدیق کریں۔"}}
CHIP = {"answer": ("ok", "✓ Verified in official source"), "clarify": ("warn", "Needs clarification"),
        "abstain": ("muted", "Not in official documents"), "unsupported": ("warn", "Cannot be guaranteed"),
        "time_sensitive": ("warn", "Check for updates"), "refused": ("bad", "Declined"),
        "action": ("bad", "Declined"), "unsafe": ("bad", "Declined")}
SUGGEST = ["Passport renew karwanay ke liye kya documents chahiye?",
           "پاسپورٹ کی تجدید کے لیے کون سی دستاویزات درکار ہیں؟",
           "Can I get a 10-year passport for my 14-year-old child?",
           "Mera passport gum ho gaya, ab kya karun?"]


@st.cache_resource(show_spinner="Loading models (first time only)…")
def load():
    return Engine()


@st.cache_data
def dataset():
    return [json.loads(l) for l in open("data/esahulat_dataset.jsonl", encoding="utf-8")]


@st.cache_data(show_spinner=False)
def to_urdu(text):
    from generate import translate_ur
    return translate_ur(text)


eng = load()
ss = st.session_state
if "convs" not in ss:
    ss.convs, ss.cur = {}, None
if ss.cur is None:
    ss.cur = str(uuid.uuid4()); ss.convs[ss.cur] = {"title": "New chat", "messages": []}

# ------------------------------ sidebar ------------------------------
with st.sidebar:
    st.markdown('<div class="es-brand">🛂 E-Sahulat AI</div>', unsafe_allow_html=True)
    if st.button("✎  New chat", key="new"):
        ss.cur = str(uuid.uuid4()); ss.convs[ss.cur] = {"title": "New chat", "messages": []}
    page = st.radio("Pages", ["💬  Chat", "🧪  Benchmark", "📊  Evaluation", "📚  Sources"], label_visibility="collapsed")
    lang = st.radio("Answer language", ["English", "اردو"], horizontal=True)
    past = [(cid, c) for cid, c in ss.convs.items() if c["messages"]]
    if past:
        st.markdown('<div class="es-section">Recent</div>', unsafe_allow_html=True)
        for cid, c in reversed(past[-8:]):
            if st.button(c["title"][:34] + ("…" if len(c["title"]) > 34 else ""), key="c" + cid):
                ss.cur = cid
    with st.expander("⚙️  Settings"):
        method = st.selectbox("Retrieval", ["hybrid", "dense", "bm25"])
        style = st.selectbox("Answer style", ["extractive", "generate"],
                             format_func=lambda s: {"extractive": "Extractive (safest)", "generate": "Generate + verify (Qwen3-0.6B)"}[s])
        mode = st.selectbox("Normalization", ["full", "lexicon", "off"], help="Live ablation: full = Unicode + lexicon + phonetic skeleton")
        saved = pathlib.Path("outputs/threshold.txt")
        threshold = st.slider("Abstain below confidence", 0.0, 1.0, float(saved.read_text()) if saved.exists() else 0.80, 0.01)
    st.caption("Local open-source models: multilingual-e5-small, xlm-roberta-base-squad2, BM25. Optional: opus-mt-en-ur, Qwen3-0.6B. No AI APIs."
               + (" · LITE mode" if LITE else "") + (f" · NO_TORCH mode ({SEMANTIC_NAME})" if NO_TORCH else ""))
L = T[lang]
ESC = lambda t: html.escape(str(t))


def ask(q):
    r = eng.answer(q, method=method, threshold=threshold, mode=mode, style=style)
    if r["status"] == "answer" and lang == "اردو":
        r["urdu"] = to_urdu(r.get("generated") or r["answer"])
    return r


def render(r, lang_used):
    Lu = T[lang_used]
    parts = []
    if r["status"] == "answer":
        shown = r.get("generated") or r["answer"]
        if r.get("urdu"):
            parts.append(f'<div dir="rtl">{ESC(r["urdu"])}</div><div class="es-note">Machine translation (opus-mt-en-ur). Official English text:</div>'
                         f'<div class="es-quote">{ESC(shown)}</div>')
        else:
            parts.append(f'<div dir="auto">{ESC(shown)}</div>')
        if r.get("generated"):
            parts.append(f'<div class="es-quote"><b>Evidence:</b> {ESC(r["answer"])}</div>')
        kind, label = CHIP["answer"] if r["verified"] else ("muted", "From official source")
        chips = f'<span class="es-chip {kind}">{label}</span>'
        if r.get("generated"):
            chips += '<span class="es-chip muted">Generated by Qwen3-0.6B, checked against evidence</span>'
        elif r.get("gen_check"):
            chips += '<span class="es-chip muted">Generated answer rejected by verifier: showing official text</span>'
        parts.append(chips)
    else:
        parts.append(f'<div dir="auto">{ESC(Lu.get(r["status"], Lu["abstain"]))}</div>')
        kind, label = CHIP.get(r["status"], CHIP["abstain"])
        parts.append(f'<span class="es-chip {kind}">{label}</span>')
    if r["conflicts"]:
        rows = "".join(f'<div>• {ESC(c["category"])}: ' + ", ".join(f'<b>{d} working days</b> ({ESC(w)})' for d, w in c["values"].items())
                       + "</div>" for c in r["conflicts"])
        parts.append(f'<div class="es-conflict">⚠️ {ESC(Lu["conflict"])}{rows}<div>Please confirm with the passport office.</div></div>')
    if r["sources"] and r["status"] in ("answer", "unsupported", "time_sensitive"):
        cited = r["source"]["passage"] if r.get("source") else None
        pills = "".join(f'<a class="es-src{" cited" if s["passage"] == cited else ""}" href="{ESC(s["url"])}" target="_blank">'
                        f'[{i}] {ESC(s["section"][:42])} · {ESC(s["passage"])}</a>' for i, s in enumerate(r["sources"][:3], 1))
        parts.append(pills)
    parts.append(f'<div class="es-meta">{r["latency_ms"]:.0f} ms · confidence {r["confidence"]:.2f} · status {ESC(r["status"])}</div>')
    st.markdown(f'<div class="es-bot"><div class="es-avatar">ES</div><div class="es-body">{"".join(parts)}</div></div>',
                unsafe_allow_html=True)
    with st.expander("Sources and how this answer was found"):
        st.markdown(f"**Search query after normalization:** `{r['normalized']}`")
        if r["steps"]:
            st.markdown("\n".join(f"- {ESC(x)}" for x in r["steps"]))
        for i, s in enumerate(r["sources"], 1):
            st.markdown(f"**[{i}] {ESC(s['title'])}** · {ESC(s['section'])} · `{s['passage']}` · [open source]({s['url']}) · retrieved {s['retrieved']}")
            st.caption(s["text"][:600])


# ------------------------------ pages ------------------------------
if page.endswith("Chat"):
    conv = ss.convs[ss.cur]
    pending = None
    if not conv["messages"]:
        st.markdown(f'<div class="es-hero"><h1 dir="auto">{ESC(L["hero"])}</h1><p dir="auto">{ESC(L["sub"])}</p></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        for i, sq in enumerate(SUGGEST):
            if (c1 if i % 2 == 0 else c2).button(sq, key=f"s{i}"):
                pending = sq
    for m in conv["messages"]:
        if m["role"] == "user":
            st.markdown(f'<div class="es-user"><div dir="auto">{ESC(m["text"])}</div></div>', unsafe_allow_html=True)
        else:
            render(m["r"], m["lang"])
    q = st.chat_input(L["ask"]) or pending
    if q:
        if not conv["messages"]:
            conv["title"] = q
        conv["messages"].append({"role": "user", "text": q})
        with st.spinner("Searching official documents…"):
            r = ask(q)
        conv["messages"].append({"role": "assistant", "r": r, "lang": lang})
        st.rerun()
    st.markdown(f'<div class="es-footnote" dir="auto">{ESC(L["footer"])}</div>', unsafe_allow_html=True)

elif page.endswith("Benchmark"):
    st.markdown("### Benchmark explorer")
    st.caption("Any query from the organizers' dataset: our live answer, then their expected behaviour.")
    data = dataset()
    split = st.radio("Split", ["test", "train_dev"], horizontal=True)
    pick = st.selectbox("Query", [f"{d['qid']} · {d['language']} · {d['query']}" for d in data if d["split"] == split])
    d = next(x for x in data if x["qid"] == pick.split(" · ")[0])
    st.markdown(f'<div class="es-user"><div dir="auto">{ESC(d["query"])}</div></div>', unsafe_allow_html=True)
    render(ask(d["query"]), lang)
    exp = d.get("expected_answer_en") or d.get("expected_behavior")
    st.markdown(f'<div class="es-card"><b>Organizers\' expectation</b> · answerable: {d["answerable"]} · topic: {ESC(d["topic"])}'
                f'<div style="margin-top:.4rem">{ESC(exp)}</div>'
                + (f'<div dir="rtl" style="margin-top:.4rem">{ESC(d["expected_answer_ur"])}</div>' if d.get("expected_answer_ur") else "")
                + (f'<div class="es-note">Gold evidence: {ESC(d["gold_evidence_en"])}</div>' if d.get("gold_evidence_en") else "")
                + '</div>', unsafe_allow_html=True)
    st.markdown("### All test-split results")
    p = pathlib.Path("outputs/answers.csv")
    if p.exists():
        df = pd.read_csv(p)
        df.insert(0, "result", df.behaviour_ok.map({True: "✅ pass", False: "❌ fail"}))
        cats = st.multiselect("Filter by category", sorted(df.category.unique()))
        if cats:
            df = df[df.category.isin(cats)]
        st.caption(f"{int((df.result == '✅ pass').sum())} of {len(df)} pass (behaviour, exact)")
        st.dataframe(df[["result", "qid", "category", "lang", "expected", "got", "cited_passage_relevant", "query", "answer", "source"]],
                     hide_index=True, width="stretch")
    else:
        st.info("Run `python evaluate.py` to produce the full results table.")

elif page.endswith("Evaluation"):
    st.markdown("### Evaluation on the organizers' test split")
    p = pathlib.Path("outputs/summary.json")
    if p.exists():
        s = json.loads(p.read_text())
        ret = pd.read_csv("outputs/retrieval.csv")

        def rec(method, norm):
            v = ret[(ret.split == "test") & (ret.method == method) & (ret.normalization == norm) & (ret.lang == "all")]["Recall@5"]
            return float(v.iloc[0]) if len(v) else 0.0
        c = st.columns(4)
        c[0].metric("Recall@5 (hybrid, full)", f"{rec('hybrid', 'full'):.2f}", f"{rec('hybrid', 'full') - rec('bm25', 'off'):+.2f} vs raw BM25")
        c[1].metric("Behaviour accuracy", f"{s['behaviour_accuracy_exact']:.2f}")
        c[2].metric("Citation support", f"{s['citation_support_rate']:.2f}" if s.get("citation_support_rate") is not None else "n/a")
        c[3].metric("Median latency", f"{s['latency_ms_median']:.0f} ms")
        st.caption(f"Threshold {s['threshold_from_train_dev']} tuned on train_dev only · answer source: "
                   f"{s.get('answer_source_selection', 'best3')} · relevance rule: {s['relevance_rule']}")
        if pathlib.Path("figures/ablation_test.png").exists():
            st.image("figures/ablation_test.png")
        st.markdown("#### Behaviour by validation category")
        st.dataframe(pd.read_csv("outputs/behaviour.csv"), hide_index=True, width="stretch")
        st.markdown("#### Retrieval: keyword vs semantic vs hybrid × normalization")
        st.dataframe(ret, hide_index=True, width="stretch")
        st.markdown("#### Cross-script consistency per intent")
        st.dataframe(pd.read_csv("outputs/consistency.csv"), hide_index=True, width="stretch")
        with st.expander("All summary values"):
            st.json(s)
    else:
        st.info("Run `python evaluate.py` first.")

else:
    st.markdown("### Official sources (provenance)")
    if pathlib.Path("data/sources.csv").exists():
        st.dataframe(pd.read_csv("data/sources.csv"), hide_index=True, width="stretch")
    st.markdown("### Organizers' dataset (150 records)")
    st.dataframe(pd.DataFrame(dataset()), hide_index=True, width="stretch")
