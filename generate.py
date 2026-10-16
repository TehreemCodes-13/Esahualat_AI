"""generate.py (owner: P1) - optional answer layer: GENERATE, then VERIFY. Never the default.
1) Urdu response (FR-8): Helsinki-NLP/opus-mt-en-ur (Apache-2.0, MarianMT, ~300 MB, fast on CPU)
   translates ONLY the shown English answer; the original English evidence is always displayed too.
2) Plain-language answer: Qwen/Qwen3-0.6B (Apache-2.0) runs on CPU and may use ONLY the cited passage.
3) Verifier (our rule): accept the generated answer only if every number in it appears in the evidence
   AND at least 70% of its content words appear in the evidence or the question. Otherwise we fall
   back to the extracted evidence sentence. Every rejection is counted and reported."""
import os, re
from normalize import normalize_text, tokens

TRANSLATOR = "Helsinki-NLP/opus-mt-en-ur"
LLM = "Qwen/Qwen3-0.6B"
SUPPORT_T = 0.70
FUNCTION = set("the a an of to for and or in on is are be with its it by as at from this that if any all will can "
               "yes no not only you your they them their have has must should may also there what which".split())
_mt, _llm = {}, {}


def translate_ur(text):
    """English -> Urdu with a local MarianMT model. Returns None if the model is unavailable."""
    try:
        if not _mt:
            from transformers import MarianMTModel, MarianTokenizer
            _mt["tok"] = MarianTokenizer.from_pretrained(TRANSLATOR)
            _mt["model"] = MarianMTModel.from_pretrained(TRANSLATOR).eval()
        batch = _mt["tok"]([text[:900]], return_tensors="pt", truncation=True)
        out = _mt["model"].generate(**batch, max_new_tokens=200, num_beams=2)
        return _mt["tok"].decode(out[0], skip_special_tokens=True)
    except Exception:
        return None


def llm_answer(question, evidence):
    """1-2 sentence answer written from the evidence only. Returns None if unavailable or NOT FOUND."""
    try:
        import torch
        if not _llm:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            torch.set_num_threads(os.cpu_count() or 4)
            _llm["tok"] = AutoTokenizer.from_pretrained(LLM)
            _llm["model"] = AutoModelForCausalLM.from_pretrained(LLM, torch_dtype=torch.float32).eval()
        msgs = [{"role": "system", "content": "You answer questions about Pakistani passports using ONLY the evidence. "
                 "Write at most 2 short sentences in English. If the question is yes/no, start with Yes or No. "
                 "Copy numbers exactly as written in the evidence. If the evidence does not answer the question, "
                 "reply exactly: NOT FOUND"},
                {"role": "user", "content": f"Evidence: {evidence[:1500]}\n\nQuestion: {question}"}]
        prompt = _llm["tok"].apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = _llm["tok"](prompt, return_tensors="pt")
        with torch.no_grad():
            out = _llm["model"].generate(**ids, max_new_tokens=80, do_sample=False)
        text = _llm["tok"].decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return None if not text or "NOT FOUND" in text.upper() else text
    except Exception:
        return None


def verify(answer, evidence, question=""):
    """Return (accepted, reason). Numbers must be copied; most content words must come from the evidence."""
    if not answer:
        return False, "no answer generated"
    ev = normalize_text(evidence).lower()
    nums = re.findall(r"\d[\d,]*\d|\d", answer)
    allowed_text = (ev + " " + normalize_text(question).lower()).replace(",", "")
    bad_nums = [n for n in nums if n.replace(",", "") not in allowed_text]
    if bad_nums:
        return False, f"numbers not in evidence: {', '.join(bad_nums)}"
    allowed = set(tokens(ev)) | set(tokens(normalize_text(question)))
    words = [w for w in tokens(normalize_text(answer)) if w not in FUNCTION and len(w) > 2 and not w.isdigit()]
    if not words:
        return True, "short answer"
    support = sum(w in allowed for w in words) / len(words)
    if support < SUPPORT_T:
        return False, f"only {support:.0%} of content words supported by evidence"
    return True, f"{support:.0%} of content words supported"
