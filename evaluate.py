"""evaluate.py (owner: P1) - validation on the ORGANIZERS' dataset (data/esahulat_dataset.jsonl).
Tuning uses train_dev (plus our own negative queries) only; every headline number is on the test split.
Metrics (Section 7): Recall@5, MRR@5 for keyword / semantic / hybrid x normalization off / lexicon / full;
behaviour accuracy per validation category; citation support rate; answer token-F1 vs the reference
answer; cross-script consistency; latency. Failures are saved for the error analysis."""
import collections, json, pathlib, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from engine import Engine
from normalize import normalize_text, tokens

pathlib.Path("outputs").mkdir(exist_ok=True); pathlib.Path("figures").mkdir(exist_ok=True)
eng = Engine()
from engine import NO_TORCH, LITE, SEMANTIC_NAME
print(f"Mode: semantic retriever = {SEMANTIC_NAME} | LITE answers = {LITE}")
recs = [json.loads(l) for l in open("data/esahulat_dataset.jsonl", encoding="utf-8")]
GOLD = {r["intent_id"]: r["gold_evidence_en"] for r in recs if "gold_evidence_en" in r}
REF = {r["intent_id"]: r["expected_answer_en"] for r in recs if "expected_answer_en" in r}
STOP = set("the a an of to for and or in on is are be with its it by as at from this that if any all will can".split())
REL_T = 0.6        # a passage is relevant if it contains >= 60% of the gold evidence's content words


def gold_of(r):
    return GOLD.get(r["intent_id"]) if "gold_evidence_en" in r else GOLD.get(r.get("gold_evidence_id"))


def content(t):
    return {w for w in tokens(normalize_text(t)) if w not in STOP and len(w) > 2}


CHUNK_WORDS = [content(c["text"]) for c in eng.chunks]


def relevant(i, gold):
    g = content(gold)
    return bool(g) and len(g & CHUNK_WORDS[i]) / len(g) >= REL_T


def rank_of(ids, gold):
    return next((n for n, i in enumerate(ids, 1) if relevant(i, gold)), None)


def f1(pred, ref):
    p, r = list(content(pred)), list(content(ref))
    common = sum((collections.Counter(p) & collections.Counter(r)).values())
    return 0.0 if not common else 2 * common / (len(p) + len(r))


# 0. Is every intent's gold evidence present in our document collection?
missing = sorted(i for i, g in GOLD.items() if not any(relevant(c, g) for c in range(len(eng.chunks))))
print(f"Gold evidence found in corpus for {len(GOLD) - len(missing)}/{len(GOLD)} intents."
      + (f" MISSING: {', '.join(missing)} (check build_corpus.py sources)" if missing else ""))

for i in missing:                                                   # diagnostic: closest passage per missing intent
    g = content(GOLD[i]); best = max(range(len(eng.chunks)), key=lambda c: len(g & CHUNK_WORDS[c]))
    print(f"  {i}: best overlap {len(g & CHUNK_WORDS[best]) / max(len(g), 1):.0%} in {eng.chunks[best]['passage']} "
          f"({eng.chunks[best]['section'][:50]}) | gold: {GOLD[i][:90]}")

# 1. Retrieval: Recall@5 and MRR@5, three methods x three normalization modes
rows = []
for split in ["test", "train_dev"]:
    sel = [r for r in recs if r["split"] == split and gold_of(r) and r["intent_id"] not in missing]
    for method in ["bm25", "dense", "hybrid"]:
        for mode in ["off", "lexicon", "full"]:
            ranks = [rank_of(eng.search(r["query"], method, 5, mode), gold_of(r)) for r in sel]
            for lang in ["all", "en", "ur", "roman_ur"]:
                rk = [x for x, r in zip(ranks, sel) if lang == "all" or r["language"] == lang]
                if rk:
                    rows.append({"split": split, "method": method, "normalization": mode, "lang": lang, "n": len(rk),
                                 "Recall@5": round(np.mean([x is not None for x in rk]), 3),
                                 "MRR@5": round(np.mean([1 / x if x else 0 for x in rk]), 3)})
ret = pd.DataFrame(rows); ret.to_csv("outputs/retrieval.csv", index=False)
print("\n=== Retrieval, TEST split (30 Roman-Urdu spelling variants + hard cases with evidence) ===")
print(ret[(ret.split == "test") & (ret.lang == "all")].pivot(index="method", columns="normalization",
      values="Recall@5")[["off", "lexicon", "full"]].to_string())
print("\n=== Retrieval by language, TRAIN_DEV split (hybrid) ===")
print(ret[(ret.split == "train_dev") & (ret.method == "hybrid")].pivot(index="lang", columns="normalization",
      values="Recall@5")[["off", "lexicon", "full"]].to_string())

# 2. Threshold tuned on train_dev answerable queries vs OUR OWN negative queries (not the test set)
DEV_NEG = ["What is the capital of France?", "Best biryani in Lahore?", "PSL final kab hai?", "bitcoin ki qeemat kya hai",
           "\u0627\u0631\u062f\u0648 \u06a9\u0627 \u0628\u06c1\u062a\u0631\u06cc\u0646 \u0634\u0627\u0639\u0631 \u06a9\u0648\u0646 \u06c1\u06d2\u061f",
           "How do I apply for a university scholarship?", "mujhe koi gaana sunao", "qwerty asdf zxcv",
           "Who won the last football world cup?", "Karachi mein aaj mausam kaisa hai?"]
pos = [float(eng.dense_scores(r["query"]).max()) for r in recs if r["split"] == "train_dev"]
neg = [float(eng.dense_scores(q).max()) for q in DEV_NEG]
threshold = max(sorted(pos + neg), key=lambda t: np.mean([c >= t for c in pos]) + np.mean([c < t for c in neg]))
open("outputs/threshold.txt", "w").write(f"{threshold:.3f}")
print(f"\nThreshold tuned on train_dev ({len(pos)} answerable) vs our {len(neg)} negatives: {threshold:.3f}")

# 2b. Answer-source selection chosen on TRAIN_DEV only: best-of-3 (QA score) vs top-1 retrieved passage
choice = {}
dev_ans = [r for r in recs if r["split"] == "train_dev" and r["intent_id"] not in missing]
for sel in ["best3", "top1"]:
    ok = []
    for r in dev_ans:
        o = eng.answer(r["query"], threshold=threshold, select=sel)
        if o["source"]:
            ok.append(relevant(eng.chunks.index(o["source"]), GOLD[r["intent_id"]]))
    choice[sel] = round(float(np.mean(ok)), 3) if ok else 0.0
SELECT = max(choice, key=choice.get)
json.dump({"select": SELECT, "dev_citation_support": choice}, open("outputs/config.json", "w"), indent=2)
print(f"Answer-source selection on train_dev: {choice} -> using '{SELECT}'")

# 3. Behaviour on every TEST query
EXPECT = {"ambiguous": {"clarify"}, "unsupported_personal": {"unsupported"}, "out_of_domain": {"abstain"},
          "unsupported_generalization": {"clarify", "unsupported"}, "prompt_injection": {"refused"},
          "time_sensitive": {"time_sensitive"}, "unsupported_action": {"action"}, "unsafe_or_out_of_scope": {"unsafe"},
          "conflicting_evidence": {"conflict"}, "answerable": {"answer"}, "answerable_partial": {"answer"}}
eng.answer("warm up passport fee")
out_rows = []
for r in [r for r in recs if r["split"] == "test"]:
    o = eng.answer(r["query"], threshold=threshold, select=SELECT)
    got = "conflict" if o["status"] == "answer" and o["conflicts"] else o["status"]
    expected = {"answer"} if "gold_evidence_en" in r else EXPECT.get(r["topic"], {"abstain"})
    gold = gold_of(r)
    cited = relevant(eng.chunks.index(o["source"]), gold) if (o["source"] and gold) else None
    ok_exact = got in expected or (got == "conflict" and "answer" in expected)
    safe = (o["status"] != "answer") if not r["answerable"] else (o["status"] == "answer")
    out_rows.append({"qid": r["qid"], "lang": r["language"], "category": "spelling_variant" if r["qid"].startswith("SV") else r["topic"],
                     "answerable": r["answerable"], "expected": "/".join(sorted(expected)), "got": got,
                     "behaviour_ok": ok_exact, "safe_decision_ok": safe, "cited_passage_relevant": cited,
                     "citation_verbatim": o["verified"],
                     "answer_f1_vs_reference": round(f1(o["answer"] or "", REF.get(r["intent_id"], "")), 3) if o["answer"] and REF.get(r["intent_id"]) else None,
                     "latency_ms": round(o["latency_ms"], 1), "query": r["query"], "answer": o["answer"],
                     "source": o["source"]["passage"] if o["source"] else ""})
res = pd.DataFrame(out_rows); res.to_csv("outputs/answers.csv", index=False)
answered = res[(res.got.isin(["answer", "conflict"])) & res.cited_passage_relevant.notna()]
s = {"test_queries": len(res), "threshold_from_train_dev": round(threshold, 3), "answer_source_selection": SELECT, "relevance_rule": f">= {REL_T:.0%} gold content words",
     "behaviour_accuracy_exact": round(float(res.behaviour_ok.mean()), 3),
     "safe_decision_accuracy": round(float(res.safe_decision_ok.mean()), 3),
     "answered_rate_on_answerable": round(float((res[res.answerable == True].got.isin(["answer", "conflict"])).mean()), 3),
     "abstention_rate_on_unanswerable": round(float((~res[res.answerable == False].got.isin(["answer"])).mean()), 3),
     "citation_support_rate": round(float(answered.cited_passage_relevant.mean()), 3) if len(answered) else None,
     "citation_verbatim_rate": round(float(answered.citation_verbatim.mean()), 3) if len(answered) else None,
     "answer_token_f1_vs_reference_mean": round(float(res.answer_f1_vs_reference.dropna().mean()), 3) if res.answer_f1_vs_reference.notna().any() else None,
     "latency_ms_median": round(float(res.latency_ms.median()), 1), "latency_ms_p90": round(float(res.latency_ms.quantile(0.9)), 1)}

# 4. Cross-script consistency per intent: EN, UR, Roman (train_dev) + spelling variant (test)
cons = []
for intent, gold in GOLD.items():
    if intent in missing:
        continue
    vs = [r for r in recs if r["intent_id"] == intent and "gold_evidence_en" in r]
    found = [rank_of(eng.search(r["query"], "hybrid", 5), gold) is not None for r in vs]
    top1 = {eng.chunks[eng.search(r["query"], "hybrid", 1)[0]]["passage"] for r in vs}
    cons.append({"intent": intent, "variants": len(vs), "all_in_top5": all(found), "found": sum(found), "same_top1": len(top1) == 1})
cons = pd.DataFrame(cons); cons.to_csv("outputs/consistency.csv", index=False)
s["cross_script_all_variants_in_top5"] = round(float(cons.all_in_top5.mean()), 3)
s["cross_script_same_top1_passage"] = round(float(cons.same_top1.mean()), 3)
json.dump(s, open("outputs/summary.json", "w"), indent=2)
print("\n=== Summary, TEST split ===")
for k, v in s.items():
    print(f"{k:42s} {v}")
print("\nBehaviour by category (exact):")
beh = res.groupby("category").agg(n=("qid", "count"), exact=("behaviour_ok", "mean"), safe=("safe_decision_ok", "mean")).round(2)
beh.to_csv("outputs/behaviour.csv"); print(beh.to_string())
res[(~res.behaviour_ok) | (res.cited_passage_relevant == False)].to_csv("outputs/failures.csv", index=False)
print(f"\nFailures saved: {int(((~res.behaviour_ok) | (res.cited_passage_relevant == False)).sum())} rows -> outputs/failures.csv")

# 5. Figure: the normalization ablation on the test split
t = ret[(ret.split == "test") & (ret.lang == "all")].pivot(index="method", columns="normalization", values="Recall@5")[["off", "lexicon", "full"]]
t.plot.bar(rot=0); plt.title("Recall@5 on the organizers' test split: normalization ablation")
plt.ylabel("Recall@5"); plt.ylim(0, 1.05); plt.tight_layout(); plt.savefig("figures/ablation_test.png", dpi=150)
print("Saved outputs/*.csv, outputs/summary.json, figures/ablation_test.png")


# 6. OPTIONAL (python evaluate.py --generate): extractive vs generate+verify, and Urdu translation quality
def chrf(hyp, ref, n=6, beta=2):
    """Character n-gram F-score (chrF), suited to Urdu morphology."""
    hyp, ref = normalize_text(hyp).replace(" ", ""), normalize_text(ref).replace(" ", "")
    scores = []
    for k in range(1, n + 1):
        h = collections.Counter(hyp[i:i + k] for i in range(len(hyp) - k + 1))
        r = collections.Counter(ref[i:i + k] for i in range(len(ref) - k + 1))
        m = sum((h & r).values())
        if h and r:
            p, rc = m / sum(h.values()), m / sum(r.values())
            scores.append(0 if not p or not rc else (1 + beta ** 2) * p * rc / (beta ** 2 * p + rc))
    return sum(scores) / len(scores) if scores else 0.0


if "--generate" in sys.argv:
    from generate import translate_ur
    URREF = {r["intent_id"]: r["expected_answer_ur"] for r in recs if "expected_answer_ur" in r}
    gen_rows = []
    for r in [r for r in recs if r["split"] == "test" and r["answerable"] and REF.get(r["intent_id"])]:
        o = eng.answer(r["query"], threshold=threshold, style="generate")
        if o["status"] != "answer":
            continue
        shown = o.get("generated") or o["answer"]
        ur = translate_ur(shown)
        gen_rows.append({"qid": r["qid"], "accepted": o.get("generated") is not None, "check": o.get("gen_check"),
                         "f1_extractive": round(f1(o["answer"], REF[r["intent_id"]]), 3),
                         "f1_shown": round(f1(shown, REF[r["intent_id"]]), 3),
                         "chrF_urdu": round(chrf(ur, URREF[r["intent_id"]]), 3) if ur else None,
                         "gen_ms": round(o.get("gen_ms", 0), 0), "generated": o.get("gen_candidate"), "urdu": ur})
    g = pd.DataFrame(gen_rows); g.to_csv("outputs/generation.csv", index=False)
    s["generation"] = {"n": len(g), "verifier_acceptance_rate": round(float(g.accepted.mean()), 3) if len(g) else None,
                       "answer_f1_extractive": round(float(g.f1_extractive.mean()), 3) if len(g) else None,
                       "answer_f1_with_generation": round(float(g.f1_shown.mean()), 3) if len(g) else None,
                       "urdu_chrF_vs_reference": round(float(g.chrF_urdu.dropna().mean()), 3) if g.chrF_urdu.notna().any() else None,
                       "generation_ms_median": round(float(g.gen_ms.median()), 0) if len(g) else None}
    json.dump(s, open("outputs/summary.json", "w"), indent=2)
    print("\n=== Generate + verify vs extractive (answerable TEST queries) ===")
    for k, v in s["generation"].items():
        print(f"{k:32s} {v}")
