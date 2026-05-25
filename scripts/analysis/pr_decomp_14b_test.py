"""14B RSFT s42 test-set precision/recall decomposition by relation type."""

import json
from collections import defaultdict
from pathlib import Path

GOLD_PATH = "/workspace/data/re-docred-repo/data/test_revised.json"
PRED_PATH = "/workspace/eval_results/test_eval_14b_rsft_s42/codalab_submission.json"
PRED_RAW_PATH = "/workspace/eval_results/test_eval_14b_rsft_s42/predictions.json"

DOCRED_REL_INFO = {
    "P6": "head of government", "P17": "country", "P19": "place of birth",
    "P20": "place of death", "P22": "father", "P25": "mother",
    "P26": "spouse", "P27": "country of citizenship", "P30": "continent",
    "P31": "instance of", "P35": "head of state", "P36": "capital",
    "P37": "official language", "P39": "position held", "P40": "child",
    "P50": "author", "P54": "member of sports team", "P57": "director",
    "P58": "screenwriter", "P69": "educated at", "P86": "composer",
    "P102": "member of political party", "P108": "employer",
    "P112": "founded by", "P118": "league", "P123": "publisher",
    "P127": "owned by",
    "P131": "located in the administrative territorial entity",
    "P136": "genre", "P137": "operator", "P140": "religion",
    "P150": "contains administrative territorial entity",
    "P155": "follows", "P156": "followed by",
    "P159": "headquarters location", "P161": "cast member",
    "P162": "producer", "P166": "award received", "P170": "creator",
    "P171": "parent taxon", "P172": "ethnic group", "P175": "performer",
    "P176": "manufacturer", "P178": "developer", "P179": "series",
    "P190": "sister city", "P194": "legislative body",
    "P205": "basin country",
    "P206": "located in or next to body of water",
    "P241": "military branch", "P264": "record label",
    "P272": "production company", "P276": "location",
    "P279": "subclass of", "P355": "subsidiary", "P361": "part of",
    "P364": "original language of work", "P400": "platform",
    "P403": "mouth of the watercourse", "P449": "original network",
    "P463": "member of", "P488": "chairperson",
    "P495": "country of origin", "P527": "has part",
    "P551": "residence", "P569": "date of birth",
    "P570": "date of death", "P571": "inception",
    "P576": "dissolved, abolished or demolished",
    "P577": "publication date", "P580": "start time",
    "P582": "end time", "P585": "point in time", "P607": "conflict",
    "P674": "characters", "P676": "lyrics by",
    "P706": "located on terrain feature", "P710": "participant",
    "P737": "influenced by", "P740": "location of formation",
    "P749": "parent organization", "P800": "notable work",
    "P807": "separated from", "P840": "narrative location",
    "P937": "work location", "P1001": "applies to jurisdiction",
    "P1056": "product or material produced",
    "P1198": "unemployment rate", "P1336": "territory claimed by",
    "P1344": "participant of", "P1365": "replaces",
    "P1366": "replaced by", "P1376": "capital of",
    "P1412": "languages spoken, written or signed",
    "P1441": "present in work", "P3373": "sibling",
}

def rel_name(code):
    return DOCRED_REL_INFO.get(code, code)

# ── Load gold ──
with open(GOLD_PATH) as f:
    gold_data = json.load(f)

gold_by_doc = {}
gold_triples = set()
gold_per_rel = defaultdict(int)
gold_evidence_info = {}  # (title, h, t, r) -> evidence list
gold_cross_sentence = set()
gold_within_sentence = set()

for doc in gold_data:
    title = doc["title"]
    n_sents = len(doc["sents"])
    for lab in doc.get("labels", []):
        key = (title, lab["h"], lab["t"], lab["r"])
        gold_triples.add(key)
        gold_per_rel[lab["r"]] += 1
        evi = lab.get("evidence", [])
        gold_evidence_info[key] = evi
        if len(evi) >= 2:
            gold_cross_sentence.add(key)
        elif len(evi) == 1:
            gold_within_sentence.add(key)
        # evi==[] means no evidence annotation (distant supervision leftovers)

# ── Load predictions (codalab format: dict of title -> list) ──
with open(PRED_PATH) as f:
    pred_data = json.load(f)

pred_triples = set()
pred_per_rel = defaultdict(int)
pred_evidence = {}  # key -> evidence list

if isinstance(pred_data, dict):
    for title, triples in pred_data.items():
        for t in triples:
            h = t.get("h_idx", t.get("h"))
            ti = t.get("t_idx", t.get("t"))
            r = t["r"]
            key = (title, h, ti, r)
            pred_triples.add(key)
            pred_per_rel[r] += 1
            pred_evidence[key] = t.get("evidence", [])
elif isinstance(pred_data, list):
    for t in pred_data:
        title = t["title"]
        h = t.get("h_idx", t.get("h"))
        ti = t.get("t_idx", t.get("t"))
        r = t["r"]
        key = (title, h, ti, r)
        pred_triples.add(key)
        pred_per_rel[r] += 1
        pred_evidence[key] = t.get("evidence", [])

# ── Overall P/R/F1 ──
tp_set = pred_triples & gold_triples
tp = len(tp_set)
prec = tp / len(pred_triples) if pred_triples else 0
rec = tp / len(gold_triples) if gold_triples else 0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

print("=" * 72)
print("  14B RSFT s42 — Test-Set Precision/Recall Decomposition")
print("=" * 72)
print(f"\n  Overall: P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
print(f"  TP={tp}  Pred={len(pred_triples)}  Gold={len(gold_triples)}")

# ── Per-relation P/R/F1 ──
all_rels = sorted(set(list(gold_per_rel.keys()) + list(pred_per_rel.keys())))

rel_metrics = {}
for r in all_rels:
    r_gold = {k for k in gold_triples if k[3] == r}
    r_pred = {k for k in pred_triples if k[3] == r}
    r_tp = r_gold & r_pred
    r_p = len(r_tp) / len(r_pred) if r_pred else 0
    r_r = len(r_tp) / len(r_gold) if r_gold else 0
    r_f1 = 2 * r_p * r_r / (r_p + r_r) if (r_p + r_r) > 0 else 0
    rel_metrics[r] = {
        "p": r_p, "r": r_r, "f1": r_f1,
        "tp": len(r_tp), "pred": len(r_pred), "gold": len(r_gold),
    }

# Top-10 by F1
sorted_by_f1 = sorted(rel_metrics.items(), key=lambda x: -x[1]["f1"])
print(f"\n{'─' * 72}")
print(f"  Top-15 relations by F1 (gold >= 10)")
print(f"{'─' * 72}")
print(f"  {'Relation':<48} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'Pred':>5} {'Gold':>5}")
for r, m in sorted_by_f1:
    if m["gold"] >= 10:
        print(f"  {rel_name(r):<48} {m['p']:.3f} {m['r']:.3f} {m['f1']:.3f} {m['tp']:>5} {m['pred']:>5} {m['gold']:>5}")

# Top-5 high-precision (pred >= 10)
print(f"\n{'─' * 72}")
print(f"  Top-10 HIGH-PRECISION relations (pred >= 10)")
print(f"{'─' * 72}")
sorted_by_p = sorted(rel_metrics.items(), key=lambda x: -x[1]["p"])
print(f"  {'Relation':<48} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'Pred':>5} {'Gold':>5}")
cnt = 0
for r, m in sorted_by_p:
    if m["pred"] >= 10:
        print(f"  {rel_name(r):<48} {m['p']:.3f} {m['r']:.3f} {m['f1']:.3f} {m['tp']:>5} {m['pred']:>5} {m['gold']:>5}")
        cnt += 1
        if cnt >= 10:
            break

# Top-5 high-recall (gold >= 10)
print(f"\n{'─' * 72}")
print(f"  Top-10 HIGH-RECALL relations (gold >= 10)")
print(f"{'─' * 72}")
sorted_by_r = sorted(rel_metrics.items(), key=lambda x: -x[1]["r"])
print(f"  {'Relation':<48} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'Pred':>5} {'Gold':>5}")
cnt = 0
for r, m in sorted_by_r:
    if m["gold"] >= 10:
        print(f"  {rel_name(r):<48} {m['p']:.3f} {m['r']:.3f} {m['f1']:.3f} {m['tp']:>5} {m['pred']:>5} {m['gold']:>5}")
        cnt += 1
        if cnt >= 10:
            break

# Bottom-10 by F1 (gold >= 20)
print(f"\n{'─' * 72}")
print(f"  Bottom-10 relations by F1 (gold >= 20)")
print(f"{'─' * 72}")
sorted_by_f1_asc = sorted(rel_metrics.items(), key=lambda x: x[1]["f1"])
print(f"  {'Relation':<48} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'Pred':>5} {'Gold':>5}")
cnt = 0
for r, m in sorted_by_f1_asc:
    if m["gold"] >= 20:
        print(f"  {rel_name(r):<48} {m['p']:.3f} {m['r']:.3f} {m['f1']:.3f} {m['tp']:>5} {m['pred']:>5} {m['gold']:>5}")
        cnt += 1
        if cnt >= 10:
            break

# Conservative (P >> R) vs aggressive (R >> P)
print(f"\n{'─' * 72}")
print(f"  Most CONSERVATIVE (P >> R, gold >= 20)")
print(f"{'─' * 72}")
pr_gap = [(r, m, m["p"] - m["r"]) for r, m in rel_metrics.items() if m["gold"] >= 20 and m["pred"] >= 5]
pr_gap.sort(key=lambda x: -x[2])
print(f"  {'Relation':<48} {'P':>6} {'R':>6} {'Gap':>6} {'TP':>5} {'Pred':>5} {'Gold':>5}")
for r, m, gap in pr_gap[:8]:
    print(f"  {rel_name(r):<48} {m['p']:.3f} {m['r']:.3f} {gap:>+.3f} {m['tp']:>5} {m['pred']:>5} {m['gold']:>5}")

print(f"\n{'─' * 72}")
print(f"  Most AGGRESSIVE (R >> P, gold >= 20)")
print(f"{'─' * 72}")
pr_gap.sort(key=lambda x: x[2])
print(f"  {'Relation':<48} {'P':>6} {'R':>6} {'Gap':>6} {'TP':>5} {'Pred':>5} {'Gold':>5}")
for r, m, gap in pr_gap[:8]:
    print(f"  {rel_name(r):<48} {m['p']:.3f} {m['r']:.3f} {gap:>+.3f} {m['tp']:>5} {m['pred']:>5} {m['gold']:>5}")

# ── Cross-sentence vs within-sentence ──
print(f"\n{'=' * 72}")
print(f"  Cross-Sentence vs Within-Sentence Analysis")
print(f"{'=' * 72}")

# Gold with evidence annotations (evi != [])
gold_with_evi = {k for k, v in gold_evidence_info.items() if len(v) > 0}
gold_no_evi = {k for k, v in gold_evidence_info.items() if len(v) == 0}

cross_tp = len(tp_set & gold_cross_sentence)
cross_gold = len(gold_cross_sentence)
cross_pred_matching = len(pred_triples & gold_cross_sentence)  # pred that match cross-sentence gold
cross_rec = cross_tp / cross_gold if cross_gold else 0

within_tp = len(tp_set & gold_within_sentence)
within_gold = len(gold_within_sentence)
within_rec = within_tp / within_gold if within_gold else 0

noevi_tp = len(tp_set & gold_no_evi)
noevi_gold = len(gold_no_evi)
noevi_rec = noevi_tp / noevi_gold if noevi_gold else 0

print(f"\n  {'Category':<35} {'TP':>6} {'Gold':>6} {'Recall':>8}")
print(f"  {'─' * 35} {'─' * 6} {'─' * 6} {'─' * 8}")
print(f"  {'Cross-sentence (evi >= 2 sents)':<35} {cross_tp:>6} {cross_gold:>6} {cross_rec:>8.4f}")
print(f"  {'Within-sentence (evi = 1 sent)':<35} {within_tp:>6} {within_gold:>6} {within_rec:>8.4f}")
print(f"  {'No evidence annotation (evi=[])':<35} {noevi_tp:>6} {noevi_gold:>6} {noevi_rec:>8.4f}")

# ── EDCR analysis on predictions ──
print(f"\n{'=' * 72}")
print(f"  EDCR-Stratified Precision Analysis")
print(f"{'=' * 72}")

# For each correct prediction, compute per-triple EDCR
# EDCR = distractor citations / total citations
# We need the gold evidence for each TP triple

tp_edcr_data = []
for key in pred_triples:
    pred_evi = set(pred_evidence.get(key, []))
    if not pred_evi:
        continue
    gold_evi = set(gold_evidence_info.get(key, []))
    if not gold_evi:
        # no gold evidence info - compute EDCR as fraction non-matching
        # skip these for EDCR analysis
        continue
    distractor = len(pred_evi - gold_evi)
    total = len(pred_evi)
    edcr = distractor / total if total > 0 else 0
    is_correct = key in tp_set
    tp_edcr_data.append({"key": key, "edcr": edcr, "correct": is_correct,
                          "pred_evi": pred_evi, "gold_evi": gold_evi})

# Also compute for all predictions (not just those with gold evi)
all_pred_edcr = []
for key in pred_triples:
    pred_evi = set(pred_evidence.get(key, []))
    if not pred_evi:
        continue
    gold_evi = set(gold_evidence_info.get(key, []))
    if gold_evi:
        distractor = len(pred_evi - gold_evi)
        total = len(pred_evi)
        edcr = distractor / total if total > 0 else 0
    else:
        edcr = None  # can't compute
    is_correct = key in tp_set
    all_pred_edcr.append({"edcr": edcr, "correct": is_correct, "n_evi": len(pred_evi)})

# Stratify by evidence count
evi_bins = [(1, 1, "1 sent"), (2, 2, "2 sents"), (3, 3, "3 sents"), (4, 99, "4+ sents")]
print(f"\n  Precision by predicted evidence count:")
print(f"  {'Evi sents':<12} {'Correct':>8} {'Total':>8} {'Precision':>10}")
for lo, hi, label in evi_bins:
    subset = [x for x in all_pred_edcr if lo <= x["n_evi"] <= hi]
    if subset:
        correct = sum(1 for x in subset if x["correct"])
        print(f"  {label:<12} {correct:>8} {len(subset):>8} {correct/len(subset):>10.4f}")

# Stratify by EDCR
if tp_edcr_data:
    edcr_bins = [(0, 0.001, "EDCR=0 (perfect)"), (0.001, 0.3, "EDCR<0.3 (low)"),
                 (0.3, 0.7, "0.3<=EDCR<0.7 (mid)"), (0.7, 1.01, "EDCR>=0.7 (high)")]
    print(f"\n  Precision by EDCR of predicted evidence (only preds with gold evi):")
    print(f"  {'EDCR bin':<25} {'Correct':>8} {'Total':>8} {'Precision':>10}")
    for lo, hi, label in edcr_bins:
        subset = [x for x in tp_edcr_data if lo <= x["edcr"] < hi]
        # Also count wrong predictions in same EDCR range
        all_in_bin = [x for x in tp_edcr_data if lo <= x["edcr"] < hi]
        if all_in_bin:
            correct = sum(1 for x in all_in_bin if x["correct"])
            print(f"  {label:<25} {correct:>8} {len(all_in_bin):>8} {correct/len(all_in_bin):>10.4f}")

# ── Relation frequency vs performance ──
print(f"\n{'=' * 72}")
print(f"  Full Per-Relation Table (gold >= 5)")
print(f"{'=' * 72}")
print(f"  {'Code':<6} {'Relation':<44} {'P':>6} {'R':>6} {'F1':>6} {'TP':>4} {'Pred':>5} {'Gold':>5}")
print(f"  {'─' * 6} {'─' * 44} {'─' * 6} {'─' * 6} {'─' * 6} {'─' * 4} {'─' * 5} {'─' * 5}")
for r, m in sorted(rel_metrics.items(), key=lambda x: -x[1]["gold"]):
    if m["gold"] >= 5:
        name = rel_name(r)
        if len(name) > 43:
            name = name[:40] + "..."
        print(f"  {r:<6} {name:<44} {m['p']:.3f} {m['r']:.3f} {m['f1']:.3f} {m['tp']:>4} {m['pred']:>5} {m['gold']:>5}")

# ── Summary stats ──
print(f"\n{'=' * 72}")
print(f"  Summary Statistics")
print(f"{'=' * 72}")
n_rels_with_gold = sum(1 for r, m in rel_metrics.items() if m["gold"] > 0)
n_rels_with_pred = sum(1 for r, m in rel_metrics.items() if m["pred"] > 0)
n_rels_with_tp = sum(1 for r, m in rel_metrics.items() if m["tp"] > 0)
zero_recall_rels = [(r, m) for r, m in rel_metrics.items() if m["gold"] >= 10 and m["tp"] == 0]
halluc_rels = [(r, m) for r, m in rel_metrics.items() if m["pred"] > 0 and m["gold"] == 0]

print(f"\n  Relations in gold: {n_rels_with_gold}")
print(f"  Relations predicted: {n_rels_with_pred}")
print(f"  Relations with TP > 0: {n_rels_with_tp}")
print(f"  Zero-recall relations (gold >= 10): {len(zero_recall_rels)}")
for r, m in zero_recall_rels:
    print(f"    {r} {rel_name(r)} (gold={m['gold']}, pred={m['pred']})")
print(f"  Hallucinated relations (pred > 0, gold = 0): {len(halluc_rels)}")
for r, m in halluc_rels:
    print(f"    {r} {rel_name(r)} (pred={m['pred']})")

# ── Macro-average F1 ──
rels_for_macro = [m for r, m in rel_metrics.items() if m["gold"] >= 5]
if rels_for_macro:
    macro_p = sum(m["p"] for m in rels_for_macro) / len(rels_for_macro)
    macro_r = sum(m["r"] for m in rels_for_macro) / len(rels_for_macro)
    macro_f1 = sum(m["f1"] for m in rels_for_macro) / len(rels_for_macro)
    print(f"\n  Macro-avg (rels with gold >= 5, N={len(rels_for_macro)}):")
    print(f"    P={macro_p:.4f}  R={macro_r:.4f}  F1={macro_f1:.4f}")

print()
