"""plan_021: per-relation / per-length / per-entity-count error analysis for RSFT vs SFT."""

import json, csv, os, sys
from collections import defaultdict
from pathlib import Path

EVAL_DIR = Path("/workspace/eval_results")
GOLD_PATH = Path("/workspace/data/docred/dev.json")
TRAIN_PATH = Path("/workspace/data/docred/train_annotated.json")
OUT_DIR = Path("/workspace/freige/scripts/analysis/results")

PRED_FILES = {
    "4B_SFT": EVAL_DIR / "d111_4b_sft_s42_reeval" / "predictions.json",
    "4B_RSFT": EVAL_DIR / "rsft_s43_dev_eval_preds" / "predictions.json",
    "8B_SFT": EVAL_DIR / "qwen3_8b_sft_eval" / "predictions.json",
    "8B_RSFT": EVAL_DIR / "d102_8b_k1_s42" / "predictions.json",
}

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


def _key(head, tail, relation):
    return (head.lower().strip(), tail.lower().strip(), relation.lower().strip())


def _doc_key(doc_id, head, tail, relation):
    return (doc_id, *_key(head, tail, relation))


def build_train_facts():
    with open(TRAIN_PATH) as f:
        train_data = json.load(f)
    facts = set()
    for doc in train_data:
        vs = doc["vertexSet"]
        for lab in doc.get("labels", []):
            h = vs[lab["h"]][0]["name"]
            t = vs[lab["t"]][0]["name"]
            r = DOCRED_REL_INFO.get(lab["r"], lab["r"])
            facts.add(_key(h, t, r))
    return facts


def load_gold_doc_info():
    with open(GOLD_PATH) as f:
        data = json.load(f)
    info = {}
    for doc in data:
        title = doc["title"]
        n_words = sum(len(s) for s in doc["sents"])
        n_entities = len(doc["vertexSet"])
        n_sents = len(doc["sents"])
        info[title] = {"n_words": n_words, "n_entities": n_entities, "n_sents": n_sents}
    return info


def load_predictions(path):
    with open(path) as f:
        data = json.load(f)
    docs = {}
    for d in data:
        doc_id = d["doc_id"]
        preds = [(t["head"], t["tail"], t["relation"]) for t in d.get("parsed_triples", [])]
        golds = [(t["head"], t["tail"], t["relation"]) for t in d.get("gold_triples", [])]
        docs[doc_id] = {"preds": preds, "golds": golds}
    return docs


def compute_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


# ──────────────────────────────────────────
# Dim 1: Per-relation F1
# ──────────────────────────────────────────
def per_relation_analysis(all_preds, train_facts):
    results = {}
    for model_name, docs in all_preds.items():
        rel_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        for doc_id, d in docs.items():
            pred_set = {_doc_key(doc_id, h, t, r) for h, t, r in d["preds"]}
            gold_set = {_doc_key(doc_id, h, t, r) for h, t, r in d["golds"]}
            tp_set = pred_set & gold_set
            fp_set = pred_set - gold_set
            fn_set = gold_set - pred_set
            for k in tp_set:
                rel_stats[k[3]]["tp"] += 1
            for k in fp_set:
                rel_stats[k[3]]["fp"] += 1
            for k in fn_set:
                rel_stats[k[3]]["fn"] += 1
        results[model_name] = dict(rel_stats)
    return results


def write_per_relation_csv(rel_results, out_path):
    all_rels = set()
    for model_stats in rel_results.values():
        all_rels.update(model_stats.keys())
    all_rels = sorted(all_rels)

    models = list(rel_results.keys())
    rows = []
    for rel in all_rels:
        row = {"relation": rel}
        for m in models:
            s = rel_results[m].get(rel, {"tp": 0, "fp": 0, "fn": 0})
            _, _, f1 = compute_f1(s["tp"], s["fp"], s["fn"])
            row[f"{m}_f1"] = round(f1, 4)
            row[f"{m}_tp"] = s["tp"]
            row[f"{m}_gold"] = s["tp"] + s["fn"]
        rows.append(row)

    for row in rows:
        row["4B_gain"] = round(row.get("4B_RSFT_f1", 0) - row.get("4B_SFT_f1", 0), 4)
        row["8B_gain"] = round(row.get("8B_RSFT_f1", 0) - row.get("8B_SFT_f1", 0), 4)

    rows.sort(key=lambda x: x.get("4B_gain", 0), reverse=True)

    fieldnames = ["relation",
                  "4B_SFT_f1", "4B_RSFT_f1", "4B_gain", "4B_SFT_tp", "4B_SFT_gold",
                  "8B_SFT_f1", "8B_RSFT_f1", "8B_gain", "8B_SFT_tp", "8B_SFT_gold"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return rows


# ──────────────────────────────────────────
# Dim 2 & 3: Per-bucket analysis
# ──────────────────────────────────────────
def bucket_analysis(all_preds, doc_info, key_fn, bucket_name, train_facts):
    values = []
    for doc_id in doc_info:
        values.append(key_fn(doc_info[doc_id]))
    values.sort()
    n = len(values)
    q33 = values[n // 3]
    q67 = values[2 * n // 3]

    def get_bucket(v):
        if v <= q33:
            return f"low(<={q33})"
        elif v <= q67:
            return f"mid({q33+1}-{q67})"
        else:
            return f"high(>{q67})"

    results = {}
    for model_name, docs in all_preds.items():
        bucket_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "n_docs": 0})
        seen_docs = defaultdict(set)
        for doc_id, d in docs.items():
            if doc_id not in doc_info:
                continue
            v = key_fn(doc_info[doc_id])
            b = get_bucket(v)
            pred_set = {_doc_key(doc_id, h, t, r) for h, t, r in d["preds"]}
            gold_set = {_doc_key(doc_id, h, t, r) for h, t, r in d["golds"]}
            # ign: filter out train facts
            pred_ign = {k for k in pred_set if k[1:] not in train_facts}
            gold_ign = {k for k in gold_set if k[1:] not in train_facts}
            tp = len(pred_ign & gold_ign)
            fp = len(pred_ign - gold_ign)
            fn = len(gold_ign - pred_ign)
            bucket_stats[b]["tp"] += tp
            bucket_stats[b]["fp"] += fp
            bucket_stats[b]["fn"] += fn
            if doc_id not in seen_docs[b]:
                bucket_stats[b]["n_docs"] += 1
                seen_docs[b].add(doc_id)
        results[model_name] = dict(bucket_stats)

    buckets_ordered = sorted(set(b for m in results.values() for b in m.keys()))
    return results, buckets_ordered, {"q33": q33, "q67": q67}


def write_bucket_csv(bucket_results, buckets_ordered, out_path, dim_name):
    models = list(bucket_results.keys())
    rows = []
    for b in buckets_ordered:
        row = {dim_name: b}
        for m in models:
            s = bucket_results[m].get(b, {"tp": 0, "fp": 0, "fn": 0, "n_docs": 0})
            _, _, f1 = compute_f1(s["tp"], s["fp"], s["fn"])
            row[f"{m}_ign_f1"] = round(f1, 4)
            row[f"{m}_n_docs"] = s["n_docs"]
        row["4B_gain"] = round(row.get("4B_RSFT_ign_f1", 0) - row.get("4B_SFT_ign_f1", 0), 4)
        row["8B_gain"] = round(row.get("8B_RSFT_ign_f1", 0) - row.get("8B_SFT_ign_f1", 0), 4)
        rows.append(row)

    fieldnames = [dim_name,
                  "4B_SFT_ign_f1", "4B_RSFT_ign_f1", "4B_gain", "4B_SFT_n_docs",
                  "8B_SFT_ign_f1", "8B_RSFT_ign_f1", "8B_gain", "8B_SFT_n_docs"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading train facts for ign_f1...")
    train_facts = build_train_facts()
    print(f"  {len(train_facts)} unique train facts")

    print("Loading gold doc info...")
    doc_info = load_gold_doc_info()
    print(f"  {len(doc_info)} dev docs")

    print("Loading predictions...")
    all_preds = {}
    for name, path in PRED_FILES.items():
        if not path.exists():
            print(f"  WARNING: {name} not found at {path}")
            continue
        all_preds[name] = load_predictions(path)
        print(f"  {name}: {len(all_preds[name])} docs")

    if not all_preds:
        print("ERROR: No prediction files found")
        sys.exit(1)

    # ── Dim 1: Per-relation ──
    print("\n=== Per-Relation F1 ===")
    rel_results = per_relation_analysis(all_preds, train_facts)
    rel_rows = write_per_relation_csv(rel_results, OUT_DIR / "per_relation_f1.csv")
    print(f"Top-5 RSFT gain (4B):")
    for r in rel_rows[:5]:
        print(f"  {r['relation']:40s}  SFT={r['4B_SFT_f1']:.3f}  RSFT={r['4B_RSFT_f1']:.3f}  gain={r['4B_gain']:+.3f}  (gold={r['4B_SFT_gold']})")
    print(f"Bottom-5 RSFT gain (4B):")
    for r in rel_rows[-5:]:
        print(f"  {r['relation']:40s}  SFT={r['4B_SFT_f1']:.3f}  RSFT={r['4B_RSFT_f1']:.3f}  gain={r['4B_gain']:+.3f}  (gold={r['4B_SFT_gold']})")

    # ── Dim 2: Per-length ──
    print("\n=== Per-Document-Length Ign-F1 ===")
    len_results, len_buckets, len_q = bucket_analysis(
        all_preds, doc_info, lambda d: d["n_words"], "doc_length", train_facts)
    len_rows = write_bucket_csv(len_results, len_buckets, OUT_DIR / "per_length_f1.csv", "doc_length")
    print(f"Quantiles: q33={len_q['q33']}, q67={len_q['q67']}")
    for r in len_rows:
        print(f"  {r['doc_length']:20s}  4B: SFT={r['4B_SFT_ign_f1']:.3f} RSFT={r['4B_RSFT_ign_f1']:.3f} gain={r['4B_gain']:+.3f}  |  8B: SFT={r['8B_SFT_ign_f1']:.3f} RSFT={r['8B_RSFT_ign_f1']:.3f} gain={r['8B_gain']:+.3f}")

    # ── Dim 3: Per-entity-count ──
    print("\n=== Per-Entity-Count Ign-F1 ===")
    ent_results, ent_buckets, ent_q = bucket_analysis(
        all_preds, doc_info, lambda d: d["n_entities"], "entity_count", train_facts)
    ent_rows = write_bucket_csv(ent_results, ent_buckets, OUT_DIR / "per_entity_count_f1.csv", "entity_count")
    print(f"Quantiles: q33={ent_q['q33']}, q67={ent_q['q67']}")
    for r in ent_rows:
        print(f"  {r['entity_count']:20s}  4B: SFT={r['4B_SFT_ign_f1']:.3f} RSFT={r['4B_RSFT_ign_f1']:.3f} gain={r['4B_gain']:+.3f}  |  8B: SFT={r['8B_SFT_ign_f1']:.3f} RSFT={r['8B_RSFT_ign_f1']:.3f} gain={r['8B_gain']:+.3f}")

    # ── Summary JSON ──
    summary = {
        "models": {
            name: {"n_docs": len(docs), "pred_file": str(PRED_FILES[name])}
            for name, docs in all_preds.items()
        },
        "per_relation": {
            "total_relations": len(rel_rows),
            "top5_4B_gain": [{"relation": r["relation"], "4B_gain": r["4B_gain"],
                              "4B_SFT_f1": r["4B_SFT_f1"], "4B_RSFT_f1": r["4B_RSFT_f1"],
                              "gold_count": r["4B_SFT_gold"]} for r in rel_rows[:5]],
            "bottom5_4B_gain": [{"relation": r["relation"], "4B_gain": r["4B_gain"],
                                 "4B_SFT_f1": r["4B_SFT_f1"], "4B_RSFT_f1": r["4B_RSFT_f1"],
                                 "gold_count": r["4B_SFT_gold"]} for r in rel_rows[-5:]],
            "top5_8B_gain": sorted(rel_rows, key=lambda x: x.get("8B_gain", 0), reverse=True)[:5],
            "bottom5_8B_gain": sorted(rel_rows, key=lambda x: x.get("8B_gain", 0))[:5],
        },
        "per_length": {
            "quantiles": len_q,
            "buckets": len_rows,
        },
        "per_entity_count": {
            "quantiles": ent_q,
            "buckets": ent_rows,
        },
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
