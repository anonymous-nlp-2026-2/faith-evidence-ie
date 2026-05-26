"""Fair evidence tax comparison.

Computes ign_f1 on the subset of documents where both with-evidence and
no-evidence predictions have format_ok=True, giving a fair measurement
of the evidence tax uncontaminated by format failures.

Usage:
    python fair_evidence_tax.py \
        --with-evi /path/to/with_evidence/predictions.json \
        --no-evi /path/to/no_evidence/predictions.json \
        [--train /path/to/train_annotated.json] \
        [--json-out /path/to/output.json]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _load_train_facts(train_path: str) -> set[tuple]:
    sys.path.insert(0, ".")
    from freige.data.docred_processor import DOCRED_REL_INFO

    with open(train_path) as f:
        data = json.load(f)
    facts = set()
    for doc in data:
        vs = doc["vertexSet"]
        for lab in doc.get("labels", []):
            h = vs[lab["h"]][0]["name"].lower().strip()
            t = vs[lab["t"]][0]["name"].lower().strip()
            r = DOCRED_REL_INFO.get(lab["r"], lab["r"]).lower().strip()
            facts.add((h, t, r))
    return facts


def _triple_key(doc_id, head, tail, relation):
    return (doc_id, head.lower().strip(), tail.lower().strip(), relation.lower().strip())


def compute_ign_f1(docs: list[dict], train_facts: set[tuple]) -> dict:
    """Compute ign_f1 over a list of prediction docs."""
    pred_set = set()
    gold_set = set()
    for d in docs:
        doc_id = d["doc_id"]
        for t in d["parsed_triples"]:
            pred_set.add(_triple_key(doc_id, t["head"], t["tail"], t["relation"]))
        for t in d["gold_triples"]:
            gold_set.add(_triple_key(doc_id, t["head"], t["tail"], t["relation"]))

    tp = pred_set & gold_set

    # full f1
    prec = len(tp) / len(pred_set) if pred_set else 0.0
    rec = len(tp) / len(gold_set) if gold_set else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # ign f1
    pred_ign = {k for k in pred_set if k[1:] not in train_facts}
    gold_ign = {k for k in gold_set if k[1:] not in train_facts}
    tp_ign = pred_ign & gold_ign
    ign_prec = len(tp_ign) / len(pred_ign) if pred_ign else 0.0
    ign_rec = len(tp_ign) / len(gold_ign) if gold_ign else 0.0
    ign_f1 = 2 * ign_prec * ign_rec / (ign_prec + ign_rec) if (ign_prec + ign_rec) > 0 else 0.0

    return {
        "f1": f1, "precision": prec, "recall": rec,
        "ign_f1": ign_f1, "ign_precision": ign_prec, "ign_recall": ign_rec,
        "tp": len(tp), "pred_count": len(pred_set), "gold_count": len(gold_set),
    }


def classify_failure(doc: dict) -> str:
    if doc.get("truncated", False):
        return "truncated"
    return "malformed_json"


def main():
    parser = argparse.ArgumentParser(description="Fair evidence tax comparison")
    parser.add_argument("--with-evi", required=True, help="With-evidence predictions.json")
    parser.add_argument("--no-evi", required=True, help="No-evidence predictions.json")
    parser.add_argument("--train", default="data/docred/train_annotated.json",
                        help="Train file for ign_f1 computation")
    parser.add_argument("--json-out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    # Load
    with open(args.with_evi) as f:
        with_preds = json.load(f)
    with open(args.no_evi) as f:
        no_preds = json.load(f)
    train_facts = _load_train_facts(args.train)

    # Index by doc_id
    with_by_doc = {d["doc_id"]: d for d in with_preds}
    no_by_doc = {d["doc_id"]: d for d in no_preds}

    all_doc_ids = sorted(set(with_by_doc.keys()) | set(no_by_doc.keys()))
    common_doc_ids = sorted(set(with_by_doc.keys()) & set(no_by_doc.keys()))

    # Format compliance
    with_ok = {did for did in common_doc_ids if with_by_doc[did].get("format_ok", True)}
    no_ok = {did for did in common_doc_ids if no_by_doc[did].get("format_ok", True)}
    fair_ids = sorted(with_ok & no_ok)

    # Failure classification (no-evidence side)
    no_failures = [no_by_doc[did] for did in common_doc_ids if did not in no_ok]
    n_trunc = sum(1 for d in no_failures if classify_failure(d) == "truncated")
    n_malformed = len(no_failures) - n_trunc

    # With-evidence failures
    with_failures = [with_by_doc[did] for did in common_doc_ids if did not in with_ok]
    with_n_trunc = sum(1 for d in with_failures if classify_failure(d) == "truncated")
    with_n_malformed = len(with_failures) - with_n_trunc

    # Compute metrics on full set
    with_full_docs = [with_by_doc[did] for did in common_doc_ids]
    no_full_docs = [no_by_doc[did] for did in common_doc_ids]
    with_full = compute_ign_f1(with_full_docs, train_facts)
    no_full = compute_ign_f1(no_full_docs, train_facts)

    # Compute metrics on fair subset
    with_fair_docs = [with_by_doc[did] for did in fair_ids]
    no_fair_docs = [no_by_doc[did] for did in fair_ids]
    with_fair = compute_ign_f1(with_fair_docs, train_facts)
    no_fair = compute_ign_f1(no_fair_docs, train_facts)

    raw_tax = (no_full["ign_f1"] - with_full["ign_f1"]) * 100
    fair_tax = (no_fair["ign_f1"] - with_fair["ign_f1"]) * 100

    # Print report
    print("=" * 55)
    print("  Evidence Tax Fair Comparison")
    print("=" * 55)
    print(f"\nFull set: {len(common_doc_ids)} docs")
    print(f"  With-evidence: ign_f1={with_full['ign_f1']:.4f}  (format_ok={len(with_ok)}/{len(common_doc_ids)})")
    print(f"  No-evidence:   ign_f1={no_full['ign_f1']:.4f}  (format_ok={len(no_ok)}/{len(common_doc_ids)})")
    print(f"  Raw tax: {raw_tax:+.2f}pp")

    print(f"\nFair subset: {len(fair_ids)} docs (both format-OK)")
    print(f"  With-evidence: ign_f1={with_fair['ign_f1']:.4f}")
    print(f"  No-evidence:   ign_f1={no_fair['ign_f1']:.4f}")
    print(f"  Fair tax: {fair_tax:+.2f}pp")

    print(f"\nFormat failures (no-evidence): {len(no_failures)} docs")
    print(f"  Truncated:      {n_trunc}")
    print(f"  Malformed JSON: {n_malformed}")

    if with_failures:
        print(f"\nFormat failures (with-evidence): {len(with_failures)} docs")
        print(f"  Truncated:      {with_n_trunc}")
        print(f"  Malformed JSON: {with_n_malformed}")

    # Detailed breakdown
    print(f"\n{'─' * 55}")
    print("Detailed metrics:")
    for label, full_m, fair_m in [("With-evidence", with_full, with_fair),
                                   ("No-evidence", no_full, no_fair)]:
        print(f"\n  {label}:")
        print(f"    Full:  P={full_m['ign_precision']:.4f}  R={full_m['ign_recall']:.4f}  F1={full_m['ign_f1']:.4f}  (pred={full_m['pred_count']}, gold={full_m['gold_count']})")
        print(f"    Fair:  P={fair_m['ign_precision']:.4f}  R={fair_m['ign_recall']:.4f}  F1={fair_m['ign_f1']:.4f}  (pred={fair_m['pred_count']}, gold={fair_m['gold_count']})")

    # JSON output
    if args.json_out:
        result = {
            "full_set": {
                "n_docs": len(common_doc_ids),
                "with_evidence": with_full,
                "no_evidence": no_full,
                "raw_tax_pp": round(raw_tax, 4),
            },
            "fair_subset": {
                "n_docs": len(fair_ids),
                "with_evidence": with_fair,
                "no_evidence": no_fair,
                "fair_tax_pp": round(fair_tax, 4),
            },
            "format_failures": {
                "no_evidence": {"total": len(no_failures), "truncated": n_trunc, "malformed_json": n_malformed},
                "with_evidence": {"total": len(with_failures), "truncated": with_n_trunc, "malformed_json": with_n_malformed},
            },
            "with_format_ok": len(with_ok),
            "no_format_ok": len(no_ok),
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nJSON output saved to {args.json_out}")


if __name__ == "__main__":
    main()
