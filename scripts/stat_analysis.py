"""Statistical analysis of FREIGE eval results."""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


# ── Method group definitions ──────────────────────────────────────────
# group_name -> list of (subdir, seed_label)
# For groups with fallback dirs, first match wins per seed slot.
METHOD_GROUPS = {
    "SFT baseline": {
        "seeds": [("sft_baseline_noquant", None)],
    },
    "SFT no-evidence": {
        "seeds": [("sft_no_evidence", None)],
    },
    "RSFT-CED": {
        "seeds": [
            ("rsft_r1", "s42"),
            ("rsft_s43", "s43"),
            ("rsft_s44", "s44"),
        ],
    },
    "RSFT-flatNLI": {
        "seeds": [
            ("rsft_flat_nli", "s42"),
            ("rsft_flat_nli_s43_reeval", "s43"),
            ("rsft_flat_nli_s43", "s43"),  # fallback
            ("rsft_flat_nli_s44", "s44"),
            ("rsft_flat_nli_s45", "s45"),
        ],
    },
    "GRPO bf16": {
        "seeds": [("grpo_g8_bf16_ckpt100", None)],
    },
    "DPO-CED 1ep": {
        "seeds": [
            ("dpo_ced_1ep_fixed", None),
            ("dpo_ced_1ep", None),
        ],
    },
    "CED reranker": {
        "seeds": [("ced_reranker_poc", None)],
        "loader": "reranker",
    },
    "RSFT top_k=3": {
        "seeds": [
            ("rsft_topk3_fixed", None),
            ("rsft_topk3", None),
        ],
    },
}

MIN_F1 = 0.01
METRICS_KEYS = ["rel_f1", "evi_f1", "edcr", "precision", "recall",
                "evi_f1_joint", "n_predictions", "n_truncated"]


# ── Loaders ───────────────────────────────────────────────────────────

def _load_standard(path: Path) -> dict | None:
    mf = path / "metrics.json"
    if not mf.exists():
        return None
    with open(mf) as f:
        m = json.load(f)
    if m.get("f1", 0) < MIN_F1:
        return None
    return {
        "rel_f1": m["f1"],
        "evi_f1": m.get("evi_f1", 0.0),
        "edcr": m.get("edcr", 0.0),
        "precision": m.get("precision", 0.0),
        "recall": m.get("recall", 0.0),
        "evi_f1_joint": m.get("evi_f1_joint", 0.0),
        "n_predictions": m.get("n_predictions", 0),
        "n_truncated": m.get("n_truncated", 0),
    }


def _load_reranker(path: Path) -> dict | None:
    rf = path / "results.json"
    if not rf.exists():
        return None
    with open(rf) as f:
        data = json.load(f)
    r = data.get("ced_reranker")
    if not r or r.get("f1", 0) < MIN_F1:
        return None
    return {
        "rel_f1": r["f1"],
        "evi_f1": r.get("evi_f1", 0.0),
        "edcr": r.get("edcr", 0.0),
        "precision": r.get("precision", 0.0),
        "recall": r.get("recall", 0.0),
        "evi_f1_joint": 0.0,
        "n_predictions": r.get("n_predictions", 0),
        "n_truncated": r.get("n_truncated", 0),
    }


LOADERS = {
    "standard": _load_standard,
    "reranker": _load_reranker,
}


# ── Data collection ──────────────────────────────────────────────────

def collect(results_dir: Path) -> dict[str, list[dict]]:
    groups = {}
    for gname, cfg in METHOD_GROUPS.items():
        loader = LOADERS[cfg.get("loader", "standard")]
        seen_seeds = set()
        results = []
        for subdir, seed in cfg["seeds"]:
            if seed and seed in seen_seeds:
                continue
            path = results_dir / subdir
            m = loader(path)
            if m is not None:
                m["subdir"] = subdir
                m["seed"] = seed
                results.append(m)
                if seed:
                    seen_seeds.add(seed)
        if results:
            groups[gname] = results
    return groups


# ── Statistics ────────────────────────────────────────────────────────

def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = np.sqrt((va + vb) / 2)
    if pooled == 0:
        return float("inf")
    return (np.mean(a) - np.mean(b)) / pooled


def ttest(group_a, group_b, metric="rel_f1"):
    va = np.array([s[metric] for s in group_a])
    vb = np.array([s[metric] for s in group_b])

    if len(va) < 2 or len(vb) < 2:
        return {
            "status": "insufficient_data",
            "n_a": len(va), "n_b": len(vb),
            "mean_a": float(np.mean(va)), "mean_b": float(np.mean(vb)),
        }

    if len(va) == len(vb):
        t, p = stats.ttest_rel(va, vb)
        kind = "paired"
    else:
        t, p = stats.ttest_ind(va, vb, equal_var=False)
        kind = "welch"

    return {
        "status": "ok",
        "test": kind,
        "t": float(t),
        "p": float(p),
        "cohens_d": float(cohens_d(va, vb)),
        "n_a": len(va), "n_b": len(vb),
        "mean_a": float(np.mean(va)),
        "mean_b": float(np.mean(vb)),
        "std_a": float(np.std(va, ddof=1)),
        "std_b": float(np.std(vb, ddof=1)),
    }


# ── Formatting ────────────────────────────────────────────────────────

def _val(mean, std, n):
    if n > 1 and std > 0:
        return f"{mean:.4f}±{std:.4f}"
    return f"{mean:.4f}"


def print_summary(groups):
    header = f"{'Method':<22} {'n':>2} {'rel_f1':>14} {'evi_f1':>14} {'EDCR':>14}"
    print(header)
    print("-" * len(header))
    for gname, seeds in groups.items():
        n = len(seeds)
        vals = {k: [s[k] for s in seeds] for k in ["rel_f1", "evi_f1", "edcr"]}
        rf = _val(np.mean(vals["rel_f1"]), np.std(vals["rel_f1"], ddof=1) if n > 1 else 0, n)
        ef = _val(np.mean(vals["evi_f1"]), np.std(vals["evi_f1"], ddof=1) if n > 1 else 0, n)
        ed = _val(np.mean(vals["edcr"]), np.std(vals["edcr"], ddof=1) if n > 1 else 0, n)
        print(f"{gname:<22} {n:>2} {rf:>14} {ef:>14} {ed:>14}")


def print_ttest(name_a, name_b, groups):
    ga = groups.get(name_a)
    gb = groups.get(name_b)
    if not ga or not gb:
        print(f"\n  Cannot compare: {name_a} has {len(ga) if ga else 0}, "
              f"{name_b} has {len(gb) if gb else 0} results")
        return

    for metric in ["rel_f1", "evi_f1", "edcr"]:
        r = ttest(ga, gb, metric)
        print(f"\n  {metric}:")
        print(f"    {name_a}: {r['mean_a']:.4f}" +
              (f" ± {r['std_a']:.4f}" if r.get("std_a") else "") +
              f" (n={r['n_a']})")
        print(f"    {name_b}: {r['mean_b']:.4f}" +
              (f" ± {r['std_b']:.4f}" if r.get("std_b") else "") +
              f" (n={r['n_b']})")
        if r["status"] == "ok":
            sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else "ns"
            print(f"    {r['test']} t={r['t']:.3f}, p={r['p']:.4f} {sig}, Cohen's d={r['cohens_d']:.3f}")
        else:
            print(f"    [insufficient data for t-test]")


def print_seed_detail(groups):
    for gname, seeds in groups.items():
        if len(seeds) <= 1:
            continue
        print(f"\n  {gname}:")
        for s in seeds:
            label = s.get("seed") or s["subdir"]
            print(f"    {label:<6} rel_f1={s['rel_f1']:.4f}  evi_f1={s['evi_f1']:.4f}  edcr={s['edcr']:.4f}")


def to_json(groups):
    out = {"summary": {}, "groups": {}, "tests": {}}
    for gname, seeds in groups.items():
        vals = {k: [s[k] for s in seeds] for k in ["rel_f1", "evi_f1", "edcr"]}
        n = len(seeds)
        out["summary"][gname] = {
            "n": n,
            "rel_f1_mean": float(np.mean(vals["rel_f1"])),
            "rel_f1_std": float(np.std(vals["rel_f1"], ddof=1)) if n > 1 else 0.0,
            "evi_f1_mean": float(np.mean(vals["evi_f1"])),
            "evi_f1_std": float(np.std(vals["evi_f1"], ddof=1)) if n > 1 else 0.0,
            "edcr_mean": float(np.mean(vals["edcr"])),
            "edcr_std": float(np.std(vals["edcr"], ddof=1)) if n > 1 else 0.0,
        }
        out["groups"][gname] = seeds

    for a, b in [("RSFT-CED", "RSFT-flatNLI"), ("RSFT-CED", "SFT baseline")]:
        if a in groups and b in groups:
            out["tests"][f"{a}_vs_{b}"] = {
                m: ttest(groups[a], groups[b], m) for m in ["rel_f1", "evi_f1", "edcr"]
            }
    return out


def generate_latex(groups):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Cross-method comparison on DocRED dev set}",
        r"\label{tab:cross-method}",
        r"\begin{tabular}{l c c c c}",
        r"\toprule",
        r"Method & $n$ & Rel F\textsubscript{1} & Evi F\textsubscript{1} & EDCR \\",
        r"\midrule",
    ]
    for gname, seeds in groups.items():
        n = len(seeds)
        vals = {k: [s[k] for s in seeds] for k in ["rel_f1", "evi_f1", "edcr"]}
        rf_m = np.mean(vals["rel_f1"])
        ef_m = np.mean(vals["evi_f1"])
        ed_m = np.mean(vals["edcr"])
        name_tex = gname.replace("_", r"\_")
        if n > 1:
            rf_s = np.std(vals["rel_f1"], ddof=1)
            ef_s = np.std(vals["evi_f1"], ddof=1)
            ed_s = np.std(vals["edcr"], ddof=1)
            lines.append(
                f"  {name_tex} & {n} & "
                f"{rf_m:.4f}$\\pm${rf_s:.4f} & "
                f"{ef_m:.4f}$\\pm${ef_s:.4f} & "
                f"{ed_m:.4f}$\\pm${ed_s:.4f} \\\\"
            )
        else:
            lines.append(
                f"  {name_tex} & {n} & {rf_m:.4f} & {ef_m:.4f} & {ed_m:.4f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="FREIGE stat analysis")
    ap.add_argument("--results_dir", default="/workspace/eval_results/")
    ap.add_argument("--format", choices=["text", "json", "latex"], default="text")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    groups = collect(Path(args.results_dir))
    if not groups:
        print("No valid results found.", file=sys.stderr)
        sys.exit(1)

    if args.format == "json":
        out = to_json(groups)
        text = json.dumps(out, indent=2, ensure_ascii=False)
    elif args.format == "latex":
        text = generate_latex(groups)
    else:
        print("=" * 72)
        print("FREIGE Cross-Method Summary")
        print("=" * 72)
        print_summary(groups)

        print("\n" + "=" * 72)
        print("Per-Seed Detail")
        print("=" * 72)
        print_seed_detail(groups)

        print("\n" + "=" * 72)
        print("t-test: RSFT-CED vs RSFT-flatNLI")
        print("=" * 72)
        print_ttest("RSFT-CED", "RSFT-flatNLI", groups)

        print("\n" + "=" * 72)
        print("t-test: RSFT-CED vs SFT baseline")
        print("=" * 72)
        print_ttest("RSFT-CED", "SFT baseline", groups)
        text = None

    if text is not None:
        if args.output:
            Path(args.output).write_text(text)
            print(f"Written to {args.output}", file=sys.stderr)
        else:
            print(text)


if __name__ == "__main__":
    main()
