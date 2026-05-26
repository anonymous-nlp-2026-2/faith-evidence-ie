import json, re, sys
sys.path.insert(0, '.')
from eval.evaluator import DocREDEvaluator

TRAIN_PATH = 'data/docred/train_annotated.json'
NOEVI_PRED = 'eval_results/qwen3_1_7b_no_evidence_eval_v3/predictions.json'
WITHEVI_PRED = 'eval_results/qwen3_1_7b_sft_eval/predictions.json'

def is_repetitive(raw_output, threshold=3):
    pattern = r'\{[^{}]+\}'
    matches = re.findall(pattern, raw_output)
    if len(matches) < threshold:
        return False
    for i in range(len(matches) - threshold + 1):
        if len(set(matches[i:i+threshold])) == 1:
            return True
    if len(matches) > 5 and len(set(matches)) / len(matches) < 0.5:
        return True
    return False

def load_and_tag(path):
    with open(path) as f:
        preds = json.load(f)
    for p in preds:
        p['repetitive'] = is_repetitive(p['raw_output'])
    return preds

def filter_preds_gold(preds, doc_ids):
    ids = set(doc_ids)
    filtered_preds = []
    filtered_gold = []
    for p in preds:
        if p['doc_id'] not in ids:
            continue
        for t in p['parsed_triples']:
            if isinstance(t, dict):
                filtered_preds.append({
                    'doc_id': p['doc_id'],
                    'head': t.get('head', ''),
                    'tail': t.get('tail', ''),
                    'relation': t.get('relation', ''),
                })
        for t in p['gold_triples']:
            if isinstance(t, dict):
                filtered_gold.append({
                    'doc_id': p['doc_id'],
                    'head': t.get('head', ''),
                    'tail': t.get('tail', ''),
                    'relation': t.get('relation', ''),
                })
    return filtered_preds, filtered_gold

# Load evaluator with train triples
evaluator = DocREDEvaluator.from_train_file(TRAIN_PATH)

noevi = load_and_tag(NOEVI_PRED)
withevi = load_and_tag(WITHEVI_PRED)

# Stats
noevi_n_trunc = sum(1 for p in noevi if p['truncated'])
noevi_n_rep = sum(1 for p in noevi if p['repetitive'])
noevi_n_both = sum(1 for p in noevi if p['truncated'] and p['repetitive'])
noevi_n_clean = sum(1 for p in noevi if not p['truncated'] and not p['repetitive'])

withevi_n_trunc = sum(1 for p in withevi if p['truncated'])
withevi_n_rep = sum(1 for p in withevi if p['repetitive'])
withevi_n_both = sum(1 for p in withevi if p['truncated'] and p['repetitive'])
withevi_n_clean = sum(1 for p in withevi if not p['truncated'] and not p['repetitive'])

print("="*65)
print("  REPETITION & TRUNCATION ANALYSIS (1.7B)")
print("="*65)
print(f"\n  {'Metric':<30} {'No-Evidence':>12} {'With-Evidence':>14}")
print(f"  {'-'*30} {'-'*12} {'-'*14}")
print(f"  {'Total docs':<30} {'985':>12} {'985':>14}")
print(f"  {'Truncated':<30} {noevi_n_trunc:>12} {withevi_n_trunc:>14}")
print(f"  {'Truncated %':<30} {noevi_n_trunc/985*100:>11.1f}% {withevi_n_trunc/985*100:>13.1f}%")
print(f"  {'Repetitive':<30} {noevi_n_rep:>12} {withevi_n_rep:>14}")
print(f"  {'Repetitive %':<30} {noevi_n_rep/985*100:>11.1f}% {withevi_n_rep/985*100:>13.1f}%")
print(f"  {'Both trunc+rep':<30} {noevi_n_both:>12} {withevi_n_both:>14}")
print(f"  {'Trunc only (not rep)':<30} {noevi_n_trunc - noevi_n_both:>12} {withevi_n_trunc - withevi_n_both:>14}")
print(f"  {'Rep only (not trunc)':<30} {noevi_n_rep - noevi_n_both:>12} {withevi_n_rep - withevi_n_both:>14}")
print(f"  {'Clean':<30} {noevi_n_clean:>12} {withevi_n_clean:>14}")
print(f"  {'Clean %':<30} {noevi_n_clean/985*100:>11.1f}% {withevi_n_clean/985*100:>13.1f}%")

# Define subsets
all_ids = [p['doc_id'] for p in noevi]

noevi_bad = set(p['doc_id'] for p in noevi if p['truncated'] or p['repetitive'])
withevi_bad = set(p['doc_id'] for p in withevi if p['truncated'] or p['repetitive'])
union_bad = noevi_bad | withevi_bad
fair_strict = [d for d in all_ids if d not in union_bad]

noevi_trunc_set = set(p['doc_id'] for p in noevi if p['truncated'])
withevi_trunc_set = set(p['doc_id'] for p in withevi if p['truncated'])
union_trunc = noevi_trunc_set | withevi_trunc_set
fair_trunc = [d for d in all_ids if d not in union_trunc]

print(f"\n{'='*65}")
print(f"  FAIR SUBSET DEFINITIONS")
print(f"{'='*65}")
print(f"  Strict (excl trunc|rep in either):  {len(fair_strict)}/985 ({len(fair_strict)/985*100:.1f}%)")
print(f"  Trunc-only (excl trunc in either):  {len(fair_trunc)}/985 ({len(fair_trunc)/985*100:.1f}%)")

# Compute ign_f1 on each subset using official evaluator
def eval_subset(noevi_preds, withevi_preds, doc_ids, label):
    ne_p, ne_g = filter_preds_gold(noevi_preds, doc_ids)
    we_p, we_g = filter_preds_gold(withevi_preds, doc_ids)
    
    ne_metrics = evaluator.compute_f1(ne_p, ne_g)
    we_metrics = evaluator.compute_f1(we_p, we_g)
    
    ne_ign = ne_metrics['ign_f1']
    we_ign = we_metrics['ign_f1']
    tax = ne_ign - we_ign
    
    print(f"\n  {label} ({len(doc_ids)} docs)")
    print(f"    No-Evidence:   Ign-P={ne_metrics['ign_precision']:.4f}  Ign-R={ne_metrics['ign_recall']:.4f}  Ign-F1={ne_ign:.4f}")
    print(f"    With-Evidence: Ign-P={we_metrics['ign_precision']:.4f}  Ign-R={we_metrics['ign_recall']:.4f}  Ign-F1={we_ign:.4f}")
    print(f"    Evidence tax (Ign-F1):  {tax*100:+.2f}pp")
    return ne_ign, we_ign

print(f"\n{'='*65}")
print(f"  IGN-F1 COMPARISON (official evaluator)")
print(f"{'='*65}")

ne_full, we_full = eval_subset(noevi, withevi, all_ids, "Full set")
ne_fair, we_fair = eval_subset(noevi, withevi, fair_strict, "Fair (strict: excl trunc+rep)")
ne_ft, we_ft = eval_subset(noevi, withevi, fair_trunc, "Fair (trunc-only exclusion)")

# Summary
print(f"\n{'='*65}")
print(f"  SUMMARY")
print(f"{'='*65}")
print(f"  Full set evidence tax:    {(ne_full - we_full)*100:+.2f}pp  (no-evi {ne_full:.4f} vs with-evi {we_full:.4f})")
print(f"  Fair strict evidence tax:  {(ne_fair - we_fair)*100:+.2f}pp  (no-evi {ne_fair:.4f} vs with-evi {we_fair:.4f})")
print(f"  Fair trunc-only tax:       {(ne_ft - we_ft)*100:+.2f}pp  (no-evi {ne_ft:.4f} vs with-evi {we_ft:.4f})")
print(f"\n  Conclusion: Excluding truncated/repetitive docs {'reduces' if abs(ne_fair-we_fair) < abs(ne_full-we_full) else 'increases'} the evidence tax gap.")
print(f"  Evidence tax direction: with-evidence {'helps' if (ne_full - we_full) < 0 else 'hurts'} at 1.7B scale on all subsets.")

