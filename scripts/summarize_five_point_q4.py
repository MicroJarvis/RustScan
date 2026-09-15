#!/usr/bin/env python3
"""Aggregate Q4 sidecars, never print per-trial or model-bit JSON.

Run from the worktree root; --output refuses overwrite. This verifies comparisons,
not universal per-pair quality dominance or portability of WGSL compensation.
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--candidate-1e3', type=Path, required=True)
    parser.add_argument('--candidate-1e2', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bs = [args.baseline_dir / 'tol1e3.json', args.baseline_dir / 'tol1e2.json']
    cs = [args.candidate_1e3, args.candidate_1e2]
    baseline, candidate = [list(map(load, paths)) for paths in (bs, cs)]
    assert not args.output.exists(), 'refusing overwrite'
    for reports, paths in [(baseline, bs), (candidate, cs)]:
        assert reports[0]['signatures'] == reports[1]['signatures']
        assert digest(paths[0].with_suffix('.bits.json')) == digest(paths[1].with_suffix('.bits.json'))
    comparisons = {}
    for b, c, bp, cp in zip(baseline, candidate, bs, cs):
        assert b['input_digest'] == c['input_digest']
        assert b['signatures']['cpu_f64'] == c['signatures']['cpu_f64']
        assert b['counts']['trials'] == c['counts']['trials'] == 65024
        raw_b, raw_c = [p.with_suffix('.stages.bin').read_bytes() for p in (bp, cp)]
        assert len(raw_b) == len(raw_c) == 65024 * 1544
        changed = sum(raw_b[i:i+1500] != raw_c[i:i+1500] for i in range(0,len(raw_b),1544))
        assert changed == 0, 'basis/A/solve/B changed'
        tol = str(b['root_match_distance']['tolerance'])
        before = b['coefficient_error_vs_same_basis_f64']['trials_with_downstream_loss']['p90']
        after = c['fixed_baseline_loss_coefficient_error']['p90']
        comparisons[tol] = {'fixed_loss_p90_improvement_factor': before/after,
                            'upstream_records_changed': changed}
        assert after < before, 'loss-cohort coefficient precision did not improve'
        assert c['root_classes_of_f64_real_roots']['MissingNoSlot'] < b['root_classes_of_f64_real_roots']['MissingNoSlot']
        assert c['model_gap_ledger']['spurious_gpu_models_no_f64_root'] < b['model_gap_ledger']['spurious_gpu_models_no_f64_root']
        assert c['counts']['cpu_has_gpu_empty'] < b['counts']['cpu_has_gpu_empty']
    bq, cq = [load(p.with_suffix('.quality.json')) for p in (bs[1], cs[1])]
    assert len(bq) == len(cq) == 65024
    assert all(b[0] == c[0] for b,c in zip(bq,cq)), 'CPU quality changed'
    cpu_pair = baseline[1]['model_quality']['pair_cpu_best']
    bucket_names = ['zero', '(0,.25)', '[.25,.5)', '[.5,.75)', '[.75,1)', '>=1']
    def bucket(x):
        return 0 if x == 0 else 1 if x < .25 else 2 if x < .5 else 3 if x < .75 else 4 if x < 1 else 5
    buckets = {}
    for name, rows in [('baseline',bq), ('candidate',cq)]:
        counts = [0]*6
        for i,row in enumerate(rows):
            counts[bucket(row[1]/max(1,cpu_pair[i//512]))] += 1
        buckets[name] = dict(zip(bucket_names,counts))
    pairs_b, pairs_c = [r[1]['model_quality']['pair_gpu_best'] for r in (baseline,candidate)]
    trial_counts = {'better':sum(c[1]>b[1] for b,c in zip(bq,cq)),
                    'worse':sum(c[1]<b[1] for b,c in zip(bq,cq)),
                    'equal':sum(c[1]==b[1] for b,c in zip(bq,cq))}
    timing = {}
    for batch in ['512','65024']:
        samples = [[x for r in reports for x in
                    r['timing_full_diagnostic_replay_seconds_by_batch'][batch]]
                   for reports in (baseline, candidate)]
        timing[batch] = {'baseline_seconds': samples[0],
                         'candidate_seconds': samples[1],
                         'baseline_median': median(samples[0]),
                         'candidate_median': median(samples[1]),
                         'candidate_over_baseline': median(samples[1])/median(samples[0])}
    pair_counts = {'better': sum(c > b for b, c in zip(pairs_b, pairs_c)),
                   'worse': sum(c < b for b, c in zip(pairs_b, pairs_c)),
                   'equal': sum(c == b for b, c in zip(pairs_b, pairs_c))}
    result = {'comparisons': comparisons, 'quality_buckets': buckets,
              'trial_quality_changes': trial_counts, 'pair_quality_changes': pair_counts,
              'timing': timing,
              'signatures': {'baseline': baseline[0]['signatures'],
                             'candidate': candidate[0]['signatures']},
              'artifact_sha256': {str(p.with_suffix(s)): digest(p.with_suffix(s))
                  for p in bs + cs
                  for s in ['.json', '.bits.json', '.trials.json', '.stages.bin']},
              'limits': ['Timings are historical baseline vs candidate, not interleaved.',
                         'Full diagnostic replay includes host/readback, not kernel-only.',
                         'Quality is fixed-input Sampson scoring, not end-to-end RANSAC.',
                         'No universal per-pair dominance or cross-device portability claim.']}
    with args.output.open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ['artifact_sha256', 'signatures']}, indent=2))


if __name__ == '__main__':
    main()