import argparse
import os
import time

import pandas as pd

import core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', default=core.DEFAULT_VARIANTS)
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--datasets', nargs='+', default=list(core.DATASET_CONFIGS))
    ap.add_argument('--figures-variant', default='V6')
    ap.add_argument('--figures-seed', type=int, default=0)
    ap.add_argument('--no-figures', action='store_true')
    args = ap.parse_args()

    results_csv = core.out_path('results.csv')
    done = set()
    if os.path.exists(results_csv):
        prev = pd.read_csv(results_csv)
        done = set(zip(prev['variant'], prev['seed'], prev['dataset']))

    jobs = [(ds, v, s) for ds in args.datasets for v in args.variants for s in args.seeds]
    print(f'{len(jobs)} runs requested, {len([j for j in jobs if (j[1], j[2], j[0]) in done])} '
          f'already finished | device: {core.get_device()}')

    t_all = time.time()
    for k, (ds, v, s) in enumerate(jobs, 1):
        if (v, s, ds) in done:
            continue
        t0 = time.time()
        want_fig = (not args.no_figures
                    and (args.figures_variant == 'all' or v == args.figures_variant)
                    and (args.figures_seed < 0 or s == args.figures_seed))
        print(f'\n[{k}/{len(jobs)}] {ds} {v} ({core.VARIANTS[v]["name"]}) seed {s}')
        info = core.train_one(v, s, ds, make_figures=want_fig)
        if info is None:
            continue
        metrics = core.evaluate_one(v, s, ds, make_figures=want_fig)
        if metrics is None:
            continue
        row = {**info, **metrics, 'clip_pred_to_rul_max': int(core.CLIP_PRED_TO_RUL_MAX),
               'minutes': round((time.time() - t0) / 60, 2)}
        pd.DataFrame([row]).to_csv(results_csv, mode='a',
                                   header=not os.path.exists(results_csv), index=False)
        print(f'  done in {row["minutes"]} min')
    print(f'\nAll finished in {(time.time() - t_all) / 60:.1f} min -> {results_csv}')


if __name__ == '__main__':
    main()
