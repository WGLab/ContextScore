"""Sweep confidence thresholds and report retention + optional label-based metrics.

Usage:
    python contextscore/threshold_sweep.py \
        --predictions /path/to/predictions.tsv \
        --output /path/to/threshold_sweep.tsv

If labels are available (embedded or provided via --labels-tsv),
precision/recall/F1 are also reported.
"""

import argparse
import logging
from typing import Dict, List

import numpy as np
import pandas as pd


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_binary_metrics(labels: np.ndarray, predicted_positive: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum((predicted_positive == 1) & (labels == 1)))
    fp = int(np.sum((predicted_positive == 1) & (labels == 0)))
    fn = int(np.sum((predicted_positive == 0) & (labels == 1)))
    tn = int(np.sum((predicted_positive == 0) & (labels == 0)))

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    specificity = safe_divide(tn, tn + fp)

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
    }


def build_threshold_grid(min_threshold: float, max_threshold: float, step: float) -> List[float]:
    thresholds = list(np.arange(min_threshold, max_threshold + (step / 2.0), step))
    return [round(float(th), 10) for th in thresholds]


def main() -> None:
    parser = argparse.ArgumentParser(description='Sweep confidence thresholds for ContextScore predictions.')
    parser.add_argument('--predictions', type=str, required=True, help='Path to predictions TSV.')
    parser.add_argument('--output', type=str, required=True, help='Output TSV path for threshold summary.')
    parser.add_argument('--score-col', type=str, default='confidence_score', help='Column containing confidence scores.')
    parser.add_argument('--label-col', type=str, default=None, help='Optional label column (0/1) for precision/recall/F1.')
    parser.add_argument('--labels-tsv', type=str, default=None, help='Optional labels TSV to merge with predictions.')
    parser.add_argument('--pred-id-col', type=str, default='id', help='ID column in predictions TSV for label merge.')
    parser.add_argument('--labels-id-col', type=str, default='id', help='ID column in labels TSV for label merge.')
    parser.add_argument(
        '--merge-on-cols',
        type=str,
        default=None,
        help='Optional comma-separated shared columns to merge labels (e.g. chrom,start,end,sv_type_str,sv_length_abs).',
    )
    parser.add_argument('--sv-length-col', type=str, default='sv_length', help='SV length column used with always-keep-large.')
    parser.add_argument('--always-keep-large', action='store_true', help='Apply keep rule: score>=threshold OR abs(sv_length)>large-cutoff.')
    parser.add_argument('--large-cutoff', type=int, default=10000, help='SV length cutoff for always-keep-large rule.')
    parser.add_argument('--min-threshold', type=float, default=0.05, help='Minimum threshold (inclusive).')
    parser.add_argument('--max-threshold', type=float, default=0.95, help='Maximum threshold (inclusive).')
    parser.add_argument('--step', type=float, default=0.05, help='Threshold step size.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    df = pd.read_csv(args.predictions, sep='\t')

    label_col = args.label_col
    if args.labels_tsv is not None:
        if label_col is None:
            label_col = 'label'

        labels_df = pd.read_csv(args.labels_tsv, sep='\t')
        if label_col not in labels_df.columns:
            raise ValueError(
                f"Label column '{label_col}' not found in {args.labels_tsv}"
            )

        if args.merge_on_cols is not None:
            merge_cols = [column.strip() for column in args.merge_on_cols.split(',') if column.strip()]
            if not merge_cols:
                raise ValueError('No merge columns were provided in --merge-on-cols')

            missing_in_pred = [column for column in merge_cols if column not in df.columns]
            missing_in_labels = [column for column in merge_cols if column not in labels_df.columns]
            if missing_in_pred:
                raise ValueError(
                    f"Merge columns missing in predictions: {missing_in_pred}. File: {args.predictions}"
                )
            if missing_in_labels:
                raise ValueError(
                    f"Merge columns missing in labels TSV: {missing_in_labels}. File: {args.labels_tsv}"
                )

            labels_df = labels_df[merge_cols + [label_col]].drop_duplicates(subset=merge_cols)
            df = df.merge(labels_df, on=merge_cols, how='left')
            logging.info('Merged labels using shared columns: %s', ','.join(merge_cols))
        else:
            if args.pred_id_col not in df.columns:
                raise ValueError(
                    f"Prediction ID column '{args.pred_id_col}' not found in {args.predictions}"
                )
            if args.labels_id_col not in labels_df.columns:
                raise ValueError(
                    f"Labels ID column '{args.labels_id_col}' not found in {args.labels_tsv}"
                )

            labels_df = labels_df[[args.labels_id_col, label_col]].drop_duplicates(subset=[args.labels_id_col])
            df = df.merge(labels_df, left_on=args.pred_id_col, right_on=args.labels_id_col, how='left')
            if args.labels_id_col != args.pred_id_col:
                df = df.drop(columns=[args.labels_id_col])
            logging.info('Merged labels using ID columns: %s (predictions) and %s (labels)', args.pred_id_col, args.labels_id_col)

        matched_labels = int(df[label_col].notna().sum())
        logging.info(
            'Loaded labels from %s. Matched labels for %d/%d predictions.',
            args.labels_tsv,
            matched_labels,
            len(df),
        )

    if args.score_col not in df.columns:
        raise ValueError(f"Score column '{args.score_col}' not found in {args.predictions}")

    scores = pd.to_numeric(df[args.score_col], errors='coerce').fillna(0.0).to_numpy(dtype=float)
    total = len(scores)

    has_labels = label_col is not None and label_col in df.columns
    labels = None
    labeled_mask = None
    if has_labels:
        label_values = pd.to_numeric(df[label_col], errors='coerce')
        labeled_mask = label_values.notna().to_numpy(dtype=bool)
        labeled_count = int(np.sum(labeled_mask))
        if labeled_count == 0:
            has_labels = False
            logging.info('Label column is present, but no non-missing labels were found. Running retention-only sweep.')
        else:
            labels = label_values.fillna(0).astype(int).to_numpy(dtype=int)
            logging.info('Using %d labeled records for metric computation.', labeled_count)

    has_sv_length = args.sv_length_col in df.columns
    sv_lengths = None
    if args.always_keep_large:
        if not has_sv_length:
            raise ValueError(
                f"--always-keep-large requested, but sv length column '{args.sv_length_col}' was not found in {args.predictions}"
            )
        sv_lengths = pd.to_numeric(df[args.sv_length_col], errors='coerce').fillna(0).to_numpy(dtype=float)

    thresholds = build_threshold_grid(args.min_threshold, args.max_threshold, args.step)

    rows = []
    for threshold in thresholds:
        keep_by_score = scores >= threshold

        if args.always_keep_large:
            large_sv = np.abs(sv_lengths) > args.large_cutoff
            keep_mask = keep_by_score | large_sv
        else:
            keep_mask = keep_by_score

        kept_count = int(np.sum(keep_mask))
        removed_count = total - kept_count

        row = {
            'threshold': threshold,
            'kept_count': kept_count,
            'removed_count': removed_count,
            'kept_fraction': safe_divide(kept_count, total),
            'removed_fraction': safe_divide(removed_count, total),
        }

        if has_labels and labels is not None and labeled_mask is not None:
            metrics = compute_binary_metrics(labels[labeled_mask], keep_mask.astype(int)[labeled_mask])
            row.update(metrics)
            row['labeled_count'] = int(np.sum(labeled_mask))

        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, sep='\t', index=False)
    logging.info('Saved threshold sweep to %s', args.output)

    # Print a concise summary to stdout for quick inspection.
    if has_labels:
        best_idx = out_df['f1'].idxmax()
        best_row = out_df.loc[best_idx]
        logging.info(
            'Best F1 threshold=%.3f | F1=%.4f | Precision=%.4f | Recall=%.4f | Kept=%d/%d | Labeled=%d',
            best_row['threshold'],
            best_row['f1'],
            best_row['precision'],
            best_row['recall'],
            int(best_row['kept_count']),
            total,
            int(best_row['labeled_count']),
        )
    else:
        logging.info('No label column provided; reported retention-only threshold sweep.')


if __name__ == '__main__':
    main()
