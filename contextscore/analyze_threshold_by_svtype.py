#!/usr/bin/env python3
"""
Analyze optimal confidence thresholds for each SV type separately.

Merges predictions with labeled benchmark data and computes precision/recall/F1
for each SV type across a range of confidence thresholds, respecting a large SV
cutoff (variants >50kb are always kept).
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path


def calculate_metrics(tp, fp, fn, tn):
    """Calculate precision, recall, F1, and specificity from confusion matrix."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, specificity


def analyze_thresholds_by_svtype(predictions_file, labels_file, output_file,
                                  min_threshold=0.05, max_threshold=0.50, step=0.05,
                                  large_cutoff=50000):
    """
    Analyze optimal thresholds for each SV type.
    
    Args:
        predictions_file: TSV with predictions (chrom, start, end, sv_type_str, sv_length_abs, confidence_score)
        labels_file: TSV with labels (chrom, start, end, sv_type_str, sv_length_abs, label)
        output_file: Output TSV file for results
        min_threshold: Minimum confidence threshold to test
        max_threshold: Maximum confidence threshold to test
        step: Step size for threshold sweep
        large_cutoff: SV size cutoff (bp); variants >this are always kept
    """
    
    logging.info(f'Loading predictions from {predictions_file}')
    predictions = pd.read_csv(predictions_file, sep='\t')
    
    logging.info(f'Loading labels from {labels_file}')
    labels = pd.read_csv(labels_file, sep='\t')
    
    # Merge on coordinate columns
    merge_cols = ['chrom', 'start', 'end', 'sv_type_str', 'sv_length_abs']
    logging.info(f'Merging predictions with labels on: {merge_cols}')
    
    merged = pd.merge(predictions, labels, on=merge_cols, how='inner')
    logging.info(f'Merged {len(merged)} variants with labels out of {len(predictions)} predictions')
    
    # Get unique SV types
    sv_types = sorted(merged['sv_type_str'].unique())
    logging.info(f'Found SV types: {sv_types}')
    
    # Generate thresholds
    thresholds = np.arange(min_threshold, max_threshold + step, step)
    
    results = []
    
    for svtype in sv_types:
        svtype_data = merged[merged['sv_type_str'] == svtype].copy()
        n_positive = (svtype_data['label'] == 1).sum()
        n_negative = (svtype_data['label'] == 0).sum()
        
        logging.info(f'\n{svtype}: {len(svtype_data)} variants (TP/TN in benchmark: {n_positive}/{n_negative})')
        
        for threshold in thresholds:
            # Apply filtering: keep if (size > large_cutoff) OR (confidence >= threshold)
            kept_mask = (svtype_data['sv_length_abs'] > large_cutoff) | (svtype_data['confidence_score'] >= threshold)
            n_kept = kept_mask.sum()
            kept_fraction = n_kept / len(svtype_data) if len(svtype_data) > 0 else 0.0
            
            # Calculate metrics on kept variants
            kept_data = svtype_data[kept_mask]
            
            if len(kept_data) > 0:
                tp = ((kept_data['confidence_score'] >= threshold) & (kept_data['label'] == 1)).sum()
                fp = ((kept_data['confidence_score'] >= threshold) & (kept_data['label'] == 0)).sum()
                fn = ((kept_data['confidence_score'] < threshold) & (kept_data['label'] == 1) & 
                      (kept_data['sv_length_abs'] <= large_cutoff)).sum()
                tn = ((kept_data['confidence_score'] < threshold) & (kept_data['label'] == 0) & 
                      (kept_data['sv_length_abs'] <= large_cutoff)).sum()
            else:
                tp = fp = fn = tn = 0
            
            precision, recall, f1, specificity = calculate_metrics(tp, fp, fn, tn)
            
            results.append({
                'sv_type': svtype,
                'threshold': threshold,
                'n_variants': len(svtype_data),
                'n_positive': n_positive,
                'n_negative': n_negative,
                'kept_count': n_kept,
                'kept_fraction': kept_fraction,
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'specificity': specificity
            })
    
    # Save results to file
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, sep='\t', index=False)
    logging.info(f'\nSaved results to {output_file}')
    
    # Print best F1 threshold for each SV type
    print("\n" + "="*80)
    print("BEST F1 THRESHOLD BY SV TYPE")
    print("="*80)
    print(f"{'SV Type':<15} {'Best Thr':>10} {'F1':>12} {'Precision':>12} {'Recall':>12} {'Kept':>10}")
    print("-"*80)
    
    for svtype in sv_types:
        svtype_results = results_df[results_df['sv_type'] == svtype]
        best_idx = svtype_results['f1'].idxmax()
        best_row = results_df.loc[best_idx]
        
        print(f"{best_row['sv_type']:<15} {best_row['threshold']:>10.2f} {best_row['f1']:>12.4f} "
              f"{best_row['precision']:>12.4f} {best_row['recall']:>12.4f} {best_row['kept_count']:>10.0f}")
    
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze optimal confidence thresholds for each SV type'
    )
    parser.add_argument('--predictions', required=True,
                        help='TSV file with predictions (output from predict.py)')
    parser.add_argument('--labels', required=True,
                        help='TSV file with benchmark labels')
    parser.add_argument('--output', required=True,
                        help='Output TSV file for threshold analysis results')
    parser.add_argument('--min-threshold', type=float, default=0.05,
                        help='Minimum confidence threshold to test (default: 0.05)')
    parser.add_argument('--max-threshold', type=float, default=0.50,
                        help='Maximum confidence threshold to test (default: 0.50)')
    parser.add_argument('--step', type=float, default=0.05,
                        help='Step size for threshold sweep (default: 0.05)')
    parser.add_argument('--large-cutoff', type=int, default=50000,
                        help='SV size cutoff in bp; variants >this are always kept (default: 50000)')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    analyze_thresholds_by_svtype(
        args.predictions,
        args.labels,
        args.output,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        step=args.step,
        large_cutoff=args.large_cutoff
    )
