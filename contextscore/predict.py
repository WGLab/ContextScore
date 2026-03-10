"""
scoring_model.py: Score the structural variants using the binary classification
model.

Usage:
    scoring_model.py <input> <output> <model>

Arguments:
    <input>     Path to the input VCF file.
    <model>     Path to the model file.
"""

import os
import sys
import logging
import numpy as np
import joblib
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from extract_features import extract_features

def create_bed(input_vcf, output_bed):
    """Create a BED file from the input VCF file. Extract the following fields:
    1. Chromosome (CHROM)
    2. Start position (POS)
    3. End position (END)
    4. SV type (SVTYPE)
    5. SV length (SVLEN)
    6. Genotype (GT)
    7. Read depth (DP)
    8. HMM log likelihood (HMM)
    9. Alignment type (ALN)
    10. Cluster size (CLUSTER)
    11. Copy number state (CN)
    12. Read alignment offset (ALNOFFSET)    
    Args:
        input_vcf (str): Path to the input VCF file.
        output_bed (str): Path to the output BED file.
    """
    logging.info('Reading VCF file: %s', input_vcf)
    vcf_df = pd.read_csv(input_vcf, sep='\t', comment='#', header=None, 
                         names=['CHROM', 'POS', 'INFO', 'FORMAT', 'SAMPLE'], usecols=[0, 1, 7, 8, 9],
                            dtype={'CHROM': str, 'POS': int, 'INFO': str, 'FORMAT': str, 'SAMPLE': str})
    
    # Add a column for the ID field with the VCF row number
    vcf_df['id'] = vcf_df.index
    
    info_df = pd.DataFrame()
    info_df['ALN'] = vcf_df['INFO'].str.extract(r'ALN=([^;]+)')
    info_df['END'] = vcf_df['INFO'].str.extract(r'END=(\d+)')
    info_df['SVTYPE'] = vcf_df['INFO'].str.extract(r'SVTYPE=([^;]+)')
    info_df['SVLEN'] = vcf_df['INFO'].str.extract(r'SVLEN=([^;]+)')
    info_df['HMM'] = vcf_df['INFO'].str.extract(r'HMM=([^;]+)')
    info_df['CLUSTER'] = vcf_df['INFO'].str.extract(r'CLUSTER=([^;]+)')
    info_df['CN'] = vcf_df['INFO'].str.extract(r'CN=([^;]+)')
    info_df['ALNOFFSET'] = vcf_df['INFO'].str.extract(r'ALNOFFSET=([^;]+)')

    # Extract the genotype (GT) and read depth (DP) from the SAMPLE column
    sample_df = pd.DataFrame()
    sample_df['GT'] = vcf_df['SAMPLE'].str.extract(r'([^:]+):')
    sample_df['DP'] = vcf_df['SAMPLE'].str.extract(r':(\d+)').astype(int)

    # Create the BED file
    bed_df = pd.DataFrame()
    bed_df['CHROM'] = vcf_df['CHROM']
    bed_df['START'] = vcf_df['POS']
    bed_df['END'] = info_df['END']
    bed_df['SVTYPE'] = info_df['SVTYPE']
    bed_df['SVLEN'] = info_df['SVLEN']
    bed_df['GT'] = sample_df['GT']
    bed_df['DP'] = sample_df['DP']
    bed_df['HMM'] = info_df['HMM']
    bed_df['ALN'] = info_df['ALN']
    bed_df['CLUSTER'] = info_df['CLUSTER']
    bed_df['CN'] = info_df['CN']
    bed_df['ALNOFFSET'] = info_df['ALNOFFSET']
    bed_df['id'] = vcf_df['id']

    # Save the BED file
    bed_df.to_csv(output_bed, sep='\t', header=False, index=False)
    logging.info('Created BED file: %s', output_bed)

def score(model, input_vcf, output_vcf, buildver='hg38', title='Probability Distribution', threshold=0.05, 
          threshold_del=None, threshold_dup=None, threshold_ins=None, threshold_inv=None, sample_coverage=None, large_cutoff=10000):
    """Score the structural variants using the binary classification model.

    Args:
        model (str): Path to the model file.
        input_vcf (str): Path to the input VCF file.
        output_vcf (str): Path to the output VCF file.
        threshold (float): Default threshold for SV types not specified.
        threshold_del (float): Optional. Threshold for DEL variants. If None, uses default threshold.
        threshold_dup (float): Optional. Threshold for DUP variants. If None, uses default threshold.
        threshold_ins (float): Optional. Threshold for INS variants. If None, uses default threshold.
        threshold_inv (float): Optional. Threshold for INV variants. If None, uses default threshold.
        sample_coverage (float): Required. Mean read depth coverage for the sample.
        large_cutoff (int): SV size cutoff in bp; variants larger than this are always kept (default: 50000).
    """
    # Build threshold dictionary with type-specific values
    threshold_by_type = {
        'DEL': threshold_del if threshold_del is not None else threshold,
        'DUP': threshold_dup if threshold_dup is not None else threshold,
        'INS': threshold_ins if threshold_ins is not None else threshold,
        'INV': threshold_inv if threshold_inv is not None else threshold,
    }
    
    prob_threshold = threshold
    logging.info('Using confidence threshold policy:')
    for svtype, thr in sorted(threshold_by_type.items()):
        logging.info('  %s: %.3f', svtype, thr)

    # Create a BED file from the input VCF file
    bed_file = os.path.splitext(input_vcf)[0] + '.bed'
    create_bed(input_vcf, bed_file)
    logging.info('Created BED file: %s', bed_file)

    # Load the model
    logging.info('Loading model from: %s', model)
    clf = joblib.load(model)
    logging.info('Model loaded successfully.')

    # Extract the features from the BED file
    annovar_path= '/mnt/isilon/wang_lab/perdomoj/softwares/annovar'
    annovar_db_path= '/mnt/isilon/wang_lab/perdomoj/annovar/humandb'
    anno_outdir= os.path.dirname(bed_file)
    anno_outdir= os.path.join(anno_outdir, 'annotations')
    if not os.path.exists(anno_outdir):
        os.makedirs(anno_outdir)
        logging.info('Created output directory: %s', anno_outdir)

    feature_df = extract_features(bed_file, annovar_path, annovar_db_path, anno_outdir, buildver, sample_coverage=sample_coverage)

    # Check if the feature extraction was successful
    if feature_df.empty:
        logging.error('Feature extraction failed. No features extracted.')
        sys.exit(1)

    # Separate the ID column and keep variant metadata for downstream evaluation joins.
    id_col = feature_df.pop('id')

    predictions_meta = pd.DataFrame({
        'id': id_col.values,
        'chrom': feature_df['chrom'].astype(str).values if 'chrom' in feature_df.columns else np.nan,
        'start': pd.to_numeric(feature_df['start'], errors='coerce').astype('Int64').values if 'start' in feature_df.columns else pd.Series([pd.NA] * len(id_col), dtype='Int64').values,
        'end': pd.to_numeric(feature_df['end'], errors='coerce').astype('Int64').values if 'end' in feature_df.columns else pd.Series([pd.NA] * len(id_col), dtype='Int64').values,
        'sv_type_str': feature_df['sv_type_str'].astype(str).values if 'sv_type_str' in feature_df.columns else np.nan,
        'sv_length': pd.to_numeric(feature_df['sv_length'], errors='coerce').astype('Int64').values if 'sv_length' in feature_df.columns else pd.Series([pd.NA] * len(id_col), dtype='Int64').values,
    })
    predictions_meta['sv_length_abs'] = predictions_meta['sv_length'].abs()
    
    # Remove other non-feature columns before prediction.
    # Keep normalized *_per_kb features; remove raw versions.
    for col in ['chrom', 'start', 'end', 'sv_type_str', 'cluster_size', 'dist_to_nearest_sv', 'read_depth']:
        if col in feature_df.columns:
            feature_df.pop(col)
    
    # Handle NaNs by filling with 0 (matching training's imputation fallback)
    logging.info('Handling NaN values in features...')
    nan_count_before = feature_df.isna().sum().sum()
    if nan_count_before > 0:
        logging.warning('Found %d NaN values in prediction features. Filling with 0.', nan_count_before)
        feature_df = feature_df.fillna(0)
    
    # Convert categorical/object columns to numeric (matching training preprocessing)
    logging.info('Converting categorical features to numeric...')
    for col in feature_df.columns:
        if feature_df[col].dtype == 'category':
            feature_df[col] = feature_df[col].cat.codes
        elif feature_df[col].dtype == 'object':
            feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
    
    # Ensure all columns are float64
    feature_df = feature_df.fillna(0).astype('float64')

    # Run the model on the features
    logging.info('Running the model on the features...')
    y_pred = clf.predict_proba(feature_df)

    output_dir = os.path.dirname(output_vcf)

    # Save per-variant probabilities for downstream threshold tuning.
    predictions_tsv = os.path.join(output_dir, 'predictions.tsv')
    predictions_df = predictions_meta.copy()
    predictions_df['confidence_score'] = y_pred[:, 1]
    predictions_df.to_csv(predictions_tsv, sep='\t', index=False)
    logging.info('Saved per-variant predictions to %s', predictions_tsv)

    # Plot a histogram of the probabilities using seaborn since it looks better
    fig, ax = plt.subplots()
    sns.histplot(y_pred[:, 1], bins=20, ax=ax)
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    ax.set_title(title)

    # Save the plot to the output directory
    plt.savefig(os.path.join(output_dir, 'probabilities_seaborn.png'))
    logging.info('Saved the plot of the probabilities to %s', os.path.join(output_dir, 'probabilities_seaborn.png'))
    
    # Build a lookup dictionary: variant_id → (confidence_score, sv_type) for type-specific filtering
    variant_lookup = {}
    for idx, row in predictions_df.iterrows():
        variant_lookup[row['id']] = (row['confidence_score'], row['sv_type_str'])
    
    logging.info('Built variant lookup with %d entries for type-specific filtering', len(variant_lookup))
    
    # For backward compatibility, also track variants below the default threshold
    filtered_indices = np.where(y_pred[:, 1] < prob_threshold)[0]
    logging.info('Number of variants under the default probability threshold %.2f: %d', prob_threshold, len(filtered_indices))

    # Get the IDs of the filtered variants (for logging/debugging)
    filtered_ids = id_col.iloc[filtered_indices].values
    filtered_ids_file = os.path.join(output_dir, 'filtered_ids.txt')
    np.savetxt(filtered_ids_file, filtered_ids, fmt='%s')
    logging.info('Saved the filtered IDs (using default threshold) to %s', filtered_ids_file)

    # Create a VCF file with only the filtered variants
    removed_svs_vcf = os.path.join(output_dir, 'removed_svs.vcf')

    # Filter the input VCF file based on type-specific thresholds and SV length
    # Keep all SVs >50kb regardless of confidence score; apply type-specific threshold to SVs <=50kb
    logging.info('Filtering the input VCF file using type-specific thresholds and SV length...')
    logging.info('Policy: Keep all SVs >50kb; apply type-specific thresholds to SVs <=50kb')
    
    current_record = 0
    pass_count = 0
    filter_count = 0
    total_records = 0
    type_filter_stats = {}  # Track filtering statistics by type
    
    with open(input_vcf, 'r') as vcf_in, open(output_vcf, 'w') as vcf_out, open(removed_svs_vcf, 'w') as removed_out:
        for line in vcf_in:
            if line.startswith('#'):
                # Write the header lines as they are
                vcf_out.write(line)
                removed_out.write(line)
            else:
                # Extract SVLEN and SVTYPE from the VCF INFO field
                info_field = line.split('\t')[7]
                svlen_match = None
                svtype_match = None
                
                for field in info_field.split(';'):
                    if field.startswith('SVLEN='):
                        try:
                            svlen_match = int(field.split('=')[1])
                        except (ValueError, IndexError):
                            svlen_match = None
                    elif field.startswith('SVTYPE='):
                        try:
                            svtype_match = field.split('=')[1]
                        except IndexError:
                            svtype_match = None
                
                # Get confidence score and sv_type from predictions lookup
                if current_record in variant_lookup:
                    confidence_score, predicted_svtype = variant_lookup[current_record]
                    # Use VCF SVTYPE if available, otherwise use predicted svtype
                    svtype = svtype_match if svtype_match else predicted_svtype
                else:
                    # Variant not in predictions (shouldn't happen, but handle gracefully)
                    logging.warning(f'Variant {current_record} not found in predictions lookup, using default threshold')
                    confidence_score = 0.0
                    svtype = svtype_match if svtype_match else 'UNKNOWN'
                
                # Get the appropriate threshold for this SV type
                type_threshold = threshold_by_type.get(svtype, prob_threshold)
                
                # Determine if variant should be kept
                is_large_sv = svlen_match is not None and abs(svlen_match) > large_cutoff
                passes_threshold = confidence_score >= type_threshold
                
                # Keep if: (large SV) OR (passes type-specific threshold)
                should_keep = is_large_sv or passes_threshold
                
                # Track statistics by type
                if svtype not in type_filter_stats:
                    type_filter_stats[svtype] = {'total': 0, 'kept': 0, 'filtered': 0}
                type_filter_stats[svtype]['total'] += 1
                
                if should_keep:
                    vcf_out.write(line)
                    pass_count += 1
                    type_filter_stats[svtype]['kept'] += 1
                else:
                    # Write the line to the removed_svs.vcf file if filtered
                    removed_out.write(line)
                    filter_count += 1
                    type_filter_stats[svtype]['filtered'] += 1

                total_records += 1
                current_record += 1

    logging.info('Filtered the input VCF file and saved it to %s', output_vcf)
    logging.info('Scoring process completed successfully. Passed %d out of %d records.', pass_count, total_records)
    logging.info('Removed %d records (low confidence and <=50kb). See %s for details.', filter_count, removed_svs_vcf)
    
    # Log filtering statistics by SV type
    logging.info('Filtering statistics by SV type:')
    for svtype in sorted(type_filter_stats.keys()):
        stats = type_filter_stats[svtype]
        kept_pct = 100.0 * stats['kept'] / stats['total'] if stats['total'] > 0 else 0
        logging.info('  %s: kept %d/%d (%.1f%%)', svtype, stats['kept'], stats['total'], kept_pct)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input VCF file.')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to the output VCF file.')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the model file.')
    parser.add_argument('--buildver', type=str, default='hg38',
                        help='Genome build version (default: hg38).')
    parser.add_argument('--title', type=str, default='Probability Distribution',
                        help='Title for the probability distribution plot (default: Probability Distribution).')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Default threshold for filtering predictions (default: 0.05). Used for SV types without specific thresholds.')
    parser.add_argument('--threshold-del', type=float, default=None,
                        help='Threshold for DEL variants (default: uses --threshold value).')
    parser.add_argument('--threshold-dup', type=float, default=None,
                        help='Threshold for DUP variants (default: uses --threshold value).')
    parser.add_argument('--threshold-ins', type=float, default=None,
                        help='Threshold for INS variants (default: uses --threshold value).')
    parser.add_argument('--threshold-inv', type=float, default=None,
                        help='Threshold for INV variants (default: uses --threshold value).')
    parser.add_argument('--sample_coverage', type=float, required=True,
                        help='Mean read depth coverage for the sample (required, used to normalize read_depth).')
    parser.add_argument('--large-cutoff', type=int, default=10000,
                        help='SV size cutoff in bp; variants larger than this are always kept (default: 50000).')

    args = parser.parse_args()
    input_vcf = args.input
    output_vcf = args.output
    model = args.model

    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Starting the scoring process...')
    logging.info('Input VCF file: %s', input_vcf)
    logging.info('Output VCF file: %s', output_vcf)
    logging.info('Model file: %s', model)

    # Check if the input VCF file exists
    if not os.path.isfile(input_vcf):
        logging.error('Input VCF file does not exist: %s', input_vcf)
        sys.exit(1)

    # Check if the model file exists
    if not os.path.isfile(model):
        logging.error('Model file does not exist: %s', model)
        sys.exit(1)

    # Check if the output directory exists, if not create it
    output_dir = os.path.dirname(output_vcf)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info('Created output directory: %s', output_dir)

    # Check if the input VCF file is a valid VCF file
    if not input_vcf.endswith('.vcf') and not input_vcf.endswith('.vcf.gz'):
        logging.error('Input file is not a valid VCF file: %s', input_vcf)
        sys.exit(1)
    if not output_vcf.endswith('.vcf'):
        logging.error('Output file must have a .vcf extension: %s', output_vcf)
        sys.exit(1)
    if not model.endswith('.pkl'):
        logging.error('Model file must have a .pkl extension: %s', model)
        sys.exit(1)

    # Check the reference genome build version
    buildver = args.buildver
    if buildver not in ['hg19', 'hg38']:
        logging.error('Unsupported genome build version: %s. Supported versions are hg19 and hg38.', buildver)
        sys.exit(1)

    # Run the scoring function
    score(model, input_vcf, output_vcf, buildver=buildver, title=args.title, 
          threshold=args.threshold, sample_coverage=args.sample_coverage,
          threshold_del=args.threshold_del, threshold_dup=args.threshold_dup,
          threshold_ins=args.threshold_ins, threshold_inv=args.threshold_inv,
          large_cutoff=args.large_cutoff)
    logging.info('Scoring process completed.')
