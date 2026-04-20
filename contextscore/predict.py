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
import argparse
import importlib
import gzip
import re
import numpy as np
import joblib
import pandas as pd

try:
    from .extract_features import extract_features
except ImportError:
    from extract_features import extract_features


USER_PREFIX = "[ContextScore]"
DEFAULT_MODEL_ENV_VAR = 'CONTEXTSCORE_MODEL_PATH'
DEFAULT_MODEL_INSTALL_PATH = os.path.join(
    sys.prefix,
    'share',
    'contextscore',
    'models',
    'contextscore_model.pkl',
)


def user_message(message):
    """Emit concise, user-facing progress messages."""
    print(f"{USER_PREFIX} {message}")


def configure_logging(verbose=False, debug=False):
    """Configure logging output level based on user-selected mode."""
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')


def resolve_annovar_paths(annovar_path, annovar_db_path):
    """Resolve ANNOVAR paths from CLI flags or environment variables."""
    resolved_path = annovar_path or os.getenv('ANNOVAR_PATH')
    resolved_db = annovar_db_path or os.getenv('ANNOVAR_DB_PATH')
    return resolved_path, resolved_db


def resolve_model_path(model_path):
    """Resolve model path from CLI, env var, or default installed location."""
    if model_path:
        return model_path, 'cli'

    env_model_path = os.getenv(DEFAULT_MODEL_ENV_VAR)
    if env_model_path:
        return env_model_path, 'env'

    return DEFAULT_MODEL_INSTALL_PATH, 'default'


def validate_annovar_paths(annovar_path, annovar_db_path):
    """Validate ANNOVAR installation paths before running feature extraction."""
    if not annovar_path:
        raise ValueError(
            'ANNOVAR path is required. Set --annovar or environment variable ANNOVAR_PATH.'
        )
    if not annovar_db_path:
        raise ValueError(
            'ANNOVAR database path is required. Set --annovar-db or environment variable ANNOVAR_DB_PATH.'
        )

    annotate_variation = os.path.join(annovar_path, 'annotate_variation.pl')
    table_annovar = os.path.join(annovar_path, 'table_annovar.pl')
    if not os.path.isfile(annotate_variation) or not os.path.isfile(table_annovar):
        raise ValueError(
            f'Invalid ANNOVAR path: {annovar_path}. Expected annotate_variation.pl and table_annovar.pl in this directory.'
        )
    if not os.path.isdir(annovar_db_path):
        raise ValueError(f'ANNOVAR database directory does not exist: {annovar_db_path}')


def try_import_plotting_libs():
    """Attempt to import plotting libraries without failing prediction flow."""
    try:
        plt = importlib.import_module('matplotlib.pyplot')
        sns = importlib.import_module('seaborn')
        return plt, sns
    except ImportError:
        return None, None


def open_vcf_text(path):
    """Open VCF text input, supporting both plain and gzipped files."""
    if str(path).endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8')
    return open(path, 'r', encoding='utf-8')


def canonicalize_chromosome(chrom_value):
    """Map CHROM values to canonical chr-prefixed labels; return None if unparseable."""
    if pd.isna(chrom_value):
        return None

    chrom_str = str(chrom_value).strip()
    if not chrom_str:
        return None

    has_chr_prefix = chrom_str.lower().startswith('chr')
    token = chrom_str[3:] if has_chr_prefix else chrom_str
    token_upper = token.upper()

    if token_upper in {'M', 'MT'}:
        return 'chrM'
    if token_upper in {'X', 'Y'}:
        return f'chr{token_upper}'
    if token_upper.isdigit():
        token_num = int(token_upper)
        if 1 <= token_num <= 22:
            return f'chr{token_num}'

    # Keep non-canonical contigs as-is (e.g., GL*, KI*, NC_*).
    if re.fullmatch(r'[A-Za-z0-9_.-]+', chrom_str):
        return chrom_str
    return None

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
    
    # Normalize CHROM labels for robust annotation intersects.
    vcf_df['CHROM_ORIG'] = vcf_df['CHROM'].astype(str)
    vcf_df['CHROM'] = vcf_df['CHROM'].apply(canonicalize_chromosome)
    invalid_chrom_mask = vcf_df['CHROM'].isna()
    skipped_chrom_ids = set(vcf_df.loc[invalid_chrom_mask, 'id'].astype(int).tolist())
    if skipped_chrom_ids:
        examples = vcf_df.loc[invalid_chrom_mask, 'CHROM_ORIG'].dropna().astype(str).unique()[:5]
        logging.warning(
            'Skipping %d variants with unparseable CHROM labels during annotation/scoring. Examples: %s',
            len(skipped_chrom_ids),
            ', '.join(examples) if len(examples) > 0 else 'N/A',
        )
        vcf_df = vcf_df.loc[~invalid_chrom_mask].copy()

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
    return skipped_chrom_ids

def score(model, input_vcf, output_vcf, buildver='hg38', threshold=0.05,
          threshold_del=None, threshold_dup=None, threshold_ins=None, threshold_inv=None,
          sample_coverage=None, large_cutoff=10000, annovar_path=None, annovar_db_path=None,
          debug_plot=False):
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
    skipped_chrom_ids = create_bed(input_vcf, bed_file)
    logging.info('Created BED file: %s', bed_file)
    if skipped_chrom_ids:
        logging.info('Variants skipped from annotation/scoring due to unparseable CHROM: %d', len(skipped_chrom_ids))

    # Load the model
    logging.info('Loading model from: %s', model)
    clf = joblib.load(model)
    logging.info('Model loaded successfully.')

    # Extract the features from the BED file.
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
        logging.info('Found %d NaN values in prediction features. Filling with 0.', nan_count_before)
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

    output_dir = os.path.dirname(os.path.abspath(output_vcf)) or '.'

    # Save per-variant probabilities for downstream threshold tuning.
    predictions_tsv = os.path.join(output_dir, 'predictions.tsv')
    predictions_df = predictions_meta.copy()
    predictions_df['confidence_score'] = y_pred[:, 1]
    predictions_df.to_csv(predictions_tsv, sep='\t', index=False)
    logging.info('Saved per-variant predictions to %s', predictions_tsv)

    if debug_plot:
        plt, sns = try_import_plotting_libs()
        if plt is None or sns is None:
            logging.warning('Debug plotting requested but matplotlib/seaborn are not installed. Skipping plot generation.')
        else:
            _, ax = plt.subplots()
            sns.histplot(y_pred[:, 1], bins=20, ax=ax)
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Count')
            ax.set_title('Probability Distribution')
            plot_path = os.path.join(output_dir, 'prob_dist.svg')
            plt.savefig(plot_path)
            plt.close()
            logging.info('Saved debug probability plot to %s', plot_path)

    # Build a lookup dictionary: variant_id → (confidence_score, sv_type) for type-specific filtering
    variant_lookup = {}
    for _, row in predictions_df.iterrows():
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
    
    with open_vcf_text(input_vcf) as vcf_in, open(output_vcf, 'w', encoding='utf-8') as vcf_out, open(removed_svs_vcf, 'w', encoding='utf-8') as removed_out:
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

                if current_record in skipped_chrom_ids:
                    svtype_for_stats = svtype_match if svtype_match else 'UNKNOWN'
                    if svtype_for_stats not in type_filter_stats:
                        type_filter_stats[svtype_for_stats] = {'total': 0, 'kept': 0, 'filtered': 0}
                    type_filter_stats[svtype_for_stats]['total'] += 1
                    type_filter_stats[svtype_for_stats]['kept'] += 1
                    vcf_out.write(line)
                    pass_count += 1
                    total_records += 1
                    current_record += 1
                    continue
                
                # Get confidence score and sv_type from predictions lookup
                if current_record in variant_lookup:
                    confidence_score, predicted_svtype = variant_lookup[current_record]
                    # Use VCF SVTYPE if available, otherwise use predicted svtype
                    svtype = svtype_match if svtype_match else predicted_svtype
                else:
                    # Variant not in predictions (shouldn't happen, but handle gracefully)
                    logging.warning('Variant %d not found in predictions lookup, using default threshold', current_record)
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

    return {
        'total_records': total_records,
        'passed_records': pass_count,
        'filtered_records': filter_count,
        'output_vcf': output_vcf,
        'removed_vcf': removed_svs_vcf,
        'predictions_tsv': predictions_tsv,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input VCF file.')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to the output VCF file.')
    parser.add_argument('--model', type=str, required=False, default=None,
                        help='Path to the model file. Optional if CONTEXTSCORE_MODEL_PATH is set or default packaged model is installed.')
    parser.add_argument('--buildver', type=str, default='hg38',
                        help='Genome build version (default: hg38).')
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='Default threshold for filtering predictions (default: 0.2). Used for SV types without specific thresholds.')
    parser.add_argument('--threshold-del', type=float, default=None,
                        help='Threshold for DEL variants (default: uses --threshold value).')
    parser.add_argument('--threshold-dup', type=float, default=None,
                        help='Threshold for DUP variants (default: uses --threshold value).')
    parser.add_argument('--threshold-ins', type=float, default=None,
                        help='Threshold for INS variants (default: uses --threshold value).')
    parser.add_argument('--threshold-inv', type=float, default=None,
                        help='Threshold for INV variants (default: uses --threshold value).')
    parser.add_argument('--sample-coverage', type=float, required=True,
                        help='Mean read depth coverage for the sample (required, used to normalize read_depth).')
    parser.add_argument('--large-cutoff', type=int, default=10000,
                        help='SV size cutoff in bp; variants larger than this are always kept (default: 50000).')
    parser.add_argument('--annovar', type=str, default=None,
                        help='Path to ANNOVAR installation directory. Can also be set via ANNOVAR_PATH.')
    parser.add_argument('--annovar-db', type=str, default=None,
                        help='Path to ANNOVAR database directory. Can also be set via ANNOVAR_DB_PATH.')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed progress logs.')
    parser.add_argument('--debug', action='store_true',
                        help='Show debug logs including subprocess details.')
    parser.add_argument('--debug-plot', action='store_true',
                        help='Generate probability distribution plot for debugging (optional, requires matplotlib and seaborn).')

    args = parser.parse_args(argv)
    input_vcf = args.input
    output_vcf = args.output
    model, model_source = resolve_model_path(args.model)

    configure_logging(verbose=args.verbose, debug=args.debug)
    user_message('Starting prediction run')

    # Check if the input VCF file exists
    if not os.path.isfile(input_vcf):
        logging.error('Input VCF file does not exist: %s', input_vcf)
        sys.exit(1)

    # Check if the model file exists
    if not os.path.isfile(model):
        logging.error('Model file does not exist: %s', model)
        user_message('Model path could not be resolved to an existing file.')
        user_message('Provide --model /path/to/model.pkl, or set CONTEXTSCORE_MODEL_PATH, or install the contextscore-models package.')
        if model_source == 'default':
            user_message(f'Default expected path: {DEFAULT_MODEL_INSTALL_PATH}')
        sys.exit(1)

    # Check if the output directory exists, if not create it
    output_dir = os.path.dirname(os.path.abspath(output_vcf)) or '.'
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

    logging.info('Using model path from %s: %s', model_source, model)

    # Check the reference genome build version
    buildver = args.buildver
    if buildver not in ['hg19', 'hg38']:
        logging.error('Unsupported genome build version: %s. Supported versions are hg19 and hg38.', buildver)
        sys.exit(1)

    annovar_path, annovar_db_path = resolve_annovar_paths(args.annovar, args.annovar_db)
    try:
        validate_annovar_paths(annovar_path, annovar_db_path)
    except ValueError as exc:
        logging.error('%s', exc)
        user_message('ANNOVAR setup is required before running prediction.')
        user_message('Example: contextscore --input sample.vcf --output out.vcf --sample_coverage 30 --annovar /path/to/annovar --annovar-db /path/to/humandb')
        user_message('Optional: add --model /path/to/model.pkl to override default model resolution.')
        user_message('You can also set ANNOVAR_PATH and ANNOVAR_DB_PATH environment variables.')
        sys.exit(2)

    user_message('Running feature extraction and scoring')

    # Run the scoring function
    summary = score(model, input_vcf, output_vcf, buildver=buildver,
                    threshold=args.threshold, sample_coverage=args.sample_coverage,
                    threshold_del=args.threshold_del, threshold_dup=args.threshold_dup,
                    threshold_ins=args.threshold_ins, threshold_inv=args.threshold_inv,
                    large_cutoff=args.large_cutoff, annovar_path=annovar_path,
                    annovar_db_path=annovar_db_path,
                    debug_plot=args.debug_plot)

    user_message(
        f"Completed. Kept {summary['passed_records']}/{summary['total_records']} variants; filtered {summary['filtered_records']}."
    )
    user_message(f"Output VCF: {summary['output_vcf']}")
    logging.info('Scoring process completed.')


if __name__ == '__main__':
    main()
