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

def score(model, input_vcf, output_vcf, scaler_path=None, buildver='hg38', title='Probability Distribution', threshold=0.05):
    """Score the structural variants using the binary classification model.

    Args:
        model (str): Path to the model file.
        input_vcf (str): Path to the input VCF file.
        output_vcf (str): Path to the output VCF file.
        scaler_path (str): Path to the scaler pkl file (optional).
    """
    prob_threshold = threshold
    logging.info('Using probability threshold: %.3f', prob_threshold)

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

    feature_df = extract_features(bed_file, annovar_path, annovar_db_path, anno_outdir, buildver)

    # Perform robust scaling on the read_depth and cluster_size columns
    logging.info('Performing robust scaling on the read_depth and cluster_size columns...')
    if scaler_path is not None and os.path.isfile(scaler_path):
        # Load the pre-fitted scaler from training
        logging.info('Loading scaler from: %s', scaler_path)
        scaler = joblib.load(scaler_path)
        feature_df[['read_depth', 'cluster_size']] = scaler.transform(feature_df[['read_depth', 'cluster_size']])
        logging.info('Applied pre-fitted scaler (trained on TP distribution).')
    else:
        # Exit with error, since the scaler is required for proper scaling of the features
        logging.error('Scaler file is required for proper scaling of the features. Please provide a valid scaler file path using the --scaler argument.')
        sys.exit(1)

    # Check if the feature extraction was successful
    if feature_df.empty:
        logging.error('Feature extraction failed. No features extracted.')
        sys.exit(1)

    # Separate the ID, chrom, start, end, SV length, read depth, and cluster size columns from the features
    id_col = feature_df.pop('id')
    
    # Remove other non-feature columns before prediction
    for col in ['chrom', 'start', 'end', 'sv_type_str']:
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

    # Plot a histogram of the probabilities using seaborn since it looks better
    output_dir = os.path.dirname(output_vcf)
    fig, ax = plt.subplots()
    sns.histplot(y_pred[:, 1], bins=20, ax=ax)
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    ax.set_title(title)

    # Save the plot to the output directory
    plt.savefig(os.path.join(output_dir, 'probabilities_seaborn.png'))
    logging.info('Saved the plot of the probabilities to %s', os.path.join(output_dir, 'probabilities_seaborn.png'))
    filtered_indices = np.where(y_pred[:, 1] < prob_threshold)[0]
    logging.info('Number of variants under the probability threshold %.2f: %d', prob_threshold, len(filtered_indices))

    # Get the IDs of the filtered variants
    filtered_ids = id_col.iloc[filtered_indices].values
    filtered_ids_file = os.path.join(output_dir, 'filtered_ids.txt')
    np.savetxt(filtered_ids_file, filtered_ids, fmt='%s')
    logging.info('Saved the filtered IDs to %s', filtered_ids_file)

    # Create a VCF file with only the filtered variants
    removed_svs_vcf = os.path.join(output_dir, 'removed_svs.vcf')

    # Filter the input VCF file based on the filtered indices
    logging.info('Filtering the input VCF file based on the filtered indices...')
    filtered_records = set(filtered_ids)
    current_record = 0
    pass_count = 0
    filter_count = 0
    total_records = 0
    with open(input_vcf, 'r') as vcf_in, open(output_vcf, 'w') as vcf_out, open(removed_svs_vcf, 'w') as removed_out:
        for line in vcf_in:
            if line.startswith('#'):
                # Write the header lines as they are
                vcf_out.write(line)
                removed_out.write(line)
            else:
                if current_record in filtered_records:
                    # Write the line to the removed_svs.vcf file if the current record is in the filtered records
                    removed_out.write(line)
                    filter_count += 1
                else:
                    # Write the line if the current record is not in the filtered records
                    vcf_out.write(line)
                    pass_count += 1

                total_records += 1
                current_record += 1

    logging.info('Filtered the input VCF file and saved it to %s', output_vcf)
    logging.info('Scoring process completed successfully. Passed %d out of %d records.', pass_count, total_records)
    logging.info('Removed %d records. See %s for details.', filter_count, removed_svs_vcf)


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
                        help='Threshold for filtering predictions (default: 0.05).')
    parser.add_argument('--scaler', type=str, default=None,
                        help='Path to the scaler pkl file (trained on TP data, optional).')

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
    logging.info('Scaler file: %s', args.scaler)

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
    score(model, input_vcf, output_vcf, scaler_path=args.scaler, buildver=buildver, title=args.title, threshold=args.threshold)
    logging.info('Scoring process completed.')
