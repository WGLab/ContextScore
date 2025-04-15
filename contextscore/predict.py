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
    # Read the VCF file
    # vcf_df = pd.read_csv(input_vcf, sep='\t', comment='#', header=None, 
    #                      names=['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER',
    #                             'INFO', 'FORMAT', 'SAMPLE'])
    logging.info('Reading VCF file: %s', input_vcf)
    vcf_df = pd.read_csv(input_vcf, sep='\t', comment='#', header=None, 
                         names=['CHROM', 'POS', 'INFO', 'FORMAT', 'SAMPLE'], usecols=[0, 1, 7, 8, 9],
                            dtype={'CHROM': str, 'POS': int, 'INFO': str, 'FORMAT': str, 'SAMPLE': str})
    
    logging.info('VCF file read successfully. Number of records: %d', len(vcf_df))
    logging.info('First few records:\n%s', vcf_df.head())
    
    # Extract the relevant fields from the INFO column
    # info_df = vcf_df['INFO'].str.split(';', expand=True)

    # Print the ALN column
    # info_df['ALN'] = vcf_df['INFO'].str.extract(r'ALN=([^;]+)')
    # logging.info('ALN col = \n%s', info_df['ALN'].head())

    # info_df.columns = ['END', 'SVTYPE', 'SVLEN', 'HMM', 'ALN', 'CLUSTER',
    # 'CN', 'ALNOFFSET']
    info_df = pd.DataFrame()
    info_df['ALN'] = vcf_df['INFO'].str.extract(r'ALN=([^;]+)')
    info_df['END'] = vcf_df['INFO'].str.extract(r'END=(\d+)')
    info_df['SVTYPE'] = vcf_df['INFO'].str.extract(r'SVTYPE=([^;]+)')
    info_df['SVLEN'] = vcf_df['INFO'].str.extract(r'SVLEN=([^;]+)')
    info_df['HMM'] = vcf_df['INFO'].str.extract(r'HMM=([^;]+)')
    info_df['CLUSTER'] = vcf_df['INFO'].str.extract(r'CLUSTER=([^;]+)')
    info_df['CN'] = vcf_df['INFO'].str.extract(r'CN=([^;]+)')
    info_df['ALNOFFSET'] = vcf_df['INFO'].str.extract(r'ALNOFFSET=([^;]+)')

    # info_df['END'] = info_df['END'].str.replace('END=', '').astype(int)
    # info_df['SVTYPE'] = info_df['SVTYPE'].str.replace('SVTYPE=', '')
    # info_df['SVLEN'] = info_df['SVLEN'].str.replace('SVLEN=', '').astype(int)
    # info_df['HMM'] = info_df['HMM'].str.replace('HMM=', '')
    # info_df['ALN'] = info_df['ALN'].str.replace('ALN=', '')
    # info_df['CLUSTER'] = info_df['CLUSTER'].str.replace('CLUSTER=', '')
    # info_df['CN'] = info_df['CN'].str.replace('CN=', '')
    # info_df['ALNOFFSET'] = info_df['ALNOFFSET'].str.replace('ALNOFFSET=', '')

    # Extract the genotype (GT) and read depth (DP) from the SAMPLE column
    sample_df = pd.DataFrame()
    sample_df['GT'] = vcf_df['SAMPLE'].str.extract(r'([^:]+):')
    sample_df['DP'] = vcf_df['SAMPLE'].str.extract(r':(\d+)').astype(int)

    logging.info('Sample GT and DP columns:\n%s', sample_df.head())
    # sample_df = vcf_df['SAMPLE'].str.split(':', expand=True)
    # sample_df.columns = ['GT', 'DP']
    # sample_df['GT'] = sample_df['GT'].str.replace('GT=', '')
    # sample_df['DP'] = sample_df['DP'].str.replace('DP=', '').astype(int)

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

    # Print the first few rows of the BED file
    logging.info('First few rows of the BED file:\n%s', bed_df.head())

    # Save the BED file
    bed_df.to_csv(output_bed, sep='\t', header=False, index=False)
    logging.info('Created BED file: %s', output_bed)

def score(model, input_vcf, output_vcf):
    """Score the structural variants using the binary classification model.

    Args:
        model (str): Path to the model file.
        input_vcf (str): Path to the input VCF file.
        output_vcf (str): Path to the output VCF file.
    """

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

    feature_df = extract_features(bed_file, annovar_path, annovar_db_path, anno_outdir)
    logging.info('Extracted features from the BED file:\n%s', feature_df.head())

    # Check if the feature extraction was successful
    if feature_df.empty:
        logging.error('Feature extraction failed. No features extracted.')
        sys.exit(1)

    # Run the model on the features
    logging.info('Running the model on the features...')
    y_pred = clf.predict_proba(feature_df)

    # Plot a histogram of the probabilities
    plt.hist(y_pred[:, 1], bins=20)
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.title('Probability Distribution')
    output_dir = os.path.dirname(output_vcf)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info('Created output directory: %s', output_dir)
    # Save the plot to the output directory
    plt.savefig(os.path.join(output_dir, 'probabilities.png'))
    logging.info('Saved the plot of the probabilities to %s.', os.path.join(output_dir, 'probabilities.png'))

    # Save the predictions to the output VCF file
    # vcf_df = pd.read_csv(input_vcf, sep='\t', comment='#', header=None,

    return

    # Extract the features from the VCF file
    X = extract_features(input_vcf)

    # Predict the labels and get the probabilities
    y_pred = clf.predict_proba(X)

    # logging.info('Predicted labels:\n%s', y_pred)

    # Plot a histogram of the probabilities
    plt.hist(y_pred[:, 1], bins=20)
    plt.xlabel('Probability')
    plt.ylabel('Count')

    # # Save the plot to the input VCF file's directory
    # output_dir = os.path.dirname(output_vcf)
    # output_filepath = os.path.join(output_dir, 'probabilities.png')
    # plt.savefig(output_filepath)
    # logging.info('Saved the plot of the probabilities to %s.', output_filepath)

    # Save the plot to the working directory
    plt.savefig('output/probabilities.png')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input VCF file.')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to the output VCF file.')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the model file.')
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

    # Run the scoring function
    score(model, input_vcf, output_vcf)
    logging.info('Scoring process completed.')
