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

from extract_features import extract_features, add_interaction_terms

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
    #                      names=['CHROM', 'POS', 'id', 'REF', 'ALT', 'QUAL', 'FILTER',
    #                             'INFO', 'FORMAT', 'SAMPLE'])
    logging.info('Reading VCF file: %s', input_vcf)
    vcf_df = pd.read_csv(input_vcf, sep='\t', comment='#', header=None, 
                         names=['CHROM', 'POS', 'INFO', 'FORMAT', 'SAMPLE'], usecols=[0, 1, 7, 8, 9],
                            dtype={'CHROM': str, 'POS': int, 'INFO': str, 'FORMAT': str, 'SAMPLE': str})
    
    # Add a column for the ID field with the VCF row number
    vcf_df['id'] = vcf_df.index

    # Print the first 10 IDs
    logging.info('First 10 IDs in the VCF file:\n%s', vcf_df['id'].head(10))
    
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
    bed_df['id'] = vcf_df['id']

    # Print the first few rows of the BED file
    logging.info('First few rows of the BED file:\n%s', bed_df.head())

    # Save the BED file
    bed_df.to_csv(output_bed, sep='\t', header=False, index=False)
    logging.info('Created BED file: %s', output_bed)

def score(model, input_vcf, output_vcf, buildver='hg38', title='Probability Distribution'):
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

    feature_df = extract_features(bed_file, annovar_path, annovar_db_path, anno_outdir, buildver)
    logging.info('Extracted features from the BED file:\n%s', feature_df.head())

    # Perform robust scaling on the read_depth and cluster_size columns
    logging.info('Performing robust scaling on the read_depth and cluster_size columns...')
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    feature_df[['read_depth', 'cluster_size']] = scaler.fit_transform(feature_df[['read_depth', 'cluster_size']])
    logging.info('Robust scaling completed.')

    # Add interaction terms to the features
    feature_df = add_interaction_terms(feature_df)
    logging.info('Added interaction terms to the features.')

    # Drop the sv_type column (imbalance especially for inversions).
    # logging.info('Dropping the sv_type column from the features.')
    # feature_df.drop(columns=['sv_type'], inplace=True)

    # # Drop the SV length column
    # feature_df.drop(columns=['sv_length'], inplace=True)

    # Drop the read_depth and cluster_size columns
    # feature_df.drop(columns=['read_depth', 'cluster_size'], inplace=True)

    # Drop the HMM log likelihood column
    # feature_df.drop(columns=['hmm_llh'], inplace=True)

    # Check if the feature extraction was successful
    if feature_df.empty:
        logging.error('Feature extraction failed. No features extracted.')
        sys.exit(1)

    # # Separate the ID column from the features
    # id_col = feature_df.pop('id')
    # logging.info('Separated ID column from the features.')

    # # Separate the chrom column from the features
    # chrom_col = feature_df.pop('chrom')
    # logging.info('Separated chrom column from the features.')

    # Separate the ID, chrom, start, end, SV length, read depth, and cluster size columns from the features
    id_col = feature_df.pop('id')
    chrom_col = feature_df.pop('chrom')
    start_col = feature_df.pop('start')
    end_col = feature_df.pop('end')
    # sv_length_col = feature_df.pop('sv_length')
    read_depth_col = feature_df['read_depth']
    cluster_size_col = feature_df['cluster_size']
    sv_type_str_col = feature_df.pop('sv_type_str')

    # # Normalize the cluster_size and read_depth columns using RobustScaler
    # logging.info('Normalizing the cluster_size and read_depth columns...')
    # from sklearn.preprocessing import RobustScaler, MinMaxScaler
    # scaler = RobustScaler()
    # robust_scaled = scaler.fit_transform(feature_df[['cluster_size', 'read_depth']])
    # feature_df[['cluster_size', 'read_depth']] = robust_scaled

    logging.info('Feature DataFrame:\n%s', feature_df.head())

    # Run the model on the features
    logging.info('Running the model on the features...')
    y_pred = clf.predict_proba(feature_df)

    # Print the first 10 predictions
    logging.info('First 10 predictions:\n%s', y_pred[:10])

    # Plot a histogram of the probabilities
    # output_dir = os.path.dirname(output_vcf)
    # plt.hist(y_pred[:, 1], bins=20)
    # plt.xlabel('Confidence Score')
    # plt.ylabel('Count')
    # plt.title('Probability Distribution')
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    #     logging.info('Created output directory: %s', output_dir)
    # # Save the plot to the output directory
    # plt.savefig(os.path.join(output_dir, 'probabilities.png'))
    # logging.info('Saved the plot of the probabilities to %s.', os.path.join(output_dir, 'probabilities.png'))

    # Plot a histogram of the probabilities using seaborn since it looks better
    output_dir = os.path.dirname(output_vcf)
    fig, ax = plt.subplots()
    sns.histplot(y_pred[:, 1], bins=20, ax=ax)
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    # ax.set_title('Probability Distribution')
    ax.set_title(title)

    # Save the plot to the output directory
    plt.savefig(os.path.join(output_dir, 'probabilities_seaborn.png'))
    logging.info('Saved the plot of the probabilities to %s', os.path.join(output_dir, 'probabilities_seaborn.png'))

    # Determine the threshold for filtering
    # 22 May 2025
    # prob_threshold = 0.001  # Does not affect results much
    # prob_threshold = 0.2  # Lowered recall (too high)
    # prob_threshold = 0.1  # Lowered recall (too high)
    # prob_threshold = 0.01  # Slightly lowered recall (too high)
    # prob_threshold = 0.005  # Even slightlier lowered recall (too high)
    # prob_threshold = 0.001

    # 10 July 2025 - Feature updates for large SVs
    # prob_threshold = 0.01  # Low precision, high recall (same recall as no filtering)
    # prob_threshold = 0.05 # Same result, improved precision
    # prob_threshold = 0.1  # Same result, improved precision
    #prob_threshold = 0.2  # Same result, improved precision
    # prob_threshold = 0.25
    # prob_threshold = 0.3  # Slightly lowered recall, improved precision
    # prob_threshold = 0.4  # Lowered recall, inversions are not highest recall anymore, but achieved overal highest F1 score
    # prob_threshold = 0.35  # Lowered recall, inversions are not highest recall
    # anymore, F1 is equal to Sniffles2
    # prob_threshold = 0.32

    # 11 July 2025 - Feature updates for large SVs
    # prob_threshold = 0.1  # Lowered recall
    # prob_threshold = 0.05

    # Engineering feature interaction terms
    # prob_threshold = 0.01
    # prob_threshold = 0.1
    # prob_threshold = 0.3  # Lowered recall
    # prob_threshold = 0.2
    # prob_threshold = 0.02  # Too many SVs
    # prob_threshold = 0.01

    # 13 July 2025 - Feature updates
    prob_threshold = 0.01  # Too high SV count for plat. ped.
    # prob_threshold = 0.1  # Too low recall
    # prob_threshold = 0.02

    filtered_indices = np.where(y_pred[:, 1] < prob_threshold)[0]

    logging.info('Number of variants under the probability threshold %.2f: %d', prob_threshold, len(filtered_indices))

    # Print all data for the filtered variants if >10kb absolute svlen
    # print_filtered = True
    # if print_filtered:
    #     logging.info('Filtered variants:\n')
    #     min_svlen = 10000
    #     filtered_variants = feature_df.iloc[filtered_indices]
    #     # Print all the features for the filtered variants
    #     for index, row in filtered_variants.iterrows():
    #         # Print if abs(svlen) > 8000 and HMM is not equal to zero
    #         if abs(row['sv_length']) > min_svlen:
    #             logging.info('Features: %s', row.to_dict())

    # Get the IDs of the filtered variants
    filtered_ids = id_col.iloc[filtered_indices].values
    # logging.info('Filtered IDs:\n%s', filtered_ids)
    # Save the filtered IDs to a text file
    filtered_ids_file = os.path.join(output_dir, 'filtered_ids.txt')
    np.savetxt(filtered_ids_file, filtered_ids, fmt='%s')
    logging.info('Saved the filtered IDs to %s', filtered_ids_file)

    # Create a VCF file with only the filtered variants
    removed_svs_vcf = os.path.join(output_dir, 'removed_svs.vcf')

    # Create a CSV file with the filtered variants (CHROM, POS, ID, SVTYPE,
    # SVLEN, [... SHAP value for each feature], Predicted probability, Predicted
    # class)
    
    logging.info('Creating a CSV file with the filtered variants...')
    import shap
    shap_df = pd.DataFrame()
    shap_df['id'] = id_col.values
    shap_df['chrom'] = chrom_col.values
    shap_df['start'] = start_col.values
    shap_df['end'] = end_col.values
    shap_df['sv_type_str'] = sv_type_str_col.values
    # shap_df['sv_length'] = sv_length_col.values
    shap_df['sv_length'] = feature_df['sv_length'].values  # Use the original sv_length from feature_df
    shap_df['read_depth'] = read_depth_col.values
    shap_df['cluster_size'] = cluster_size_col.values
    shap_df['predicted_probability'] = y_pred[:, 1]
    shap_df['predicted_class'] = (y_pred[:, 1] >= prob_threshold).astype(int)

    # Calculate SHAP values
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(feature_df)
    shap_df_shap = pd.DataFrame(shap_values, columns=feature_df.columns)

    # Combine the SHAP values with the filtered variants DataFrame
    shap_df = pd.concat([shap_df, shap_df_shap], axis=1)

    # for col in shap_df.columns:
    #     if col not in ['id', 'chrom', 'predicted_probability', 'predicted_class']:
    #         shap_df[col] = shap_df[col].astype(float)
    shap_df = shap_df[shap_df['id'].isin(filtered_ids)]
    logging.info('Filtered SHAP values DataFrame:\n%s', shap_df.head())
    logging.info('Number of filtered variants: %d', len(shap_df))

    # Save the SHAP values to a CSV file
    logging.info('Saving the filtered variant SHAP values to a CSV file...')

    # Move the CHROM, START, END, SVTYPE, SVLEN, READ_DEPTH, CLUSTER_SIZE, PREDICTED_PROBABILITY, PREDICTED_CLASS columns to the front
    # shap_df = shap_df[['chrom', 'id', 'start', 'end', 'sv_type_str', 'sv_length', 'read_depth', 'cluster_size',
    #                    'predicted_probability', 'predicted_class'] + 
    #                    [col for col in shap_df.columns if col not in ['chrom', 'id', 'start', 'end', 'sv_type_str', 'sv_length', 'read_depth', 'cluster_size', 'predicted_probability', 'predicted_class']]]
    # # shap_df = shap_df[['chrom', 'id', 'predicted_probability', 'predicted_class'] + [col for col in shap_df.columns if col not in ['chrom', 'id', 'predicted_probability', 'predicted_class']]]

    shap_csv_file = os.path.join(output_dir, 'filtered_variants.csv')
    shap_df.to_csv(shap_csv_file, index=False)
    logging.info('Saved the filtered variant SHAP values to %s', shap_csv_file)

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
    score(model, input_vcf, output_vcf, buildver=buildver, title=args.title)
    logging.info('Scoring process completed.')
