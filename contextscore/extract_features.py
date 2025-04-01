"""
extract_features.py: Extract features from the input VCF file.

Usage:
    extract_features.py <input>

Arguments:
    <input>     Path to the input VCF file.

Output:
    A dataframe with a column for each feature.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

import joblib
import seaborn as sb
import matplotlib.pyplot as plt

# data = joblib.load("test.jl")
# p = sb.countplot(data=data[data["state"] == 'tp'], x="szbin", hue="svtype", hue_order=["DEL", "INS"])
# plt.xticks(rotation=45, ha='right')
# p.set(title="True Positives by svtype and szbin")

def read_vcf(filepath):
    """Read in the VCF file."""
    vcf_df = pd.read_csv(filepath, sep='\t', comment='#', header=None, usecols=[0, 1, 7, 8], \
                         names=['CHROM', 'POS', 'INFO', 'FORMAT'], \
                            dtype={'CHROM': str, 'POS': np.int64, 'INFO': str, 'FORMAT': str})
    return vcf_df

def extract_features(input_vcf):
    """Extract the features from the VCF file's data."""
    # Read in the VCF file.
    vcf_df = read_vcf(input_vcf)

    # Extract the alignment type (string) from the INFO column.
    aln_type = vcf_df['INFO'].str.extract(r'ALN=(\w+)', expand=False)

    # Extract the cluster size from the INFO column.
    cluster_size = vcf_df['INFO'].str.extract(r'CLUSTER=(\d+)', expand=False).astype(np.int32)

    # Set 0 cluster size to nan.
    cluster_size[cluster_size == 0] = np.nan

    # Extract GT from the FORMAT column.
    gt = vcf_df['FORMAT'].str.extract(r'GT=(\d+)', expand=False).astype(np.int32)

    # Set ./. GT to nan.
    gt[gt == './.'] = np.nan

    # Check if any GT values are missing.
    if gt.isnull().values.any():
        logging.info('Number of missing GT values: ' + str(gt.isnull().sum()))

    # Extract DP from the FORMAT column.
    dp = vcf_df['FORMAT'].str.extract(r'DP=(\d+)', expand=False).astype(np.int32)

    # Get the array of chromosome names.
    chrom = vcf_df['CHROM']

    # Create a key to map the chromosome names to a unique integer.

    # First, get all unique chromosome names.
    chrom_unique = chrom.unique()

    # Next, create a dictionary to map the chromosome names to integers.
    chrom_dict = {chrom: i for i, chrom in enumerate(chrom_unique)}

    # Finally, map the chromosome names to integers.
    chrom = chrom.map(chrom_dict)


    # Check if any chromosome names are missing.
    if chrom.isnull().values.any():
        logging.error('Chromosome name is missing.')
        sys.exit(1)
    else:
        # Print space-separated chromosome names.
        logging.info('Chromosomes: ' + ' '.join(chrom.unique().astype(str)))

    # Get the start and end positions.
    start = vcf_df['POS']

    # Check if any start positions are missing.
    if start.isnull().values.any():
        logging.error('Start position is missing.')
        sys.exit(1)

    # Get the SV length from the INFO column.
    sv_length = vcf_df['INFO'].str.extract(r'SVLEN=(-?\d+)', expand=False).astype(np.int32)

    # Check if any SV lengths are missing.
    if sv_length.isnull().values.any():
        logging.error('SV length is missing.')
        sys.exit(1)

    # Get the SV type from the INFO column.
    sv_type = vcf_df['INFO'].str.extract(r'SVTYPE=(\w+)', expand=False)

    # If INFO/REPTYPE=DUP, then the SV type is a duplication.
    sv_type[vcf_df['INFO'].str.contains('REPTYPE=DUP')] = 'DUP'

    # Convert the SV type to integers.
    sv_type = sv_type.replace('DEL', '0')
    sv_type = sv_type.replace('DUP', '1')
    sv_type = sv_type.replace('INV', '2')
    sv_type = sv_type.replace('INS', '3')
    sv_type = sv_type.replace('BND', '4')
    sv_type = sv_type.astype(np.int32)

    # Check if any SV types are missing.
    if sv_type.isnull().values.any():
        logging.error('SV type is missing.')
        sys.exit(1)

    # Loop through the columns and check if any values are missing for all of
    # the feature arrays.
    for col in [chrom, start, sv_length, sv_type, read_support, clipped_bases]:
        if col.isnull().values.all():
            logging.error('All values are missing for a feature.')
            logging.error(col)
            sys.exit(1)

    # Print the first 4 rows of the features.
    logging.info('Features:')
    logging.info(pd.DataFrame({'chrom': chrom.head(4), 'start': start.head(4), 'sv_length': sv_length.head(4), \
                               'sv_type': sv_type.head(4), 'read_support': read_support.head(4), \
                               'clipped_bases': clipped_bases.head(4)}))
    
    # Check that all features have the same length.
    if not all(len(col) == len(chrom) for col in [start, sv_length, sv_type, read_support, clipped_bases]):
        logging.error('Features do not have the same length.')

        # Print the length of each feature.
        logging.error('Chromosomes: ' + str(len(chrom)))
        logging.error('Start positions: ' + str(len(start)))
        logging.error('SV lengths: ' + str(len(sv_length)))
        logging.error('SV types: ' + str(len(sv_type)))
        logging.error('Read support: ' + str(len(read_support)))
        logging.error('Clipped bases: ' + str(len(clipped_bases)))

        sys.exit(1)

    # Create a dataframe of the features.
    features = pd.DataFrame({'chrom': chrom, 'start': start, 'sv_length': sv_length, 'sv_type': sv_type, \
                             'read_support': read_support, 'clipped_bases': clipped_bases})

    # Check if any features are missing.
    if features.isnull().values.any():
        logging.error('Features are missing.')

        # Get the rows with missing features.
        missing_features = features[features.isnull().any(axis=1)]

        # Print the rows with missing features.
        logging.error(missing_features)
        sys.exit(1)

    # Return the features.
    return features
