"""
train_model.py - Train the binary classification model.

This script trains the binary classification model using the true positive and
false positive data. The true positive data is obtained from a benchmarking
dataset. The false positive data is obtained from running the caller on data
that is known to be negative for SVs. This data can be obtained by running the
caller on a normal sample with known SVs accounted for in the reference genome.

For example for HG002, the true positive data is obtained from the Genome in a
Bottle benchmarking dataset, and the false positive data is obtained from
running the caller on the HG002 normal sample and extracting the SV calls that
are not in the benchmarking dataset. This can be repeated for other samples such
as HG001 and HG005 as long as the known SVs are accounted for.

In the HG002 SV v0.6 dataset, there are low-confidence regions which
are excluded from the true positive data. Thus, we must include true SVs from
other publicly available normal samples with information from complex regions,
such as those aligned to CHM13. 

The model is trained using logistic regression. The features are the LRR and
BAF values. The labels are 1 for true positives and 0 for false positives.

The model is saved to the output directory as a pickle file.

Usage:
    python train_model.py <true_positives_filepath> <false_positives_filepath>
    <output_directory>
    
    true_positives_filepath: Path to the VCF of true positive SV calls obtained
        from a benchmarking dataset.
    false_positives_filepath: Path to the VCF of false positive SV calls
        obtained from running the caller on data that is known to be negative
        for SVs. This data can be obtained by running the caller on a normal
        sample with known SVs accounted for in the reference genome.

    output_directory: Path to the output directory.

Output:
    model.pkl: The binary classification model.

Example:
    python train_model.py data/sv_scoring_dataset/true_positives.vcf
    sv_scoring_dataset/false_positives.vcf data/sv_scoring_dataset/model
"""

import os
import sys
import logging
import numpy as np
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from extract_features import extract_features

# Set up the logger.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train(tp_files, fp_files):
    """Train the binary classification model."""

    # Extract the features from the VCF files.
    logging.info('Extracting features from the true positive VCF file.')
    
    # Set up the dataframe with all the features.
    feature_cols=[
        "label",
        "chrom",
        "start",
        "sv_length",
        "sv_type",
        "cluster_size",
        "hmm_llh",
        "segdup",
        "repeatregions",
        "telomere",
        "centromere",
        "fragile_site",
        "conserved_region"
    ]
    tp_data = pd.DataFrame(columns=feature_cols)
    for tp_file in tp_files:
        # Extract the features from the true positive VCF file.
        tp_data = pd.concat([tp_data, extract_features(tp_file)], ignore_index=True)
        logging.info('Extracted features from %s', tp_file)

    # Check if the true positive data is empty.
    if tp_data.empty:
        logging.error('True positive data is empty.')
        sys.exit(1)

    for fp_file in fp_files:
        logging.info('Extracting features from the false positive VCF file.')
        # Extract the features from the false positive VCF file.
        fp_data = extract_features(fp_file)
        logging.info('Extracted features from %s', fp_file)

    # Check if the false positive data is empty.
    if fp_data.empty:
        logging.error('False positive data is empty.')
        sys.exit(1)

    

    # Check if any features are missing.
    if tp_data.isnull().values.any():
        logging.error('Features are missing.')

        # Get the rows with missing features.
        missing_features = tp_data[tp_data.isnull().any(axis=1)]

        # Print the rows with missing features.
        logging.error(missing_features)
        sys.exit(1)

    logging.info('Extracting features from the false positive VCF file.')
    fp_data = extract_features(false_positives_filepath)

    # Check if any features are missing.
    if fp_data.isnull().values.any():
        logging.error('Features are missing.')

        # Get the rows with missing features.
        missing_features = fp_data[fp_data.isnull().any(axis=1)]

        # Print the rows with missing features.
        logging.error(missing_features)
        sys.exit(1)

    # Add the labels.
    tp_data['label'] = 1
    fp_data['label'] = 0

    # Print the number of true positives and false positives.
    logging.info('Number of true labels: %d', tp_data.shape[0])
    logging.info('Number of false labels: %d', fp_data.shape[0])

    # Combine the true positive and false positive data.
    data = pd.concat([tp_data, fp_data])

    # Get the features and labels.
    features = data[["chrom", "start", "sv_length", "sv_type", "read_support", "clipped_bases"]]
    labels = data["label"]

    # Check if any features are missing.
    if features.isnull().values.any():
        logging.error('Features are missing.')

        # Get the rows with missing features.
        missing_features = features[features.isnull().any(axis=1)]

        # Print the rows with missing features.
        logging.error(missing_features)
        sys.exit(1)

    # Check if any labels are missing.
    if labels.isnull().values.any():
        logging.error('Labels are missing.')
        sys.exit(1)

    # Train the model.
    model = LogisticRegression()
    model.fit(features, labels)

    # Return the model.
    return model

# Run the program.
def run(tp_dir, fp_dir, output_directory, annovar_path, db_path):
    """Train the binary classification model."""
    logging.info('Getting the true positive and false positive VCF files.')

    # Get all *.jl files in the true positive directory.
    tp_files = [os.path.join(tp_dir, f) for f in os.listdir(tp_dir) if f.endswith('.jl')]
    # Get all *.jl files in the false positive directory.
    fp_files = [os.path.join(fp_dir, f) for f in os.listdir(fp_dir) if f.endswith('.jl')]
    
    # Check if the true positive and false positive directories are empty.
    if not tp_files:
        logging.error('No true positive VCF files found in the directory.')
        return
    
    if not fp_files:
        logging.error('No false positive VCF files found in the directory.')
        return

    # Print the files.
    logging.info('True positive VCF files:')
    for file in tp_files:
        logging.info(file)

    logging.info('False positive VCF files:')
    for file in fp_files:
        logging.info(file)

    logging.info('Output directory: %s', output_directory)
    logging.info('ANNOVAR path: %s', annovar_path)
    logging.info('ANNOVAR database path: %s', db_path)

    # Check if the output directory exists.
    if not os.path.exists(output_directory):
        logging.info('Creating the output directory.')
        os.makedirs(output_directory)

    model = train(tp_files, fp_files)

    logging.info('Model failed successfully!')

    # logging.info('Training the model.')
    # # Train the model using the true positive and false positive VCF files.
    # model = train(true_positives_files, false_positives_files)
    # logging.info('Model trained successfully.')
    # # Save the model to the output directory.
    # model_path = os.path.join(output_directory, "model.pkl")
    # logging.info('Saving the model to %s', model_path)
    # joblib.dump(model, model_path)
    # logging.info('Model saved successfully.')

    # # Print the model.
    # logging.info('Model: %s', model)

    # # Print the model coefficients.
    # logging.info('Model coefficients: %s', model.coef_)

    # # Print the model intercept.
    # logging.info('Model intercept: %s', model.intercept_)

    # # Print the model score.
    # logging.info('Model score: %s', model.score(features, labels))

    # # Print the model accuracy.
    # logging.info('Model accuracy: %s', model.score(features, labels))

    # # Print the model precision.
    # logging.info('Model precision: %s', model.score(features, labels))

    # # Print the model recall.
    # logging.info('Model recall: %s', model.score(features, labels))

    # # Print the model F1 score.
    # logging.info('Model F1 score: %s', model.score(features, labels))

    # Check if the output directory is empty.
    # Train the model.
    # model = train(true_positives_filepath, false_positives_filepath)

    # # Create the output directory if it does not exist.
    # if not os.path.exists(output_directory):
    #     os.makedirs(output_directory)

    # # Save the model
    # model_path = os.path.join(output_directory, "model.pkl")
    # joblib.dump(model, model_path)

    # # Print the model.
    # print(model)

    # Return the model.
    # return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpdir", required=True, help="Directory containing benchmark VCF files of real SVs (true positives and false negatives)")
    parser.add_argument("--fpdir", required=True, help="Directory containing false positive VCF files from running the caller on normal samples")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--annovar", required=True, help="Path to ANNOVAR")
    parser.add_argument("--annovar_db", required=True, help="Path to ANNOVAR database")
    args = parser.parse_args()
    # Get the command line arguments.
    # if len(sys.argv) != 4:
    #     logging.error('Usage: python train_model.py <true_positives_filepath> <false_positives_filepath> <output_directory>\n')
    #     sys.exit(1)

    # # Input VCF of true positive SV calls obtained from a benchmarking dataset.
    # tp_filepath = sys.argv[1]

    # # Input VCF of false positive SV calls obtained from running the caller on
    # # data that is known to be negative for SVs. This data can be obtained by
    # # running the caller on a normal sample with known SVs accounted for in the
    # # reference genome.
    # fp_filepath = sys.argv[2]
    # output_dir = sys.argv[3]

    # Run the program.
    logging.info('Training the model...')
    run(args.tpdir, args.fpdir, args.outdir, args.annovar, args.annovar_db)
    # run(tp_filepath, fp_filepath, output_dir)
    logging.info('done.')
