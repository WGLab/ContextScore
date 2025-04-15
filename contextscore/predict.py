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


def score(model, input_vcf, output_vcf):
    """Score the structural variants using the binary classification model.

    Args:
        model (str): Path to the model file.
        input_vcf (str): Path to the input VCF file.
        output_vcf (str): Path to the output VCF file.
    """
    # Load the model
    clf = joblib.load(model)

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
    