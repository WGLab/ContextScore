"""
score_vcf.py - Score structural variants in a VCF file using a binary classification model.

This script prioritizes structural variants in a VCF file by scoring them using
a binary classification model. The model is trained using a VCF file of true
positive structural variants and a VCF file of false positive structural
variants. The model is trained using the following features extracted from the
VCF files: chromosome, start position, structural variant length, structural
variant type, read support, and clipped bases. The model is a logistic
regression model.

Usage:
    python score_vcf.py <model_path> <vcf_filepath>

Arguments:
    model_path: str
        Path to the trained model file.
    vcf_filepath: str
        Path to the VCF file to score.

Example:
    python score_vcf.py model.pkl structural_variants.vcf

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


def score(region_model_path, caller_model_path, vcf_filepath, output_vcf):
    """Load the model and VCF file and score the structural variants."""
    # Load the VCF file.
    logging.info('Extracting features from the VCF file.')
    features = extract_features(vcf_filepath)

    # Load the model.
    logging.info('Loading the region-based model.')
    region_model = joblib.load(region_model_path)

    logging.info('Loading the caller-based model.')
    caller_model = joblib.load(caller_model_path)

    # Score the structural variants separately using the region-based and
    # caller-based models.
    logging.info('Scoring the structural variants using the region-based model.')
    y_region = region_model.predict_proba(features)[:, 1]  # index 0=negative, 1=positive probabilities
    y_caller = caller_model.predict_proba(features)[:, 1]  # index 0=negative, 1=positive probabilities

    # Compute the confidence scores.
    conf_region = np.abs(y_region - 0.5)
    conf_caller = np.abs(y_caller - 0.5)

    # Normalize the confidence scores to get the weights (higher confidence =
    # higher weight)
    # (weights sum to 1)
    weights_region = conf_region / (conf_region + conf_caller)
    weights_caller = conf_caller / (conf_region + conf_caller)

    # Handle NaN values in the weights when conf=0 (0.5 SV prediction from both models), equal weight (0.5) for both models
    weights_region = np.nan_to_num(weights_region, nan=0.5)
    weights_caller = np.nan_to_num(weights_caller, nan=0.5)

    # Combine the scores from the two models using the weights.
    P_final = (weights_region * y_region + weights_caller * y_caller)

    # Second approach: Model performance-based weights.
    # auc_region = 0.85  # Performance of Model 1
    # auc_caller = 0.75  # Performance of Model 2

    # w1 = auc_region / (auc_region + auc_caller)
    # w2 = auc_caller / (auc_region + auc_caller)

    # P_final = w1 * y_region + w2 * y_caller

    # Plot a histogram of the scores.
    logging.info('Plotting the distribution of scores.')
    plt.hist(scores)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Scores')
    
    # Save the plot as a PNG file.
    output_png = "scores.png"
    plt.tight_layout()
    plt.savefig(output_png)
    logging.info('Saved the plot as %s.', output_png)


if __name__ == '__main__':
    # Get the command line arguments.
    if len(sys.argv) != 4:
        logging.error('Usage: python score_vcf.py <model_path> <input_vcf> <output_vcf>\n')
        sys.exit(1)

    # Get the model path and VCF file path.
    model_path = sys.argv[1]
    vcf_filepath = sys.argv[2]
    output_vcf = sys.argv[3]

    # Run the program.
    score(model_path, vcf_filepath, output_vcf)
    logging.info('done.')
