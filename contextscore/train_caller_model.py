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
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# from extract_features import extract_features

# Set up the logger.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_features(input_bed):
    """Extract the features from the BED file, columns are in the first row:
    chrom, start, end, sv_type, sv_length, genotype, read_depth, hmm_llh, aln_type, cluster_size
    """
    logging.info('Extracting features from the BED file %s', input_bed)

    # Load a dictionary mapping chromosome names to numbers.
    chrom_dict_path="/mnt/isilon/wang_lab/perdomoj/projects/ContextScore/Train/Model/chrom_map.pkl"
    chrom_dict = joblib.load(chrom_dict_path)

    # Read in the BED file.
    bed_df = pd.read_csv(input_bed, sep='\t', header=0, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                         names=['chrom', 'start', 'end', 'sv_type', 'sv_length', 'genotype', 'read_depth', 'hmm_llh', 'aln_type', 'cluster_size'],
                         dtype={'chrom': str, 'start': np.int32, 'end': np.int32, 'sv_type': str, 'sv_length': np.int32, 'genotype': str, 'read_depth': np.int32, 'hmm_llh': np.float32, 'aln_type': str, 'cluster_size': np.int32})

    # Print the number of NaN values
    logging.info('Number of NaN values: %d', bed_df.isnull().sum().sum())

    # Map the chromosome names to numbers.
    bed_df['chrom'] = bed_df['chrom'].map(chrom_dict)

    # Print the number of NaN values
    logging.info('Number of NaN values after chr mapping: %d', bed_df.isnull().sum().sum())

    # Create a map of alignment types to numbers.
    # Alignment types are: "CIGARINS", "CIGARDEL", "CIGARCLIP", "SPLIT",
    # "SPLITDIST1", "SPLITDIST2", "SPLITINV", "SUPPINV", "HMM", "UNKNOWN"
    aln_type_map = {
        'CIGARINS': 0,
        'CIGARDEL': 1,
        'CIGARCLIP': 2,
        'SPLIT': 3,
        'SPLITDIST1': 4,
        'SPLITDIST2': 5,
        'SPLITINV': 6,
        'SUPPINV': 7,
        'HMM': 8,
        'UNKNOWN': 9
    }

    # Map the alignment types to numbers.
    bed_df['aln_type'] = bed_df['aln_type'].map(aln_type_map)

    # Print the number of NaN values
    logging.info('Number of NaN values after aln_type mapping: %d', bed_df.isnull().sum().sum())

    # Create a map of SV types to numbers.
    # SV types are: "DEL", "DUP", "INV", "INS", "BND", "UNKNOWN"
    sv_type_map = {
        'DEL': 0,
        'DUP': 1,
        'INV': 2,
        'INS': 3,
        'BND': 4,
        'UNKNOWN': 5
    }

    # Map the SV types to numbers.
    bed_df['sv_type'] = bed_df['sv_type'].map(sv_type_map)

    # Print the number of NaN values
    logging.info('Number of NaN values after sv_type mapping: %d', bed_df.isnull().sum().sum())

    # Create a map of genotypes to numbers.
    # Genotypes are: "0/0", "0/1", "1/1", "./."
    genotype_map = {
        '0/0': 0,
        '0/1': 1,
        '1/1': 2,
        './.': 3
    }

    # Map the genotypes to numbers.
    bed_df['genotype'] = bed_df['genotype'].map(genotype_map)

    # Print the number of NaN values
    logging.info('Number of NaN values after genotype mapping: %d', bed_df.isnull().sum().sum())

    # Check if any features are missing.
    if bed_df.isnull().values.any():
        logging.error('Features are missing.')

        # Get the rows with missing features.
        missing_features = bed_df[bed_df.isnull().any(axis=1)]

        # Print the rows with missing features.
        logging.error(missing_features)
        sys.exit(1)

    # Return the features.
    return bed_df


def train(tp_bed, fp_bed, output_directory):
    """Train the binary classification model."""

    # Extract the features from the VCF files.
    logging.info('Extracting features from the true positive file %s', tp_bed)
    tp_data = extract_features(tp_bed)

    # Check if any features are missing.
    if tp_data.isnull().values.any():
        logging.error('Features are missing.')

        # Get the rows with missing features.
        missing_features = tp_data[tp_data.isnull().any(axis=1)]

        # Print the rows with missing features.
        logging.error(missing_features)
        sys.exit(1)

    logging.info('Extracting features from the false positive file %s', fp_bed)
    fp_data = extract_features(fp_bed)

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

    # Drop NaN values from the data.
    logging.info('Dropping NaN values from the data.')
    tp_data.dropna(inplace=True)
    fp_data.dropna(inplace=True)

    # Balance the dataset by undersampling the true positives.
    logging.info('Balancing the dataset by undersampling the true positives (count = %d) to match the false positives (count = %d)', tp_data.shape[0], fp_data.shape[0])
    tp_data = tp_data.sample(fp_data.shape[0], random_state=42)

    # Combine the true positive and false positive data.
    data = pd.concat([tp_data, fp_data])

    # Get the features and labels.
    features = data.drop(columns=['label'])
    labels = data["label"]

    # Train different models.
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "SVC": SVC(kernel='rbf', class_weight='balanced', probability=True)
    }

    # models = {
    #     "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    # }

    for model_name, model in models.items():
        logging.info('Training the %s model.', model_name)
        # logging.info('Training the model.')
        # # model = LogisticRegression()
        # model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(features, labels)

        # Get predicted probabilities.
        logging.info('Getting predicted probabilities.')
        y_pred = model.predict(features)
        y_prob = model.predict_proba(features)[:, 1]

        # Get the ROC curve.
        fpr, tpr, thresholds = roc_curve(labels, y_prob)
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve.
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        # Save the plot to the output directory.
        roc_plot_path = os.path.join(output_directory, model_name + '_roc_curve.png')
        plt.savefig(roc_plot_path)
        plt.close()
        logging.info('Saved the ROC curve to %s', roc_plot_path)

        # Get the precision-recall curve.
        precision, recall, thresholds = precision_recall_curve(labels, y_prob)
        pr_auc = auc(recall, precision)

        # Plot the precision-recall curve.
        plt.figure()
        plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (area = %0.2f)' % pr_auc)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        # Save the plot to the output directory.
        pr_plot_path = os.path.join(output_directory, model_name + '_pr_curve.png')
        plt.savefig(pr_plot_path)
        plt.close()
        logging.info('Saved the Precision-Recall curve to %s', pr_plot_path)

        # Get the confusion matrix.
        cm = confusion_matrix(labels, y_pred)
        logging.info('Confusion matrix:\n%s', cm)

        # Plot the confusion matrix using seaborn.
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        # Save the plot to the output directory.
        cm_plot_path = os.path.join(output_directory, model_name + '_confusion_matrix.png')
        plt.savefig(cm_plot_path)
        plt.close()
        logging.info('Saved the confusion matrix to %s', cm_plot_path)

        # Print the classification report.
        logging.info('Classification report:\n%s', classification_report(labels, y_pred))

        # Save the report to a file.
        report_path = os.path.join(output_directory, model_name + '_classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(classification_report(labels, y_pred))

        logging.info('Saved the classification report to %s', report_path)

        # Save the model.
        model_path = os.path.join(output_directory, model_name + '_caller_model.pkl')
        logging.info('Saving the model to %s', model_path)
        joblib.dump(model, model_path)
        logging.info('Saved the model to %s', model_path)

        # Run cross-validation by splitting the data into 5 folds and training
        # the model on each fold.
        from sklearn.model_selection import cross_val_score
        logging.info('Running cross-validation.')
        scores = cross_val_score(model, features, labels, cv=5)
        logging.info('Cross-validation scores: %s', scores)
        logging.info('Mean cross-validation score: %f', scores.mean())


# Run the program.
def run(tp_bed, fp_bed, output_directory):
    """Run the program."""
    # Train the model.
    train(tp_bed, fp_bed, output_directory)
    # Create the output directory if it does not exist.
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tpbed', type=str, required=True, help='Path to the VCF of true positive SV calls obtained from a benchmarking dataset.')
    parser.add_argument('--fpbed', type=str, required=True, help='Path to the VCF of false positive SV calls obtained from running the caller on data that is known to be negative for SVs.')
    parser.add_argument('--outdir', type=str, required=True, help='Path to the output directory.')
    args = parser.parse_args()

    # Run the program.
    logging.info('Training the model...')
    run(args.tpbed, args.fpbed, args.outdir)
    logging.info('done.')
