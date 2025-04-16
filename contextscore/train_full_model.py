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
import subprocess
import logging
import numpy as np
import joblib
import pandas as pd
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from extract_features import extract_features

# Set up the logger.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train(tp_bed, fp_bed, output_directory, annovar_path, db_path, outdiranno, tp_bed_hg19=None, fp_bed_hg19=None):
    """Train the binary classification model."""

    # ---------------------------------------------------------------
    # SV Feature Extraction
    # ---------------------------------------------------------------


    # Extract the features from the VCF files.
    logging.info('Extracting features from the true positive and false positive VCF files (GRCh38).')
    buildversion = 'hg38'
    tp_anno_outdir = os.path.join(outdiranno, "tp_anno")
    tp_data = extract_features(tp_bed, annovar_path, db_path, tp_anno_outdir, buildversion=buildversion)
    logging.info('Extracted %d features from the true positive VCF file.', tp_data.shape[0])
    fp_anno_outdir = os.path.join(outdiranno, "fp_anno")
    fp_data = extract_features(fp_bed, annovar_path, db_path, fp_anno_outdir, buildversion=buildversion)
    logging.info('Extracted %d features from the false positive VCF file.', fp_data.shape[0])

    logging.info('Extracting features from the true positive and false positive VCF files (HG002-GRCh19).')
    buildversion = 'hg19'
    if tp_bed_hg19 is not None and fp_bed_hg19 is not None:
        tp_anno_outdir_hg19 = os.path.join(outdiranno, "tp_anno_hg19")
        tp_data_hg19 = extract_features(tp_bed_hg19, annovar_path, db_path, tp_anno_outdir_hg19, buildversion=buildversion)
        logging.info('Extracted %d features from the true positive VCF file (hg19).', tp_data_hg19.shape[0])
        fp_anno_outdir_hg19 = os.path.join(outdiranno, "fp_anno_hg19")
        fp_data_hg19 = extract_features(fp_bed_hg19, annovar_path, db_path, fp_anno_outdir_hg19, buildversion=buildversion)
        logging.info('Extracted %d features from the false positive VCF file (hg19).', fp_data_hg19.shape[0])

        # Concatenate the data from hg38 and hg19.
        logging.info('Concatenating the data from hg38 and hg19.')
        tp_data = pd.concat([tp_data, tp_data_hg19], ignore_index=True)
        fp_data = pd.concat([fp_data, fp_data_hg19], ignore_index=True)
    else:
        logging.info('No hg19 data provided. Using only hg38 data.')
    logging.info('Feature extraction completed. True positives: %d, False positives: %d',
                 tp_data.shape[0], fp_data.shape[0])
    
    # ---------------------------------------------------------------
    # Data Preprocessing
    # ---------------------------------------------------------------
    # Drop the genotype column from the data.
    logging.info('Dropping the genotype column from the data.')
    tp_data.drop(columns=['genotype'], inplace=True, errors='ignore')
    fp_data.drop(columns=['genotype'], inplace=True, errors='ignore')

    # Drop the cn_state column from the data.
    logging.info('Dropping the cn_state column from the data.')
    tp_data.drop(columns=['cn_state'], inplace=True, errors='ignore')
    fp_data.drop(columns=['cn_state'], inplace=True, errors='ignore')

    # Analyze feature correlations in the collected data.
    logging.info('Analyzing feature correlations in the collected data.')
    corr_matrix = tp_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix (True Positives)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'feature_correlation_tp.png'))
    plt.close()
    corr_matrix = fp_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix (False Positives)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'feature_correlation_fp.png'))
    plt.close()
    logging.info('Feature correlation analysis completed. TP saved to %s and FP saved to %s',
                 os.path.join(output_directory, 'feature_correlation_tp.png'),
                 os.path.join(output_directory, 'feature_correlation_fp.png'))
    
    # Analyze feature correlations in the combined data.
    logging.info('Analyzing feature correlations in the combined data.')
    combined_data = pd.concat([tp_data, fp_data])
    corr_matrix_combined = combined_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix_combined, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix (Combined Data)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'feature_correlation_combined.png'))
    plt.close()
    logging.info('Feature correlation analysis completed for combined data. Saved to %s',
                 os.path.join(output_directory, 'feature_correlation_combined.png'))

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
    logging.info('Number of true labels after dropping NaN values: %d', tp_data.shape[0])
    logging.info('Number of false labels after dropping NaN values: %d', fp_data.shape[0])

    # Balance the dataset by undersampling the true positives.
    logging.info('Balancing the dataset by undersampling the true positives (count = %d) to match the false positives (count = %d)', tp_data.shape[0], fp_data.shape[0])
    tp_data = tp_data.sample(fp_data.shape[0], random_state=42)

    # Plot the differences in correlation between true positives and false
    # positives.
    # diff_corr = tp_data.corr() - fp_data.corr()
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(diff_corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    # plt.title('Difference in Feature Correlation (True Positives - False Positives)')
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_directory, 'feature_correlation_difference.png'))
    # plt.close()
    # logging.info('Feature correlation difference analysis completed. Saved to %s',
    #              os.path.join(output_directory, 'feature_correlation_difference.png'))

    # [TEST] Exit after this step to verify the feature extraction and data
    # preprocessing.
    # sys.exit(0)

    # Combine the true positive and false positive data.
    data = pd.concat([tp_data, fp_data])

    # Get the features and labels.
    features = data.drop(columns=['label'])
    labels = data["label"]

    # Train different models.
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "SVC": SVC(kernel='linear', class_weight='balanced', probability=True)
    }

    for model_name, model in models.items():
        # Split the data into training and testing sets.
        logging.info('Splitting the data into training and testing sets (0.8/0.2).')
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        # If SVC, scale the data.
        if model_name == "SVC":
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Train the model.
        logging.info('Training the %s model.', model_name)
        # model.fit(features, labels)
        model.fit(X_train, y_train)

        # Get predicted probabilities for the training and testing sets.
        y_train_prob = model.predict_proba(X_train)[:, 1]
        y_test_prob = model.predict_proba(X_test)[:, 1]

        # Compute the ROC curve and ROC area for the training set.
        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
        roc_auc_train = auc(fpr_train, tpr_train)

        # Compute the ROC curve and ROC area for the testing set.
        fpr_test, tpr_test, thresholds = roc_curve(y_test, y_test_prob)
        roc_auc_test = auc(fpr_test, tpr_test)

        # Use Youden's J statistic to find the optimal threshold.
        j_scores = tpr_test - fpr_test
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        logging.info('Optimal threshold (Youden\'s J statistic): %f', optimal_threshold)
        logging.info('True positive rate (sensitivity): %f', tpr_test[optimal_idx])
        logging.info('False positive rate (1 - specificity): %f', fpr_test[optimal_idx])

        # Print the ROC AUC scores.
        logging.info('ROC AUC score for the training set: %f', roc_auc_train)
        logging.info('ROC AUC score for the testing set: %f', roc_auc_test)

        # Plot the ROC curve for the training set.
        plt.figure()
        plt.plot(fpr_train, tpr_train, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_train)
        # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (Training Set)')
        plt.legend(loc='lower right')
        # Save the plot to the output directory.
        roc_plot_path = os.path.join(output_directory, model_name + '_roc_curve.png')
        plt.savefig(roc_plot_path)
        plt.close()
        logging.info('Saved the ROC curve to %s', roc_plot_path)

        # Plot the ROC curve for the testing set.
        plt.figure()
        plt.plot(fpr_test, tpr_test, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_test)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (Testing Set)')
        plt.legend(loc='lower right')
        # Save the plot to the output directory.
        roc_plot_path = os.path.join(output_directory, model_name + '_roc_curve_test.png')
        plt.savefig(roc_plot_path)
        plt.close()
        logging.info('Saved the ROC curve to %s', roc_plot_path)

        # Compute precision-recall curve
        precision, recall, thresholds_pr = precision_recall_curve(y_test, y_test_prob)

        logging.info('precision size: %d', len(precision))
        logging.info('recall size: %d', len(recall))
        logging.info('thresholds size: %d', len(thresholds_pr))

        # Find the threshold that gives the highest precision (ideally with
        # recall > 0) or where precision == 1.0 (0 false positives).
        precision_1_indices = np.where(precision[:-1] == 1.0)[0]
        if len(precision_1_indices) > 0:
            # If there are indices where precision == 1.0, use the one with the
            # highest recall.
            optimal_index = precision_1_indices[np.argmax(recall[precision_1_indices])]
            optimal_threshold_pr = thresholds_pr[optimal_index]
            logging.info('Optimal threshold (highest precision = 1.0): %f with recall %f',
                        optimal_threshold_pr, recall[optimal_index])
        else:
            # If no indices where precision == 1.0, use the one with the highest
            # precision.
            optimal_index = np.argmax(precision[:-1])
            optimal_threshold_pr = thresholds_pr[optimal_index]
            logging.info('Optimal threshold (highest precision = %f): %f with recall %f',
                        optimal_threshold_pr, precision[optimal_index], recall[optimal_index])

        # Get the feature names.
        feature_names = features.columns.tolist()
        logging.info('Feature names: %s', feature_names)
        logging.info('Number of features: %d', len(feature_names))

        # Feature importance for Random_Forest and XGBoost.
        if model_name in ["Random_Forest", "XGBoost"]:
            # Get feature importances.
            importances = model.feature_importances_

            # Sort the feature importances in descending order.
            indices = np.argsort(importances)[::-1]

            # Print the feature ranking.
            logging.info('Feature ranking:')
            for f in range(X_train.shape[1]):
                logging.info('%d. Feature %s (%f)', f + 1, feature_names[indices[f]], importances[indices[f]])

            # Plot the feature importances.
            plt.figure()
            plt.title('Feature Importances')
            plt.bar(range(X_train.shape[1]), importances[indices], align='center')
            plt.xticks(range(X_train.shape[1]), indices)
            plt.xlim([-1, X_train.shape[1]])
            # Save the plot to the output directory.
            importance_plot_path = os.path.join(output_directory, model_name + '_feature_importances.png')
            plt.savefig(importance_plot_path)
            plt.close()
            logging.info('Saved the feature importances plot to %s', importance_plot_path)

        # For SVC, get the coefficients.
        if model_name == "SVC":
            # Get the coefficients.
            # coefficients = model.coef_[0]

            # Sort the coefficients in descending order.
            indices = np.argsort(coefficients)[::-1]

            # Print the feature ranking.
            logging.info('Feature ranking:')
            for f in range(X_train.shape[1]):
                logging.info('%d. Feature %s (%f)', f + 1, feature_names[indices[f]], coefficients[indices[f]])

            # Plot the coefficients.
            plt.figure()
            plt.title('Feature Coefficients')
            plt.bar(range(X_train.shape[1]), coefficients[indices], align='center')
            plt.xticks(range(X_train.shape[1]), indices)
            plt.xlim([-1, X_train.shape[1]])
            # Save the plot to the output directory.
            coeff_plot_path = os.path.join(output_directory, model_name + '_feature_coefficients.png')
            plt.savefig(coeff_plot_path)
            plt.close()
            logging.info('Saved the feature coefficients plot to %s', coeff_plot_path)

        # For logistic regression, get the coefficients.
        if model_name == "Logistic Regression":
            # Get the coefficients.
            coefficients = model.coef_[0]

            # Sort the coefficients in descending order.
            indices = np.argsort(coefficients)[::-1]

            # Print the feature ranking.
            logging.info('Feature ranking:')
            for f in range(X_train.shape[1]):
                logging.info('%d. Feature %s (%f)', f + 1, feature_names[indices[f]], coefficients[indices[f]])

            # Plot the coefficients.
            plt.figure()
            plt.title('Feature Coefficients')
            plt.bar(range(X_train.shape[1]), coefficients[indices], align='center')
            plt.xticks(range(X_train.shape[1]), indices)
            plt.xlim([-1, X_train.shape[1]])
            # Save the plot to the output directory.
            coeff_plot_path = os.path.join(output_directory, model_name + '_feature_coefficients.png')
            plt.savefig(coeff_plot_path)
            plt.close()
            logging.info('Saved the feature coefficients plot to %s', coeff_plot_path)

        # Get the precision-recall curve f
        # precision, recall, thresholds = precision_recall_curve(labels, y_prob)
        # pr_auc = auc(recall, precision)

        # Plot the precision-recall curve.
        # plt.figure()
        # plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (area = %0.2f)' % pr_auc)
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.title('Precision-Recall Curve')
        # plt.legend(loc='lower left')
        # # Save the plot to the output directory.
        # pr_plot_path = os.path.join(output_directory, model_name + '_pr_curve.png')
        # plt.savefig(pr_plot_path)
        # plt.close()
        # logging.info('Saved the Precision-Recall curve to %s', pr_plot_path)

        # # Get the confusion matrix.
        # cm = confusion_matrix(labels, y_pred)
        # logging.info('Confusion matrix:\n%s', cm)

        # # Plot the confusion matrix using seaborn.
        # plt.figure()
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        # plt.xlabel('Predicted')
        # plt.ylabel('True')
        # plt.title('Confusion Matrix')
        # # Save the plot to the output directory.
        # cm_plot_path = os.path.join(output_directory, model_name + '_confusion_matrix.png')
        # plt.savefig(cm_plot_path)
        # plt.close()
        # logging.info('Saved the confusion matrix to %s', cm_plot_path)

        # # Print the classification report.
        # logging.info('Classification report:\n%s', classification_report(labels, y_pred))

        # # Save the report to a file.
        # report_path = os.path.join(output_directory, model_name + '_classification_report.txt')
        # with open(report_path, 'w') as f:
        #     f.write(classification_report(labels, y_pred))

        # logging.info('Saved the classification report to %s', report_path)

        # Save the model.
        model_path = os.path.join(output_directory, model_name + '_caller_model.pkl')
        logging.info('Saving the model to %s', model_path)
        joblib.dump(model, model_path)
        logging.info('Saved the model to %s', model_path)

        # Run cross-validation by splitting the data into 5 folds and training
        # the model on each fold.
        # from sklearn.model_selection import cross_val_score
        # logging.info('Running cross-validation.')
        # scores = cross_val_score(model, features, labels, cv=5, scoring='f1')
        # logging.info('Cross-validation scores: %s', scores)
        # logging.info('Mean cross-validation score: %f', scores.mean())


def run(tp_bed, fp_bed, output_directory, annovar_path, db_path, outdiranno, tp_bed_hg19=None, fp_bed_hg19=None):
    """Run the training process."""
    train(tp_bed, fp_bed, output_directory, annovar_path, db_path, outdiranno, tp_bed_hg19, fp_bed_hg19)


if __name__ == '__main__':
    # Parse the command line arguments.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpbed", required=True, help="Directory containing true positive SVs in hg38")
    parser.add_argument("--fpbed", required=True, help="Directory containing false positive SVs in hg38")
    parser.add_argument("--tpbed_hg19", required=False, help="Directory containing true positive SVs in hg19")
    parser.add_argument("--fpbed_hg19", required=False, help="Directory containing false positive SVs in hg19")
    parser.add_argument("--outdiranno", required=True, help="Output directory for saving the ANNOVAR annotations")
    parser.add_argument("--outdir", required=True, help="Output directory for saving the model")
    parser.add_argument("--annovar", required=True, help="Path to ANNOVAR")
    parser.add_argument("--annovar_db", required=True, help="Path to ANNOVAR database")
    args = parser.parse_args()

    # Run the program.
    logging.info('Training the model...')
    run(args.tpbed, args.fpbed, args.outdir, args.annovar, args.annovar_db, args.outdiranno, args.tpbed_hg19, args.fpbed_hg19)
    logging.info('done.')
