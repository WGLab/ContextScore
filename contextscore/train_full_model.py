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

from extract_features import extract_features, add_interaction_terms, normalize_column

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

    # Normalize cluster_size and read_depth using Robust scaling.
    # logging.info('Normalizing cluster_size and read_depth using Robust scaling.')
    # from sklearn.preprocessing import RobustScaler, MinMaxScaler
    # # Combine the data.
    # combined_data = pd.concat([tp_data, fp_data])
    # # Create a RobustScaler object.
    # scaler = RobustScaler()
    # # Fit the scaler to the data.
    # robust_scaled = scaler.fit_transform(combined_data[['cluster_size', 'read_depth']])

    # # Update the data with the scaled values.
    # combined_data[['cluster_size', 'read_depth']] = robust_scaled
    # # Split the data back into true positives and false positives.
    # tp_data[['cluster_size', 'read_depth']] = combined_data[['cluster_size', 'read_depth']].iloc[:tp_data.shape[0]]
    # fp_data[['cluster_size', 'read_depth']] = combined_data[['cluster_size', 'read_depth']].iloc[tp_data.shape[0]:]

    # logging.info('Normalization completed.')

    # # Drop the cluster_size column
    # logging.info('Dropping the cluster_size column from the data.')
    # tp_data.drop(columns=['cluster_size'], inplace=True)
    # fp_data.drop(columns=['cluster_size'], inplace=True)

    # Plot the distributions of cluster_size in the TP vs. FP data.
    # logging.info('Plotting the distributions of cluster_size in the TP vs. FP data.')
    # plt.figure(figsize=(10, 6))
    # sns.histplot(tp_data['cluster_size'], color='blue', label='True Positives', kde=True, stat="density", bins=30)
    # sns.histplot(fp_data['cluster_size'], color='red', label='False Positives', kde=True, stat="density", bins=30)
    # plt.xlabel('Cluster Size')
    # plt.ylabel('Density')
    # plt.title('Distribution of Cluster Size (True Positives vs False Positives)')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_directory, 'cluster_size_distribution.png'))
    # plt.close()
    # logging.info('Cluster size distribution plot saved to %s', os.path.join(output_directory, 'cluster_size_distribution.png'))

    # Analyze feature correlations in the collected data.
    # logging.info('Analyzing feature correlations in the collected data.')
    # corr_matrix = tp_data.corr()
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    # plt.title('Feature Correlation Matrix (True Positives)')
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_directory, 'feature_correlation_tp.png'))
    # plt.close()
    # corr_matrix = fp_data.corr()
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    # plt.title('Feature Correlation Matrix (False Positives)')
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_directory, 'feature_correlation_fp.png'))
    # plt.close()
    # logging.info('Feature correlation analysis completed. TP saved to %s and FP saved to %s',
    #              os.path.join(output_directory, 'feature_correlation_tp.png'),
    #              os.path.join(output_directory, 'feature_correlation_fp.png'))
    
    # Analyze feature correlations in the combined data.
    # logging.info('Analyzing feature correlations in the combined data.')
    # combined_data = pd.concat([tp_data, fp_data])
    # corr_matrix_combined = combined_data.corr()
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(corr_matrix_combined, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    # plt.title('Feature Correlation Matrix (Combined Data)')
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_directory, 'feature_correlation_combined.png'))
    # plt.close()
    # logging.info('Feature correlation analysis completed for combined data. Saved to %s',
    #              os.path.join(output_directory, 'feature_correlation_combined.png'))

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

    logging.info('Number of true labels after balancing: %d', tp_data.shape[0])
    logging.info('Number of false labels after balancing: %d', fp_data.shape[0])

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
    data = pd.concat([tp_data, fp_data], ignore_index=True)  # Ignore the index to realign the indices.

    # Add interaction terms to the data.
    data = add_interaction_terms(data)

    # Drop the chromosome column from the data.
    # data.drop(columns=['chrom'], inplace=True)

    # Drop the chrom, start, end, sv_length, read_depth, and cluster_size
    # columns
    # logging.info('Dropping the chrom, start, end, sv_length, read_depth, and cluster_size columns from the data.')
    # data.drop(columns=['chrom', 'start', 'end', 'sv_length', 'read_depth',
    # 'cluster_size', 'sv_type_str'], inplace=True)
    data.drop(columns=['chrom', 'start', 'end', 'sv_type_str'], inplace=True)

    # Drop the SV type column (imbalance especially for inversions).
    # logging.info('Dropping the sv_type column from the data.')
    # data.drop(columns=['sv_type'], inplace=True)

    # Normalize cluster_size and read_depth
    # data = normalize_column(data, 'cluster_size')
    # data = normalize_column(data, 'read_depth')

    # Drop cluster_size
    # data.drop(columns=['cluster_size'], inplace=True)

    # # Drop the SV length column
    # data.drop(columns=['sv_length'], inplace=True)

    # Drop the read_depth and cluster_size columns
    data.drop(columns=['read_depth', 'cluster_size'], inplace=True)

    # Drop the hmm log likelihood column
    # data.drop(columns=['hmm_llh'], inplace=True)

    # Drop the SV length and aln_type columns.
    # data.drop(columns=['sv_length', 'aln_type'], inplace=True)

    logging.info('Columns list after preprocessing: %s', data.columns.tolist())

    # Print duplicate columns if any.
    duplicate_columns = data.columns[data.columns.duplicated()].tolist()
    logging.info('Duplicate columns found: %s', duplicate_columns)

    # Get the features and labels.
    features = data.drop(columns=['label'])
    labels = data["label"]
     
    # Print the number of features.
    logging.info('Number of features: %d', features.shape[1])
    logging.info('Feature names: %s', features.columns.tolist())

    # Train different models.
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', enable_categorical=True),
        "SVC": SVC(kernel='linear', class_weight='balanced', probability=True)
    }

    # Split the data into training and testing sets.
    logging.info('Splitting the data into training and testing sets (0.8/0.2).')
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    logging.info('Data split completed. Training set size: %d, Testing set size: %d',
                 X_train.shape[0], X_test.shape[0])

    # Compute sample weights based on SV length.
    # svlen_weights = np.log1p(np.abs(X_train['sv_length']))
    # svlen_weights = np.log1p(np.abs(X_train['sv_length'])) ** 2  # Square the
    # weights to emphasize larger SV lengths.
    # conf = np.clip(np.exp(X_train['hmm_llh'] / 1000), 1e-6, 1)
    # # Replace NaN values in conf with 1.0
    # conf = conf.fillna(1.0)
    # svlen = np.log1p(np.abs(X_train['sv_length']) + 1e-6)  # Add a small value to avoid log(0)
    # svlen_weights = conf * svlen

    #     sv_type_map = {
    #     'DEL': 0,
    #     'DUP': 1,
    #     'INV': 2,
    #     'INS': 3,
    #     'BND': 4,
    #     'UNKNOWN': 5
    # }
    # Weights based on the SV type (weight inversions more heavily since they are
    # less common in the dataset).
    sv_type_weights = {
        0: 1.0,  # DEL
        1: 1.0,  # DUP
        2: 5.0,  # INV
        3: 1.0,  # INS
        4: 1.0,  # BND
        5: 1.0   # UNKNOWN
    }
    # Create a sample weight array based on the SV type.
    # sample_weights = np.array([sv_type_weights.get(sv_type, 1.0) for sv_type in X_train['sv_type']])

    # Split the data into 1/2 >10kb abs(sv_length) and 1/2 <10kb abs(sv_length),
    # then train the model with 80% of the data and test with 20% of the data.
    # logging.info('Splitting the data by size of SVs (1/2 >10kb abs(sv_length) and 1/2 <10kb abs(sv_length)).')
    # large_sv_mask = features['sv_length'].abs() > 10000
    # small_sv_mask = features['sv_length'].abs() <= 10000
    
    
    # # Drop the chromosome column from the features.
    # X_train.drop(columns=['chrom'], inplace=True)
    # X_test.drop(columns=['chrom'], inplace=True)

    # # Print the number of features.
    # logging.info('Number of features: %d', X_train.shape[1])
    # # Print the feature names.
    # feature_names = features.columns.tolist()
    # logging.info('Feature names: %s', feature_names)

    for model_name, model in models.items():
        model_name_fp = model_name.replace(" ", "_")

        # Skip SVC and logistic regression for now.
        if model_name == "SVC":
            logging.info('Skipping SVC model.')
            continue

        # if model_name == "Logistic Regression":
        #     logging.info('Skipping Logistic Regression model.')
        #     continue

        # Skip all but XGBoost
        if model_name != "XGBoost":
            logging.info('Skipping %s model.', model_name)
            continue

        # # Split the data into training and testing sets.
        # logging.info('Splitting the data into training and testing sets (0.8/0.2).')
        # X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        # Normalize the data since some features such as read depth and LRR vary across
        # different samples and can have different ranges.
        # from sklearn.preprocessing import StandardScaler
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)
        # logging.info('Data split and scaled. Training set size: %d, Testing set size: %d',
        #              X_train.shape[0], X_test.shape[0])
        
        # Print the number of features.
        logging.info('Number of features: %d', X_train.shape[1])
        logging.info('Training set size: %d', X_train.shape[0])
        logging.info('Testing set size: %d', X_test.shape[0])

        # Print the feature names.
        feature_names = features.columns.tolist()
        logging.info('Feature names: %s', feature_names)
        logging.info('Number of features: %d', len(feature_names))

        # Print the number of true positives and false positives in the training
        # and testing sets.
        logging.info('Number of true positives in the training set: %d', np.sum(y_train == 1))
        logging.info('Number of false positives in the training set: %d', np.sum(y_train == 0))
        logging.info('Number of true positives in the testing set: %d', np.sum(y_test == 1))
        logging.info('Number of false positives in the testing set: %d', np.sum(y_test == 0))
        logging.info('Training set size: %d', X_train.shape[0])
        logging.info('Testing set size: %d', X_test.shape[0])
        logging.info('Number of features: %d', X_train.shape[1])

        # If SVC, scale the data.
        # if model_name == "SVC":
        #     from sklearn.preprocessing import StandardScaler
        #     scaler = StandardScaler()
        #     X_train = scaler.fit_transform(X_train)
        #     X_test = scaler.transform(X_test)

        # Train the model.
        logging.info('Training the %s model.', model_name)
        # model.fit(features, labels)
        model.fit(X_train, y_train)
        # model.fit(X_train, y_train, sample_weight=sample_weights)

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
        # j_scores = tpr_test - fpr_test
        # optimal_idx = np.argmax(j_scores)
        # optimal_threshold = thresholds[optimal_idx]
        # logging.info('Optimal threshold (Youden\'s J statistic): %f', optimal_threshold)
        # logging.info('True positive rate (sensitivity): %f', tpr_test[optimal_idx])
        # logging.info('False positive rate (1 - specificity): %f', fpr_test[optimal_idx])

        # Print the ROC AUC scores.
        logging.info('ROC AUC score for the training set: %f', roc_auc_train)
        logging.info('ROC AUC score for the testing set: %f', roc_auc_test)

        # # Plot the ROC curve for the training set.
        # plt.figure()
        # plt.plot(fpr_train, tpr_train, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_train)
        # # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('{} Receiver Operating Characteristic (Training Set)'.format(model_name))
        # plt.legend(loc='lower right')
        # # Save the plot to the output directory.
        # roc_plot_path = os.path.join(output_directory, model_name_fp + '_roc_curve.png')
        # plt.savefig(roc_plot_path)
        # plt.close()
        # logging.info('Saved the ROC curve to %s', roc_plot_path)

        # # Plot the ROC curve for the testing set.
        # plt.figure()
        # plt.plot(fpr_test, tpr_test, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_test)
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('{} Receiver Operating Characteristic (Testing Set)'.format(model_name_fp))
        # plt.legend(loc='lower right')
        # # Save the plot to the output directory.
        # roc_plot_path = os.path.join(output_directory, model_name + '_roc_curve_test.png')
        # plt.savefig(roc_plot_path)
        # plt.close()
        # logging.info('Saved the ROC curve to %s', roc_plot_path)

        # Compute precision-recall curve
        # precision, recall, thresholds_pr = precision_recall_curve(y_test, y_test_prob)

        # logging.info('precision size: %d', len(precision))
        # logging.info('recall size: %d', len(recall))
        # logging.info('thresholds size: %d', len(thresholds_pr))

        # # Plot Recall vs Thresholds
        # plt.figure()
        # plt.plot(thresholds_pr, recall[1:], color='blue', lw=2, label='Recall')
        # plt.xlabel('Threshold')
        # plt.ylabel('Recall')
        # plt.title('%s Recall vs Thresholds' % model_name)
        # # plt.legend(loc='lower right')

        # # Remove the legend
        # plt.legend().remove()

        # # Save the plot to the output directory.
        # recall_plot_path = os.path.join(output_directory, model_name_fp + '_recall_vs_thresholds.png')
        # plt.savefig(recall_plot_path)
        # plt.close()
        # logging.info('Saved the Recall vs Thresholds plot to %s', recall_plot_path)

        # # Plot Precision vs Thresholds
        # plt.figure()
        # plt.plot(thresholds_pr, precision[1:], color='blue', lw=2, label='Precision')
        # plt.xlabel('Threshold')
        # plt.ylabel('Precision')
        # plt.title('%s Precision vs Thresholds' % model_name)
        # # plt.legend(loc='lower right')
        # # Remove the legend
        # plt.legend().remove()
        # # Save the plot to the output directory.
        # precision_plot_path = os.path.join(output_directory, model_name_fp + '_precision_vs_thresholds.png')
        # plt.savefig(precision_plot_path)
        # plt.close()
        # logging.info('Saved the Precision vs Thresholds plot to %s', precision_plot_path)

        # Get the feature names.
        feature_names = features.columns.tolist()

        # Create a dictionary of feature names and their labels.
        feature_name_dict = {
            "aln_type": "Alignment Type",
            "aln_type_hmm": "HMM Prediction",
            "simpleRepeat": "Simple Repeat",
            "segdup": "Segmental Duplications",
            "cluster_size": "Cluster Size",
            "read_depth": "Read Depth",
            "aln_offset": "Alignment Offset",
            "hmm_llh": "HMM Log Likelihood",
            "phastCons": "PhastCons Conservation Score",
            "sv_length": "Structural Variant Length",
            "sv_type": "Structural Variant Type",
            "fragile_site": "Fragile Site",
            "centromere": "Centromere",
            "telomere": "Telomere"
        }

        # Map the feature names to their labels.
        feature_names = [feature_name_dict.get(name, name) for name in feature_names]

        logging.info('Feature names: %s', feature_names)
        logging.info('Number of features: %d', len(feature_names))

        # Feature importance for Random_Forest and XGBoost.
        if model_name in ["Random_Forest", "XGBoost"]:
            # Get feature importances.
            importances = model.feature_importances_

            # Sort the feature importances in descending order.
            indices = np.argsort(importances)[::-1]
            top_features = [feature_names[i] for i in indices]
            top_importances = [importances[i] for i in indices]

            # Print the feature ranking.
            logging.info('Feature ranking:')
            for f in range(X_train.shape[1]):
                logging.info('%d. Feature %s (%f)', f + 1, feature_names[indices[f]], importances[indices[f]])

            # Plot the feature importances.
            plt.figure()
            plt.title('XGBoost Feature Importances')
            plt.bar(range(len(top_features)), top_importances, align='center')
            plt.xticks(range(len(top_features)), top_features, rotation=45, ha='right')
            # plt.bar(range(X_train.shape[1]), importances[indices], align='center')
            # plt.xticks(range(X_train.shape[1]), indices)
            # plt.xlim([-1, X_train.shape[1]])

            # Set the x ticks as the feature names
            # plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=45)
            # plt.xlim([-1, X_train.shape[1]])
            plt.xlabel('')
            plt.ylabel('Importance')
            plt.tight_layout()

            # Save the plot to the output directory.
            importance_plot_path = os.path.join(output_directory, model_name_fp + '_feature_importances.png')
            plt.savefig(importance_plot_path, bbox_inches='tight')
            plt.close()
            logging.info('Saved the feature importances plot to %s', importance_plot_path)

            # Plot the % of SVs (TPs and FPs) overlapping with the genomic
            # context regions (simpleRepeat, segdup, fragile_site, phastCons >
            # 0.5)
            # print("Number of TPs: ", tp_data.shape[0])
            # print("Number of FPs: ", fp_data.shape[0])
            # logging.info('Plotting the percentage of SVs (TPs and FPs) overlapping with the genomic context regions.')
            # for feature in ['simpleRepeat', 'segdup', 'fragile_site', 'phastCons']:
            #     if feature == 'phastCons':
            #         tp_data_feature = tp_data[tp_data[feature] > 0.5]
            #         fp_data_feature = fp_data[fp_data[feature] > 0.5]
            #     else:
            #         tp_data_feature = tp_data[tp_data[feature] == 1]
            #         fp_data_feature = fp_data[fp_data[feature] == 1]
            #     tp_pcnt = tp_data_feature.shape[0] / tp_data.shape[0] * 100
            #     fp_pcnt = fp_data_feature.shape[0] / fp_data.shape[0] * 100
            #     logging.info('Feature %s: TP = %.2f%%, FP = %.2f%%', feature, tp_pcnt, fp_pcnt)
            #     plt.figure()
            #     plt.bar(['TP', 'FP'], [tp_pcnt, fp_pcnt], color=['#0072B2', '#D55E00'])  # Blue, Vermillion (colorblind-friendly)
            #     plt.xlabel('SV Type')
            #     plt.ylabel('Percentage of SVs')
            #     plt.title('Percentage of SVs Overlapping with %s' % feature)
            #     plt.ylim([0, 100])
            #     # Save the plot to the output directory.
            #     feature_plot_path = os.path.join(output_directory, model_name_fp + '_%s.png' % feature)
            #     plt.savefig(feature_plot_path, bbox_inches='tight')
            #     plt.close()
            #     logging.info('Saved the %s plot to %s', feature, feature_plot_path)

            # # Exit early to verify the feature importances.
            # sys.exit(0)

            # Convert bool columns to int for SHAP analysis.
            # bool_cols = X_train.select_dtypes(include=['bool']).columns
            # X_train[bool_cols] = X_train[bool_cols].astype(int)

            # Figure out which column has dtype object in X_train.
            # print("X_train dtypes:")
            # print(X_train.dtypes)
            # print("X_train columns:")
            # print(X_train.columns)

            # Analyze the feature importances using SHAP values.
            # import shap
            # explainer = shap.Explainer(model, X_train)
            # shap_values = explainer(X_train)
            # # Plot the SHAP values.
            # plt.figure(figsize=(10, 6))
            # shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
            # # Save the SHAP summary plot to the output directory.
            # shap_plot_path = os.path.join(output_directory, model_name_fp + '_shap_summary_plot.png')
            # plt.savefig(shap_plot_path, bbox_inches='tight')
            # plt.close()
            # logging.info('Saved the SHAP summary plot to %s', shap_plot_path)

            # -----------------------------------------------
            # SV Length vs SHAP values
            # -----------------------------------------------

            # # Plot 1: SHAP values vs SV length.
            # plt.figure(figsize=(10, 6))
            # sns.scatterplot(data=X_train, x='abs_SVLEN', y='shap_SVLEN', hue='true_label', alpha=0.6)
            # plt.xscale('log')
            # plt.xlabel("SV Length (bp, log scale)")
            # plt.ylabel("SHAP value for SV Length")
            # plt.title("SHAP value vs. SV length")
            # plt.axhline(0, color='gray', linestyle='--')
            # plt.legend(title="True Label")
            # plt.tight_layout()
            # shap_svlen_plot_path = os.path.join(output_directory, model_name_fp + '_shap_svlen.png')
            # plt.savefig(shap_svlen_plot_path, bbox_inches='tight')
            # plt.close()
            # logging.info('Saved the SHAP value vs. SV length plot to %s', shap_svlen_plot_path)

            # # Plot 2: Predicted probability vs SV length.
            # X_train['y_prob'] = model.predict_proba(X_train)[:, 1]
            # plt.figure(figsize=(10, 6))
            # sns.scatterplot(data=X_train, x='abs_SVLEN', y='y_prob', hue='true_label', alpha=0.6)
            # plt.xscale('log')
            # plt.xlabel("SV Length (bp, log scale)")
            # plt.ylabel("Predicted Probability of Being True Positive")
            # plt.title("Predicted Probability vs. SV length")
            # plt.axhline(0.5, color='gray', linestyle='--')
            # plt.legend(title="True Label")
            # plt.tight_layout()
            # prob_svlen_plot_path = os.path.join(output_directory, model_name_fp + '_prob_svlen.png')
            # plt.savefig(prob_svlen_plot_path, bbox_inches='tight')
            # plt.close()
            # logging.info('Saved the predicted probability vs. SV length plot to %s', prob_svlen_plot_path)

            # plt.title('Feature Importances')

            # [TEST] Exit after this step to verify the feature importances.
            # sys.exit(0)

        # For SVC, get the coefficients.
        # if model_name == "SVC":
        #     # Get the coefficients.
        #     coefficients = model.coef_[0]

        #     # Sort the coefficients in descending order.
        #     indices = np.argsort(coefficients)[::-1]

        #     # Print the feature ranking.
        #     logging.info('Feature ranking:')
        #     for f in range(X_train.shape[1]):
        #         logging.info('%d. Feature %s (%f)', f + 1, feature_names[indices[f]], coefficients[indices[f]])

        #     # Plot the coefficients.
        #     plt.figure()
        #     plt.title('Feature Coefficients')
        #     plt.bar(range(X_train.shape[1]), coefficients[indices], align='center')
        #     plt.xticks(range(X_train.shape[1]), indices)
        #     plt.xlim([-1, X_train.shape[1]])
        #     # Save the plot to the output directory.
        #     coeff_plot_path = os.path.join(output_directory, model_name + '_feature_coefficients.png')
        #     plt.savefig(coeff_plot_path)
        #     plt.close()
        #     logging.info('Saved the feature coefficients plot to %s', coeff_plot_path)

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
            coeff_plot_path = os.path.join(output_directory, model_name_fp + '_feature_coefficients.png')
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
        model_path = os.path.join(output_directory, model_name_fp + '_caller_model.pkl')
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

    # Exit early to test the saved model.
    sys.exit(0)

    # Run a cross-validation analysis splitting the data by chromosome.
    logging.info('Running cross-validation analysis splitting the data by chromosome.')
    # chromosomes = features['chrom'].unique()

    # Specify the chromosomes to not include non-standard chromosomes.
    chromosomes = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
                  'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY']
    
    logging.info('Chromosomes: %s', chromosomes)
    f1_scores = {}
    for model_name, model in models.items():
        # Skip SVC
        if model_name == "SVC":
            logging.info('Skipping SVC model for cross-validation analysis.')
            continue

        for chrom in chromosomes:
            logging.info('Training the %s model on chromosome %s.', model_name, chrom)
            # Split the data into training and testing sets by chromosome.
            X_train_chrom = features[features['chrom'] != chrom].copy()
            y_train_chrom = labels[features['chrom'] != chrom].copy()
            X_test_chrom = features[features['chrom'] == chrom].copy()
            y_test_chrom = labels[features['chrom'] == chrom].copy()

            # Drop the chromosome column from the features.
            X_train_chrom.drop(columns=['chrom'], inplace=True)
            X_test_chrom.drop(columns=['chrom'], inplace=True)

            logging.info('Training set size: %d, Testing set size: %d',
                         X_train_chrom.shape[0], X_test_chrom.shape[0])
            # Train the model.
            model.fit(X_train_chrom, y_train_chrom)
            # Get the predicted probabilities for the testing set.
            y_test_chrom_prob = model.predict_proba(X_test_chrom)[:, 1]
            # Compute the ROC curve and ROC area for the testing set.
            fpr_chrom, tpr_chrom, _ = roc_curve(y_test_chrom, y_test_chrom_prob)
            roc_auc_chrom = auc(fpr_chrom, tpr_chrom)
            logging.info('ROC AUC score for the %s model on chromosome %s: %f', model_name, chrom, roc_auc_chrom)

            # Compute the F1 score for the testing set.
            from sklearn.metrics import f1_score
            y_test_chrom_pred = (y_test_chrom_prob >= optimal_threshold).astype(int)
            f1 = f1_score(y_test_chrom, y_test_chrom_pred)
            # f1_scores.append(f1)
            f1_scores[(model_name, chrom)] = f1
            logging.info('F1 score for the %s model on chromosome %s: %f', model_name, chrom, f1)
            
    logging.info('Cross-validation analysis completed. F1 scores: %s', f1_scores)

    # Plot the F1 scores for each model and chromosome (one plot per model).
    logging.info('Plotting the F1 scores for each model and chromosome.')
    for model_name in models.keys():
        model_f1_scores = {chrom: f1_scores[(model_name, chrom)] for chrom in chromosomes if (model_name, chrom) in f1_scores}
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(model_f1_scores.keys()), y=list(model_f1_scores.values()), palette='viridis')
        plt.xlabel('Chromosome')
        plt.ylabel('F1 Score')
        plt.title('F1 Scores for %s Model by Chromosome' % model_name)
        plt.xticks(rotation=45)
        plt.tight_layout()
        # Save the plot to the output directory.
        f1_plot_path = os.path.join(output_directory, model_name + '_f1_scores_by_chromosome.png')
        plt.savefig(f1_plot_path)
        plt.close()
        logging.info('Saved the F1 scores plot to %s', f1_plot_path)

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
