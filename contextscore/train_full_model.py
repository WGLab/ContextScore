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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from extract_features import extract_features, add_interaction_terms, normalize_column

# Set up the logger.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def balance_tp_fp_datasets(tp_data, fp_data):
    """Balance the true positive and false positive datasets by undersampling the lower-count class."""
    tp_count = tp_data.shape[0]
    fp_count = fp_data.shape[0]

    if tp_count > fp_count:
        logging.info('Balancing the dataset by undersampling the true positives (count = %d) to match the false positives (count = %d)', tp_count, fp_count)
        tp_data = tp_data.sample(fp_count, random_state=42)
    elif fp_count > tp_count:
        logging.info('Balancing the dataset by undersampling the false positives (count = %d) to match the true positives (count = %d)', fp_count, tp_count)
        fp_data = fp_data.sample(tp_count, random_state=42)
    else:
        logging.info('The dataset is already balanced. True positives: %d, False positives: %d', tp_count, fp_count)

    return tp_data, fp_data

def train(tp_hg002_grch37, fp_hg002_grch37, tp_visor_grch38, fp_visor_grch38, tp_platinum_grch38, fp_platinum_grch38, output_directory, annovar_path, db_path, outdiranno, leave_out="none", split_80_20=False):
    """Train the binary classification model."""

    # ---------------------------------------------------------------
    # SV Feature Extraction
    # ---------------------------------------------------------------

    # Set paths to none if leave_out is set to the corresponding dataset
    if leave_out == "hg002":
        logging.info('Leaving out HG002 dataset from training.')
        tp_hg002_grch37 = None
        fp_hg002_grch37 = None
    elif leave_out == "visor":
        logging.info('Leaving out Visor dataset from training.')
        tp_visor_grch38 = None
        fp_visor_grch38 = None
    elif leave_out == "platinum":
        logging.info('Leaving out Platinum Pedigree dataset from training.')
        tp_platinum_grch38 = None
        fp_platinum_grch38 = None

    # ===============================================================
    # Extract the features from the VCF files.
    # ===============================================================
    # GRCh38 data.
    logging.info('Extracting features from the true positive and false positive VCF files (GRCh38).')
    buildversion = 'hg38'
    tp_visor_anno = extract_features(tp_visor_grch38, annovar_path, db_path, os.path.join(outdiranno, "tp_anno_grch38"), buildversion=buildversion) if tp_visor_grch38 is not None else None
    fp_visor_anno = extract_features(fp_visor_grch38, annovar_path, db_path, os.path.join(outdiranno, "fp_anno_grch38"), buildversion=buildversion) if fp_visor_grch38 is not None else None
    # Balance datasets before concatenation. This is important to prevent the model from being biased towards the class with more samples.
    # if tp_visor_anno is not None and fp_visor_anno is not None:
        # tp_visor_anno, fp_visor_anno = balance_tp_fp_datasets(tp_visor_anno, fp_visor_anno)
    
    tp_platinum_anno = extract_features(tp_platinum_grch38, annovar_path, db_path, os.path.join(outdiranno, "tp_anno_grch38"), buildversion=buildversion) if tp_platinum_grch38 is not None else None
    fp_platinum_anno = extract_features(fp_platinum_grch38, annovar_path, db_path, os.path.join(outdiranno, "fp_anno_grch38"), buildversion=buildversion) if fp_platinum_grch38 is not None else None
    # Balance datasets before concatenation.
    # if tp_platinum_anno is not None and fp_platinum_anno is not None:
        # tp_platinum_anno, fp_platinum_anno = balance_tp_fp_datasets(tp_platinum_anno, fp_platinum_anno)

    # HG002 data (GRCh37).
    logging.info('Extracting features from the true positive and false positive VCF files (HG002-GRCh37).')
    buildversion = 'hg19'
    tp_hg002_anno = extract_features(tp_hg002_grch37, annovar_path, db_path, os.path.join(outdiranno, "tp_anno_grch37"), buildversion=buildversion) if tp_hg002_grch37 is not None else None
    fp_hg002_anno = extract_features(fp_hg002_grch37, annovar_path, db_path, os.path.join(outdiranno, "fp_anno_grch37"), buildversion=buildversion) if fp_hg002_grch37 is not None else None
    # Balance datasets before concatenation.
    # if tp_hg002_anno is not None and fp_hg002_anno is not None:
        # tp_hg002_anno, fp_hg002_anno = balance_tp_fp_datasets(tp_hg002_anno, fp_hg002_anno)

    # Concatenate the data from all datasets.
    logging.info('Concatenating the data from all datasets.')
    tp_data = pd.concat([df for df in [tp_visor_anno, tp_platinum_anno, tp_hg002_anno] if df is not None], ignore_index=True)
    fp_data = pd.concat([df for df in [fp_visor_anno, fp_platinum_anno, fp_hg002_anno] if df is not None], ignore_index=True)

    # Extract the features from the VCF files.
    # logging.info('Extracting features from the true positive and false positive VCF files (GRCh38).')
    # buildversion = 'hg38'
    # tp_anno_outdir = os.path.join(outdiranno, "tp_anno")
    # tp_data = extract_features(tp_bed, annovar_path, db_path, tp_anno_outdir, buildversion=buildversion)
    # logging.info('Extracted %d features from the true positive VCF file.', tp_data.shape[0])
    # fp_anno_outdir = os.path.join(outdiranno, "fp_anno")
    # fp_data = extract_features(fp_bed, annovar_path, db_path, fp_anno_outdir, buildversion=buildversion)
    # logging.info('Extracted %d features from the false positive VCF file.', fp_data.shape[0])

    # logging.info('Extracting features from the true positive and false positive VCF files (HG002-GRCh19).')
    # buildversion = 'hg19'
    # if tp_bed_hg19 is not None and fp_bed_hg19 is not None:
    #     tp_anno_outdir_hg19 = os.path.join(outdiranno, "tp_anno_hg19")
    #     tp_data_hg19 = extract_features(tp_bed_hg19, annovar_path, db_path, tp_anno_outdir_hg19, buildversion=buildversion)
    #     logging.info('Extracted %d features from the true positive VCF file (hg19).', tp_data_hg19.shape[0])
    #     fp_anno_outdir_hg19 = os.path.join(outdiranno, "fp_anno_hg19")
    #     fp_data_hg19 = extract_features(fp_bed_hg19, annovar_path, db_path, fp_anno_outdir_hg19, buildversion=buildversion)
    #     logging.info('Extracted %d features from the false positive VCF file (hg19).', fp_data_hg19.shape[0])

    #     # Concatenate the data from hg38 and hg19.
    #     logging.info('Concatenating the data from hg38 and hg19.')
    #     tp_data = pd.concat([tp_data, tp_data_hg19], ignore_index=True)
    #     fp_data = pd.concat([fp_data, fp_data_hg19], ignore_index=True)

    # else:
    #     logging.info('No hg19 data provided. Using only hg38 data.')
    # logging.info('Feature extraction completed. True positives: %d, False positives: %d',
    #              tp_data.shape[0], fp_data.shape[0])

    # ---------------------------------------------------------------
    # Data Preprocessing
    # ---------------------------------------------------------------

    # Remove duplicate rows from the concatenated data.
    tp_count_before = tp_data.shape[0]
    tp_data.drop_duplicates(inplace=True)
    tp_count_after = tp_data.shape[0]
    fp_count_before = fp_data.shape[0]
    fp_data.drop_duplicates(inplace=True)
    fp_count_after = fp_data.shape[0]
    logging.info('Removed %d tp duplicates and %d fp duplicates from the concatenated data. Remaining true positives: %d, remaining false positives: %d', tp_count_before - tp_count_after, fp_count_before - fp_count_after, tp_data.shape[0], fp_data.shape[0])

    # Perform robust scaling on the read_depth and cluster_size columns using
    # the RobustScaler from sklearn.
    logging.info('Normalizing read_depth and cluster_size using Robust scaling.')
    # First combine the data.
    combined_data = pd.concat([tp_data, fp_data], ignore_index=True)
    # Create a RobustScaler object.
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    # Fit the scaler to the data.
    robust_scaled = scaler.fit_transform(combined_data[['read_depth', 'cluster_size']])
    # Update the data with the scaled values.
    combined_data[['read_depth', 'cluster_size']] = robust_scaled
    # Split the data back into true positives and false positives.
    tp_data = combined_data.iloc[:tp_data.shape[0]]
    fp_data = combined_data.iloc[tp_data.shape[0]:]
    logging.info('Normalization completed. True positives: %d, False positives: %d', tp_data.shape[0], fp_data.shape[0])

    # Drop the genotype column from the data.
    logging.info('Dropping the genotype column from the data.')
    tp_data = tp_data.drop(columns=['genotype'], errors='ignore')
    fp_data = fp_data.drop(columns=['genotype'], errors='ignore')

    # Drop the cn_state column from the data.
    logging.info('Dropping the cn_state column from the data.')
    tp_data = tp_data.drop(columns=['cn_state'], errors='ignore')
    fp_data = fp_data.drop(columns=['cn_state'], errors='ignore')

    # Add the labels.
    tp_data['label'] = 1
    fp_data['label'] = 0

    # Print the number of true positives and false positives.
    logging.info('Number of true labels: %d', tp_data.shape[0])
    logging.info('Number of false labels: %d', fp_data.shape[0])

    # Drop NaN values from the data.
    logging.info('Dropping NaN values from the data.')
    tp_data = tp_data.dropna()
    fp_data = fp_data.dropna()
    logging.info('Number of true labels after dropping NaN values: %d', tp_data.shape[0])
    logging.info('Number of false labels after dropping NaN values: %d', fp_data.shape[0])

    # Balance the dataset by undersampling the true positives.
    # logging.info('Balancing the dataset by undersampling the true positives (count = %d) to match the false positives (count = %d)', tp_data.shape[0], fp_data.shape[0])
    # tp_data = tp_data.sample(fp_data.shape[0], random_state=42)

    # logging.info('Number of true labels after balancing: %d', tp_data.shape[0])
    # logging.info('Number of false labels after balancing: %d', fp_data.shape[0])

    # Combine the true positive and false positive data.
    data = pd.concat([tp_data, fp_data], ignore_index=True)  # Ignore the index to realign the indices.

    # Add interaction terms to the data.
    data = add_interaction_terms(data)

    # Drop columns not needed for training.
    # data.drop(columns=['chrom', 'start', 'end', 'sv_type_str'], inplace=True)

    # Pop the chrom column to use it later for cross-validation.
    chrom_col = data.pop('chrom')

    # Drop columns that are not needed for training.
    data = data.drop(columns=['start', 'end', 'sv_type_str'], errors='ignore')

    # Drop the read_depth and cluster_size columns
    # data = data.drop(columns=['read_depth', 'cluster_size'], errors='ignore')

    logging.info('Columns list after preprocessing: %s', data.columns.tolist())

    # Print duplicate columns if any.
    duplicate_columns = data.columns[data.columns.duplicated()].tolist()
    if duplicate_columns:
        logging.warning('Duplicate columns found: %s', duplicate_columns)

    # Get the features and labels.
    features = data.drop(columns=['label'])
    labels = data["label"]
     
    # Print the number of features.
    logging.info('Number of features: %d', features.shape[1])
    logging.info('Feature names: %s', features.columns.tolist())
    if split_80_20:
        # Split the data into training and testing sets using stratified sampling to maintain the class balance.
        logging.info('Splitting the data into training and testing sets using an 80-20 split with stratified sampling to maintain class balance.')
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)
    else:
        # Use all the data for training and testing. We will use cross-validation to evaluate the model performance.
        logging.info('Using all the data for training and testing. Cross-validation will be used to evaluate the model performance.')
        X_train, y_train = features, labels
        X_test, y_test = features, labels

    # If not 80/20 split, use XGBoost only (highest performing model) to save time.
    if split_80_20:
        pipelines = {
            "Logistic_Regression": Pipeline([('classifier', LogisticRegression(max_iter=1000, random_state=42))]),
            "Random_Forest": Pipeline([('classifier', RandomForestClassifier(n_estimators=100, random_state=42))]),
            "XGBoost": Pipeline([('classifier', XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42, enable_categorical=True))])
        }
    else:
        pipelines = {
            "XGBoost": Pipeline([('classifier', XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42, enable_categorical=True))])
        }

    param_grids = {
        "Logistic_Regression": {
            'classifier__C': [0.01, 0.1, 1, 10],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear']
        },
        "Random_Forest": {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2]
        },
        "XGBoost": {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 6],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__subsample': [0.8, 1]
        }
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for model_name, pipeline in pipelines.items():
        logging.info('Training the %s model.', model_name)
        model_name_fp = "contextscore_" + model_name.lower() + "_leaveout_" + leave_out

        if split_80_20:
            model_name_fp += "_80_20_split"

        # Perform grid search to find the best hyperparameters for the model, optimizing for precision to prioritize reducing false positives.
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grids[model_name], cv=cv, scoring='precision', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        logging.info('Best hyperparameters for %s: %s', model_name, grid_search.best_params_)

        # Get predicted probabilities for the training and testing sets.
        best_model = grid_search.best_estimator_

        # Save plots only for 80-20 split since the ROC curve will be overly optimistic when using all the data for training and testing.
        if split_80_20:
            y_train_prob = best_model.predict_proba(X_train)[:, 1]
            y_test_prob = best_model.predict_proba(X_test)[:, 1]

            # Compute the ROC curve and ROC area for the training set.
            fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
            roc_auc_train = auc(fpr_train, tpr_train)

            # Compute the ROC curve and ROC area for the testing set.
            fpr_test, tpr_test, thresholds = roc_curve(y_test, y_test_prob)
            roc_auc_test = auc(fpr_test, tpr_test)

            # Print the ROC AUC scores.
            logging.info('ROC AUC score for the training set: %f', roc_auc_train)
            logging.info('ROC AUC score for the testing set: %f', roc_auc_test)

            # Plot the ROC curve for the training set.
            plt.figure()
            plt.plot(fpr_train, tpr_train, color='blue', lw=2, label='ROC curve (area = %0.3f)' % roc_auc_train)
            # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            model_name_label = model_name.replace("_", " ")
            plt.title('{} Receiver Operating Characteristic (Training Set)'.format(model_name_label))
            plt.legend(loc='lower right')
            # Save the plot to the output directory.
            roc_plot_path = os.path.join(output_directory, model_name_fp + '_roc_curve.png')
            plt.savefig(roc_plot_path)
            plt.close()
            logging.info('Saved the ROC curve to %s', roc_plot_path)

            # Plot the ROC curve for the testing set.
            plt.figure()
            plt.plot(fpr_test, tpr_test, color='blue', lw=2, label='ROC curve (area = %0.3f)' % roc_auc_test)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('{} Receiver Operating Characteristic (Testing Set)'.format(model_name_label))
            plt.legend(loc='lower right')
            # Save the plot to the output directory.
            roc_plot_path = os.path.join(output_directory, model_name + '_roc_curve_test.png')
            plt.savefig(roc_plot_path)
            plt.close()
            logging.info('Saved the ROC curve to %s', roc_plot_path)
        else:
            # Save the model to the output directory as a pickle file.
            model_path = os.path.join(output_directory, model_name_fp + '_model.pkl')
            joblib.dump(best_model, model_path)
            logging.info('Saved the %s model to %s', model_name, model_path)


        # Continue if not running SHAP analysis.
        logging.info('Completed training and evaluation for %s. Continuing to the next model.', model_name)
        continue

        # Feature importance for Random_Forest and XGBoost
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
            "telomere": "Telomere",
            "call_type": "Alignment Type",
            "simple_repeat_cs": "Simple Repeat x Cluster Size",
            "simple_repeat_rd": "Simple Repeat x Read Depth",
            "cs_hmm": "Cluster Size x HMM LLH",
            "fragile_site_cs": "Fragile Site x Cluster Size",
            "fragile_site_rd": "Fragile Site x Read Depth",
            "segdup_cs": "Seg. Dup. x Cluster Size",
            "segdup_rd": "Seg. Dup. x Read Depth"
        }

        # Map the feature names to their labels.
        feature_names = [feature_name_dict.get(name, name) for name in feature_names]

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
            bool_cols = X_train.select_dtypes(include=['bool']).columns
            X_train[bool_cols] = X_train[bool_cols].astype(int)

            # Analyze the feature importances using SHAP values.
            import shap
            # explainer = shap.Explainer(model, X_train)
            # shap_values = explainer(X_train)

            # SHAP doesn't support XGBoost with categorical features directly,
            # so we need to use their suggested workaround.
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            shap_values = explainer.shap_values(X_train)

            # Plot the SHAP values.
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
            # Save the SHAP summary plot to the output directory.
            shap_plot_path = os.path.join(output_directory, model_name_fp + '_shap_summary_plot.png')
            plt.savefig(shap_plot_path, bbox_inches='tight')
            plt.close()
            logging.info('Saved the SHAP summary plot to %s', shap_plot_path)

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

    # Exit early if not running per-chromosome cross-validation analysis.
    return

    # Run a cross-validation analysis splitting the data by chromosome.
    logging.info('Running cross-validation analysis splitting the data by chromosome.')
    # chromosomes = features['chrom'].unique()

    # Specify the chromosomes to not include non-standard chromosomes.
    # chromosomes = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
    #               'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY']
    
    # 4 August 2025: Remove chrY from the analysis. More than half is missing in
    # GRCh38 and leads to high false positive rates.
    chromosomes = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
                  'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']

    logging.info('Chromosomes: %s', chromosomes)
    f1_scores = {}
    precision_scores = {}
    recall_scores = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipelines = {
        "Logistic Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression())
        ]),
        "Random_Forest": Pipeline([
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        "XGBoost": Pipeline([
            ('classifier', XGBClassifier(eval_metric='logloss', enable_categorical=True))
        ]),
        "SVC": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='linear', class_weight='balanced', probability=True))
        ])
    }

    # =================================================================
    # Hyperparameter grids
    # =================================================================
    param_grids = {
        "Logistic Regression": {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear']
        },
        "Random_Forest": {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5, 10]
        },
        "XGBoost": {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 6, 9],
            'classifier__learning_rate': [0.1, 0.2, 0.3]
        },
        "SVC": {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__gamma': ['scale', 'auto']
        }
    }

    for name in pipelines.keys():
        logging.info(f"\n=============================")
        logging.info(f"Training pipeline: {name}")
        logging.info(f"==============================\n")
        pipe = pipelines[name]
        grid = param_grids[name]
        grid_search = GridSearchCV(pipe, grid, cv=cv, scoring='f1', n_jobs=-1)
        grid_search.fit(features, labels)
        logging.info(f"Best parameters for {name}: {grid_search.best_params_}")
        # # Skip SVC
        # if model_name == "SVC":
        #     logging.info('Skipping SVC model for cross-validation analysis.')
        #     continue

        # # Skip all but XGBoost
        # if model_name != "XGBoost":
        #     logging.info('Skipping %s model for cross-validation analysis.', model_name)
        #     continue

        # Dictionary with number of SVs in the training set for each chromosome.
        sv_counts = {chrom: features[chrom_col == chrom].shape[0] for chrom in chromosomes}
        logging.info('Number of SVs in the training set for each chromosome: %s', sv_counts)

        for chrom in chromosomes:
            logging.info('Training the %s model on chromosome %s.', model_name, chrom)
            # Split the data into training and testing sets by chromosome.
            X_train_chrom = features[chrom_col != chrom].copy()
            y_train_chrom = labels[chrom_col != chrom].copy()
            X_test_chrom = features[chrom_col == chrom].copy()
            y_test_chrom = labels[chrom_col == chrom].copy()

            # Drop the chromosome column from the features.
            # X_train_chrom.drop(columns=['chrom'], inplace=True)
            # X_test_chrom.drop(columns=['chrom'], inplace=True)

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
            y_test_chrom_pred = (y_test_chrom_prob >= 0.5).astype(int)  # Use a threshold of 0.5 for classification.
            f1 = f1_score(y_test_chrom, y_test_chrom_pred)
            f1_scores[(model_name, chrom)] = f1
            logging.info('F1 score for the %s model on chromosome %s: %f', model_name, chrom, f1)

            # Compute precision and recall for the testing set.
            from sklearn.metrics import precision_score, recall_score
            precision = precision_score(y_test_chrom, y_test_chrom_pred)
            recall = recall_score(y_test_chrom, y_test_chrom_pred)
            precision_scores[(model_name, chrom)] = precision
            recall_scores[(model_name, chrom)] = recall
            logging.info('Precision for the %s model on chromosome %s: %f', model_name, chrom, precision)
            logging.info('Recall for the %s model on chromosome %s: %f', model_name, chrom, recall)

            # Compute the F1 score for the testing set.
            # from sklearn.metrics import f1_score
            # y_test_chrom_pred = (y_test_chrom_prob >= optimal_threshold).astype(int)
            # f1 = f1_score(y_test_chrom, y_test_chrom_pred)
            # # f1_scores.append(f1)
            # f1_scores[(model_name, chrom)] = f1
            # logging.info('F1 score for the %s model on chromosome %s: %f', model_name, chrom, f1)
            
    logging.info('Cross-validation analysis completed. F1 scores: %s', f1_scores)

    # Plot the F1 scores for each model and chromosome (one plot per model).
    logging.info('Plotting the scores for each model and chromosome.')
    metrics = ['F1 Score', 'Precision', 'Recall']
    for model_name in models.keys():
        # Skip if not XGBoost
        if model_name != "XGBoost":
            logging.info('Skipping %s model for plotting scores by chromosome.', model_name)
            continue

        # Save a plot with F1, Precision, and Recall scores for chrY
        if 'chrY' in chromosomes:
            logging.info('Plotting scores for %s model on chrY.', model_name)
            # Create a bar plot for the F1 scores by chromosome.
            chry_f1 = f1_scores.get((model_name, 'chrY'), 0)
            chry_precision = precision_scores.get((model_name, 'chrY'), 0)
            chry_recall = recall_scores.get((model_name, 'chrY'), 0)

            # plt.figure(figsize=(10, 6))

            # Make it way smaller for better visibility.
            plt.figure(figsize=(6, 4))

            # Plot F1, Precision, and Recall scores for chrY.
            sns.barplot(x=['F1 Score', 'Precision', 'Recall'], y=[chry_f1, chry_precision, chry_recall], color='black')

            # plt.xlabel('Metric')
            plt.ylabel('Score')
            plt.title('%s Scores for %s Model on chrY' % (model_name, model_name))
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            # Save the plot to the output directory.
            score_plot_path = os.path.join(output_directory, model_name + '_scores_chrY.png')
            plt.savefig(score_plot_path)
            plt.close()
            logging.info('Saved the scores plot for chrY to %s', score_plot_path)


        for metric, scores in zip(metrics, [f1_scores, precision_scores, recall_scores]):
            logging.info('Plotting %s for %s model by chromosome.', metric, model_name)
            # Create a bar plot for the F1 scores by chromosome.
            # model_f1_scores = {chrom: f1_scores[(model_name, chrom)] for chrom
            # in chromosomes if (model_name, chrom) in f1_scores}
            model_scores = {chrom: scores[(model_name, chrom)] for chrom in chromosomes if (model_name, chrom) in scores}

            plt.figure(figsize=(10, 6))
            # Smaller figure size for better visibility.
            # plt.figure(figsize=(8, 5))
            ax = sns.barplot(x=list(model_scores.keys()), y=list(model_scores.values()), color='black')

            # Annotate each bar with the number of SVs in the training set for that
            # chromosome.
            # Put the number of SVs above each bar.
            # for i, (chrom, score) in enumerate(model_scores.items()):
            #     num_sv = sv_counts[chrom]
            #     ax.text(i, score + 0.01, f'{num_sv}', ha='center', va='bottom', fontsize=8)

            plt.xlabel('Chromosome')
            plt.ylabel(metric)
            plt.title('%s for %s Model by Chromosome' % (metric, model_name))
            plt.xticks(rotation=45)
            plt.tight_layout()
            # Save the plot to the output directory.
            score_plot_path = os.path.join(output_directory, model_name + '_%s_by_chromosome.png' % metric.lower().replace(' ', '_'))
            plt.savefig(score_plot_path)
            plt.close()
            logging.info('Saved the %s plot to %s', metric, score_plot_path)

if __name__ == '__main__':
    # Parse the command line arguments.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp_hg002_grch37", required=True, help="Path to the true positive BED file for HG002 in GRCh37")
    parser.add_argument("--fp_hg002_grch37", required=True, help="Path to the false positive BED file for HG002 in GRCh37")
    parser.add_argument("--tp_visor_grch38", required=True, help="Path to the true positive BED file for Visor in GRCh38")
    parser.add_argument("--fp_visor_grch38", required=True, help="Path to the false positive BED file for Visor in GRCh38")
    parser.add_argument("--tp_platinum_grch38", required=True, help="Path to the true positive BED file for Platinum in GRCh38")
    parser.add_argument("--fp_platinum_grch38", required=True, help="Path to the false positive BED file for Platinum in GRCh38")
    parser.add_argument("--outdiranno", required=True, help="Output directory for saving the ANNOVAR annotations")
    parser.add_argument("--outdir", required=True, help="Output directory for saving the model")
    parser.add_argument("--annovar", required=True, help="Path to ANNOVAR")
    parser.add_argument("--annovar_db", required=True, help="Path to ANNOVAR database")
    parser.add_argument("--leave_out", required=True, help="Which dataset to leave out for training")
    parser.add_argument("--split_80_20", action='store_true', help="Whether to split the data into training and testing sets using an 80-20 split. If not specified, all the data will be used for training and testing, and cross-validation will be used to evaluate the model performance.")
    args = parser.parse_args()

    # Run the program.
    logging.info('Training the model, split_80_20 = %s.', args.split_80_20)
    train(args.tp_hg002_grch37, args.fp_hg002_grch37, args.tp_visor_grch38, args.fp_visor_grch38, args.tp_platinum_grch38, args.fp_platinum_grch38, args.outdir, args.annovar, args.annovar_db, args.outdiranno, args.leave_out, args.split_80_20)
    logging.info('done.')

