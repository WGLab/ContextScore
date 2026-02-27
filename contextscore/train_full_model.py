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
import logging
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Import SHAP for model interpretation.
import shap
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

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
    no_leave_out = False
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
    else:
        logging.info('Not leaving out any dataset from training.')
        no_leave_out = True

    # ===============================================================
    # Extract the features from the VCF files.
    # ===============================================================
    # GRCh38 data.
    logging.info('Extracting features from the true positive and false positive VCF files (GRCh38).')
    buildversion = 'hg38'
    tp_visor_anno = extract_features(tp_visor_grch38, annovar_path, db_path, os.path.join(outdiranno, "tp_anno_grch38"), buildversion=buildversion) if tp_visor_grch38 is not None else None
    fp_visor_anno = extract_features(fp_visor_grch38, annovar_path, db_path, os.path.join(outdiranno, "fp_anno_grch38"), buildversion=buildversion) if fp_visor_grch38 is not None else None
    
    tp_platinum_anno = extract_features(tp_platinum_grch38, annovar_path, db_path, os.path.join(outdiranno, "tp_anno_grch38"), buildversion=buildversion) if tp_platinum_grch38 is not None else None
    fp_platinum_anno = extract_features(fp_platinum_grch38, annovar_path, db_path, os.path.join(outdiranno, "fp_anno_grch38"), buildversion=buildversion) if fp_platinum_grch38 is not None else None

    # HG002 data (GRCh37).
    logging.info('Extracting features from the true positive and false positive VCF files (HG002-GRCh37).')
    buildversion = 'hg19'
    tp_hg002_anno = extract_features(tp_hg002_grch37, annovar_path, db_path, os.path.join(outdiranno, "tp_anno_grch37"), buildversion=buildversion) if tp_hg002_grch37 is not None else None
    fp_hg002_anno = extract_features(fp_hg002_grch37, annovar_path, db_path, os.path.join(outdiranno, "fp_anno_grch37"), buildversion=buildversion) if fp_hg002_grch37 is not None else None

    # Concatenate the data from all datasets.
    logging.info('Concatenating the data from all datasets.')
    tp_data = pd.concat([df for df in [tp_visor_anno, tp_platinum_anno, tp_hg002_anno] if df is not None], ignore_index=True)
    fp_data = pd.concat([df for df in [fp_visor_anno, fp_platinum_anno, fp_hg002_anno] if df is not None], ignore_index=True)

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

    # Drop SV length features since they are highly correlated with the SV type feature and may lead to overfitting.
    # logging.info('Dropping SV length feature from the data.')
    # tp_data = tp_data.drop(columns=['sv_length'], errors='ignore')
    # fp_data = fp_data.drop(columns=['sv_length'], errors='ignore')

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
            "XGBoost": Pipeline([('classifier', XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42, enable_categorical=False))])
        }
    else:
        pipelines = {
            "XGBoost": Pipeline([('classifier', XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42, enable_categorical=False))])
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
            'classifier__n_estimators': [150, 250],  # Slightly more trees
            'classifier__max_depth': [3, 6],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__subsample': [0.8, 1]
        }
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for model_name, pipeline in pipelines.items():
        logging.info('Training model class %s', model_name)
        model_name_fp = "contextscore_" + model_name.lower() + "_leaveout_" + leave_out

        if split_80_20:
            model_name_fp += "_80_20_split"

        # Perform grid search to find the best hyperparameters for the model, optimizing for precision to prioritize reducing false positives.
        # Convert categorical columns to numeric
        X_train_processed = X_train.copy()
        for col in X_train_processed.columns:
            if X_train_processed[col].dtype == 'category':
                X_train_processed[col] = X_train_processed[col].cat.codes
            elif X_train_processed[col].dtype == 'object':
                X_train_processed[col] = pd.to_numeric(X_train_processed[col], errors='coerce')

        X_train_processed = X_train_processed.fillna(0).astype('float64')

        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grids[model_name], cv=cv, scoring='precision', n_jobs=-1)
        grid_search.fit(X_train_processed, y_train)
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


        logging.info('Completed training and evaluation for %s model.', model_name)

        # Run SHAP if full analysis and no leave-outs (SHAP is slow)
        if not split_80_20 and no_leave_out:
            # SHAP doesn't support XGBoost with categorical features directly,
            # so we need to use their suggested workaround.
            classifier = best_model.named_steps['classifier']

            # Prepare numeric data for SHAP
            X_train_numeric = X_train.copy()
            for col in X_train_numeric.columns:
                if X_train_numeric[col].dtype == 'object':
                    X_train_numeric[col] = pd.to_numeric(X_train_numeric[col], errors='coerce')

            X_train_numeric = X_train_numeric.fillna(0).astype('float64')

            # Use a larger background sample to cover all tree leaves
            sample_size = min(5000, len(X_train_numeric))  # Larger sample
            X_background = shap.sample(X_train_numeric, sample_size, random_state=42)

            # Create explainer and calculate SHAP values
            explainer = shap.TreeExplainer(classifier, X_background)

            # Calculate SHAP values
            shap_values = explainer.shap_values(X_train_numeric)

            # 1. Summary plot (existing)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_train_numeric, show=False)
            shap_plot_path = os.path.join(output_directory, model_name_fp + '_shap_summary_plot.png')
            plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info('Saved the SHAP summary plot to %s', shap_plot_path)

            # 2. Bar plot showing mean absolute SHAP values (feature importance)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_train_numeric, plot_type="bar", show=False)
            bar_plot_path = os.path.join(output_directory, model_name_fp + '_shap_importance_plot.png')
            plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info('Saved the SHAP importance plot to %s', bar_plot_path)


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

