"""
train_model.py - Train the binary classification model and evaluate using per-chromosome cross-validation and 80/20 train/test split.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc

try:
    from .extract_features import extract_features
except ImportError:
    from extract_features import extract_features

# Set up the logger.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Manuscript-friendly display labels for model features.
FEATURE_DISPLAY_NAMES = {
    'dist_nearest_sv_per_kb': 'Nearest SV distance / kb',
    'cluster_size_per_kb': 'Cluster size / kb',
    'sv_length': 'SV length (bp)',
    'read_depth_normalized': 'Normalized depth',
    'segdup_left': 'SegDup overlap (left)',
    'segdup_right': 'SegDup overlap (right)',
    'dist_to_telomere': 'Telomere distance',
    'dist_to_centromere': 'Centromere distance',
    'call_type': 'Call evidence type',
    'simpleRepeat_left': 'Simple repeat (left)',
    'simpleRepeat_right': 'Simple repeat (right)',
    'sv_type': 'SV type',
    'repeat_span_density': 'Repeat span density',
    'fragile_site': 'Fragile-site overlap',
    'phastCons': 'phastCons score',
    'hmm_llh': 'HMM log-likelihood',
    'aln_offset': 'Alignment offset',
    'svlen_50_500': 'SV length 50-500bp',
    'svlen_500_5000': 'SV length 500-5,000bp',
    'svlen_5000_50000': 'SV length 5,000-50,000bp',
    'svlen_50000_plus': 'SV length ≥50,000bp',
}

ENABLE_SHAP = False

if ENABLE_SHAP:
    import shap

def get_display_feature_name(feature_name):
    """Map internal feature keys to human-readable labels for plots/tables."""
    return FEATURE_DISPLAY_NAMES.get(feature_name, feature_name.replace('_', ' '))


def get_display_feature_names(feature_names):
    """Return human-readable labels in the same order as input feature names."""
    return [get_display_feature_name(name) for name in feature_names]


def preprocess_feature_matrix(feature_df):
    """Convert mixed-type feature columns to numeric values for model fitting/inference."""
    processed_df = feature_df.copy()
    for col in processed_df.columns:
        if processed_df[col].dtype == 'category':
            processed_df[col] = processed_df[col].cat.codes
        elif processed_df[col].dtype == 'object':
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

    return processed_df.fillna(0).astype('float64')


def get_cv_splits(y, max_splits=5):
    """Choose a valid number of stratified CV folds for the provided labels."""
    class_counts = y.value_counts()
    if class_counts.empty or len(class_counts) < 2:
        return None

    n_splits = min(max_splits, int(class_counts.min()))
    if n_splits < 2:
        return None

    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

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


def impute_missing_values(tp_data, fp_data):
    """Impute missing values using TP-referenced statistics to avoid excessive row drops."""
    logging.info('Imputing NaN values using TP-referenced statistics...')

    # Report NaNs by column before imputation.
    tp_nan = tp_data.isna().sum()
    fp_nan = fp_data.isna().sum()
    tp_nan = tp_nan[tp_nan > 0].sort_values(ascending=False)
    fp_nan = fp_nan[fp_nan > 0].sort_values(ascending=False)
    if not tp_nan.empty:
        logging.info('TP NaN counts by column before imputation: %s', tp_nan.to_dict())
    if not fp_nan.empty:
        logging.info('FP NaN counts by column before imputation: %s', fp_nan.to_dict())

    bool_like_cols = {
        'fragile_site', 'phastCons', 'telomere', 'centromere',
        'simpleRepeat_left', 'simpleRepeat_right'
    }

    shared_cols = [col for col in tp_data.columns if col in fp_data.columns and col != 'label']
    for col in shared_cols:
        if not (tp_data[col].isna().any() or fp_data[col].isna().any()):
            continue

        if col in bool_like_cols:
            tp_data[col] = tp_data[col].fillna(False)
            fp_data[col] = fp_data[col].fillna(False)
            continue

        if pd.api.types.is_numeric_dtype(tp_data[col]):
            fill_value = tp_data[col].median(skipna=True)
            if pd.isna(fill_value):
                fill_value = 0.0
            tp_data[col] = tp_data[col].fillna(fill_value)
            fp_data[col] = fp_data[col].fillna(fill_value)
            continue

        # Categorical/object fallback: use TP mode, else placeholder.
        mode_values = tp_data[col].mode(dropna=True)
        fill_value = mode_values.iloc[0] if not mode_values.empty else 'UNKNOWN'
        tp_data[col] = tp_data[col].fillna(fill_value)
        fp_data[col] = fp_data[col].fillna(fill_value)

    # Report NaNs after imputation.
    tp_remaining = int(tp_data.isna().sum().sum())
    fp_remaining = int(fp_data.isna().sum().sum())
    logging.info('NaN imputation complete. Remaining NaNs - TP: %d, FP: %d', tp_remaining, fp_remaining)

    return tp_data, fp_data


def stratified_undersample_fp(fp_data, target_count, random_state=42):
    """Undersample false positives using stratified sampling to preserve SV type and length distribution.
    
    Args:
        fp_data (pd.DataFrame): False positive data to undersample.
        target_count (int): Target number of samples to retain.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        pd.DataFrame: Undersampled false positive data.
    """
    logging.info('Performing stratified undersampling of false positives (count = %d) to target (count = %d)', 
                 fp_data.shape[0], target_count)
    
    # Create length bins for stratification
    fp_data_temp = fp_data.copy()
    fp_data_temp['length_bin'] = pd.cut(fp_data_temp['sv_length'], 
                                         bins=[0, 1000, 10000, 100000, float('inf')],
                                         labels=['<1kb', '1-10kb', '10-100kb', '>100kb'])
    
    # Create stratification column combining SV type and length bin
    fp_data_temp['stratum'] = fp_data_temp['sv_type'].astype(str) + '_' + fp_data_temp['length_bin'].astype(str)
    
    # Calculate target sample size per stratum (proportional to original distribution)
    stratum_counts = fp_data_temp['stratum'].value_counts()
    stratum_fracs = stratum_counts / len(fp_data_temp)
    
    logging.info('Sampling from %d strata with proportional allocation', len(stratum_counts))
    
    # Sample from each stratum proportionally
    sampled_dfs = []
    for stratum, frac in stratum_fracs.items():
        stratum_data = fp_data_temp[fp_data_temp['stratum'] == stratum]
        n_samples = max(1, int(round(frac * target_count)))  # At least 1 sample per stratum
        n_samples = min(n_samples, len(stratum_data))  # Can't sample more than available
        sampled = stratum_data.sample(n=n_samples, random_state=random_state)
        sampled_dfs.append(sampled)
    
    fp_data_balanced = pd.concat(sampled_dfs, ignore_index=True)
    
    # Drop temporary columns
    fp_data_balanced = fp_data_balanced.drop(columns=['length_bin', 'stratum'])
    
    # If we're slightly off from target due to rounding, adjust by random sampling
    if len(fp_data_balanced) > target_count:
        fp_data_balanced = fp_data_balanced.sample(n=target_count, random_state=random_state)
    elif len(fp_data_balanced) < target_count:
        # Sample additional rows to reach target
        n_additional = target_count - len(fp_data_balanced)
        additional = fp_data.sample(n=n_additional, random_state=random_state+1)
        fp_data_balanced = pd.concat([fp_data_balanced, additional], ignore_index=True)
    
    logging.info('Stratified undersampling complete. Final count: %d', len(fp_data_balanced))
    
    return fp_data_balanced


def train(tp_hg002_grch37, fp_hg002_grch37, tp_visor_grch38, fp_visor_grch38, tp_na12877_grch38, fp_na12877_grch38, tp_na12878_grch38, fp_na12878_grch38, tp_na12879_grch38, fp_na12879_grch38, output_directory, annovar_path, db_path, outdiranno, leave_out="none", split_80_20=False, per_chr_validation=False, sample_coverage_hg002=None, sample_coverage_visor=None, sample_coverage_na12877=None, sample_coverage_na12878=None, sample_coverage_na12879=None):
    """Train the binary classification model.
    
    Args:
        sample_coverage_hg002 (float): Required. Mean read depth coverage for HG002 sample.
        sample_coverage_visor (float): Required. Mean read depth coverage for Visor sample.
        sample_coverage_na12877 (float): Required. Mean read depth coverage for NA12877 sample.
        sample_coverage_na12878 (float): Required. Mean read depth coverage for NA12878 sample.
        sample_coverage_na12879 (float): Required. Mean read depth coverage for NA12879 sample.
    """

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
        logging.info('Leaving out Platinum Pedigree datasets (all 3 samples) from training.')
        tp_na12877_grch38 = None
        fp_na12877_grch38 = None
        tp_na12878_grch38 = None
        fp_na12878_grch38 = None
        tp_na12879_grch38 = None
        fp_na12879_grch38 = None
    else:
        logging.info('Not leaving out any dataset from training.')
        no_leave_out = True

    # ===============================================================
    # Extract the features from the VCF files.
    # ===============================================================
    # GRCh38 data.
    logging.info('Extracting features from the true positive and false positive VCF files (GRCh38).')
    buildversion = 'hg38'
    tp_visor_anno = extract_features(tp_visor_grch38, annovar_path, db_path, os.path.join(outdiranno, "tp_visor_anno_grch38"), buildversion=buildversion, sample_coverage=sample_coverage_visor) if tp_visor_grch38 is not None else None
    fp_visor_anno = extract_features(fp_visor_grch38, annovar_path, db_path, os.path.join(outdiranno, "fp_visor_anno_grch38"), buildversion=buildversion, sample_coverage=sample_coverage_visor) if fp_visor_grch38 is not None else None
    
    tp_na12877_anno = extract_features(tp_na12877_grch38, annovar_path, db_path, os.path.join(outdiranno, "tp_na12877_anno_grch38"), buildversion=buildversion, sample_coverage=sample_coverage_na12877) if tp_na12877_grch38 is not None else None
    fp_na12877_anno = extract_features(fp_na12877_grch38, annovar_path, db_path, os.path.join(outdiranno, "fp_na12877_anno_grch38"), buildversion=buildversion, sample_coverage=sample_coverage_na12877) if fp_na12877_grch38 is not None else None
    
    tp_na12878_anno = extract_features(tp_na12878_grch38, annovar_path, db_path, os.path.join(outdiranno, "tp_na12878_anno_grch38"), buildversion=buildversion, sample_coverage=sample_coverage_na12878) if tp_na12878_grch38 is not None else None
    fp_na12878_anno = extract_features(fp_na12878_grch38, annovar_path, db_path, os.path.join(outdiranno, "fp_na12878_anno_grch38"), buildversion=buildversion, sample_coverage=sample_coverage_na12878) if fp_na12878_grch38 is not None else None
    
    tp_na12879_anno = extract_features(tp_na12879_grch38, annovar_path, db_path, os.path.join(outdiranno, "tp_na12879_anno_grch38"), buildversion=buildversion, sample_coverage=sample_coverage_na12879) if tp_na12879_grch38 is not None else None
    fp_na12879_anno = extract_features(fp_na12879_grch38, annovar_path, db_path, os.path.join(outdiranno, "fp_na12879_anno_grch38"), buildversion=buildversion, sample_coverage=sample_coverage_na12879) if fp_na12879_grch38 is not None else None

    # HG002 data (GRCh37).
    logging.info('Extracting features from the true positive and false positive VCF files (HG002-GRCh37).')
    buildversion = 'hg19'
    tp_hg002_anno = extract_features(tp_hg002_grch37, annovar_path, db_path, os.path.join(outdiranno, "tp_anno_grch37"), buildversion=buildversion, sample_coverage=sample_coverage_hg002) if tp_hg002_grch37 is not None else None
    fp_hg002_anno = extract_features(fp_hg002_grch37, annovar_path, db_path, os.path.join(outdiranno, "fp_anno_grch37"), buildversion=buildversion, sample_coverage=sample_coverage_hg002) if fp_hg002_grch37 is not None else None

    # Concatenate the data from all datasets.
    logging.info('Concatenating the data from all datasets.')
    tp_data = pd.concat([df for df in [tp_visor_anno, tp_na12877_anno, tp_na12878_anno, tp_na12879_anno, tp_hg002_anno] if df is not None], ignore_index=True)
    fp_data = pd.concat([df for df in [fp_visor_anno, fp_na12877_anno, fp_na12878_anno, fp_na12879_anno, fp_hg002_anno] if df is not None], ignore_index=True)

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

    # Add the labels.
    tp_data['label'] = 1
    fp_data['label'] = 0

    # Print the number of true positives and false positives.
    logging.info('Number of true labels: %d', tp_data.shape[0])
    logging.info('Number of false labels: %d', fp_data.shape[0])

    # Impute NaN values from the data using TP-referenced statistics.
    tp_data, fp_data = impute_missing_values(tp_data, fp_data)

    # Safety drop for any residual NaNs that could break downstream training.
    logging.info('Dropping any residual NaN rows after imputation.')
    tp_data = tp_data.dropna()
    fp_data = fp_data.dropna()
    logging.info('Number of true labels after impute+dropna: %d', tp_data.shape[0])
    logging.info('Number of false labels after impute+dropna: %d', fp_data.shape[0])

    # Instead of undersampling, use class_weight='balanced' in Random Forest
    # to handle class imbalance while preserving all training data.
    logging.info('Skipping undersampling - will use class_weight="balanced" instead')
    logging.info('Final class counts - TP: %d, FP: %d', tp_data.shape[0], fp_data.shape[0])

    # Combine the true positive and false positive data.
    data = pd.concat([tp_data, fp_data], ignore_index=True)  # Ignore the index to realign the indices.

    # Pop the chrom column to use it later for cross-validation.
    chrom_col = data.pop('chrom')

    # Drop columns that are not needed for training.
    # Keep normalized *_per_kb features; remove raw versions.
    data = data.drop(columns=['start', 'end', 'sv_type_str', 'cluster_size', 'dist_to_nearest_sv', 'read_depth'], errors='ignore')

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

    # If not 80/20 split, use XGBoost and Random Forest only (highest performing models) to save time.
    if split_80_20:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                'xgboost is required when --split-80-20 is enabled. Install xgboost to train with this option.'
            ) from exc
        pipelines = {
            "Random_Forest": Pipeline([('classifier', RandomForestClassifier(n_estimators=100, random_state=42))]),
            "XGBoost": Pipeline([('classifier', XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42, enable_categorical=False))])
        }
        # pipelines = {
        #     "Logistic_Regression": Pipeline([('classifier', LogisticRegression(max_iter=1000, random_state=42))]),
        #     "Random_Forest": Pipeline([('classifier', RandomForestClassifier(n_estimators=100, random_state=42))]),
        #     "XGBoost": Pipeline([('classifier', XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42, enable_categorical=False))])
        # }
    else:
        pipelines = {
            "Random_Forest": Pipeline([('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))]),
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

    if per_chr_validation:
        # ======================================================
        # Evaluate the model using per-chromosome cross-validation, but don't save.
        # ======================================================
        logging.info('Evaluating the model using per-chromosome cross-validation.')

        # Remove chrY from the analysis. More than half is missing in GRCh38 and leads to high false positive rates.
        chromosomes = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
                    'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']

        logging.info('Chromosomes: %s', chromosomes)
        f1_scores = {}
        precision_scores = {}
        recall_scores = {}
        for model_name, pipeline in pipelines.items():
            # Dictionary with number of SVs in the held-out set for each chromosome.
            sv_counts = {chrom: features[chrom_col == chrom].shape[0] for chrom in chromosomes}
            logging.info('Number of SVs in the held-out set for each chromosome: %s', sv_counts)

            for chrom in chromosomes:
                logging.info('Training the %s model on chromosome %s.', model_name, chrom)

                # Split the data into training and testing sets by chromosome.
                X_train_chrom = features[chrom_col != chrom].copy()
                y_train_chrom = labels[chrom_col != chrom].copy()
                X_test_chrom = features[chrom_col == chrom].copy()
                y_test_chrom = labels[chrom_col == chrom].copy()

                logging.info('Training set size: %d, Testing set size: %d',
                            X_train_chrom.shape[0], X_test_chrom.shape[0])

                X_train_chrom_processed = preprocess_feature_matrix(X_train_chrom)
                X_test_chrom_processed = preprocess_feature_matrix(X_test_chrom)

                fold_cv = get_cv_splits(y_train_chrom)
                if fold_cv is None:
                    logging.warning(
                        'Skipping chromosome %s for %s: insufficient class balance for stratified CV.',
                        chrom,
                        model_name
                    )
                    continue

                grid_search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grids[model_name],
                    cv=fold_cv,
                    scoring='precision',
                    n_jobs=-1
                )
                grid_search.fit(X_train_chrom_processed, y_train_chrom)
                best_model = grid_search.best_estimator_
                logging.info(
                    'Best hyperparameters for %s on held-out chromosome %s: %s',
                    model_name,
                    chrom,
                    grid_search.best_params_
                )

                # Get the predicted probabilities for the testing set.
                y_test_chrom_prob = best_model.predict_proba(X_test_chrom_processed)[:, 1]

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
                
        logging.info('Cross-validation analysis completed. F1 scores: %s', f1_scores)

        # Plot the F1 scores for each model and chromosome (one plot per model).
        logging.info('Plotting the scores for each model and chromosome.')
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError as exc:
            raise ImportError(
                'matplotlib and seaborn are required for per-chromosome validation plots.'
            ) from exc
        metrics = ['F1 Score', 'Precision', 'Recall']
        for model_name in pipelines.keys():

            # Save a plot with F1, Precision, and Recall scores for chrY
            if 'chrY' in chromosomes:
                logging.info('Plotting scores for %s model on chrY.', model_name)

                # Create a bar plot for the F1 scores by chromosome.
                chry_f1 = f1_scores.get((model_name, 'chrY'), 0)
                chry_precision = precision_scores.get((model_name, 'chrY'), 0)
                chry_recall = recall_scores.get((model_name, 'chrY'), 0)
                plt.figure(figsize=(6, 4))

                # Plot F1, Precision, and Recall scores for chrY.
                sns.barplot(x=['F1 Score', 'Precision', 'Recall'], y=[chry_f1, chry_precision, chry_recall], color='black')

                # plt.xlabel('Metric')
                plt.ylabel('Score')
                plt.title('%s Scores for %s Model on chrY' % (model_name, model_name))
                plt.xticks(rotation=45)
                plt.tight_layout()
                # Save the plot to the output directory.
                score_plot_path = os.path.join(output_directory, model_name + '_scores_chrY.svg')
                plt.savefig(score_plot_path)
                plt.close()
                logging.info('Saved the scores plot for chrY to %s', score_plot_path)

            for metric, scores in zip(metrics, [f1_scores, precision_scores, recall_scores]):
                logging.info('Plotting %s for %s model by chromosome.', metric, model_name)
                # Create a bar plot for the F1 scores by chromosome.
                model_scores = {chrom: scores[(model_name, chrom)] for chrom in chromosomes if (model_name, chrom) in scores}
                plt.figure(figsize=(10, 6))
                ax = sns.barplot(x=list(model_scores.keys()), y=list(model_scores.values()), color='black')

                plt.xlabel('Chromosome')
                plt.ylabel(metric)
                plt.title('%s for %s Model by Chromosome' % (metric, model_name))
                plt.xticks(rotation=45)
                plt.tight_layout()
                score_plot_path = os.path.join(output_directory, model_name + '_%s_by_chromosome.svg' % metric.lower().replace(' ', '_'))
                plt.savefig(score_plot_path)
                plt.close()
                logging.info('Saved the %s plot to %s', metric, score_plot_path)

    else:
        # =======================================================
        # Train the model using cross-validation and grid search for hyperparameter tuning.
        # =======================================================
        cv = get_cv_splits(y_train)
        if cv is None:
            raise ValueError('Unable to run training: need at least two classes with at least two samples each for stratified CV.')

        for model_name, pipeline in pipelines.items():
            logging.info('Training model class %s', model_name)
            model_name_fp = "contextscore_" + model_name.lower() + "_leaveout_" + leave_out

            if split_80_20:
                model_name_fp += "_80_20_split"

            # Perform grid search to find the best hyperparameters for the model, optimizing for precision to prioritize reducing false positives.
            X_train_processed = preprocess_feature_matrix(X_train)
            X_test_processed = preprocess_feature_matrix(X_test)

            grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grids[model_name], cv=cv, scoring='precision', n_jobs=-1)
            grid_search.fit(X_train_processed, y_train)
            logging.info('Best hyperparameters for %s: %s', model_name, grid_search.best_params_)

            # Get predicted probabilities for the training and testing sets.
            best_model = grid_search.best_estimator_

            # Save plots only for 80-20 split since the ROC curve will be overly optimistic when using all the data for training and testing.
            if split_80_20:
                try:
                    import matplotlib.pyplot as plt
                except ImportError as exc:
                    raise ImportError(
                        'matplotlib is required when --split-80-20 is enabled to generate ROC plots.'
                    ) from exc
                y_train_prob = best_model.predict_proba(X_train_processed)[:, 1]
                y_test_prob = best_model.predict_proba(X_test_processed)[:, 1]

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
                roc_plot_path = os.path.join(output_directory, model_name_fp + '_roc_curve_train.svg')
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
                roc_plot_path = os.path.join(output_directory, model_name + '_roc_curve_test.svg')
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
                logging.info('Running feature importance analysis for %s model.', model_name)
                classifier = best_model.named_steps['classifier']

                # For Random Forest, use both native importance and SHAP (with aggressive sampling)
                if model_name == 'Random_Forest':
                    try:
                        import matplotlib.pyplot as plt

                        # 1. Native Random Forest feature importance (instant)
                        feature_importances = classifier.feature_importances_
                        feature_names = X_train.columns.tolist()
                        display_feature_names = get_display_feature_names(feature_names)
                        
                        importance_df = pd.DataFrame({
                            'feature': feature_names,
                            'feature_display': display_feature_names,
                            'importance': feature_importances
                        }).sort_values('importance', ascending=True)
                        
                        plt.figure(figsize=(10, 8))
                        plt.bar(importance_df['feature_display'], importance_df['importance'])
                        plt.ylabel('Feature Importance')
                        plt.xlabel('Feature')
                        plt.title('Random Forest Feature Importance')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        importance_plot_path = os.path.join(output_directory, model_name_fp + '_feature_importance_plot.svg')
                        plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        logging.info('Saved Random Forest feature-importance plot to %s', importance_plot_path)
                        
                        importance_csv_path = os.path.join(output_directory, model_name_fp + '_feature_importances.csv')
                        importance_df.sort_values('importance', ascending=False).to_csv(importance_csv_path, index=False)
                        logging.info('Saved feature importances to %s', importance_csv_path)
                        
                        if ENABLE_SHAP:
                            # 2. SHAP analysis with aggressive sampling for efficiency
                            logging.info('Computing SHAP values for Random Forest (with sampling)...')
                            X_train_numeric = X_train.copy()
                            for col in X_train_numeric.columns:
                                if X_train_numeric[col].dtype == 'object':
                                    X_train_numeric[col] = pd.to_numeric(X_train_numeric[col], errors='coerce')
                            X_train_numeric = X_train_numeric.fillna(0).astype('float64')
                            
                            # Aggressive sampling for RF SHAP: reduce from 148k to ~300 samples
                            explain_size = min(300, len(X_train_numeric))
                            background_size = min(50, len(X_train_numeric) // 100)  # ~1% of data
                            X_explain = shap.sample(X_train_numeric, explain_size, random_state=42)
                            X_background = shap.sample(X_train_numeric, background_size, random_state=42)
                            
                            logging.info('SHAP RF: explain_size=%d, background_size=%d (from %d total)', 
                                        explain_size, background_size, len(X_train_numeric))
                            
                            # Use interventional mode for standard SHAP values (not interactions)
                            explainer = shap.TreeExplainer(classifier)
                            shap_values = explainer.shap_values(X_explain, check_additivity=False)
                            
                            logging.info('SHAP raw output type: %s, raw shape: %s', 
                                        type(shap_values), 
                                        shap_values.shape if hasattr(shap_values, 'shape') else 'N/A')
                            
                            # Handle different output formats
                            if isinstance(shap_values, list):
                                # List of arrays for each class
                                shap_values = shap_values[1]  # Use positive class
                            elif len(shap_values.shape) == 3:
                                # 3D array: (n_samples, n_features, n_classes)
                                shap_values = shap_values[:, :, 1]  # Select positive class
                            
                            logging.info('SHAP debug: shap_values shape=%s (final), X_explain shape=%s', 
                                        shap_values.shape, X_explain.shape)
                            
                            # Ensure X_explain is explicitly indexed by feature names
                            X_explain_for_plot = X_explain.reset_index(drop=True)
                            X_explain_display = X_explain_for_plot.rename(columns=get_display_feature_name)
                            
                            # SHAP summary plot
                            plt.figure(figsize=(12, 8))
                            shap.summary_plot(shap_values, X_explain_display, show=False, max_display=15)
                            shap_plot_path = os.path.join(output_directory, model_name_fp + '_shap_summary_plot.svg')
                            plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
                            plt.close()
                            logging.info('Saved SHAP summary plot to %s', shap_plot_path)
                            
                            # SHAP bar plot (mean |SHAP|)
                            plt.figure(figsize=(10, 8))
                            shap.summary_plot(shap_values, X_explain_display, plot_type='bar', show=False)
                            bar_plot_path = os.path.join(output_directory, model_name_fp + '_shap_importance_plot.svg')
                            plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
                            plt.close()
                            logging.info('Saved SHAP importance plot to %s', bar_plot_path)
                        
                    except Exception as exc:
                        logging.warning('SHAP analysis skipped for %s: %s', model_name, exc)
                
                # For other models, use SHAP
                else:
                    if ENABLE_SHAP:
                        logging.info('Computing SHAP values for %s model...', model_name)
                        # Prepare numeric data for SHAP
                        X_train_numeric = X_train.copy()
                        for col in X_train_numeric.columns:
                            if X_train_numeric[col].dtype == 'object':
                                X_train_numeric[col] = pd.to_numeric(X_train_numeric[col], errors='coerce')

                        X_train_numeric = X_train_numeric.fillna(0).astype('float64')

                        # Bound SHAP workload to avoid OOM/core-dump on large full-model runs.
                        explain_size = min(5000, len(X_train_numeric))
                        background_size = min(300, len(X_train_numeric))
                        X_explain = shap.sample(X_train_numeric, explain_size, random_state=42)
                        X_background = shap.sample(X_train_numeric, background_size, random_state=42)

                        logging.info(
                            'SHAP sampling: explain_size=%d, background_size=%d (from %d training rows)',
                            len(X_explain), len(X_background), len(X_train_numeric)
                        )

                        try:
                            if model_name == 'XGBoost':
                                explainer = shap.TreeExplainer(classifier, feature_perturbation='tree_path_dependent')
                                shap_values = explainer.shap_values(X_explain)
                            elif model_name == 'Logistic_Regression':
                                explainer = shap.LinearExplainer(classifier, X_background)
                                shap_values = explainer.shap_values(X_explain)
                            else:
                                explainer = shap.Explainer(classifier, X_background)
                                shap_values = explainer(X_explain)

                            # Some SHAP explainers return one array per class. For binary
                            # classification plots, use positive class values.
                            if isinstance(shap_values, list) and len(shap_values) > 1:
                                shap_values_to_plot = shap_values[1]
                            else:
                                shap_values_to_plot = shap_values

                            X_explain_display = X_explain.rename(columns=get_display_feature_name)

                            # 1. Summary plot
                            plt.figure(figsize=(10, 8))
                            shap.summary_plot(shap_values_to_plot, X_explain_display, show=False)
                            shap_plot_path = os.path.join(output_directory, model_name_fp + '_shap_summary_plot.svg')
                            plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
                            plt.close()
                            logging.info('Saved the SHAP summary plot to %s', shap_plot_path)

                            # 2. Bar plot showing mean absolute SHAP values (feature importance)
                            plt.figure(figsize=(10, 8))
                            shap.summary_plot(shap_values_to_plot, X_explain_display, plot_type='bar', show=False)
                            bar_plot_path = os.path.join(output_directory, model_name_fp + '_shap_importance_plot.svg')
                            plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
                            plt.close()
                            logging.info('Saved the SHAP importance plot to %s', bar_plot_path)
                        except Exception as exc:
                            logging.warning('SHAP analysis failed for %s: %s. Continuing without SHAP outputs.', model_name, exc)

if __name__ == '__main__':
    # Parse the command line arguments.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp_hg002_grch37", required=True, help="Path to the true positive BED file for HG002 in GRCh37")
    parser.add_argument("--fp_hg002_grch37", required=True, help="Path to the false positive BED file for HG002 in GRCh37")
    parser.add_argument("--tp_visor_grch38", required=True, help="Path to the true positive BED file for Visor in GRCh38")
    parser.add_argument("--fp_visor_grch38", required=True, help="Path to the false positive BED file for Visor in GRCh38")
    parser.add_argument("--tp_na12877_grch38", required=True, help="Path to the true positive BED file for NA12877 in GRCh38")
    parser.add_argument("--fp_na12877_grch38", required=True, help="Path to the false positive BED file for NA12877 in GRCh38")
    parser.add_argument("--tp_na12878_grch38", required=True, help="Path to the true positive BED file for NA12878 in GRCh38")
    parser.add_argument("--fp_na12878_grch38", required=True, help="Path to the false positive BED file for NA12878 in GRCh38")
    parser.add_argument("--tp_na12879_grch38", required=True, help="Path to the true positive BED file for NA12879 in GRCh38")
    parser.add_argument("--fp_na12879_grch38", required=True, help="Path to the false positive BED file for NA12879 in GRCh38")
    parser.add_argument("--outdiranno", required=True, help="Output directory for saving the ANNOVAR annotations")
    parser.add_argument("--outdir", required=True, help="Output directory for saving the model")
    parser.add_argument("--annovar", required=True, help="Path to ANNOVAR")
    parser.add_argument("--annovar_db", required=True, help="Path to ANNOVAR database")
    parser.add_argument("--leave_out", required=True, help="Which dataset to leave out for training")
    parser.add_argument("--sample_coverage_hg002", type=float, required=True, help="Mean read depth coverage for HG002 sample (required)")
    parser.add_argument("--sample_coverage_visor", type=float, required=True, help="Mean read depth coverage for Visor sample (required)")
    parser.add_argument("--sample_coverage_na12877", type=float, required=True, help="Mean read depth coverage for NA12877 sample (required)")
    parser.add_argument("--sample_coverage_na12878", type=float, required=True, help="Mean read depth coverage for NA12878 sample (required)")
    parser.add_argument("--sample_coverage_na12879", type=float, required=True, help="Mean read depth coverage for NA12879 sample (required)")
    parser.add_argument("--split_80_20", action='store_true', help="Whether to split the data into training and testing sets using an 80-20 split. If not specified, all the data will be used for training and testing, and cross-validation will be used to evaluate the model performance.")
    parser.add_argument("--per_chr_validation", action='store_true', help="Whether to run per-chromosome cross-validation.")
    args = parser.parse_args()

    # Run the program.
    logging.info('Training the model, split_80_20 = %s, leave_out = %s, per_chr_validation = %s', args.split_80_20, args.leave_out, args.per_chr_validation)
    train(args.tp_hg002_grch37, args.fp_hg002_grch37, args.tp_visor_grch38, args.fp_visor_grch38, args.tp_na12877_grch38, args.fp_na12877_grch38, args.tp_na12878_grch38, args.fp_na12878_grch38, args.tp_na12879_grch38, args.fp_na12879_grch38, args.outdir, args.annovar, args.annovar_db, args.outdiranno, args.leave_out, args.split_80_20, args.per_chr_validation, args.sample_coverage_hg002, args.sample_coverage_visor, args.sample_coverage_na12877, args.sample_coverage_na12878, args.sample_coverage_na12879)
    logging.info('done.')
