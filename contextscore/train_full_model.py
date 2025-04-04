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

def read_cytoband_file(cytoband_file):
    """Get the centromere and telomere regions for each chromosome."""
    cytobands = pd.read_csv(cytoband_file, sep='\t', header=None, names=["chrom", "start", "end", "name", "gieStain"])
    chrom_dict = {}
    for chrom in cytobands['chrom'].unique():
        
        # Skip chrM
        if chrom == 'chrM':
            continue

        chrom_df = cytobands[cytobands['chrom'] == chrom]
        # First and last bands are the telomeres.
        # First telomere:
        chrom_dict[chrom] = {
            'telomerep': chrom_df.iloc[0]['name'],
            'telomereq': chrom_df.iloc[-1]['name']
        }

        # Identify the 2 centromeres for p and q (contain "acen").
        centromere_p = chrom_df[chrom_df['name'].str.contains('acen') & chrom_df['name'].str.contains('p')]
        centromere_q = chrom_df[chrom_df['name'].str.contains('acen') & chrom_df['name'].str.contains('q')]
        if not centromere_p.empty:
            chrom_dict[chrom]['centromerep'] = centromere_p.iloc[0]['name']
        if not centromere_q.empty:
            chrom_dict[chrom]['centromereq'] = centromere_q.iloc[0]['name']

        # print("Chromosome:", chrom)
        # print(chrom_dict[chrom])

    return chrom_dict

def extract_features(input_bed):
    """Extract the features from the BED file, columns are in the first row:
    chrom, start, end, sv_type, sv_length, genotype, read_depth, hmm_llh, aln_type, cluster_size
    """
    logging.info('Extracting features from the BED file %s', input_bed)

    # Load a dictionary mapping chromosome names to numbers.
    # chrom_dict_path="/mnt/isilon/wang_lab/perdomoj/projects/ContextScore/Train/Model/chrom_map.pkl"
    # chrom_dict = joblib.load(chrom_dict_path)

    # Read in the BED file.
    bed_df = pd.read_csv(input_bed, sep='\t', header=0, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                         names=['chrom', 'start', 'end', 'sv_type', 'sv_length', 'genotype', 'read_depth', 'hmm_llh', 'aln_type', 'cluster_size'],
                         dtype={'chrom': str, 'start': np.int32, 'end': np.int32, 'sv_type': str, 'sv_length': np.int32, 'genotype': str, 'read_depth': np.int32, 'hmm_llh': np.float32, 'aln_type': str, 'cluster_size': np.int32})

    # # Print the number of NaN values
    # logging.info('Number of NaN values: %d', bed_df.isnull().sum().sum())

    # # Map the chromosome names to numbers.
    # bed_df['chrom'] = bed_df['chrom'].map(chrom_dict)

    # # Print the number of NaN values
    # logging.info('Number of NaN values after chr mapping: %d', bed_df.isnull().sum().sum())

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


def run_bedtools_intersect(input_bed, table_bed):
    """Run bedtools intersect to annotate the BED file."""
    # Check if bedtools is installed.
    try:
        subprocess.run(["bedtools", "--version"], check=True)
    except subprocess.CalledProcessError:
        logging.error('bedtools is not installed. Please install bedtools.')
        sys.exit(1)

    # Check if the input BED file exists.
    if not os.path.exists(input_bed):
        logging.error('Input BED file does not exist: %s', input_bed)
        sys.exit(1)

    # Check if the table BED file exists.
    if not os.path.exists(table_bed):
        logging.error('Table BED file does not exist: %s', table_bed)
        sys.exit(1)

    # Run bedtools intersect to annotate the BED file.
    cmd = [
        "bedtools", "intersect",
        "-a", input_bed,
        "-b", table_bed,
        "-wa", "-wb"
    ]
    logging.info('Running the command to annotate the BED file: %s', " ".join(cmd))
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True)

        # Parse the output of bedtools intersect into a pandas DataFrame.
        logging.info('Parsing the output of bedtools intersect.')
        annotated_bed = pd.read_csv(
            StringIO(result.stdout),
            sep='\t',
            header=None,
            names=["chrom", "start", "end", "chr_anno", "start_anno", "end_anno", "name"],
            usecols=[0, 1, 2, 10, 11, 12, 13],
            dtype={'chrom': str, 'start': np.int32, 'end': np.int32, 'chr_anno': str, 'start_anno': np.int32, 'end_anno': np.int32, 'name': str}
        )

        # Print the first few rows of the annotated BED file.
        logging.info('Annotated BED file:\n%s', annotated_bed.head())

        return annotated_bed

    except subprocess.CalledProcessError as e:
        logging.error('Error annotating the BED file: %s', e)
        logging.error('Please check the input and table BED files.')
        sys.exit(1)


def bed_to_annovar_input(bed_file):
    """Convert the BED file to ANNOVAR input format."""
    output_file = bed_file.replace('.bed', '.avinput')
    logging.info('Converting the BED file to ANNOVAR input format.')

    # Read the BED file using pandas (first line is the header with the column names).
    df = pd.read_csv(bed_file, sep='\t', header=None, comment='#', names=["CHROM", "POS", "END", "SVTYPE", "SVLEN"], skiprows=1)
    logging.info('Number of rows in the BED file: %d', df.shape[0])
    logging.info('First 5 rows of the BED file:\n%s', df.head())

    # The ANNOVAR input format requires the following columns:
    # 1. Chromosome
    # 2. Start position
    # 3. End position
    # 4. Reference allele
    # 5. Alternate allele
    # We will use the first three columns from the BED file and add two dummy
    # columns for the reference and alternate alleles (0, and -) since gnomAD does not
    # provide the sequence information for the SVs.

    # Create a new dataframe with the required columns.
    annovar_df = pd.DataFrame()
    annovar_df['chrom'] = df['CHROM']
    annovar_df['start'] = df['POS']
    annovar_df['end'] = df['END']
    annovar_df['ref'] = '0'
    annovar_df['alt'] = '-'

    # Save the tab-delimited dataframe to a file.
    logging.info('Saving the ANNOVAR input file to %s', output_file)
    annovar_df.to_csv(output_file, sep='\t', index=False, header=False)
    logging.info('Number of rows in the ANNOVAR input file: %d', annovar_df.shape[0])
    logging.info('First 5 rows of the ANNOVAR input file:\n%s', annovar_df.head())
    logging.info('Saved the ANNOVAR input file to %s', output_file)

    return output_file


def download_annovar_db(annovar_path, db_path, db_name):
    """Download the ANNOVAR database if it does not exist."""
    logging.info('Downloading the database:' + db_name)
    cmd = [
        f"{annovar_path}/annotate_variation.pl",
        "-buildver", "hg38",
        "-downdb", db_name,
        db_path
    ]
    # annotate_variation.pl -build hg19 -downdb phastConsElements46way humandb/

    # Run the command to download the database.
    logging.info('Running the command to download the database: %s', " ".join(cmd))
    try:
        subprocess.run(" ".join(cmd), shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logging.error('Error downloading the database: %s', e)
        logging.error('Please check the ANNOVAR path and database path.')
        sys.exit(1)
    logging.info('Downloaded the database %s successfully.', db_name)


def annotate(annovar_input, annovar_path, db_path, output_dir):
    """Annotate regions."""
    logging.info('Annotating regions using ANNOVAR.')

    annotations_dir = os.path.join(output_dir, 'regions')
    logging.info('Creating the output directory: %s', annotations_dir)
    cmd = [
        f"{annovar_path}/table_annovar.pl",
        annovar_input,
        db_path,
        "--buildver hg38",
        "--out", annotations_dir,
        "--remove",
        "--protocol genomicSuperDups,cytoBand",
        "--operation r,r",
        "--nastring .",
        "-polish"
    ]
    # "--protocol genomicSuperDups",

    try:
        subprocess.run(" ".join(cmd), shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logging.error('Error annotating: %s', e)
        logging.error('Please check the ANNOVAR path and database path.')
        sys.exit(1)

    logging.info('Completed annotations.')


def get_cytoband_is_c_t(chrom_dict, chrom, cytoband):
    """Check if the cytoband is a telomere or centromere."""
    if chrom not in chrom_dict:
        return False, False  # Not in any region.

    is_telomere = False
    is_centromere = False
    # Check if the cytoband is a telomere.
    try:
        if 'telomerep' in chrom_dict[chrom] and chrom_dict[chrom]['telomerep'] in cytoband:
            is_telomere = True
    except TypeError:
        # Handle the case where telomerep is not defined.
        logging.warning('chrom_dict[%s] does not have telomerep defined.', chrom)
        logging.warning('Cytoband: %s', cytoband)
        logging.warning('chrom_dict[%s]: %s', chrom, chrom_dict[chrom])
        is_telomere = False
    if 'telomereq' in chrom_dict[chrom] and chrom_dict[chrom]['telomereq'] in cytoband:
        is_telomere = True
    if 'centromerep' in chrom_dict[chrom] and chrom_dict[chrom]['centromerep'] in cytoband:
        is_centromere = True
    if 'centromereq' in chrom_dict[chrom] and chrom_dict[chrom]['centromereq'] in cytoband:
        is_centromere = True
    
    return is_telomere, is_centromere


def add_annotations(data, input_bed, annovar_path, db_path, anno_outdir):
    """Add annotations to the features."""
    logging.info('Adding annotations to the features.')

    # ---------------------------------------------------------------
    # Annotate the fragile sites using a BED file from HumCFS (GRCh38/hg38).
    # https://webs.iiitd.edu.in/raghava/humcfs/download.html
    # ANNOVAR instructions are here: https://annovar.openbioinformatics.org/en/latest/user-guide/region/
    fragile_sites_bed="/mnt/isilon/wang_lab/perdomoj/projects/ContextScore/Train/FragileSites/FragileSites_merged.bed"
    logging.info('Annotating the fragile sites using the BED file (GRCh38): %s', fragile_sites_bed)
    fragile_sites_df = run_bedtools_intersect(input_bed, fragile_sites_bed)

    # Merge the fragile sites annotations with the true positive data.
    data['fragile_site'] = data.merge(fragile_sites_df, on=['chrom', 'start', 'end'], how='left')['chr_anno'].notna()

    logging.info('Number of records with fragile sites: %d', data['fragile_site'].sum())
    logging.info('Total number of records: %d', data.shape[0])

    # ---------------------------------------------------------------
    # Annotate conserved regions using a UCSC Table Browser BED file for
    # phastCons100way (GRCh38/hg38).
    phastCons_bed = "/mnt/isilon/wang_lab/perdomoj/data/UCSC_Tables/phastCons100way_hg38.bed"
    logging.info('Annotating conserved regions using the BED file (GRCh38): %s', phastCons_bed)
    phastCons_df = run_bedtools_intersect(input_bed, phastCons_bed)

    # Merge the phastCons annotations with the true positive data.
    data['phastCons'] = data.merge(phastCons_df, on=['chrom', 'start', 'end'], how='left')['chr_anno'].notna()

    logging.info('Number of records with conserved regions: %d', data['phastCons'].sum())
    logging.info('Total number of records: %d', data.shape[0])

    # ---------------------------------------------------------------
    # Annotate simple repeats using a UCSC Table Browser BED file for
    # simpleRepeat (GRCh38/hg38).
    simpleRepeat_bed = "/mnt/isilon/wang_lab/perdomoj/data/UCSC_Tables/simple_repeats_hg38.bed"
    logging.info('Annotating simple repeats using the BED file (GRCh38): %s', simpleRepeat_bed)
    simpleRepeat_df = run_bedtools_intersect(input_bed, simpleRepeat_bed)

    # Merge the simpleRepeat annotations with the true positive data.
    data['simpleRepeat'] = data.merge(simpleRepeat_df, on=['chrom', 'start', 'end'], how='left')['chr_anno'].notna()

    logging.info('Number of records with simple repeats: %d', data['simpleRepeat'].sum())
    logging.info('Total number of records: %d', data.shape[0])

    # ---------------------------------------------------------------
    # Annotate the SVs using ANNOVAR.
    
    # Download the segmental duplication database
    download_annovar_db(annovar_path, db_path, "genomicSuperDups")

    # Download the cytoband database
    download_annovar_db(annovar_path, db_path, "cytoBand")

    # Set up a dictionary for each chromosome, mapping the cytoband to the
    # centromere and telomere regions.
    cytoband_file = "/home/perdomoj/github/ContextScore/data/hg38_cytoband.txt"  # Downloaded from UCSC.
    cytoband_dict = read_cytoband_file(cytoband_file)

    logging.info('Converting the true positive BED file to ANNOVAR input format.')
    annovar_file = bed_to_annovar_input(input_bed)

    logging.info('Annotating the SVs using ANNOVAR.')
    if not os.path.exists(anno_outdir):
        os.makedirs(anno_outdir)

    annotate(annovar_file, annovar_path, db_path, anno_outdir)

    anno_file = os.path.join(anno_outdir, 'regions.hg38_multianno.txt')
    if not os.path.exists(anno_file):
        logging.error('ANNOVAR annotation file does not exist: %s', anno_file)
        sys.exit(1)

    # Read the ANNOVAR output file.
    logging.info('Reading the ANNOVAR output file: %s', anno_file)
    anno_df = pd.read_csv(anno_file, sep='\t', header=0, comment='#')

    # Replace NaN values for the genomicSuperDups column with 0.
    # anno_df['genomicSuperDups'].fillna(0, inplace=True)

    # # Replace NaN values for the cytoBand column with ""
    # # anno_df['cytoBand'].fillna("", inplace=True).astype(str)
    # anno_df['cytoBand'] = anno_df['cytoBand'].fillna("").astype(str)

    # Convert chr, start, end to the same data types as the data.
    anno_df['Chr'] = anno_df['Chr'].astype(str)
    anno_df['Start'] = anno_df['Start'].astype(np.int32)
    anno_df['End'] = anno_df['End'].astype(np.int32)

    print("[TEST] Data types:")
    print(data.dtypes[['chrom', 'start', 'end']])
    print(anno_df.dtypes[['Chr', 'Start', 'End']])

    # Merge the ANNOVAR annotations with the data.
    logging.info('Merging the ANNOVAR annotations with the data.')
    data = data.merge(anno_df, left_on=['chrom', 'start', 'end'], right_on=['Chr', 'Start', 'End'], how='left')

    # Extract segmental duplication scores.
    def extract_max_score(score_series):
        """Extract and return the maximum Score= value from a series."""
        scores = score_series.str.extract(r'Score=([\d\.]+)')[0].dropna().astype(float)
        return scores.max() if not scores.empty else 0
    
    # Extract the maximum score from the segmental duplication annotations.
    data['segdup'] = extract_max_score(data['genomicSuperDups'])

    # Extract the cytoband annotations.
    def get_cyto_info(row):
        """Get telomere and centromere information for a row."""
        if pd.notna(row['cytoBand']):
            return get_cytoband_is_c_t(cytoband_dict, row['chrom'], row['cytoBand'])
        
        return (False, False)
    
    data['telomere'], data['centromere'] = data.apply(get_cyto_info, axis=1, result_type='expand')

    # Print the current columns in the data.
    logging.info('Current columns in the data: %s', data.columns)

    # Drop the unnecessary columns.
    data.drop(columns=['Chr', 'Start', 'End', 'cytoBand', 'genomicSuperDups', 'Ref', 'Alt'], inplace=True)

    logging.info('Number of records after adding annotations: %d', data.shape[0])
    logging.info('First 5 rows of the data after adding annotations:\n%s', data.head())


def train(tp_bed, fp_bed, output_directory, annovar_path, db_path, outdiranno):
    """Train the binary classification model."""

    # ---------------------------------------------------------------
    # SV Feature Extraction
    # ---------------------------------------------------------------

    # Extract the features from the VCF files.
    tp_data = extract_features(tp_bed)
    fp_data = extract_features(fp_bed)

    # ---------------------------------------------------------------
    # Annotate the features
    # ---------------------------------------------------------------

    # Add annotations to the features.
    tp_anno_outdir = os.path.join(outdiranno, "tp_anno")
    add_annotations(tp_data, tp_bed, annovar_path, db_path, tp_anno_outdir)
    fp_anno_outdir = os.path.join(outdiranno, "fp_anno")
    add_annotations(fp_data, fp_bed, annovar_path, db_path, fp_anno_outdir)

    # ---------------------------------------------------------------
    # Feature preparation
    # ---------------------------------------------------------------

    # Finally map chromosome names to numbers.
    # Load a dictionary mapping chromosome names to numbers.
    chrom_dict_path="/mnt/isilon/wang_lab/perdomoj/projects/ContextScore/Train/Model/chrom_map.pkl"
    chrom_dict = joblib.load(chrom_dict_path)

    # Print the number of NaN values
    logging.info('Number of NaN values: %d', tp_data.isnull().sum().sum())

    # Map the chromosome names to numbers.
    tp_data['chrom'] = tp_data['chrom'].map(chrom_dict)
    fp_data['chrom'] = fp_data['chrom'].map(chrom_dict)

    # Print the number of NaN values
    logging.info('Number of NaN values after chr mapping: %d', tp_data.isnull().sum().sum())

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
def run(tp_bed, fp_bed, output_directory, annovar_path, db_path, outdiranno):
    """Run the program."""
    # Train the model.
    train(tp_bed, fp_bed, output_directory, annovar_path, db_path, outdiranno)
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
    # Parse the command line arguments.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpbed", required=True, help="Directory containing benchmark VCF files of real SVs (true positives and false negatives)")
    parser.add_argument("--fpbed", required=True, help="Directory containing false positive VCF files from running the caller on normal samples")
    parser.add_argument("--outdiranno", required=True, help="Output directory for saving the ANNOVAR annotations")
    parser.add_argument("--outdir", required=True, help="Output directory for saving the model")
    parser.add_argument("--annovar", required=True, help="Path to ANNOVAR")
    parser.add_argument("--annovar_db", required=True, help="Path to ANNOVAR database")
    args = parser.parse_args()

    # Run the program.
    logging.info('Training the model...')
    run(args.tpbed, args.fpbed, args.outdir, args.annovar, args.annovar_db, args.outdiranno)
    logging.info('done.')
