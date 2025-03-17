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

def download_annovar_db(annovar_path, db_path, db_name, buildver):
    """Download the ANNOVAR database if it does not exist."""
    logging.info('Downloading the database:' + db_name)
    cmd = [
        f"{annovar_path}/annotate_variation.pl",
        "-buildver", buildver,
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

def annotate_bed(annovar_input, annovar_path, db_path, output_dir, bed_file):
    """Annotate from the BED file using ANNOVAR."""
    # Use the example: annotate_variation.pl ex1.hg18.avinput humandb/ -bedfile hg18_SureSelect_All_Exon_G3362_with_names.bed -dbtype bed -regionanno -out ex1 
    logging.info('Annotating from the BED file using ANNOVAR.')

    annotations_dir = os.path.join(output_dir, 'bed')
    logging.info('Creating the output directory: %s', annotations_dir)
    cmd = [
        f"{annovar_path}/annotate_variation.pl",
        annovar_input,
        db_path,
        "-buildver hg38",
        "-bedfile", bed_file,
        "-dbtype bed",
        "-regionanno",
        "-out", annotations_dir
    ]

    try:
        subprocess.run(" ".join(cmd), shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logging.error('Error annotating from the BED file: %s', e)
        logging.error('Please check the ANNOVAR path and database path.')
        sys.exit(1)

    logging.info('Completed annotating from the BED file using ANNOVAR.')
    logging.info('Output directory: %s', annotations_dir)

# Run the program.
def run(tp_bed, fp_bed, output_directory, output_directory_annovar, annovar_path, db_path):
    """Train the binary classification model."""

    # TODO: Make this an input parameter.
    buildver = 'hg38'

    logging.info('Getting the true positive and false positive VCF files.')

    # Convert the BED files to ANNOVAR input format.
    logging.info('Converting the true positive BED file to ANNOVAR input format.')
    true_positives_file = bed_to_annovar_input(tp_bed)

    logging.info('Converting the false positive BED file to ANNOVAR input format.')
    false_positives_file = bed_to_annovar_input(fp_bed)


    # Annotate the fragile sites using a BED file from HumCFS (GRCh38/hg38).
    # https://webs.iiitd.edu.in/raghava/humcfs/download.html
    # ANNOVAR instructions are here: https://annovar.openbioinformatics.org/en/latest/user-guide/region/
    # fragile_sites_bed="/mnt/isilon/wang_lab/perdomoj/projects/ContextScore/Train/FragileSites/FragileSites_merged.bed"
    # fragile_sites_bed =
    # "/mnt/isilon/wang_lab/perdomoj/projects/ContextScore/Train/FragileSites/FragileSites_merged.bed"
    # fragile_sites_bed =
    # "/mnt/isilon/wang_lab/perdomoj/projects/ContextScore/Train/FragileSites/FragileSites_merged.bed"
    fragile_sites_bed = "fragileSites.bed"
    logging.info('Annotating the fragile sites using the BED file (GRCh38): %s', fragile_sites_bed)

    logging.info('Annotating fragile sites in true positives.')
    tp_fs_dir = os.path.join(output_directory_annovar, 'TP_FS')
    if not os.path.exists(tp_fs_dir):
        os.makedirs(tp_fs_dir)

    annotate_bed(true_positives_file, annovar_path, db_path, tp_fs_dir, fragile_sites_bed)
    # Output is bed.hg38_bed
    tp_fs_annotation = os.path.join(tp_fs_dir, 'bed.hg38_bed')
    if not os.path.exists(tp_fs_annotation):
        logging.error('Annotation file does not exist: %s', tp_fs_annotation)
        logging.error('Please check the ANNOVAR path and database path.')
        sys.exit(1)
    logging.info('Successfully annotated true positives to file: %s', tp_fs_annotation)

    logging.info('Annotating fragile sites in false positives.')
    fp_fs_dir = os.path.join(output_directory_annovar, 'FP_FS')
    if not os.path.exists(fp_fs_dir):
        os.makedirs(fp_fs_dir)

    annotate_bed(false_positives_file, annovar_path, db_path, fp_fs_dir, fragile_sites_bed)
    # Output is bed.hg38_bed
    fp_fs_annotation = os.path.join(fp_fs_dir, 'bed.hg38_bed')
    if not os.path.exists(fp_fs_annotation):
        logging.error('Annotation file does not exist: %s', fp_fs_annotation)
        logging.error('Please check the ANNOVAR path and database path.')
        sys.exit(1)
    logging.info('Successfully annotated false positives to file: %s', fp_fs_annotation)

    # ---------------------------------------
    # Annotate conserved regions using a UCSC Table Browser BED file for
    # phastCons100way
    phastCons_bed = "phastCons100wayHG38_fixed.bed"
    logging.info('Annotating conserved regions using the BED file (GRCh38): %s', phastCons_bed)

    logging.info('Annotating conserved regions in true positives.')
    tp_cr_dir = os.path.join(output_directory_annovar, 'TP_CR')
    if not os.path.exists(tp_cr_dir):
        os.makedirs(tp_cr_dir)
    annotate_bed(true_positives_file, annovar_path, db_path, tp_cr_dir, phastCons_bed)
    # Output is bed.hg38_bed
    tp_cr_annotation = os.path.join(tp_cr_dir, 'bed.hg38_bed')
    if not os.path.exists(tp_cr_annotation):
        logging.error('Annotation file does not exist: %s', tp_cr_annotation)

    # ---------------------------------------
    # Region-based annotation using ANNOVAR databases.

    # genomicSuperDups is the segmental duplication database.
    # phastConsElements46way is the conservation database.
    # cytoBand is used to annotate the centromere and telomere regions.

    # Download the segmental duplication database
    download_annovar_db(annovar_path, db_path, "genomicSuperDups", buildver)

    # Download the conservation database
    # download_annovar_db(annovar_path, db_path, "phastConsElements46way", buildver)

    # Download the cytoband database
    download_annovar_db(annovar_path, db_path, "cytoBand", buildver)

    logging.info('Annotating true positivess using ANNOVAR.')
    tp_anno_dir = os.path.join(output_directory_annovar, 'TP')
    if not os.path.exists(tp_anno_dir):
        os.makedirs(tp_anno_dir)

    annotate(true_positives_file, annovar_path, db_path, tp_anno_dir)
    tp_annotation = os.path.join(tp_anno_dir, 'regions.' + buildver + '_multianno.txt')

    # Check if the annotation file exists.
    if not os.path.exists(tp_annotation):
        logging.error('Annotation file does not exist: %s', tp_annotation)
        logging.error('Please check the ANNOVAR path and database path.')
        sys.exit(1)

    logging.info('Successfully annotated true positives to file: %s', tp_annotation)

    logging.info('Annotating false positives using ANNOVAR.')
    fp_anno_dir = os.path.join(output_directory_annovar, 'FP')
    if not os.path.exists(fp_anno_dir):
        os.makedirs(fp_anno_dir)
    annotate(false_positives_file, annovar_path, db_path, fp_anno_dir)
    fp_annotation = os.path.join(fp_anno_dir, 'regions.' + buildver + '_multianno.txt')

    # Check if the annotation file exists.
    if not os.path.exists(fp_annotation):
        logging.error('Annotation file does not exist: %s', fp_annotation)
        logging.error('Please check the ANNOVAR path and database path.')
        sys.exit(1)

    logging.info('Successfully annotated false positives to file: %s', fp_annotation)

    # Now extract features into a combined dataframe.
    columns = [
        "label",
        "chrom",
        "start",
        "sv_length",
        "sv_type",
        "segdup",
        "telomere",
        "centromere",
        "fragile_site",
        "conserved_region"
    ]

    

    # BELOW IS A WIP
    # -------------------------------


    # logging.info('Output directory: %s', output_directory)
    # logging.info('ANNOVAR path: %s', annovar_path)
    # logging.info('ANNOVAR database path: %s', db_path)

    # # Check if the output directory exists.
    # if not os.path.exists(output_directory):
    #     logging.info('Creating the output directory.')
    #     os.makedirs(output_directory)

    # model = train(tp_files, fp_files)

    logging.info('All complete!')

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
    parser.add_argument("--tpbed", required=True, help="Directory containing benchmark VCF files of real SVs (true positives and false negatives)")
    parser.add_argument("--fpbed", required=True, help="Directory containing false positive VCF files from running the caller on normal samples")
    parser.add_argument("--outdiranno", required=True, help="Output directory for saving the ANNOVAR annotations")
    parser.add_argument("--outdir", required=True, help="Output directory for saving the model")
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
    run(args.tpbed, args.fpbed, args.outdir, args.outdiranno, args.annovar, args.annovar_db)
    # run(tp_filepath, fp_filepath, output_dir)
    logging.info('done.')
