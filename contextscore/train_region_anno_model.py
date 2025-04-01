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
import pybedtools  # For annotating BED files.
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from extract_features import extract_features

# Set up the logger.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_cytoband_file(cytoband_file):
    """Get the centromere and telomere regions for each chromosome."""
    cytobands = pd.read_csv(cytoband_file, sep='\t', header=None, names=["chrom", "start", "end", "name", "gieStain"])
    chrom_dict = {}
    for chrom in cytobands['chrom'].unique():
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

def annotate_bed(input_bed, table_bed):
    """Annotate a BED file using bedtools."""
    logging.info('Input BED file: %s', input_bed)
    logging.info('Table BED file: %s', table_bed)

    input_bed = pybedtools.BedTool(input_bed)
    input_count = input_bed.count()

    table_bed = pybedtools.BedTool(table_bed)
    table_count = table_bed.count()

    # Perform the annotation using bedtools intersect.
    logging.info('Annotating the BED file using bedtools intersect.')
    annotated_bed = input_bed.intersect(table_bed, wa=True, wb=True)
    df = annotated_bed.to_dataframe(
        names=["chrom", "start", "end", "chr_anno", "start_anno", "end_anno", "name"],
        usecols=[0, 1, 2, 3, 4, 5, 6],  # Only keep the relevant columns.
    )
    
    # Print first 5 rows of the annotated dataframe.
    logging.info('Annotated BED dataframe:\n%s', df.head())
    anno_count = df.shape[0]
    logging.info('Number of rows in the input BED file: %d', input_count)
    logging.info('Number of rows in the table BED file: %d', table_count)
    logging.info('Number of rows in the annotated BED dataframe: %d', anno_count)
    logging.info("Annotated " + str(anno_count) + " rows from the input BED file with " + str(table_count) + " rows from the table BED file (Percentage: %.2f%%)" % ((anno_count / input_count) * 100))

    return df

def add_annotations(df, annotation_file):
    """Add annotations to the dataframe from the ANNOVAR output file."""
    logging.info('Adding annotations from: %s', annotation_file)

    # Read the annotation file into a dataframe.
    anno_df = pd.read_csv(annotation_file, sep='\t', header=None, names=["chrom", "start", "end", "annotation"], comment='#')
    logging.info('Annotation dataframe:\n%s', anno_df.head())


def add_telomere_centromere_segdup(input_df, anno_df, cytoband_dict):
    """Add telomere, centromere, and segmental duplication annotations to the
    input dataframe."""
    
    # Merge the input dataframe with the annotation dataframe
    logging.info('Merging the input dataframe with the annotation dataframe.')
    merged_df = input_df.merge(anno_df, left_on=['chrom', 'start', 'end'], right_on=['Chr', 'Start', 'End'], how='left')
    logging.info('Merged dataframe:\n%s', merged_df.head())

    # Extract segmental duplication scores
    def extract_max_score(score_series):
        """Extract and return the maximum Score= value from a series."""
        scores = score_series.str.extract(r'Score=([\d\.]+)')[0].dropna().astype(float)
        return scores.max() if not scores.empty else np.nan
        
    # Get the maximum score for segmental duplications.
    merged_df['segdup'] = extract_max_score(merged_df['genomicSuperDups'])

    # Determine centromere and telomere regions using vectorized operations.
    def get_cyto_info(row):
        """Get telomere and centromere information for a row."""
        if pd.notna(row['cytoBand']):
            return get_cytoband_is_c_t(cytoband_dict, row['chrom'], row['cytoBand'])
        return (np.nan, np.nan)
    
    merged_df[['telomere', 'centromere']] = merged_df.apply(get_cyto_info, axis=1, result_type='expand')

    # Select only necessary columns for the final output.
    final_cols = list(input_df.columns) + ['segdup', 'telomere', 'centromere']
    final_df = merged_df[final_cols]
    logging.info('Final dataframe:\n%s', final_df.head())

    return final_df

    # for index, row in input_df.iterrows():
    #     chrom = row['chrom']
    #     start = row['start']
    #     end = row['end']

    #     # Get all corresponding rows with the same chromosome and start/end positions.
    #     matching_rows = anno_df[(anno_df['Chr'] == chrom) & (anno_df['Start'] == start) & (anno_df['End'] == end)]
    #     if matching_rows.empty:
    #         logging.warning('No matching annotation found for index %d: Chromosome: %s, Start: %d, End: %d', index, chrom, start, end)
    #         continue  # Skip if no matching annotation is found.

    #     # Check if any rows have the annotation for segmental duplication.
    #     segdup = matching_rows['genomicSuperDups'].dropna().any()  # Check if any value is not NaN.      
    #     if segdup:
    #         # Get the value after Score=
    #         max_score = 0
    #         score_found = False
    #         for score in matching_rows['genomicSuperDups']:
    #             if score == '.':
    #                 continue

    #             # Check if the score starts with 'Score='.
    #             elif score.startswith('Score='):
    #                 score_found = True
    #                 try:
    #                     score_value = score.split('Score=')[1]
    #                     if ';' in score_value:
    #                         score_value = score_value.split(';')[0]
    #                     score_value = float(score_value)  # Convert to float.

    #                     # Keep track of the maximum score found.
    #                     if score_value > max_score:
    #                         max_score = score_value

    #                 except (IndexError, ValueError):
    #                     logging.warning('Could not parse Score from: %s', score)

    #         if score_found:
    #             input_df.at[index, 'segdup'] = max_score

    #     # Check if the region is a telomere or centromere.
    #     cytoband = matching_rows['cytoBand'].dropna().unique()
    #     if cytoband.size > 0:
    #         # Get the first cytoband annotation (if multiple are present).
    #         cytoband = cytoband[0]  # Take the first one if there are multiple.
            
    #         # Check if the cytoband is a telomere or centromere.
    #         is_telomere, is_centromere = get_cytoband_is_c_t(cytoband_dict, chrom, cytoband)
            
    #         # Update the telomere and centromere columns.
    #         input_df.at[index, 'telomere'] = is_telomere
    #         input_df.at[index, 'centromere'] = is_centromere


# Run the program.
def run(tp_bed, fp_bed, output_directory, output_directory_annovar, annovar_path, db_path):
    """Train the binary classification model."""

    # Set up a dictionary for each chromosome, mapping the cytoband to the
    # centromere and telomere regions.
    cytoband_file = "/home/perdomoj/github/ContextScore/data/hg38_cytoband.txt"  # Downloaded from UCSC.
    cytoband_dict = read_cytoband_file(cytoband_file)

    # TODO: Make this an input parameter.
    buildver = 'hg38'

    logging.info('Getting the true positive and false positive VCF files.')

    # Convert the BED files to ANNOVAR input format.
    logging.info('Converting the true positive BED file to ANNOVAR input format.')
    true_positives_file = bed_to_annovar_input(tp_bed)

    logging.info('Converting the false positive BED file to ANNOVAR input format.')
    false_positives_file = bed_to_annovar_input(fp_bed)

    # HPRC tracks:
    # https://genome.ucsc.edu/cgi-bin/hgTracks?hgsid=2497626981_YO5LtOenyXcMHylL5pvsY90WzIkJ&c=chr6&hgTracksConfigPage=configure&hgtgroup_hprc_close=0#hprcGroup
    # Current error with hprc90way Multiple Alignment download from UCSC (https://genome.ucsc.edu/cgi-bin/hgTables):
    # Can't start query:
    # select bin,chrom,chromStart,chromEnd,extFile,offset,score from hprc90way where chrom='chr1'
    # mySQL error 1064: You have an error in your SQL syntax; check the manual that corresponds to your MariaDB server version for the right syntax to use near 'offset,score from hprc90way where chrom='chr1'' at line 1 (profile=<noProfile>, host=localhost, db=hg38)

    # Annotate the fragile sites using a BED file from HumCFS (GRCh38/hg38).
    # https://webs.iiitd.edu.in/raghava/humcfs/download.html
    # ANNOVAR instructions are here: https://annovar.openbioinformatics.org/en/latest/user-guide/region/
    fragile_sites_bed="/mnt/isilon/wang_lab/perdomoj/projects/ContextScore/Train/FragileSites/FragileSites_merged.bed"
    logging.info('Annotating the fragile sites using the BED file (GRCh38): %s', fragile_sites_bed)

    logging.info('Annotating fragile sites in true positives.')
    tp_fragile_sites_df = annotate_bed(tp_bed, fragile_sites_bed)

    logging.info('Annotating fragile sites in false positives.')
    fp_fragile_sites_df = annotate_bed(fp_bed, fragile_sites_bed)

    # ---------------------------------------
    # Annotate conserved regions using a UCSC Table Browser BED file for
    # phastCons100way
    phastCons_bed = "/mnt/isilon/wang_lab/perdomoj/data/UCSC_Tables/phastCons100way_hg38.bed"
    logging.info('Annotating conserved regions using the BED file (GRCh38): %s', phastCons_bed)

    logging.info('Annotating conserved regions in true positives.')
    tp_cons_df = annotate_bed(tp_bed, phastCons_bed)

    logging.info('Annotating conserved regions in false positives.')
    fp_cons_df = annotate_bed(fp_bed, phastCons_bed)

    # ---------------------------------------
    # Annotate simple repeats using a UCSC Table Browser BED file for
    # simpleRepeat
    simpleRepeat_bed = "/mnt/isilon/wang_lab/perdomoj/data/UCSC_Tables/simple_repeats_hg38.bed"
    logging.info('Annotating simple repeats using the BED file (GRCh38): %s', simpleRepeat_bed)

    logging.info('Annotating simple repeats in true positives.')
    tp_sr_df = annotate_bed(tp_bed, simpleRepeat_bed)

    logging.info('Annotating simple repeats in false positives.')
    fp_sr_df = annotate_bed(fp_bed, simpleRepeat_bed)

    # ---------------------------------------
    # Region-based annotation using ANNOVAR databases.

    # genomicSuperDups is the segmental duplication database.
    # phastConsElements46way is the conservation database.
    # cytoBand is used to annotate the centromere and telomere regions.

    # Download the segmental duplication database
    download_annovar_db(annovar_path, db_path, "genomicSuperDups", buildver)

    # Download the cytoband database
    download_annovar_db(annovar_path, db_path, "cytoBand", buildver)

    logging.info('Annotating true positives using ANNOVAR.')
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

    # Read the true positive input into a dataframe.
    logging.info('Reading the true positive input file: %s', true_positives_file)
    tp_df = pd.read_csv(true_positives_file, sep='\t', header=None, names=["chrom", "start", "end"], usecols=[0, 1, 2])

    # Check if the true positive dataframe is empty.
    if tp_df.empty:
        logging.error('True positive dataframe is empty.')
        sys.exit(1)
    
    logging.info('True positive dataframe:\n%s', tp_df.head())

    # Read the false positive input into a dataframe.
    logging.info('Reading the false positive input file: %s', false_positives_file)
    fp_df = pd.read_csv(false_positives_file, sep='\t', header=None, names=["chrom", "start", "end"], usecols=[0, 1, 2])

    # Check if the false positive dataframe is empty.
    if fp_df.empty:
        logging.error('False positive dataframe is empty.')
        sys.exit(1)

    # Read the annovar output into a dataframe.
    logging.info('Reading the true positive annotation file: %s', tp_annotation)
    tp_anno_df = pd.read_csv(tp_annotation, sep='\t', header=0, comment='#')
    logging.info('True positive annotation header:\n%s', tp_anno_df.columns)
    logging.info('True positive annotation dataframe:\n%s', tp_anno_df.head())

    logging.info('Reading the false positive annotation file: %s', fp_annotation)
    fp_anno_df = pd.read_csv(fp_annotation, sep='\t', header=0, comment='#')
    logging.info('False positive annotation header:\n%s', fp_anno_df.columns)
    logging.info('False positive annotation dataframe:\n%s', fp_anno_df.head())

    # Add the labels to the dataframes.
    tp_df['label'] = 1
    tp_df['telomere'] = False
    tp_df['centromere'] = False
    tp_df['segdup'] = False
    tp_df['fragile_site'] = False
    tp_df['conserved_region'] = False
    tp_df['simple_repeat'] = False

    fp_df['label'] = 0
    fp_df['telomere'] = False
    fp_df['centromere'] = False
    fp_df['segdup'] = False
    fp_df['fragile_site'] = False
    fp_df['conserved_region'] = False
    fp_df['simple_repeat'] = False

    logging.info('Adding telomere, centromere, and segmental duplication annotations to true positives.')
    tp_df = add_telomere_centromere_segdup(tp_df, tp_anno_df, cytoband_dict)

    logging.info('Adding telomere, centromere, and segmental duplication annotations to false positives.')
    fp_df = add_telomere_centromere_segdup(fp_df, fp_anno_df, cytoband_dict)

    # Add the fragile site and conserved region annotations
    logging.info('Adding fragile site annotations to true positives.')
    tp_df['fragile_site'] = tp_df.merge(tp_fragile_sites_df, on=['chrom', 'start', 'end'], how='left')['chr_anno'].notna()
    
    logging.info('Updated df:\n%s', tp_df.head())

    logging.info('Adding fragile site annotations to false positives.')
    fp_df['fragile_site'] = fp_df.merge(fp_fragile_sites_df, on=['chrom', 'start', 'end'], how='left')['chr_anno'].notna()
    logging.info('Updated df:\n%s', fp_df.head())

    # Add the conserved region annotations.
    logging.info('Adding conserved region annotations to true positives.')
    tp_df['conserved_region'] = tp_df.merge(tp_cons_df, on=['chrom', 'start', 'end'], how='left')['chr_anno'].notna()

    logging.info('Adding conserved region annotations to false positives.')
    fp_df['conserved_region'] = fp_df.merge(fp_cons_df, on=['chrom', 'start', 'end'], how='left')['chr_anno'].notna()

    # Add the simple repeat annotations.
    logging.info('Adding simple repeat annotations to true positives.')
    tp_df['simple_repeat'] = tp_df.merge(tp_sr_df, on=['chrom', 'start', 'end'], how='left')['chr_anno'].notna()

    logging.info('Adding simple repeat annotations to false positives.')
    fp_df['simple_repeat'] = fp_df.merge(fp_sr_df, on=['chrom', 'start', 'end'], how='left')['chr_anno'].notna()

    # Check if the annotations were added correctly.
    logging.info('True positive dataframe after adding annotations:\n%s', tp_df.head())
    logging.info('TP fragile sites: %d', tp_df['fragile_site'].sum())
    logging.info('TP telomeres: %d', tp_df['telomere'].sum())
    logging.info('TP centromeres: %d', tp_df['centromere'].sum())
    logging.info('TP segmental duplications: %d', tp_df['segdup'].sum())
    logging.info('TP conserved regions: %d', tp_df['conserved_region'].sum())
    logging.info('TP simple repeats: %d', tp_df['simple_repeat'].sum())


    # Print a tab-delimited table with the number of rows in each category.
    logging.info('True positives:')
    logging.info('Total\tFragile Sites\tTelomeres\tCentromeres\tSegmental Duplications\tConserved Regions')
    logging.info('%d\t%d\t%d\t%d\t%d\t%d', tp_df.shape[0], tp_df['fragile_site'].sum(), tp_df['telomere'].sum(), tp_df['centromere'].sum(), tp_df['segdup'].sum(), tp_df['conserved_region'].sum())

    logging.info('False positives:')
    logging.info('Total\tFragile Sites\tTelomeres\tCentromeres\tSegmental Duplications\tConserved Regions')
    logging.info('%d\t%d\t%d\t%d\t%d\t%d', fp_df.shape[0], fp_df['fragile_site'].sum(), fp_df['telomere'].sum(), fp_df['centromere'].sum(), fp_df['segdup'].sum(), fp_df['conserved_region'].sum())

    # Save the same table to a file.
    annot_summary = 'TrainingAnnotationsSummary.tsv'
    logging.info('Saving the true positive summary to %s', annot_summary)
    with open(annot_summary, 'w') as f:
        f.write('True Positives\n')
        f.write('Total\tFragile Sites\tTelomeres\tCentromeres\tSegmental Duplications\tConserved Regions\n')
        f.write('%d\t%d\t%d\t%d\t%d\t%d\n' % (tp_df.shape[0], tp_df['fragile_site'].sum(), tp_df['telomere'].sum(), tp_df['centromere'].sum(), tp_df['segdup'].sum(), tp_df['conserved_region'].sum()))
        f.write('\n')
        f.write('False Positives\n')
        f.write('Total\tFragile Sites\tTelomeres\tCentromeres\tSegmental Duplications\tConserved Regions\n')
        f.write('%d\t%d\t%d\t%d\t%d\t%d\n' % (fp_df.shape[0], fp_df['fragile_site'].sum(), fp_df['telomere'].sum(), fp_df['centromere'].sum(), fp_df['segdup'].sum(), fp_df['conserved_region'].sum()))

    logging.info('Saved the summary to %s', annot_summary)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpbed", required=True, help="Directory containing benchmark VCF files of real SVs (true positives and false negatives)")
    parser.add_argument("--fpbed", required=True, help="Directory containing false positive VCF files from running the caller on normal samples")
    parser.add_argument("--outdiranno", required=True, help="Output directory for saving the ANNOVAR annotations")
    parser.add_argument("--outdir", required=True, help="Output directory for saving the model")
    parser.add_argument("--annovar", required=True, help="Path to ANNOVAR")
    parser.add_argument("--annovar_db", required=True, help="Path to ANNOVAR database")

    # Add flag for whether to train the genomic context model. If not specified,
    # the default is False (train the model using SV caller features).
    parser.add_argument("--train_genomic_context", action="store_true", help="Train the genomic context model", default=False)
    args = parser.parse_args()

    # Run the program.
    logging.info('Training the model...')
    run(args.tpbed, args.fpbed, args.outdir, args.outdiranno, args.annovar, args.annovar_db)
    logging.info('done.')
