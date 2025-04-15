"""
extract_features.py: Extract features from the input VCF file.

Usage:
    extract_features.py <input>

Arguments:
    <input>     Path to the input VCF file.

Output:
    A dataframe with a column for each feature.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import subprocess
import joblib
from io import StringIO


def read_cytoband_file(cytoband_file):
    """Get the centromere and telomere regions for each chromosome."""
    cytobands = pd.read_csv(cytoband_file, sep='\t', header=None, names=["chrom", "start", "end", "name", "gieStain"])
    chrom_dict = {}
    for chrom in cytobands['chrom'].unique():
        
        # Skip chrM, and other non-standard chromosomes.
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

# tp_data, tp_bed, annovar_path, db_path, tp_anno_outdir

def extract_features(input_bed, annovar_path, db_path, outdiranno):
    """Extract the features from the BED file, columns are in the first row:
    chrom, start, end, sv_type, sv_length, genotype, read_depth, hmm_llh, aln_type, cluster_size
    """
    logging.info('Extracting features from the BED file %s', input_bed)

    # Load a dictionary mapping chromosome names to numbers.
    # chrom_dict_path="/mnt/isilon/wang_lab/perdomoj/projects/ContextScore/Train/Model/chrom_map.pkl"
    # chrom_dict = joblib.load(chrom_dict_path)

    # Read in the BED file.
    bed_df = pd.read_csv(input_bed, sep='\t', header=0, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                         names=['chrom', 'start', 'end', 'sv_type', 'sv_length', 'genotype', 'read_depth', 'hmm_llh', 'aln_type', 'cluster_size', 'cn_state', 'aln_offset', 'id'],
                         dtype={'chrom': str, 'start': np.int32, 'end': np.int32, 'sv_type': str, 'sv_length': np.int32, 'genotype': str, 'read_depth': np.int32, 'hmm_llh': np.float32, 'aln_type': str, 'cluster_size': np.int32, 'cn_state': np.int32, 'aln_offset': np.int32, 'id': np.int32})

    logging.info("[TEST1] columns in the BED file: %s", bed_df.columns)

    # Drop the genotype column and cn_state columns (due to redundancy).
    bed_df.drop(columns=['genotype', 'cn_state'], inplace=True)
    logging.info('[TEST] Dropped the genotype and cn_state columns. Current columns: %s', bed_df.columns)

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
    # genotype_map = {
    #     '0/0': 0,
    #     '0/1': 1,
    #     '1/1': 2,
    #     './.': 3
    # }

    # Map the genotypes to numbers.
    # bed_df['genotype'] = bed_df['genotype'].map(genotype_map)

    # Check if any features are missing.
    if bed_df.isnull().values.any():
        logging.error('Features are missing.')

        # Get the rows with missing features.
        missing_features = bed_df[bed_df.isnull().any(axis=1)]

        # Print the rows with missing features.
        logging.error(missing_features)
        sys.exit(1)

    # Add annotations to the features.
    add_annotations(bed_df, input_bed, annovar_path, db_path, outdiranno)

    # Finally map chromosome names to numbers.
    # Load a dictionary mapping chromosome names to numbers.
    chrom_dict_path="/mnt/isilon/wang_lab/perdomoj/projects/ContextScore/Train/Model/chrom_map.pkl"
    chrom_dict = joblib.load(chrom_dict_path)

    # Print the number of NaN values
    logging.info('Number of NaN values: %d', bed_df.isnull().sum().sum())

    # Map the chromosome names to numbers.
    bed_df['chrom'] = bed_df['chrom'].map(chrom_dict)

    # Actually drop the chrom, start, end columns.
    bed_df.drop(columns=['chrom', 'start', 'end'], inplace=True)

    logging.info('[TEST] Dropped the chrom, start, end columns. Final columns (TP): %s', bed_df.columns)

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
            usecols=[0, 1, 2, 13, 14, 15, 16],#12, 13, 14, 15], #10, 11, 12, 13],
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
    # df = pd.read_csv(bed_file, sep='\t', header=None, comment='#',
    # names=["CHROM", "POS", "END", "SVTYPE", "SVLEN"], skiprows=1)
    # df = pd.read_csv(bed_file, sep='\t', header=0, comment='#',
    #                  names=["CHROM", "POS", "END", "SVTYPE", "SVLEN"], usecols=[0, 1, 2, 3, 4],
    #                  dtype={'CHROM': str, 'POS': np.int32, 'END': np.int32,
    #                  'SVTYPE': str, 'SVLEN': np.int32})
    df = pd.read_csv(bed_file, sep='\t', usecols=[0, 1, 2],
                     names=["CHROM", "POS", "END"],
                     dtype={'CHROM': str, 'POS': np.int32, 'END': np.int32})
    
    # Check if the BED file is empty.
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

        if 'telomereq' in chrom_dict[chrom] and chrom_dict[chrom]['telomereq'] in cytoband:
            is_telomere = True

        if 'centromerep' in chrom_dict[chrom] and chrom_dict[chrom]['centromerep'] in cytoband:
            is_centromere = True

        if 'centromereq' in chrom_dict[chrom] and chrom_dict[chrom]['centromereq'] in cytoband:
            is_centromere = True

    except KeyError:
        # Handle the case where chrom_dict[chrom] is not defined.
        logging.warning('chrom_dict[%s] is not defined.', chrom)
        logging.warning('Cytoband: %s', cytoband)
        logging.warning('chrom_dict[%s]: %s', chrom, chrom_dict.get(chrom, 'Not found'))

    except TypeError:
        # Handle the case where telomerep is not defined.
        logging.warning('chrom_dict[%s] does not have telomerep defined.', chrom)
        logging.warning('Cytoband: %s', cytoband)
        logging.warning('chrom_dict[%s]: %s', chrom, chrom_dict[chrom])
    #     is_telomere = False
    # if 'telomereq' in chrom_dict[chrom] and chrom_dict[chrom]['telomereq'] in cytoband:
    #     is_telomere = True
    # if 'centromerep' in chrom_dict[chrom] and chrom_dict[chrom]['centromerep'] in cytoband:
    #     is_centromere = True
    # if 'centromereq' in chrom_dict[chrom] and chrom_dict[chrom]['centromereq'] in cytoband:
    #     is_centromere = True
    
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
