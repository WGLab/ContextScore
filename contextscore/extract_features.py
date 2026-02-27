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

import pandas as pd
import numpy as np

def normalize_column(df, column):
    """Normalize a column using z-score normalization."""
    mean = df[column].mean()
    std = df[column].std()
    df[column] = (df[column] - mean) / std
    return df

def add_interaction_terms(df):
    """Add interaction terms to the dataframe."""

    # Replace cluster_size with log transformed values to reduce the range.
    # df['log_cs'] = np.log1p(np.abs(df['cluster_size']))

    # df['log_rd'] = np.log1p(np.abs(df['read_depth']))
    # Log-transform the sv_length column to reduce the range.
    # df['log_svlen'] = np.log1p(np.abs(df['sv_length']))

    # Log-transform the read_depth column to reduce the range.
    # df['log_rd'] = np.log1p(np.abs(df['read_depth']))

    # Log-transform the cluster_size column to reduce the range.
    # df['log_cs'] = np.log1p(np.abs(df['cluster_size']))

    # Add a feature for whether the SV is a CNV (DUP, DEL with non-zero HMM log
    # likelihood).
    # df['is_cnv_hmm'] = df['sv_type'].apply(lambda x: 1 if x in [0, 1] else 0)  # Assuming 0 is DEL and 1 is DUP
    # df['is_cnv_hmm'] = df['is_cnv_hmm'] & (df['hmm_llh'] != 0)

    # Cluster size * hmm_llh interaction term
    df['cs_hmm'] = df['cluster_size'] * df['hmm_llh']

    # Replace hmm_llh with likelihood
    # df['hmm_llh'] = np.clip(np.exp(df['hmm_llh']), 1e-6, 0.999999)


    # Boolean for whether the SV is an inversion (INV).
    # df['is_inv'] = df['sv_type'].apply(lambda x: 1 if x == 2 else 0)  # Assuming 2 is INV

    # SV length interaction terms
    # df['svlenkb_cs'] = np.abs(df['sv_length']) / 1000 * df['cluster_size']
    # df['svlenkb_rd'] = np.abs(df['sv_length']) / 1000 * df['read_depth']
    # df['svlenkb_hmm'] = np.abs(df['sv_length']) / 1000 * df['hmm_llh']
    # df['hmm_llh_scaled'] = df['hmm_llh'] / np.log1p(np.abs(df['sv_length']))

    # df['hmm_llh_scaled'] = df['hmm_llh'] / (np.log1p(np.abs(df['sv_length'])))
    # df['hmm_llh_per_kb'] = df['hmm_llh_scaled'] / (np.abs(df['sv_length']) / 1000 + 1e-6)

    # Cluster size / read depth interaction terms
    # df['cs_rd'] = df['cluster_size'] / (df['read_depth'] + 1e-6)

    # Segdup * HMM llh interaction term
    # df['segdup_hmm'] = df['segdup'] * df['hmm_llh']

    # Segdup * cs/rd interaction term
    # df['segdup_cs_rd'] = df['segdup'] * df['cs_rd']

    # Simple repeat * cs/rd interaction terfm
    # df['simple_repeat_cs_rd'] = df['simpleRepeat'] * df['cs_rd']

    # Segdup * sv_length interaction term
    # df['segdup_svlen'] = df['segdup'] * np.abs(df['sv_length'])

    # Replace nans in segdup_hmm with 0
    # df['segdup_hmm'] = df['segdup_hmm'].fillna(0)

    # Segdup * cs
    df['segdup_cs'] = df['segdup'] * df['cluster_size']

    # Segdup * rd
    df['segdup_rd'] = df['segdup'] * df['read_depth']

    # Simple repeat * cs
    df['simple_repeat_cs'] = df['simpleRepeat'] * df['cluster_size']

    # Simple repeat * rd
    df['simple_repeat_rd'] = df['simpleRepeat'] * df['read_depth']

    # Fragile site * cs
    df['fragile_site_cs'] = df['fragile_site'] * df['cluster_size']

    # Fragile site * rd
    df['fragile_site_rd'] = df['fragile_site'] * df['read_depth']

    # Drop the segdup column
    # df.drop(columns=['segdup'], inplace=True)

    # # Drop the simple_repeat column
    # df.drop(columns=['simpleRepeat'], inplace=True)

    # # Drop the fragile_site column
    # df.drop(columns=['fragile_site'], inplace=True)

    # Drop cluster_size
    # df.drop(columns=['cluster_size'], inplace=True)

    # Drop the simple_repeat column
    # df.drop(columns=['simpleRepeat'], inplace=True)

    # Drop sv_type
    # df.drop(columns=['sv_type'], inplace=True)
    
    # ---
    # Cluster size per kb
    # df['cs_per_kb'] = df['cluster_size'] / (np.abs(df['sv_length']) / 1000 + 1e-6)

    # # Read depth per kb
    # df['rd_per_kb'] = df['read_depth'] / (np.abs(df['sv_length']) / 1000 +
    # 1e-6)
    # ---

    # Segmental duplication interaction terms
    # CNVs are mostly in segmental duplications, so we can use the
    # segmental duplication score to create interaction terms.
    # df['is_dup_and']

    # Cluster size * sv_length
    # df['log_svlen_cs'] = df['log_svlen'] + df['log_cs']

    # HMM log likelihood * sv_length
    # df['log_svlen_hmm'] = df['log_svlen'] + df['hmm_llh']

    # Remove log_cs
    # df.drop(columns=['log_cs'], inplace=True)

    # Read depth * sv_length
    # df['log_svlen_rd'] = df['log_svlen'] + df['log_rd']

    # Remove the log_svlen, log_rd, and log_cs columns, keeping the interaction
    # terms only.
    # df.drop(columns=['log_svlen', 'log_rd', 'log_cs'], inplace=True)

    # df['rd_cs'] = df['read_depth'] * df['cluster_size']
    # df['svlen_hmm'] = df['log_svlen'] * df['hmm_llh']
    # df['cs_hmm'] = df['cluster_size'] * df['hmm_llh']
    # df['rd_hmm'] = df['read_depth'] * df['hmm_llh']
    # df['hmm_per_kb'] = df['hmm_llh'] / (np.abs(df['sv_length']) / 1000 + 1)

    return df

def add_overlap_count(df, chrom_col='chrom', start_col='start', end_col='end'):
    """Add 'overlap_count' = number of other SVs on same chr that overlap each SV."""
    out = pd.Series(0, index=df.index, dtype=np.int32)

    for chrom, group in df.groupby(chrom_col, sort=False):
        starts = group[start_col].to_numpy()
        ends   = group[end_col].to_numpy()

        # overlap if start_i < end_j  AND  start_j < end_i
        overlap_matrix = (starts[:, None] < ends[None, :]) & \
                         (starts[None, :] < ends[:, None])

        # subtract 1 to drop the self-overlap on the diagonal
        counts = overlap_matrix.sum(axis=1) - 1
        out.loc[group.index] = counts.astype(np.int32)

    df['overlap_count'] = out
    return df


def extract_features(input_bed, annovar_path, db_path, outdiranno, buildversion='hg38'):
    """Extract the features from the BED file, columns are in the first row:
    chrom, start, end, sv_type, sv_length, genotype, read_depth, hmm_llh, aln_type, cluster_size
    """
    logging.info('Extracting features from the BED file %s', input_bed)

    # Get the number of columns in the BED file.
    with open(input_bed, 'r') as f:
        first_line = f.readline().strip()
        num_columns = len(first_line.split('\t'))
        logging.info('Number of columns in the BED file: %d', num_columns)
    
    training_format = False
    if num_columns == 12:  # Standard training format.
        training_format = True
        logging.info('Training format detected.')
    elif num_columns == 13:  # Contains additional 'id' column.
        logging.info('Prediction format detected.')

    # Read in the BED file.
    if training_format:
        bed_df = pd.read_csv(input_bed, sep='\t', header=0, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                         names=['chrom', 'start', 'end', 'sv_type', 'sv_length', 'genotype', 'read_depth', 'hmm_llh', 'aln_type', 'cluster_size', 'cn_state', 'aln_offset'],
                            dtype={'chrom': str, 'start': np.int32, 'end': np.int32, 'sv_type': str, 'sv_length': np.int32, 'genotype': str, 'read_depth': np.int32, 'hmm_llh': np.float32, 'aln_type': str, 'cluster_size': np.int32, 'cn_state': np.int32, 'aln_offset': np.int32})
    else:
        bed_df = pd.read_csv(input_bed, sep='\t', header=0, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                         names=['chrom', 'start', 'end', 'sv_type', 'sv_length', 'genotype', 'read_depth', 'hmm_llh', 'aln_type', 'cluster_size', 'cn_state', 'aln_offset', 'id'],
                         dtype={'chrom': str, 'start': np.int32, 'end': np.int32, 'sv_type': str, 'sv_length': np.int32, 'genotype': str, 'read_depth': np.int32, 'hmm_llh': np.float32, 'aln_type': str, 'cluster_size': np.int32, 'cn_state': np.int32, 'aln_offset': np.int32, 'id': np.int32})

    # Drop the genotype column and cn_state columns (due to redundancy).
    bed_df.drop(columns=['genotype', 'cn_state'], inplace=True)

    # Create alignment type feature, 0 for CIGAR alignment types (contains
    # CIGAR), 1 for CIGARCLIP (contains CIGARCLIP), 2 for SPLIT alignment (all
    # others)
    bed_df['call_type'] = bed_df['aln_type'].apply(lambda x: 1 if 'CIGARCLIP' in x else (0 if 'CIGAR' in x else 2))
    # Change call type to categorical.
    bed_df['call_type'] = bed_df['call_type'].astype('category')

    # Drop the original aln_type column.
    bed_df.drop(columns=['aln_type'], inplace=True)

    # Drop the sv_length column since it is noisy
    bed_df.drop(columns=['sv_length'], inplace=True)

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

    bed_df['sv_type_str'] = bed_df['sv_type'].astype(str)

    # Map the SV types to numbers.
    bed_df['sv_type'] = bed_df['sv_type'].map(sv_type_map).astype('category')

    # Check if any features are missing.
    if bed_df.isnull().values.any():
        logging.error('Features are missing.')

        # Get the rows with missing features.
        missing_features = bed_df[bed_df.isnull().any(axis=1)]

        # Print the rows with missing features.
        logging.error(missing_features)
        sys.exit(1)

    # Add annotations to the features.
    bed_df = add_annotations(bed_df, input_bed, annovar_path, db_path, outdiranno, buildversion, training_format)
    logging.info('Added ANNOVAR annotations to the features. Updated columns: %s', bed_df.columns)

    # Print the number of NaN values
    logging.info('Number of NaN values: %d', bed_df.isnull().sum().sum())

    # -------------------------------------------------------------------
    # Fix the chromosome names to all start with 'chr' if they don't already.
    bed_df['chrom'] = bed_df['chrom'].apply(lambda x: 'chr' + x if not x.startswith('chr') else x)

    # Drop telomere and centromere columns (they don't affect predictions).
    bed_df.drop(columns=['telomere', 'centromere'], inplace=True)

    # Return the features.
    return bed_df


def run_bedtools_intersect(input_bed, table_bed, training_format=False):
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
        if training_format:
            annotated_bed = pd.read_csv(
                StringIO(result.stdout),
                sep='\t',
                header=None,
                names=["chrom", "start", "end", "chr_anno", "start_anno", "end_anno", "name"],
                usecols=[0, 1, 2, 12, 13, 14, 15],
                dtype={'chrom': str, 'start': np.int32, 'end': np.int32, 'chr_anno': str, 'start_anno': np.int32, 'end_anno': np.int32, 'name': str}
            )
        else:
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


    # Post-processing the features:
    # Cap hmm log likelihood to avoid extreme values.
    df['hmm_llh'] = np.clip(df['hmm_llh'], -1e6, 0)

    # Update hmm_llh, set 0 to np.nan
    df['hmm_llh'] = df['hmm_llh'].replace(0, np.nan)

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


def download_annovar_db(annovar_path, db_path, db_name, buildversion='hg38'):
    """Download the ANNOVAR database if it does not exist."""
    logging.info('Downloading the database:' + db_name + ' for build version: ' + buildversion)
    cmd = [
        f"{annovar_path}/annotate_variation.pl",
        "-buildver", buildversion,
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


def annotate(annovar_input, annovar_path, db_path, output_dir, buildversion='hg38'):
    """Annotate regions."""
    logging.info('Annotating regions using ANNOVAR.')

    annotations_dir = os.path.join(output_dir, 'regions')
    logging.info('Creating the output directory: %s', annotations_dir)
    cmd = [
        f"{annovar_path}/table_annovar.pl",
        annovar_input,
        db_path,
        "--buildver " + buildversion,
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
        pass
        # Handle the case where chrom_dict[chrom] is not defined.
        # logging.warning('chrom_dict[%s] is not defined.', chrom)
        # logging.warning('Cytoband: %s', cytoband)
        # logging.warning('chrom_dict[%s]: %s', chrom, chrom_dict.get(chrom, 'Not found'))

    except TypeError:
        pass
        # Handle the case where telomerep is not defined.
        # logging.warning('chrom_dict[%s] does not have telomerep defined.', chrom)
        # logging.warning('Cytoband: %s', cytoband)
        # logging.warning('chrom_dict[%s]: %s', chrom, chrom_dict[chrom])

    return is_telomere, is_centromere


def add_annotations(data, input_bed, annovar_path, db_path, anno_outdir, buildversion='hg38', training_format=False):
    """Add annotations to the features."""
    logging.info('Adding annotations to the features.')

    # ---------------------------------------------------------------
    # Annotate the fragile sites using a BED file from HumCFS (GRCh38/hg38).
    # https://webs.iiitd.edu.in/raghava/humcfs/download.html
    # ANNOVAR instructions are here:
    # https://annovar.openbioinformatics.org/en/latest/user-guide/region/
    if buildversion == 'hg38':
        fragile_sites_bed="/mnt/isilon/wang_lab/perdomoj/projects/ContextScore/Train/FragileSites/FragileSites_merged.bed"
    elif buildversion == 'hg19':
        fragile_sites_bed="/mnt/isilon/wang_lab/perdomoj/projects/ContextScore/Train/FragileSites/FragileSites_hg19.bed"
    else:
        logging.error('Unsupported build version: %s. Please use hg38 or hg19.', buildversion)
        sys.exit(1)

    logging.info('Annotating the fragile sites using the BED file (GRCh38): %s', fragile_sites_bed)
    fragile_sites_df = run_bedtools_intersect(input_bed, fragile_sites_bed, training_format)

    # Merge the fragile sites annotations with the true positive data.
    data['fragile_site'] = data.merge(fragile_sites_df, on=['chrom', 'start', 'end'], how='left')['chr_anno'].notna()

    logging.info('Number of records with fragile sites: %d', data['fragile_site'].sum())
    logging.info('Total number of records: %d', data.shape[0])

    # ---------------------------------------------------------------
    # Annotate conserved regions using a UCSC Table Browser BED file for
    # phastCons100way (GRCh38/hg38).
    if buildversion == 'hg38':
        phastCons_bed = "/mnt/isilon/wang_lab/perdomoj/data/UCSC_Tables/phastCons100way_hg38.bed"
    elif buildversion == 'hg19':
        phastCons_bed = "/mnt/isilon/wang_lab/perdomoj/data/UCSC_Tables/phastCons100way_hg19.bed"
    else:
        logging.error('Unsupported build version: %s. Please use hg38 or hg19.', buildversion)
        sys.exit(1)
    logging.info('Annotating conserved regions using the BED file (GRCh38): %s', phastCons_bed)
    phastCons_df = run_bedtools_intersect(input_bed, phastCons_bed, training_format)

    # Merge the phastCons annotations with the true positive data.
    data['phastCons'] = data.merge(phastCons_df, on=['chrom', 'start', 'end'], how='left')['chr_anno'].notna()

    logging.info('Number of records with conserved regions: %d', data['phastCons'].sum())
    logging.info('Total number of records: %d', data.shape[0])

    # ---------------------------------------------------------------
    # Annotate simple repeats using a UCSC Table Browser BED file for
    # simpleRepeat (GRCh38/hg38).
    if buildversion == 'hg38':
        simpleRepeat_bed = "/mnt/isilon/wang_lab/perdomoj/data/UCSC_Tables/simple_repeats_hg38.bed"
    elif buildversion == 'hg19':
        simpleRepeat_bed = "/mnt/isilon/wang_lab/perdomoj/data/UCSC_Tables/simple_repeats_hg19.bed"
    else:
        logging.error('Unsupported build version: %s. Please use hg38 or hg19.', buildversion)
        sys.exit(1)
    logging.info('Annotating simple repeats using the BED file (GRCh38): %s', simpleRepeat_bed)
    simpleRepeat_df = run_bedtools_intersect(input_bed, simpleRepeat_bed, training_format)

    # Merge the simpleRepeat annotations with the true positive data.
    data['simpleRepeat'] = data.merge(simpleRepeat_df, on=['chrom', 'start', 'end'], how='left')['chr_anno'].notna()

    logging.info('Number of records with simple repeats: %d', data['simpleRepeat'].sum())
    logging.info('Total number of records: %d', data.shape[0])

    # ---------------------------------------------------------------
    # Annotate the SVs using ANNOVAR.
    
    # Download the segmental duplication database
    download_annovar_db(annovar_path, db_path, "genomicSuperDups", buildversion)

    # Download the cytoband database
    download_annovar_db(annovar_path, db_path, "cytoBand", buildversion)

    # Set up a dictionary for each chromosome, mapping the cytoband to the
    # centromere and telomere regions.
    cytoband_file = "/home/perdomoj/github/ContextScore/data/hg38_cytoband.txt"  # Downloaded from UCSC.
    cytoband_dict = read_cytoband_file(cytoband_file)

    logging.info('Converting the true positive BED file to ANNOVAR input format.')
    annovar_file = bed_to_annovar_input(input_bed)

    logging.info('Annotating the SVs using ANNOVAR.')
    if not os.path.exists(anno_outdir):
        os.makedirs(anno_outdir)

    annotate(annovar_file, annovar_path, db_path, anno_outdir, buildversion)

    # anno_file = os.path.join(anno_outdir, 'regions.hg38_multianno.txt')
    anno_file = os.path.join(anno_outdir, 'regions.{}_multianno.txt'.format(buildversion))
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

    # Print the first 20 segdup values.
    # logging.info('First 20 values of the segdup column: %s', data['genomicSuperDups'].head(20))

    # Extract segmental duplication scores.
    # def extract_max_score(score_series):
    #     """Extract and return the maximum Score= value from a series."""
    #     scores = score_series.str.extract(r'Score=([\d\.]+)')[0].dropna().astype(float)
    #     return scores.max() if not scores.empty else 0
    
    # Extract segmental duplication scores.
    def extract_scores(score_str):
        """Extract and return the segmental duplication scores from a string."""
        if pd.isna(score_str) or score_str == '.':
            return 0
        # Extract the Score= value from the string.
        try:
            score = score_str.split('Score=')[1].split(';')[0]
        except IndexError:
            logging.warning('Score= not found in the string: %s', score_str)
            return 0
        return float(score) if score else 0

    # Extract the segmental duplication scores.
    data['segdup'] = data['genomicSuperDups'].apply(extract_scores)

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

    logging.info('Dropped the unnecessary columns. Current columns: %s', data.columns)

    logging.info('Number of records after adding annotations: %d', data.shape[0])
    # logging.info('First 5 rows of the data after adding annotations:\n%s', data.head())

    return data
