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
import heapq
import numpy as np
import pandas as pd
import subprocess
from io import StringIO


def read_cytoband_file(cytoband_file):
    """Get the centromere and telomere regions for each chromosome."""
    cytobands = pd.read_csv(cytoband_file, sep='\t', header=0, names=["chrom", "start", "end", "name", "gieStain"], dtype={"chrom": str, "start": int, "end": int, "name": str, "gieStain": str})
    chrom_dict = {}
    for chrom in cytobands['chrom'].unique():
        
        # Skip chrM, and other non-standard chromosomes.
        if chrom == 'chrM':
            continue

        chrom_df = cytobands[cytobands['chrom'] == chrom].sort_values('start')
        # Store chromosome boundaries and terminal bands.
        chrom_dict[chrom] = {
            'chrom_start': int(chrom_df['start'].min()),
            'chrom_end': int(chrom_df['end'].max()),
            'telomerep_start': int(chrom_df.iloc[0]['start']),
            'telomerep_end': int(chrom_df.iloc[0]['end']),
            'telomereq_start': int(chrom_df.iloc[-1]['start']),
            'telomereq_end': int(chrom_df.iloc[-1]['end'])
        }

        # Identify centromeres from cytobands with gieStain == "acen".
        acen_df = chrom_df[chrom_df['gieStain'] == 'acen']
        centromere_p = acen_df[acen_df['name'].str.startswith('p', na=False)]
        centromere_q = acen_df[acen_df['name'].str.startswith('q', na=False)]
        if not centromere_p.empty:
            chrom_dict[chrom]['centromerep_start'] = int(centromere_p.iloc[0]['start'])
            chrom_dict[chrom]['centromerep_end'] = int(centromere_p.iloc[0]['end'])
        if not centromere_q.empty:
            chrom_dict[chrom]['centromereq_start'] = int(centromere_q.iloc[0]['start'])
            chrom_dict[chrom]['centromereq_end'] = int(centromere_q.iloc[0]['end'])

        # Combined centromere span (union of acen blocks) for distance calculation.
        if not acen_df.empty:
            chrom_dict[chrom]['centromere_start'] = int(acen_df['start'].min())
            chrom_dict[chrom]['centromere_end'] = int(acen_df['end'].max())

    return chrom_dict


def normalize_chrom_label(chrom):
    """Normalize chromosome labels for robust joins/lookups (e.g., 1 vs chr1)."""
    if pd.isna(chrom):
        return None
    chrom_str = str(chrom).strip()
    if not chrom_str:
        return None
    chrom_str = chrom_str[3:] if chrom_str.lower().startswith('chr') else chrom_str
    return chrom_str.upper()

def extract_features(input_bed, annovar_path, db_path, outdiranno, buildversion='hg38', sample_coverage=None):
    """Extract the features from the BED file, columns are in the first row:
    chrom, start, end, sv_type, sv_length, genotype, read_depth, hmm_llh, aln_type, cluster_size
    
    Args:
        sample_coverage (float): Required. Mean read depth coverage for the sample, used to normalize read_depth.
    """
    logging.info('Extracting features from the BED file %s', input_bed)
    
    if sample_coverage is None or sample_coverage <= 0:
        logging.error('sample_coverage is required and must be > 0')
        raise ValueError('sample_coverage is required and must be > 0')

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

    # Ensure SV length is positive
    bed_df['sv_length'] = bed_df['sv_length'].abs()

    # Throw error if any SV lengths are negative
    if (bed_df['sv_length'] < 0).any():
        logging.error('Negative SV lengths found in the BED file.')
        sys.exit(1)

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

    # Read depth normalized by sample coverage
    bed_df['read_depth_normalized'] = np.where(
        sample_coverage > 0,
        bed_df['read_depth'] / sample_coverage,
        bed_df['read_depth']
    )

    # Keep sv_length temporarily for later distance normalization
    # Will be dropped after all normalized features are created

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

    # Print the number of NaN values
    logging.info('Number of NaN values: %d', bed_df.isnull().sum().sum())

    # -------------------------------------------------------------------
    # Fix the chromosome names to all start with 'chr' if they don't already.
    bed_df['chrom'] = bed_df['chrom'].apply(lambda x: 'chr' + x if not x.startswith('chr') else x)

    # Drop telomere and centromere columns (they don't affect predictions).
    bed_df.drop(columns=['telomere', 'centromere'], inplace=True)

    # Drop the genotype column from the data.
    bed_df = bed_df.drop(columns=['genotype'], errors='ignore')

    # Drop the cn_state column from the data.
    bed_df = bed_df.drop(columns=['cn_state'], errors='ignore')

    # Add distance to nearest other SV call, clustered false positives often appear near real SVs.
    # Vectorized by chromosome to avoid row-wise apply.
    logging.info('Computing distance to nearest other SV call (same chromosome)...')
    logging.info('Applying distance calculation to all rows...')
    bed_df['dist_to_nearest_sv'] = np.nan

    for chrom, idx in bed_df.groupby('chrom', sort=False).groups.items():
        chrom_df = bed_df.loc[idx, ['start', 'end']].sort_values(['start', 'end'])
        n = chrom_df.shape[0]

        if n <= 1:
            continue

        starts = chrom_df['start'].to_numpy(dtype=np.int64)
        ends = chrom_df['end'].to_numpy(dtype=np.int64)

        # Previous interval summary.
        prev_max_end = np.maximum.accumulate(ends)
        prev_max_end_excl = np.empty(n, dtype=np.int64)
        prev_max_end_excl[0] = np.iinfo(np.int64).min
        prev_max_end_excl[1:] = prev_max_end[:-1]

        # Next interval summary.
        next_start_excl = np.empty(n, dtype=np.int64)
        next_start_excl[:-1] = starts[1:]
        next_start_excl[-1] = np.iinfo(np.int64).max

        # Overlap checks with prior/next intervals.
        overlap_prev = prev_max_end_excl > starts
        overlap_next = ends > next_start_excl
        overlap_any = overlap_prev | overlap_next

        # Gap to closest left/right neighbor (touching intervals yield 0).
        left_gap = starts - prev_max_end_excl
        right_gap = next_start_excl - ends

        # No-left/no-right sentinels.
        left_gap[0] = np.iinfo(np.int64).max
        right_gap[-1] = np.iinfo(np.int64).max

        nearest = np.minimum(left_gap, right_gap).astype(np.float64)
        nearest[overlap_any] = 0.0

        # Any remaining sentinel values are undefined (should only happen in degenerate cases).
        sentinel = float(np.iinfo(np.int64).max)
        nearest[nearest >= sentinel] = np.nan

        bed_df.loc[chrom_df.index, 'dist_to_nearest_sv'] = nearest

    logging.info('Distance to nearest SV calculated. Coverage: %.1f%%', (bed_df['dist_to_nearest_sv'].notna().sum() / len(bed_df) * 100))

    # Print statistics about the distance to nearest SV feature.
    logging.info('Distance to nearest SV - mean: %.2f, median: %.2f, std: %.2f', bed_df['dist_to_nearest_sv'].mean(), bed_df['dist_to_nearest_sv'].median(), bed_df['dist_to_nearest_sv'].std())

    # Now drop sv_length since all normalizations are complete
    # bed_df.drop(columns=['sv_length'], inplace=True)

    # Save the first 500 features to a new file.
    features_file = os.path.join(outdiranno, 'features.tsv')
    logging.info('Saving the features to %s', features_file)
    # Save only the first 500 rows to avoid saving too many records.
    bed_df.head(500).to_csv(features_file, sep='\t', index=False)
    logging.info('Saved the features to %s', features_file)

    # Return the features dataframe.
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
    df = pd.read_csv(bed_file, sep='\t', usecols=[0, 1, 2],
                     names=["CHROM", "POS", "END"],
                     dtype={'CHROM': str, 'POS': np.int32, 'END': np.int32})
    
    # Check if the BED file is empty.
    logging.info('Number of rows in the BED file: %d', df.shape[0])

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
    logging.info('Saved the ANNOVAR input file to %s', output_file)

    return output_file


def download_annovar_db(annovar_path, db_path, db_name, buildversion='hg38'):
    """Download the ANNOVAR database if it does not exist.
    
    Returns True if successful or database already exists, False if download failed.
    """
    logging.info('Downloading the database: %s for build version: %s', db_name, buildversion)
    
    # Check if database files already exist
    expected_files = [
        os.path.join(db_path, f"{buildversion}_{db_name}.txt"),
        os.path.join(db_path, f"{buildversion}_{db_name}.txt.idx"),
    ]
    
    if all(os.path.exists(f) for f in expected_files):
        logging.info('Database %s already exists, skipping download.', db_name)
        return True
    
    # Ensure the database directory exists
    os.makedirs(db_path, exist_ok=True)
    
    cmd = [
        f"{annovar_path}/annotate_variation.pl",
        "-buildver", buildversion,
        "-downdb", db_name,
        "."  # Download to current directory (we'll set cwd=db_path)
    ]

    # Run the command to download the database from the db_path directory
    # This ensures files are downloaded directly to the correct location
    logging.info('Running the command to download the database: %s (in directory: %s)', " ".join(cmd), db_path)
    try:
        result = subprocess.run(" ".join(cmd), shell=True, check=True, capture_output=True, text=True, cwd=db_path)
        if result.stdout:
            logging.debug('Download stdout: %s', result.stdout)
        logging.info('Downloaded the database %s successfully.', db_name)
        return True
    except subprocess.CalledProcessError as e:
        logging.warning('Failed to download the database %s: %s', db_name, e)
        if e.stderr:
            logging.warning('Error output: %s', e.stderr)
        logging.warning('Continuing without this database. Some features may be missing.')
        return False


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
    # Check if the cytoband annotation indicates telomere or centromere regions.
    try:
        # Centromeres contain 'acen' in their names.
        if 'acen' in cytoband:
            is_centromere = True
        # Telomeres are at the extreme bands - simplistic check for p/q terminal regions
        # (This is a simplified heuristic; a more robust method would use actual position data)
        elif 'p11' in cytoband or 'p12' in cytoband or 'p13' in cytoband:  # p-arm terminal
            is_telomere = True
        elif 'q13' in cytoband or 'q14' in cytoband:  # q-arm terminal (varies by chromosome)
            is_telomere = True

    except TypeError:
        pass
        # Handle the case where cytoband is not a string.

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

    # Check if record has any simple repeats (boolean indicator).
    data['simpleRepeat'] = data.merge(simpleRepeat_df, on=['chrom', 'start', 'end'], how='left')['chr_anno'].notna()

    logging.info('Number of records with simple repeats: %d', data['simpleRepeat'].sum())
    logging.info('Total number of records: %d', data.shape[0])

    # ---------------------------------------------------------------
    # Annotate the SVs using ANNOVAR.
    
    # Download the segmental duplication database
    segdup_success = download_annovar_db(annovar_path, db_path, "genomicSuperDups", buildversion)

    # Download the cytoband database
    cytoband_success = download_annovar_db(annovar_path, db_path, "cytoBand", buildversion)

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

    # Convert chr, start, end to the same data types as the data.
    anno_df['Chr'] = anno_df['Chr'].astype(str)
    anno_df['Start'] = anno_df['Start'].astype(np.int32)
    anno_df['End'] = anno_df['End'].astype(np.int32)

    # Merge the ANNOVAR annotations with the data.
    logging.info('Merging ANNOVAR annotations (%d records) with data (%d records)...', anno_df.shape[0], data.shape[0])
    data = data.merge(anno_df, left_on=['chrom', 'start', 'end'], right_on=['Chr', 'Start', 'End'], how='left')
    logging.info('ANNOVAR merge completed.')
    
    # Extract segmental duplication scores.
    logging.info('Extracting segmental duplication scores...')
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
    logging.info('Segmental duplication scores extracted. Mean: %.3f', data['segdup'].mean())

    # Extract the cytoband annotations.
    logging.info('Processing cytoband annotations for telomere/centromere detection...')
    def get_cyto_info(row):
        """Get telomere and centromere information for a row."""
        if pd.notna(row['cytoBand']):
            return get_cytoband_is_c_t(cytoband_dict, row['chrom'], row['cytoBand'])
        
        return (False, False)
    
    cyto_flags = data.apply(get_cyto_info, axis=1, result_type='expand')
    data[['telomere', 'centromere']] = cyto_flags
    logging.info('Telomere/centromere annotation complete. Telomeres: %d, Centromeres: %d', data['telomere'].sum(), data['centromere'].sum())

    # Add feature dist_to_telomere, dist_to_centromere using vectorized operations.
    logging.info('Computing distances to chromosome telomeres and centromeres...')
    chrom_bounds = pd.DataFrame([
        {
            'chrom': chrom,
            'chrom_norm': normalize_chrom_label(chrom),
            'chrom_start': values.get('chrom_start', np.nan),
            'chrom_end': values.get('chrom_end', np.nan),
            'centromere_start': values.get('centromere_start', np.nan),
            'centromere_end': values.get('centromere_end', np.nan)
        }
        for chrom, values in cytoband_dict.items()
    ])

    data_with_bounds = data.copy()
    data_with_bounds['chrom_norm'] = data_with_bounds['chrom'].apply(normalize_chrom_label)
    data_with_bounds = data_with_bounds.merge(
        chrom_bounds[['chrom_norm', 'chrom_start', 'chrom_end', 'centromere_start', 'centromere_end']],
        on='chrom_norm',
        how='left'
    )

    starts = data_with_bounds['start'].to_numpy(dtype=np.float64)
    ends = data_with_bounds['end'].to_numpy(dtype=np.float64)
    chrom_starts = data_with_bounds['chrom_start'].to_numpy(dtype=np.float64)
    chrom_ends = data_with_bounds['chrom_end'].to_numpy(dtype=np.float64)
    centromere_starts = data_with_bounds['centromere_start'].to_numpy(dtype=np.float64)
    centromere_ends = data_with_bounds['centromere_end'].to_numpy(dtype=np.float64)

    # Telomere distance: nearest interval-to-point distance to chromosome start/end.
    dist_left_tel = np.minimum(np.abs(starts - chrom_starts), np.abs(ends - chrom_starts))
    dist_right_tel = np.minimum(np.abs(starts - chrom_ends), np.abs(ends - chrom_ends))
    dist_to_telomere = np.minimum(dist_left_tel, dist_right_tel)
    tel_valid = (~np.isnan(chrom_starts)) & (~np.isnan(chrom_ends))
    dist_to_telomere[~tel_valid] = np.nan

    # Centromere distance: 0 if overlapping centromere span, else gap to nearest boundary.
    cen_valid = (~np.isnan(centromere_starts)) & (~np.isnan(centromere_ends))
    dist_to_centromere = np.full(len(data_with_bounds), np.nan, dtype=np.float64)
    left_of_centromere = ends < centromere_starts
    right_of_centromere = starts > centromere_ends
    overlap_centromere = (~left_of_centromere) & (~right_of_centromere)
    dist_to_centromere[cen_valid & left_of_centromere] = (centromere_starts - ends)[cen_valid & left_of_centromere]
    dist_to_centromere[cen_valid & right_of_centromere] = (starts - centromere_ends)[cen_valid & right_of_centromere]
    dist_to_centromere[cen_valid & overlap_centromere] = 0.0

    data['dist_to_telomere'] = dist_to_telomere
    data['dist_to_centromere'] = dist_to_centromere
    tel_zero_pct = (data['dist_to_telomere'] == 0).mean() * 100
    cen_zero_pct = (data['dist_to_centromere'] == 0).mean() * 100
    cen_le1_pct = (data['dist_to_centromere'] <= 1).mean() * 100
    cen_desc = data['dist_to_centromere'].describe(percentiles=[0.5, 0.9, 0.99])
    # Diagnostics for coordinate issues that can indicate malformed records.
    out_of_bounds_pct = ((data_with_bounds['start'] < data_with_bounds['chrom_start']) | (data_with_bounds['end'] > data_with_bounds['chrom_end'])).mean() * 100
    logging.info(
        'Telomere/centromere distances calculated. Mean dist_to_telomere: %.2f, Mean dist_to_centromere: %.2f, telomere zeros: %.2f%%, centromere zeros: %.2f%%, out-of-bounds coords: %.2f%%',
        data['dist_to_telomere'].mean(),
        data['dist_to_centromere'].mean(),
        tel_zero_pct,
        cen_zero_pct,
        out_of_bounds_pct
    )
    logging.info(
        'Centromere distance distribution: min=%.2f, p50=%.2f, p90=%.2f, p99=%.2f, max=%.2f, <=1bp: %.2f%%',
        cen_desc['min'],
        cen_desc['50%'],
        cen_desc['90%'],
        cen_desc['99%'],
        cen_desc['max'],
        cen_le1_pct
    )

    # Log-transform long-tailed distance features for model stability.
    logging.info('Applying log1p transform to dist_to_telomere and dist_to_centromere...')
    data['dist_to_telomere'] = np.log1p(data['dist_to_telomere'])
    data['dist_to_centromere'] = np.log1p(data['dist_to_centromere'])
    logging.info(
        'Distance log-transform complete. Mean log-dist_to_telomere: %.3f, Mean log-dist_to_centromere: %.3f',
        data['dist_to_telomere'].mean(),
        data['dist_to_centromere'].mean()
    )

    # Helper function to compute repeat density across entire SV span
    def compute_repeat_density_span(data_df, repeat_overlap_df):
        """Compute repeat span density as the fraction of the SV covered by simple repeats."""
        repeat_copy = repeat_overlap_df.copy()
        repeat_copy['overlap_length'] = repeat_copy['end_anno'] - repeat_copy['start_anno']
        
        # Group by original SV coordinates and sum total overlapping lengths
        density_df = repeat_copy.groupby(['chrom', 'start', 'end'])['overlap_length'].sum().reset_index()
        density_df.columns = ['chrom', 'start', 'end', 'total_repeat_length']
        
        # Merge with data and calculate density
        merged = data_df.merge(density_df, on=['chrom', 'start', 'end'], how='left')
        merged['total_repeat_length'] = merged['total_repeat_length'].fillna(0)
        span_length = (merged['end'] - merged['start']).astype(float)
        zero_span_count = (span_length <= 0).sum()
        if zero_span_count > 0:
            logging.info('Found %d SV records with non-positive span; setting repeat_span_density to 0 for these records.', zero_span_count)
        valid_span = span_length > 0
        density_values = pd.Series(0.0, index=merged.index)
        density_values.loc[valid_span] = merged.loc[valid_span, 'total_repeat_length'] / span_length.loc[valid_span]
        density_values = density_values.clip(lower=0, upper=1)
        
        return density_values
    
    # Add breakpoint features from both breakpoints (vectorized by chromosome).
    logging.info('Computing breakpoint features (segdup and simple repeat at left/right breakpoints)...')

    def point_max_overlap_score(points, starts, ends, scores):
        """For each query point, return max score among overlapping intervals."""
        if len(starts) == 0:
            return np.zeros(len(points), dtype=np.float64)

        order = np.argsort(starts, kind='mergesort')
        starts = starts[order]
        ends = ends[order]
        scores = scores[order]

        point_order = np.argsort(points, kind='mergesort')
        result = np.zeros(len(points), dtype=np.float64)

        active = []  # max-heap via negative score: (-score, interval_end)
        interval_idx = 0
        n_intervals = len(starts)

        for point_idx in point_order:
            point = points[point_idx]

            while interval_idx < n_intervals and starts[interval_idx] <= point:
                heapq.heappush(active, (-scores[interval_idx], ends[interval_idx]))
                interval_idx += 1

            while active and active[0][1] < point:
                heapq.heappop(active)

            if active:
                result[point_idx] = -active[0][0]

        return result

    def point_in_any_interval(points, starts, ends):
        """For each query point, return whether it is covered by any interval."""
        if len(starts) == 0:
            return np.zeros(len(points), dtype=bool)

        order = np.argsort(starts, kind='mergesort')
        starts_sorted = starts[order]
        ends_sorted = ends[order]
        max_end_prefix = np.maximum.accumulate(ends_sorted)

        idx = np.searchsorted(starts_sorted, points, side='right') - 1
        covered = np.zeros(len(points), dtype=bool)
        valid = idx >= 0
        covered[valid] = max_end_prefix[idx[valid]] >= points[valid]

        return covered

    # Precompute interval arrays by chromosome for fast lookup.
    anno_segdup = anno_df[['Chr', 'Start', 'End', 'genomicSuperDups']].copy()
    anno_segdup['segdup_score'] = anno_segdup['genomicSuperDups'].apply(extract_scores)
    segdup_intervals = {
        normalize_chrom_label(chrom): (
            grp['Start'].to_numpy(dtype=np.int64),
            grp['End'].to_numpy(dtype=np.int64),
            grp['segdup_score'].to_numpy(dtype=np.float64)
        )
        for chrom, grp in anno_segdup.groupby('Chr', sort=False)
    }

    repeat_intervals = {
        normalize_chrom_label(chrom): (
            grp['start_anno'].to_numpy(dtype=np.int64),
            grp['end_anno'].to_numpy(dtype=np.int64)
        )
        for chrom, grp in simpleRepeat_df.groupby('chrom', sort=False)
    }

    # Allocate result columns.
    data['segdup_left'] = 0.0
    data['segdup_right'] = 0.0
    data['simpleRepeat_left'] = False
    data['simpleRepeat_right'] = False

    logging.info('Computing left breakpoint features...')
    for chrom, chrom_idx in data.groupby('chrom', sort=False).groups.items():
        idx = list(chrom_idx)
        left_points = data.loc[idx, 'start'].to_numpy(dtype=np.int64)
        right_points = data.loc[idx, 'end'].to_numpy(dtype=np.int64)
        chrom_norm = normalize_chrom_label(chrom)

        seg_starts, seg_ends, seg_scores = segdup_intervals.get(
            chrom_norm, (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float64))
        )
        rep_starts, rep_ends = repeat_intervals.get(
            chrom_norm, (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        )

        data.loc[idx, 'segdup_left'] = point_max_overlap_score(left_points, seg_starts, seg_ends, seg_scores)
        data.loc[idx, 'simpleRepeat_left'] = point_in_any_interval(left_points, rep_starts, rep_ends)

        data.loc[idx, 'segdup_right'] = point_max_overlap_score(right_points, seg_starts, seg_ends, seg_scores)
        data.loc[idx, 'simpleRepeat_right'] = point_in_any_interval(right_points, rep_starts, rep_ends)

    logging.info('Breakpoint features complete. segdup_left mean: %.3f, segdup_right mean: %.3f', data['segdup_left'].mean(), data['segdup_right'].mean())
    
    # Calculate repeat span density feature using the simpleRepeat annotations. For each record, calculate the repeat span density as the total overlapping length of all simple repeats divided by the length of the record (end - start).
    logging.info('Computing repeat span density (total repeat coverage across SV)...')
    data['repeat_span_density'] = compute_repeat_density_span(data, simpleRepeat_df)  # across entire SV
    logging.info('Repeat span density calculated. Mean: %.3f, Max: %.3f', data['repeat_span_density'].mean(), data['repeat_span_density'].max())

    # Drop the unnecessary/redundant columns.
    data.drop(columns=['Chr', 'Start', 'End', 'cytoBand', 'genomicSuperDups', 'Ref', 'Alt', 'segdup', 'simpleRepeat'], inplace=True)

    logging.info('Number of records after adding annotations: %d', data.shape[0])

    return data
