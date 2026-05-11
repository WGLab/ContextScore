import pandas as pd
import pymysql
from pathlib import Path

def download_ucsc(table_name: str, 
                 genome_version: str = "hg38",
                 output_file: str = "ucsc_table.bed") -> None:
    """
    Downloads the UCSC Simple Repeats table and saves it as a BED file for use with BEDTools.
    Note: This function requires access to the UCSC MySQL database.
    """
    print("Downloading UCSC " + table_name + " table for " + genome_version + "...")
    
    # Connect to UCSC MySQL database
    conn = pymysql.connect(host="genome-mysql.soe.ucsc.edu",
                        user="genome",
                        password="",
                        database="hg38")  # Change to the desired genome version (e.g., hg19, mm10)

    query = f"""
    SELECT
        chrom AS chr, 
        chromStart AS start, 
        chromEnd AS end, 
        name
    FROM
        {table_name}
    WHERE
        chrom IS NOT NULL AND
        chromStart IS NOT NULL AND
        chromEnd IS NOT NULL
    AND
        chromStart >= 0 AND
        chromEnd > chromStart
    AND
        chromStart < chromEnd;
    """
    df = pd.read_sql(query, conn)

    # Close connection
    conn.close()

    # Save as BED file for BEDTools
    df.to_csv(output_file, sep="\t", index=False, header=False)
    print("Downloaded UCSC " + table_name + " table for " + genome_version + " and saved as " + output_file)

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download the UCSC Simple Repeats table for hg38
    simple_repeat_file = str(data_dir / "simple_repeats_hg38.bed")
    download_ucsc(table_name="simpleRepeat",
                 genome_version="hg38",
                 output_file=simple_repeat_file)
    
    # Download the UCSC phastCons100way table for hg38
    phastcons_file = str(data_dir / "phastcons100way_hg38.bed")
    download_ucsc(table_name="phastCons100way",
                 genome_version="hg38",
                 output_file=phastcons_file)
