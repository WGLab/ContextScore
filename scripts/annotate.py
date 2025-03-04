import os
import subprocess

def annotate_with_annovar(input_vcf, output_prefix, annovar_path, db_path):
    cmd = [
        f"{annovar_path}/table_annovar.pl",
        input_vcf,
        db_path,
        "--buildver hg19",
        "--out", output_prefix,
        "--remove",
        "--protocol refGene,cytoBand,dbnsfp35a",
        "--operation g,r,f",
        "--nastring ."
    ]
    subprocess.run(" ".join(cmd), shell=True, check=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", required=True, help="Input VCF file")
    parser.add_argument("--out", required=True, help="Output prefix")
    parser.add_argument("--annovar", required=True, help="Path to ANNOVAR")
    parser.add_argument("--db", required=True, help="Path to ANNOVAR database")
    args = parser.parse_args()
    
    annotate_with_annovar(args.vcf, args.out, args.annovar, args.db)
