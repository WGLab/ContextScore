import sys
import subprocess

def main(vcf_file, annovar_path, annovar_db_path):
    # Construct the command to run the contextscore module
    command = [annovar_path, 'contextscore', vcf_file, '-db', annovar_db_path]
    
    # Run the command
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print(f"Error running contextscore: {e.stderr.decode()}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python __main__.py <vcf_file> <annovar_path> <annovar_db_path>")
        sys.exit(1)
    
    vcf_file = sys.argv[1]
    annovar_path = sys.argv[2]
    annovar_db_path = sys.argv[3]
    
    main(vcf_file, annovar_path, annovar_db_path)
    