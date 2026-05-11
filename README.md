# ContextScore
Assign confidence scores to SV datasets based on coverage, genomic context, and other important alignment features

[![unit tests](https://github.com/WGLab/ContextScore/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/WGLab/ContextScore/actions/workflows/unit-tests.yml)

## Installation
```bash
conda install -c wglab -c bioconda -c conda-forge contextscore

# Or using mamba (faster dependency resolution):
mamba install -c wglab contextscore
```

## ANNOVAR setup
[ANNOVAR](https://annovar.openbioinformatics.org/en/latest/user-guide/download/) is required for prediction and must be installed separately.

These are the required ANNOVAR components for ContextScore:
- `--annovar`: directory containing `annotate_variation.pl` and `table_annovar.pl`
- `--annovar-db`: ANNOVAR database directory

## User Workflow
```bash
contextscore --input input.vcf --output scored.vcf --sample-coverage 30 --buildver {hg38,hg19} --threshold 0.2 \
	--annovar /path/to/annovar --annovar-db /path/to/humandb
```

## Sources for additional annotations (under `data/` directory):
| File | Source | Description | Link |
| --- | --- | --- | --- |
| `cytobands_hg{19,38}.txt` | UCSC Genome Browser | Cytoband annotations for human genome builds hg19 and hg38 | [UCSC hg19](https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/cytoBand.txt.gz) / [UCSC hg38](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cytoBand.txt.gz) |
| `hg{19,38}_segmental_duplications.bed` | UCSC Genome Browser | Segmental duplication annotations for human genome builds hg19 and hg38 | [UCSC hg19](https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/segmentalDuplications.txt.gz) / [UCSC hg38](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/segmentalDuplications.txt.gz) |
| `phastcons100way_hg{19,38}.bed` | UCSC Genome Browser | PhastCons conservation scores for human genome builds hg19 and hg38 | [UCSC hg19](https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/phastCons100way.txt.gz) / [UCSC hg38](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/phastCons100way.txt.gz) |
| `simple_repeats_hg{19,38}.bed` | UCSC Genome Browser | Simple repeat annotations for human genome builds hg19 and hg38 | [UCSC hg19](https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/simpleRepeat.txt.gz) / [UCSC hg38](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/simpleRepeat.txt.gz) |
| `fragile_sites_hg38.bed` / `fragile_sites_hg19_liftover.bed` | [HumCFS](https://webs.iiitd.edu.in/raghava/humcfs/download.html) | Fragile site annotations for human genome builds hg38 and hg19 (liftover) | [HumCFS](https://webs.iiitd.edu.in/raghava/humcfs/fragile_site_bed.zip) |

