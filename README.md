# ContextScore
Assign confidence scores to SV datasets based on coverage, genomic context, and other important alignment features

[![unit tests](https://github.com/WGLab/ContextScore/actions/workflows/predict-test.yml/badge.svg)](https://github.com/WGLab/ContextScore/actions/workflows/predict-test.yml)

## Installation
```bash
conda install -c wglab -c bioconda -c conda-forge contextscore

# Or using mamba (faster than conda for large environments)
mamba install -c wglab contextscore
```

## Sources for annotation files used in the model (under data/ directory):
| File | Source | Description | Link |
| --- | --- | --- | --- |
| `hg{19,38}_cytoband.txt` | UCSC Genome Browser | Cytoband annotations for human genome builds hg19 and hg38 | [UCSC hg19](https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/cytoBand.txt.gz) / [UCSC hg38](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cytoBand.txt.gz) |
| `hg{19,38}_segmental_duplications.bed` | UCSC Genome Browser | Segmental duplication annotations for human genome builds hg19 and hg38 | [UCSC hg19](https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/segmentalDuplications.txt.gz) / [UCSC hg38](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/segmentalDuplications.txt.gz) |
| `phastcons100way_hg{19,38}.bed` | UCSC Genome Browser | PhastCons conservation scores for human genome builds hg19 and hg38 | [UCSC hg19](https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/phastCons100way.txt.gz) / [UCSC hg38](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/phastCons100way.txt.gz) |
| `simple_repeats_hg{19,38}.bed` | UCSC Genome Browser | Simple repeat annotations for human genome builds hg19 and hg38 | [UCSC hg19](https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/simpleRepeat.txt.gz) / [UCSC hg38](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/simpleRepeat.txt.gz) |
| `fragile_sites_hg38.bed (with hg19 liftover)` | [HumCFS](https://webs.iiitd.edu.in/raghava/humcfs/download.html) | Fragile site annotations for human genome build hg38 | [HumCFS](https://webs.iiitd.edu.in/raghava/humcfs/fragile_site_bed.zip) |

## User Workflow
```bash
contextscore --input input.vcf --output scored.vcf --model full_model.pkl --sample_coverage 30 \
	--annovar /path/to/annovar --annovar-db /path/to/humandb
```

ANNOVAR is required for prediction and must be installed separately.

You can provide ANNOVAR locations using flags:

- `--annovar`: directory containing `annotate_variation.pl` and `table_annovar.pl`
- `--annovar-db`: ANNOVAR database directory

Or using environment variables:

```bash
export ANNOVAR_PATH=/path/to/annovar
export ANNOVAR_DB_PATH=/path/to/humandb
contextscore --input input.vcf --output scored.vcf --model full_model.pkl --sample_coverage 30
```
