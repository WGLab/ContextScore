# ContextScore
Assign confidence scores to SV datasets based on coverage, genomic context, and other important alignment features

[![unit tests](https://github.com/WGLab/ContextScore/actions/workflows/predict-test.yml/badge.svg)](https://github.com/WGLab/ContextScore/actions/workflows/predict-test.yml)

## User Workflow

ContextScore exposes a single user-facing command for prediction:

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

## Conda Package Setup

This repository includes a conda recipe at `conda-recipe/meta.yaml`.

Build steps:

```bash
conda env create -f environment.yml
conda activate contextscore
conda build conda-recipe
```

If resolving the environment is slow, try mamba:
```bash
conda install -n base -c conda-forge mamba


Install locally built package:

```bash
conda install --use-local contextscore
```
