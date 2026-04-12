# ContextScore
Assign confidence scores to SV datasets based on coverage, genomic context, and other important alignment features

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

### Logging Modes

- Default: concise user-focused progress messages
- `--verbose`: detailed processing logs
- `--debug`: debugging logs including subprocess details

## Developer Workflow

Model training is developer-only and is intentionally not exposed as a public installed command.
Use module/script invocation for training pipelines and private datasets.

## Conda Package Setup

This repository includes a Linux conda recipe at `conda-recipe/meta.yaml`.

Build steps:

```bash
conda env create -f environment.yml
conda activate contextscore-build
conda build conda-recipe
```

Install locally built package:

```bash
conda install --use-local contextscore
```

Notes:

- The recipe is Linux-only (`skip: true  # [win]`) because `bedtools` is required at runtime.
- Optional debug plotting dependencies are not part of base runtime; install with pip extra if needed:

```bash
pip install .[plot]
```
