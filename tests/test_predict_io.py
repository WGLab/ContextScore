from pathlib import Path

import numpy as np
import pandas as pd

from contextscore import predict


FIXTURE_VCF_GZ = Path(__file__).parent / 'fixtures' / 'output.vcf.gz'
TEST_OUTPUT_DIR = Path(__file__).parent / 'output'
FILTERED_VCF = TEST_OUTPUT_DIR / 'output_filtered.vcf'
REMOVED_VCF = TEST_OUTPUT_DIR / 'removed_svs.vcf'
PREDICTIONS_TSV = TEST_OUTPUT_DIR / 'predictions.tsv'
FILTERED_IDS = TEST_OUTPUT_DIR / 'filtered_ids.txt'


class DummyModel:
    def predict_proba(self, feature_df):
        length_signal = feature_df['length_signal'].to_numpy(dtype=float)
        probabilities = np.where(length_signal >= 200, 0.95, 0.05)
        return np.column_stack([1.0 - probabilities, probabilities])


def _fake_extract_features(bed_file, annovar_path, annovar_db_path, anno_outdir, buildver, sample_coverage=None):
    bed_df = pd.read_csv(
        bed_file,
        sep='\t',
        header=None,
        names=['chrom', 'start', 'end', 'sv_type_str', 'sv_length', 'gt', 'dp', 'hmm', 'aln', 'cluster', 'cn', 'alnoffset', 'id'],
    )
    sv_length = pd.to_numeric(bed_df['sv_length'], errors='coerce').fillna(0)
    read_depth = pd.to_numeric(bed_df['dp'], errors='coerce').fillna(0)

    return pd.DataFrame({
        'id': bed_df['id'].astype(int),
        'chrom': bed_df['chrom'].astype(str),
        'start': pd.to_numeric(bed_df['start'], errors='coerce').fillna(0).astype(int),
        'end': pd.to_numeric(bed_df['end'], errors='coerce').fillna(0).astype(int),
        'sv_type_str': bed_df['sv_type_str'].astype(str),
        'sv_length': sv_length.astype(int),
        'length_signal': sv_length.abs().astype(float),
        'depth_signal': read_depth.astype(float),
    })


def _count_vcf_records(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return sum(1 for line in handle if line.strip() and not line.startswith('#'))


def _prepare_output_dir():
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)
    for path in [FILTERED_VCF, REMOVED_VCF, PREDICTIONS_TSV, FILTERED_IDS, FIXTURE_VCF_GZ.with_suffix('.bed')]:
        if path.exists():
            path.unlink()


def test_open_vcf_text_gz():
    with predict.open_vcf_text(FIXTURE_VCF_GZ) as handle:
        lines = [line for line in handle]
    assert len(lines) > 0
    assert any(line.startswith('#') for line in lines)


def test_open_vcf_text_plain(tmp_path):
    vcf_path = tmp_path / 'test.vcf'
    vcf_path.write_text(
        '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n'
        'chr1\t100\t.\tA\tT\t.\tPASS\tSVTYPE=DEL;END=200\n',
        encoding='utf-8',
    )
    with predict.open_vcf_text(str(vcf_path)) as handle:
        lines = [line for line in handle]
    assert lines[0].startswith('#')
    assert 'SVTYPE=DEL' in lines[1]


def test_vcf_gz_header_and_records():
    with predict.open_vcf_text(FIXTURE_VCF_GZ) as handle:
        lines = [line.rstrip() for line in handle]
    header_lines = [line for line in lines if line.startswith('#')]
    variant_lines = [line for line in lines if line and not line.startswith('#')]

    assert any(line.startswith('##fileformat=VCFv4.2') for line in header_lines)
    assert any(line.startswith('#CHROM') for line in header_lines)
    assert len(variant_lines) > 0
    assert 'SVTYPE=' in variant_lines[0].split('\t')[7]
    assert 'END=' in variant_lines[0].split('\t')[7]


def test_vcf_gz_svtype_counts():
    svtypes = []
    with predict.open_vcf_text(FIXTURE_VCF_GZ) as handle:
        for line in handle:
            if line.startswith('#'):
                continue
            info = line.rstrip().split('\t')[7]
            for entry in info.split(';'):
                if entry.startswith('SVTYPE='):
                    svtypes.append(entry.split('=')[1])
                    break

    assert len(svtypes) > 0
    assert 'INS' in svtypes
    assert 'DEL' in svtypes


def test_score_generates_outputs_in_tests_output(monkeypatch):
    _prepare_output_dir()
    input_bed_path = FIXTURE_VCF_GZ.with_suffix('.bed')
    monkeypatch.setattr(predict, 'extract_features', _fake_extract_features)
    monkeypatch.setattr(predict.joblib, 'load', lambda model_path: DummyModel())

    summary = predict.score(
        model='tests/fixtures/dummy_model.pkl',
        input_vcf=str(FIXTURE_VCF_GZ),
        output_vcf=str(FILTERED_VCF),
        sample_coverage=30,
        annovar_path='unused',
        annovar_db_path='unused',
    )

    assert summary['output_vcf'] == str(FILTERED_VCF)
    assert summary['removed_vcf'] == str(REMOVED_VCF)
    assert summary['predictions_tsv'] == str(PREDICTIONS_TSV)
    assert FILTERED_VCF.exists()
    assert REMOVED_VCF.exists()
    assert PREDICTIONS_TSV.exists()
    assert FILTERED_IDS.exists()
    assert not input_bed_path.exists()

    predictions_df = pd.read_csv(PREDICTIONS_TSV, sep='\t')
    kept_records = _count_vcf_records(FILTERED_VCF)
    removed_records = _count_vcf_records(REMOVED_VCF)

    assert not predictions_df.empty
    assert set(['id', 'chrom', 'start', 'end', 'sv_type_str', 'sv_length', 'sv_length_abs', 'confidence_score']).issubset(predictions_df.columns)
    assert predictions_df['confidence_score'].between(0, 1).all()
    assert predictions_df['confidence_score'].max() == 0.95
    assert predictions_df['confidence_score'].min() == 0.05
    assert kept_records > 0
    assert removed_records > 0
    assert summary['total_records'] == len(predictions_df)
    assert summary['passed_records'] == kept_records
    assert summary['filtered_records'] == removed_records
    assert summary['passed_records'] + summary['filtered_records'] == summary['total_records']


def test_generated_predictions_include_multiple_svtypes():
    assert PREDICTIONS_TSV.exists()
    predictions_df = pd.read_csv(PREDICTIONS_TSV, sep='\t')

    assert predictions_df['sv_type_str'].nunique() >= 2
    assert {'DEL', 'INS'}.issubset(set(predictions_df['sv_type_str'].unique()))
