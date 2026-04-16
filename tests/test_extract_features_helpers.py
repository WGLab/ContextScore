import numpy as np
import pandas as pd

from contextscore import extract_features as extract_features_module
from contextscore.extract_features import bed_to_annovar_input, normalize_chrom_label


def test_normalize_chrom_label_handles_none_like_values():
    assert normalize_chrom_label(None) is None
    assert normalize_chrom_label(np.nan) is None
    assert normalize_chrom_label(" ") is None


def test_normalize_chrom_label_normalizes_prefix_and_case():
    assert normalize_chrom_label("chr1") == "1"
    assert normalize_chrom_label("1") == "1"
    assert normalize_chrom_label("chrx") == "X"
    assert normalize_chrom_label("x") == "X"


def test_bed_to_annovar_input_preserves_first_record_from_headerless_bed(tmp_path):
    bed_path = tmp_path / "input.bed"
    bed_path.write_text(
        "chr1\t100\t150\tINS\t50\t./.\t10\t0\tCIGARINS\t3\t0\t0\t0\n"
        "chr2\t200\t260\tDEL\t-60\t./.\t11\t0\tCIGARDEL\t4\t0\t0\t1\n",
        encoding="utf-8",
    )

    output_path = bed_to_annovar_input(str(bed_path))
    annovar_df = pd.read_csv(
        output_path,
        sep='\t',
        header=None,
        names=['chrom', 'start', 'end', 'ref', 'alt'],
    )

    assert annovar_df[['chrom', 'start', 'end']].values.tolist() == [
        ['chr1', 100, 150],
        ['chr2', 200, 260],
    ]


def test_extract_features_preserves_id_zero_from_headerless_prediction_bed(tmp_path, monkeypatch):
    bed_path = tmp_path / "input.bed"
    bed_path.write_text(
        "chr1\t100\t150\tINS\t50\t./.\t10\t0\tCIGARINS\t3\t0\t0\t0\n"
        "chr1\t300\t370\tDEL\t-70\t./.\t11\t0\tCIGARDEL\t4\t0\t0\t1\n",
        encoding="utf-8",
    )

    def fake_add_annotations(data, input_bed, annovar_path, db_path, anno_outdir, buildversion='hg38', training_format=False):
        annotated = data.copy()
        annotated['telomere'] = False
        annotated['centromere'] = False
        return annotated

    monkeypatch.setattr(extract_features_module, 'add_annotations', fake_add_annotations)

    feature_df = extract_features_module.extract_features(
        str(bed_path),
        annovar_path='unused',
        db_path='unused',
        outdiranno=str(tmp_path),
        sample_coverage=30,
    )

    assert feature_df['id'].tolist() == [0, 1]
