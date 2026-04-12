import numpy as np

from contextscore.extract_features import normalize_chrom_label


def test_normalize_chrom_label_handles_none_like_values():
    assert normalize_chrom_label(None) is None
    assert normalize_chrom_label(np.nan) is None
    assert normalize_chrom_label(" ") is None


def test_normalize_chrom_label_normalizes_prefix_and_case():
    assert normalize_chrom_label("chr1") == "1"
    assert normalize_chrom_label("1") == "1"
    assert normalize_chrom_label("chrx") == "X"
    assert normalize_chrom_label("x") == "X"
