import importlib
from pathlib import Path

import pytest

from contextscore.predict import (
    DEFAULT_MODEL_INSTALL_PATH,
    DEFAULT_MODEL_ENV_VAR,
    resolve_model_path,
    resolve_annovar_paths,
    try_import_plotting_libs,
    validate_annovar_paths,
)


def test_example_vcf_fixture_exists():
    fixture_path = Path(__file__).parent / "fixtures" / "example.vcf"
    assert fixture_path.exists()


def test_resolve_annovar_paths_prefers_cli_over_env(monkeypatch):
    monkeypatch.setenv("ANNOVAR_PATH", "/env/annovar")
    monkeypatch.setenv("ANNOVAR_DB_PATH", "/env/db")

    annovar_path, annovar_db = resolve_annovar_paths("/cli/annovar", "/cli/db")

    assert annovar_path == "/cli/annovar"
    assert annovar_db == "/cli/db"


def test_resolve_annovar_paths_uses_env_when_cli_missing(monkeypatch):
    monkeypatch.setenv("ANNOVAR_PATH", "/env/annovar")
    monkeypatch.setenv("ANNOVAR_DB_PATH", "/env/db")

    annovar_path, annovar_db = resolve_annovar_paths(None, None)

    assert annovar_path == "/env/annovar"
    assert annovar_db == "/env/db"


def test_resolve_model_path_prefers_cli_over_env(monkeypatch):
    monkeypatch.setenv(DEFAULT_MODEL_ENV_VAR, '/env/model.pkl')

    resolved, source = resolve_model_path('/cli/model.pkl')

    assert resolved == '/cli/model.pkl'
    assert source == 'cli'


def test_resolve_model_path_uses_env_when_cli_missing(monkeypatch):
    monkeypatch.setenv(DEFAULT_MODEL_ENV_VAR, '/env/model.pkl')

    resolved, source = resolve_model_path(None)

    assert resolved == '/env/model.pkl'
    assert source == 'env'


def test_resolve_model_path_uses_default_when_cli_and_env_missing(monkeypatch):
    monkeypatch.delenv(DEFAULT_MODEL_ENV_VAR, raising=False)

    resolved, source = resolve_model_path(None)

    assert resolved == DEFAULT_MODEL_INSTALL_PATH
    assert source == 'default'


def test_validate_annovar_paths_requires_path_and_db():
    with pytest.raises(ValueError, match="ANNOVAR path is required"):
        validate_annovar_paths(None, "/db")

    with pytest.raises(ValueError, match="ANNOVAR database path is required"):
        validate_annovar_paths("/annovar", None)


def test_validate_annovar_paths_accepts_valid_layout(tmp_path):
    annovar_dir = tmp_path / "annovar"
    db_dir = tmp_path / "humandb"
    annovar_dir.mkdir()
    db_dir.mkdir()

    (annovar_dir / "annotate_variation.pl").write_text("#!/usr/bin/env perl\n", encoding="utf-8")
    (annovar_dir / "table_annovar.pl").write_text("#!/usr/bin/env perl\n", encoding="utf-8")

    validate_annovar_paths(str(annovar_dir), str(db_dir))


def test_try_import_plotting_libs_graceful_when_missing(monkeypatch):
    real_import_module = importlib.import_module

    def fake_import_module(name):
        if name in {"matplotlib.pyplot", "seaborn"}:
            raise ImportError("not installed")
        return real_import_module(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    plt, sns = try_import_plotting_libs()

    assert plt is None
    assert sns is None
