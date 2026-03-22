import pytest
from pydantic import ValidationError

from app.config import Settings


@pytest.fixture(autouse=True)
def clear_provider_env(monkeypatch):
    for key in (
        "YIELD_PROVIDER",
        "EXPLANATION_PROVIDER",
        "RISK_PROVIDER",
        "EXTRACTION_PROVIDER",
        "AI_YIELD_PROVIDER",
        "AI_EXPLANATION_PROVIDER",
        "AI_RISK_PROVIDER",
        "AI_EXTRACTION_PROVIDER",
        "INGESTION_LOG_FORMAT",
        "INGESTION_ENABLED_SOURCES",
        "INGESTION_DISABLED_SOURCES",
    ):
        monkeypatch.delenv(key, raising=False)


def test_settings_defaults_preserve_current_provider_behavior():
    config = Settings(_env_file=None)

    assert config.AI_YIELD_PROVIDER == "xgboost"
    assert config.AI_EXPLANATION_PROVIDER == "rule_based"
    assert config.AI_RISK_PROVIDER == "rule_based"
    assert config.AI_EXTRACTION_PROVIDER == "rule_based"


def test_short_provider_env_var_overrides_legacy_env_var(monkeypatch):
    monkeypatch.setenv("YIELD_PROVIDER", "stub")
    monkeypatch.setenv("AI_YIELD_PROVIDER", "rule_based")

    config = Settings(_env_file=None)

    assert config.AI_YIELD_PROVIDER == "stub"


def test_settings_normalize_supported_provider_aliases():
    config = Settings(
        _env_file=None,
        YIELD_PROVIDER="ml",
        EXPLANATION_PROVIDER="deterministic",
        EXTRACTION_PROVIDER="manual",
    )

    assert config.AI_YIELD_PROVIDER == "xgboost"
    assert config.AI_EXPLANATION_PROVIDER == "deterministic"
    assert config.AI_EXTRACTION_PROVIDER == "rule_based"


def test_settings_accept_stub_provider_ids():
    config = Settings(
        _env_file=None,
        YIELD_PROVIDER="stub",
        EXPLANATION_PROVIDER="deterministic",
        RISK_PROVIDER="stub",
        EXTRACTION_PROVIDER="stub",
    )

    assert config.AI_YIELD_PROVIDER == "stub"
    assert config.AI_EXPLANATION_PROVIDER == "deterministic"
    assert config.AI_RISK_PROVIDER == "stub"
    assert config.AI_EXTRACTION_PROVIDER == "stub"


def test_settings_accept_deterministic_yield_provider():
    config = Settings(
        _env_file=None,
        YIELD_PROVIDER="deterministic",
    )

    assert config.AI_YIELD_PROVIDER == "deterministic"


@pytest.mark.parametrize(
    ("setting_name", "setting_value", "expected_message"),
    [
        (
            "YIELD_PROVIDER",
            "rule_based",
            "YIELD_PROVIDER='rule_based' is not supported. Allowed values: deterministic, stub, ml, xgboost",
        ),
        (
            "EXPLANATION_PROVIDER",
            "llm",
            "EXPLANATION_PROVIDER='llm' is not supported. Allowed values: deterministic, rule_based",
        ),
        ("RISK_PROVIDER", "ml", "RISK_PROVIDER='ml' is not supported. Allowed values: rule_based, stub"),
        (
            "EXTRACTION_PROVIDER",
            "llm",
            "EXTRACTION_PROVIDER='llm' is not supported. Allowed values: manual, rule_based, stub",
        ),
    ],
)
def test_invalid_provider_selection_fails_clearly(setting_name, setting_value, expected_message):
    with pytest.raises(ValidationError, match=expected_message):
        Settings(_env_file=None, **{setting_name: setting_value})


def test_faostat_defaults_are_present():
    config = Settings(_env_file=None)

    assert config.FAOSTAT_SOURCE_NAME == "FAOSTAT Crops and Livestock"
    assert config.FAOSTAT_API_BASE_URL.endswith("/api/v1/en/data/QCL")
    assert "Production_Crops_Livestock" in config.FAOSTAT_BULK_DOWNLOAD_URL
    assert config.FAOSTAT_DEFAULT_LOOKBACK_YEARS == 1
    assert config.FAOSTAT_BATCH_SIZE == 500


def test_climate_ranking_defaults_are_present():
    config = Settings(_env_file=None)

    assert config.CLIMATE_LOOKBACK_DAYS == 30
    assert config.HEAT_DAY_THRESHOLD == 30.0
    assert config.AGRONOMIC_SCORE_WEIGHT == 0.65
    assert config.CLIMATE_SCORE_WEIGHT == 0.35


def test_yield_training_defaults_are_present():
    config = Settings(_env_file=None)

    assert config.YIELD_MODEL_PATH.endswith("yield_model.json")
    assert config.YIELD_MODEL_VERSION == "yield-xgb-v1"
    assert config.YIELD_TRAINING_SAMPLE_COUNT == 600
    assert config.YIELD_TRAINING_RANDOM_SEED == 20260321
    assert config.YIELD_MIN_REAL_TRAINING_SAMPLES == 25


def test_ingestion_source_filters_support_allow_and_deny_lists():
    allowlisted_config = Settings(
        _env_file=None,
        INGESTION_ENABLED_SOURCES="NASA POWER Daily, FAOSTAT Crops and Livestock",
    )
    denylisted_config = Settings(
        _env_file=None,
        INGESTION_DISABLED_SOURCES="FAOSTAT Crops and Livestock",
    )

    assert allowlisted_config.get_ingestion_enabled_sources() == {
        "nasa power daily",
        "faostat crops and livestock",
    }
    assert allowlisted_config.is_ingestion_source_enabled("NASA POWER Daily") is True
    assert allowlisted_config.is_ingestion_source_enabled("Unknown Source") is False
    assert denylisted_config.get_ingestion_disabled_sources() == {"faostat crops and livestock"}
    assert denylisted_config.is_ingestion_source_enabled("FAOSTAT Crops and Livestock") is False
    assert denylisted_config.is_ingestion_source_enabled("NASA POWER Daily") is True


@pytest.mark.parametrize(
    ("setting_name", "setting_value", "expected_message"),
    [
        ("FAOSTAT_DEFAULT_LOOKBACK_YEARS", 0, "FAOSTAT_DEFAULT_LOOKBACK_YEARS must be greater than 0"),
        ("FAOSTAT_BATCH_SIZE", 0, "FAOSTAT_BATCH_SIZE must be greater than 0"),
        (
            "INGESTION_LOG_FORMAT",
            "yaml",
            "INGESTION_LOG_FORMAT must be either 'json' or 'text'",
        ),
        ("CLIMATE_LOOKBACK_DAYS", 0, "CLIMATE_LOOKBACK_DAYS must be greater than 0"),
        ("HEAT_DAY_THRESHOLD", 8, "HEAT_DAY_THRESHOLD must be between 10 and 70"),
        ("YIELD_MODEL_VERSION", "   ", "YIELD_MODEL_VERSION must not be blank"),
        ("YIELD_TRAINING_SAMPLE_COUNT", 0, "YIELD_TRAINING_SAMPLE_COUNT must be greater than 0"),
        ("YIELD_MIN_REAL_TRAINING_SAMPLES", 0, "YIELD_MIN_REAL_TRAINING_SAMPLES must be greater than 0"),
    ],
)
def test_invalid_faostat_settings_fail_clearly(setting_name, setting_value, expected_message):
    with pytest.raises(ValidationError, match=expected_message):
        Settings(_env_file=None, **{setting_name: setting_value})


def test_overlapping_ingestion_source_filters_fail_clearly():
    with pytest.raises(
        ValidationError,
        match="INGESTION_ENABLED_SOURCES and INGESTION_DISABLED_SOURCES cannot overlap: nasa power daily",
    ):
        Settings(
            _env_file=None,
            INGESTION_ENABLED_SOURCES="NASA POWER Daily",
            INGESTION_DISABLED_SOURCES="nasa power daily",
        )


def test_ranking_weights_must_sum_to_more_than_zero():
    with pytest.raises(
        ValidationError,
        match="CLIMATE_SCORE_WEIGHT and AGRONOMIC_SCORE_WEIGHT must sum to more than 0",
    ):
        Settings(
            _env_file=None,
            CLIMATE_SCORE_WEIGHT=0.0,
            AGRONOMIC_SCORE_WEIGHT=0.0,
        )
