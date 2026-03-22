from dataclasses import dataclass
from typing import ClassVar, Self

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings


@dataclass(frozen=True, slots=True)
class ProviderSettingSpec:
    """Describe how a provider setting is loaded, normalized, and validated."""

    short_field: str
    canonical_field: str
    allowed_values: tuple[str, ...]
    aliases: dict[str, str]


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/agrimind"
    DATABASE_ECHO: bool = False
    DATABASE_POOL_PRE_PING: bool = True
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_POOL_TIMEOUT_SECONDS: int = 30
    DATABASE_CONNECT_TIMEOUT_SECONDS: int = 10
    APP_NAME: str = "AgriMind"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    YIELD_PROVIDER: str | None = None
    EXPLANATION_PROVIDER: str | None = None
    RISK_PROVIDER: str | None = None
    EXTRACTION_PROVIDER: str | None = None
    AI_SUITABILITY_PROVIDER: str = "rule_based"
    AI_RISK_PROVIDER: str = "rule_based"
    AI_EXPLANATION_PROVIDER: str = "rule_based"
    AI_RANKING_AUGMENTATION_PROVIDER: str = "rule_based"
    AI_YIELD_PROVIDER: str = "xgboost"
    AI_ASSISTANT_PROVIDER: str = "openai"
    AI_EXTRACTION_PROVIDER: str = "rule_based"
    YIELD_MODEL_DIR: str = "artifacts/yield_model"
    YIELD_MODEL_PATH: str = "artifacts/yield_model/yield_model.json"
    YIELD_MODEL_VERSION: str = "yield-xgb-v1"
    YIELD_TRAINING_SAMPLE_COUNT: int = 600
    YIELD_TRAINING_RANDOM_SEED: int = 20260321
    YIELD_MIN_REAL_TRAINING_SAMPLES: int = 25
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4.1-mini"
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_TIMEOUT_SECONDS: float = 20.0
    NASA_POWER_SOURCE_NAME: str = "NASA POWER Daily"
    NASA_POWER_BASE_URL: str = "https://power.larc.nasa.gov/api/temporal/daily/point"
    NASA_POWER_COMMUNITY: str = "AG"
    NASA_POWER_TIMEOUT_SECONDS: float = 60.0
    NASA_POWER_MAX_RETRIES: int = 2
    NASA_POWER_RETRY_BACKOFF_SECONDS: float = 1.0
    NASA_POWER_DEFAULT_LOOKBACK_DAYS: int = 30
    NASA_POWER_MAX_WINDOW_SHIFTS: int = 18
    NASA_POWER_TIME_STANDARD: str = "UTC"
    CLIMATE_LOOKBACK_DAYS: int = 30
    HEAT_DAY_THRESHOLD: float = 30.0
    CLIMATE_SCORE_WEIGHT: float = 0.35
    AGRONOMIC_SCORE_WEIGHT: float = 0.65
    ECONOMIC_SCORE_WEIGHT: float = 0.3
    DEFAULT_CROP_PRICE_SOURCE: str = "static"
    ENABLE_ECONOMIC_SCORING: bool = True
    FAOSTAT_SOURCE_NAME: str = "FAOSTAT Crops and Livestock"
    FAOSTAT_API_BASE_URL: str = "https://faostatservices.fao.org/api/v1/en/data/QCL"
    FAOSTAT_API_TOKEN: str | None = None
    FAOSTAT_BULK_DOWNLOAD_URL: str = (
        "https://fenixservices.fao.org/faostat/static/bulkdownloads/"
        "Production_Crops_Livestock_E_All_Data_(Normalized).zip"
    )
    FAOSTAT_TIMEOUT_SECONDS: float = 120.0
    FAOSTAT_DEFAULT_LOOKBACK_YEARS: int = 1
    FAOSTAT_BATCH_SIZE: int = 500
    FAOSTAT_DEFAULT_COUNTRIES: str = ""
    FAOSTAT_DEFAULT_CROPS: str = ""
    INGESTION_LOG_FORMAT: str = "json"
    INGESTION_ENABLED_SOURCES: str = ""
    INGESTION_DISABLED_SOURCES: str = ""
    INGESTION_AUTO_CREATE_TABLES: bool = True
    INGESTION_STALE_RUN_MINUTES: int = 60

    _PROVIDER_SETTING_SPECS: ClassVar[dict[str, ProviderSettingSpec]] = {
        "yield": ProviderSettingSpec(
            short_field="YIELD_PROVIDER",
            canonical_field="AI_YIELD_PROVIDER",
            allowed_values=("deterministic", "stub", "ml", "xgboost"),
            aliases={
                "deterministic": "deterministic",
                "stub": "stub",
                "ml": "xgboost",
                "xgboost": "xgboost",
            },
        ),
        "explanation": ProviderSettingSpec(
            short_field="EXPLANATION_PROVIDER",
            canonical_field="AI_EXPLANATION_PROVIDER",
            allowed_values=("deterministic", "rule_based"),
            aliases={
                "deterministic": "deterministic",
                "rule_based": "rule_based",
            },
        ),
        "risk": ProviderSettingSpec(
            short_field="RISK_PROVIDER",
            canonical_field="AI_RISK_PROVIDER",
            allowed_values=("rule_based", "stub"),
            aliases={
                "rule_based": "rule_based",
                "stub": "stub",
            },
        ),
        "extraction": ProviderSettingSpec(
            short_field="EXTRACTION_PROVIDER",
            canonical_field="AI_EXTRACTION_PROVIDER",
            allowed_values=("manual", "rule_based", "stub"),
            aliases={
                "manual": "rule_based",
                "rule_based": "rule_based",
                "stub": "stub",
            },
        ),
    }

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "debug", "dev", "development"}:
                return True
            if normalized in {"0", "false", "no", "off", "release", "prod", "production", ""}:
                return False
        return value

    @field_validator("NASA_POWER_DEFAULT_LOOKBACK_DAYS")
    @classmethod
    def validate_nasa_power_lookback_days(cls, value: int) -> int:
        """Ensure the default NASA POWER date window is positive."""

        if value <= 0:
            raise ValueError("NASA_POWER_DEFAULT_LOOKBACK_DAYS must be greater than 0")
        return value

    @field_validator("NASA_POWER_MAX_RETRIES")
    @classmethod
    def validate_nasa_power_max_retries(cls, value: int) -> int:
        """Ensure NASA POWER retry counts are non-negative."""

        if value < 0:
            raise ValueError("NASA_POWER_MAX_RETRIES must be greater than or equal to 0")
        return value

    @field_validator("NASA_POWER_RETRY_BACKOFF_SECONDS")
    @classmethod
    def validate_nasa_power_retry_backoff_seconds(cls, value: float) -> float:
        """Ensure NASA POWER retry backoff is non-negative."""

        if value < 0:
            raise ValueError("NASA_POWER_RETRY_BACKOFF_SECONDS must be greater than or equal to 0")
        return value

    @field_validator("NASA_POWER_MAX_WINDOW_SHIFTS")
    @classmethod
    def validate_nasa_power_max_window_shifts(cls, value: int) -> int:
        """Ensure the NASA POWER availability search window count is non-negative."""

        if value < 0:
            raise ValueError("NASA_POWER_MAX_WINDOW_SHIFTS must be greater than or equal to 0")
        return value

    @field_validator("CLIMATE_LOOKBACK_DAYS")
    @classmethod
    def validate_climate_lookback_days(cls, value: int) -> int:
        """Ensure the climate lookback window is positive."""

        if value <= 0:
            raise ValueError("CLIMATE_LOOKBACK_DAYS must be greater than 0")
        return value

    @field_validator("HEAT_DAY_THRESHOLD")
    @classmethod
    def validate_heat_day_threshold(cls, value: float) -> float:
        """Ensure the heat-day threshold is within a sane agronomic range."""

        if value < 10 or value > 70:
            raise ValueError("HEAT_DAY_THRESHOLD must be between 10 and 70")
        return value

    @field_validator("CLIMATE_SCORE_WEIGHT", "AGRONOMIC_SCORE_WEIGHT", "ECONOMIC_SCORE_WEIGHT")
    @classmethod
    def validate_ranking_score_weights(cls, value: float, info) -> float:
        """Ensure climate and agronomic score weights are non-negative."""

        if value < 0:
            raise ValueError(f"{info.field_name} must be greater than or equal to 0")
        return value

    @field_validator("DEFAULT_CROP_PRICE_SOURCE")
    @classmethod
    def validate_default_crop_price_source(cls, value: str) -> str:
        """Normalize the configured crop price source identifier."""

        normalized = value.strip().lower()
        if normalized not in {"static", "manual"}:
            raise ValueError("DEFAULT_CROP_PRICE_SOURCE must be either 'static' or 'manual'")
        return normalized

    @field_validator("YIELD_MODEL_VERSION")
    @classmethod
    def validate_yield_model_version(cls, value: str) -> str:
        """Ensure the configured yield model version is not blank."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("YIELD_MODEL_VERSION must not be blank")
        return normalized

    @field_validator("YIELD_TRAINING_SAMPLE_COUNT", "YIELD_MIN_REAL_TRAINING_SAMPLES")
    @classmethod
    def validate_yield_training_counts(cls, value: int, info) -> int:
        """Ensure configured yield training sample counts are positive."""

        if value <= 0:
            raise ValueError(f"{info.field_name} must be greater than 0")
        return value

    @field_validator("FAOSTAT_DEFAULT_LOOKBACK_YEARS")
    @classmethod
    def validate_faostat_lookback_years(cls, value: int) -> int:
        """Ensure the default FAOSTAT year window is positive."""

        if value <= 0:
            raise ValueError("FAOSTAT_DEFAULT_LOOKBACK_YEARS must be greater than 0")
        return value

    @field_validator("FAOSTAT_BATCH_SIZE")
    @classmethod
    def validate_faostat_batch_size(cls, value: int) -> int:
        """Ensure the FAOSTAT raw payload batch size is positive."""

        if value <= 0:
            raise ValueError("FAOSTAT_BATCH_SIZE must be greater than 0")
        return value

    @field_validator("INGESTION_STALE_RUN_MINUTES")
    @classmethod
    def validate_ingestion_stale_run_minutes(cls, value: int) -> int:
        """Ensure the stale-run timeout is positive."""

        if value <= 0:
            raise ValueError("INGESTION_STALE_RUN_MINUTES must be greater than 0")
        return value

    @field_validator("INGESTION_LOG_FORMAT")
    @classmethod
    def validate_ingestion_log_format(cls, value: str) -> str:
        """Normalize and validate the ingestion CLI log output format."""

        normalized = value.strip().lower()
        if normalized not in {"json", "text"}:
            raise ValueError("INGESTION_LOG_FORMAT must be either 'json' or 'text'")
        return normalized

    @model_validator(mode="after")
    def normalize_provider_settings(self) -> Self:
        """Resolve provider aliases into canonical AI_* settings."""

        for spec in self._PROVIDER_SETTING_SPECS.values():
            resolved_value = self._resolve_provider_setting(spec)
            setattr(self, spec.canonical_field, resolved_value)
        return self

    @model_validator(mode="after")
    def validate_ingestion_source_filters(self) -> Self:
        """Reject overlapping enabled and disabled ingestion source lists."""

        overlapping_sources = self.get_ingestion_enabled_sources().intersection(
            self.get_ingestion_disabled_sources()
        )
        if overlapping_sources:
            duplicates = ", ".join(sorted(overlapping_sources))
            raise ValueError(
                "INGESTION_ENABLED_SOURCES and INGESTION_DISABLED_SOURCES "
                f"cannot overlap: {duplicates}"
            )
        return self

    @model_validator(mode="after")
    def validate_ranking_weight_sum(self) -> Self:
        """Ensure climate and agronomic weights produce a usable composite score."""

        total_weight = self.CLIMATE_SCORE_WEIGHT + self.AGRONOMIC_SCORE_WEIGHT
        if total_weight <= 0:
            raise ValueError(
                "CLIMATE_SCORE_WEIGHT and AGRONOMIC_SCORE_WEIGHT must sum to more than 0"
            )
        return self

    def _resolve_provider_setting(self, spec: ProviderSettingSpec) -> str:
        short_value = getattr(self, spec.short_field)
        legacy_value = getattr(self, spec.canonical_field)

        if short_value is not None:
            source_name = spec.short_field
            source_value = short_value
        else:
            source_name = spec.canonical_field
            source_value = legacy_value

        normalized_value = self._normalize_provider_value(source_name, source_value, spec)
        resolved_value = spec.aliases.get(normalized_value)
        if resolved_value is None:
            allowed_values = ", ".join(spec.allowed_values)
            raise ValueError(
                f"{source_name}={normalized_value!r} is not supported. Allowed values: {allowed_values}"
            )
        return resolved_value

    @staticmethod
    def _normalize_provider_value(
        source_name: str,
        source_value: str,
        spec: ProviderSettingSpec,
    ) -> str:
        if not isinstance(source_value, str):
            raise ValueError(f"{source_name} must be a string provider id")

        normalized_value = source_value.strip().lower()
        if normalized_value:
            return normalized_value

        allowed_values = ", ".join(spec.allowed_values)
        raise ValueError(f"{source_name} cannot be blank. Allowed values: {allowed_values}")

    @staticmethod
    def _normalize_source_name(source_name: str) -> str:
        """Normalize a source name for case-insensitive config matching."""

        return source_name.strip().casefold()

    def _parse_source_name_csv(self, raw_value: str) -> frozenset[str]:
        """Parse a comma-separated source-name setting into normalized names."""

        names = {
            self._normalize_source_name(part)
            for part in raw_value.split(",")
            if part.strip()
        }
        return frozenset(names)

    def get_ingestion_enabled_sources(self) -> frozenset[str]:
        """Return the normalized allowlist of enabled ingestion source names."""

        return self._parse_source_name_csv(self.INGESTION_ENABLED_SOURCES)

    def get_ingestion_disabled_sources(self) -> frozenset[str]:
        """Return the normalized denylist of disabled ingestion source names."""

        return self._parse_source_name_csv(self.INGESTION_DISABLED_SOURCES)

    def is_ingestion_source_enabled(self, source_name: str, *, default_enabled: bool = True) -> bool:
        """Return whether an ingestion source is enabled by config."""

        normalized_name = self._normalize_source_name(source_name)
        if normalized_name in self.get_ingestion_disabled_sources():
            return False

        enabled_sources = self.get_ingestion_enabled_sources()
        if enabled_sources:
            return normalized_name in enabled_sources

        return default_enabled

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
