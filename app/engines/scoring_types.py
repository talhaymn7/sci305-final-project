"""Structured internal types for rule-based suitability scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID


class ScoreStatus(str, Enum):
    """Supported component score states."""

    IDEAL = "ideal"
    ACCEPTABLE = "acceptable"
    LIMITED = "limited"
    UNCONSTRAINED = "unconstrained"
    MISSING = "missing"
    BLOCKED = "blocked"


@dataclass(slots=True)
class ScoreComponent:
    """Detailed score information for a single scoring dimension."""

    key: str
    label: str
    weight: float
    awarded_points: float
    max_points: float
    status: ScoreStatus
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ScorePenalty:
    """Informational penalty derived from score lost within a dimension."""

    dimension: str
    points_lost: float
    message: str


@dataclass(slots=True)
class ScoreBlocker:
    """Hard blocker that forces the final score to zero."""

    code: str
    dimension: str
    message: str


@dataclass(slots=True)
class SuitabilityResult:
    """Structured suitability result for a single field and crop."""

    field_id: int | str | UUID
    crop_id: int | str | UUID
    soil_test_id: int | str | UUID | None
    total_score: float
    score_breakdown: dict[str, ScoreComponent] = field(default_factory=dict)
    penalties: list[ScorePenalty] = field(default_factory=list)
    blockers: list[ScoreBlocker] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    agronomic_score_value: float | None = None
    climate_score_value: float | None = None
    confidence_score: float | None = None
    climate_reasons: list[str] = field(default_factory=list)
    climate_warnings: list[str] = field(default_factory=list)
    climate_penalties: list[dict[str, float | str]] = field(default_factory=list)
    climate_strengths: list[str] = field(default_factory=list)
    climate_weaknesses: list[str] = field(default_factory=list)
    climate_risks: list[str] = field(default_factory=list)

    @property
    def component_scores(self) -> dict[str, float]:
        """Compatibility view keyed to numeric awarded points per dimension."""

        return {
            key: component.awarded_points
            for key, component in self.score_breakdown.items()
        }

    @property
    def blocking_constraints(self) -> list[str]:
        """Compatibility view of blocker messages."""

        return [blocker.message for blocker in self.blockers]

    @property
    def climate_score(self) -> float | None:
        """Return the climate component normalized to a 0-100 score."""

        if self.climate_score_value is not None:
            return self.climate_score_value
        component = self.score_breakdown.get("climate_compatibility")
        if component is None or component.max_points <= 0:
            return None
        return round((component.awarded_points / component.max_points) * 100, 2)

    @property
    def agronomic_score(self) -> float:
        """Return the agronomic score before any external ranking augmentation."""

        if self.agronomic_score_value is not None:
            return self.agronomic_score_value
        return self.total_score
