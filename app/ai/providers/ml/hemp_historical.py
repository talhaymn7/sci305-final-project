# app/ai/providers/ml/hemp_historical.py
"""Historical correction provider for hemp prescription.

Adjusts the cold-start XGBoost prediction using completed crop cycle data.
The core mechanism is an exponential-weighted average of (actual / predicted)
yield ratios across all cycles for the field, with more recent cycles
receiving higher weight.

The N correction uses the most recent cycle's efficiency signal: if the
farmer applied less N than recommended but still achieved near-expected
yield, the recommendation is reduced accordingly.
"""
from __future__ import annotations

from app.schemas.hemp_cycle import CycleRecord
from app.schemas.hemp_prescription import HempPrescriptionRequest, HempPrescriptionResult

_DECAY = 0.6           # weight decay per cycle (older cycles count less)
_MIN_FACTOR = 0.50     # correction floor
_MAX_FACTOR = 2.00     # correction ceiling
_BASE_CONFIDENCE = 0.65
_CONFIDENCE_PER_CYCLE = 0.10
_MAX_CONFIDENCE = 0.95


class HistoricalHempProvider:
    """Corrects cold-start predictions using actual field cycle outcomes.

    Pass ``cycles`` sorted oldest-first. The provider weights recent
    cycles more heavily via exponential decay.
    """

    def predict(
        self,
        request: HempPrescriptionRequest,
        cycles: list[CycleRecord],
        cold_start: HempPrescriptionResult,
    ) -> HempPrescriptionResult:
        if not cycles:
            return cold_start

        yield_factor = self._yield_correction_factor(cycles)
        n_factor     = self._n_correction_factor(cycles)

        corrected_yield = round(
            max(0.0, cold_start.expected_yield_ton_dekar * yield_factor), 3
        )
        corrected_n = round(
            max(0.0, cold_start.rec_nitrogen_kg_dekar * n_factor), 2
        )
        confidence = min(_MAX_CONFIDENCE, _BASE_CONFIDENCE + len(cycles) * _CONFIDENCE_PER_CYCLE)

        notes = list(cold_start.notes)
        notes.append(
            f"Recommendation adjusted from {len(cycles)} completed cycle(s) on this field. "
            f"Yield correction factor: {yield_factor:.2f}x."
        )

        return HempPrescriptionResult(
            rec_nitrogen_kg_dekar=corrected_n,
            rec_phosphorus_kg_dekar=cold_start.rec_phosphorus_kg_dekar,
            rec_potassium_kg_dekar=cold_start.rec_potassium_kg_dekar,
            rec_irrigation_mm_week=cold_start.rec_irrigation_mm_week,
            expected_yield_ton_dekar=corrected_yield,
            suitable=True,
            confidence=confidence,
            blockers=[],
            notes=notes,
            provider="historical",
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _yield_correction_factor(self, cycles: list[CycleRecord]) -> float:
        """Exponential-weighted average of (actual / predicted) ratios.

        cycles[0] is oldest; cycles[-1] is most recent.
        """
        n = len(cycles)
        weights = [_DECAY ** (n - 1 - i) for i in range(n)]

        weighted_sum = 0.0
        usable_weight = 0.0
        for cycle, w in zip(cycles, weights):
            if cycle.predicted_yield_ton_dekar <= 0.0:
                continue
            ratio = cycle.actual_yield_ton_dekar / cycle.predicted_yield_ton_dekar
            weighted_sum  += ratio * w
            usable_weight += w

        if usable_weight == 0.0:
            return 1.0

        raw_factor = weighted_sum / usable_weight
        return max(_MIN_FACTOR, min(_MAX_FACTOR, raw_factor))

    def _n_correction_factor(self, cycles: list[CycleRecord]) -> float:
        """Adjust N recommendation based on the most recent cycle's efficiency signal."""
        recent = cycles[-1]
        rec     = recent.rec_nitrogen_kg_dekar
        applied = recent.applied_nitrogen_kg_dekar

        if rec <= 0.0 or applied <= 0.0:
            return 1.0

        application_ratio = applied / rec
        yield_ratio = (
            recent.actual_yield_ton_dekar / max(0.001, recent.predicted_yield_ton_dekar)
        )

        # Applied <88% of recommendation yet yield held up >= 93% -> reduce recommendation
        if application_ratio < 0.88 and yield_ratio >= 0.93:
            return max(0.70, application_ratio + 0.05)

        # Applied >115% of recommendation AND yield improved >10% -> slight increase
        if application_ratio > 1.15 and yield_ratio > 1.10:
            return min(1.35, application_ratio * 0.90)

        return 1.0
