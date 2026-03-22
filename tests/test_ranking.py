from types import SimpleNamespace

from app.engines.ranking_engine import rank_fields_for_crop
from app.schemas.weather_history import ClimateSummary
from app.services.economic_service import EconomicAssessment


def make_field(field_id: int, name: str, **kwargs):
    values = {
        "id": field_id,
        "name": name,
        "irrigation_available": True,
        "drainage_quality": "good",
        "slope_percent": 3.0,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def make_crop(**kwargs):
    values = {
        "id": 1,
        "crop_name": "Corn",
        "ideal_ph_min": 6.0,
        "ideal_ph_max": 6.8,
        "tolerable_ph_min": 5.8,
        "tolerable_ph_max": 7.0,
        "water_requirement_level": "medium",
        "drainage_requirement": "moderate",
        "salinity_tolerance": "moderate",
        "rooting_depth_cm": 100.0,
        "slope_tolerance": 5.0,
        "organic_matter_preference": "moderate",
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def make_soil(**kwargs):
    values = {
        "id": 1,
        "ph": 6.5,
        "organic_matter_percent": 4.0,
        "depth_cm": 120.0,
        "ec": 1.5,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def make_climate_summary(**kwargs):
    values = {
        "avg_temp": 24.0,
        "total_rainfall": 650.0,
        "frost_days": 1,
        "heat_days": 10,
    }
    values.update(kwargs)
    return ClimateSummary(**values)


def make_economic_assessment(**kwargs):
    values = {
        "estimated_revenue": 20000.0,
        "estimated_cost": 9000.0,
        "estimated_profit": 11000.0,
        "reasons": ["High profit due to high yield and low cost."],
        "strengths": ["High profit due to high yield and low cost."],
        "weaknesses": [],
    }
    values.update(kwargs)
    return EconomicAssessment(**values)


def test_rank_fields_for_crop_sorts_descending_by_total_score():
    field_a = make_field(1, "Field Alpha", irrigation_available=True, drainage_quality="excellent", slope_percent=2.0)
    field_b = make_field(2, "Field Beta", irrigation_available=False, drainage_quality="poor", slope_percent=9.0)
    crop = make_crop(water_requirement_level="high")
    soil_map = {
        1: make_soil(ph=6.4, organic_matter_percent=4.0, depth_cm=140.0, ec=1.2),
        2: make_soil(ph=5.2, organic_matter_percent=1.0, depth_cm=60.0, ec=3.0),
    }

    ranked = rank_fields_for_crop([field_a, field_b], crop, soil_map)

    assert [entry.field_id for entry in ranked.ranked_fields] == [1, 2]
    assert ranked.ranked_fields[0].rank == 1
    assert ranked.ranked_fields[0].total_score > ranked.ranked_fields[1].total_score


def test_rank_fields_for_crop_preserves_input_order_for_ties():
    fields = [
        make_field(1, "First Field"),
        make_field(2, "Second Field"),
    ]
    crop = make_crop()
    soil_map = {
        1: make_soil(),
        2: make_soil(id=2),
    }

    ranked = rank_fields_for_crop(fields, crop, soil_map)

    assert [entry.field_id for entry in ranked.ranked_fields] == [1, 2]


def test_rank_fields_for_crop_supports_top_n():
    fields = [
        make_field(1, "Field One"),
        make_field(2, "Field Two", irrigation_available=False),
        make_field(3, "Field Three", slope_percent=20.0),
    ]
    crop = make_crop(water_requirement_level="high")
    soil_map = {
        1: make_soil(),
        2: make_soil(id=2),
        3: make_soil(id=3),
    }

    ranked = rank_fields_for_crop(fields, crop, soil_map, top_n=2)

    assert len(ranked.ranked_fields) == 2
    assert [entry.rank for entry in ranked.ranked_fields] == [1, 2]


def test_rank_fields_for_crop_handles_missing_soil_data_gracefully():
    fields = [
        make_field(1, "With Soil"),
        make_field(2, "Without Soil"),
    ]
    crop = make_crop()
    soil_map = {
        1: make_soil(),
    }

    ranked = rank_fields_for_crop(fields, crop, soil_map)

    missing_soil_entry = ranked.ranked_fields[1]
    assert missing_soil_entry.field_id == 2
    assert missing_soil_entry.total_score == 0.0
    assert any(blocker.code == "missing_soil_test" for blocker in missing_soil_entry.blockers)
    assert "No soil test available for suitability scoring." in missing_soil_entry.reasons


def test_rank_fields_for_crop_accepts_lookup_callable():
    fields = [
        make_field(1, "Callable A"),
        make_field(2, "Callable B", irrigation_available=False),
    ]
    crop = make_crop(water_requirement_level="high")
    lookup_calls: list[int] = []

    def soil_lookup(field_obj):
        lookup_calls.append(field_obj.id)
        if field_obj.id == 1:
            return make_soil(id=1)
        return None

    ranked = rank_fields_for_crop(fields, crop, soil_lookup)

    assert lookup_calls == [1, 2]
    assert len(ranked.ranked_fields) == 2
    assert ranked.ranked_fields[0].field_name == "Callable A"


def test_rank_fields_for_crop_uses_climate_summaries_when_other_inputs_match():
    field_a = make_field(1, "Climate Alpha")
    field_b = make_field(2, "Climate Beta")
    crop = make_crop(
        optimal_temp_min_c=18.0,
        optimal_temp_max_c=30.0,
        rainfall_requirement_mm=650.0,
        frost_tolerance_days=2,
        heat_tolerance_days=20,
    )
    soil_map = {
        1: make_soil(id=1),
        2: make_soil(id=2),
    }
    climate_map = {
        1: make_climate_summary(avg_temp=24.0, total_rainfall=650.0, frost_days=1, heat_days=10),
        2: make_climate_summary(avg_temp=34.0, total_rainfall=250.0, frost_days=6, heat_days=30),
    }

    ranked = rank_fields_for_crop(
        [field_a, field_b],
        crop,
        soil_map,
        climate_summaries=climate_map,
    )

    assert [entry.field_id for entry in ranked.ranked_fields] == [1, 2]
    assert "climate_compatibility" in ranked.ranked_fields[0].breakdown
    assert ranked.ranked_fields[0].total_score > ranked.ranked_fields[1].total_score


def test_rank_fields_for_crop_uses_economic_score_for_composite_ordering():
    field_a = make_field(1, "Profit Alpha")
    field_b = make_field(2, "Profit Beta")
    crop = make_crop()
    soil_map = {
        1: make_soil(id=1, ph=6.4, depth_cm=130.0),
        2: make_soil(id=2, ph=6.5, depth_cm=125.0),
    }
    economic_map = {
        1: make_economic_assessment(estimated_profit=5000.0),
        2: make_economic_assessment(
            estimated_profit=15000.0,
            estimated_revenue=26000.0,
            estimated_cost=11000.0,
        ),
    }

    ranked = rank_fields_for_crop(
        [field_a, field_b],
        crop,
        soil_map,
        economic_assessments=economic_map,
    )

    assert [entry.field_id for entry in ranked.ranked_fields] == [2, 1]
    assert ranked.ranked_fields[0].economic_score > ranked.ranked_fields[1].economic_score
    assert ranked.ranked_fields[0].ranking_score >= ranked.ranked_fields[0].total_score * 0.7


def test_ranked_result_includes_breakdown_blockers_and_reasons():
    field_obj = make_field(1, "Detailed Field", irrigation_available=False)
    crop = make_crop(water_requirement_level="high")
    soil_map = {1: make_soil(ph=6.5)}

    ranked = rank_fields_for_crop([field_obj], crop, soil_map)
    entry = ranked.ranked_fields[0]

    assert entry.field_name == "Detailed Field"
    assert "ph_compatibility" in entry.breakdown
    assert isinstance(entry.reasons, list)
    assert entry.economic_score == 0.0
    assert entry.estimated_profit is None
    assert entry.ranking_score == entry.total_score
    assert any(blocker.code == "no_irrigation_high_water_crop" for blocker in entry.blockers)


def test_rank_fields_api_returns_wrapped_ranking_response(client, db):
    from app.models.crop_price import CropPrice
    from app.models.input_cost import InputCost
    from app.services.crop_service import get_crop

    field1 = client.post("/api/v1/fields/", json={
        "name": "Field Alpha",
        "location_name": "Zone A",
        "latitude": 39.0997,
        "longitude": -94.5786,
        "area_hectares": 15.0,
        "slope_percent": 2.0,
        "irrigation_available": True,
        "drainage_quality": "excellent",
    }).json()
    field2 = client.post("/api/v1/fields/", json={
        "name": "Field Beta",
        "location_name": "Zone B",
        "latitude": 36.1627,
        "longitude": -86.7816,
        "area_hectares": 8.0,
        "slope_percent": 8.0,
        "irrigation_available": False,
        "drainage_quality": "poor",
    }).json()

    client.post("/api/v1/soil-tests/", json={
        "field_id": field1["id"],
        "ph": 6.5,
        "organic_matter_percent": 4.0,
        "nitrogen_ppm": 50.0,
        "phosphorus_ppm": 35.0,
        "potassium_ppm": 220.0,
        "texture_class": "loamy",
        "depth_cm": 120.0,
        "ec": 1.2,
    })
    client.post("/api/v1/soil-tests/", json={
        "field_id": field2["id"],
        "ph": 5.0,
        "organic_matter_percent": 1.0,
        "nitrogen_ppm": 10.0,
        "phosphorus_ppm": 5.0,
        "potassium_ppm": 60.0,
        "texture_class": "sandy",
        "depth_cm": 50.0,
        "ec": 3.0,
    })

    crop = client.post("/api/v1/crops/", json={
        "crop_name": "Corn",
        "scientific_name": "Zea mays",
        "ideal_ph_min": 6.0,
        "ideal_ph_max": 6.8,
        "tolerable_ph_min": 5.8,
        "tolerable_ph_max": 7.0,
        "water_requirement_level": "high",
        "drainage_requirement": "moderate",
        "frost_sensitivity": "high",
        "heat_sensitivity": "medium",
        "salinity_tolerance": "moderate",
        "rooting_depth_cm": 150.0,
        "slope_tolerance": 5.0,
        "organic_matter_preference": "moderate",
    }).json()

    crop_model = get_crop(db, crop["id"])
    crop_model.crop_price = CropPrice(price_per_ton=210.0)
    crop_model.input_cost = InputCost(fertilizer_cost=240.0, water_cost=165.0, labor_cost=130.0)
    db.commit()

    response = client.post("/api/v1/rank-fields/", json={
        "crop_id": crop["id"],
        "top_n": 5,
    })

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "ranking.v2"
    assert payload["crop"] == {
        "id": crop["id"],
        "crop_name": "Corn",
        "scientific_name": "Zea mays",
    }
    assert payload["total_fields_evaluated"] == 2
    assert payload["ranked_results"][0]["field_id"] == field1["id"]
    assert set(payload["ranked_results"][0].keys()) == {
        "rank",
        "field_id",
        "field_name",
        "total_score",
        "agronomic_score",
        "climate_score",
        "economic_score",
        "risk_score",
        "confidence_score",
        "estimated_revenue",
        "estimated_cost",
        "estimated_profit",
        "predicted_yield",
        "predicted_yield_min",
        "predicted_yield_max",
        "predicted_yield_range",
        "ranking_score",
        "strengths",
        "weaknesses",
        "risks",
        "climate_reasons",
        "climate_warnings",
        "climate_strengths",
        "climate_weaknesses",
        "climate_risks",
        "breakdown",
        "blockers",
        "reasons",
        "metadata",
        "provider_metadata",
        "explanation",
    }
    assert set(payload["ranked_results"][0]["explanation"].keys()) == {
        "short_explanation",
        "detailed_explanation",
        "strengths",
        "weaknesses",
        "risks",
        "metadata",
    }
    assert set(payload["ranked_results"][0]["provider_metadata"].keys()) == {
        "agronomic_provider",
        "ranking_provider",
        "explanation_provider",
        "yield_provider",
        "risk_provider",
    }
    assert "No irrigation available for a high water-demand crop." in payload["ranked_results"][1]["explanation"]["risks"]
    assert payload["ranked_results"][0]["estimated_profit"] is not None
    assert payload["ranked_results"][0]["estimated_revenue"] is not None
    assert payload["ranked_results"][0]["estimated_cost"] is not None
    assert isinstance(payload["ranked_results"][0]["economic_score"], float)
    assert payload["ranked_results"][0]["ranking_score"] == payload["ranked_results"][0]["total_score"]
    assert isinstance(payload["ranked_results"][0]["climate_reasons"], list)
    assert isinstance(payload["ranked_results"][0]["climate_warnings"], list)
    assert isinstance(payload["ranked_results"][0]["strengths"], list)
    assert isinstance(payload["ranked_results"][0]["metadata"]["provider_name"], str)
    assert isinstance(payload["ranked_results"][0]["provider_metadata"]["explanation_provider"]["provider_name"], str)


def test_rank_fields_api_respects_field_filter_and_keeps_missing_soil_entries(client, db):
    from app.models.crop_price import CropPrice
    from app.models.input_cost import InputCost
    from app.services.crop_service import get_crop

    field1 = client.post("/api/v1/fields/", json={
        "name": "Field One",
        "location_name": "Zone A",
        "latitude": 39.0997,
        "longitude": -94.5786,
        "area_hectares": 15.0,
        "slope_percent": 2.0,
        "irrigation_available": True,
        "drainage_quality": "excellent",
    }).json()
    field2 = client.post("/api/v1/fields/", json={
        "name": "Field Two",
        "location_name": "Zone B",
        "latitude": 36.1627,
        "longitude": -86.7816,
        "area_hectares": 8.0,
        "slope_percent": 8.0,
        "irrigation_available": True,
        "drainage_quality": "good",
    }).json()
    field3 = client.post("/api/v1/fields/", json={
        "name": "Field Three",
        "location_name": "Zone C",
        "latitude": 34.0522,
        "longitude": -118.2437,
        "area_hectares": 5.0,
        "slope_percent": 4.0,
        "irrigation_available": True,
        "drainage_quality": "good",
    }).json()

    client.post("/api/v1/soil-tests/", json={
        "field_id": field1["id"],
        "ph": 6.5,
        "organic_matter_percent": 4.0,
        "nitrogen_ppm": 50.0,
        "phosphorus_ppm": 35.0,
        "potassium_ppm": 220.0,
        "texture_class": "loamy",
        "depth_cm": 120.0,
        "ec": 1.2,
    })

    crop = client.post("/api/v1/crops/", json={
        "crop_name": "Corn",
        "scientific_name": "Zea mays",
        "ideal_ph_min": 6.0,
        "ideal_ph_max": 6.8,
        "tolerable_ph_min": 5.8,
        "tolerable_ph_max": 7.0,
        "water_requirement_level": "medium",
        "drainage_requirement": "moderate",
        "frost_sensitivity": "high",
        "heat_sensitivity": "medium",
        "salinity_tolerance": "moderate",
        "rooting_depth_cm": 150.0,
        "slope_tolerance": 5.0,
        "organic_matter_preference": "moderate",
    }).json()

    crop_model = get_crop(db, crop["id"])
    crop_model.crop_price = CropPrice(price_per_ton=210.0)
    crop_model.input_cost = InputCost(fertilizer_cost=240.0, water_cost=165.0, labor_cost=130.0)
    db.commit()

    response = client.post("/api/v1/rank-fields/", json={
        "crop_id": crop["id"],
        "field_ids": [field1["id"], field2["id"]],
        "top_n": 5,
    })

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_fields_evaluated"] == 2
    assert [entry["field_id"] for entry in payload["ranked_results"]] == [field1["id"], field2["id"]]
    assert all(entry["field_id"] != field3["id"] for entry in payload["ranked_results"])
    assert any(
        blocker["code"] == "missing_soil_test"
        for blocker in payload["ranked_results"][1]["blockers"]
    )


def test_rank_fields_api_returns_404_for_missing_crop(client):
    response = client.post("/api/v1/rank-fields/", json={"crop_id": 9999})

    assert response.status_code == 404
    assert response.json()["detail"] == "Crop with id 9999 not found"


def test_rank_fields_api_returns_404_when_no_fields_match_filter(client):
    crop = client.post("/api/v1/crops/", json={
        "crop_name": "Corn",
        "scientific_name": "Zea mays",
        "ideal_ph_min": 6.0,
        "ideal_ph_max": 6.8,
        "tolerable_ph_min": 5.8,
        "tolerable_ph_max": 7.0,
        "water_requirement_level": "medium",
        "drainage_requirement": "moderate",
        "frost_sensitivity": "high",
        "heat_sensitivity": "medium",
        "salinity_tolerance": "moderate",
        "rooting_depth_cm": 150.0,
        "slope_tolerance": 5.0,
        "organic_matter_preference": "moderate",
    }).json()

    response = client.post("/api/v1/rank-fields/", json={
        "crop_id": crop["id"],
        "field_ids": [9999],
    })

    assert response.status_code == 404
    assert response.json()["detail"] == "No fields found for the provided field filter"
