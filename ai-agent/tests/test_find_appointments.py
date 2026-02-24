"""Tests for the find_appointments tool."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from ai_agent.tools.find_appointments import (
    FindAppointmentsInput,
    _find_appointments_impl,
    _format_appointment,
)


# -- helpers -------------------------------------------------------------------

def _make_appointment(**overrides: Any) -> dict[str, Any]:
    base = {
        "pc_eid": 1,
        "fname": "John",
        "lname": "Doe",
        "pc_pid": 10,
        "pid": 10,
        "pce_aid_fname": "Dr",
        "pce_aid_lname": "Smith",
        "pc_eventDate": "2026-03-01",
        "pc_startTime": "09:00:00",
        "pc_endTime": "09:30:00",
        "pc_apptstatus": "@",
        "pc_title": "Office Visit",
        "facility_name": "Main Clinic",
        "pc_hometext": "Follow-up",
    }
    base.update(overrides)
    return base


def _make_patient(**overrides: Any) -> dict[str, Any]:
    base = {"pid": 10, "fname": "John", "lname": "Doe", "DOB": "1980-01-15"}
    base.update(overrides)
    return base


def _mock_client(
    patients: list | None = None,
    appointments: list | None = None,
    patient_appointments: dict[int, list] | None = None,
) -> AsyncMock:
    """Build a mock OpenEMRClient with configurable GET responses."""
    client = AsyncMock()

    async def mock_get(path: str, params: dict | None = None) -> dict:
        if "/api/patient" in path and "/appointment" in path:
            # /api/patient/<pid>/appointment
            pid = int(path.split("/patient/")[1].split("/")[0])
            data = (patient_appointments or {}).get(pid, [])
            return {"data": data}
        if "/api/patient" in path:
            return {"data": patients or []}
        if "/api/appointment" in path:
            return {"data": appointments or []}
        return {"data": []}

    client.get = AsyncMock(side_effect=mock_get)
    return client


# -- _format_appointment -------------------------------------------------------


def test_format_appointment_basic():
    raw = _make_appointment()
    out = _format_appointment(raw)
    assert out["appointment_id"] == 1
    assert out["patient_name"] == "John Doe"
    assert out["patient_id"] == 10
    assert out["provider_name"] == "Dr Smith"
    assert out["date"] == "2026-03-01"
    assert out["start_time"] == "09:00:00"
    assert out["end_time"] == "09:30:00"
    assert out["status"] == "@"
    assert out["status_label"] == "Arrived"
    assert out["category"] == "Office Visit"
    assert out["facility"] == "Main Clinic"
    assert out["reason"] == "Follow-up"


def test_format_appointment_unknown_status():
    raw = _make_appointment(pc_apptstatus="Z")
    out = _format_appointment(raw)
    assert out["status_label"] == "Z"


# -- search by patient_id (direct) --------------------------------------------


@pytest.mark.asyncio
async def test_search_by_patient_id():
    appts = [_make_appointment(), _make_appointment(pc_eid=2)]
    client = _mock_client(patient_appointments={10: appts})

    result = await _find_appointments_impl(client, patient_id=10)

    assert result["total_count"] == 2
    assert len(result["appointments"]) == 2
    assert result["appointments"][0]["appointment_id"] == 1


# -- search by patient_name ---------------------------------------------------


@pytest.mark.asyncio
async def test_search_by_patient_name_single_match():
    patients = [_make_patient()]
    appts = [_make_appointment()]
    client = _mock_client(patients=patients, patient_appointments={10: appts})

    result = await _find_appointments_impl(client, patient_name="Doe")

    assert result["total_count"] == 1
    assert result["appointments"][0]["patient_name"] == "John Doe"


@pytest.mark.asyncio
async def test_search_by_patient_name_no_match():
    client = _mock_client(patients=[])

    result = await _find_appointments_impl(client, patient_name="Nobody")

    assert result["total_count"] == 0
    assert "No patients found" in result["message"]


@pytest.mark.asyncio
async def test_search_by_patient_name_ambiguous():
    """More than 5 matches triggers disambiguation."""
    patients = [_make_patient(pid=i, fname=f"P{i}") for i in range(6)]
    client = _mock_client(patients=patients)

    result = await _find_appointments_impl(client, patient_name="P")

    assert result["total_count"] == 0
    assert "Multiple patients" in result["message"]
    assert "matching_patients" in result
    assert len(result["matching_patients"]) == 6


# -- search all appointments (no patient filter) -------------------------------


@pytest.mark.asyncio
async def test_search_all_appointments():
    appts = [
        _make_appointment(pc_eid=1),
        _make_appointment(pc_eid=2, pc_eventDate="2026-03-02"),
    ]
    client = _mock_client(appointments=appts)

    result = await _find_appointments_impl(client)

    assert result["total_count"] == 2


# -- date filter ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_filter_by_date():
    appts = [
        _make_appointment(pc_eid=1, pc_eventDate="2026-03-01"),
        _make_appointment(pc_eid=2, pc_eventDate="2026-03-02"),
    ]
    client = _mock_client(appointments=appts)

    result = await _find_appointments_impl(client, date="2026-03-01")

    assert result["total_count"] == 1
    assert result["appointments"][0]["date"] == "2026-03-01"


# -- status filter -------------------------------------------------------------


@pytest.mark.asyncio
async def test_filter_by_status():
    appts = [
        _make_appointment(pc_eid=1, pc_apptstatus="@"),
        _make_appointment(pc_eid=2, pc_apptstatus="-"),
    ]
    client = _mock_client(appointments=appts)

    result = await _find_appointments_impl(client, status="-")

    assert result["total_count"] == 1
    assert result["appointments"][0]["status"] == "-"


# -- provider filter -----------------------------------------------------------


@pytest.mark.asyncio
async def test_filter_by_provider():
    appts = [
        _make_appointment(pc_eid=1, pce_aid_fname="Dr", pce_aid_lname="Smith"),
        _make_appointment(pc_eid=2, pce_aid_fname="Dr", pce_aid_lname="Jones"),
    ]
    client = _mock_client(appointments=appts)

    result = await _find_appointments_impl(client, provider_name="Jones")

    assert result["total_count"] == 1
    assert result["appointments"][0]["provider_name"] == "Dr Jones"


# -- combined filters ----------------------------------------------------------


@pytest.mark.asyncio
async def test_combined_filters():
    appts = [
        _make_appointment(pc_eid=1, pc_eventDate="2026-03-01", pc_apptstatus="@"),
        _make_appointment(pc_eid=2, pc_eventDate="2026-03-01", pc_apptstatus="-"),
        _make_appointment(pc_eid=3, pc_eventDate="2026-03-02", pc_apptstatus="@"),
    ]
    client = _mock_client(patient_appointments={10: appts})

    result = await _find_appointments_impl(
        client, patient_id=10, date="2026-03-01", status="@"
    )

    assert result["total_count"] == 1
    assert result["appointments"][0]["appointment_id"] == 1


# -- no results ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_results_message():
    client = _mock_client(appointments=[])

    result = await _find_appointments_impl(client, date="2099-01-01")

    assert result["total_count"] == 0
    assert "No appointments found" in result["message"]


# -- input schema validation ---------------------------------------------------


def test_input_schema_all_optional():
    """All fields are optional — empty input is valid."""
    inp = FindAppointmentsInput()
    assert inp.patient_name is None
    assert inp.date is None
    assert inp.patient_id is None


def test_input_schema_with_values():
    inp = FindAppointmentsInput(patient_name="Doe", date="2026-03-01", patient_id=5)
    assert inp.patient_name == "Doe"
    assert inp.date == "2026-03-01"
    assert inp.patient_id == 5
