"""Tests for serving the React page from the API, and the /examples endpoint.
They use a small stand-in page and no model, so they run on any machine."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from backend.app import FRONTEND_DIST, create_app


def _built_page(folder):
    (folder / "assets").mkdir(parents=True)
    (folder / "index.html").write_text("<html><body><div id='root'></div></body></html>", encoding="utf-8")
    (folder / "assets" / "app.js").write_text("console.log('page');", encoding="utf-8")
    return folder


def test_built_page_and_its_files_are_served(tmp_path):
    with TestClient(create_app(tmp_path / "no_model", _built_page(tmp_path / "dist"))) as client:
        page, script = client.get("/"), client.get("/assets/app.js")

    assert page.status_code == 200 and "id='root'" in page.text
    assert script.status_code == 200 and "console.log" in script.text


def test_api_routes_still_answer_when_the_page_is_mounted(tmp_path):
    with TestClient(create_app(tmp_path / "no_model", _built_page(tmp_path / "dist"))) as client:
        # no model in this test, so the API answers 503 in JSON rather than the page answering
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["error"] == "model_not_available"


def test_missing_build_shows_how_to_build_it(tmp_path):
    with TestClient(create_app(tmp_path / "no_model", tmp_path / "missing_dist")) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "npm run build" in response.text


def test_examples_endpoint_returns_one_applicant_per_band(tmp_path):
    with TestClient(create_app(tmp_path / "no_model", tmp_path / "missing_dist")) as client:
        examples = client.get("/examples").json()

    assert [e["expected_band"] for e in examples] == ["Low", "Medium", "High"]
    assert all("age_years" in e["application"] for e in examples)


def test_the_real_build_is_present():
    # frontend/dist/ is kept in git so the system runs with Python only
    assert (FRONTEND_DIST / "index.html").exists(), "run npm run build in frontend/"
