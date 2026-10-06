"""Tests for the risk bands: boundaries, consistency with the flag, and the summary."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.risk_bands import band_summary, risk_band

THRESHOLD = 0.52


def test_bands_at_and_around_each_boundary():
    assert risk_band(0.0, THRESHOLD) == "Low"
    assert risk_band(0.2499, THRESHOLD) == "Low"
    assert risk_band(0.25, THRESHOLD) == "Medium"
    assert risk_band(0.5199, THRESHOLD) == "Medium"
    assert risk_band(THRESHOLD, THRESHOLD) == "High"
    assert risk_band(1.0, THRESHOLD) == "High"


def test_high_band_is_exactly_the_flagged_applicants():
    for score in (0.1, 0.3, 0.51, 0.52, 0.9):
        assert (risk_band(score, THRESHOLD) == "High") == (score >= THRESHOLD)


def test_rejects_cut_offs_in_the_wrong_order():
    with pytest.raises(ValueError):
        risk_band(0.3, operating_threshold=0.2)


def test_band_summary_shares_add_up():
    y = [0, 0, 0, 1, 0, 1, 1, 0]
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    summary = band_summary(y, scores, THRESHOLD)

    assert list(summary.index) == ["Low", "Medium", "High"]
    assert summary["share_of_applicants"].sum() == pytest.approx(1.0)
    assert summary["share_of_defaulters"].sum() == pytest.approx(1.0)
    assert summary.loc["Low", "default_rate"] == 0.0
    assert summary.loc["High", "default_rate"] == pytest.approx(2 / 3)
