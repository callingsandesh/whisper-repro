"""Tests for the dataset layer: the Sample contract and the registry."""

import json

import pytest

from whisper_repro.datasets import Sample, available, get_adapter
from whisper_repro.datasets.base import read_manifest, write_manifest


def test_sample_round_trips_through_json():
    s = Sample(sample_id="np_0001", audio_path="np_0001.flac",
               reference="म नेपाली बोल्छु", language="ne",
               duration_s=2.5, speaker_id="spk1", meta={"device": "phone"})
    assert Sample.from_json(s.to_json()) == s


def test_devanagari_survives_serialisation():
    """ensure_ascii=False matters: manifests must stay human-readable."""
    s = Sample("x", "x.flac", "नेपाली", "ne")
    assert "नेपाली" in s.to_json()
    assert json.loads(s.to_json())["reference"] == "नेपाली"


def test_registry_lists_every_adapter():
    names = available()
    assert "librispeech_test_clean" in names
    assert "nepali_custom" in names


def test_unknown_dataset_names_the_alternatives():
    with pytest.raises(KeyError, match="librispeech"):
        get_adapter("does_not_exist")


def test_manifest_round_trip(tmp_path, monkeypatch):
    import whisper_repro.paths as paths
    monkeypatch.setattr(paths, "MANIFEST_DIR", tmp_path)
    monkeypatch.setattr("whisper_repro.datasets.base.manifest_path",
                        lambda name: tmp_path / f"{name}.jsonl")
    samples = [Sample(f"s{i}", f"s{i}.flac", f"ref {i}", "en") for i in range(3)]
    write_manifest("tmp_ds", samples)
    assert read_manifest("tmp_ds") == samples
    assert read_manifest("tmp_ds", limit=2) == samples[:2]
