from __future__ import annotations

from pathlib import Path

from config.settings import load_settings


def test_load_settings_resolves_paths(monkeypatch):
    monkeypatch.delenv("NIM_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NIM_MODEL", raising=False)
    settings = load_settings("config.yaml")

    assert settings.categories[0].name == "sutures"
    assert settings.output.dir == Path.cwd() / "output"
    assert settings.output.checkpoint_db == Path.cwd() / "checkpoint.db"


def test_load_settings_accepts_nvidia_api_key_alias(monkeypatch):
    monkeypatch.delenv("NIM_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-test")
    monkeypatch.setenv("NIM_MODEL", "meta/llama-4-maverick-17b-128e-instruct")
    monkeypatch.setenv(
        "NIM_BASE_URL", "https://integrate.api.nvidia.com/v1/chat/completions"
    )
    settings = load_settings("config.yaml")

    assert settings.llm.api_key == "nv-test"
    assert settings.llm.base_url == "https://integrate.api.nvidia.com/v1"
    assert settings.llm.model == "meta/llama-4-maverick-17b-128e-instruct"
