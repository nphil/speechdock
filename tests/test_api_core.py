from types import SimpleNamespace

import pytest

from speechdock.app import clean_filename, model_for, normalize_url


class Models:
    async def list_models(self):
        return [
            {"model_id": "whisper-1", "registered": True, "translation_capable": False},
            {"model_id": "speechdock-large-v3", "registered": False, "translation_capable": True},
        ]


@pytest.mark.asyncio
async def test_translation_never_uses_turbo_or_unregistered_model():
    app = SimpleNamespace(state=SimpleNamespace(models=Models()))
    with pytest.raises(ValueError, match="cannot translate"):
        await model_for(app, "whisper-1", "translate")
    with pytest.raises(ValueError, match="not available"):
        await model_for(app, "speechdock-large-v3", "translate")
    assert (await model_for(app, "whisper-1", "transcribe"))["model_id"] == "whisper-1"


def test_url_and_display_filename_are_safe():
    assert normalize_url("https://host.example/v1/") == "https://host.example"
    with pytest.raises(ValueError):
        normalize_url("https://user:password@host.example")
    assert clean_filename("/private/video\nname.mp4") == "videoname.mp4"
