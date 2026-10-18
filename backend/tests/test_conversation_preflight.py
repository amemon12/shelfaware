"""Preflight is opt-in, neutral, provider-selected, and redacts failures."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from app.conversation_brain import BrainSettings, Intent, VisionCandidates


@pytest.fixture
def preflight(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[2] / "scripts" / "conversation_preflight.py"
    spec = importlib.util.spec_from_file_location("isolated_conversation_preflight", path)
    module = importlib.util.module_from_spec(spec)
    # Never read the developer's private .env in this test module.
    with patch("dotenv.load_dotenv"):
        spec.loader.exec_module(module)
    state = SimpleNamespace(calls=[], instances=[], effects=[])
    state.settings = BrainSettings(provider="anthropic", model="claude-sonnet-5-5",
        anthropic_key="test-secret-claude", deepseek_key="test-secret-unused-deepseek",
        catalog_path=str(tmp_path / "empty-catalog.json"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-secret-claude")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-secret-speech")
    monkeypatch.setattr(module.BrainSettings, "from_env", lambda: state.settings)
    monkeypatch.setattr(module.ConversationConfig, "from_env", lambda:
        SimpleNamespace(transcription_provider="elevenlabs", speech_provider="elevenlabs"))

    class FakeBrain:
        def __init__(self, profile, *, settings):
            self.profile, self.settings, self.closed = profile, settings, False
            state.instances.append(self)

        async def _json(self, schema, name, prompt, content, timeout):
            state.calls.append((schema, name, prompt, content, timeout))
            if state.effects:
                result = state.effects.pop(0)
                if isinstance(result, BaseException):
                    raise result
                return result
            return Intent(action="greeting") if schema is Intent else VisionCandidates(candidate_ids=[])

        async def close(self):
            self.closed = True

    monkeypatch.setattr(module, "ShoppingBrain", FakeBrain)
    state.module = module
    return state


def read_result(capsys):
    output = capsys.readouterr().out
    result, _ = json.JSONDecoder().raw_decode(output)
    return result, output


@pytest.mark.asyncio
async def test_default_never_constructs_provider_or_makes_requests(preflight, capsys):
    await preflight.module.check(False)
    result, output = read_result(capsys)
    assert result["cloud_requests_made"] == 0
    assert result["reasoning_provider"] == "anthropic"
    assert result["reasoning_model"] == "claude-sonnet-5-5"
    assert result["hardware_verified"] is False
    assert preflight.calls == [] and preflight.instances == []
    assert "test-secret" not in output


@pytest.mark.asyncio
async def test_warmup_sends_both_actual_schemas_neutral_text_only(preflight, capsys):
    await preflight.module.check(False, warm_schemas=True)
    result, output = read_result(capsys)
    assert result["cloud_requests_made"] == 2 and result["live_reasoning"] == "passed"
    assert [call[0] for call in preflight.calls] == [Intent, VisionCandidates]
    assert [check["schema"] for check in result["schema_checks"]] == ["Intent", "VisionCandidates"]
    for _, _, prompt, content, timeout in preflight.calls:
        assert timeout == 30
        assert len(content) == 1 and content[0]["type"] == "input_text"
        combined = (prompt + json.dumps(content)).lower()
        assert not any(word in combined for word in ("diabetes", "medical", "glucose", "medication", "data:image", "test-secret"))
    assert preflight.instances[0].profile == {"consented": True, "goals": [], "avoid_ingredients": []}
    assert preflight.instances[0].settings.provider == "anthropic"
    assert preflight.instances[0].closed
    assert "does not test vision" in output and "test-secret" not in output


@pytest.mark.asyncio
async def test_live_is_exactly_one_intent_probe(preflight, capsys):
    await preflight.module.check(True)
    result, _ = read_result(capsys)
    assert result["cloud_requests_made"] == 1 and len(preflight.calls) == 1
    assert preflight.calls[0][0] is Intent and preflight.instances[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_type", [TimeoutError, httpx.ReadTimeout, httpx.ConnectTimeout])
async def test_timeout_is_distinct_redacted_and_second_schema_still_checked(preflight, capsys, timeout_type):
    preflight.effects = [timeout_type("test-secret-claude private request payload"), VisionCandidates(candidate_ids=[])]
    await preflight.module.check(False, warm_schemas=True)
    result, output = read_result(capsys)
    assert result["schema_checks"][0]["status"] == "timed_out_possible_cold_schema_or_network"
    assert result["schema_checks"][1]["status"] == "passed"
    assert result["live_reasoning"] == "failed" and result["cloud_requests_made"] == 2
    assert "test-secret" not in output and "private request payload" not in output
    assert preflight.instances[0].closed


@pytest.mark.asyncio
async def test_provider_exception_does_not_print_details(preflight, capsys):
    preflight.effects = [ValueError("private health record; Authorization: test-secret-claude")]
    await preflight.module.check(True)
    result, output = read_result(capsys)
    assert result["schema_checks"][0]["status"] == "failed_check_key_model_quota_or_network_on_provider_dashboard"
    assert result["live_reasoning"] == "failed"
    assert all(value not in output for value in ("private health record", "Authorization", "test-secret"))
    assert preflight.instances[0].closed


@pytest.mark.asyncio
async def test_missing_selected_key_never_falls_back_to_present_deepseek_key(preflight, capsys):
    preflight.settings = BrainSettings(provider="anthropic", model="claude-sonnet-5-5",
        deepseek_key="test-secret-unused-deepseek", catalog_path=preflight.settings.catalog_path)
    await preflight.module.check(False, warm_schemas=True)
    result, output = read_result(capsys)
    assert result["live_reasoning"] == "not_run_missing_key"
    assert result["cloud_requests_made"] == 0 and preflight.instances == []
    assert "test-secret" not in output


@pytest.mark.asyncio
async def test_unexpected_semantic_answer_is_not_success(preflight, capsys):
    preflight.effects = [Intent(action="medical"), VisionCandidates(candidate_ids=["invented"])]
    await preflight.module.check(False, warm_schemas=True)
    result, _ = read_result(capsys)
    assert result["live_reasoning"] == "failed"
    assert all(check["status"] == "unexpected_result" for check in result["schema_checks"])
    assert preflight.instances[0].closed
