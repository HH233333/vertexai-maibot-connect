import os
import sys

plugin_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if plugin_root not in sys.path:
    sys.path.insert(0, plugin_root)

from vertexai_maibot.client_manager import VertexClientManager, _build_http_options


def test_build_http_options_disables_sdk_retries():
    options = _build_http_options({})
    assert options.retry_options is not None
    assert options.retry_options.attempts == 1


def test_build_http_options_timeout_and_base_url():
    options = _build_http_options({
        "timeout": 30,
        "base_url": "https://aiplatform.us.rep.googleapis.com/v1",
    })
    assert options.timeout == 30000
    assert options.base_url == "https://aiplatform.us.rep.googleapis.com/"
    assert options.api_version == "v1"
    assert options.retry_options is not None
    assert options.retry_options.attempts == 1


def test_client_manager_creates_client_with_retry_disabled():
    manager = VertexClientManager(default_project="test-proj", default_location="us-central1")
    client, auth_info = manager.get_client({
        "api_key": "AQ.dummy_api_key_for_test",
        "project": "test-proj",
        "location": "us-central1",
    })
    assert client._api_client._http_options.retry_options.attempts == 1
