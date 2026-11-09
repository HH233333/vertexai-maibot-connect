import os
import sys

# 将插件目录加入 sys.path
plugin_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if plugin_root not in sys.path:
    sys.path.insert(0, plugin_root)

import pytest
from plugin import create_plugin, VertexAIConnectPlugin, CLIENT_TYPE


def test_plugin_creation():
    plugin = create_plugin()
    assert isinstance(plugin, VertexAIConnectPlugin)
    assert CLIENT_TYPE == "vertex_connect"
    assert hasattr(plugin, "handle_llm_provider")
    assert hasattr(plugin, "status_command")
    assert hasattr(plugin, "models_command")
