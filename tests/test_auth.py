import os
import sys

# 将插件目录加入 sys.path
plugin_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if plugin_root not in sys.path:
    sys.path.insert(0, plugin_root)

import json
import pytest
from vertexai_maibot.auth import (
    VertexAuthInfo,
    resolve_vertex_auth,
)

SAMPLE_PRIVATE_KEY = (
    "-----BEGIN PRIVATE KEY-----\n"
    "MIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQDK9yZ3ZTOvXM9R\n"
    "KgGDnHOTXtmJt152e7GXRVkmKyTWOZBvWvrjyp5MMraGi/w79Zx3EJVGB8asZkcS\n"
    "ITPMDfbZ5WIQskIg040fMCOd3ZF2e3Hs1L8WmXaQVRfOh+iZ1w1Ty85OLRJns/Vs\n"
    "xyMIM8dDUk2jYL4gl/GHTBBnlKMy0eBSHP1kpjhG+G16HiTSlWOq+oo1V4HS5Kjb\n"
    "zwcD2i6o8/W1KFF/FUWwQ/zF5pudaH4Tdor9YrGzcPAbAb3Hbc2OxWzsIrggLQdx\n"
    "WGALviQh4RP+Zk116VX3E/ay+G6euK2e8CEl8JCOFFAln5fv+g7WcGkXfBKaM2RW\n"
    "KMh1OcX1AgMBAAECggEACw+P+Za6y9Y3eBldbsIybDgRi9dB3xwbj/5X/uQABGqr\n"
    "q93d6VcL7GQhHVD4pH3ckoKFWAUpdxC108+aZobmSnadMwK75tUQjkAN2NcPy9Cr\n"
    "Yzgu17jyRKhWfUtbyjJ1h/kF263KIHumptBtYM+NjuYn0/z8L4OIRpLHHBX2FNlR\n"
    "OHaziI3lZX3AKhYCPjKzreT1ByfBOtQMnttLXiH3VakwIQkrjUEOnSpY8YsEzsGp\n"
    "QPsSm5erK7fvTOPUdsI0Mxt3pMu+qV/xNXVBXNir64AS3JCXmLSZppoRgshH/sF6\n"
    "Y/vFkJU8aY8zh7dFD8mATUwFJrNQfEF7FEiC7iCGawKBgQD59Qoz1mHEcWBJzVrM\n"
    "PUvTViLfQ4UJMZTpq3dKkBoDgmUw45c0Iv9owUjaLQZWx7T6FIbiESdyJJgXkw0r\n"
    "qWhEkqyPJRdLoEyqz1Z2jU/Rd8Oj6IgQ03fPGxxW+mR1o6yjveGT2mEt/vH57eSi\n"
    "0fd67leeXprmboRUECNZeTVTUwKBgQDP30h5i0w7rqmLzLgj/fpU+YOhVOXCz709\n"
    "UGOeY2JOzTVvwYZ20erLLVgYLQCkeFeWTK8bYbFbDgh5kc1D+zWrlUxQQ+F+soKH\n"
    "UxeJOOZ5EmhjPrpr/iRgB+P51Ofsk94eM+fXkId3AI6sKo5IJxvWi9v58UsHSXf9\n"
    "i+a9ZA/glwKBgFaP7cCV9bW6LcZhP5iLgb7mjZ0UHPZaHV/u6ZkA3oFonvKXXr3H\n"
    "gPweM8Sa4x8aeipBAZU4q5hz49gGLQF54Yod4XSRO8gM84AT3bWgqou4FiClXrm1\n"
    "AkDO+rmEbVXwINFz6LLzZcQwDkynfg6E+FeyQtWtnac/jsQt8pAB0d3RAoGAKMVM\n"
    "e7pci2AolOjkPPr1fHEr5hq2rcz8+uengktP3tDU6ER20Rs/B/auzp8U/y0zUKFd\n"
    "vuscVBWeh+mW7vx4fFzeCOgndCgi/dPIS/vDCIAZxU1qVwATtloV1sxxaK9f42Ce\n"
    "2mLYjvinAxGdsn0pMp1f2df/tUAtrqcf7ImRq6UCgYB2o4csRsE1yTil7+D41SwJ\n"
    "Ta1scXmAl0IFV5fty8GvYuAknkIViRe1t7vT/edaRmnpi6da6VTKuY2ABzy4SiFn\n"
    "N6CQFzeIpPASAUYpBx1HM+JEDArC1rwGYHUiBJ635HtssNAk85vv9xssvxubWxiw\n"
    "WgUBPvu6X+DZ7ruYzej3QQ==\n"
    "-----END PRIVATE KEY-----\n"
)


def test_auth_ya29_token():
    auth_info = resolve_vertex_auth(
        provider_data={"api_key": "ya29.sample_token_value", "project": "my-test-proj", "location": "us-central1"}
    )
    assert auth_info.project == "my-test-proj"
    assert auth_info.location == "us-central1"
    assert auth_info.credentials is not None
    assert auth_info.credentials.token == "ya29.sample_token_value"


def test_auth_express_key():
    auth_info = resolve_vertex_auth(
        provider_data={"api_key": "AIzaSy_sample_key", "project": "my-test-proj", "location": "europe-west1"}
    )
    assert auth_info.project == "my-test-proj"
    assert auth_info.location == "europe-west1"
    assert auth_info.api_key == "AIzaSy_sample_key"


def test_auth_sa_json_string():
    sa_data = {
        "type": "service_account",
        "project_id": "sa-proj",
        "token_uri": "https://oauth2.googleapis.com/token",
        "private_key": SAMPLE_PRIVATE_KEY,
        "client_email": "test@sa-proj.iam.gserviceaccount.com",
    }
    sa_json = json.dumps(sa_data)
    auth_info = resolve_vertex_auth(
        provider_data={"api_key": sa_json, "location": "asia-east1"}
    )
    assert auth_info.project == "sa-proj"
    assert auth_info.location == "asia-east1"
    assert auth_info.credentials is not None


def test_auth_location_from_base_url():
    auth_info = resolve_vertex_auth(
        provider_data={
            "api_key": "AIzaSy_sample_key",
            "project": "proj-1",
            "base_url": "https://us-east4-aiplatform.googleapis.com/v1",
        }
    )
    assert auth_info.location == "us-east4"
