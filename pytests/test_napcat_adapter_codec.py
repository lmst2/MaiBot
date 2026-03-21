from pathlib import Path
from typing import Any, Dict

import importlib
import sys
from types import SimpleNamespace

import pytest


BUILT_IN_PLUGIN_ROOT = Path(__file__).resolve().parents[1] / "src" / "plugins" / "built_in"
if str(BUILT_IN_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(BUILT_IN_PLUGIN_ROOT))

NapCatInboundCodec = importlib.import_module("napcat_adapter.codec_inbound").NapCatInboundCodec
NapCatOutboundCodec = importlib.import_module("napcat_adapter.codec_outbound").NapCatOutboundCodec


def test_napcat_outbound_codec_supports_binary_and_forward_segments() -> None:
    codec = NapCatOutboundCodec()
    raw_message = [
        {"type": "text", "data": "hello"},
        {"type": "image", "data": "", "hash": "h1", "binary_data_base64": "aW1hZ2U="},
        {"type": "emoji", "data": "", "hash": "h2", "binary_data_base64": "ZW1vamk="},
        {"type": "voice", "data": "", "hash": "h3", "binary_data_base64": "dm9pY2U="},
        {
            "type": "reply",
            "data": {
                "target_message_id": "origin-1",
                "target_message_content": "origin text",
            },
        },
        {
            "type": "forward",
            "data": [
                {
                    "user_id": "42",
                    "user_nickname": "alice",
                    "user_cardname": "Alice",
                    "message_id": "fwd-1",
                    "content": [{"type": "text", "data": "node-text"}],
                }
            ],
        },
    ]

    converted = codec.convert_segments(raw_message)

    assert converted[0] == {"type": "text", "data": {"text": "hello"}}
    assert converted[1]["type"] == "image"
    assert converted[1]["data"]["file"] == "base64://aW1hZ2U="
    assert converted[2]["type"] == "image"
    assert converted[2]["data"]["subtype"] == 1
    assert converted[3] == {"type": "record", "data": {"file": "base64://dm9pY2U="}}
    assert converted[4] == {"type": "reply", "data": {"id": "origin-1"}}
    assert converted[5]["type"] == "node"
    assert converted[5]["data"]["name"] == "alice"
    assert converted[5]["data"]["content"] == [{"type": "text", "data": {"text": "node-text"}}]


def test_napcat_outbound_codec_builds_private_action_from_route_metadata() -> None:
    codec = NapCatOutboundCodec()
    message: Dict[str, Any] = {
        "message_info": {
            "user_info": {"user_id": "10001", "user_nickname": "tester"},
            "additional_config": {},
        },
        "raw_message": [{"type": "text", "data": "hello"}],
    }

    action_name, params = codec.build_outbound_action(message, {"target_user_id": "30001"})

    assert action_name == "send_private_msg"
    assert params == {"message": [{"type": "text", "data": {"text": "hello"}}], "user_id": "30001"}


class DummyQueryService:
    """用于测试的轻量查询服务。"""

    async def download_binary(self, url: str) -> bytes:
        """返回固定图片二进制。

        Args:
            url: 图片地址。

        Returns:
            bytes: 固定测试图片二进制。
        """
        if url:
            return b"image-bytes"
        return b""

    async def get_message_detail(self, message_id: str) -> Dict[str, Any] | None:
        """返回空消息详情。

        Args:
            message_id: 目标消息 ID。

        Returns:
            Dict[str, Any] | None: 固定空结果。
        """
        del message_id
        return None

    async def get_record_detail(self, file_name: str, file_id: str | None = None) -> Dict[str, Any] | None:
        """返回空语音详情。

        Args:
            file_name: 语音文件名。
            file_id: 可选文件 ID。

        Returns:
            Dict[str, Any] | None: 固定空结果。
        """
        del file_name
        del file_id
        return None

    async def get_forward_message(self, message_id: str) -> Dict[str, Any] | None:
        """返回空转发详情。

        Args:
            message_id: 转发消息 ID。

        Returns:
            Dict[str, Any] | None: 固定空结果。
        """
        del message_id
        return None


@pytest.mark.asyncio
async def test_napcat_inbound_codec_parses_cq_string_image_segments() -> None:
    codec = NapCatInboundCodec(SimpleNamespace(debug=lambda message: None), DummyQueryService())
    payload = {
        "message": "[CQ:image,file=test.png,sub_type=0,url=https://example.com/test.png][CQ:at,qq=10001] 看到是国人直接给你封了",
    }

    raw_message, is_at = await codec.convert_segments(payload, "10001")

    assert raw_message[0]["type"] == "image"
    assert raw_message[1] == {
        "type": "at",
        "data": {
            "target_user_id": "10001",
            "target_user_nickname": None,
            "target_user_cardname": None,
        },
    }
    assert raw_message[2] == {"type": "text", "data": " 看到是国人直接给你封了"}
    assert is_at is True
