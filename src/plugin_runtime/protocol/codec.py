"""MsgPack / JSON 编解码器

提供统一的消息编解码接口，生产环境默认使用 MsgPack，
开发调试模式可切换为 JSON（仅编解码切换，传输层不变）。
"""

from typing import Any

import json

import msgpack

from .envelope import Envelope


class Codec:
    """消息编解码器基类"""

    def encode_envelope(self, envelope: Envelope) -> bytes:
        raise NotImplementedError

    def decode_envelope(self, data: bytes) -> Envelope:
        raise NotImplementedError

    def encode(self, obj: dict[str, Any]) -> bytes:
        raise NotImplementedError

    def decode(self, data: bytes) -> dict[str, Any]:
        raise NotImplementedError


class MsgPackCodec(Codec):
    """MsgPack 编解码器（生产默认）"""

    def encode(self, obj: dict[str, Any]) -> bytes:
        return msgpack.packb(obj, use_bin_type=True)

    def decode(self, data: bytes) -> dict[str, Any]:
        result = msgpack.unpackb(data, raw=False)
        if not isinstance(result, dict):
            raise ValueError(f"期望解码为 dict，实际为 {type(result)}")
        return result

    def encode_envelope(self, envelope: Envelope) -> bytes:
        return self.encode(envelope.model_dump())

    def decode_envelope(self, data: bytes) -> Envelope:
        raw = self.decode(data)
        return Envelope.model_validate(raw)


class JsonCodec(Codec):
    """JSON 编解码器（开发调试用）"""

    def encode(self, obj: dict[str, Any]) -> bytes:
        return json.dumps(obj, ensure_ascii=False).encode("utf-8")

    def decode(self, data: bytes) -> dict[str, Any]:
        result = json.loads(data.decode("utf-8"))
        if not isinstance(result, dict):
            raise ValueError(f"期望解码为 dict，实际为 {type(result)}")
        return result

    def encode_envelope(self, envelope: Envelope) -> bytes:
        return self.encode(envelope.model_dump())

    def decode_envelope(self, data: bytes) -> Envelope:
        raw = self.decode(data)
        return Envelope.model_validate(raw)


def create_codec(use_json: bool = False) -> Codec:
    """创建编解码器实例

    Args:
        use_json: 是否使用 JSON（开发模式）。默认使用 MsgPack。
    """
    if use_json:
        return JsonCodec()
    return MsgPackCodec()
