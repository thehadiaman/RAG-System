"""Model-agnostic chat messages and images, shared by the Ollama and Bedrock clients."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ImageInput:
    data: bytes
    format: str  # "png", "jpeg", "gif" or "webp"

    @classmethod
    def from_bytes(cls, data: bytes) -> ImageInput:
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return cls(data, "png")
        if data.startswith(b"\xff\xd8\xff"):
            return cls(data, "jpeg")
        if data[:6] in (b"GIF87a", b"GIF89a"):
            return cls(data, "gif")
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return cls(data, "webp")
        raise ValueError("Unsupported image format (expected PNG, JPEG, GIF or WebP)")


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    text: str
    images: tuple[ImageInput, ...] = ()


def solid_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """A plain single-colour PNG. Used for connection checks and tests."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    row = b"\x00" + bytes(rgb) * width  # filter byte + RGB pixels
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(row * height))
        + chunk(b"IEND", b"")
    )
