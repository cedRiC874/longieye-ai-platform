"""Strict, synthetic-only raster contracts for the offline multimodal demo.

The decoder intentionally accepts a very small PNG subset.  It does not retain
metadata, accept paths or URLs, or attempt to recover malformed images.  This
keeps image handling deterministic and prevents the public HTTP contract from
becoming an image-upload surface.
"""

from __future__ import annotations

import hashlib
import math
import struct
import zlib
from dataclasses import dataclass
from types import MappingProxyType


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
IMAGE_SIDE = 128
PREPARED_SIDE = 32
RGB_CHANNELS = 3
MAX_ENCODED_BYTES = 512 * 1024
CANONICAL_FIXTURE_PIXEL_SHA256 = MappingProxyType(
    {
        "od": "2c2e1da93031d9f9855886158bd3b1732e3bf8976bc62d2a8606d6a19d6ce9b5",
        "os": "4ac1c9e10ba2119ae983d97ac6f9cb7683667f4ab05eeb6dd2b8e1f06e1a81a4",
    }
)
_ROW_BYTES = IMAGE_SIDE * RGB_CHANNELS
_EXPECTED_RGB_BYTES = IMAGE_SIDE * _ROW_BYTES
_EXPECTED_SCANLINE_BYTES = IMAGE_SIDE * (_ROW_BYTES + 1)

_SAFE_MESSAGES = {
    "image_invalid": "合成影像不符合受支持的 PNG 合同。",
    "image_too_large": "合成影像文件超过允许大小。",
    "image_crc_mismatch": "合成影像分块完整性校验失败。",
    "image_integrity_mismatch": "合成影像像素完整性校验失败。",
    "image_laterality_invalid": "合成影像眼别无效。",
    "image_pixel_invalid": "合成影像像素数据无效。",
    "image_quality_rejected": "合成影像未通过质量检查。",
}

_QUALITY_BRIGHTNESS_MIN = 0.08
_QUALITY_BRIGHTNESS_MAX = 0.92
_QUALITY_CONTRAST_MIN = 0.08
_QUALITY_CLIPPED_MAX = 0.55
_QUALITY_FIELD_COVERAGE_MIN = 0.35
_QUALITY_SHARPNESS_MIN = 0.006
_QUALITY_REASON_CODES = {
    "provenance_unverified",
    "brightness_low",
    "brightness_high",
    "contrast_low",
    "clipped_fraction_high",
    "field_coverage_low",
    "sharpness_low",
}


class ImageArtifactError(ValueError):
    """Stable, privacy-safe image error that never includes submitted content."""

    def __init__(self, code: str) -> None:
        if code not in _SAFE_MESSAGES:
            raise ValueError("unsupported image artifact error code")
        self.code = code
        super().__init__(_SAFE_MESSAGES[code])


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_eye(eye: object) -> str:
    if not isinstance(eye, str) or eye not in {"od", "os"}:
        raise ImageArtifactError("image_laterality_invalid")
    return str(eye)


@dataclass(frozen=True)
class RasterImage:
    """Decoded 128x128 RGB image with externally supplied OD/OS laterality."""

    eye: str
    width: int
    height: int
    rgb: bytes
    encoded_sha256: str
    pixel_sha256: str
    provenance_verified: bool

    def __post_init__(self) -> None:
        _validate_eye(self.eye)
        if (
            type(self.width) is not int
            or type(self.height) is not int
            or self.width != IMAGE_SIDE
            or self.height != IMAGE_SIDE
            or type(self.rgb) is not bytes
            or len(self.rgb) != _EXPECTED_RGB_BYTES
            or not _valid_sha256(self.encoded_sha256)
            or not _valid_sha256(self.pixel_sha256)
            or type(self.provenance_verified) is not bool
            or hashlib.sha256(
                encode_canonical_png(self.width, self.height, self.rgb)
            ).hexdigest()
            != self.encoded_sha256
            or hashlib.sha256(self.rgb).hexdigest() != self.pixel_sha256
        ):
            raise ImageArtifactError("image_pixel_invalid")


@dataclass(frozen=True)
class ImageQualityReport:
    """Aggregate, non-pixel quality evidence for one synthetic image."""

    accepted: bool
    reason_codes: tuple[str, ...]
    brightness: float
    contrast: float
    clipped_fraction: float
    field_coverage: float
    sharpness: float

    def __post_init__(self) -> None:
        metrics = (
            self.brightness,
            self.contrast,
            self.clipped_fraction,
            self.field_coverage,
            self.sharpness,
        )
        if (
            type(self.accepted) is not bool
            or type(self.reason_codes) is not tuple
            or any(reason not in _QUALITY_REASON_CODES for reason in self.reason_codes)
            or len(set(self.reason_codes)) != len(self.reason_codes)
            or self.accepted != (not self.reason_codes)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
                for value in metrics
            )
        ):
            raise ImageArtifactError("image_pixel_invalid")


@dataclass(frozen=True)
class PreparedImage:
    """Quality-approved 32x32, row-major, interleaved RGB values in [0, 1]."""

    eye: str
    side: int
    pixels: tuple[float, ...]
    quality: ImageQualityReport
    source_pixel_sha256: str

    def __post_init__(self) -> None:
        _validate_eye(self.eye)
        if (
            type(self.side) is not int
            or self.side != PREPARED_SIDE
            or type(self.pixels) is not tuple
            or len(self.pixels) != PREPARED_SIDE * PREPARED_SIDE * RGB_CHANNELS
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
                for value in self.pixels
            )
            or not isinstance(self.quality, ImageQualityReport)
            or not self.quality.accepted
            or not _valid_sha256(self.source_pixel_sha256)
        ):
            raise ImageArtifactError("image_pixel_invalid")


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", checksum)
    )


def _stored_zlib_stream(payload: bytes) -> bytes:
    """Return a canonical zlib stream containing one uncompressed DEFLATE block."""

    if len(payload) > 0xFFFF:
        raise ImageArtifactError("image_pixel_invalid")
    length = len(payload)
    stored_block = (
        b"\x01"
        + struct.pack("<H", length)
        + struct.pack("<H", length ^ 0xFFFF)
        + payload
    )
    adler32 = zlib.adler32(payload) & 0xFFFFFFFF
    return b"\x78\x01" + stored_block + struct.pack(">I", adler32)


def encode_canonical_png(width: int, height: int, rgb: bytes) -> bytes:
    """Encode the fixed synthetic RGB contract into byte-stable canonical PNG."""

    if (
        type(width) is not int
        or type(height) is not int
        or width != IMAGE_SIDE
        or height != IMAGE_SIDE
        or type(rgb) is not bytes
        or len(rgb) != _EXPECTED_RGB_BYTES
    ):
        raise ImageArtifactError("image_pixel_invalid")
    pixel_bytes = rgb
    scanlines = b"".join(
        b"\x00" + pixel_bytes[offset : offset + _ROW_BYTES]
        for offset in range(0, len(pixel_bytes), _ROW_BYTES)
    )
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"".join(
        (
            PNG_SIGNATURE,
            _png_chunk(b"IHDR", ihdr),
            _png_chunk(b"IDAT", _stored_zlib_stream(scanlines)),
            _png_chunk(b"IEND", b""),
        )
    )


def _parse_exact_chunks(raw_bytes: bytes) -> tuple[bytes, bytes]:
    offset = len(PNG_SIGNATURE)
    chunks: list[tuple[bytes, bytes]] = []
    expected_types = (b"IHDR", b"IDAT", b"IEND")
    while offset < len(raw_bytes):
        if len(chunks) >= len(expected_types):
            raise ImageArtifactError("image_invalid")
        if len(raw_bytes) - offset < 12:
            raise ImageArtifactError("image_invalid")
        length = struct.unpack(">I", raw_bytes[offset : offset + 4])[0]
        chunk_end = offset + 12 + length
        if chunk_end > len(raw_bytes):
            raise ImageArtifactError("image_invalid")
        chunk_type = raw_bytes[offset + 4 : offset + 8]
        payload = raw_bytes[offset + 8 : offset + 8 + length]
        supplied_crc = struct.unpack(">I", raw_bytes[offset + 8 + length : chunk_end])[0]
        actual_crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
        if supplied_crc != actual_crc:
            raise ImageArtifactError("image_crc_mismatch")
        if chunk_type != expected_types[len(chunks)]:
            raise ImageArtifactError("image_invalid")
        chunks.append((chunk_type, payload))
        offset = chunk_end
        if chunk_type == b"IEND":
            break

    if offset != len(raw_bytes) or tuple(item[0] for item in chunks) != expected_types:
        raise ImageArtifactError("image_invalid")
    ihdr, idat, iend = (item[1] for item in chunks)
    if len(ihdr) != 13 or not idat or iend:
        raise ImageArtifactError("image_invalid")
    return ihdr, idat


def _decode_scanlines(idat: bytes) -> bytes:
    try:
        decompressor = zlib.decompressobj()
        decoded = decompressor.decompress(idat, _EXPECTED_SCANLINE_BYTES + 1)
    except (MemoryError, zlib.error):
        raise ImageArtifactError("image_invalid") from None
    if (
        len(decoded) != _EXPECTED_SCANLINE_BYTES
        or decompressor.unconsumed_tail
        or decompressor.unused_data
        or not decompressor.eof
    ):
        raise ImageArtifactError("image_invalid")

    rgb = bytearray(_EXPECTED_RGB_BYTES)
    input_offset = 0
    output_offset = 0
    for _ in range(IMAGE_SIDE):
        if decoded[input_offset] != 0:
            raise ImageArtifactError("image_invalid")
        input_offset += 1
        rgb[output_offset : output_offset + _ROW_BYTES] = decoded[
            input_offset : input_offset + _ROW_BYTES
        ]
        input_offset += _ROW_BYTES
        output_offset += _ROW_BYTES
    return bytes(rgb)


def decode_synthetic_png(
    raw_bytes: bytes,
    expected_eye: str,
    expected_pixel_sha256: str | None = None,
) -> RasterImage:
    """Decode one fixed-size synthetic PNG without accepting metadata or paths."""

    eye = _validate_eye(expected_eye)
    if type(raw_bytes) is not bytes:
        raise ImageArtifactError("image_invalid")
    if len(raw_bytes) > MAX_ENCODED_BYTES:
        raise ImageArtifactError("image_too_large")
    encoded = raw_bytes
    if not encoded.startswith(PNG_SIGNATURE):
        raise ImageArtifactError("image_invalid")

    ihdr, idat = _parse_exact_chunks(encoded)
    width, height, bit_depth, color_type, compression, filtering, interlace = (
        struct.unpack(">IIBBBBB", ihdr)
    )
    if (
        width != IMAGE_SIDE
        or height != IMAGE_SIDE
        or bit_depth != 8
        or color_type != 2
        or compression != 0
        or filtering != 0
        or interlace != 0
    ):
        raise ImageArtifactError("image_invalid")

    rgb = _decode_scanlines(idat)
    if encoded != encode_canonical_png(width, height, rgb):
        raise ImageArtifactError("image_invalid")
    pixel_sha256 = hashlib.sha256(rgb).hexdigest()
    provenance_verified = expected_pixel_sha256 is not None
    if expected_pixel_sha256 is not None and (
        not _valid_sha256(expected_pixel_sha256)
        or expected_pixel_sha256 != pixel_sha256
    ):
        raise ImageArtifactError("image_integrity_mismatch")
    return RasterImage(
        eye=eye,
        width=width,
        height=height,
        rgb=rgb,
        encoded_sha256=hashlib.sha256(encoded).hexdigest(),
        pixel_sha256=pixel_sha256,
        provenance_verified=provenance_verified,
    )


def _luminance_values(rgb: bytes) -> tuple[float, ...]:
    return tuple(
        (77 * rgb[index] + 150 * rgb[index + 1] + 29 * rgb[index + 2]) / 256.0
        for index in range(0, len(rgb), RGB_CHANNELS)
    )


class ImageQualityGate:
    """Deterministic engineering checks for the fully synthetic image branch."""

    def evaluate(self, image: RasterImage) -> ImageQualityReport:
        if not isinstance(image, RasterImage):
            raise ImageArtifactError("image_pixel_invalid")
        luminance = _luminance_values(image.rgb)
        count = len(luminance)
        mean_byte = math.fsum(luminance) / count
        brightness = mean_byte / 255.0
        variance = math.fsum((value - mean_byte) ** 2 for value in luminance) / count
        contrast = math.sqrt(variance) / 255.0
        clipped_fraction = sum(
            value <= 5.0 or value >= 250.0 for value in luminance
        ) / count
        field_coverage = sum(12.0 <= value <= 245.0 for value in luminance) / count

        horizontal = math.fsum(
            abs(luminance[row * IMAGE_SIDE + column + 1] - luminance[row * IMAGE_SIDE + column])
            for row in range(IMAGE_SIDE)
            for column in range(IMAGE_SIDE - 1)
        )
        vertical = math.fsum(
            abs(luminance[(row + 1) * IMAGE_SIDE + column] - luminance[row * IMAGE_SIDE + column])
            for row in range(IMAGE_SIDE - 1)
            for column in range(IMAGE_SIDE)
        )
        edge_count = 2 * IMAGE_SIDE * (IMAGE_SIDE - 1)
        sharpness = (horizontal + vertical) / edge_count / 255.0

        reasons: list[str] = []
        if not image.provenance_verified:
            reasons.append("provenance_unverified")
        if brightness < _QUALITY_BRIGHTNESS_MIN:
            reasons.append("brightness_low")
        if brightness > _QUALITY_BRIGHTNESS_MAX:
            reasons.append("brightness_high")
        if contrast < _QUALITY_CONTRAST_MIN:
            reasons.append("contrast_low")
        if clipped_fraction > _QUALITY_CLIPPED_MAX:
            reasons.append("clipped_fraction_high")
        if field_coverage < _QUALITY_FIELD_COVERAGE_MIN:
            reasons.append("field_coverage_low")
        if sharpness < _QUALITY_SHARPNESS_MIN:
            reasons.append("sharpness_low")

        return ImageQualityReport(
            accepted=not reasons,
            reason_codes=tuple(reasons),
            brightness=round(brightness, 6),
            contrast=round(contrast, 6),
            clipped_fraction=round(clipped_fraction, 6),
            field_coverage=round(field_coverage, 6),
            sharpness=round(sharpness, 6),
        )

    def prepare(self, image: RasterImage) -> PreparedImage:
        quality = self.evaluate(image)
        if not quality.accepted:
            raise ImageArtifactError("image_quality_rejected")

        scale = IMAGE_SIDE // PREPARED_SIDE
        area = scale * scale * 255.0
        pooled: list[float] = []
        for output_y in range(PREPARED_SIDE):
            source_y = output_y * scale
            for output_x in range(PREPARED_SIDE):
                source_x = output_x * scale
                for channel in range(RGB_CHANNELS):
                    total = 0
                    for row in range(source_y, source_y + scale):
                        start = (row * IMAGE_SIDE + source_x) * RGB_CHANNELS + channel
                        total += sum(
                            image.rgb[start + offset * RGB_CHANNELS]
                            for offset in range(scale)
                        )
                    pooled.append(total / area)

        return PreparedImage(
            eye=image.eye,
            side=PREPARED_SIDE,
            pixels=tuple(pooled),
            quality=quality,
            source_pixel_sha256=image.pixel_sha256,
        )
