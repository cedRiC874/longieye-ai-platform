from __future__ import annotations

import hashlib
import math
import struct
import zlib
from pathlib import Path

import pytest

from longieye.imaging import (
    CANONICAL_FIXTURE_PIXEL_SHA256,
    IMAGE_SIDE,
    MAX_ENCODED_BYTES,
    ImageArtifactError,
    ImageQualityGate,
    RasterImage,
    decode_synthetic_png,
    encode_canonical_png,
)


def synthetic_fundus_rgb(*, mirror: bool = False) -> bytes:
    pixels = bytearray()
    center = (IMAGE_SIDE - 1) / 2.0
    radius = 58.0
    for y in range(IMAGE_SIDE):
        for x in range(IMAGE_SIDE):
            source_x = IMAGE_SIDE - 1 - x if mirror else x
            distance = math.hypot(source_x - center, y - center)
            if distance > radius:
                pixels.extend((0, 0, 0))
                continue
            radial = 1.0 - distance / radius
            texture = ((source_x * 17 + y * 31) % 17) - 8
            red = 135 + int(55 * radial) + texture
            green = 42 + int(38 * radial) + texture // 3
            blue = 24 + int(24 * radial) + texture // 4

            disc_x = 91
            if math.hypot(source_x - disc_x, y - 64) <= 10:
                red, green, blue = 232, 181, 96
            vessel_a = abs((y - 64) - int(0.18 * (source_x - disc_x))) <= 1
            vessel_b = abs((y - 64) + int(0.32 * (source_x - disc_x))) <= 1
            if source_x < disc_x and (vessel_a or vessel_b):
                red, green, blue = 52, 14, 12
            pixels.extend((red, green, blue))
    return bytes(pixels)


def chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )


def chunks(png: bytes) -> list[tuple[bytes, bytes]]:
    parsed = []
    offset = 8
    while offset < len(png):
        length = struct.unpack(">I", png[offset : offset + 4])[0]
        chunk_type = png[offset + 4 : offset + 8]
        payload = png[offset + 8 : offset + 8 + length]
        parsed.append((chunk_type, payload))
        offset += 12 + length
    return parsed


def rebuild_png(parts: list[tuple[bytes, bytes]]) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"".join(chunk(kind, data) for kind, data in parts)


def encoded_image(rgb: bytes | None = None) -> bytes:
    return encode_canonical_png(
        IMAGE_SIDE,
        IMAGE_SIDE,
        synthetic_fundus_rgb() if rgb is None else rgb,
    )


def decoded_image(rgb: bytes | None = None, *, eye: str = "od"):
    pixels = synthetic_fundus_rgb() if rgb is None else rgb
    return decode_synthetic_png(
        encoded_image(pixels), eye, hashlib.sha256(pixels).hexdigest()
    )


def assert_error_code(expected: str, callback) -> None:
    with pytest.raises(ImageArtifactError) as exc_info:
        callback()
    assert exc_info.value.code == expected


def test_canonical_png_roundtrip_is_byte_stable_and_hash_bound():
    rgb = synthetic_fundus_rgb()
    first = encoded_image(rgb)
    second = encoded_image(rgb)
    expected_pixel_sha256 = hashlib.sha256(rgb).hexdigest()

    image = decode_synthetic_png(first, "od", expected_pixel_sha256)

    assert first == second
    assert image.eye == "od"
    assert (image.width, image.height) == (128, 128)
    assert image.rgb == rgb
    assert image.pixel_sha256 == expected_pixel_sha256
    assert image.provenance_verified is True
    assert image.encoded_sha256 == hashlib.sha256(first).hexdigest()
    assert [kind for kind, _ in chunks(first)] == [b"IHDR", b"IDAT", b"IEND"]


def test_decoder_rejects_invalid_crc_without_echoing_input():
    corrupted = bytearray(encoded_image())
    corrupted[29] ^= 0x01
    with pytest.raises(ImageArtifactError) as exc_info:
        decode_synthetic_png(bytes(corrupted), "od")
    assert exc_info.value.code == "image_crc_mismatch"
    assert "IHDR" not in str(exc_info.value)


def test_decoder_rejects_metadata_and_duplicate_or_reordered_chunks():
    original = chunks(encoded_image())
    metadata = [original[0], (b"tEXt", b"patient=private"), *original[1:]]
    duplicate_idat = [original[0], original[1], original[1], original[2]]
    reordered = [original[1], original[0], original[2]]

    for candidate in (metadata, duplicate_idat, reordered):
        assert_error_code(
            "image_invalid", lambda candidate=candidate: decode_synthetic_png(rebuild_png(candidate), "od")
        )


def test_decoder_rejects_oversize_bad_dimensions_and_truncation():
    assert_error_code(
        "image_too_large",
        lambda: decode_synthetic_png(b"\x89PNG\r\n\x1a\n" + b"x" * MAX_ENCODED_BYTES, "od"),
    )

    original = chunks(encoded_image())
    ihdr = bytearray(original[0][1])
    ihdr[:4] = struct.pack(">I", 64)
    wrong_dimensions = rebuild_png([(b"IHDR", bytes(ihdr)), *original[1:]])
    assert_error_code(
        "image_invalid", lambda: decode_synthetic_png(wrong_dimensions, "od")
    )
    assert_error_code(
        "image_invalid", lambda: decode_synthetic_png(encoded_image()[:-1], "od")
    )


def test_decoder_rejects_nonzero_filter_decompression_overrun_and_zlib_tail():
    original = chunks(encoded_image())
    scanlines = bytearray(zlib.decompress(original[1][1]))
    scanlines[0] = 1
    nonzero_filter = rebuild_png(
        [original[0], (b"IDAT", zlib.compress(bytes(scanlines))), original[2]]
    )
    overrun = rebuild_png(
        [
            original[0],
            (b"IDAT", zlib.compress(bytes(scanlines) + b"x")),
            original[2],
        ]
    )
    trailing = rebuild_png(
        [original[0], (b"IDAT", zlib.compress(bytes(scanlines)) + b"tail"), original[2]]
    )

    for candidate in (nonzero_filter, overrun, trailing):
        assert_error_code(
            "image_invalid", lambda candidate=candidate: decode_synthetic_png(candidate, "od")
        )


def test_decoder_rejects_noncanonical_but_otherwise_valid_deflate_stream():
    original = chunks(encoded_image())
    scanlines = zlib.decompress(original[1][1])
    candidate = rebuild_png(
        [original[0], (b"IDAT", zlib.compress(scanlines)), original[2]]
    )

    assert candidate != encoded_image()
    assert_error_code("image_invalid", lambda: decode_synthetic_png(candidate, "od"))


def test_encoder_and_decoder_reject_bad_pixel_contract_laterality_and_hash():
    assert_error_code(
        "image_pixel_invalid", lambda: encode_canonical_png(64, 128, b"x")
    )
    assert_error_code(
        "image_laterality_invalid", lambda: decode_synthetic_png(encoded_image(), "right")
    )
    assert_error_code(
        "image_integrity_mismatch",
        lambda: decode_synthetic_png(encoded_image(), "os", "0" * 64),
    )


def test_unbound_decode_is_marked_unverified_and_cannot_be_prepared():
    image = decode_synthetic_png(encoded_image(), "od")
    assert image.provenance_verified is False
    assert ImageQualityGate().evaluate(image).reason_codes[0] == "provenance_unverified"
    assert_error_code("image_quality_rejected", lambda: ImageQualityGate().prepare(image))


def test_raster_constructor_recomputes_both_content_hashes():
    rgb = synthetic_fundus_rgb()
    encoded = encoded_image(rgb)
    values = {
        "eye": "od",
        "width": 128,
        "height": 128,
        "rgb": rgb,
        "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
        "pixel_sha256": hashlib.sha256(rgb).hexdigest(),
        "provenance_verified": True,
    }
    assert RasterImage(**values).rgb == rgb
    assert_error_code(
        "image_pixel_invalid",
        lambda: RasterImage(**(values | {"encoded_sha256": "0" * 64})),
    )
    assert_error_code(
        "image_pixel_invalid",
        lambda: RasterImage(**(values | {"pixel_sha256": "0" * 64})),
    )


@pytest.mark.parametrize(
    ("rgb", "reason"),
    [
        (bytes([0, 0, 0]) * (IMAGE_SIDE * IMAGE_SIDE), "brightness_low"),
        (bytes([255, 255, 255]) * (IMAGE_SIDE * IMAGE_SIDE), "brightness_high"),
        (bytes([128, 128, 128]) * (IMAGE_SIDE * IMAGE_SIDE), "contrast_low"),
    ],
    ids=("dark", "bright", "low-contrast"),
)
def test_quality_gate_rejects_dark_bright_and_low_contrast(rgb, reason):
    report = ImageQualityGate().evaluate(decoded_image(rgb))
    assert report.accepted is False
    assert reason in report.reason_codes
    assert_error_code(
        "image_quality_rejected",
        lambda: ImageQualityGate().prepare(decoded_image(rgb)),
    )


def test_quality_gate_rejects_contrasty_but_blurry_low_frequency_image():
    pixels = bytearray()
    for _y in range(IMAGE_SIDE):
        for x in range(IMAGE_SIDE):
            value = round(128 + 75 * math.sin(2 * math.pi * x / IMAGE_SIDE))
            pixels.extend((value, value, value))
    report = ImageQualityGate().evaluate(decoded_image(bytes(pixels)))

    assert report.contrast >= 0.08
    assert report.accepted is False
    assert "sharpness_low" in report.reason_codes


def test_valid_synthetic_image_quality_and_area_pooling_are_deterministic():
    image = decoded_image()
    gate = ImageQualityGate()

    first_report = gate.evaluate(image)
    second_report = gate.evaluate(image)
    first = gate.prepare(image)
    second = gate.prepare(image)

    assert first_report == second_report
    assert first_report.accepted is True
    assert first_report.reason_codes == ()
    assert first == second
    assert first.eye == "od"
    assert first.side == 32
    assert len(first.pixels) == 32 * 32 * 3
    assert all(0.0 <= value <= 1.0 for value in first.pixels)
    assert first.quality == first_report
    assert first.source_pixel_sha256 == image.pixel_sha256


def test_laterality_is_explicit_and_mirrored_pixels_have_distinct_hashes():
    od_rgb = synthetic_fundus_rgb()
    os_rgb = synthetic_fundus_rgb(mirror=True)
    od = decode_synthetic_png(encoded_image(od_rgb), "od", hashlib.sha256(od_rgb).hexdigest())
    os = decode_synthetic_png(encoded_image(os_rgb), "os", hashlib.sha256(os_rgb).hexdigest())

    assert od.eye == "od"
    assert os.eye == "os"
    assert od.pixel_sha256 != os.pixel_sha256


def test_canonical_eye_registry_is_immutable():
    with pytest.raises(TypeError):
        CANONICAL_FIXTURE_PIXEL_SHA256["os"] = CANONICAL_FIXTURE_PIXEL_SHA256["od"]


def test_demo_card_hash_table_matches_both_published_fixtures():
    project_root = Path(__file__).resolve().parents[1]
    card = (project_root / "docs" / "MULTIMODAL_DEMO_CARD.md").read_text(
        encoding="utf-8"
    )
    for eye in ("od", "os"):
        relative_path = f"examples/synthetic_fundus/{eye}.png"
        encoded = (project_root / relative_path).read_bytes()
        encoded_sha256 = hashlib.sha256(encoded).hexdigest()
        image = decode_synthetic_png(
            encoded, eye, CANONICAL_FIXTURE_PIXEL_SHA256[eye]
        )
        expected_row = (
            f"| {eye.upper()} | `{relative_path}` | `{encoded_sha256}` | "
            f"`{image.pixel_sha256}` |"
        )
        assert expected_row in card
