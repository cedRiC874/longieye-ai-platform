"""Generate two deterministic, unmistakably synthetic fundus-like PNG fixtures.

The generator intentionally uses only integer arithmetic and fixed drawing
instructions.  It does not read source images, use randomness, or embed PNG
metadata.  The resulting fixtures exist solely to exercise Sprint 3 image
quality, encoding, fusion, and fallback paths; they are not medical images.
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from longieye.imaging import (  # noqa: E402
    decode_synthetic_png,
    encode_canonical_png,
)


WIDTH = 128
HEIGHT = 128
CHANNELS = 3
MAX_PNG_BYTES = 64 * 1024
OUTPUT_DIRECTORY = PROJECT_ROOT / "examples" / "synthetic_fundus"
OUTPUT_PATHS = {
    "od": OUTPUT_DIRECTORY / "od.png",
    "os": OUTPUT_DIRECTORY / "os.png",
}


# Five columns by seven rows.  Only glyphs used by the fixture labels are
# included, which keeps the generator small and makes the watermark auditable.
FONT_5X7 = {
    " ": (
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
    ),
    "C": (
        "01110",
        "10001",
        "10000",
        "10000",
        "10000",
        "10001",
        "01110",
    ),
    "D": (
        "11110",
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "11110",
    ),
    "E": (
        "11111",
        "10000",
        "10000",
        "11110",
        "10000",
        "10000",
        "11111",
    ),
    "H": (
        "10001",
        "10001",
        "10001",
        "11111",
        "10001",
        "10001",
        "10001",
    ),
    "I": (
        "11111",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
        "11111",
    ),
    "N": (
        "10001",
        "11001",
        "11001",
        "10101",
        "10011",
        "10011",
        "10001",
    ),
    "O": (
        "01110",
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "01110",
    ),
    "S": (
        "01111",
        "10000",
        "10000",
        "01110",
        "00001",
        "00001",
        "11110",
    ),
    "T": (
        "11111",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
    ),
    "Y": (
        "10001",
        "10001",
        "01010",
        "00100",
        "00100",
        "00100",
        "00100",
    ),
}


class FixtureError(ValueError):
    """Raised when a tracked fixture is not the canonical synthetic image."""


def _configure_utf8_output() -> None:
    """Keep Chinese status messages deterministic on redirected Windows output."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")


def _pixel_offset(x: int, y: int) -> int:
    return (y * WIDTH + x) * CHANNELS


def _set_pixel(
    pixels: bytearray, x: int, y: int, color: tuple[int, int, int]
) -> None:
    if not 0 <= x < WIDTH or not 0 <= y < HEIGHT:
        return
    offset = _pixel_offset(x, y)
    pixels[offset : offset + CHANNELS] = bytes(color)


def _draw_filled_circle(
    pixels: bytearray,
    center_x: int,
    center_y: int,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    radius_squared = radius * radius
    for y in range(center_y - radius, center_y + radius + 1):
        for x in range(center_x - radius, center_x + radius + 1):
            delta_x = x - center_x
            delta_y = y - center_y
            if delta_x * delta_x + delta_y * delta_y <= radius_squared:
                _set_pixel(pixels, x, y, color)


def _draw_line(
    pixels: bytearray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    width: int,
) -> None:
    """Rasterize a fixed-width segment with integer Bresenham stepping."""

    x0, y0 = start
    x1, y1 = end
    delta_x = abs(x1 - x0)
    step_x = 1 if x0 < x1 else -1
    delta_y = -abs(y1 - y0)
    step_y = 1 if y0 < y1 else -1
    error = delta_x + delta_y
    brush_radius = width // 2

    while True:
        for brush_y in range(-brush_radius, brush_radius + 1):
            for brush_x in range(-brush_radius, brush_radius + 1):
                _set_pixel(pixels, x0 + brush_x, y0 + brush_y, color)
        if x0 == x1 and y0 == y1:
            break
        doubled_error = error * 2
        if doubled_error >= delta_y:
            error += delta_y
            x0 += step_x
        if doubled_error <= delta_x:
            error += delta_x
            y0 += step_y


def _draw_polyline(
    pixels: bytearray,
    points: tuple[tuple[int, int], ...],
    color: tuple[int, int, int],
    width: int,
) -> None:
    for start, end in zip(points, points[1:]):
        _draw_line(pixels, start, end, color, width)


def _mirror_points(
    points: tuple[tuple[int, int], ...], eye: str
) -> tuple[tuple[int, int], ...]:
    if eye == "od":
        return points
    return tuple((WIDTH - 1 - x, y) for x, y in points)


def _draw_text(
    pixels: bytearray,
    text: str,
    start_x: int,
    start_y: int,
    color: tuple[int, int, int],
) -> None:
    cursor_x = start_x
    for character in text:
        glyph = FONT_5X7[character]
        for row, bits in enumerate(glyph):
            for column, bit in enumerate(bits):
                if bit == "1":
                    _set_pixel(pixels, cursor_x + column, start_y + row, color)
        cursor_x += 6


def generate_eye_rgb(eye: str) -> bytes:
    """Return canonical RGB8 pixels for one explicitly named synthetic eye."""

    if eye not in OUTPUT_PATHS:
        raise ValueError("eye must be od or os")

    pixels = bytearray(WIDTH * HEIGHT * CHANNELS)
    retina_x = 64
    retina_y = 57
    retina_radius = 52
    retina_radius_squared = retina_radius * retina_radius
    eye_texture_offset = 0 if eye == "od" else 7

    # A deterministic radial red/orange field.  The small modular texture makes
    # the illustration visibly synthetic while giving the quality gate useful
    # brightness and contrast structure.
    for y in range(HEIGHT):
        for x in range(WIDTH):
            delta_x = x - retina_x
            delta_y = y - retina_y
            distance_squared = delta_x * delta_x + delta_y * delta_y
            if distance_squared > retina_radius_squared:
                continue
            vignette = (distance_squared * 72) // retina_radius_squared
            texture = ((x * 17 + y * 29 + eye_texture_offset) % 13) - 6
            red = max(0, min(255, 184 - vignette + texture))
            green = max(0, min(255, 70 - vignette // 3 + texture // 2))
            blue = max(0, min(255, 42 - vignette // 4))
            _set_pixel(pixels, x, y, (red, green, blue))

    # Anatomically inspired but deliberately schematic macula, optic disc and
    # mirrored vessel trees.  No source photograph contributes any pixel.
    macula_x = 82 if eye == "od" else 45
    _draw_filled_circle(pixels, macula_x, 58, 8, (82, 20, 24))
    _draw_filled_circle(pixels, macula_x, 58, 3, (48, 10, 16))

    disc_x = 42 if eye == "od" else 85
    _draw_filled_circle(pixels, disc_x, 57, 9, (238, 177, 92))
    _draw_filled_circle(pixels, disc_x, 57, 5, (250, 207, 126))

    vessel_paths = (
        ((42, 57), (55, 45), (74, 35), (98, 27)),
        ((42, 57), (58, 55), (81, 51), (106, 43)),
        ((42, 57), (59, 65), (79, 76), (100, 91)),
        ((42, 57), (52, 75), (65, 91), (79, 103)),
        ((42, 57), (36, 41), (31, 27), (27, 15)),
    )
    for path in vessel_paths:
        _draw_polyline(
            pixels,
            _mirror_points(path, eye),
            (91, 16, 24),
            width=3,
        )

    branch_paths = (
        ((64, 40), (72, 28), (79, 18)),
        ((76, 52), (88, 59), (101, 65)),
        ((73, 73), (82, 86), (88, 98)),
        ((57, 82), (48, 93), (42, 101)),
    )
    for path in branch_paths:
        _draw_polyline(
            pixels,
            _mirror_points(path, eye),
            (112, 23, 30),
            width=1,
        )

    _draw_filled_circle(pixels, disc_x, 57, 2, (255, 226, 153))

    label = f"SYNTHETIC {eye.upper()}"
    label_width = len(label) * 6 - 1
    _draw_text(
        pixels,
        label,
        (WIDTH - label_width) // 2,
        118,
        (255, 235, 64),
    )
    return bytes(pixels)


def _read_bounded_regular_file(path: Path) -> bytes:
    try:
        if path.is_symlink():
            raise FixtureError("合成图像不能是符号链接")
        file_stat = path.stat()
        if not stat.S_ISREG(file_stat.st_mode):
            raise FixtureError("合成图像必须是普通文件")
        if file_stat.st_size > MAX_PNG_BYTES:
            raise FixtureError("PNG 文件超过允许大小")
        with path.open("rb") as handle:
            contents = handle.read(MAX_PNG_BYTES + 1)
    except FileNotFoundError:
        raise FixtureError("合成图像文件不存在") from None
    except OSError:
        raise FixtureError("无法读取合成图像文件") from None
    if len(contents) > MAX_PNG_BYTES:
        raise FixtureError("PNG 文件超过允许大小")
    return contents


def _decode_checked(png_bytes: bytes, eye: str):
    try:
        decoded = decode_synthetic_png(png_bytes, expected_eye=eye)
    except ValueError as exc:
        raise FixtureError(str(exc)) from None
    if decoded.width != WIDTH or decoded.height != HEIGHT:
        raise FixtureError("PNG 尺寸必须为 128x128")
    if decoded.eye != eye:
        raise FixtureError(f"{eye.upper()} 文件的眼别验证失败")
    return decoded


def _canonical_png(eye: str, rgb: bytes) -> bytes:
    png_bytes = encode_canonical_png(WIDTH, HEIGHT, rgb)
    if _decode_checked(png_bytes, eye).rgb != rgb:
        raise FixtureError("规范 PNG 编码器未能保持 RGB 像素")
    return png_bytes


def write_fixtures() -> None:
    try:
        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        resolved_root = PROJECT_ROOT.resolve(strict=True)
        resolved_output = OUTPUT_DIRECTORY.resolve(strict=True)
        resolved_output.relative_to(resolved_root)
        current = PROJECT_ROOT
        for part in OUTPUT_DIRECTORY.relative_to(PROJECT_ROOT).parts:
            current /= part
            if current.is_symlink():
                raise FixtureError("合成图像输出目录不能包含符号链接")
        for eye, path in OUTPUT_PATHS.items():
            if path.is_symlink():
                raise FixtureError("合成图像输出不能是符号链接")
            if path.exists() and not stat.S_ISREG(path.lstat().st_mode):
                raise FixtureError("合成图像输出必须是普通文件")
            contents = _canonical_png(eye, generate_eye_rgb(eye))
            descriptor, temporary_name = tempfile.mkstemp(
                dir=OUTPUT_DIRECTORY, prefix=f".{eye}.", suffix=".tmp"
            )
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(contents)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_name, path)
            except BaseException:
                try:
                    Path(temporary_name).unlink(missing_ok=True)
                except OSError:
                    pass
                raise
            print(
                f"已生成全合成 {eye.upper()} 图像："
                f"{path.relative_to(PROJECT_ROOT)}"
            )
    except FixtureError:
        raise
    except (OSError, ValueError):
        raise FixtureError("无法写入合成图像文件") from None


def check_fixtures() -> None:
    expected = {eye: generate_eye_rgb(eye) for eye in OUTPUT_PATHS}
    if expected["od"] == expected["os"]:
        raise FixtureError("OD 与 OS 合成图像必须可区分")

    for eye, path in OUTPUT_PATHS.items():
        decoded = _decode_checked(_read_bounded_regular_file(path), eye).rgb
        opposite_eye = "os" if eye == "od" else "od"
        if decoded == expected[opposite_eye]:
            raise FixtureError(f"{eye.upper()} 文件包含错误眼别图像")
        if decoded != expected[eye]:
            raise FixtureError(f"{eye.upper()} 图像像素与确定性生成结果不一致")
        print(f"已验证全合成 {eye.upper()} 图像：RGB、眼别和 PNG chunk 均符合规范")


def main() -> int:
    _configure_utf8_output()
    parser = argparse.ArgumentParser(
        description="生成或校验固定的 128x128 RGB 全合成 fundus-like PNG。"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只校验现有 OD/OS 文件，不写入任何文件。",
    )
    args = parser.parse_args()

    try:
        if args.check:
            check_fixtures()
        else:
            write_fixtures()
    except FixtureError as exc:
        print(f"合成图像操作失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
