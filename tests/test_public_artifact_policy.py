import pytest
import subprocess
import sys
from pathlib import Path

from scripts.check_public_artifacts import (
    content_looks_like_denied_image,
    policy_violations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_policy_accepts_public_code_docs_and_synthetic_json():
    assert policy_violations(
        [
            "src/longieye/research.py",
            "docs/RESEARCH_MODEL_CARD_TEMPLATE.md",
            "configs/demo_model.json",
            "examples/synthetic_fundus/od.png",
            "examples/synthetic_fundus/os.png",
        ]
    ) == []


def test_policy_rejects_research_data_checkpoints_and_oof_outputs():
    paths = [
        "data/participants.json",
        "models/research.pt",
        "reports/fold_oof.json",
        "artifacts/private/manifest.json",
        "exports/predictions.parquet",
        "exports/participants.json",
        "models/preprocessing.json",
        "configs/research_manifest.json",
        ".env.production",
        "secrets/client.pem",
        "secrets/client.p12",
        "examples/real_fundus.png",
        "examples/private.jpg",
        "incoming/scan.jpeg",
        "incoming/scan.tiff",
        "incoming/scan.webp",
        "incoming/scan.dcm",
        "incoming/scan.nii",
        "incoming/scan.nii.gz",
        "incoming/scan.ppm",
        "payload/scan.jp2",
        "payload/scan.avif",
        "payload/scan.heic",
        "payload/scan.raw",
        "examples/real_patient.dat",
        "docs/real_patient.svg",
    ]

    assert policy_violations(paths) == sorted(paths)


def test_synthetic_image_and_vector_allowlists_are_case_sensitive():
    paths = [
        "examples/synthetic_fundus/OD.png",
        "examples/SYNTHETIC_fundus/os.png",
        "docs/assets/Architecture.svg",
    ]
    assert policy_violations(paths) == sorted(paths)


@pytest.mark.parametrize(
    "payload",
    [
        b"\x89PNG\r\n\x1a\nrenamed",
        b"\xff\xd8\xff\xe0renamed",
        b"II*\x00renamed",
        b"x" * 128 + b"DICM" + b"renamed",
        b"\x00\x00\x00\x0cjP  \r\n\x87\nrenamed",
        b"\x00\x00\x00\x18ftypavifrenamed",
        b"  <?xml version='1.0'?><svg viewBox='0 0 1 1'></svg>",
        b"%PDF-1.7 renamed container",
        b"PK\x03\x04renamed container",
        b"\x00" * 257 + b"ustar\x00renamed container",
    ],
    ids=(
        "png",
        "jpeg",
        "tiff",
        "dicom",
        "jpeg2000",
        "avif",
        "svg",
        "pdf",
        "zip",
        "tar",
    ),
)
def test_policy_detects_images_renamed_to_an_unrelated_suffix(payload):
    assert content_looks_like_denied_image(payload) is True


def test_policy_magic_guard_does_not_treat_regular_source_as_an_image():
    assert content_looks_like_denied_image(b"from pathlib import Path\n") is False


def test_public_artifact_policy_cli_passes_for_the_candidate_tree():
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "check_public_artifacts.py")],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Public-artifact policy passed" in completed.stdout
