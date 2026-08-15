from scripts.check_public_artifacts import policy_violations


def test_policy_accepts_public_code_docs_and_synthetic_json():
    assert policy_violations(
        [
            "src/longieye/research.py",
            "docs/RESEARCH_MODEL_CARD_TEMPLATE.md",
            "configs/demo_model.json",
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
    ]

    assert policy_violations(paths) == sorted(paths)
