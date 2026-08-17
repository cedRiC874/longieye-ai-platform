from __future__ import annotations

import hashlib
import itertools
import math
from pathlib import Path

import pytest

from longieye.features import extract_features
from longieye.fusion import (
    CANONICAL_ENCODER_ID,
    CANONICAL_ENCODER_TRAINING_DATA,
    EMBEDDING_ORDER,
    IMAGE_EMBEDDING_CONTRACT_VERSION,
    DeterministicFundusEncoder,
    EyeFusionResult,
    FusionContractError,
    ImageEncoder,
    MultimodalFusionResult,
    StructuredAnchoredFusionAdapter,
)
from longieye.imaging import (
    ImageArtifactError,
    PreparedImage,
    RasterImage,
    decode_synthetic_png,
)
from longieye.model import DemoRiskModel
from longieye.model_contract import AdapterReadiness, RiskModelBackend
from longieye.service import RiskPredictionService
from scripts.generate_synthetic_fundus import OUTPUT_PATHS, generate_eye_rgb
from scripts.run_demo import load_case
from scripts.run_multimodal_demo import load_fixture_images


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def model_and_features():
    model = DemoRiskModel.from_path(PROJECT_ROOT / "configs" / "demo_model.json")
    case, _ = load_case(PROJECT_ROOT / "examples" / "request.json")
    return model, extract_features(case)


def valid_descriptor():
    return {
        "encoder_id": CANONICAL_ENCODER_ID,
        "model_stage": "synthetic_reference_encoder",
        "embedding_contract_version": IMAGE_EMBEDDING_CONTRACT_VERSION,
        "embedding_order": EMBEDDING_ORDER,
        "clinical_use": False,
        "training_data": CANONICAL_ENCODER_TRAINING_DATA,
    }


def test_reference_encoder_is_deterministic_finite_and_contract_shaped():
    image = load_fixture_images("both")["od"]
    from longieye.imaging import ImageQualityGate

    prepared = ImageQualityGate().prepare(image)
    encoder = DeterministicFundusEncoder()

    first = encoder.encode(prepared)
    second = encoder.encode(prepared)

    assert isinstance(encoder, ImageEncoder)
    assert first == second
    assert len(first) == len(EMBEDDING_ORDER)
    assert all(math.isfinite(value) for value in first)
    assert encoder.readiness().status == "ready"


def test_no_image_path_is_bit_exact_structured_fallback():
    model, features = model_and_features()
    anchor = model.predict(features)
    adapter = StructuredAnchoredFusionAdapter(model)

    result = adapter.predict_with_images(features, None)

    assert isinstance(adapter, RiskModelBackend)
    assert adapter.predict(features) == anchor
    assert result.mode == "structured_fallback"
    assert result.od.demo_score == anchor["od"]
    assert result.os.demo_score == anchor["os"]
    assert result.od.reason_code == "image_missing"
    assert result.os.reason_code == "image_missing"


def test_two_canonical_images_enable_bounded_synthetic_fusion():
    model, features = model_and_features()
    adapter = StructuredAnchoredFusionAdapter(model)

    result = adapter.predict_with_images(features, load_fixture_images("both"))
    payload = result.as_dict()

    assert result.mode == "multimodal"
    for eye_result in (result.od, result.os):
        assert eye_result.mode == "tabular_plus_synthetic_image"
        assert eye_result.reason_code is None
        assert 0.0 <= eye_result.demo_score <= 1.0
        assert abs(eye_result.logit_adjustment) <= 0.35
    assert payload["clinical_use"] is False
    assert "未经训练或临床验证" in payload["disclaimer"]
    rendered = str(payload).lower()
    for forbidden in ("sha256", "pixel", "embedding", "path", "filename"):
        assert forbidden not in rendered


def test_missing_one_eye_falls_back_only_for_that_eye():
    model, features = model_and_features()
    anchor = model.predict(features)

    result = StructuredAnchoredFusionAdapter(model).predict_with_images(
        features, load_fixture_images("missing-os")
    )

    assert result.mode == "partial_fallback"
    assert result.od.mode == "tabular_plus_synthetic_image"
    assert result.os.mode == "tabular_only"
    assert result.os.demo_score == anchor["os"]
    assert result.os.structured_anchor_score == anchor["os"]
    assert result.os.reason_code == "image_missing"


def test_quality_rejection_preserves_the_exact_eye_anchor():
    class RejectingGate:
        def prepare(self, image):
            raise ImageArtifactError("image_quality_rejected")

    model, features = model_and_features()
    anchor = model.predict(features)
    adapter = StructuredAnchoredFusionAdapter(model, quality_gate=RejectingGate())

    result = adapter.predict_with_images(features, load_fixture_images("both"))

    assert result.mode == "structured_fallback"
    assert result.od.demo_score == anchor["od"]
    assert result.os.demo_score == anchor["os"]
    assert result.od.reason_code == "image_quality_rejected"
    assert result.os.reason_code == "image_quality_rejected"


def test_encoder_contract_error_locks_image_branch_not_structured_anchor():
    class BrokenEncoder:
        descriptor = valid_descriptor()

        def readiness(self):
            return AdapterReadiness(status="ready", self_test="passed")

        def encode(self, image):
            return (math.nan,) * len(EMBEDDING_ORDER)

    model, features = model_and_features()
    anchor = model.predict(features)
    adapter = StructuredAnchoredFusionAdapter(model, image_encoder=BrokenEncoder())

    result = adapter.predict_with_images(features, load_fixture_images("both"))

    assert result.mode == "structured_fallback"
    assert result.od.demo_score == anchor["od"]
    assert result.os.demo_score == anchor["os"]
    assert result.od.reason_code == "image_encoder_contract_error"
    assert result.os.reason_code == "image_encoder_contract_error"
    assert adapter.readiness().status == "not_ready"


@pytest.mark.parametrize("failing_eye", ["od", "os"])
def test_one_eye_encoder_failure_is_order_independent_for_current_request(
    failing_eye
):
    class EyeSpecificBrokenEncoder:
        descriptor = valid_descriptor()

        def readiness(self):
            return AdapterReadiness(status="ready", self_test="passed")

        def encode(self, image):
            if image.eye == failing_eye:
                raise OSError("C:/private/person-name.png")
            return (0.2,) * len(EMBEDDING_ORDER)

    model, features = model_and_features()
    anchor = model.predict(features)
    adapter = StructuredAnchoredFusionAdapter(
        model, image_encoder=EyeSpecificBrokenEncoder()
    )

    result = adapter.predict_with_images(features, load_fixture_images("both"))

    failed = getattr(result, failing_eye)
    succeeded = getattr(result, "os" if failing_eye == "od" else "od")
    assert result.mode == "partial_fallback"
    assert failed.mode == "tabular_only"
    assert failed.demo_score == anchor[failing_eye]
    assert failed.reason_code == "image_encoder_contract_error"
    assert succeeded.mode == "tabular_plus_synthetic_image"
    assert adapter.readiness().error_code == "image_encoder_contract_error"
    assert "person-name" not in str(result.as_dict())


def test_infinite_encoder_output_is_bounded_and_locks_safely():
    class InfiniteEncoder:
        descriptor = valid_descriptor()

        def readiness(self):
            return AdapterReadiness(status="ready", self_test="passed")

        def encode(self, image):
            return itertools.repeat(0.2)

    model, features = model_and_features()
    adapter = StructuredAnchoredFusionAdapter(model, image_encoder=InfiniteEncoder())

    result = adapter.predict_with_images(features, load_fixture_images("both"))

    assert result.mode == "structured_fallback"
    assert adapter.readiness().error_code == "image_encoder_contract_error"


def test_encoder_unavailable_is_explicit_and_exactly_falls_back():
    class UnavailableEncoder:
        descriptor = valid_descriptor()

        def readiness(self):
            return AdapterReadiness(
                status="not_ready",
                self_test="not_run",
                error_code="fixture_unavailable",
            )

        def encode(self, image):
            raise AssertionError("must not be called")

    model, features = model_and_features()
    anchor = model.predict(features)
    adapter = StructuredAnchoredFusionAdapter(
        model, image_encoder=UnavailableEncoder()
    )

    result = adapter.predict_with_images(features, load_fixture_images("both"))

    assert result.mode == "structured_fallback"
    assert result.od.demo_score == anchor["od"]
    assert result.os.demo_score == anchor["os"]
    assert result.od.reason_code == "image_encoder_unavailable"
    assert result.os.reason_code == "image_encoder_unavailable"


def test_encoder_readiness_exception_is_safely_normalized():
    class LeakingReadinessEncoder:
        descriptor = valid_descriptor()

        def readiness(self):
            raise OSError("C:/private/person-name.png")

        def encode(self, image):
            raise AssertionError("must not be called")

    model, features = model_and_features()
    adapter = StructuredAnchoredFusionAdapter(
        model, image_encoder=LeakingReadinessEncoder()
    )

    result = adapter.predict_with_images(features, load_fixture_images("both"))
    readiness = adapter.readiness()

    assert result.mode == "structured_fallback"
    assert readiness.error_code == "image_encoder_contract_error"
    assert "person-name" not in str(result.as_dict())


def test_provenance_is_bound_to_eye_registry_not_caller_boolean():
    model, features = model_and_features()
    od_bytes = OUTPUT_PATHS["od"].read_bytes()
    od_pixels = generate_eye_rgb("od")
    forged_os = decode_synthetic_png(
        od_bytes,
        expected_eye="os",
        expected_pixel_sha256=hashlib.sha256(od_pixels).hexdigest(),
    )
    assert forged_os.provenance_verified is True

    with pytest.raises(FusionContractError) as exc_info:
        StructuredAnchoredFusionAdapter(model).predict_with_images(
            features, {"os": forged_os}
        )
    assert exc_info.value.code == "image_provenance_invalid"


def test_wrong_container_duplicate_or_swapped_eye_fails_closed():
    model, features = model_and_features()
    adapter = StructuredAnchoredFusionAdapter(model)
    images = load_fixture_images("both")

    invalid_inputs = [
        [],
        {"right": images["od"]},
        {"od": images["os"]},
        {"od": images["od"], "os": images["od"]},
    ]
    for invalid in invalid_inputs:
        with pytest.raises(FusionContractError) as exc_info:
            adapter.predict_with_images(features, invalid)
        assert exc_info.value.code == "image_provenance_invalid"


def test_raster_subclass_or_post_construction_mutation_cannot_change_pixels():
    images = load_fixture_images("both")

    class SwappableRaster(RasterImage):
        def __getattribute__(self, name):
            if name == "rgb":
                try:
                    active = object.__getattribute__(self, "active")
                except AttributeError:
                    active = False
                if active:
                    return images["os"].rgb
            return super().__getattribute__(name)

    forged = SwappableRaster(
        eye=images["od"].eye,
        width=images["od"].width,
        height=images["od"].height,
        rgb=images["od"].rgb,
        encoded_sha256=images["od"].encoded_sha256,
        pixel_sha256=images["od"].pixel_sha256,
        provenance_verified=True,
    )
    object.__setattr__(forged, "active", True)

    mutated = load_fixture_images("both")["od"]
    object.__setattr__(mutated, "rgb", images["os"].rgb)

    model, features = model_and_features()
    adapter = StructuredAnchoredFusionAdapter(model)
    for invalid in ({"od": forged}, {"od": mutated}):
        with pytest.raises(FusionContractError) as exc_info:
            adapter.predict_with_images(features, invalid)
        assert exc_info.value.code == "image_provenance_invalid"


def test_bad_mapping_value_and_quality_contract_errors_are_safe():
    class LeakingMapping(dict):
        def __getitem__(self, key):
            raise OSError("C:/private/person-name.png")

    class InvalidGate:
        def prepare(self, image):
            raise ImageArtifactError("image_integrity_mismatch")

    model, features = model_and_features()
    adapter = StructuredAnchoredFusionAdapter(model)
    for invalid in ({"od": object()}, LeakingMapping(od=object())):
        with pytest.raises(FusionContractError) as exc_info:
            adapter.predict_with_images(features, invalid)
        assert exc_info.value.code == "image_provenance_invalid"
        assert exc_info.value.__cause__ is None
        assert "person-name" not in str(exc_info.value)

    adapter = StructuredAnchoredFusionAdapter(model, quality_gate=InvalidGate())
    with pytest.raises(FusionContractError) as exc_info:
        adapter.predict_with_images(features, load_fixture_images("both"))
    assert exc_info.value.code == "fusion_contract_invalid"
    assert exc_info.value.__cause__ is None


def test_falsey_injected_encoder_and_gate_are_not_silently_replaced():
    class FalseyEncoder:
        descriptor = valid_descriptor()

        def __init__(self):
            self.called = 0

        def __bool__(self):
            return False

        def readiness(self):
            return AdapterReadiness(status="ready", self_test="passed")

        def encode(self, image):
            self.called += 1
            return (0.2,) * len(EMBEDDING_ORDER)

    class FalseyGate:
        def __init__(self):
            from longieye.imaging import ImageQualityGate

            self.delegate = ImageQualityGate()
            self.called = 0

        def __bool__(self):
            return False

        def prepare(self, image):
            self.called += 1
            return self.delegate.prepare(image)

    model, features = model_and_features()
    encoder = FalseyEncoder()
    gate = FalseyGate()
    result = StructuredAnchoredFusionAdapter(
        model, image_encoder=encoder, quality_gate=gate
    ).predict_with_images(features, load_fixture_images("both"))

    assert result.mode == "multimodal"
    assert encoder.called == 2
    assert gate.called == 2


def test_injected_gate_cannot_swap_or_substitute_prepared_eye_pixels():
    from dataclasses import replace

    from longieye.imaging import ImageQualityGate

    images = load_fixture_images("both")
    os_prepared = ImageQualityGate().prepare(images["os"])

    class SwappingGate:
        def prepare(self, image):
            return os_prepared

    class SubstitutingGate:
        def prepare(self, image):
            return replace(
                os_prepared,
                eye=image.eye,
                source_pixel_sha256=image.pixel_sha256,
            )

    class EqualityBypassPrepared(PreparedImage):
        def __eq__(self, other):
            return True

    class EqualityBypassGate:
        def prepare(self, image):
            return EqualityBypassPrepared(
                eye=image.eye,
                side=os_prepared.side,
                pixels=os_prepared.pixels,
                quality=os_prepared.quality,
                source_pixel_sha256=image.pixel_sha256,
            )

    model, features = model_and_features()
    exact_instance_gate = ImageQualityGate()
    exact_instance_gate.prepare = SubstitutingGate().prepare

    for gate in (
        SwappingGate(),
        SubstitutingGate(),
        exact_instance_gate,
        EqualityBypassGate(),
    ):
        with pytest.raises(FusionContractError) as exc_info:
            StructuredAnchoredFusionAdapter(
                model, quality_gate=gate
            ).predict_with_images(features, images)
        assert exc_info.value.code == "image_provenance_invalid"


def test_public_service_refuses_multimodal_adapter_stage():
    model, _ = model_and_features()
    adapter = StructuredAnchoredFusionAdapter(model)

    with pytest.raises(ValueError, match="demo_synthetic"):
        RiskPredictionService(adapter)


def test_feature_validation_and_adjustment_type_errors_are_safely_normalized():
    class LeakingFeatures(dict):
        def __iter__(self):
            raise OSError("C:/private/person-name.png")

    class LeakingFloat:
        def __float__(self):
            raise OSError("C:/private/person-name.png")

    model, _ = model_and_features()
    adapter = StructuredAnchoredFusionAdapter(model)

    with pytest.raises(FusionContractError) as exc_info:
        adapter.predict(LeakingFeatures())
    assert exc_info.value.code == "fusion_contract_invalid"
    assert exc_info.value.__cause__ is None
    assert "person-name" not in str(exc_info.value)

    with pytest.raises(FusionContractError) as adjustment_error:
        StructuredAnchoredFusionAdapter(
            model, max_logit_adjustment=LeakingFloat()
        )
    assert adjustment_error.value.code == "fusion_contract_invalid"
    assert "person-name" not in str(adjustment_error.value)


def test_eye_result_rejects_impossible_active_fusion_state():
    with pytest.raises(ValueError, match="active fusion state"):
        EyeFusionResult(
            demo_score=0.0,
            structured_anchor_score=1.0,
            mode="tabular_plus_synthetic_image",
            reason_code=None,
            logit_adjustment=0.5,
        )
    with pytest.raises(ValueError, match="active fusion state"):
        EyeFusionResult(
            demo_score=0.5,
            structured_anchor_score=0.5,
            mode="tabular_plus_synthetic_image",
            reason_code=None,
            logit_adjustment=999.0,
        )


def test_multimodal_result_rejects_noncanonical_eye_result_types():
    class LeakingEyeResult:
        mode = "tabular_only"

        def as_dict(self):
            return {"path": "C:/private/person-name.png"}

    with pytest.raises(ValueError, match="invalid type"):
        MultimodalFusionResult(
            mode="structured_fallback",
            od=LeakingEyeResult(),
            os=LeakingEyeResult(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("encoder_id", "self-declared-other-encoder"),
        ("training_data", "trained_on_unapproved_images"),
    ],
)
def test_encoder_descriptor_cannot_relabel_another_backend(field, value):
    class MislabelledEncoder:
        descriptor = valid_descriptor() | {field: value}

        def readiness(self):
            return AdapterReadiness(status="ready", self_test="passed")

        def encode(self, image):
            return (0.2,) * len(EMBEDDING_ORDER)

    model, _ = model_and_features()
    with pytest.raises(FusionContractError) as exc_info:
        StructuredAnchoredFusionAdapter(model, image_encoder=MislabelledEncoder())
    assert exc_info.value.code == "fusion_contract_invalid"


def test_infinite_feature_key_iterator_is_bounded_and_safe():
    class InfiniteFeatures(dict):
        def __iter__(self):
            return itertools.repeat("sex_y1")

    model, _ = model_and_features()
    with pytest.raises(FusionContractError) as exc_info:
        StructuredAnchoredFusionAdapter(model).predict(InfiniteFeatures())
    assert exc_info.value.code == "fusion_contract_invalid"


def test_structured_backend_readiness_is_rechecked_after_construction():
    class MutableBackend:
        metadata = {
            "model_id": "mutable-synthetic-anchor",
            "model_stage": "demo_synthetic",
            "training_data": "test-only",
            "clinical_use": False,
        }

        def __init__(self):
            self.ready = True

        def readiness(self):
            return AdapterReadiness(
                status="ready" if self.ready else "not_ready",
                self_test="passed" if self.ready else "failed",
                error_code=None if self.ready else "C:/private/person-name.png",
            )

        def predict(self, features):
            return {"od": 0.2, "os": 0.3}

    backend = MutableBackend()
    adapter = StructuredAnchoredFusionAdapter(backend)
    backend.ready = False

    readiness = adapter.readiness()
    assert readiness.status == "not_ready"
    assert readiness.error_code == "structured_backend_unavailable"
    assert "person-name" not in str(readiness)
    with pytest.raises(FusionContractError) as exc_info:
        adapter.predict({name: 0.0 for name in model_and_features()[1]})
    assert exc_info.value.code == "fusion_contract_invalid"
