"""Synthetic-only multimodal fusion kept outside the public HTTP service."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Protocol, runtime_checkable

from .features import FEATURE_ORDER
from .imaging import (
    CANONICAL_FIXTURE_PIXEL_SHA256,
    ImageArtifactError,
    ImageQualityGate,
    PreparedImage,
    RasterImage,
)
from .model_contract import (
    AdapterReadiness,
    RiskModelBackend,
    validated_feature_values,
    validated_scores,
)


EYE_ORDER = ("od", "os")
IMAGE_EMBEDDING_CONTRACT_VERSION = "synthetic-fundus-statistics-v1"
FUSION_CONTRACT_VERSION = "structured-anchor-logit-residual-v1"
CANONICAL_ENCODER_ID = "longieye-synthetic-fundus-statistics-v1"
CANONICAL_ENCODER_TRAINING_DATA = "none_procedural_statistics_only"
EMBEDDING_ORDER = (
    "mean_intensity",
    "intensity_contrast",
    "edge_energy",
    "center_periphery_delta",
    "left_right_asymmetry",
)
MULTIMODAL_DISCLAIMER = (
    "仅用于全合成多模态工程演示；图像编码器未经训练或临床验证，"
    "不可用于诊断、筛查或治疗决策。"
)
FALLBACK_REASON_CODES = {
    "image_missing",
    "image_quality_rejected",
    "image_encoder_unavailable",
    "image_encoder_contract_error",
}


_SAFE_MESSAGES = {
    "fusion_contract_invalid": "多模态融合合同无效。",
    "image_provenance_invalid": "合成图像的眼别或来源绑定无效。",
}


class FusionContractError(ValueError):
    """A stable, non-echoing multimodal boundary error."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(_SAFE_MESSAGES.get(code, "多模态融合失败。"))


def _adjusted_score(anchor_score: float, adjustment: float) -> float:
    bounded_anchor = min(1.0 - 1e-12, max(1e-12, anchor_score))
    anchor_logit = math.log(bounded_anchor / (1.0 - bounded_anchor))
    return 1.0 / (1.0 + math.exp(-(anchor_logit + adjustment)))


@runtime_checkable
class ImageEncoder(Protocol):
    """Minimal interface for an already-initialized image encoder."""

    @property
    def descriptor(self) -> Mapping[str, object]: ...

    def encode(self, image: PreparedImage) -> Sequence[float]: ...

    def readiness(self) -> AdapterReadiness: ...


class DeterministicFundusEncoder:
    """Inspectable statistics encoder for synthetic fixtures, not a learned model."""

    _descriptor = {
        "encoder_id": CANONICAL_ENCODER_ID,
        "model_stage": "synthetic_reference_encoder",
        "embedding_contract_version": IMAGE_EMBEDDING_CONTRACT_VERSION,
        "embedding_order": EMBEDDING_ORDER,
        "clinical_use": False,
        "training_data": CANONICAL_ENCODER_TRAINING_DATA,
    }

    @property
    def descriptor(self) -> Mapping[str, object]:
        return dict(self._descriptor)

    def readiness(self) -> AdapterReadiness:
        return AdapterReadiness(status="ready", self_test="passed")

    def encode(self, image: PreparedImage) -> tuple[float, ...]:
        if image.side != 32 or len(image.pixels) != image.side * image.side * 3:
            raise FusionContractError("fusion_contract_invalid")
        rgb = tuple(float(value) for value in image.pixels)
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in rgb):
            raise FusionContractError("fusion_contract_invalid")
        pixels = tuple(
            (77.0 * rgb[index] + 150.0 * rgb[index + 1] + 29.0 * rgb[index + 2])
            / 256.0
            for index in range(0, len(rgb), 3)
        )

        side = image.side
        mean = sum(pixels) / len(pixels)
        contrast = math.sqrt(
            sum((value - mean) ** 2 for value in pixels) / len(pixels)
        )

        edge_total = 0.0
        edge_count = 0
        for row in range(side):
            offset = row * side
            for column in range(side):
                value = pixels[offset + column]
                if column + 1 < side:
                    edge_total += abs(value - pixels[offset + column + 1])
                    edge_count += 1
                if row + 1 < side:
                    edge_total += abs(value - pixels[offset + side + column])
                    edge_count += 1
        edge_energy = edge_total / edge_count

        center_values: list[float] = []
        periphery_values: list[float] = []
        left_values: list[float] = []
        right_values: list[float] = []
        center = (side - 1) / 2.0
        center_radius_squared = (side / 4.0) ** 2
        for row in range(side):
            for column in range(side):
                value = pixels[row * side + column]
                radius_squared = (row - center) ** 2 + (column - center) ** 2
                if radius_squared <= center_radius_squared:
                    center_values.append(value)
                elif radius_squared >= (side * 0.40) ** 2:
                    periphery_values.append(value)
                if column < side // 2:
                    left_values.append(value)
                else:
                    right_values.append(value)

        center_periphery_delta = (
            sum(center_values) / len(center_values)
            - sum(periphery_values) / len(periphery_values)
        )
        left_right_asymmetry = (
            sum(left_values) / len(left_values)
            - sum(right_values) / len(right_values)
        )
        embedding = (
            mean,
            contrast,
            edge_energy,
            center_periphery_delta,
            left_right_asymmetry,
        )
        if not all(math.isfinite(value) for value in embedding):
            raise FusionContractError("fusion_contract_invalid")
        return embedding


@dataclass(frozen=True)
class EyeFusionResult:
    """One eye's explicit branch decision and synthetic score."""

    demo_score: float
    structured_anchor_score: float
    mode: str
    reason_code: str | None
    logit_adjustment: float

    def __post_init__(self) -> None:
        if self.mode not in {"tabular_only", "tabular_plus_synthetic_image"}:
            raise ValueError("unsupported eye fusion mode")
        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in (self.demo_score, self.structured_anchor_score)
        ):
            raise ValueError("fusion scores must be finite and bounded")
        if not math.isfinite(self.logit_adjustment):
            raise ValueError("logit adjustment must be finite")
        if self.mode == "tabular_only" and self.demo_score != self.structured_anchor_score:
            raise ValueError("fallback must exactly preserve the structured score")
        if self.mode == "tabular_only" and (
            self.reason_code not in FALLBACK_REASON_CODES
            or self.logit_adjustment != 0.0
        ):
            raise ValueError("fallback state is inconsistent")
        if self.mode == "tabular_plus_synthetic_image" and self.reason_code is not None:
            raise ValueError("an active image branch cannot have a fallback reason")
        if self.mode == "tabular_plus_synthetic_image" and (
            abs(self.logit_adjustment) > 0.5
            or not math.isclose(
                self.demo_score,
                _adjusted_score(
                    self.structured_anchor_score, self.logit_adjustment
                ),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("active fusion state is inconsistent")

    def as_dict(self) -> dict[str, object]:
        return {
            "demo_score": round(self.demo_score, 6),
            "structured_anchor_score": round(self.structured_anchor_score, 6),
            "mode": self.mode,
            "reason_code": self.reason_code,
            "logit_adjustment": round(self.logit_adjustment, 6),
        }


@dataclass(frozen=True)
class MultimodalFusionResult:
    """Bilateral offline result with no paths, pixels, embeddings or identifiers."""

    mode: str
    od: EyeFusionResult
    os: EyeFusionResult

    def __post_init__(self) -> None:
        if type(self.od) is not EyeFusionResult or type(self.os) is not EyeFusionResult:
            raise ValueError("multimodal eye results have an invalid type")
        if self.mode not in {
            "multimodal",
            "partial_fallback",
            "structured_fallback",
        }:
            raise ValueError("unsupported multimodal result mode")
        active_count = sum(
            result.mode == "tabular_plus_synthetic_image"
            for result in (self.od, self.os)
        )
        expected_mode = (
            "multimodal"
            if active_count == 2
            else "partial_fallback"
            if active_count == 1
            else "structured_fallback"
        )
        if self.mode != expected_mode:
            raise ValueError("multimodal result mode is inconsistent")

    def as_dict(self) -> dict[str, object]:
        return {
            "model_stage": "demo_multimodal_synthetic",
            "mode": self.mode,
            "predictions": {"od": self.od.as_dict(), "os": self.os.as_dict()},
            "clinical_use": False,
            "disclaimer": MULTIMODAL_DISCLAIMER,
        }


class StructuredAnchoredFusionAdapter:
    """Add a bounded synthetic image residual while preserving exact fallback."""

    _head_weights = {
        "od": (-0.35, 0.80, 0.65, 0.45, 0.30),
        "os": (-0.30, 0.75, 0.70, 0.40, -0.30),
    }

    def __init__(
        self,
        structured_backend: RiskModelBackend,
        image_encoder: ImageEncoder | None = None,
        quality_gate: ImageQualityGate | None = None,
        *,
        max_logit_adjustment: float = 0.35,
    ) -> None:
        try:
            structured_metadata = structured_backend.metadata
            structured_readiness = structured_backend.readiness()
            if (
                not isinstance(structured_metadata, Mapping)
                or not isinstance(structured_readiness, AdapterReadiness)
                or structured_metadata.get("model_stage") != "demo_synthetic"
                or structured_metadata.get("clinical_use") is not False
                or structured_readiness.status != "ready"
            ):
                raise FusionContractError("fusion_contract_invalid")
            structured_model_id = structured_metadata.get("model_id")
            if not isinstance(structured_model_id, str) or not structured_model_id:
                raise FusionContractError("fusion_contract_invalid")
        except FusionContractError:
            raise
        except Exception:
            raise FusionContractError("fusion_contract_invalid") from None

        if type(max_logit_adjustment) not in {int, float}:
            raise FusionContractError("fusion_contract_invalid")
        adjustment_limit = float(max_logit_adjustment)
        if not math.isfinite(adjustment_limit) or not 0.0 < adjustment_limit <= 0.5:
            raise FusionContractError("fusion_contract_invalid")

        encoder = (
            DeterministicFundusEncoder() if image_encoder is None else image_encoder
        )
        try:
            descriptor = encoder.descriptor
            if not isinstance(descriptor, Mapping):
                raise FusionContractError("fusion_contract_invalid")
            encoder_id = descriptor.get("encoder_id")
            raw_embedding_order = descriptor.get("embedding_order")
            if isinstance(raw_embedding_order, (str, bytes)):
                raise FusionContractError("fusion_contract_invalid")
            embedding_order = tuple(
                islice(iter(raw_embedding_order), len(EMBEDDING_ORDER) + 1)
            )
            if (
                encoder_id != CANONICAL_ENCODER_ID
                or descriptor.get("training_data")
                != CANONICAL_ENCODER_TRAINING_DATA
                or descriptor.get("model_stage") != "synthetic_reference_encoder"
                or descriptor.get("clinical_use") is not False
                or descriptor.get("embedding_contract_version")
                != IMAGE_EMBEDDING_CONTRACT_VERSION
                or embedding_order != EMBEDDING_ORDER
            ):
                raise FusionContractError("fusion_contract_invalid")
        except FusionContractError:
            raise
        except Exception:
            raise FusionContractError("fusion_contract_invalid") from None

        self._structured_backend = structured_backend
        self._image_encoder = encoder
        self._quality_gate = ImageQualityGate() if quality_gate is None else quality_gate
        self._max_logit_adjustment = adjustment_limit
        self._image_error_code: str | None = None
        self._metadata = {
            "model_id": "longieye-synthetic-multimodal-fusion-v0",
            "model_stage": "demo_multimodal_synthetic",
            "training_data": "none_procedural_images_and_synthetic_anchor",
            "clinical_use": False,
            "fusion_contract_version": FUSION_CONTRACT_VERSION,
            "image_embedding_contract_version": IMAGE_EMBEDDING_CONTRACT_VERSION,
            "structured_anchor_model_id": structured_model_id,
        }

    @property
    def metadata(self) -> Mapping[str, object]:
        return dict(self._metadata)

    def _encoder_readiness(self) -> AdapterReadiness:
        if self._image_error_code is not None:
            return AdapterReadiness(
                status="not_ready",
                self_test="failed",
                error_code=self._image_error_code,
            )
        try:
            readiness = self._image_encoder.readiness()
            if not isinstance(readiness, AdapterReadiness):
                raise TypeError("invalid readiness")
            return readiness
        except Exception:
            self._image_error_code = "image_encoder_contract_error"
            return AdapterReadiness(
                status="not_ready",
                self_test="failed",
                error_code=self._image_error_code,
            )

    def _structured_readiness(self) -> AdapterReadiness:
        try:
            metadata = self._structured_backend.metadata
            readiness = self._structured_backend.readiness()
            if (
                not isinstance(metadata, Mapping)
                or not isinstance(readiness, AdapterReadiness)
                or metadata.get("model_stage") != "demo_synthetic"
                or metadata.get("clinical_use") is not False
            ):
                raise TypeError("invalid structured backend state")
            return readiness
        except Exception:
            return AdapterReadiness(
                status="not_ready",
                self_test="failed",
                error_code="structured_backend_unavailable",
            )

    def readiness(self) -> AdapterReadiness:
        structured_readiness = self._structured_readiness()
        if structured_readiness.status != "ready":
            return AdapterReadiness(
                status="not_ready",
                self_test=structured_readiness.self_test,
                error_code="structured_backend_unavailable",
            )
        encoder_readiness = self._encoder_readiness()
        if encoder_readiness.status != "ready":
            return AdapterReadiness(
                status="not_ready",
                self_test=encoder_readiness.self_test,
                error_code=(
                    self._image_error_code or "image_encoder_unavailable"
                ),
            )
        return AdapterReadiness(status="ready", self_test="passed")

    def predict(self, features: Mapping[str, float]) -> dict[str, float]:
        """Return the exact structured anchor when no image evidence is supplied."""

        try:
            if self._structured_readiness().status != "ready":
                raise FusionContractError("fusion_contract_invalid")
            if not isinstance(features, Mapping):
                raise FusionContractError("fusion_contract_invalid")
            keys = tuple(islice(iter(features), len(FEATURE_ORDER) + 1))
            if len(keys) != len(FEATURE_ORDER) or set(keys) != set(FEATURE_ORDER):
                raise FusionContractError("fusion_contract_invalid")
            copied_features = {name: features[name] for name in FEATURE_ORDER}
            values = validated_feature_values(copied_features)
            normalized_features = dict(zip(FEATURE_ORDER, values, strict=True))
            scores = self._structured_backend.predict(normalized_features)
            return validated_scores(scores)
        except FusionContractError:
            raise
        except Exception:
            raise FusionContractError("fusion_contract_invalid") from None

    @staticmethod
    def _validated_images(
        images: Mapping[str, RasterImage] | None,
    ) -> dict[str, RasterImage]:
        if images is None:
            return {}
        if not isinstance(images, Mapping):
            raise FusionContractError("image_provenance_invalid")
        try:
            keys = tuple(islice(iter(images), len(EYE_ORDER) + 1))
            if len(keys) > len(EYE_ORDER) or len(set(keys)) != len(keys):
                raise FusionContractError("image_provenance_invalid")
            copied: dict[str, RasterImage] = {}
            for eye in keys:
                if eye not in EYE_ORDER:
                    raise FusionContractError("image_provenance_invalid")
                image = images[eye]
                if type(image) is not RasterImage:
                    raise FusionContractError("image_provenance_invalid")
                copied[eye] = RasterImage(
                    eye=image.eye,
                    width=image.width,
                    height=image.height,
                    rgb=image.rgb,
                    encoded_sha256=image.encoded_sha256,
                    pixel_sha256=image.pixel_sha256,
                    provenance_verified=image.provenance_verified,
                )
        except FusionContractError:
            raise
        except ImageArtifactError:
            raise FusionContractError("image_provenance_invalid") from None
        except Exception:
            raise FusionContractError("image_provenance_invalid") from None

        hashes: list[str] = []
        for eye, image in copied.items():
            if (
                image.eye != eye
                or not image.provenance_verified
                or image.pixel_sha256 != CANONICAL_FIXTURE_PIXEL_SHA256[eye]
            ):
                raise FusionContractError("image_provenance_invalid")
            hashes.append(image.pixel_sha256)
        if len(hashes) != len(set(hashes)):
            raise FusionContractError("image_provenance_invalid")
        return copied

    def predict_with_images(
        self,
        features: Mapping[str, float],
        images: Mapping[str, RasterImage] | None,
    ) -> MultimodalFusionResult:
        image_mapping = self._validated_images(images)
        anchor_scores = self.predict(features)
        encoder_readiness = self._encoder_readiness()
        encoder_reason = (
            None
            if encoder_readiness.status == "ready"
            else self._image_error_code or "image_encoder_unavailable"
        )

        results: dict[str, EyeFusionResult] = {}
        request_encoder_failed = False
        for eye in EYE_ORDER:
            results[eye], eye_encoder_failed = self._predict_eye(
                eye,
                anchor_scores[eye],
                image_mapping.get(eye),
                encoder_reason=encoder_reason,
            )
            request_encoder_failed = request_encoder_failed or eye_encoder_failed
        if request_encoder_failed:
            self._image_error_code = "image_encoder_contract_error"

        active_count = sum(
            result.mode == "tabular_plus_synthetic_image"
            for result in results.values()
        )
        mode = (
            "multimodal"
            if active_count == 2
            else "partial_fallback"
            if active_count == 1
            else "structured_fallback"
        )
        return MultimodalFusionResult(
            mode=mode,
            od=results["od"],
            os=results["os"],
        )

    def _predict_eye(
        self,
        eye: str,
        anchor_score: float,
        image: RasterImage | None,
        *,
        encoder_reason: str | None,
    ) -> tuple[EyeFusionResult, bool]:
        if image is None:
            return self._fallback(anchor_score, "image_missing"), False
        if encoder_reason is not None:
            return self._fallback(anchor_score, encoder_reason), False

        try:
            prepared = self._quality_gate.prepare(image)
        except ImageArtifactError as exc:
            if exc.code == "image_quality_rejected":
                return self._fallback(anchor_score, "image_quality_rejected"), False
            raise FusionContractError("fusion_contract_invalid") from None
        except Exception:
            raise FusionContractError("fusion_contract_invalid") from None
        if (
            type(prepared) is not PreparedImage
            or prepared.eye != eye
            or prepared.source_pixel_sha256 != image.pixel_sha256
        ):
            raise FusionContractError("image_provenance_invalid") from None
        try:
            canonical_prepared = ImageQualityGate().prepare(image)
        except Exception:
            raise FusionContractError("fusion_contract_invalid") from None
        if prepared != canonical_prepared:
            raise FusionContractError("image_provenance_invalid") from None
        prepared = canonical_prepared

        try:
            raw_embedding = self._image_encoder.encode(prepared)
            iterator = iter(raw_embedding)
            sampled = tuple(islice(iterator, len(EMBEDDING_ORDER) + 1))
            if len(sampled) != len(EMBEDDING_ORDER):
                raise ValueError("invalid embedding length")
            embedding: list[float] = []
            for index, raw_value in enumerate(sampled):
                if isinstance(raw_value, bool):
                    raise ValueError("invalid embedding value")
                value = float(raw_value)
                lower, upper = ((0.0, 1.0) if index < 3 else (-1.0, 1.0))
                if not math.isfinite(value) or not lower <= value <= upper:
                    raise ValueError("invalid embedding value")
                embedding.append(value)
        except Exception:
            return (
                self._fallback(anchor_score, "image_encoder_contract_error"),
                True,
            )

        raw_adjustment = sum(
            weight * value
            for weight, value in zip(
                self._head_weights[eye], embedding, strict=True
            )
        )
        adjustment = self._max_logit_adjustment * math.tanh(raw_adjustment)
        fused_score = _adjusted_score(anchor_score, adjustment)
        return (
            EyeFusionResult(
                demo_score=fused_score,
                structured_anchor_score=anchor_score,
                mode="tabular_plus_synthetic_image",
                reason_code=None,
                logit_adjustment=adjustment,
            ),
            False,
        )

    @staticmethod
    def _fallback(anchor_score: float, reason_code: str) -> EyeFusionResult:
        return EyeFusionResult(
            demo_score=anchor_score,
            structured_anchor_score=anchor_score,
            mode="tabular_only",
            reason_code=reason_code,
            logit_adjustment=0.0,
        )
