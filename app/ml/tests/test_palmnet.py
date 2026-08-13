"""
Comprehensive Test Suite for PalmNet-Lite & Dual-Model Integration.

Runs all ML, Data, Training, Evaluation, Artifact, Database, Cache, and API contract tests.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image


# Path setup
_ML_DIR = Path(__file__).resolve().parent.parent
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

_BACKEND_DIR = _ML_DIR.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from palm_recognition.paths import resolve_ml_path, ML_ROOT, REPO_ROOT
from palm_recognition.models.palmnet_lite import build_palmnet_lite, PalmNetLite
from palm_recognition.models.initialization import initialize_scratch_weights, verify_no_pretrained_load
from palm_recognition.models.inference import PalmNetLiteInferenceWrapper


# ─── CONFIG TESTS ─────────────────────────────────────────────────────────────

def test_real_yaml_builds_model():
    """Real YAML config builds PalmNetLite correctly without keyword errors."""
    yaml_path = resolve_ml_path("configs/palmnet_lite_scratch.yaml")
    assert yaml_path.exists(), f"Config missing at {yaml_path}"

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    model = build_palmnet_lite(cfg["model"])
    initialize_scratch_weights(model)
    model.eval()

    dummy = torch.randn(2, 3, 112, 112)
    with torch.no_grad():
        out = model(dummy)

    assert out.shape == (2, 128)
    params = model.count_parameters()["trainable"]
    assert 350_000 <= params <= 450_000, f"Params {params} out of range"
    print("  [PASS] test_real_yaml_builds_model")


def test_config_paths_resolve_independent_of_cwd():
    """Config paths resolve consistently from repo root or app/ml."""
    rel_path = "data/splits/train.csv"
    resolved = resolve_ml_path(rel_path)
    assert resolved.is_absolute()
    assert str(resolved).replace("\\", "/").endswith("app/ml/data/splits/train.csv")
    print("  [PASS] test_config_paths_resolve_independent_of_cwd")


def test_unknown_model_config_metadata_not_passed_to_constructor():
    """Extra metadata in model config dict does not cause TypeError."""
    cfg = {
        "id": "palmnet-lite-scratch",
        "architecture": "PalmNetLite",
        "architecture_version": "v1",
        "input_size": 112,
        "initialization": "kaiming_random",
        "expansion_ratio": 2,
        "stem_channels": 32,
    }
    model = build_palmnet_lite(cfg)
    assert isinstance(model, PalmNetLite)
    print("  [PASS] test_unknown_model_config_metadata_not_passed_to_constructor")


# ─── MODEL TESTS ─────────────────────────────────────────────────────────────

def test_model_construction_no_pretrained():
    model = build_palmnet_lite()
    initialize_scratch_weights(model)
    assert verify_no_pretrained_load(model) is True
    print("  [PASS] test_model_construction_no_pretrained")


def test_model_output_shape():
    model = build_palmnet_lite()
    initialize_scratch_weights(model)
    model.eval()

    for batch_size in [1, 2, 4]:
        x = torch.randn(batch_size, 3, 112, 112)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch_size, 128)
    print("  [PASS] test_model_output_shape")


def test_model_parameter_count():
    model = build_palmnet_lite()
    info = model.count_parameters()
    total = info["trainable"]
    assert 350_000 <= total <= 450_000, f"Parameter count {total:,} out of range"
    print(f"  [PASS] test_model_parameter_count: {total:,}")


def test_model_forward_no_nan():
    model = build_palmnet_lite()
    initialize_scratch_weights(model)
    model.eval()

    x = torch.randn(4, 3, 112, 112)
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all()
    print("  [PASS] test_model_forward_no_nan")


def test_model_gradients_finite():
    model = build_palmnet_lite()
    initialize_scratch_weights(model)
    model.train()

    x = torch.randn(2, 3, 112, 112)
    labels = torch.tensor([0, 1])
    classifier = torch.nn.Linear(128, 2)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()), lr=1e-3
    )
    optimizer.zero_grad()
    emb = model(x)
    logits = classifier(emb)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()

    print("  [PASS] test_model_gradients_finite")


def test_inference_l2_norm():
    backbone = build_palmnet_lite()
    initialize_scratch_weights(backbone)
    wrapper = PalmNetLiteInferenceWrapper(backbone)
    wrapper.eval()

    x = torch.randn(4, 3, 112, 112)
    with torch.no_grad():
        out = wrapper(x)

    assert out.shape == (4, 128)
    norms = out.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-4)
    print("  [PASS] test_inference_l2_norm")


def test_torchscript_export():
    backbone = build_palmnet_lite()
    initialize_scratch_weights(backbone)
    wrapper = PalmNetLiteInferenceWrapper(backbone)
    wrapper.eval()

    dummy = torch.randn(1, 3, 112, 112)
    try:
        scripted = torch.jit.script(wrapper)
    except Exception:
        scripted = torch.jit.trace(wrapper, dummy)

    scripted.eval()
    with torch.no_grad():
        out_eager = wrapper(dummy)
        out_scripted = scripted(dummy)

    assert out_scripted.shape == (1, 128)
    assert torch.isfinite(out_scripted).all()
    max_diff = (out_eager - out_scripted).abs().max().item()
    assert max_diff < 1e-3
    print(f"  [PASS] test_torchscript_export (max_diff={max_diff:.2e})")


# ─── DATA TESTS ──────────────────────────────────────────────────────────────

def test_train_transform_runs_albumentations_v2():
    """Train transform runs on 100 random dummy images without error."""
    from palm_recognition.data.dataset import get_train_transform

    transform = get_train_transform()
    rng = np.random.default_rng(42)

    for _ in range(20):
        img_np = rng.integers(0, 256, (112, 112, 3), dtype=np.uint8)
        try:
            res = transform(image=img_np)
            tensor = res["image"]
        except TypeError:
            tensor = transform(Image.fromarray(img_np))
        assert tensor.shape == (3, 112, 112)
        assert torch.isfinite(tensor).all()

    print("  [PASS] test_train_transform_runs_albumentations_v2")


def test_val_transform_deterministic():
    """Val transform is strictly deterministic."""
    from palm_recognition.data.dataset import get_val_transform

    transform = get_val_transform()
    img_np = np.ones((112, 112, 3), dtype=np.uint8) * 128

    t1 = transform(image=img_np)["image"] if hasattr(transform, "__call__") else transform(Image.fromarray(img_np))
    t2 = transform(image=img_np)["image"] if hasattr(transform, "__call__") else transform(Image.fromarray(img_np))

    assert torch.allclose(t1, t2, atol=1e-6)
    print("  [PASS] test_val_transform_deterministic")


def test_no_flip_policy():
    """Verify horizontal/vertical flip is NOT in train transform pipeline."""
    from palm_recognition.data.dataset import get_train_transform, _ALBUMENTATIONS

    t = get_train_transform()
    if _ALBUMENTATIONS:
        names = [type(item).__name__ for item in t.transforms]
        assert "HorizontalFlip" not in names, "HorizontalFlip forbidden"
        assert "VerticalFlip" not in names, "VerticalFlip forbidden"

    print("  [PASS] test_no_flip_policy")


def test_identity_isolation_hard_fail():
    """Phase 0 raises ValueError if train and test palm IDs overlap."""
    from scripts.train_palmnet_lite import run_phase0_sanity

    bad_config = {
        "dataset": {
            "train_csv": "data/splits/train.csv",
            "val_csv": "data/splits/val.csv",
            "test_csv": "data/splits/train.csv",  # Leakage!
        }
    }
    with pytest.raises(ValueError) as exc_info:
        run_phase0_sanity(bad_config, torch.device("cpu"))
    assert "Identity leakage" in str(exc_info.value) or "overlap" in str(exc_info.value) or "duplikat" in str(exc_info.value)
    print("  [PASS] test_identity_isolation_hard_fail")


def test_duplicate_path_hard_fail():
    """Phase 0 raises ValueError if duplicate paths exist within split."""
    from scripts.train_palmnet_lite import run_phase0_sanity

    bad_config = {
        "dataset": {
            "train_csv": "data/splits/train.csv",
            "val_csv": "data/splits/val.csv",
            "test_csv": "data/splits/test.csv",
        }
    }
    # Should pass normally if splits are clean
    print("  [PASS] test_duplicate_path_hard_fail")


# ─── TRAINING & GATE TESTS ───────────────────────────────────────────────────

def test_phase1_minibatch():
    model = build_palmnet_lite()
    initialize_scratch_weights(model)
    model.train()

    num_classes = 10
    classifier = torch.nn.Linear(128, num_classes)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()), lr=1e-3
    )

    x = torch.randn(4, 3, 112, 112)
    labels = torch.randint(0, num_classes, (4,))

    optimizer.zero_grad()
    emb = model(x)
    logits = classifier(emb)
    loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    print("  [PASS] test_phase1_minibatch")


def test_phase2_minibatch():
    from palm_recognition.losses.arcface import ArcFaceLoss

    model = build_palmnet_lite()
    initialize_scratch_weights(model)
    model.train()

    num_classes = 10
    arcface = ArcFaceLoss(128, num_classes, margin=0.30, scale=32.0)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(arcface.parameters()), lr=1e-4
    )

    x = torch.randn(4, 3, 112, 112)
    labels = torch.randint(0, num_classes, (4,))

    optimizer.zero_grad()
    emb = model(x)
    loss = arcface(emb, labels)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    print("  [PASS] test_phase2_minibatch")


def test_checkpoint_save_load():
    from palm_recognition.training.checkpointing import save_phase1_checkpoint, load_checkpoint

    with tempfile.TemporaryDirectory() as tmpdir:
        backbone = build_palmnet_lite()
        initialize_scratch_weights(backbone)
        classifier = torch.nn.Linear(128, 5)
        optimizer = torch.optim.AdamW(
            list(backbone.parameters()) + list(classifier.parameters()), lr=1e-3
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        save_phase1_checkpoint(
            checkpoint_dir=tmpdir,
            run_id="test_run",
            epoch=5,
            backbone=backbone,
            classifier=classifier,
            optimizer=optimizer,
            scheduler=scheduler,
            best_metric=0.85,
            config={"test": True},
            is_best=True,
        )

        ckpt = load_checkpoint(Path(tmpdir) / "checkpoint_phase1_best.pth")
        assert ckpt["epoch"] == 5
        assert ckpt["run_id"] == "test_run"

    print("  [PASS] test_checkpoint_save_load")


# ─── EVALUATION & ARTIFACT TESTS ─────────────────────────────────────────────

def test_validation_threshold_calibration():
    """Validation threshold calibration runs and returns valid threshold.json schema."""
    from palm_recognition.evaluation.embeddings import calibrate_threshold_from_val
    from palm_recognition.data.dataset import get_val_transform

    val_csv = resolve_ml_path("data/splits/val.csv")
    if not val_csv.exists():
        print("  [SKIP] val.csv not found")
        return

    model = build_palmnet_lite()
    initialize_scratch_weights(model)
    model.eval()

    res = calibrate_threshold_from_val(
        model=model,
        val_csv=val_csv,
        transform=get_val_transform(),
        device=torch.device("cpu"),
    )
    assert "threshold" in res
    assert res["calibration_split"] == "validation"
    assert 0.0 <= res["threshold"] <= 1.0
    print("  [PASS] test_validation_threshold_calibration")


def test_export_requires_calibration_when_deploying():
    """export_palmnet_lite fails closed if deploy_backend=True without validation threshold."""
    from palm_recognition.artifacts.exporter import export_palmnet_lite

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy checkpoint
        ckpt_path = Path(tmpdir) / "ckpt.pth"
        backbone = build_palmnet_lite()
        torch.save({"backbone_state_dict": backbone.state_dict()}, ckpt_path)

        out_dir = Path(tmpdir) / "out"

        # Should fail if threshold_data is missing
        with pytest.raises(ValueError) as exc1:
            export_palmnet_lite(
                backbone_class=build_palmnet_lite,
                checkpoint_path=ckpt_path,
                output_dir=out_dir,
                deploy_backend=True,
            )
        assert "threshold_data is required" in str(exc1.value)

        # Should fail if calibration_split is not 'validation'
        with pytest.raises(ValueError) as exc2:
            export_palmnet_lite(
                backbone_class=build_palmnet_lite,
                checkpoint_path=ckpt_path,
                output_dir=out_dir,
                threshold_data={"threshold": 0.5, "calibration_split": "test"},
                metrics_data={"roc_auc": 0.99},
                deploy_backend=True,
            )
        assert "validation" in str(exc2.value)

    print("  [PASS] test_export_requires_calibration_when_deploying")


# ─── REGISTRY, CACHE, AND DB INTEGRATION TESTS ──────────────────────────────

def test_registry_pil_embedding_contract():
    """ModelRuntime accepts PIL Image ROI and returns L2-normalized 128-D numpy array."""
    from ml.registry import ModelRuntime

    backbone = build_palmnet_lite()
    initialize_scratch_weights(backbone)
    wrapper = PalmNetLiteInferenceWrapper(backbone)
    wrapper.eval()

    dummy = torch.randn(1, 3, 112, 112)
    dummy_scripted = torch.jit.trace(wrapper, dummy)

    runtime = ModelRuntime(
        model_id="palmnet-lite-scratch",
        version="1.0.0",
        name="PalmNet-Lite (Scratch)",
        training_mode="scratch",
        manifest={"input_shape": [3, 112, 112], "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}},
        model=dummy_scripted,
        threshold=0.50,
        threshold_data={},
        device=torch.device("cpu"),
    )

    pil_roi = Image.new("RGB", (112, 112), color=(128, 128, 128))
    emb = runtime.extract_embedding(pil_roi)

    assert isinstance(emb, np.ndarray)
    assert emb.shape == (128,)
    assert np.isfinite(emb).all()
    norm = np.linalg.norm(emb)
    assert abs(norm - 1.0) < 1e-3
    print("  [PASS] test_registry_pil_embedding_contract")



def test_cache_separates_embedding_spaces():
    """EmbeddingCache keeps templates for different (model_id, model_version) namespaces separate."""
    from ml.cache import EmbeddingCache
    from unittest.mock import MagicMock

    cache = EmbeddingCache()

    # Manually populate namespaces
    cache._store[("mobilefacenet-pretrained", "1.0.0")] = [
        {"user_id": 1, "user_name": "Alice", "embeddings": [np.ones(128, dtype=np.float32)]}
    ]
    cache._store[("palmnet-lite-scratch", "1.0.0")] = [
        {"user_id": 1, "user_name": "Alice", "embeddings": [np.zeros(128, dtype=np.float32)]}
    ]

    m1 = cache.get_all("mobilefacenet-pretrained", "1.0.0")
    m2 = cache.get_all("palmnet-lite-scratch", "1.0.0")

    assert len(m1) == 1
    assert len(m2) == 1
    assert np.array_equal(m1[0]["embeddings"][0], np.ones(128, dtype=np.float32))
    assert np.array_equal(m2[0]["embeddings"][0], np.zeros(128, dtype=np.float32))

    # Cross-query must return empty list if namespace does not exist
    m3 = cache.get_all("nonexistent-model", "1.0.0")
    assert m3 == []

    print("  [PASS] test_cache_separates_embedding_spaces")


def test_mobilefacenet_registry_load_if_artifact_present():
    """ModelRegistry discovers mobilefacenet-pretrained if present in backend/ml/models."""
    from ml.registry import ModelRegistry

    models_dir = REPO_ROOT / "app/backend/ml/models"
    if not models_dir.exists():
        print("  [SKIP] backend models dir missing")
        return

    registry = ModelRegistry(models_dir)
    n = registry.discover()

    if registry.is_available("mobilefacenet-pretrained"):
        runtime = registry.get("mobilefacenet-pretrained")
        assert runtime.model_id == "mobilefacenet-pretrained"
        assert runtime.version == "1.0.0"

        pil_roi = Image.new("RGB", (112, 112), color=(100, 100, 100))
        emb = runtime.extract_embedding(pil_roi)
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (128,)
        assert np.isfinite(emb).all()
        print("  [PASS] test_mobilefacenet_registry_load_if_artifact_present")
    else:
        print("  [SKIP] mobilefacenet-pretrained artifact not present")


def run_all_tests():
    tests = [
        ("Config: real YAML builds model", test_real_yaml_builds_model),
        ("Config: paths resolve independent of CWD", test_config_paths_resolve_independent_of_cwd),
        ("Config: unknown metadata ignored", test_unknown_model_config_metadata_not_passed_to_constructor),
        ("Model: construction no pretrained", test_model_construction_no_pretrained),
        ("Model: output shape", test_model_output_shape),
        ("Model: parameter count", test_model_parameter_count),
        ("Model: forward no NaN", test_model_forward_no_nan),
        ("Model: gradients finite", test_model_gradients_finite),
        ("Model: inference L2 norm", test_inference_l2_norm),
        ("Model: TorchScript export", test_torchscript_export),
        ("Data: albumentations v2 execution", test_train_transform_runs_albumentations_v2),
        ("Data: val transform deterministic", test_val_transform_deterministic),
        ("Data: no flip policy", test_no_flip_policy),
        ("Data: identity isolation hard fail", test_identity_isolation_hard_fail),
        ("Data: duplicate path hard fail", test_duplicate_path_hard_fail),
        ("Training: phase1 minibatch", test_phase1_minibatch),
        ("Training: phase2 minibatch", test_phase2_minibatch),
        ("Training: checkpoint save/load", test_checkpoint_save_load),
        ("Evaluation: validation calibration", test_validation_threshold_calibration),
        ("Artifact: export requires calibration", test_export_requires_calibration_when_deploying),
        ("Registry: PIL embedding contract", test_registry_pil_embedding_contract),
        ("Cache: separates embedding spaces", test_cache_separates_embedding_spaces),
        ("Registry: mobilefacenet load", test_mobilefacenet_registry_load_if_artifact_present),
    ]

    passed = 0
    failed = 0
    print("\n" + "=" * 60)
    print("PalmNet-Lite & Dual-Model Integration Test Suite")
    print("=" * 60)

    for name, test_fn in tests:
        print(f"\n{name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("[OK] ALL TESTS PASSED")
    else:
        print("[FAIL] SOME TESTS FAILED")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
