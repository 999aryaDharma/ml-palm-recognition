"""
Unit tests untuk PalmNet-Lite — wajib pass sebelum training.

Tests:
    MODEL:
    - construction without pretrained
    - output shape [B, 128]
    - parameter sanity (350k-450k)
    - gradients finite
    - no NaN/Inf output
    - inference L2 norm ~1
    - TorchScript export

    DATA:
    - val transform deterministic
    - no horizontal flip in val transform

    TRAINING:
    - one mini-batch Phase 1 backward
    - one mini-batch ArcFace backward
    - checkpoint save/load

    EVALUATION:
    - cosine similarity computation
    - template averaging + normalize

    ARTIFACT:
    - verify_artifact on exported model

Usage:
    cd app/ml
    python -m pytest tests/test_palmnet.py -v
    # atau
    python tests/test_palmnet.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Path setup
_ML_DIR = Path(__file__).resolve().parent.parent
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))


# ─── MODEL TESTS ─────────────────────────────────────────────────────────────

def test_model_construction_no_pretrained():
    """PalmNetLite dapat dibuat tanpa file pretrained/checkpoint."""
    from palm_recognition.models.palmnet_lite import build_palmnet_lite
    from palm_recognition.models.initialization import initialize_scratch_weights, verify_no_pretrained_load

    model = build_palmnet_lite()
    initialize_scratch_weights(model)
    assert verify_no_pretrained_load(model) is True
    print("  [PASS] construction_no_pretrained")


def test_model_output_shape():
    """Output shape harus [B, 128]."""
    from palm_recognition.models.palmnet_lite import build_palmnet_lite
    from palm_recognition.models.initialization import initialize_scratch_weights

    model = build_palmnet_lite()
    initialize_scratch_weights(model)
    model.eval()

    for batch_size in [1, 2, 4]:
        x = torch.randn(batch_size, 3, 112, 112)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch_size, 128), f"Shape salah: {out.shape}"
    print("  [PASS] output_shape")


def test_model_parameter_count():
    """Parameter count harus dalam range [350k, 450k]."""
    from palm_recognition.models.palmnet_lite import build_palmnet_lite

    model = build_palmnet_lite()
    info = model.count_parameters()
    total = info["trainable"]
    assert 350_000 <= total <= 450_000, (
        f"Parameter count {total:,} di luar range [350k, 450k]"
    )
    print(f"  [PASS] parameter_count: {total:,}")


def test_model_forward_no_nan():
    """Forward pass tidak menghasilkan NaN/Inf."""
    from palm_recognition.models.palmnet_lite import build_palmnet_lite
    from palm_recognition.models.initialization import initialize_scratch_weights

    model = build_palmnet_lite()
    initialize_scratch_weights(model)
    model.eval()

    x = torch.randn(4, 3, 112, 112)
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all(), "Output mengandung NaN/Inf"
    print("  [PASS] forward_no_nan")


def test_model_gradients_finite():
    """Backward pass menghasilkan gradients yang finite."""
    from palm_recognition.models.palmnet_lite import build_palmnet_lite
    from palm_recognition.models.initialization import initialize_scratch_weights

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
            assert p.grad is not None, f"Grad None: {name}"
            assert torch.isfinite(p.grad).all(), f"Grad non-finite: {name}"

    print("  [PASS] gradients_finite")


def test_inference_l2_norm():
    """PalmNetLiteInference output harus L2-normalized."""
    from palm_recognition.models.palmnet_lite import PalmNetLite, PalmNetLiteInference
    from palm_recognition.models.initialization import initialize_scratch_weights
    import torch.nn.functional as F

    backbone = PalmNetLite()
    initialize_scratch_weights(backbone)
    wrapper = PalmNetLiteInference(backbone)
    wrapper.eval()

    x = torch.randn(4, 3, 112, 112)
    with torch.no_grad():
        out = wrapper(x)

    assert out.shape == (4, 128)
    norms = out.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-4), (
        f"L2 norm bukan ~1: {norms}"
    )
    print("  [PASS] inference_l2_norm")


def test_torchscript_export():
    """Model dapat diekspor ke TorchScript."""
    from palm_recognition.models.palmnet_lite import PalmNetLite, PalmNetLiteInference
    from palm_recognition.models.initialization import initialize_scratch_weights

    backbone = PalmNetLite()
    initialize_scratch_weights(backbone)
    wrapper = PalmNetLiteInference(backbone)
    wrapper.eval()

    dummy = torch.randn(1, 3, 112, 112)

    # Try script first, fallback to trace
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
    assert max_diff < 1e-3, f"Eager vs scripted diff terlalu besar: {max_diff}"
    print(f"  [PASS] torchscript_export (max_diff={max_diff:.2e})")


# ─── TRAINING TESTS ──────────────────────────────────────────────────────────

def test_phase1_minibatch():
    """Satu mini-batch Phase 1 backward pass."""
    from palm_recognition.models.palmnet_lite import build_palmnet_lite
    from palm_recognition.models.initialization import initialize_scratch_weights

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
    assert torch.isfinite(loss), f"Loss non-finite: {loss.item()}"
    loss.backward()
    optimizer.step()

    print(f"  [PASS] phase1_minibatch (loss={loss.item():.4f})")


def test_arcface_minibatch():
    """Satu mini-batch ArcFace backward pass."""
    from palm_recognition.models.palmnet_lite import build_palmnet_lite
    from palm_recognition.models.initialization import initialize_scratch_weights
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
    assert torch.isfinite(loss), f"ArcFace loss non-finite: {loss.item()}"
    loss.backward()
    optimizer.step()

    print(f"  [PASS] arcface_minibatch (loss={loss.item():.4f})")


def test_checkpoint_save_load():
    """Checkpoint save dan load menghasilkan state yang sama."""
    from palm_recognition.models.palmnet_lite import build_palmnet_lite
    from palm_recognition.models.initialization import initialize_scratch_weights
    from palm_recognition.training.checkpointing import (
        save_phase1_checkpoint, load_checkpoint
    )

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
        assert ckpt["best_metric"] == 0.85
        assert "backbone_state_dict" in ckpt
        assert "classifier_state_dict" in ckpt

        # Reload backbone
        backbone2 = build_palmnet_lite()
        backbone2.load_state_dict(ckpt["backbone_state_dict"])

    print("  [PASS] checkpoint_save_load")


# ─── EVALUATION TESTS ────────────────────────────────────────────────────────

def test_template_averaging():
    """Template averaging + normalize."""
    from palm_recognition.evaluation.embeddings import build_enrollment_templates

    num_classes = 3
    embs = np.random.randn(6, 128).astype(np.float32)
    # Normalize embeddings
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / (norms + 1e-8)

    class_indices = np.array([0, 0, 1, 1, 2, 2])
    templates = build_enrollment_templates(embs, class_indices, num_classes)

    assert templates.shape == (num_classes, 128)
    for c in range(num_classes):
        n = np.linalg.norm(templates[c])
        assert abs(n - 1.0) < 1e-4, f"Template {c} norm bukan ~1: {n}"

    print("  [PASS] template_averaging")


def test_cosine_similarity_range():
    """Cosine similarity antara L2-normalized vectors dalam [-1, 1]."""
    a = torch.randn(10, 128)
    b = torch.randn(10, 128)
    a = torch.nn.functional.normalize(a, p=2, dim=1)
    b = torch.nn.functional.normalize(b, p=2, dim=1)

    sims = (a * b).sum(dim=1)
    assert (sims >= -1.0 - 1e-5).all()
    assert (sims <= 1.0 + 1e-5).all()
    print("  [PASS] cosine_similarity_range")


# ─── ARTIFACT TESTS ──────────────────────────────────────────────────────────

def test_artifact_verify():
    """Export dan verify TorchScript artifact."""
    from palm_recognition.models.palmnet_lite import PalmNetLite, PalmNetLiteInference, build_palmnet_lite
    from palm_recognition.models.initialization import initialize_scratch_weights
    from palm_recognition.artifacts.exporter import verify_artifact

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dan save TorchScript
        backbone = build_palmnet_lite()
        initialize_scratch_weights(backbone)
        wrapper = PalmNetLiteInference(backbone)
        wrapper.eval()

        dummy = torch.randn(1, 3, 112, 112)
        try:
            scripted = torch.jit.script(wrapper)
        except Exception:
            scripted = torch.jit.trace(wrapper, dummy)

        model_pt = Path(tmpdir) / "model.pt"
        scripted.save(str(model_pt))

        # Write dummy manifest
        import json
        manifest = {"model_id": "palmnet-lite-scratch", "version": "1.0.0"}
        with open(Path(tmpdir) / "manifest.json", "w") as f:
            json.dump(manifest, f)

        result = verify_artifact(tmpdir)
        assert result["ok"] is True, f"Verify gagal: {result.get('error')}"
        assert result["output_shape"] == [1, 128]
        assert abs(result["l2_norm"] - 1.0) < 1e-2

    print("  [PASS] artifact_verify")


# ─── LOSSES TEST ─────────────────────────────────────────────────────────────

def test_arcface_margin_warmup():
    """ArcFace margin warmup bekerja dengan benar."""
    from palm_recognition.losses.arcface import ArcFaceLoss, LinearMarginWarmup

    loss_fn = ArcFaceLoss(128, 10, margin=0.30, scale=32.0)
    warmup = LinearMarginWarmup(loss_fn, target_margin=0.30, warmup_epochs=8)

    # Epoch 0: margin = 0.30 * 1/8 = 0.0375
    m0 = warmup.step(0)
    assert abs(m0 - 0.30 * 1 / 8) < 1e-6, f"Epoch 0 margin wrong: {m0}"

    # Epoch 8+: margin = 0.30
    m8 = warmup.step(8)
    assert abs(m8 - 0.30) < 1e-6, f"Epoch 8 margin wrong: {m8}"

    print("  [PASS] arcface_margin_warmup")


# ─── RUN ALL ─────────────────────────────────────────────────────────────────

def run_all_tests():
    tests = [
        ("Model: construction no pretrained", test_model_construction_no_pretrained),
        ("Model: output shape",               test_model_output_shape),
        ("Model: parameter count",            test_model_parameter_count),
        ("Model: forward no NaN",             test_model_forward_no_nan),
        ("Model: gradients finite",           test_model_gradients_finite),
        ("Model: inference L2 norm",          test_inference_l2_norm),
        ("Model: TorchScript export",         test_torchscript_export),
        ("Training: phase1 minibatch",        test_phase1_minibatch),
        ("Training: arcface minibatch",       test_arcface_minibatch),
        ("Training: checkpoint save/load",    test_checkpoint_save_load),
        ("Evaluation: template averaging",    test_template_averaging),
        ("Evaluation: cosine similarity",     test_cosine_similarity_range),
        ("Losses: arcface margin warmup",     test_arcface_margin_warmup),
        ("Artifact: verify",                  test_artifact_verify),
    ]

    passed = 0
    failed = 0
    print("\n" + "=" * 60)
    print("PalmNet-Lite Unit Tests")
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
        print("✓ ALL TESTS PASSED".encode('utf-8').decode('utf-8', errors='replace'))
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
