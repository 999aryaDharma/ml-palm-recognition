"""
Smoke Test Gate (Phase M2)

Tujuan: Validasi kritis sebelum invest waktu untuk fine-tuning ArcFace.

Pertanyaan: Apakah MobileFaceNet pretrained (dilatih di wajah MS-Celeb-1M)
bisa menghasilkan embedding yang membedakan telapak tangan?

Metric: Separation gap = mean(positive_similarity) - mean(negative_similarity)
- positive: pair gambar dari individu sama
- negative: pair dari individu berbeda

Decision Gate:
  gap > 0.2  → PATH C (Fine-tune ArcFace di Minggu 3) — direkomendasikan
  0.1 ≤ gap ≤ 0.2  → Fine-tune dengan data tambahan, atau adjust LR
  gap < 0.1  → PATH B (Deploy pretrained as-is, ~85% accuracy expected)

Output:
- ml/artifacts/smoke_test_histogram.png — visualisasi distribusi
- ml/docs/decision_gate_report.md — rekomendasi tertulis
- print summary ke stdout
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    RAW_DIR,
    PROCESSED_DIR,
    ARTIFACTS_DIR,
    PROJECT_ROOT,
    PRETRAINED_PATH,
    SMOKE_TEST_GAP_THRESHOLD_GOOD,
    SMOKE_TEST_GAP_THRESHOLD_MIN,
    SMOKE_TEST_SAMPLE_INDIVIDUALS,
    SMOKE_TEST_IMAGES_PER_INDIVIDUAL,
)
from mobilefacenet import load_mobilefacenet
from data_loader import get_val_transform
from extract_landmarks import LandmarkExtractor
from extract_roi import extract_palm_roi


def compute_embedding_for_image(img_path, landmark_extractor, model, transform, device):
    """End-to-end: image → landmark → ROI → embedding. None kalau gagal."""
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return None

    lm_result = landmark_extractor.extract(img)
    if lm_result is None:
        return None

    roi = extract_palm_roi(img, lm_result["landmarks"])
    if roi is None:
        return None

    img_np = np.array(roi)
    try:
        tensor = transform(image=img_np)["image"]
    except (TypeError, KeyError):
        tensor = transform(roi)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(tensor)
        emb = F.normalize(emb, dim=1)
    return emb.squeeze(0).cpu().numpy()


def run_smoke_test(
    data_dir: Path,
    pretrained_path: Path,
    n_individuals: int = SMOKE_TEST_SAMPLE_INDIVIDUALS,
    n_images: int = SMOKE_TEST_IMAGES_PER_INDIVIDUAL,
    use_processed: bool = False,
):
    """
    Args:
        data_dir: folder dengan struktur individual_XXX/(images or left|right/)
        pretrained_path: path ke MobileFaceNet pretrained weights
        n_individuals: jumlah individu yang di-sample
        n_images: jumlah gambar per individu
        use_processed: kalau True, asumsikan dataset sudah ROI-extracted
                       (skip landmark/ROI step)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_mobilefacenet(
        str(pretrained_path) if pretrained_path.exists() else None,
        strict=False,
    ).to(device)
    model.eval()

    if not pretrained_path.exists():
        print(f"⚠️  Pretrained tidak ditemukan: {pretrained_path}")
        print(f"   Smoke test akan menggunakan RANDOM-INIT model (akan gagal!).")
        print(f"   Download pretrained dulu, mis. dari face.evoLVe atau Hugging Face.")

    transform = get_val_transform()

    if not use_processed:
        landmark_extractor = LandmarkExtractor()
    else:
        landmark_extractor = None

    # List individu
    data_dir = Path(data_dir)
    individuals = sorted([d for d in data_dir.iterdir() if d.is_dir()])[:n_individuals]
    if len(individuals) == 0:
        raise RuntimeError(f"Tidak ada folder individu di {data_dir}")
    print(f"Sampling {len(individuals)} individu dari {data_dir}")

    # Extract embedding per individu
    all_embeddings = {}  # {ind_name: [emb, ...]}

    for ind in individuals:
        # Cari semua gambar (rekursif untuk handle left/right subfolder)
        images = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            images.extend(ind.rglob(ext))
        images = sorted(images)[:n_images]
        if not images:
            print(f"  [skip] {ind.name}: tidak ada gambar")
            continue

        embeddings = []
        for img_path in images:
            if use_processed:
                # Skip landmark/ROI, langsung ke embedding
                img = Image.open(img_path).convert("RGB")
                img_np = np.array(img)
                try:
                    tensor = transform(image=img_np)["image"]
                except (TypeError, KeyError):
                    tensor = transform(img)
                tensor = tensor.unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(tensor)
                    emb = F.normalize(emb, dim=1)
                emb = emb.squeeze(0).cpu().numpy()
            else:
                emb = compute_embedding_for_image(
                    img_path, landmark_extractor, model, transform, device
                )

            if emb is not None:
                embeddings.append(emb)

        if len(embeddings) < 2:
            print(f"  [skip] {ind.name}: hanya {len(embeddings)} embedding valid")
            continue

        all_embeddings[ind.name] = embeddings
        print(f"  {ind.name}: {len(embeddings)} embedding")

    if len(all_embeddings) < 2:
        raise RuntimeError("Butuh minimal 2 individu untuk smoke test")

    # Hitung positive pairs (same individual)
    positive_sims = []
    for ind_name, embs in all_embeddings.items():
        for i, j in combinations(range(len(embs)), 2):
            sim = float(np.dot(embs[i], embs[j]))
            positive_sims.append(sim)

    # Hitung negative pairs (different individuals, sampling untuk efisiensi)
    negative_sims = []
    ind_names = list(all_embeddings.keys())
    for i in range(len(ind_names)):
        for j in range(i + 1, len(ind_names)):
            embs_i = all_embeddings[ind_names[i]][:3]
            embs_j = all_embeddings[ind_names[j]][:3]
            for ei in embs_i:
                for ej in embs_j:
                    sim = float(np.dot(ei, ej))
                    negative_sims.append(sim)

    if landmark_extractor is not None:
        landmark_extractor.close()

    # Statistics
    pos_mean = float(np.mean(positive_sims))
    neg_mean = float(np.mean(negative_sims))
    gap = pos_mean - neg_mean

    print("\n" + "=" * 60)
    print("=== SMOKE TEST RESULTS ===")
    print(f"Positive pairs:  {len(positive_sims)}  (same individual)")
    print(f"Negative pairs:  {len(negative_sims)}  (different individuals)")
    print(f"Mean pos sim:    {pos_mean:.4f}")
    print(f"Mean neg sim:    {neg_mean:.4f}")
    print(f"Separation gap:  {gap:.4f}")
    print()

    if gap > SMOKE_TEST_GAP_THRESHOLD_GOOD:
        decision = "PATH C"
        message = "✅ Gap > 0.2 → PATH C: Fine-tune ArcFace di Minggu 3."
        recommendation = "Lanjut ke fine-tuning. Expected hasil EER < 1% setelah training."
    elif gap > SMOKE_TEST_GAP_THRESHOLD_MIN:
        decision = "PATH C (cautious)"
        message = "⚠️  Gap 0.1–0.2 → Fine-tune masih recommended tapi butuh perhatian."
        recommendation = (
            "Pertimbangkan: data augmentation lebih agresif, lebih banyak epoch, "
            "atau pakai pretrained dari domain lebih dekat (mis. fingerprint)."
        )
    else:
        decision = "PATH B"
        message = "❌ Gap < 0.1 → PATH B: Deploy pretrained, accept ~85% accuracy."
        recommendation = (
            "Pretrained tidak transfer well ke palm. Coba: verify ROI extraction "
            "quality, atau pertimbangkan training from scratch dengan data lebih banyak."
        )

    print(message)
    print(f"Rekomendasi: {recommendation}")
    print("=" * 60)

    return {
        "positive_sims": positive_sims,
        "negative_sims": negative_sims,
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
        "gap": gap,
        "decision": decision,
        "message": message,
        "recommendation": recommendation,
        "n_individuals": len(all_embeddings),
    }


def save_histogram(results, output_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(
        results["positive_sims"], bins=30, alpha=0.7, color="green",
        label=f"Positive (same)\nMean: {results['pos_mean']:.3f}", density=True,
    )
    plt.hist(
        results["negative_sims"], bins=30, alpha=0.7, color="red",
        label=f"Negative (diff)\nMean: {results['neg_mean']:.3f}", density=True,
    )
    plt.axvline(x=results["pos_mean"], color="darkgreen", linestyle="--", lw=1)
    plt.axvline(x=results["neg_mean"], color="darkred", linestyle="--", lw=1)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.title(f"Smoke Test: Pretrained MobileFaceNet on Palmprint\nGap = {results['gap']:.4f} → {results['decision']}")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Histogram saved: {output_path}")


def save_decision_report(results, output_path: Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = f"""# Decision Gate Report — Smoke Test (Phase M2)

## Setup
- Pretrained model: MobileFaceNet (MS-Celeb-1M)
- Sample: {results['n_individuals']} individu
- Positive pairs: {len(results['positive_sims'])}
- Negative pairs: {len(results['negative_sims'])}

## Results
| Metric | Value |
|---|---|
| Mean positive similarity (same palm) | {results['pos_mean']:.4f} |
| Mean negative similarity (diff palm) | {results['neg_mean']:.4f} |
| **Separation gap** | **{results['gap']:.4f}** |

## Decision: {results['decision']}

{results['message']}

**Rekomendasi:** {results['recommendation']}

## Threshold Reference
- gap > 0.2 → PATH C (fine-tune, recommended)
- 0.1 ≤ gap ≤ 0.2 → PATH C cautious (extra care)
- gap < 0.1 → PATH B (deploy pretrained, accept lower accuracy)

## Next Steps
{'Lanjut ke training Phase 1 (softmax warm-up) lalu Phase 2 (ArcFace fine-tune).' if 'PATH C' in results['decision'] else 'Skip training, langsung export pretrained ke palm_recognizer.pt dan dokumentasikan limitasi di laporan.'}
"""
    output_path.write_text(report)
    print(f"Report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Smoke test gate — decide Path B vs C")
    parser.add_argument("--data-dir", type=str, default=str(RAW_DIR))
    parser.add_argument("--pretrained", type=str, default=str(PRETRAINED_PATH))
    parser.add_argument("--use-processed", action="store_true",
                        help="Dataset sudah di-ROI-extract, skip MediaPipe step")
    parser.add_argument("--n-individuals", type=int, default=SMOKE_TEST_SAMPLE_INDIVIDUALS)
    parser.add_argument("--n-images", type=int, default=SMOKE_TEST_IMAGES_PER_INDIVIDUAL)
    parser.add_argument("--histogram-out", type=str,
                        default=str(ARTIFACTS_DIR / "smoke_test_histogram.png"))
    parser.add_argument("--report-out", type=str,
                        default=str(PROJECT_ROOT / "docs" / "decision_gate_report.md"))
    args = parser.parse_args()

    if args.use_processed:
        data_dir = Path(args.data_dir)
        if data_dir == Path(RAW_DIR):
            data_dir = PROCESSED_DIR
    else:
        data_dir = Path(args.data_dir)

    results = run_smoke_test(
        data_dir=data_dir,
        pretrained_path=Path(args.pretrained),
        n_individuals=args.n_individuals,
        n_images=args.n_images,
        use_processed=args.use_processed,
    )

    save_histogram(results, args.histogram_out)
    save_decision_report(results, args.report_out)


if __name__ == "__main__":
    main()
