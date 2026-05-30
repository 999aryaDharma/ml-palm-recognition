"""
Konfigurasi terpusat untuk pipeline ML Palm Biometric.

Semua hyperparameter, path, dan magic number HARUS didefinisikan di sini.
Jangan hardcode di script lain — import dari config.
"""
from pathlib import Path

# ============================================================================
# Path Configuration
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # ml/ (config.py ada di ml/scripts/)

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
LOGS_DIR = ARTIFACTS_DIR / "training_logs"

# Pretrained weights (akan didownload manual user)
PRETRAINED_PATH = MODELS_DIR / "mobilefacenet_pretrained.pth"

# MediaPipe Hand Landmarker task file
# Download: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
HAND_LANDMARKER_PATH = MODELS_DIR / "hand_landmarker.task"

# Output artifacts
RECOGNIZER_OUTPUT = ARTIFACTS_DIR / "palm_recognizer.pt"
THRESHOLD_OUTPUT = ARTIFACTS_DIR / "threshold.json"

# Backend copy targets (relative — disesuaikan saat export)
BACKEND_MODEL_DIR = PROJECT_ROOT.parent / "backend" / "ml" / "models"

# ============================================================================
# Image / ROI Configuration
# ============================================================================
ROI_SIZE = 112  # Input MobileFaceNet: 112×112
ROI_CHANNELS = 3  # RGB
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]

# Quality gates (ROI extraction)
BLUR_THRESHOLD_LAPLACIAN = 50.0  # var < ini → reject (blur)
MIN_PALM_WIDTH_PX = 50  # p5↔p17 distance minimum
HAND_BBOX_MIN_FRAC = 0.25  # hand bbox diagonal / frame diagonal

# ============================================================================
# MediaPipe Configuration
# ============================================================================
MP_MIN_DETECTION_CONFIDENCE = 0.7
MP_MIN_PRESENCE_CONFIDENCE = 0.7
MP_MIN_TRACKING_CONFIDENCE = 0.5
MP_NUM_HANDS = 1
MP_LANDMARK_VISIBILITY_MIN = 0.5

# ============================================================================
# Model Architecture
# ============================================================================
EMBEDDING_DIM = 128  # Output MobileFaceNet
INPUT_CHANNELS = 3

# ============================================================================
# Dataset Split (Tongji)
# ============================================================================
TOTAL_INDIVIDUALS = 300
TRAIN_INDIVIDUALS = 200       # Untuk fine-tuning (closed-set)
TEST_INDIVIDUALS = 100        # Hold-out untuk cross-session eval
HANDS_PER_INDIVIDUAL = 2      # kiri + kanan
NUM_CLASSES = TRAIN_INDIVIDUALS * HANDS_PER_INDIVIDUAL  # 400 classes
RANDOM_SEED = 42

# Per-individu split (sesi 1 → train, sesi 2 → val)
TRAIN_IMGS_PER_HAND = 8
VAL_IMGS_PER_HAND = 2  # sisanya

# ============================================================================
# Training Phase 1 (Softmax Warm-up)
# ============================================================================
PHASE1_EPOCHS_FROZEN = 5
PHASE1_EPOCHS_UNFROZEN = 10
PHASE1_LR_FROZEN = 1e-3
PHASE1_LR_UNFROZEN = 1e-4
PHASE1_BATCH_SIZE = 64
PHASE1_TARGET_VAL_ACC = 0.75  # gate: kalau < ini, debug pipeline

# ============================================================================
# Training Phase 2 (ArcFace Fine-tune)
# ============================================================================
PHASE2_EPOCHS = 60
PHASE2_BATCH_SIZE = 64
PHASE2_LR = 1e-4
PHASE2_WEIGHT_DECAY = 1e-4
PHASE2_GRAD_CLIP_NORM = 5.0

ARCFACE_MARGIN = 0.7
ARCFACE_SCALE = 64.0
ARCFACE_MARGIN_WARMUP_EPOCHS = 10  # linear 0 → 0.7 selama 10 epoch

PHASE2_TARGET_COSINE_GAP = 0.4  # gate

# ============================================================================
# Evaluation
# ============================================================================
EVAL_FAR_TARGETS = [0.001, 0.0001]  # 0.1% dan 0.01%
TARGET_EER = 0.01                    # ≤ 1%
TARGET_TAR_AT_FAR_001 = 0.95         # ≥ 95% di FAR=0.1%

# Matching
DEFAULT_THRESHOLD = 0.50  # fallback kalau threshold.json belum ada
MATCH_TOP_K = 3  # multi-sample matching: avg top-3 per user

# ============================================================================
# Augmentation (Tongji palm data — NO horizontal flip!)
# ============================================================================
AUG_ROTATE_DEG = 15
AUG_CROP_SCALE = (0.9, 1.0)
AUG_CROP_RATIO = (0.85, 1.15)
AUG_BRIGHTNESS = 0.25
AUG_CONTRAST = 0.25
AUG_SATURATION = 0.15
AUG_HUE = 0.05

# ============================================================================
# Smoke Test Gate (Phase M2)
# ============================================================================
SMOKE_TEST_GAP_THRESHOLD_GOOD = 0.2  # > ini → Path C (fine-tune)
SMOKE_TEST_GAP_THRESHOLD_MIN = 0.1   # < ini → Path B (skip fine-tune)
SMOKE_TEST_SAMPLE_INDIVIDUALS = 5
SMOKE_TEST_IMAGES_PER_INDIVIDUAL = 10
