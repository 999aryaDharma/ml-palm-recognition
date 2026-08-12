"""
Scratch Weight Initialization untuk PalmNet-Lite.

Semua weight dimulai dari random initialization.
TIDAK ADA pretrained weights, external checkpoint, atau auto-download.

Policy (sesuai docs/02-model-architecture.md §14):
    Conv2d   : Kaiming Normal, fan_out, nonlinearity='leaky_relu'
    Linear   : Xavier Normal
    BatchNorm: weight=1, bias=0
    PReLU    : initial slope=0.25 (diset saat konstruksi modul)

Gunakan:
    model = PalmNetLite(...)
    initialize_scratch_weights(model)
"""
import torch.nn as nn
from torch.nn import init


def initialize_scratch_weights(model: nn.Module, prelu_init: float = 0.25) -> None:
    """Apply scratch initialization ke seluruh model.

    Berjalan secara rekursif ke semua submodule.
    Harus dipanggil SEKALI setelah model dikonstruksi.
    TIDAK memanggil load_state_dict, download, atau pretrained API apa pun.

    Args:
        model: nn.Module yang akan diinisialisasi
        prelu_init: initial negative slope untuk PReLU
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            init.kaiming_normal_(
                module.weight,
                mode="fan_out",
                nonlinearity="leaky_relu",
            )
            if module.bias is not None:
                init.zeros_(module.bias)

        elif isinstance(module, nn.Linear):
            init.xavier_normal_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)

        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if module.weight is not None:
                init.ones_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)

        elif isinstance(module, nn.PReLU):
            # Reset ke initial slope (sudah diset di konstruksi, reset saja)
            init.constant_(module.weight, prelu_init)


def verify_no_pretrained_load(model: nn.Module) -> bool:
    """Verifikasi bahwa model tidak memuat external checkpoint.

    Fungsi ini adalah assertion semantik — tidak bisa mendeteksi
    semua kasus, tetapi memastikan tidak ada attribute yang mengindikasikan
    pretrained loading.

    Returns:
        True jika ok (tidak ada flag pretrained)
    """
    # Cek tidak ada attribute pretrained pada model
    forbidden_attrs = ["pretrained", "pretrained_path", "_loaded_from_checkpoint"]
    for attr in forbidden_attrs:
        if hasattr(model, attr) and getattr(model, attr):
            raise ValueError(
                f"Model memiliki attribute '{attr}' yang mengindikasikan "
                "pretrained weight. PalmNetLite harus dari scratch."
            )
    return True
