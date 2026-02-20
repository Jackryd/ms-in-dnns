from medmnist_equivariant.data import MedMNISTDataModule
from medmnist_equivariant.baseline_model import BaselineCNN, PLBaselineModule
__all__ = [
    "MedMNISTDataModule",
    "BaselineCNN",
    "PLBaselineModule",
]

try:
    from medmnist_equivariant.equivariant_model import C4EquivariantCNN, PLC4EquivariantModule

    __all__.extend(["C4EquivariantCNN", "PLC4EquivariantModule"])
except ModuleNotFoundError:
    # Allow baseline training even if escnn dependencies are not fully available yet.
    pass
