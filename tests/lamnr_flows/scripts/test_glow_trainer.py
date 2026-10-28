import pytest
import torch
import torch.nn as nn
from antstorch.lamnr_flows.scripts.train_lamnr_glow_2d import LAMNrGlow2DTrainer
from antstorch.lamnr_flows.scripts.train_lamnr_glow_3d import LAMNrGlow3DTrainer

# 1. Mock minimal pour le modèle et le loader
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)
    def forward(self, x): return self.conv(x)
    def forward_and_log_det(self, x): return x, torch.zeros(x.shape[0])

@pytest.fixture
def dummy_args():
    class Args:
        def __init__(self):
            self.num_views = 2
            self.H, self.W = 32, 32
            self.D = 32 # Pour le 3D
            self.device = "cpu"
    return Args()

# 2. Test unitaire pour le Trainer 2D
def test_glow_trainer_2d(dummy_args):
    # Instanciation à vide, puis injection
    trainer = LAMNrGlow2DTrainer() 
    trainer.args = dummy_args
    trainer.models = [DummyModel(), DummyModel()]

    # Mock d'un batch simple
    batch = [torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32)]
    
    # Test d'extraction de vue
    from antstorch.lamnr_flows.core.train_lamnr_glow_base import _extract_views_from_batch
    views = _extract_views_from_batch(batch, num_views=2)

    assert len(views) == 2
    assert views[0].shape == (1, 1, 32, 32)
    
    print("Test 2D: Extraction réussie.")


def test_glow_trainer_2d_infers_channels_after_splitting_packed_views(
    dummy_args, monkeypatch
):
    """Packed NIfTI views are not input channels of each Glow model."""
    from antstorch.lamnr_flows.scripts import train_lamnr_glow_2d as trainer_module

    packed_batch = torch.randn(4, 2, 32, 32)
    loader = [packed_batch]
    monkeypatch.setattr(
        trainer_module,
        "build_loaders_from_globs",
        lambda **kwargs: (loader, loader, object()),
    )

    dummy_args.view = [["view0"], ["view1"]]
    dummy_args.train_samples = 4
    dummy_args.val_samples = 4
    dummy_args.batch = 4
    dummy_args.num_workers = 0
    dummy_args.slice_idx = 0
    dummy_args.val_frac = 0.1
    dummy_args.subject_limit = 0
    dummy_args.aug_schedules = None
    dummy_args.disable_aug_anneal = False
    dummy_args.seed = 0

    trainer = LAMNrGlow2DTrainer()
    trainer.is_ddp = False
    trainer.rank = 0
    trainer.world_size = 1
    trainer.build_loaders(dummy_args)

    assert trainer.input_shape == (1, 32, 32)
    assert dummy_args.C == 1

# 3. Test unitaire pour le Trainer 3D
def test_glow_trainer_3d(dummy_args):
    # Instanciation à vide, puis injection
    trainer = LAMNrGlow3DTrainer() 
    trainer.args = dummy_args
    trainer.models = [DummyModel(), DummyModel()]  

    # Mock d'un batch simple
    batch = [torch.randn(1, 1, 32, 32, 32), torch.randn(1, 1, 32, 32, 32)]
    
    # Test d'extraction de vue
    from antstorch.lamnr_flows.core.train_lamnr_glow_base import _extract_views_from_batch
    views = _extract_views_from_batch(batch, num_views=2)

    assert len(views) == 2
    assert views[0].shape == (1, 1, 32, 32, 32)

    print("Test 3D: Initialisation réussie.")
