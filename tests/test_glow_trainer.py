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
    from antstorch.lamnr_flows.core.base_trainer import _extract_views_from_batch
    views = _extract_views_from_batch(batch, num_views=2)

    assert len(views) == 2
    assert views[0].shape == (1, 1, 32, 32)
    
    print("Test 2D: Extraction réussie.")

# 3. Test unitaire pour le Trainer 3D
def test_glow_trainer_3d(dummy_args):
    # Instanciation à vide, puis injection
    trainer = LAMNrGlow3DTrainer() 
    trainer.args = dummy_args
    trainer.models = [DummyModel(), DummyModel()]  

    # Mock d'un batch simple
    batch = [torch.randn(1, 1, 32, 32, 32), torch.randn(1, 1, 32, 32, 32)]
    
    # Test d'extraction de vue
    from antstorch.lamnr_flows.core.base_trainer import _extract_views_from_batch
    views = _extract_views_from_batch(batch, num_views=2)

    assert len(views) == 2
    assert views[0].shape == (1, 1, 32, 32, 32)

    print("Test 3D: Initialisation réussie.")