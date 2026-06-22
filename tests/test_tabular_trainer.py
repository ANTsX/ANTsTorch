import pytest
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

from sklearn.datasets import load_iris
from torch.utils.data import DataLoader, TensorDataset

# Importation de votre nouveau trainer (ajustez le chemin selon votre structure)
from antstorch.lamnr_flows.scripts.train_lamnr_flows_tabular import TabularLAMNrTrainer

# ---------------------------------------------------------
# 1. Faux Modèle (Mock Model) pour contourner Glow/RealNVP
# ---------------------------------------------------------
class DummyFlowModel(nn.Module):
    """Un faux flux bijectif ultra-léger pour les tests unitaires."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward_and_log_det(self, z):
        # Fausse passe directe (recon) : multiplie par 2
        x_rec = z * 2.0
        log_det = torch.zeros(z.shape[0], device=z.device)
        return x_rec, log_det

    def inverse_and_log_det(self, x):
        # Fausse passe inverse (encodage) : divise par 2
        z = x * 0.5
        log_det = torch.zeros(x.shape[0], device=x.device)
        return z, log_det

# ---------------------------------------------------------
# 2. Fixtures Pytest (Préparation des données/arguments)
# ---------------------------------------------------------
@pytest.fixture
def dummy_args(tmp_path):
    """Génère de faux arguments CLI en utilisant le dossier temporaire de pytest."""
    class Args:
        pass
    
    args = Args()
    args.num_views = 2
    args.views = ["clinique_t1", "clinique_t2"]
    args.out_dir = str(tmp_path / "runs_test")
    
    # Arguments pour l'exportation que nous venons de créer
    args.save_z = True
    args.save_whitened = True
    args.save_recon = True
    
    # Arguments requis par BaseLAMNrTrainer
    args.align = "vicreg"
    args.align_weight = 1.0
    args.weighting = "fixed"
    args.device = "cpu"  # Force le CPU pour les tests CI
    
    return args

@pytest.fixture
def dummy_tabular_trainer(dummy_args):
    """Instancie le trainer en contournant le chargement réel des données."""
    trainer = TabularLAMNrTrainer()
    trainer.args = dummy_args
    trainer.num_views = 2
    trainer.dev = "cpu"
    
    trainer.models = [DummyFlowModel(dim=2), DummyFlowModel(dim=2)]
    
    return trainer

@pytest.fixture
def dummy_tabular_trainer(dummy_args):
    """Instancie le trainer avec les attributs minimaux nécessaires."""
    trainer = TabularLAMNrTrainer()
    trainer.args = dummy_args
    trainer.num_views = len(dummy_args.views)
    trainer.dev = "cpu"
    
    # Injection des modèles (ce que vous avez déjà)
    trainer.models = [DummyFlowModel(dim=2), DummyFlowModel(dim=2)]
    
    # AJOUTEZ CE BLOC : Injection du DataLoader factice pour l'export
    # On crée un faux DataLoader qui contient les données que le trainer attend
    t0 = torch.randn(10, 2) # Faux batch de 10 échantillons, 2 features
    t1 = torch.randn(10, 2)
    dataset = TensorDataset(t0, t1)
    trainer.train_loader = DataLoader(dataset, batch_size=32)
    
    return trainer

# ---------------------------------------------------------
# 3. Les Tests Unitaires
# ---------------------------------------------------------
def test_tabular_trainer_inheritance(dummy_tabular_trainer):
    """Vérifie que le TabularTrainer hérite bien de la base unifiée."""
    from antstorch.lamnr_flows.core.base_trainer import BaseLAMNrTrainer
    assert isinstance(dummy_tabular_trainer, BaseLAMNrTrainer), "TabularLAMNrTrainer doit hériter de BaseLAMNrTrainer"

def test_export_tabular_results(dummy_tabular_trainer, dummy_args):
    """Vérifie que la nouvelle fonction d'exportation génère bien les bons fichiers CSV."""
    # 1. Exécution de la fonction d'exportation
    dummy_tabular_trainer.export_tabular_results()
    
    out_dir = Path(dummy_args.out_dir)
    
    # 2. Assertions : Vérifier que les fichiers existent sur le disque
    for vi in range(dummy_args.num_views):
        path_z = out_dir / f"view_{vi}_latent_z.csv"
        path_w = out_dir / f"view_{vi}_whitened_epsilon.csv"
        path_r = out_dir / f"view_{vi}_reconstructed_x.csv"
        
        assert path_z.exists(), f"Le fichier Z {path_z} n'a pas été créé."
        assert path_w.exists(), f"Le fichier W {path_w} n'a pas été créé."
        assert path_r.exists(), f"Le fichier R {path_r} n'a pas été créé."

        # 3. Assertions : Vérifier le contenu (Optionnel mais recommandé)
        df_z = pd.read_csv(path_z)
        assert df_z.shape == (10, 2), "La forme du DataFrame Z exporté est incorrecte."


from antstorch.lamnr_flows.scripts.train_lamnr_flows_tabular import TabularLAMNrTrainer

# ---------------------------------------------------------
# 1. Faux Modèle (Mock Model) pour exécution rapide
# ---------------------------------------------------------
class DummyFlowModel(nn.Module):
    """Un faux flux bijectif ultra-léger pour les tests unitaires."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward_and_log_det(self, z):
        # Fausse passe directe
        x_rec = z * 2.0
        log_det = torch.zeros(z.shape[0], device=z.device)
        return x_rec, log_det

    def inverse_and_log_det(self, x):
        # Fausse passe inverse
        z = x * 0.5
        log_det = torch.zeros(x.shape[0], device=x.device)
        return z, log_det

# ---------------------------------------------------------
# 2. Le Test d'Intégration Principal
# ---------------------------------------------------------
def test_tabular_trainer_export_iris(tmp_path, monkeypatch):
    """
    Test validant que le trainer tabulaire peut charger des données réelles (Iris),
    exécuter sa logique interne, et exporter les fichiers CSV avec succès.
    """
    # ÉTAPE 1 : Génération des données dans le dossier temporaire de pytest
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    
    view0_path = tmp_path / "iris_view0.csv"
    view1_path = tmp_path / "iris_view1.csv"
    out_dir_path = tmp_path / "test_run_iris"
    out_dir_path.mkdir(exist_ok=True)
    
    df[['sepal length (cm)', 'sepal width (cm)']].to_csv(view0_path, index=False)
    df[['petal length (cm)', 'petal width (cm)']].to_csv(view1_path, index=False)
    
    # ÉTAPE 2 : Simulation des arguments
    class DummyArgsIris:
        views = [str(view0_path), str(view1_path)]
        out_dir = str(out_dir_path)
        max_iter = 5
        batch_size = 32
        align = "vicreg"
        normalization = "0mean"
        save_z = True
        save_whitened = False
        save_recon = True
        
        # Champs techniques
        H, W, D = None, None, None
        sample_mode = "off"
        distributed = False
        rank = 0
        local_rank = 0
        device = "cpu"
        num_views = 2
        lr = 1e-4
        weight_decay = 0.0

    args = DummyArgsIris()

    # ÉTAPE 3 : Patch de la fonction de création du modèle
    # (Ajustez la chaîne si votre fonction se trouve ailleurs, ex: "antstorch.create_...")
    monkeypatch.setattr(
        "antstorch.architectures.create_real_nvp_normalizing_flow_model",
        lambda *a, **kw: DummyFlowModel(dim=2)
    )

    # ÉTAPE 4 : Initialisation du Trainer (Option 2 : à vide)
    trainer = TabularLAMNrTrainer()
    
    # Injection manuelle des paramètres requis (contournement du __init__)
    trainer.args = args
    trainer.num_views = len(args.views)
    trainer.models = [DummyFlowModel(dim=2), DummyFlowModel(dim=2)]
    trainer.dev = "cpu"
    
    # Injection du DataLoader
    t0 = torch.tensor(pd.read_csv(view0_path).values, dtype=torch.float32)
    t1 = torch.tensor(pd.read_csv(view1_path).values, dtype=torch.float32)
    dataset = TensorDataset(t0, t1)
    trainer.train_loader = DataLoader(dataset, batch_size=32)

    # ÉTAPE 5 : Assertions Préliminaires
    assert trainer.num_views == 2, "Le trainer doit avoir 2 vues."
    assert len(trainer.train_loader.dataset) == 150, "Le dataset doit contenir 150 lignes."

    # ÉTAPE 6 : Exécution de la fonction testée
    trainer.export_tabular_results()

    # ÉTAPE 7 : Assertions Finales (Vérification de la création des fichiers)
    assert (out_dir_path / "view_0_latent_z.csv").exists(), "Le fichier Z pour la vue 0 est manquant."
    assert (out_dir_path / "view_1_latent_z.csv").exists(), "Le fichier Z pour la vue 1 est manquant."
    assert (out_dir_path / "view_0_reconstructed_x.csv").exists(), "Le fichier Recon pour la vue 0 est manquant."
    assert (out_dir_path / "view_1_reconstructed_x.csv").exists(), "Le fichier Recon pour la vue 1 est manquant."