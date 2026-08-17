import numpy as np
import pandas as pd
import torch

from antstorch.lamnr_flows.scripts.train_lamnr_flows_tabular import TabularNormalizer
from antstorch.utilities.dataframe_dataset import MultiViewDataFrameDataset


def test_tabular_normalizer_matches_dataframe_dataset_without_jitter():
    frame = pd.DataFrame({
        "a": [1.0, 2.0, 4.0, 8.0],
        "b": [2.0, 3.0, 5.0, 9.0],
    })
    dataset = MultiViewDataFrameDataset(
        {"view": frame}, normalization="0mean", alpha=0.0,
        add_noise_in="none", impute="mean",
    )
    normalizer = TabularNormalizer("0mean").fit(frame.to_numpy())
    expected = normalizer.transform(frame.iloc[[0]].to_numpy()).squeeze(0)
    actual = dataset[0]["views"]["view"]
    assert torch.allclose(actual, expected, atol=1e-6)


def test_tabular_normalizer_matches_dataframe_dataset_normalized_jitter():
    frame = pd.DataFrame({
        "a": [1.0, 2.0, 4.0, 8.0],
        "b": [2.0, 3.0, 5.0, 9.0],
    })
    dataset = MultiViewDataFrameDataset(
        {"view": frame}, normalization="0mean", alpha=0.1,
        add_noise_in="normalized", impute="mean",
    )
    normalizer = TabularNormalizer("0mean").fit(frame.to_numpy())
    np.random.seed(11)
    actual = dataset[0]["views"]["view"]
    np.random.seed(11)
    expected = normalizer.transform(
        frame.iloc[[0]].to_numpy(), noise_std=0.1
    ).squeeze(0)
    assert torch.allclose(actual, expected, atol=1e-6)
