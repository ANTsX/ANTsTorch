"""Offline U-Net forward pass; random weights are not a trained segmentation model."""
import torch
from antstorch import create_unet_model_2d


def main():
    torch.manual_seed(42)
    model = create_unet_model_2d(
        input_channel_size=1, number_of_outputs=3,
        number_of_filters=(8, 16, 32), mode="classification",
    ).eval()
    image = torch.rand(1, 1, 64, 64)
    with torch.inference_mode():
        probabilities = model(image)
    assert probabilities.shape == (1, 3, 64, 64)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(1, 64, 64))
    print("Probability tensor:", tuple(probabilities.shape))


if __name__ == "__main__":
    main()
