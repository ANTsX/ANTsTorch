"""Extract a brain probability map from a 3-D T1 MRI; downloads pretrained assets."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to a 3-D T1 MRI")
    parser.add_argument("--output-dir", type=Path, default=Path("example_outputs/brain"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    import ants
    import antstorch

    image = ants.image_read(str(args.image)).clone("float")
    if image.dimension != 3 or image.has_components:
        parser.error("Expected a scalar 3-D T1 image")
    probability = antstorch.brain_extraction(image, modality="t1", device=args.device, verbose=True)
    mask = ants.threshold_image(probability, 0.5, 1.0, 1, 0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ants.image_write(probability, str(args.output_dir / "brain_probability.nii.gz"))
    ants.image_write(mask, str(args.output_dir / "brain_mask.nii.gz"))
    ants.image_write(image * mask, str(args.output_dir / "brain.nii.gz"))
    print("Results:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
