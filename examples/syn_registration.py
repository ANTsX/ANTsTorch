"""Register ANTsPy sample images; downloads r16/r64 on first use."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("example_outputs/syn"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    import ants
    from antstorch.syn import syn_registration

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed = ants.image_read(ants.get_ants_data("r16")).clone("float")
    moving = ants.image_read(ants.get_ants_data("r64")).clone("float")
    result = syn_registration(
        fixed, moving, type_of_transform="SyN", device=args.device,
        levels=(4, 2, 1), reg_iterations=(20, 10, 5),
        outprefix=str(args.output_dir / "r64_to_r16_"), verbose=True,
    )
    ants.image_write(result["warpedmovout"], str(args.output_dir / "warped.nii.gz"))
    ants.image_write(result["jacobian"], str(args.output_dir / "jacobian.nii.gz"))
    print("Forward transforms:", result["fwdtransforms"])
    print("Demonstration iteration counts; inspect alignment before downstream use.")


if __name__ == "__main__":
    main()
