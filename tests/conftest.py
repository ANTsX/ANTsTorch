# tests/conftest.py
def pytest_addoption(parser):
    parser.addoption(
        "--dump-aug-samples",
        action="store_true",
        default=False,
        help="If set, dump augmented samples per step to the test temp dir.",
    )
    parser.addoption(
        "--aug-steps",
        type=int,
        default=4,
        help="Number of scheduling steps to iterate in the integration test.",
    )
    parser.addoption(
        "--mods",
        type=str,
        default="T1",
        help='Comma-separated modalities to load (e.g., "T2" or "T2,T1,FA").',
    )

    parser.addoption(
        "--aug-schedules",
        type=str,
        default="",
        help=("Scheduler DSL, e.g. "
              '"noise_std:cos:0.05->0.00@150k,sd_affine:linear:0.05->0.00@80k,'
              'sd_deformation:cos:10.0->0.00@100k,sd_simulated_bias_field:cos:1e-8->0.00@120k,'
              'sd_histogram_warping:exp:0.025->0.00@120k"'),
    )
    parser.addoption(
        "--grid",
        type=int,
        default=10,
        help="Grid size for mosaics (rows=cols=grid).",
    )
    parser.addoption(
        "--tile-size",
        type=int,
        default=128,
        help="Tile size (and dataset H=W) for mosaic PNGs.",
    )
    parser.addoption(
        "--preview-channel",
        type=int,
        default=0,
        help="Which modality/channel index to render in the mosaic.",
    )
