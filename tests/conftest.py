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

