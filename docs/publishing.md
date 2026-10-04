# Building and publishing documentation

## Local preview

From the repository root:

```bash
python3 -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -r docs/requirements.txt
python -m sphinx -b html -n -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser. This build installs only the
pinned documentation dependencies; it neither imports ANTsTorch nor runs examples.
The documentation CI workflow runs the same strict build and uploads the HTML.

## Read the Docs

The root `.readthedocs.yaml` selects Python 3.12, installs
`docs/requirements.txt`, and builds `docs/conf.py`, treating warnings as errors.

1. Commit and push the documentation files to the GitHub repository and branch
   that Read the Docs will build. Local files are not visible to the service.
2. Sign in to [Read the Docs](https://app.readthedocs.org/dashboard/) and choose
   **Add project**. Select `https://github.com/ANTsX/ANTsTorch.git`.
3. Set the default branch to the branch containing `.readthedocs.yaml` and
   complete the import. If using a different upstream repository, push the
   documentation there and select that repository instead.
4. Confirm the first build succeeds. Enable additional versions or pull-request
   previews in the project settings as desired.
5. Use the project's assigned public URL for the README badge and package
   documentation URL after publication; the project slug may already be taken.

Automatic imports configure a Git integration. Manual imports also require a
webhook for automatic rebuilds. Follow the official
[project import guide](https://docs.readthedocs.com/platform/stable/intro/add-project.html)
and [configuration reference](https://docs.readthedocs.com/platform/stable/config-file/v2.html).
