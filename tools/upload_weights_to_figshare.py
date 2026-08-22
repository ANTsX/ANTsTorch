#!/usr/bin/env python3
"""
upload_weights_to_figshare.py

Uploads your converted ANTsTorch `_pytorch.pt` weight files to figshare via
figshare's REST API (https://api.figshare.com/v2), using a personal access
token from a standard individual figshare account. Publishes them as files
under a single figshare "article" (dataset), then rewrites the matching
`"<id>_pytorch": ""` placeholder entries in get_pretrained_network.py with
the real `https://ndownloader.figshare.com/files/<id>` download URLs figshare
assigns -- the same URL format ANTsPyNet's own get_pretrained_network.py
already uses for every other id.

Why the API and not FTPS: FTPS is best suited to bulk-uploading raw files
with metadata added later by hand; the API lets this script create the
article, attach every file with a real title, AND read back each file's
public download URL in the same run -- which is what's needed to patch
get_pretrained_network.py automatically. If you'd rather use FTPS or the
web UI yourself, you'll still need the API (or the web UI) afterward just
to look up each file's download URL, so the API ends up being the whole
job in one script rather than two.

SECURITY: your personal access token is read ONLY from the FIGSHARE_TOKEN
environment variable -- never pass it as a command-line argument (it would
end up in your shell history and in `ps`). Generate one at
https://figshare.com/account/applications (Personal Tokens).

METADATA: the article gets a fixed set of base tags (ANTsX/ANTsTorch/
ANTsPyNet/PyTorch/pretrained-weights/deep-learning/medical-imaging) plus,
dynamically, the exact ids being uploaded in this run (so a partial run via
--only still tags accurately). Add more with --tags. `license` and `authors`
are deliberately NOT set: figshare automatically lists the uploading account
as the author, and falls back to that account's default license when none
is given at article creation -- setting them here would just be duplicating
account-level defaults that already apply. If you want a *different*
license than your account default, pass its figshare license id with
--license-id (see https://api.figshare.com/v2/account/licenses for a list
of ids valid for your account).

Only .pt files that (a) actually exist locally and (b) currently have an
empty "" URL in get_pretrained_network.py are uploaded by default -- so it
naturally skips anything you haven't converted yet (e.g. the allen_*/SR
ids you said you don't need).

CAVEAT: this script was written against figshare's published API docs
(https://docs.figshare.com/) but could not be tested end-to-end against a
real account (no token available in the environment this was written in).
The multi-part upload step in particular (reading byte ranges per `parts`
entry from the uploader service) follows the documented field names
(`partNo`, `startOffset`, `endOffset`) but its exact edge-case behavior
(byte-range inclusivity, retry semantics) is not independently verified.
Strongly recommended: run with `--only <one_stem>` first and confirm the
uploaded file downloads correctly before doing the full batch.

Usage:
    export FIGSHARE_TOKEN="your-personal-access-token"

    # dry run: see what would be uploaded, no network calls to figshare
    python upload_weights_to_figshare.py --pt-dir ~/.antstorch --dry-run

    # test on one file first (strongly recommended)
    python upload_weights_to_figshare.py --pt-dir ~/.antstorch --only hyperMapp3r

    # then the full batch
    python upload_weights_to_figshare.py --pt-dir ~/.antstorch

    # reuse an article you already created/published in a partial run
    python upload_weights_to_figshare.py --pt-dir ~/.antstorch --article-id 12345678

    # add your own tags on top of the base set, and/or a specific license
    python upload_weights_to_figshare.py --pt-dir ~/.antstorch --tags segmentation lung --license-id 1

    # figshare REQUIRES at least one category before an article can be published
    # (there's no account default for this, unlike license). Find one:
    python upload_weights_to_figshare.py --list-categories
    # ... then either pass it on a normal run:
    python upload_weights_to_figshare.py --pt-dir ~/.antstorch --category-ids 68 --only hyperMapp3r_pytorch
    # ... or, if files are already uploaded to an article whose publish failed
    # on the missing-categories error, finish it WITHOUT re-uploading:
    python upload_weights_to_figshare.py --pt-dir ~/.antstorch --article-id 12345678 \\
        --finalize-only --category-ids 68
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

API_BASE = "https://api.figshare.com/v2"

# Always-applied tags, regardless of which ids are uploaded in a given run.
TAGS_BASE = [
    "ANTsX", "ANTsTorch", "ANTsPyNet", "PyTorch",
    "pretrained-weights", "deep-learning", "medical-imaging",
]

# The 50 lung/mouse/WMH ids this session added as "" placeholders in
# get_pretrained_network.py. Only ones with a matching local .pt file are
# actually uploaded (see main()).
KNOWN_IDS = [
    "protonLungMri_pytorch", "protonLobes_pytorch", "maskLobes_pytorch",
    "lungCtWithPriorsSegmentationWeights_pytorch", "wholeLungMaskFromVentilation_pytorch",
    "xrayLungExtraction_pytorch", "elBicho_pytorch", "pulmonaryArteryWeights_pytorch",
    "pulmonaryAirwayWeights_pytorch", "mouseT2wBrainExtraction3D_pytorch",
    "ex5_coronal_weights_pytorch", "ex5_sagittal_weights_pytorch",
    "mouseT2wBrainParcellation3DNick_pytorch", "mouseT2wBrainParcellation3DTct_pytorch",
    "mouseSTPTBrainParcellation3DJay_pytorch", "allen_brain_mask_weights_pytorch",
    "allen_brain_leftright_coronal_mask_weights_pytorch",
    "allen_cerebellum_sagittal_mask_weights_pytorch", "allen_cerebellum_coronal_mask_weights_pytorch",
    "allen_sr_weights_pytorch",
    "sysuMediaWmhFlairOnlyModel0_pytorch", "sysuMediaWmhFlairOnlyModel1_pytorch",
    "sysuMediaWmhFlairOnlyModel2_pytorch", "sysuMediaWmhFlairT1Model0_pytorch",
    "sysuMediaWmhFlairT1Model1_pytorch", "sysuMediaWmhFlairT1Model2_pytorch",
    "hyperMapp3r_pytorch", "antsxnetWmhOr_pytorch", "antsxnetWmh_pytorch",
    "pvs_shiva_t1_0_pytorch", "pvs_shiva_t1_1_pytorch", "pvs_shiva_t1_2_pytorch",
    "pvs_shiva_t1_3_pytorch", "pvs_shiva_t1_4_pytorch", "pvs_shiva_t1_5_pytorch",
    "pvs_shiva_t1_flair_0_pytorch", "pvs_shiva_t1_flair_1_pytorch", "pvs_shiva_t1_flair_2_pytorch",
    "pvs_shiva_t1_flair_3_pytorch", "pvs_shiva_t1_flair_4_pytorch",
    "wmh_shiva_flair_0_pytorch", "wmh_shiva_flair_1_pytorch", "wmh_shiva_flair_2_pytorch",
    "wmh_shiva_flair_3_pytorch", "wmh_shiva_flair_4_pytorch",
    "wmh_shiva_t1_flair_0_pytorch", "wmh_shiva_t1_flair_1_pytorch", "wmh_shiva_t1_flair_2_pytorch",
    "wmh_shiva_t1_flair_3_pytorch", "wmh_shiva_t1_flair_4_pytorch",
]


def api_request(method, url, token, json_body=None, headers=None, context=None, parse_json=True):
    data = json.dumps(json_body).encode() if json_body is not None else None
    hdrs = {"Authorization": f"token {token}", "Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
    tag = f"[{context}] " if context else ""
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            if not raw:
                return resp.status, None, dict(resp.headers)
            if not parse_json:
                # Some endpoints (e.g. the "mark upload complete" call, which figshare
                # answers with HTTP 202 and a plain-text "Accepted for processing" body,
                # not JSON) never return a JSON body -- don't even try to parse one.
                return resp.status, raw.decode(errors="replace"), dict(resp.headers)
            try:
                return resp.status, json.loads(raw), dict(resp.headers)
            except json.JSONDecodeError as e:
                # Surface the raw body -- a bare "Extra data"/"Expecting value" message
                # gives no clue which call failed or what figshare actually sent back
                # (e.g. an HTML error page, a truncated body, or a non-JSON response).
                snippet = raw[:500].decode(errors="replace")
                raise RuntimeError(
                    f"{tag}{method} {url} -> HTTP {resp.status} but response body is not "
                    f"valid JSON ({e}). First 500 bytes of body:\n{snippet!r}"
                ) from None
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"{tag}{method} {url} -> HTTP {e.code}: {body}") from None


def md5_and_size(path):
    h = hashlib.md5()
    size = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def create_article(token, title, description, tags=None, license_id=None, category_ids=None):
    json_body = {"title": title, "description": description, "defined_type": "dataset"}
    if tags:
        json_body["tags"] = tags
    if license_id is not None:
        json_body["license"] = license_id
    if category_ids:
        json_body["categories"] = category_ids
    status, body, headers = api_request(
        "POST", f"{API_BASE}/account/articles", token, json_body=json_body, context="create-article",
    )
    # figshare returns the new article's API URL either in the JSON body's
    # "location" field or the Location header, depending on version.
    location = (body or {}).get("location") or headers.get("Location")
    if not location:
        raise RuntimeError(f"Could not determine new article location from response: {body}")
    article_id = location.rstrip("/").split("/")[-1]
    return article_id


def upload_one_file(token, article_id, path, verbose):
    fname = os.path.basename(path)
    md5, size = md5_and_size(path)
    if verbose:
        print(f"    md5={md5} size={size}")

    # Step 1: initiate
    status, body, headers = api_request(
        "POST", f"{API_BASE}/account/articles/{article_id}/files", token,
        json_body={"name": fname, "md5": md5, "size": size},
        context="1-initiate",
    )
    file_location = (body or {}).get("location") or headers.get("Location")
    if not file_location:
        raise RuntimeError(f"No file location returned for {fname}: status={status} body={body} headers={headers}")
    if verbose:
        print(f"    [1-initiate] status={status} file_location={file_location}")

    # Step 2: fetch upload_url
    status, file_info, _ = api_request("GET", file_location, token, context="2-fetch-upload_url")
    upload_url = file_info["upload_url"]
    if verbose:
        print(f"    [2-fetch-upload_url] status={status} upload_url={upload_url}")

    # Step 3: fetch parts list from the uploader service
    status, upload_info, _ = api_request("GET", upload_url, token, context="3-fetch-parts")
    parts = upload_info["parts"]
    if verbose:
        print(f"    [3-fetch-parts] status={status} n_parts={len(parts)}")

    with open(path, "rb") as f:
        for part in parts:
            start, end, part_no = part["startOffset"], part["endOffset"], part["partNo"]
            f.seek(start)
            chunk = f.read(end - start + 1)
            req = urllib.request.Request(
                f"{upload_url}/{part_no}", data=chunk, method="PUT",
                headers={"Authorization": f"token {token}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=300):
                    pass
            except urllib.error.HTTPError as e:
                raise RuntimeError(
                    f"[3b-put-part {part_no}] PUT {upload_url}/{part_no} -> HTTP {e.code}: "
                    f"{e.read().decode(errors='replace')}"
                ) from None
            if verbose:
                print(f"    part {part_no}: {len(chunk)} bytes uploaded (offsets {start}-{end})")

    # Step 4: mark upload complete -- figshare answers with HTTP 202 and a
    # plain-text "Accepted for processing" body, not JSON, so parse_json=False.
    status, complete_body, _ = api_request(
        "POST", file_location, token, context="4-complete", parse_json=False,
    )
    if status not in (200, 202):
        raise RuntimeError(f"[4-complete] unexpected status {status} for {file_location}: {complete_body!r}")
    if verbose:
        print(f"    [4-complete] status={status} body={complete_body!r}")
    return file_location


def list_account_categories(token):
    # figshare requires every article to have at least one category before
    # it can be published; this lists the categories your account may pick
    # from (id + human-readable title) so you can choose one explicitly --
    # nothing here guesses which one fits your data.
    status, body, _ = api_request(
        "GET", f"{API_BASE}/account/categories", token, context="list-categories",
    )
    return body  # list of {id, title, parent_id, source_id, taxonomy_id}


def update_article_categories(token, article_id, category_ids):
    api_request(
        "PUT", f"{API_BASE}/account/articles/{article_id}", token,
        json_body={"categories": category_ids}, context="update-categories", parse_json=False,
    )


def publish_article(token, article_id):
    try:
        api_request("POST", f"{API_BASE}/account/articles/{article_id}/publish", token, context="publish")
    except RuntimeError as e:
        if "categories" in str(e).lower():
            raise RuntimeError(
                f"{e}\n\n"
                "figshare requires at least one category before an article can be published. "
                "Find a category id with:\n"
                "    python upload_weights_to_figshare.py --list-categories\n"
                "then set it explicitly and re-finalize this article without re-uploading:\n"
                f"    python upload_weights_to_figshare.py --pt-dir <dir> --article-id {article_id} "
                f"--finalize-only --category-ids <id>"
            ) from None
        raise


def get_article_files(token, article_id):
    status, body, _ = api_request(
        "GET", f"{API_BASE}/account/articles/{article_id}", token, context="get-article-files",
    )
    return body["files"]  # each has at least: id, name, download_url


def patch_get_pretrained_network(gpn_path, id_to_url, verbose):
    with open(gpn_path, "r") as f:
        content = f.read()

    n_patched = 0
    for file_id, url in id_to_url.items():
        # Replace only the specific empty-string placeholder line for this id,
        # e.g.  "hyperMapp3r_pytorch": "",   ->   "hyperMapp3r_pytorch": "https://...",
        pattern = re.compile(rf'("{re.escape(file_id)}"\s*:\s*)""')
        new_content, n = pattern.subn(rf'\1"{url}"', content)
        if n == 0:
            print(f"    [warn] no \"\" placeholder found for {file_id!r} in {gpn_path} -- not patched")
            continue
        content = new_content
        n_patched += n
        if verbose:
            print(f"    patched {file_id} -> {url}")

    with open(gpn_path, "w") as f:
        f.write(content)
    return n_patched


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pt-dir", default="~/.antstorch", help="Directory containing the converted <id>.pt files (default: ~/.antstorch)")
    p.add_argument("--gpn-path", default=None, help="Path to get_pretrained_network.py to patch (default: auto-detected next to this script's repo layout)")
    p.add_argument("--only", nargs="+", default=None, help="Only upload these ids (space-separated, e.g. hyperMapp3r_pytorch)")
    p.add_argument("--article-id", default=None, help="Reuse an existing (unpublished) article id instead of creating a new one -- e.g. to resume after a partial run")
    p.add_argument("--tags", nargs="+", default=None, help="Extra tags to add on top of the base set (ANTsX, ANTsTorch, ANTsPyNet, PyTorch, ...) and the uploaded ids")
    p.add_argument("--license-id", type=int, default=None, help="figshare license id to set explicitly (default: your account's default license -- see https://api.figshare.com/v2/account/licenses)")
    p.add_argument("--category-ids", nargs="+", type=int, default=None, help="figshare category id(s) -- REQUIRED before an article can be published (figshare has no default). Find ids with --list-categories.")
    p.add_argument("--list-categories", action="store_true", help="Print your account's available figshare categories (id + title) and exit -- no upload, no article created")
    p.add_argument("--finalize-only", action="store_true", help="Skip (re-)uploading files -- assume they're already attached to --article-id from a prior run. Just set categories, publish, and patch URLs.")
    p.add_argument("--dry-run", action="store_true", help="Print what would be uploaded/patched; make no network calls to figshare")
    p.add_argument("--no-publish", action="store_true", help="Upload files but don't publish the article yet (download URLs won't be public/resolvable until you publish)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    token = os.environ.get("FIGSHARE_TOKEN")
    if not token and not (args.dry_run and not args.list_categories):
        print("Error: set FIGSHARE_TOKEN in your environment first (never pass it as a CLI argument).")
        print('  export FIGSHARE_TOKEN="your-personal-access-token"')
        sys.exit(1)

    if args.list_categories:
        print("Your account's available figshare categories:\n")
        for cat in list_account_categories(token):
            print(f"  {cat['id']:>10}  {cat['title']}")
        print("\nPass one or more with --category-ids <id> [<id> ...].")
        return

    if args.finalize_only and not args.article_id:
        print("Error: --finalize-only requires --article-id <id> (the article whose files are already uploaded).")
        sys.exit(1)

    pt_dir = os.path.expanduser(args.pt_dir)

    gpn_path = args.gpn_path
    if gpn_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        gpn_path = os.path.normpath(os.path.join(here, "..", "antstorch", "utilities", "get_pretrained_network.py"))

    wanted_ids = args.only if args.only else KNOWN_IDS
    to_upload = []
    for file_id in wanted_ids:
        path = os.path.join(pt_dir, f"{file_id}.pt")
        if os.path.exists(path):
            to_upload.append((file_id, path))
        elif args.only:
            print(f"[skip] {file_id}: {path} not found")

    if not to_upload:
        print(f"No .pt files found in {pt_dir} matching the requested id(s). Nothing to do.")
        return

    print(f"Found {len(to_upload)} file(s) to upload:")
    for file_id, path in to_upload:
        print(f"  {file_id}  <-  {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")
    print(f"\nget_pretrained_network.py to patch: {gpn_path}")

    if args.dry_run:
        print("\n--dry-run: stopping here, no network calls made.")
        return

    if not os.path.exists(gpn_path):
        print(f"Error: {gpn_path} not found. Pass --gpn-path explicitly.")
        sys.exit(1)

    if args.article_id:
        article_id = args.article_id
        print(f"\nReusing article {article_id}")
        existing_names = set()
        if not args.finalize_only:
            existing_names = {f["name"] for f in get_article_files(token, article_id)}
            if existing_names:
                print(f"  already has {len(existing_names)} file(s) attached -- these will be skipped, not re-uploaded")
    else:
        existing_names = set()
        # Dedup while preserving order: base tags first, then the ids actually
        # being uploaded this run (so a partial --only run tags accurately
        # instead of claiming all 50), then any extra --tags from the user.
        tags = list(dict.fromkeys(TAGS_BASE + [file_id for file_id, _ in to_upload] + (args.tags or [])))
        print("\nCreating figshare article...")
        print(f"  tags: {tags}")
        article_id = create_article(
            token,
            title="ANTsTorch pretrained weights (lung/mouse/WMH)",
            description=(
                "PyTorch (.pt) weight files for the ANTsTorch port of ANTsPyNet's "
                "lung_extraction, lung_segmentation, mouse.py, and "
                "white_matter_hyperintensity_segmentation applications. "
                "See https://github.com/ANTsX/ANTsTorch."
            ),
            tags=tags,
            license_id=args.license_id,
            category_ids=args.category_ids,
        )
        print(f"  article id: {article_id}  (https://figshare.com/account/articles/{article_id})")

    if args.finalize_only:
        print(f"\n--finalize-only: skipping upload, assuming these id(s) are already attached to article {article_id}:")
        for file_id, _ in to_upload:
            print(f"  {file_id}")
        uploaded = dict(to_upload)
    else:
        uploaded = {}
        for file_id, path in to_upload:
            if f"{file_id}.pt" in existing_names:
                print(f"\n[skip] {file_id}: already attached to article {article_id}")
                uploaded[file_id] = path
                continue
            print(f"\nUploading {file_id} ...")
            try:
                upload_one_file(token, article_id, path, args.verbose)
                uploaded[file_id] = path
                print(f"  done.")
            except Exception as e:
                print(f"  FAILED: {e}")

        if not uploaded:
            print("\nNothing uploaded successfully -- not publishing, not patching.")
            sys.exit(1)

    if args.no_publish:
        print(f"\n--no-publish: leaving article {article_id} unpublished. "
              f"Publish it later (or rerun with --article-id {article_id}) before the URLs will resolve.")
        return

    if args.category_ids:
        print(f"\nSetting categories {args.category_ids} on article {article_id} ...")
        update_article_categories(token, article_id, args.category_ids)

    print(f"\nPublishing article {article_id} ...")
    publish_article(token, article_id)

    print("Reading back public download URLs...")
    files = get_article_files(token, article_id)
    by_name = {f["name"]: f for f in files}

    id_to_url = {}
    for file_id in uploaded:
        fname = f"{file_id}.pt"
        info = by_name.get(fname)
        if info is None:
            print(f"  [warn] could not find {fname} in the published article's file list")
            continue
        url = info.get("download_url") or f"https://ndownloader.figshare.com/files/{info['id']}"
        id_to_url[file_id] = url
        print(f"  {file_id} -> {url}")

    print(f"\nPatching {gpn_path} ...")
    n = patch_get_pretrained_network(gpn_path, id_to_url, args.verbose)
    print(f"Patched {n} entries.")


if __name__ == "__main__":
    main()
