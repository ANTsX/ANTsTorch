#!/usr/bin/env python3
"""
convert_harvard_oxford_atlas_labeling_bespoke.py

Standalone (h5py + torch only -- no TensorFlow/antspynet needed) weight
converter for harvard_oxford_atlas_labeling's single weight file,
HarvardOxfordAtlasSubcortical.h5 -> HarvardOxfordAtlasSubcortical_pytorch.pt.

Written 2026-08-24 while triaging a real parity FAIL: the parity framework
(ANTsPyNet-ANTsTorch-Comparison) showed segmentation_image dice=0.63 for
this function even though all 33 probability_images individually agreed
well, and brain_extraction(modality="t1hemi") -- the other big external
input this function depends on -- was independently confirmed clean
(dice=0.9871). That leaves this function's OWN weight conversion as the
leading remaining suspect. Rather than debug an existing converter whose
location Nick couldn't immediately find, this is a fresh, dedicated,
self-contained script in the same style as this toolkit's other bespoke
converters (convert_wmh_bespoke.py, convert_hippmapp3r_hypothalamus_
claustrum_bespoke.py) -- easy to find (named after the one function it
converts) and easy to audit end-to-end in one file.

Architecture (confirmed against the real antspynet source,
antspynet/utilities/harvard_oxford_atlas_labeling.py, and antstorch's own
port, antstorch/utilities/harvard_oxford_atlas_labeling.py -- both fetched
and read directly, not guessed):

    unet_model_pre = create_unet_model_3d(
        (160, 176, 160, 1), number_of_outputs=23, mode="classification",
        number_of_filters=(16, 32, 64, 128), dropout_rate=0.0,
        convolution_kernel_size=(3,3,3), deconvolution_kernel_size=(2,2,2))
    penultimate_layer = unet_model_pre.layers[-2].output   # 16-channel features
    output2 = Conv3D(1, (1,1,1), activation='sigmoid')(penultimate_layer)
    unet_model = Model(inputs=unet_model_pre.input,
                        outputs=[unet_model_pre.output, output2])

i.e. a plain 4-level 3D U-Net (23-channel softmax classification head) with
ONE extra 1-channel sigmoid "foreground probability" head grafted onto the
same penultimate (pre-classification-conv) feature map -- exactly what
antstorch's create_multihead_unet_model_3d(base_unet=..., n_aux_heads=1,
use_sigmoid=True, n_main_outputs=23) wraps. Number_of_outputs=23 is not a
guess -- it is recomputed below from harvard_oxford_atlas_labeling.py's own
label tuples (hoa_lateral_labels + hoa_lateral_left_labels +
hoa_extra_labels = 7+13+3 = 23), so this script fails loudly if that ever
changes rather than silently assuming a stale number.

Keras layer-name risk (see convert_hippmapp3r_hypothalamus_claustrum_
bespoke.py's docstring for why this matters -- hypothalamus.h5 turned out
to be a full *training* graph with hand-picked layer names, not a plain
inference save_weights(), which broke a purely positional assumption):
create_unet_model_3d and harvard_oxford_atlas_labeling's inline `output2 =
Conv3D(...)` never pass an explicit `name=` kwarg anywhere, so -- UNLIKE
hypothalamus -- there is no reason to expect hand-picked names here. But
this script does NOT rely on that expectation alone. Every conv/deconv
layer is instead classified by its KERNEL SHAPE, read directly from the
real .h5 (3x3x3 = regular conv, 2x2x2 = deconv, 1x1x1 with 23 output
channels = main classification head, 1x1x1 with 1 output channel = aux
head) -- channel counts are unambiguous and self-verifying, so a wrong
guess raises immediately instead of silently mis-assigning weights. Only
the RELATIVE ORDER *within* the 14 same-shaped (3x3x3) conv layers relies
on positional reasoning (encoder's 8 built strictly before decoder's 6, in
the real Keras `layer_names` file order) -- the same "no secondary branch,
so creation order == file order" reasoning already verified for
`lung_proton` in this project, and unlike hippmapp3r/hypermapp3r's
Add()-branch reordering surprise, HOA's decoder has no such branch (no
attention gating, no side-output merged back in).

⚠️  NOT YET VERIFIED AGAINST THE REAL FILE: no HarvardOxfordAtlasSubcortical.h5
was available in the sandbox this was written in. Run with --self-test
first (builds a synthetic .h5 with the exact expected shapes, PLUS decoy
non-weight layers like Activation/MaxPooling3D/Concatenate/UpSampling3D and
a decoy conv, to prove the shape-based classifier ignores what it should
and finds exactly one main/aux head) to sanity-check this script's logic
end-to-end without needing the real weights. Then run for real and inspect
the printed layer classification (--verbose) before trusting the output --
if the encoder/decoder split or channel counts look wrong, that print is
the signal, not a downstream crash.

Usage:

    python convert_harvard_oxford_atlas_labeling_bespoke.py --self-test

    python convert_harvard_oxford_atlas_labeling_bespoke.py \\
        --weights-dir ~/.keras/ANTsXNet \\
        --out-dir ~/.antstorch \\
        --antstorch-src ~/Pkg/ANTsTorch
"""
import argparse
import importlib.util
import os
import sys
import tempfile
import types

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_wmh_bespoke import h5_layer_names, h5_get_wb, set_conv, natkey  # noqa: E402


# ---------------------------------------------------------------------------
# HOA's own label bookkeeping, copied verbatim from harvard_oxford_atlas_
# labeling.py (both antspynet and antstorch agree on these tuples) so
# number_of_outputs is derived, not hard-coded.
# ---------------------------------------------------------------------------

HOA_LATERAL_LABELS = (0, 3, 4, 5, 6, 15, 24)
HOA_LATERAL_LEFT_LABELS = (1, 7, 9, 11, 13, 16, 18, 20, 22, 25, 27, 29, 31)
HOA_EXTRA_LABELS = (33, 34, 35)
NUMBER_OF_OUTPUTS = len(sorted((*HOA_LATERAL_LABELS, *HOA_LATERAL_LEFT_LABELS, *HOA_EXTRA_LABELS)))
assert NUMBER_OF_OUTPUTS == 23, NUMBER_OF_OUTPUTS

NUMBER_OF_FILTERS = (16, 32, 64, 128)
CROPPED_TEMPLATE_SIZE = (160, 176, 160)


def _load_create_unet_model(antstorch_src):
    """Import antstorch/architectures/create_unet_model.py directly by file
    path -- no dependency on the rest of the antstorch package being
    importable (mirrors _load_arch_modules's approach in
    convert_wmh_bespoke.py, but that helper returns create_custom_unet_model,
    not create_unet_model, which is where create_unet_model_3d and
    create_multihead_unet_model_3d actually live)."""
    path = os.path.join(antstorch_src, "antstorch", "architectures", "create_unet_model.py")
    spec = importlib.util.spec_from_file_location("_hoa_create_unet_model", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Classify every weighted Conv3D/Conv3DTranspose layer in the real .h5 by
# kernel shape -- see module docstring for why this is safer than a pure
# name-pattern regex for this architecture.
# ---------------------------------------------------------------------------

def _classify_hoa_conv_layers(h5_path, verbose=True):
    names = h5_layer_names(h5_path)
    conv3x3, deconv2x2 = [], []
    main_out, aux_out = None, None
    skipped = []

    with h5py.File(h5_path, "r") as f:
        for n in names:
            try:
                W, b = h5_get_wb(f, n)
            except Exception:
                skipped.append(n)
                continue
            if W.ndim != 5:
                skipped.append(n)
                continue
            kD, kH, kW, c_in, c_out = W.shape
            if (kD, kH, kW) == tuple((3, 3, 3)):
                conv3x3.append(n)
            elif (kD, kH, kW) == (2, 2, 2):
                deconv2x2.append(n)
            elif (kD, kH, kW) == (1, 1, 1):
                if c_out == NUMBER_OF_OUTPUTS:
                    assert main_out is None, f"2 candidate main-output layers: {main_out!r}, {n!r}"
                    main_out = n
                elif c_out == 1:
                    assert aux_out is None, f"2 candidate aux-head layers: {aux_out!r}, {n!r}"
                    aux_out = n
                else:
                    skipped.append(n)
            else:
                skipped.append(n)

    if verbose:
        print(f"[classify] {len(names)} total h5 layer entries")
        print(f"[classify] 3x3x3 conv layers (encoder+decoder): {len(conv3x3)} -> {conv3x3}")
        print(f"[classify] 2x2x2 deconv layers: {len(deconv2x2)} -> {deconv2x2}")
        print(f"[classify] main output (23ch, softmax) layer: {main_out}")
        print(f"[classify] aux head (1ch, sigmoid) layer: {aux_out}")
        print(f"[classify] ignored (no kernel, or unrecognized shape): {len(skipped)} layers")

    assert len(conv3x3) == 14, f"expected 14 3x3x3 conv layers (8 encoder + 6 decoder), got {len(conv3x3)}: {conv3x3}"
    assert len(deconv2x2) == 3, f"expected 3 2x2x2 deconv layers, got {len(deconv2x2)}: {deconv2x2}"
    assert main_out is not None, "no 23-channel 1x1x1 conv found (main classification head)"
    assert aux_out is not None, "no 1-channel 1x1x1 conv found (aux foreground-probability head)"
    return conv3x3, deconv2x2, main_out, aux_out


def _torch_conv3x3_prefixes():
    """base.encoding_convolution_layers.{i}.{0,2} then
    base.decoding_convolution_layers.{i}.{0,2} -- see create_unet_model_3d
    in antstorch/architectures/create_unet_model.py: each encoder/decoder
    block is nn.Sequential(conv1, activation, conv2, activation), so the
    conv weights live at sub-indices 0 and 2 (dropout_rate=0.0 for HOA, so
    no extra Dropout layer shifts these indices)."""
    prefixes = []
    for i in range(len(NUMBER_OF_FILTERS)):
        prefixes.append(f"base.encoding_convolution_layers.{i}.0")
        prefixes.append(f"base.encoding_convolution_layers.{i}.2")
    for i in range(len(NUMBER_OF_FILTERS) - 1):
        prefixes.append(f"base.decoding_convolution_layers.{i}.0")
        prefixes.append(f"base.decoding_convolution_layers.{i}.2")
    return prefixes


def _torch_deconv_prefixes():
    return [f"base.decoding_convolution_transpose_layers.{i}.deconv"
            for i in range(len(NUMBER_OF_FILTERS) - 1)]


def convert_harvard_oxford_atlas_labeling(h5_path, model, verbose=True):
    # create_multihead_unet_model_3d lazily builds self.heads on its first
    # forward call -- run one warmup pass first so "heads.0.weight"/"heads.0.bias"
    # actually exist in model.state_dict() before we try to assign into them.
    model.eval()
    with torch.no_grad():
        _ = model(torch.zeros(1, 1, *CROPPED_TEMPLATE_SIZE))

    conv3x3, deconv2x2, main_out, aux_out = _classify_hoa_conv_layers(h5_path, verbose=verbose)

    torch_conv3x3 = _torch_conv3x3_prefixes()
    torch_deconv = _torch_deconv_prefixes()
    assert len(torch_conv3x3) == len(conv3x3) == 14
    assert len(torch_deconv) == len(deconv2x2) == 3

    sd = {k: v.clone() for k, v in model.state_dict().items()}
    with h5py.File(h5_path, "r") as f:
        for prefix, hname in zip(torch_conv3x3, conv3x3):
            W, b = h5_get_wb(f, hname)
            set_conv(sd, prefix, W, b, expect_bias=True)
        for prefix, hname in zip(torch_deconv, deconv2x2):
            W, b = h5_get_wb(f, hname)
            set_conv(sd, prefix, W, b, expect_bias=True)
        W, b = h5_get_wb(f, main_out)
        set_conv(sd, "base.output.0", W, b, expect_bias=True)
        W, b = h5_get_wb(f, aux_out)
        set_conv(sd, "heads.0", W, b, expect_bias=True)

    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return model


# ---------------------------------------------------------------------------
# Verification: the multihead wrapper returns (main, aux), not a single
# tensor -- can't reuse convert_wmh_bespoke.verify_and_save() as-is.
# ---------------------------------------------------------------------------

def verify_and_save_multihead(model, out_path, build_fn, x):
    model.eval()
    with torch.no_grad():
        y_main, y_aux = model(x)
    s = y_main.sum(dim=1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-3), "main softmax output not normalized"
    assert torch.all((y_aux >= 0) & (y_aux <= 1)), "aux sigmoid output out of [0,1]"

    torch.save(model.state_dict(), out_path)

    model2 = build_fn()
    with torch.no_grad():
        _ = model2(x)  # warmup model2's heads too, before loading state dict into them
    model2.load_state_dict(torch.load(out_path, map_location="cpu", weights_only=True), strict=True)
    model2.eval()
    with torch.no_grad():
        y2_main, y2_aux = model2(x)
    assert torch.allclose(y_main, y2_main, atol=1e-6), f"reload mismatch (main) for {out_path}"
    assert torch.allclose(y_aux, y2_aux, atol=1e-6), f"reload mismatch (aux) for {out_path}"
    print(f"OK -> {out_path}  main shape {tuple(y_main.shape)}  aux shape {tuple(y_aux.shape)}")


def build_model(ccu):
    base = ccu.create_unet_model_3d(
        input_channel_size=1,
        number_of_outputs=NUMBER_OF_OUTPUTS,
        number_of_filters=NUMBER_OF_FILTERS,
        dropout_rate=0.0,
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        mode="classification",
        pad_crop="keras",
    )
    return ccu.create_multihead_unet_model_3d(
        base_unet=base, n_aux_heads=1, use_sigmoid=True, n_main_outputs=NUMBER_OF_OUTPUTS,
    )


# ---------------------------------------------------------------------------
# Self-test: build a synthetic .h5 with the exact expected real-file shape,
# PLUS decoy layers (activation/pooling/concat/upsampling names with no
# kernel, and one decoy conv with an unrelated channel count) so a passing
# self-test proves the classifier both finds the right layers AND correctly
# ignores what it should -- without needing Nick's real weights.
# ---------------------------------------------------------------------------

def _write_keras_conv_layer(root, name, kernel_shape, bias_shape, rng):
    g = root.create_group(name).create_group(name)
    g.create_dataset("kernel:0", data=rng.standard_normal(kernel_shape).astype(np.float32))
    if bias_shape is not None:
        g.create_dataset("bias:0", data=rng.standard_normal(bias_shape).astype(np.float32))


def _build_synthetic_h5(path, rng):
    layer_names = []
    with h5py.File(path, "w") as f:
        f.attrs["layer_names"] = []  # placeholder, fixed up at the end

        # Decoy non-weight layers, interleaved, matching typical Keras names.
        for decoy in ("input_1", "max_pooling3d", "up_sampling3d", "concatenate", "activation"):
            f.create_group(decoy).create_group(decoy)  # empty: no kernel:0 inside
            layer_names.append(decoy)

        in_ch = 1
        for i, nf in enumerate(NUMBER_OF_FILTERS):
            c1_in = in_ch if i == 0 else NUMBER_OF_FILTERS[i - 1]
            name1 = f"enc_conv_{i}_0"
            name2 = f"enc_conv_{i}_1"
            _write_keras_conv_layer(f, name1, (3, 3, 3, c1_in, nf), (nf,), rng)
            _write_keras_conv_layer(f, name2, (3, 3, 3, nf, nf), (nf,), rng)
            layer_names += [name1, name2]

        for i in range(len(NUMBER_OF_FILTERS) - 1):
            out_ch = NUMBER_OF_FILTERS[len(NUMBER_OF_FILTERS) - i - 2]
            in_ch_deconv = NUMBER_OF_FILTERS[len(NUMBER_OF_FILTERS) - i - 1]
            dname = f"dec_deconv_{i}"
            # Keras Conv3DTranspose kernel shape: (kD,kH,kW, filters=out, input_dim=in)
            _write_keras_conv_layer(f, dname, (2, 2, 2, out_ch, in_ch_deconv), (out_ch,), rng)
            layer_names.append(dname)

            c1name, c2name = f"dec_conv_{i}_0", f"dec_conv_{i}_1"
            _write_keras_conv_layer(f, c1name, (3, 3, 3, 2 * out_ch, out_ch), (out_ch,), rng)
            _write_keras_conv_layer(f, c2name, (3, 3, 3, out_ch, out_ch), (out_ch,), rng)
            layer_names += [c1name, c2name]

        # Decoy conv with an unrelated 1x1x1 channel count (neither 23 nor 1) --
        # must be ignored, not mistaken for main/aux head.
        _write_keras_conv_layer(f, "decoy_1x1_conv", (1, 1, 1, NUMBER_OF_FILTERS[0], 5), (5,), rng)
        layer_names.append("decoy_1x1_conv")

        _write_keras_conv_layer(f, "main_output", (1, 1, 1, NUMBER_OF_FILTERS[0], NUMBER_OF_OUTPUTS),
                                 (NUMBER_OF_OUTPUTS,), rng)
        layer_names.append("main_output")
        _write_keras_conv_layer(f, "aux_output", (1, 1, 1, NUMBER_OF_FILTERS[0], 1), (1,), rng)
        layer_names.append("aux_output")

        f.attrs["layer_names"] = layer_names


def run_self_test(antstorch_src=None):
    print("[self-test] building synthetic .h5 with known shapes + decoy layers...")
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        h5_path = os.path.join(tmp, "HarvardOxfordAtlasSubcortical.h5")
        _build_synthetic_h5(h5_path, rng)

        conv3x3, deconv2x2, main_out, aux_out = _classify_hoa_conv_layers(h5_path, verbose=True)
        assert main_out == "main_output", main_out
        assert aux_out == "aux_output", aux_out
        assert "decoy_1x1_conv" not in (conv3x3 + deconv2x2 + [main_out, aux_out])
        print("[self-test] classification OK (decoys correctly ignored)")

        if antstorch_src is None:
            antstorch_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        create_unet_model_path = os.path.join(antstorch_src, "antstorch", "architectures", "create_unet_model.py")
        if not os.path.exists(create_unet_model_path):
            print(f"[self-test] antstorch source not found at {create_unet_model_path} -- "
                  "skipping the model-build/load/verify stage. Re-run with "
                  "--antstorch-src pointing at a real checkout to test that stage too.")
            return
        ccu = _load_create_unet_model(antstorch_src)
        model = build_model(ccu)
        convert_harvard_oxford_atlas_labeling(h5_path, model, verbose=True)

        out_path = os.path.join(tmp, "HarvardOxfordAtlasSubcortical_pytorch.pt")
        x = torch.randn(1, 1, *CROPPED_TEMPLATE_SIZE)
        verify_and_save_multihead(model, out_path, lambda: build_model(ccu), x)
        print("[self-test] ALL OK -- load/forward/reload round-trip succeeded on synthetic data")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights-dir", help="Directory with HarvardOxfordAtlasSubcortical.h5 (e.g. ~/.keras/ANTsXNet)")
    p.add_argument("--out-dir", help="Directory to write HarvardOxfordAtlasSubcortical_pytorch.pt (e.g. ~/.antstorch)")
    p.add_argument("--antstorch-src", help="Path to the ANTsTorch repo root (contains antstorch/architectures/...)")
    p.add_argument("--self-test", action="store_true",
                    help="Run against a synthetic in-memory .h5 instead of a real file -- no other args needed "
                         "except optionally --antstorch-src (to also exercise the build/load/verify stage).")
    args = p.parse_args()

    if args.self_test:
        run_self_test(os.path.expanduser(args.antstorch_src) if args.antstorch_src else None)
        return

    missing = [n for n in ("weights_dir", "out_dir", "antstorch_src") if getattr(args, n) is None]
    if missing:
        p.error(f"--{missing[0].replace('_','-')} is required (unless --self-test)")

    weights_dir = os.path.expanduser(args.weights_dir)
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    antstorch_src = os.path.expanduser(args.antstorch_src)

    h5_path = os.path.join(weights_dir, "HarvardOxfordAtlasSubcortical.h5")
    out_path = os.path.join(out_dir, "HarvardOxfordAtlasSubcortical_pytorch.pt")
    if not os.path.exists(h5_path):
        print(f"[skip] HarvardOxfordAtlasSubcortical: {h5_path} not found")
        return

    ccu = _load_create_unet_model(antstorch_src)

    try:
        model = build_model(ccu)
        convert_harvard_oxford_atlas_labeling(h5_path, model, verbose=True)
        x = torch.randn(1, 1, *CROPPED_TEMPLATE_SIZE)
        verify_and_save_multihead(model, out_path, lambda: build_model(ccu), x)
        print("HarvardOxfordAtlasSubcortical: OK")
    except Exception as e:
        print(f"HarvardOxfordAtlasSubcortical: FAILED: {e!r}")
        raise


if __name__ == "__main__":
    main()