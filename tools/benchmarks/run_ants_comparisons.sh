#!/usr/bin/env bash
# Run the ANTs vs ANTsTorch primitive comparisons on one device:
#   denoise_image, n4_bias_field_correction, cortical_thickness
#
# Usage (from anywhere):
<<<<<<< HEAD
#   tools/benchmarks/run_ants_comparisons.sh [image.nii.gz]
#
# The optional image (or IMAGE=...) is used by all three comparisons. Without
# it, each script uses its own default: r16 for denoise_image and N4,
# S_template3 for cortical_thickness. cortical_thickness needs a 3-D T1.
#
# Environment overrides:
#   IMAGE=path             Input image shared by the three comparisons
=======
#   tools/benchmarks/run_ants_comparisons.sh
#
# Environment overrides:
>>>>>>> 2a4eac54389193467a5b83055848f05ae22db7f3
#   DEVICE=cuda:0          PyTorch device (default: cuda:0)
#   OUT_DIR=path           Output root (default: results/ants_comparisons_<date>_<time>)
#   PYTHON=python          Python interpreter
#   ANTS_THREADS=N         ITK threads for ANTs denoising (default: ITK default)
#   REPEATS=3              Timed runs for denoise_image (default: 3)
#
# Each comparison writes to its own subdirectory with a log file. A failing
# comparison does not stop the others; the exit status is nonzero if any failed.

set -uo pipefail

<<<<<<< HEAD
# Resolve the image before changing directory so relative paths still work.
IMAGE="${1:-${IMAGE:-}}"
if [[ -n "${IMAGE}" ]]; then
    if [[ ! -f "${IMAGE}" ]]; then
        echo "Image not found: ${IMAGE}" >&2
        exit 1
    fi
    IMAGE="$(cd "$(dirname "${IMAGE}")" && pwd)/$(basename "${IMAGE}")"
fi
IMAGE_ARGS=()
[[ -n "${IMAGE}" ]] && IMAGE_ARGS=("${IMAGE}")

=======
>>>>>>> 2a4eac54389193467a5b83055848f05ae22db7f3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-python}"
REPEATS="${REPEATS:-3}"
OUT_DIR="${OUT_DIR:-results/ants_comparisons_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_DIR}"

# Fail fast if the requested CUDA device is not usable.
if [[ "${DEVICE}" == cuda* ]]; then
    if ! "${PYTHON}" - "${DEVICE}" <<'PY'
import sys, torch
device = torch.device(sys.argv[1])
if not torch.cuda.is_available():
    sys.exit("CUDA is not available to PyTorch")
index = device.index or 0
if index >= torch.cuda.device_count():
    sys.exit(f"{device} requested, but only {torch.cuda.device_count()} CUDA device(s) found")
print(f"Using {device}: {torch.cuda.get_device_name(index)} (torch {torch.__version__}, CUDA {torch.version.cuda})")
PY
    then
        exit 1
    fi
fi

{
    echo "date:   $(date -Iseconds)"
    echo "host:   $(hostname)"
    echo "device: ${DEVICE}"
<<<<<<< HEAD
    echo "image:  ${IMAGE:-script defaults (r16 / S_template3)}"
=======
>>>>>>> 2a4eac54389193467a5b83055848f05ae22db7f3
    echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo ' (modified)')"
} | tee "${OUT_DIR}/run_info.txt"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv >> "${OUT_DIR}/run_info.txt"

declare -A STATUS
declare -A SECONDS_USED

run() {
    local name="$1"; shift
    local dir="${OUT_DIR}/${name}"
    mkdir -p "${dir}"
    echo
    echo "=================================================================="
    echo " ${name}  (${DEVICE})"
    echo "=================================================================="
    local start=${SECONDS}
    "$@" 2>&1 | tee "${dir}/log.txt"
    local rc=${PIPESTATUS[0]}
    SECONDS_USED[${name}]=$(( SECONDS - start ))
    if (( rc == 0 )); then STATUS[${name}]="OK"; else STATUS[${name}]="FAILED (exit ${rc})"; fi
}

DENOISE_ARGS=(--device "${DEVICE}" --repeats "${REPEATS}" --output-dir "${OUT_DIR}/denoise_image")
[[ -n "${ANTS_THREADS:-}" ]] && DENOISE_ARGS+=(--ants-threads "${ANTS_THREADS}")

run denoise_image \
<<<<<<< HEAD
    "${PYTHON}" tools/benchmarks/compare_denoise_image.py ${IMAGE_ARGS[@]+"${IMAGE_ARGS[@]}"} "${DENOISE_ARGS[@]}"

run n4_bias_field_correction \
    "${PYTHON}" tools/benchmarks/compare_n4_bias_field_correction.py ${IMAGE_ARGS[@]+"${IMAGE_ARGS[@]}"} \
=======
    "${PYTHON}" tools/benchmarks/compare_denoise_image.py "${DENOISE_ARGS[@]}"

run n4_bias_field_correction \
    "${PYTHON}" tools/benchmarks/compare_n4_bias_field_correction.py \
>>>>>>> 2a4eac54389193467a5b83055848f05ae22db7f3
        --device "${DEVICE}" \
        --output-dir "${OUT_DIR}/n4_bias_field_correction"

run cortical_thickness \
<<<<<<< HEAD
    "${PYTHON}" tools/benchmarks/compare_cortical_thickness.py ${IMAGE_ARGS[@]+"${IMAGE_ARGS[@]}"} \
=======
    "${PYTHON}" tools/benchmarks/compare_cortical_thickness.py \
>>>>>>> 2a4eac54389193467a5b83055848f05ae22db7f3
        --device "${DEVICE}" \
        --output-dir "${OUT_DIR}/cortical_thickness"

echo
echo "Summary (${OUT_DIR})" | tee "${OUT_DIR}/summary.txt"
failed=0
for name in denoise_image n4_bias_field_correction cortical_thickness; do
    printf "  %-26s %-18s %5ss\n" "${name}" "${STATUS[${name}]}" "${SECONDS_USED[${name}]}" | tee -a "${OUT_DIR}/summary.txt"
    [[ "${STATUS[${name}]}" == OK ]] || failed=1
done
exit ${failed}
