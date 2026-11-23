#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash build_opp.sh <ops-transformer checkout> <new build directory> [vendor]" >&2
    exit 2
fi

operator_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
upstream_dir=$(git -C "$1" rev-parse --show-toplevel)
build_dir=$2
vendor=${3:-irs_port}
revision=a44c2d02dd8626a656162228fff63956a07e4ec7
upstream_op=experimental/attention/indexer_region_score
: "${ASCEND_HOME_PATH:?Source the CANN environment first}"
: "${FLASHINFER_CATLASS_ROOT:?Set FLASHINFER_CATLASS_ROOT to the CATLASS checkout}"
catlass_include=$(realpath -- "$FLASHINFER_CATLASS_ROOT/include")
third_party_dir=$(realpath -- "${IRS_THIRD_PARTY_DIR:-$upstream_dir/third_party}")
test -r "$catlass_include/catlass/detail/macros.hpp"
test -d "$third_party_dir"
git -C "$upstream_dir" cat-file -e "$revision:build.sh"
mkdir -- "$build_dir"
build_dir=$(cd -- "$build_dir" && pwd -P)

git -C "$upstream_dir" archive "$revision" | tar -xf - -C "$build_dir" \
    --exclude="$upstream_op/op_kernel" \
    --exclude="$upstream_op/op_host/indexer_region_score_tiling.cpp"
cp -a -- "$operator_dir/CMakeLists.txt" "$operator_dir/op_host" "$operator_dir/op_kernel" \
    "$build_dir/$upstream_op/"
# The upstream OPP builder resolves the device entry directly under op_kernel.
printf '#include "arch35/indexer_region_score.cpp"\n' > "$build_dir/$upstream_op/op_kernel/indexer_region_score.cpp"
# Keep the upstream host source path available to its build glob and contract test.
printf '#include "arch35/indexer_region_score_tiling.cpp"\n' \
    > "$build_dir/$upstream_op/op_host/indexer_region_score_tiling.cpp"
mkdir -p -- "$build_dir/experimental/attention/common"
cp -a -- "$operator_dir/../common/include" "$build_dir/experimental/attention/common/"

export CPLUS_INCLUDE_PATH="$catlass_include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
cd -- "$build_dir"
bash build.sh --experimental --pkg --soc=ascend950 --ops=indexer_region_score \
    --vendor_name="$vendor" --cann_3rd_lib_path="$third_party_dir" "-j${IRS_BUILD_JOBS:-4}" -O3
