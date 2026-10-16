#!/bin/bash
# Every family that does NOT set ATTN_BLOCK_ONLY must compile attn.h byte for byte
# (OPEN-ATTN-CONTEXT: a family joins the path by measurement, and everything else
# compiles what it compiled). Prove it at the source level: preprocess each attention
# translation unit at a family's own flags with a reference ref's sources and with the
# working tree's, and compare token streams. Identical tokens = identical objects, the
# same evidence a --check re-export gives for one tenth of the time.
#
# Run from Git Bash on Windows or from WSL/Linux, with the mlir-aie environment active
# (its python resolves Peano and the AIE headers):
#
#   bash utilities/attn_pp_identical.sh [git ref]      # default: main
#   ALL_SPECS=1 bash utilities/attn_pp_identical.sh    # also every recipes/specs/*.json at its own flags
#
# PEANO / HDR in the environment override the python lookup. OLD_DIR=<an open_kernels tree>
# takes the reference sources from that directory instead of `git show` (WSL cannot open a
# worktree whose .git names a Windows path).
#
# Add a family by adding its ATTN_* flags below -- they are recipes/attnknobs.py's
# geometry for that spec -- or run with ALL_SPECS=1 (utilities/attn_flags.py).
set -e
REF=${1:-main}
W=$(cd "$(dirname "$0")/.." && { pwd -W 2>/dev/null || pwd; })
if [ -z "$OLD_DIR" ]; then
  git -C "$W" rev-parse --verify -q "$REF^{commit}" > /dev/null || { echo "no such ref: $REF"; exit 2; }
fi
if [ -z "$PEANO" ] || [ -z "$HDR" ]; then
  eval "$(${PYTHON:-python} - <<'PY'
from aie.utils import config
print(f'PEANO="{config.peano_install_dir()}"')
print(f'HDR="{config.cxx_header_path()}"')
PY
)"
fi
PEANO=${PEANO//\\//}; HDR=${HDR//\\//}
OUT=$W/.claude/tmp/ppcheck
rm -rf "$OUT"; mkdir -p "$OUT/old/attn" "$OUT/old/include"
if [ -n "$OLD_DIR" ]; then
  cp "$OLD_DIR"/designs/attn/* "$OUT/old/attn/"
  cp "$OLD_DIR"/include/* "$OUT/old/include/"
  REF=$OLD_DIR
else
  for f in $(git -C "$W" ls-tree --name-only "$REF" open_kernels/designs/attn/); do
    git -C "$W" show "$REF:$f" > "$OUT/old/attn/$(basename "$f")"
  done
  for f in $(git -C "$W" ls-tree --name-only "$REF" open_kernels/include/); do
    git -C "$W" show "$REF:$f" > "$OUT/old/include/$(basename "$f")"
  done
fi
CLANG="$PEANO/bin/clang++"
base() { echo "-I$HDR -I$HDR/aie_kernels -I$HDR/aie_kernels/aie2p -I$1 -D__AIE_API_AIE_ADF_HPP__ --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG -Wno-macro-redefined"; }

# Gemma3-4B: head dim 256, no gate, RB 2 -- keeps the single-row kernel at a blocked RB.
G3="-DATTN_NH=8 -DATTN_KVH=4 -DATTN_HD=256 -DATTN_ROT=128 -DATTN_GATE=0 -DATTN_VEXP=1 -DATTN_NHL=2 -DATTN_RB=2 -DATTN_QKNORM=1"
# Qwen3-4B: head dim 128, RB 4 -- a family not at head dim 256.
Q3="-DATTN_NH=32 -DATTN_KVH=8 -DATTN_HD=128 -DATTN_ROT=128 -DATTN_GATE=0 -DATTN_VEXP=1 -DATTN_NHL=8 -DATTN_RB=4 -DATTN_QKNORM=1"
# Qwen3.5-9B: the 35B's design, still on the single-row kernel.
Q35="-DATTN_NH=16 -DATTN_KVH=4 -DATTN_HD=256 -DATTN_ROT=64 -DATTN_GATE=1 -DATTN_VEXP=1 -DATTN_NHL=4"
# The 35B's previous flags: RB 1, the single-row path.
Q36RB1="-DATTN_NH=16 -DATTN_KVH=2 -DATTN_HD=256 -DATTN_ROT=64 -DATTN_GATE=1 -DATTN_VEXP=1 -DATTN_NHL=4"

SETS=("gemma3-4b|$G3" "qwen3-4b|$Q3" "qwen35-9b|$Q35" "qwen36-rb1|$Q36RB1")
if [ -n "$ALL_SPECS" ]; then
  declare -A SEEN
  while IFS= read -r l; do
    tag=${l%%|*}
    stem=${tag%%@*}
    # a spec the reference tree does not have is a new point: nothing to compare against
    if [ -n "$OLD_DIR" ]; then [ -f "$OLD_DIR/recipes/specs/$stem.json" ] || continue
    else git -C "$W" cat-file -e "$REF:open_kernels/recipes/specs/$stem.json" 2>/dev/null || continue; fi
    [ -n "${SEEN[${l#*|}]}" ] && { echo "same flags as ${SEEN[${l#*|}]}: $tag"; continue; }
    SEEN[${l#*|}]=$tag
    SETS+=("spec-$l")
  done < <(${PYTHON:-python} "$W/utilities/attn_flags.py")
fi

one() {   # one TU at one flag set: preprocess both trees, compare token streams
  local tag=$1 tu=$2 flags=$3
  "$CLANG" -E -P $(base "$OUT/old/include") -I"$OUT/old/attn" $flags "$OUT/old/attn/$tu.cc" -o "$OUT/$tag.$tu.old.i"
  "$CLANG" -E -P $(base "$W/open_kernels/include") -I"$W/open_kernels/designs/attn" $flags "$W/open_kernels/designs/attn/$tu.cc" -o "$OUT/$tag.$tu.new.i"
  if cmp -s <(tr -s ' \t\r\n' '\n' < "$OUT/$tag.$tu.old.i") <(tr -s ' \t\r\n' '\n' < "$OUT/$tag.$tu.new.i"); then
    echo "identical  $tag  $tu"
  else
    echo "DIFFERENT  $tag  $tu"
  fi
}
export -f one base
export CLANG OUT W HDR
for name in "${SETS[@]}"; do
  tag=${name%%|*}; flags=${name#*|}
  for tu in attn_meta attn_q attn_k attn_v attn_init attn_fin attn_fin_ng attn_step attn_step_new attn_stepb attn_stepb_new; do
    if [ $tu = attn_stepb ]; then case "$flags" in *ATTN_RB=*) ;; *) continue;; esac; fi
    if [ $tu = attn_stepb_new ]; then case "$flags" in *ATTN_BLOCK_ONLY=1*) ;; *) continue;; esac; fi
    printf '%s\t%s\t%s\n' "$tag" "$tu" "$flags"
  done
done | xargs -d '\n' -P "${JOBS:-8}" -n 1 bash -c 'IFS=$(printf "\t") read -r a b c <<< "$1"; one "$a" "$b" "$c"' _ | tee "$OUT/result.txt"
n=$(wc -l < "$OUT/result.txt")
echo "$n TUs compared against $REF"
! grep -q "^DIFFERENT" "$OUT/result.txt"
