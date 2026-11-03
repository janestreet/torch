#!/bin/bash
# This script weakens all undefined torch symbols in an object file *except*
# for aoti_torch_abi_version and hides all defined C++ symbols. The point is to allow
# Python environments to move to a newer torch without forcing the torch_refcounted
# ocaml library to upgrade as well.
#
# Crucially, torch_refcounted will fail with a nice error message if someone
# actually tries to use it when we are linked against the wrong torch version.
# This is possible thanks to aoti_torch_abi_version, which is a strong symbol.
#
# Usage: post-process-wrapper-symbols.sh OBJECT_FILE LIB_DIR [LIB_DIR...]
#
# OBJECT_FILE is the object file whose symbols you want to weaken.
#
# LIB_DIR(s) point to the Torch shared libraries corresponding to the torch
# headers used for building torch_refcounted. We compute the intersection of
# symbols defined in those libraries and symbols defined in OBJECT_FILE to
# determine the set of symbols we want to weaken.
#
# The actual weakening and hiding happen via llvm-objcopy.

set -euo pipefail

obj="$1"
shift

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

# All undefined (mangled) symbols in the object.
nm --undefined-only --format=posix "$obj" | awk '{print $1}' | sort -u \
  > "$tmpdir/undefined"

# The Torch shared libraries we compiled against.
for dir in "$@"; do
  [ -d "$dir" ] || continue
  find -L "$dir" -maxdepth 1 \
    \( -name 'libtorch*.so*' -o -name 'libc10*.so*' \
    -o -name 'libcaffe2*.so*' -o -name 'libshm*.so*' \) \
    -print
done | sort -u > "$tmpdir/torch-libs"

if ! [ -s "$tmpdir/torch-libs" ]; then
  echo "post-process-wrapper-symbols.sh: no Torch libraries found under: $*" >&2
  exit 1
fi

# All dynamic symbols defined by those libraries. Strip symbol-version
# suffixes (name@@VERSION) so the names compare equal to the object's
# unversioned references.
while IFS= read -r lib; do
  nm -D --defined-only --format=posix "$lib" | awk '{print $1}' | sed 's/@.*//'
done < "$tmpdir/torch-libs" | sort -u > "$tmpdir/torch-defined"

# Symbols that must stay strong (never weakened), one exact name per line.
# aoti_torch_abi_version must remain a strong reference so that loading fails
# loudly when the AOTI ABI is absent/incompatible.
cat > "$tmpdir/keep-strong" << 'EOF'
aoti_torch_abi_version
EOF

comm -12 "$tmpdir/undefined" "$tmpdir/torch-defined" > "$tmpdir/torch-undefined"

if [ -s "$tmpdir/torch-undefined" ]; then
  grep -Fxv -f "$tmpdir/keep-strong" "$tmpdir/torch-undefined" \
    > "$tmpdir/weaken" || true
  if [ -s "$tmpdir/weaken" ]; then
    llvm-objcopy --weaken-symbols="$tmpdir/weaken" "$obj"
  fi
fi

# Hide C++ symbols to avoid issues with inline torch symbols. C++ symbols
# always start with the "_Z" prefix on x86 / aarch64 linux. All symbols
# that we actually want to be exported are either native C symbols or are
# C++ symbols with extern "C" linkage.
nm --defined-only --format=posix "$obj" \
  | awk '$1 ~ /^_Z/ {print $1}' \
  | sort -u > "$tmpdir/hide"

if [ -s "$tmpdir/hide" ]; then
  llvm-objcopy --set-symbols-visibility="$tmpdir/hide=hidden" "$obj"
fi

# Check for strong torch symbols
nm --format=posix "$obj" > "$tmpdir/after"
awk '$2 == "U" {print $1}' "$tmpdir/after" | sort -u > "$tmpdir/strong"
if comm -12 "$tmpdir/strong" "$tmpdir/torch-defined" \
  | grep -Fxv -f "$tmpdir/keep-strong"; then
  echo "$obj: unexpected strong Torch references" >&2
  exit 1
fi

# Check for visible C++ symbols
readelf --wide --syms "$obj" | awk -v obj="$obj" '
  $8 ~ /^_Z/ && $5 != "LOCAL" && $7 != "UND" && ($6 == "DEFAULT" || $6 == "PROTECTED") {
    print obj ": visible C++ symbol: " $8 > "/dev/stderr"
    bad = 1
  }
  END {exit bad}
'

# If you are weakening symbols, you should be checking for abi safety
if ! grep -Eq '^torch_refcounted_check_abi_once [TU]( |$)' "$tmpdir/after"; then
  echo "$obj: missing strong reference to torch_refcounted_check_abi_once" >&2
  exit 1
fi
