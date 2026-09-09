#!/usr/bin/env python3
"""Standalone release-mode gates; no production FFI or workspace API added.

Run from any directory: python3 RustSFM/tools/test_vlfeat_native.py [--sanitize]
CC and VLFEAT_ROOT can select the same compiler/source tree as Cargo. The
frozen scalar oracle must not be regenerated when evaluating a candidate.
"""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitize", action="store_true", help="enable AddressSanitizer")
    parser.add_argument("--ubsan", action="store_true", help="also check upstream undefined behavior")
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent
    root = tools.parents[1]
    vlfeat = Path(os.environ.get("VLFEAT_ROOT", root / "third_party/vlfeat")).resolve()
    cc = shlex.split(os.environ.get("CC", "clang"))
    flags = ["-O3", "-DNDEBUG", "-DVL_DISABLE_AVX", "-DVL_DISABLE_SSE2",
             "-DVL_DISABLE_OPENMP", "-I", str(vlfeat)]
    if args.sanitize:
        flags += ["-fsanitize=address", "-fno-omit-frame-pointer"]
    if args.ubsan:
        flags += ["-fsanitize=undefined"]
    env = dict(os.environ, VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", UBSAN_OPTIONS="halt_on_error=1")

    def run(command, **kwargs):
        return subprocess.run(command, env=env, timeout=90, check=True, **kwargs)

    run(cc + ["--version"])
    with tempfile.TemporaryDirectory(prefix="rustsfm-vlfeat-") as tmp:
        tmp = Path(tmp)
        objects = []
        for source in ["generic", "host", "mathop", "imopv", "sift", "scalespace",
                       "stringop", "random", "covdet"]:
            obj = tmp / (source + ".o")
            run(cc + flags + ["-c", str(vlfeat / (source + ".c")), "-o", str(obj)])
            objects.append(str(obj))
        for name, sources in [
            ("imconv", [tools / "vlfeat_imconv_test.c", tools / "vlfeat_imconv_scalar.c"]),
            ("covdet", [tools / "vlfeat_covdet_alloc_test.c"]),
        ]:
            exe = tmp / name
            run(cc + flags + list(map(str, sources)) + objects + ["-lpthread", "-lm", "-o", str(exe)])
            run([str(exe)])

        # Negative control: prove the allocation gate catches precisely the
        # original missing free, without ever modifying the workspace bridge.
        bridge = tools.parent / "src/native/vlfeat_sift.c"
        text = bridge.read_text()
        fixed = "    free(patch);\n    free(patch_xy);\n    free(scaled_descriptors);\n    vl_sift_delete(sift);"
        if text.count(fixed) != 1:
            raise RuntimeError("success cleanup changed; review negative control")
        broken = tmp / "bridge.c"
        broken.write_text(text.replace(fixed, fixed.replace("    free(patch_xy);\n", "")))
        harness = tmp / "negative.c"
        harness.write_text((tools / "vlfeat_covdet_alloc_test.c").read_text().replace(
            '#include "../src/native/vlfeat_sift.c"', '#include "bridge.c"'))
        exe = tmp / "negative"
        run(cc + flags + ["-I", str(bridge.parent), str(harness)] + objects +
            ["-lpthread", "-lm", "-o", str(exe)])
        result = subprocess.run([str(exe)], env=env, timeout=90, capture_output=True, text=True)
        if result.returncode != 1 or "live=1" not in result.stderr:
            raise RuntimeError(f"negative control failed: {result.returncode}: {result.stderr}")
        print("PASS: removed-free negative control detected: " + result.stderr.strip())


if __name__ == "__main__":
    main()
