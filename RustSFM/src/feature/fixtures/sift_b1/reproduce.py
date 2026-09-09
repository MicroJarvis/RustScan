#!/usr/bin/env python3
"""Verify byte-for-byte provenance; --write explicitly restores the frozen subsets."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="original frozen corpus directory")
    parser.add_argument("--write", action="store_true", help="restore fixture binaries")
    args = parser.parse_args()
    destination = Path(__file__).resolve().parent
    provenance = json.loads((destination / "provenance.json").read_text())
    for source in provenance["sources"]:
        original = (args.corpus / source["file"]).read_bytes()
        if len(original) != source["rows"] * 128:
            raise ValueError(
                f"{source['file']}: source size mismatch: expected "
                f"{source['rows'] * 128} bytes, got {len(original)}"
            )
        if hashlib.sha256(original).hexdigest() != source["source_corpus_sha256"]:
            raise ValueError(f"{source['file']}: source SHA-256 does not match provenance")
        indices = source["selected_rows"]
        if indices != sorted(set(indices)):
            raise ValueError(f"{source['file']}: selected rows must be sorted and unique")
        if not all(0 <= i < source["rows"] for i in indices):
            raise ValueError(
                f"{source['file']}: selected rows must be in range [0, {source['rows']})"
            )
        subset = b"".join(original[i * 128 : (i + 1) * 128] for i in indices)
        if len(subset) != source["fixture_rows"] * 128:
            raise ValueError(
                f"{source['file']}: subset size mismatch: expected "
                f"{source['fixture_rows'] * 128} bytes, got {len(subset)}"
            )
        if hashlib.sha256(subset).hexdigest() != source["fixture_sha256"]:
            raise ValueError(f"{source['file']}: subset SHA-256 does not match provenance")
        target = destination / source["file"]
        if args.write:
            target.write_bytes(subset)
        if target.read_bytes() != subset:
            raise ValueError(f"{target}: fixture bytes do not match the selected source rows")
        print(f"{source['file']}: {len(indices)} rows; exact source bytes and SHA-256 verified")


if __name__ == "__main__":
    main()
