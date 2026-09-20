# flowers2_colmap reference sparse text (2026-06-30)

Pinned COLMAP sparse **text** export used by
`scripts/provision_flowers2_colmap_fixture.sh` to populate the external
`artifacts/inputs/flowers2_colmap/sparse/text` tree for ignored
`real_colmap_sparse_*` RustSFM tests.

Content is verified by SHA-256 in the provision script. Do not edit these files
in place; replace the whole directory and update the digests together.

Default provision (from workspace root):

```bash
./scripts/provision_flowers2_colmap_fixture.sh
cargo test -p rustsfm --lib -- --ignored
```
