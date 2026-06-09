# Unitree G1 baked viewer mesh

This directory contains a baked static viewer mesh derived from Unitree's
official `unitree_ros` G1 robot description.

Source:
- Repository: `https://github.com/unitreerobotics/unitree_ros`
- Source asset path: `robots/g1_description`
- Source revision used: `ae2b2a7e9ba1b5814b5cb78b3c5800829c5591dd`
- Model variant: `g1_29dof_rev_1_0`

License:
- The source repository is BSD 3-Clause licensed.
- See `LICENSE.unitree_ros`.

The baked `.bmesh` file is a static render asset for RustViewer. It combines
the visual STL meshes from the URDF/MJCF hierarchy, applies the fixed body
transforms, and converts coordinates into RustViewer's scene convention. The
mesh keeps the original STL triangles so the official model does not appear
fragmented or hole-filled from naive face decimation.
