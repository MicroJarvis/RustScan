struct Uniforms {
    num_intersections: u32,
    num_tiles: u32,
    pad: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> tile_id_from_isect: array<u32>;
@group(0) @binding(1) var<storage, read_write> tile_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> uniforms: Uniforms;
@group(0) @binding(3) var<storage, read> logical_intersections: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let num_intersections = logical_intersections[0];
    if index >= num_intersections {
        return;
    }

    let tile_id = tile_id_from_isect[index];
    if tile_id >= uniforms.num_tiles {
        return;
    }

    if index == 0u || tile_id != tile_id_from_isect[index - 1u] {
        tile_offsets[tile_id * 2u] = index;
    }

    if index + 1u == num_intersections || tile_id != tile_id_from_isect[index + 1u] {
        tile_offsets[tile_id * 2u + 1u] = index + 1u;
    }
}
