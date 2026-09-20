//! PLY File Format Support
//!
//! Stanford PLY (Polygon File Format) supports both ASCII and binary formats.
//!
//! Format specification:
//! - Header with element definitions
//! - Vertex list (x, y, z, nx, ny, nz, red, green, blue)
//! - Face list (vertex count + indices)

use crate::core::attrib_soa_kernel::VertexPropertyRef;
use crate::{AttribSoAKernel, FaceHandle, Point3, RustMesh, VPropHandle, VertexHandle};
use crate::{Vec3, Vec4};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// PLY file format variant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlyFormat {
    /// ASCII text format (human-readable)
    Ascii,
    /// Binary little-endian format
    BinaryLittleEndian,
    /// Binary big-endian format
    BinaryBigEndian,
}

/// PLY property type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyPropertyType {
    Char,
    UChar,
    Short,
    UShort,
    Int,
    UInt,
    Float,
    Double,
}

impl PlyPropertyType {
    /// Parse from string
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "char" | "int8" => Some(Self::Char),
            "uchar" | "uint8" => Some(Self::UChar),
            "short" | "int16" => Some(Self::Short),
            "ushort" | "uint16" => Some(Self::UShort),
            "int" | "int32" => Some(Self::Int),
            "uint" | "uint32" => Some(Self::UInt),
            "float" | "float32" => Some(Self::Float),
            "double" | "float64" => Some(Self::Double),
            _ => None,
        }
    }

    /// Size in bytes
    #[allow(dead_code)]
    fn size(&self) -> usize {
        match self {
            Self::Char | Self::UChar => 1,
            Self::Short | Self::UShort => 2,
            Self::Int | Self::UInt | Self::Float => 4,
            Self::Double => 8,
        }
    }
}

/// PLY property definition
#[derive(Debug, Clone)]
struct PlyProperty {
    name: String,
    data_type: PlyPropertyType,
    is_list: bool,
    list_size_type: Option<PlyPropertyType>,
}

/// PLY element definition
#[derive(Debug, Clone)]
struct PlyElement {
    name: String,
    count: usize,
    properties: Vec<PlyProperty>,
}

/// PLY header information
#[derive(Debug)]
struct PlyHeader {
    format: PlyFormat,
    #[allow(dead_code)]
    version: String,
    elements: Vec<PlyElement>,
    comments: Vec<String>,
}

const ATTRIB_VERTEX_COMMENT_PREFIX: &str = "rustmesh_attrib";

#[derive(Debug, Clone)]
enum AttribVertexPropertySchema {
    Float {
        name: String,
    },
    Int {
        name: String,
    },
    Vec3 {
        name: String,
        field_names: [String; 3],
    },
}

impl AttribVertexPropertySchema {
    fn name(&self) -> &str {
        match self {
            Self::Float { name } | Self::Int { name } | Self::Vec3 { name, .. } => name,
        }
    }

    fn comment_text(&self) -> String {
        match self {
            Self::Float { name } => {
                format!("{ATTRIB_VERTEX_COMMENT_PREFIX} vertex float {name}")
            }
            Self::Int { name } => {
                format!("{ATTRIB_VERTEX_COMMENT_PREFIX} vertex int {name}")
            }
            Self::Vec3 { name, field_names } => format!(
                "{ATTRIB_VERTEX_COMMENT_PREFIX} vertex vec3 {} {} {} {}",
                name, field_names[0], field_names[1], field_names[2]
            ),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum AttribVertexPropertyHandle {
    Float(VPropHandle<f32>),
    Int(VPropHandle<i32>),
    Vec3(VPropHandle<Vec3>),
}

#[derive(Debug, Clone)]
struct AttribVertexPropertyBinding {
    schema: AttribVertexPropertySchema,
    handle: AttribVertexPropertyHandle,
}

#[derive(Debug, Clone)]
struct AttribVertexPropertyWrite<'a> {
    schema: AttribVertexPropertySchema,
    values: VertexPropertyRef<'a>,
}

// ============================================================================
// PLY Reading Implementation
// ============================================================================

/// Read mesh from PLY file
pub fn read_ply(path: impl AsRef<Path>) -> io::Result<RustMesh> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // Parse header
    let (header, mut data_reader) = parse_ply_header(reader)?;

    // Read data based on format
    match header.format {
        PlyFormat::Ascii => read_ply_ascii_data(data_reader, &header),
        PlyFormat::BinaryLittleEndian => {
            // For binary, we need to read remaining buffered data first
            read_ply_binary_data_from_buffered(&mut data_reader, &header, false)
        }
        PlyFormat::BinaryBigEndian => {
            read_ply_binary_data_from_buffered(&mut data_reader, &header, true)
        }
    }
}

/// Read AttribSoAKernel from PLY file.
pub fn read_attrib_ply(path: impl AsRef<Path>) -> io::Result<AttribSoAKernel> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let (header, mut data_reader) = parse_ply_header(reader)?;

    match header.format {
        PlyFormat::Ascii => read_attrib_ply_ascii_data(data_reader, &header),
        PlyFormat::BinaryLittleEndian => {
            read_attrib_ply_binary_data_from_buffered(&mut data_reader, &header, false)
        }
        PlyFormat::BinaryBigEndian => {
            read_attrib_ply_binary_data_from_buffered(&mut data_reader, &header, true)
        }
    }
}

/// Parse PLY header
/// Returns the header info and the remaining reader for data
fn parse_ply_header(mut reader: BufReader<File>) -> io::Result<(PlyHeader, BufReader<File>)> {
    let mut line = String::new();

    // First line must be "ply"
    reader.read_line(&mut line)?;
    if !line.trim().eq_ignore_ascii_case("ply") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Not a PLY file: missing 'ply' header",
        ));
    }

    let mut format = PlyFormat::Ascii;
    let mut version = String::new();
    let mut elements: Vec<PlyElement> = Vec::new();
    let mut comments: Vec<String> = Vec::new();
    let mut current_element: Option<PlyElement> = None;

    loop {
        line.clear();
        reader.read_line(&mut line)?;
        let trimmed = line.trim();

        if trimmed.is_empty() {
            continue;
        }

        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0].to_lowercase().as_str() {
            "format" => {
                if parts.len() >= 3 {
                    format = match parts[1].to_lowercase().as_str() {
                        "ascii" => PlyFormat::Ascii,
                        "binary_little_endian" => PlyFormat::BinaryLittleEndian,
                        "binary_big_endian" => PlyFormat::BinaryBigEndian,
                        _ => {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!("Unknown PLY format: {}", parts[1]),
                            ))
                        }
                    };
                    version = parts[2].to_string();
                }
            }
            "element" => {
                // Save previous element
                if let Some(elem) = current_element.take() {
                    elements.push(elem);
                }

                if parts.len() >= 3 {
                    current_element = Some(PlyElement {
                        name: parts[1].to_string(),
                        count: parts[2].parse().unwrap_or(0),
                        properties: Vec::new(),
                    });
                }
            }
            "property" => {
                if let Some(ref mut elem) = current_element {
                    if parts.len() >= 3 {
                        if parts[1].to_lowercase() == "list" {
                            // List property: "property list <size_type> <data_type> <name>"
                            if parts.len() >= 5 {
                                let list_size_type = PlyPropertyType::from_str(parts[2]);
                                let data_type = PlyPropertyType::from_str(parts[3]);
                                if let (Some(lst), Some(dt)) = (list_size_type, data_type) {
                                    elem.properties.push(PlyProperty {
                                        name: parts[4].to_string(),
                                        data_type: dt,
                                        is_list: true,
                                        list_size_type: Some(lst),
                                    });
                                }
                            }
                        } else {
                            // Regular property: "property <type> <name>"
                            if let Some(dt) = PlyPropertyType::from_str(parts[1]) {
                                elem.properties.push(PlyProperty {
                                    name: parts[2].to_string(),
                                    data_type: dt,
                                    is_list: false,
                                    list_size_type: None,
                                });
                            }
                        }
                    }
                }
            }
            "comment" => {
                comments.push(trimmed[7..].to_string());
            }
            "end_header" => {
                // Save last element
                if let Some(elem) = current_element.take() {
                    elements.push(elem);
                }
                break;
            }
            _ => {}
        }
    }

    Ok((
        PlyHeader {
            format,
            version,
            elements,
            comments,
        },
        reader,
    ))
}

/// Build vertex property index map
fn build_vertex_property_map(element: &PlyElement) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    for (i, prop) in element.properties.iter().enumerate() {
        map.insert(prop.name.clone(), i);
    }
    map
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn validate_property_name(name: &str, used_names: &mut HashSet<String>) -> io::Result<()> {
    if name.is_empty() {
        return Err(invalid_input("PLY vertex property names cannot be empty"));
    }
    if name.split_whitespace().count() != 1 {
        return Err(invalid_input(format!(
            "PLY vertex property '{name}' contains whitespace and cannot be serialized"
        )));
    }
    if !used_names.insert(name.to_string()) {
        return Err(invalid_input(format!(
            "PLY vertex property name '{name}' conflicts with another serialized property"
        )));
    }
    Ok(())
}

fn build_attrib_vertex_property_schemas(
    mesh: &AttribSoAKernel,
) -> io::Result<Vec<AttribVertexPropertySchema>> {
    let mut used_names: HashSet<String> = ["x", "y", "z"].into_iter().map(str::to_string).collect();

    if mesh.has_vertex_normals() {
        used_names.extend(["nx", "ny", "nz"].into_iter().map(str::to_string));
    }
    if mesh.has_vertex_colors() {
        used_names.extend(
            ["red", "green", "blue", "alpha"]
                .into_iter()
                .map(str::to_string),
        );
    }

    let mut schemas = Vec::new();
    for prop in mesh.vertex_property_refs() {
        match prop {
            VertexPropertyRef::Float { name, .. } => {
                validate_property_name(name, &mut used_names)?;
                schemas.push(AttribVertexPropertySchema::Float {
                    name: name.to_string(),
                });
            }
            VertexPropertyRef::Int { name, .. } => {
                validate_property_name(name, &mut used_names)?;
                schemas.push(AttribVertexPropertySchema::Int {
                    name: name.to_string(),
                });
            }
            VertexPropertyRef::Vec3 { name, .. } => {
                let field_names = [
                    format!("{name}_x"),
                    format!("{name}_y"),
                    format!("{name}_z"),
                ];
                for field_name in &field_names {
                    validate_property_name(field_name, &mut used_names)?;
                }
                schemas.push(AttribVertexPropertySchema::Vec3 {
                    name: name.to_string(),
                    field_names,
                });
            }
            VertexPropertyRef::Vec2 { name, .. } => {
                return Err(invalid_input(format!(
                    "PLY vertex-property persistence does not support Vec2 property '{name}'"
                )));
            }
            VertexPropertyRef::Vec4 { name, .. } => {
                return Err(invalid_input(format!(
                    "PLY vertex-property persistence does not support Vec4 property '{name}'"
                )));
            }
        }
    }

    Ok(schemas)
}

fn build_attrib_vertex_property_writes(
    mesh: &AttribSoAKernel,
) -> io::Result<Vec<AttribVertexPropertyWrite<'_>>> {
    let schemas = build_attrib_vertex_property_schemas(mesh)?;
    let props = mesh.vertex_property_refs();

    Ok(schemas
        .iter()
        .zip(props.into_iter())
        .map(|(schema, values)| AttribVertexPropertyWrite {
            schema: schema.clone(),
            values,
        })
        .collect())
}

fn parse_attrib_vertex_property_schemas(
    header: &PlyHeader,
    vertex_element: Option<&PlyElement>,
) -> io::Result<Vec<AttribVertexPropertySchema>> {
    let Some(vertex_element) = vertex_element else {
        return Ok(Vec::new());
    };
    let prop_map = build_vertex_property_map(vertex_element);
    let mut seen_names = HashSet::new();
    let mut schemas = Vec::new();

    for comment in &header.comments {
        let parts: Vec<&str> = comment.split_whitespace().collect();
        if parts.first().copied() != Some(ATTRIB_VERTEX_COMMENT_PREFIX) {
            continue;
        }

        if parts.len() < 4 {
            return Err(invalid_data(format!(
                "Malformed RustMesh attribute comment: '{comment}'"
            )));
        }
        if parts[1] != "vertex" {
            continue;
        }

        let schema = match parts[2] {
            "float" if parts.len() == 4 => AttribVertexPropertySchema::Float {
                name: parts[3].to_string(),
            },
            "int" if parts.len() == 4 => AttribVertexPropertySchema::Int {
                name: parts[3].to_string(),
            },
            "vec3" if parts.len() == 7 => AttribVertexPropertySchema::Vec3 {
                name: parts[3].to_string(),
                field_names: [
                    parts[4].to_string(),
                    parts[5].to_string(),
                    parts[6].to_string(),
                ],
            },
            kind => {
                return Err(invalid_data(format!(
                    "Unsupported RustMesh attribute schema '{kind}' in comment '{comment}'"
                )))
            }
        };

        if !seen_names.insert(schema.name().to_string()) {
            return Err(invalid_data(format!(
                "Duplicate RustMesh attribute schema for '{}'",
                schema.name()
            )));
        }

        match &schema {
            AttribVertexPropertySchema::Float { name }
            | AttribVertexPropertySchema::Int { name } => {
                if !prop_map.contains_key(name) {
                    return Err(invalid_data(format!(
                        "PLY vertex property '{name}' referenced by RustMesh metadata is missing"
                    )));
                }
            }
            AttribVertexPropertySchema::Vec3 { name, field_names } => {
                for field_name in field_names {
                    if !prop_map.contains_key(field_name) {
                        return Err(invalid_data(format!(
                            "PLY vertex property '{}' referenced by RustMesh metadata for '{}' is missing",
                            field_name, name
                        )));
                    }
                }
            }
        }

        schemas.push(schema);
    }

    Ok(schemas)
}

fn create_attrib_vertex_property_bindings(
    mesh: &mut AttribSoAKernel,
    schemas: &[AttribVertexPropertySchema],
) -> Vec<AttribVertexPropertyBinding> {
    schemas
        .iter()
        .map(|schema| {
            let handle = match schema {
                AttribVertexPropertySchema::Float { name } => {
                    AttribVertexPropertyHandle::Float(mesh.add_vertex_property::<f32>(name))
                }
                AttribVertexPropertySchema::Int { name } => {
                    AttribVertexPropertyHandle::Int(mesh.add_vertex_property::<i32>(name))
                }
                AttribVertexPropertySchema::Vec3 { name, .. } => {
                    AttribVertexPropertyHandle::Vec3(mesh.add_vertex_property::<Vec3>(name))
                }
            };

            AttribVertexPropertyBinding {
                schema: schema.clone(),
                handle,
            }
        })
        .collect()
}

fn read_ascii_value<T>(parts: &[&str], index: Option<&usize>, default: T) -> T
where
    T: std::str::FromStr + Copy,
{
    index
        .and_then(|&i| parts.get(i).and_then(|s| s.parse::<T>().ok()))
        .unwrap_or(default)
}

fn read_value_buf(
    reader: &mut BufReader<File>,
    ty: PlyPropertyType,
    big_endian: bool,
) -> io::Result<f64> {
    Ok(match ty {
        PlyPropertyType::Char => {
            let mut buf = [0u8; 1];
            reader.read_exact(&mut buf)?;
            buf[0] as i8 as f64
        }
        PlyPropertyType::UChar => {
            let mut buf = [0u8; 1];
            reader.read_exact(&mut buf)?;
            buf[0] as f64
        }
        PlyPropertyType::Short => {
            let mut buf = [0u8; 2];
            reader.read_exact(&mut buf)?;
            let val = if big_endian {
                i16::from_be_bytes(buf)
            } else {
                i16::from_le_bytes(buf)
            };
            val as f64
        }
        PlyPropertyType::UShort => {
            let mut buf = [0u8; 2];
            reader.read_exact(&mut buf)?;
            let val = if big_endian {
                u16::from_be_bytes(buf)
            } else {
                u16::from_le_bytes(buf)
            };
            val as f64
        }
        PlyPropertyType::Int => {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            let val = if big_endian {
                i32::from_be_bytes(buf)
            } else {
                i32::from_le_bytes(buf)
            };
            val as f64
        }
        PlyPropertyType::UInt => {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            let val = if big_endian {
                u32::from_be_bytes(buf)
            } else {
                u32::from_le_bytes(buf)
            };
            val as f64
        }
        PlyPropertyType::Float => {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            let val = if big_endian {
                f32::from_be_bytes(buf)
            } else {
                f32::from_le_bytes(buf)
            };
            val as f64
        }
        PlyPropertyType::Double => {
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf)?;
            if big_endian {
                f64::from_be_bytes(buf)
            } else {
                f64::from_le_bytes(buf)
            }
        }
    })
}

fn orient_halfedge(
    mesh: &AttribSoAKernel,
    halfedge: crate::HalfedgeHandle,
    from: VertexHandle,
    to: VertexHandle,
) -> io::Result<crate::HalfedgeHandle> {
    if mesh.from_vertex_handle(halfedge) == from && mesh.to_vertex_handle(halfedge) == to {
        return Ok(halfedge);
    }

    let opposite = mesh
        .opposite_halfedge_handle(halfedge)
        .ok_or_else(|| invalid_data("Halfedge is missing an opposite handle"))?;
    if mesh.from_vertex_handle(opposite) == from && mesh.to_vertex_handle(opposite) == to {
        Ok(opposite)
    } else {
        Err(invalid_data(
            "Failed to orient face halfedge during PLY import",
        ))
    }
}

fn add_attrib_face(mesh: &mut AttribSoAKernel, vertices: &[VertexHandle]) -> io::Result<()> {
    if vertices.len() < 3 {
        return Ok(());
    }

    let mut halfedges = Vec::with_capacity(vertices.len());
    for i in 0..vertices.len() {
        let from = vertices[i];
        let to = vertices[(i + 1) % vertices.len()];
        let heh = mesh.add_edge(from, to);
        halfedges.push(orient_halfedge(mesh, heh, from, to)?);
    }

    let fh = mesh.add_face(halfedges.first().copied());
    for i in 0..halfedges.len() {
        let curr = halfedges[i];
        let next = halfedges[(i + 1) % halfedges.len()];
        let prev = halfedges[(i + halfedges.len() - 1) % halfedges.len()];

        mesh.set_face_handle(curr, fh);
        mesh.set_next_halfedge_handle(curr, next);
        if let Some(halfedge) = mesh.halfedge_mut(curr) {
            halfedge.prev_halfedge_handle = Some(prev);
        }
    }

    Ok(())
}

fn attrib_face_vertices(mesh: &AttribSoAKernel, fh: FaceHandle) -> io::Result<Vec<VertexHandle>> {
    let start = mesh
        .face_halfedge_handle(fh)
        .ok_or_else(|| invalid_data(format!("Face {} is missing a halfedge handle", fh.idx())))?;
    let mut current = start;
    let mut vertices = Vec::new();

    for _ in 0..mesh.n_halfedges().max(1) {
        let from = mesh.from_vertex_handle(current);
        if !from.is_valid() {
            return Err(invalid_data(format!(
                "Halfedge {} in face {} is missing a valid source vertex",
                current.idx(),
                fh.idx()
            )));
        }
        vertices.push(from);

        current = mesh.next_halfedge_handle(current).ok_or_else(|| {
            invalid_data(format!(
                "Halfedge {} in face {} is missing a next pointer",
                current.idx(),
                fh.idx()
            ))
        })?;

        if current == start {
            return Ok(vertices);
        }
    }

    Err(invalid_data(format!(
        "Face {} traversal did not close while writing PLY",
        fh.idx()
    )))
}

/// Read ASCII PLY data
fn read_ply_ascii_data(mut reader: BufReader<File>, header: &PlyHeader) -> io::Result<RustMesh> {
    let mut mesh = RustMesh::new();

    // Find vertex and face elements
    let vertex_element = header
        .elements
        .iter()
        .find(|e| e.name.to_lowercase() == "vertex");
    let face_element = header
        .elements
        .iter()
        .find(|e| e.name.to_lowercase() == "face");

    // Build property maps
    let vertex_props: Option<HashMap<String, usize>> = vertex_element
        .as_ref()
        .map(|e| build_vertex_property_map(e));

    // Request attributes if present
    let has_normals = vertex_props
        .as_ref()
        .map(|p| p.contains_key("nx") || p.contains_key("normal_x"))
        .unwrap_or(false);
    let has_colors = vertex_props
        .as_ref()
        .map(|p| p.contains_key("red") || p.contains_key("diffuse_red"))
        .unwrap_or(false);

    if has_normals {
        mesh.request_vertex_normals();
    }
    if has_colors {
        mesh.request_vertex_colors();
    }

    // Read vertices
    if let Some(ref elem) = vertex_element {
        let prop_map = vertex_props.as_ref().unwrap();

        for _ in 0..elem.count {
            let mut line = String::new();
            reader.read_line(&mut line)?;
            let parts: Vec<&str> = line.split_whitespace().collect();

            // Parse position
            let x_idx = prop_map.get("x").or_else(|| prop_map.get("pos_x"));
            let y_idx = prop_map.get("y").or_else(|| prop_map.get("pos_y"));
            let z_idx = prop_map.get("z").or_else(|| prop_map.get("pos_z"));

            let x = x_idx
                .and_then(|&i| parts.get(i).and_then(|s| s.parse::<f32>().ok()))
                .unwrap_or(0.0);
            let y = y_idx
                .and_then(|&i| parts.get(i).and_then(|s| s.parse::<f32>().ok()))
                .unwrap_or(0.0);
            let z = z_idx
                .and_then(|&i| parts.get(i).and_then(|s| s.parse::<f32>().ok()))
                .unwrap_or(0.0);

            let vh = mesh.add_vertex(Point3::new(x, y, z));

            // Parse normal
            if has_normals {
                let nx_idx = prop_map.get("nx").or_else(|| prop_map.get("normal_x"));
                let ny_idx = prop_map.get("ny").or_else(|| prop_map.get("normal_y"));
                let nz_idx = prop_map.get("nz").or_else(|| prop_map.get("normal_z"));

                let nx = nx_idx
                    .and_then(|&i| parts.get(i).and_then(|s| s.parse::<f32>().ok()))
                    .unwrap_or(0.0);
                let ny = ny_idx
                    .and_then(|&i| parts.get(i).and_then(|s| s.parse::<f32>().ok()))
                    .unwrap_or(0.0);
                let nz = nz_idx
                    .and_then(|&i| parts.get(i).and_then(|s| s.parse::<f32>().ok()))
                    .unwrap_or(1.0);

                mesh.set_vertex_normal_by_index(vh.idx_usize(), Vec3::new(nx, ny, nz));
            }

            // Parse color
            if has_colors {
                let r_idx = prop_map.get("red").or_else(|| prop_map.get("diffuse_red"));
                let g_idx = prop_map
                    .get("green")
                    .or_else(|| prop_map.get("diffuse_green"));
                let b_idx = prop_map
                    .get("blue")
                    .or_else(|| prop_map.get("diffuse_blue"));
                let a_idx = prop_map
                    .get("alpha")
                    .or_else(|| prop_map.get("diffuse_alpha"));

                let r = r_idx
                    .and_then(|&i| parts.get(i).and_then(|s| s.parse::<u8>().ok()))
                    .unwrap_or(255);
                let g = g_idx
                    .and_then(|&i| parts.get(i).and_then(|s| s.parse::<u8>().ok()))
                    .unwrap_or(255);
                let b = b_idx
                    .and_then(|&i| parts.get(i).and_then(|s| s.parse::<u8>().ok()))
                    .unwrap_or(255);
                let a = a_idx
                    .and_then(|&i| parts.get(i).and_then(|s| s.parse::<u8>().ok()))
                    .unwrap_or(255);

                mesh.set_vertex_color_by_index(
                    vh.idx_usize(),
                    Vec4::new(
                        r as f32 / 255.0,
                        g as f32 / 255.0,
                        b as f32 / 255.0,
                        a as f32 / 255.0,
                    ),
                );
            }
        }
    }

    // Read faces
    if let Some(ref elem) = face_element {
        for _ in 0..elem.count {
            let mut line = String::new();
            reader.read_line(&mut line)?;
            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts.is_empty() {
                continue;
            }

            // First value is the vertex count for this face
            let n_verts: usize = parts[0].parse().unwrap_or(0);
            if n_verts < 3 || n_verts > parts.len() - 1 {
                continue;
            }

            let mut indices = Vec::with_capacity(n_verts);
            for i in 1..=n_verts {
                if let Ok(idx) = parts[i].parse::<u32>() {
                    indices.push(crate::VertexHandle::new(idx));
                }
            }

            if indices.len() >= 3 {
                mesh.add_face(&indices);
            }
        }
    }

    Ok(mesh)
}

fn read_attrib_ply_ascii_data(
    mut reader: BufReader<File>,
    header: &PlyHeader,
) -> io::Result<AttribSoAKernel> {
    let mut mesh = AttribSoAKernel::new();

    let vertex_element = header
        .elements
        .iter()
        .find(|e| e.name.to_lowercase() == "vertex");
    let face_element = header
        .elements
        .iter()
        .find(|e| e.name.to_lowercase() == "face");

    let vertex_props: Option<HashMap<String, usize>> = vertex_element
        .as_ref()
        .map(|e| build_vertex_property_map(e));
    let attrib_schemas = parse_attrib_vertex_property_schemas(header, vertex_element)?;

    let has_normals = vertex_props
        .as_ref()
        .map(|p| p.contains_key("nx") || p.contains_key("normal_x"))
        .unwrap_or(false);
    let has_colors = vertex_props
        .as_ref()
        .map(|p| p.contains_key("red") || p.contains_key("diffuse_red"))
        .unwrap_or(false);

    if has_normals {
        mesh.request_vertex_normals();
    }
    if has_colors {
        mesh.request_vertex_colors();
    }

    let attrib_bindings = create_attrib_vertex_property_bindings(&mut mesh, &attrib_schemas);

    if let Some(ref elem) = vertex_element {
        let prop_map = vertex_props.as_ref().unwrap();

        for _ in 0..elem.count {
            let mut line = String::new();
            reader.read_line(&mut line)?;
            let parts: Vec<&str> = line.split_whitespace().collect();

            let x = read_ascii_value(parts.as_slice(), prop_map.get("x"), 0.0f32);
            let y = read_ascii_value(parts.as_slice(), prop_map.get("y"), 0.0f32);
            let z = read_ascii_value(parts.as_slice(), prop_map.get("z"), 0.0f32);
            let vh = mesh.add_vertex(Point3::new(x, y, z));

            if has_normals {
                let nx = read_ascii_value(
                    parts.as_slice(),
                    prop_map.get("nx").or_else(|| prop_map.get("normal_x")),
                    0.0f32,
                );
                let ny = read_ascii_value(
                    parts.as_slice(),
                    prop_map.get("ny").or_else(|| prop_map.get("normal_y")),
                    0.0f32,
                );
                let nz = read_ascii_value(
                    parts.as_slice(),
                    prop_map.get("nz").or_else(|| prop_map.get("normal_z")),
                    1.0f32,
                );
                mesh.set_vertex_normal(vh, Vec3::new(nx, ny, nz));
            }

            if has_colors {
                let r = read_ascii_value(
                    parts.as_slice(),
                    prop_map.get("red").or_else(|| prop_map.get("diffuse_red")),
                    255u8,
                );
                let g = read_ascii_value(
                    parts.as_slice(),
                    prop_map
                        .get("green")
                        .or_else(|| prop_map.get("diffuse_green")),
                    255u8,
                );
                let b = read_ascii_value(
                    parts.as_slice(),
                    prop_map
                        .get("blue")
                        .or_else(|| prop_map.get("diffuse_blue")),
                    255u8,
                );
                let a = read_ascii_value(
                    parts.as_slice(),
                    prop_map
                        .get("alpha")
                        .or_else(|| prop_map.get("diffuse_alpha")),
                    255u8,
                );
                mesh.set_vertex_color(
                    vh,
                    Vec4::new(
                        r as f32 / 255.0,
                        g as f32 / 255.0,
                        b as f32 / 255.0,
                        a as f32 / 255.0,
                    ),
                );
            }

            for binding in &attrib_bindings {
                match (&binding.schema, binding.handle) {
                    (
                        AttribVertexPropertySchema::Float { name },
                        AttribVertexPropertyHandle::Float(handle),
                    ) => {
                        let value =
                            read_ascii_value(parts.as_slice(), prop_map.get(name.as_str()), 0.0f32);
                        mesh.set_vertex_property(handle, vh, value);
                    }
                    (
                        AttribVertexPropertySchema::Int { name },
                        AttribVertexPropertyHandle::Int(handle),
                    ) => {
                        let value =
                            read_ascii_value(parts.as_slice(), prop_map.get(name.as_str()), 0i32);
                        mesh.set_vertex_property(handle, vh, value);
                    }
                    (
                        AttribVertexPropertySchema::Vec3 { field_names, .. },
                        AttribVertexPropertyHandle::Vec3(handle),
                    ) => {
                        let x = read_ascii_value(
                            parts.as_slice(),
                            prop_map.get(field_names[0].as_str()),
                            0.0f32,
                        );
                        let y = read_ascii_value(
                            parts.as_slice(),
                            prop_map.get(field_names[1].as_str()),
                            0.0f32,
                        );
                        let z = read_ascii_value(
                            parts.as_slice(),
                            prop_map.get(field_names[2].as_str()),
                            0.0f32,
                        );
                        mesh.set_vertex_property(handle, vh, Vec3::new(x, y, z));
                    }
                    _ => unreachable!("attribute schema/handle mismatch"),
                }
            }
        }
    }

    if let Some(ref elem) = face_element {
        for _ in 0..elem.count {
            let mut line = String::new();
            reader.read_line(&mut line)?;
            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts.is_empty() {
                continue;
            }

            let n_verts: usize = parts[0].parse().unwrap_or(0);
            if n_verts < 3 || n_verts > parts.len() - 1 {
                continue;
            }

            let mut indices = Vec::with_capacity(n_verts);
            for i in 1..=n_verts {
                if let Ok(idx) = parts[i].parse::<u32>() {
                    indices.push(VertexHandle::new(idx));
                }
            }

            add_attrib_face(&mut mesh, &indices)?;
        }
    }

    Ok(mesh)
}

/// Read binary PLY data from buffered reader
fn read_ply_binary_data_from_buffered(
    reader: &mut BufReader<File>,
    header: &PlyHeader,
    big_endian: bool,
) -> io::Result<RustMesh> {
    let mut mesh = RustMesh::new();

    // Find vertex and face elements
    let vertex_element = header
        .elements
        .iter()
        .find(|e| e.name.to_lowercase() == "vertex");
    let face_element = header
        .elements
        .iter()
        .find(|e| e.name.to_lowercase() == "face");

    // Build property maps
    let vertex_props: Option<HashMap<String, usize>> = vertex_element
        .as_ref()
        .map(|e| build_vertex_property_map(e));

    // Request attributes if present
    let has_normals = vertex_props
        .as_ref()
        .map(|p| p.contains_key("nx") || p.contains_key("normal_x"))
        .unwrap_or(false);
    let has_colors = vertex_props
        .as_ref()
        .map(|p| p.contains_key("red") || p.contains_key("diffuse_red"))
        .unwrap_or(false);

    if has_normals {
        mesh.request_vertex_normals();
    }
    if has_colors {
        mesh.request_vertex_colors();
    }

    // Read vertices
    if let Some(ref elem) = vertex_element {
        let prop_map = vertex_props.as_ref().unwrap();

        for _ in 0..elem.count {
            let mut values = Vec::with_capacity(elem.properties.len());

            for prop in &elem.properties {
                let val = read_value_buf(reader, prop.data_type, big_endian)?;
                values.push(val);
            }

            // Parse position
            let x_idx = prop_map.get("x").or_else(|| prop_map.get("pos_x"));
            let y_idx = prop_map.get("y").or_else(|| prop_map.get("pos_y"));
            let z_idx = prop_map.get("z").or_else(|| prop_map.get("pos_z"));

            let x = x_idx.and_then(|&i| values.get(i)).copied().unwrap_or(0.0) as f32;
            let y = y_idx.and_then(|&i| values.get(i)).copied().unwrap_or(0.0) as f32;
            let z = z_idx.and_then(|&i| values.get(i)).copied().unwrap_or(0.0) as f32;

            let vh = mesh.add_vertex(Point3::new(x, y, z));

            // Parse normal
            if has_normals {
                let nx_idx = prop_map.get("nx").or_else(|| prop_map.get("normal_x"));
                let ny_idx = prop_map.get("ny").or_else(|| prop_map.get("normal_y"));
                let nz_idx = prop_map.get("nz").or_else(|| prop_map.get("normal_z"));

                let nx = nx_idx.and_then(|&i| values.get(i)).copied().unwrap_or(0.0) as f32;
                let ny = ny_idx.and_then(|&i| values.get(i)).copied().unwrap_or(0.0) as f32;
                let nz = nz_idx.and_then(|&i| values.get(i)).copied().unwrap_or(1.0) as f32;

                mesh.set_vertex_normal_by_index(vh.idx_usize(), Vec3::new(nx, ny, nz));
            }

            // Parse color
            if has_colors {
                let r_idx = prop_map.get("red").or_else(|| prop_map.get("diffuse_red"));
                let g_idx = prop_map
                    .get("green")
                    .or_else(|| prop_map.get("diffuse_green"));
                let b_idx = prop_map
                    .get("blue")
                    .or_else(|| prop_map.get("diffuse_blue"));
                let a_idx = prop_map
                    .get("alpha")
                    .or_else(|| prop_map.get("diffuse_alpha"));

                let r = r_idx.and_then(|&i| values.get(i)).copied().unwrap_or(255.0) as f32 / 255.0;
                let g = g_idx.and_then(|&i| values.get(i)).copied().unwrap_or(255.0) as f32 / 255.0;
                let b = b_idx.and_then(|&i| values.get(i)).copied().unwrap_or(255.0) as f32 / 255.0;
                let a = a_idx.and_then(|&i| values.get(i)).copied().unwrap_or(255.0) as f32 / 255.0;

                mesh.set_vertex_color_by_index(vh.idx_usize(), Vec4::new(r, g, b, a));
            }
        }
    }

    // Read faces
    if let Some(ref elem) = face_element {
        for _ in 0..elem.count {
            // For faces, we need to handle list properties
            // First read the list size
            let list_prop = elem.properties.iter().find(|p| p.is_list);
            if let Some(prop) = list_prop {
                // Read list size
                let size_type = prop.list_size_type.unwrap_or(PlyPropertyType::UChar);
                let n_verts = read_value_buf(reader, size_type, big_endian)? as usize;

                // Read indices
                let mut indices = Vec::with_capacity(n_verts);
                for _ in 0..n_verts {
                    let idx = read_value_buf(reader, prop.data_type, big_endian)? as u32;
                    indices.push(crate::VertexHandle::new(idx));
                }

                if indices.len() >= 3 {
                    mesh.add_face(&indices);
                }
            }
        }
    }

    Ok(mesh)
}

fn read_attrib_ply_binary_data_from_buffered(
    reader: &mut BufReader<File>,
    header: &PlyHeader,
    big_endian: bool,
) -> io::Result<AttribSoAKernel> {
    let mut mesh = AttribSoAKernel::new();

    let vertex_element = header
        .elements
        .iter()
        .find(|e| e.name.to_lowercase() == "vertex");
    let face_element = header
        .elements
        .iter()
        .find(|e| e.name.to_lowercase() == "face");

    let vertex_props: Option<HashMap<String, usize>> = vertex_element
        .as_ref()
        .map(|e| build_vertex_property_map(e));
    let attrib_schemas = parse_attrib_vertex_property_schemas(header, vertex_element)?;

    let has_normals = vertex_props
        .as_ref()
        .map(|p| p.contains_key("nx") || p.contains_key("normal_x"))
        .unwrap_or(false);
    let has_colors = vertex_props
        .as_ref()
        .map(|p| p.contains_key("red") || p.contains_key("diffuse_red"))
        .unwrap_or(false);

    if has_normals {
        mesh.request_vertex_normals();
    }
    if has_colors {
        mesh.request_vertex_colors();
    }

    let attrib_bindings = create_attrib_vertex_property_bindings(&mut mesh, &attrib_schemas);

    if let Some(ref elem) = vertex_element {
        let prop_map = vertex_props.as_ref().unwrap();

        for _ in 0..elem.count {
            let mut values = Vec::with_capacity(elem.properties.len());
            for prop in &elem.properties {
                let val = read_value_buf(reader, prop.data_type, big_endian)?;
                values.push(val);
            }

            let x = values[prop_map["x"]] as f32;
            let y = values[prop_map["y"]] as f32;
            let z = values[prop_map["z"]] as f32;
            let vh = mesh.add_vertex(Point3::new(x, y, z));

            if has_normals {
                let nx_idx = prop_map.get("nx").or_else(|| prop_map.get("normal_x"));
                let ny_idx = prop_map.get("ny").or_else(|| prop_map.get("normal_y"));
                let nz_idx = prop_map.get("nz").or_else(|| prop_map.get("normal_z"));

                let nx = nx_idx.and_then(|&i| values.get(i)).copied().unwrap_or(0.0) as f32;
                let ny = ny_idx.and_then(|&i| values.get(i)).copied().unwrap_or(0.0) as f32;
                let nz = nz_idx.and_then(|&i| values.get(i)).copied().unwrap_or(1.0) as f32;
                mesh.set_vertex_normal(vh, Vec3::new(nx, ny, nz));
            }

            if has_colors {
                let r_idx = prop_map.get("red").or_else(|| prop_map.get("diffuse_red"));
                let g_idx = prop_map
                    .get("green")
                    .or_else(|| prop_map.get("diffuse_green"));
                let b_idx = prop_map
                    .get("blue")
                    .or_else(|| prop_map.get("diffuse_blue"));
                let a_idx = prop_map
                    .get("alpha")
                    .or_else(|| prop_map.get("diffuse_alpha"));

                let r = r_idx.and_then(|&i| values.get(i)).copied().unwrap_or(255.0) as f32 / 255.0;
                let g = g_idx.and_then(|&i| values.get(i)).copied().unwrap_or(255.0) as f32 / 255.0;
                let b = b_idx.and_then(|&i| values.get(i)).copied().unwrap_or(255.0) as f32 / 255.0;
                let a = a_idx.and_then(|&i| values.get(i)).copied().unwrap_or(255.0) as f32 / 255.0;

                mesh.set_vertex_color(vh, Vec4::new(r, g, b, a));
            }

            for binding in &attrib_bindings {
                match (&binding.schema, binding.handle) {
                    (
                        AttribVertexPropertySchema::Float { name },
                        AttribVertexPropertyHandle::Float(handle),
                    ) => {
                        let value = prop_map
                            .get(name)
                            .and_then(|&i| values.get(i))
                            .copied()
                            .unwrap_or(0.0) as f32;
                        mesh.set_vertex_property(handle, vh, value);
                    }
                    (
                        AttribVertexPropertySchema::Int { name },
                        AttribVertexPropertyHandle::Int(handle),
                    ) => {
                        let value = prop_map
                            .get(name)
                            .and_then(|&i| values.get(i))
                            .copied()
                            .unwrap_or(0.0) as i32;
                        mesh.set_vertex_property(handle, vh, value);
                    }
                    (
                        AttribVertexPropertySchema::Vec3 { field_names, .. },
                        AttribVertexPropertyHandle::Vec3(handle),
                    ) => {
                        let x = prop_map
                            .get(field_names[0].as_str())
                            .and_then(|&i| values.get(i))
                            .copied()
                            .unwrap_or(0.0) as f32;
                        let y = prop_map
                            .get(field_names[1].as_str())
                            .and_then(|&i| values.get(i))
                            .copied()
                            .unwrap_or(0.0) as f32;
                        let z = prop_map
                            .get(field_names[2].as_str())
                            .and_then(|&i| values.get(i))
                            .copied()
                            .unwrap_or(0.0) as f32;
                        mesh.set_vertex_property(handle, vh, Vec3::new(x, y, z));
                    }
                    _ => unreachable!("attribute schema/handle mismatch"),
                }
            }
        }
    }

    if let Some(ref elem) = face_element {
        for _ in 0..elem.count {
            let list_prop = elem.properties.iter().find(|p| p.is_list);
            if let Some(prop) = list_prop {
                let size_type = prop.list_size_type.unwrap_or(PlyPropertyType::UChar);
                let n_verts = read_value_buf(reader, size_type, big_endian)? as usize;

                let mut indices = Vec::with_capacity(n_verts);
                for _ in 0..n_verts {
                    let idx = read_value_buf(reader, prop.data_type, big_endian)? as u32;
                    indices.push(VertexHandle::new(idx));
                }

                add_attrib_face(&mut mesh, &indices)?;
            }
        }
    }

    Ok(mesh)
}

// ============================================================================
// PLY Writing Implementation
// ============================================================================

/// Write mesh to PLY file
pub fn write_ply(mesh: &RustMesh, path: impl AsRef<Path>, format: PlyFormat) -> io::Result<()> {
    match format {
        PlyFormat::Ascii => write_ply_ascii(mesh, path),
        PlyFormat::BinaryLittleEndian => write_ply_binary(mesh, path, false),
        PlyFormat::BinaryBigEndian => write_ply_binary(mesh, path, true),
    }
}

/// Write AttribSoAKernel to PLY file.
pub fn write_attrib_ply(
    mesh: &AttribSoAKernel,
    path: impl AsRef<Path>,
    format: PlyFormat,
) -> io::Result<()> {
    match format {
        PlyFormat::Ascii => write_attrib_ply_ascii(mesh, path),
        PlyFormat::BinaryLittleEndian => write_attrib_ply_binary(mesh, path, false),
        PlyFormat::BinaryBigEndian => write_attrib_ply_binary(mesh, path, true),
    }
}

/// Write PLY in ASCII format
fn write_ply_ascii(mesh: &RustMesh, path: impl AsRef<Path>) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let has_normals = mesh.has_vertex_normals();
    let has_colors = mesh.has_vertex_colors();

    // Write header
    writeln!(writer, "ply")?;
    writeln!(writer, "format ascii 1.0")?;
    writeln!(writer, "comment RustMesh PLY Export")?;
    writeln!(writer, "element vertex {}", mesh.n_vertices())?;
    writeln!(writer, "property float x")?;
    writeln!(writer, "property float y")?;
    writeln!(writer, "property float z")?;

    if has_normals {
        writeln!(writer, "property float nx")?;
        writeln!(writer, "property float ny")?;
        writeln!(writer, "property float nz")?;
    }

    if has_colors {
        writeln!(writer, "property uchar red")?;
        writeln!(writer, "property uchar green")?;
        writeln!(writer, "property uchar blue")?;
        writeln!(writer, "property uchar alpha")?;
    }

    writeln!(writer, "element face {}", mesh.n_faces())?;
    writeln!(writer, "property list uchar int vertex_indices")?;
    writeln!(writer, "end_header")?;

    // Write vertices
    for v_idx in 0..mesh.n_vertices() {
        if let Some(point) = mesh.point_by_index(v_idx) {
            write!(writer, "{} {} {}", point.x, point.y, point.z)?;

            if has_normals {
                if let Some(normal) = mesh.vertex_normal_by_index(v_idx) {
                    write!(writer, " {} {} {}", normal.x, normal.y, normal.z)?;
                } else {
                    write!(writer, " 0 0 1")?;
                }
            }

            if has_colors {
                if let Some(color) = mesh.vertex_color_by_index(v_idx) {
                    let r = (color.x.clamp(0.0, 1.0) * 255.0) as u8;
                    let g = (color.y.clamp(0.0, 1.0) * 255.0) as u8;
                    let b = (color.z.clamp(0.0, 1.0) * 255.0) as u8;
                    let a = (color.w.clamp(0.0, 1.0) * 255.0) as u8;
                    write!(writer, " {} {} {} {}", r, g, b, a)?;
                } else {
                    write!(writer, " 255 255 255 255")?;
                }
            }

            writeln!(writer)?;
        }
    }

    // Write faces
    for f_idx in 0..mesh.n_faces() {
        let face_handle = crate::FaceHandle::new(f_idx as u32);
        let vertices = mesh.face_vertices_vec(face_handle);

        if vertices.is_empty() {
            continue;
        }

        write!(writer, "{}", vertices.len())?;
        for vh in vertices {
            write!(writer, " {}", vh.idx_usize())?;
        }
        writeln!(writer)?;
    }

    writer.flush()?;
    Ok(())
}

fn write_attrib_ply_ascii(mesh: &AttribSoAKernel, path: impl AsRef<Path>) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let attrib_props = build_attrib_vertex_property_writes(mesh)?;

    writeln!(writer, "ply")?;
    writeln!(writer, "format ascii 1.0")?;
    writeln!(writer, "comment RustMesh AttribSoAKernel PLY Export")?;
    for prop in &attrib_props {
        writeln!(writer, "comment {}", prop.schema.comment_text())?;
    }
    writeln!(writer, "element vertex {}", mesh.n_vertices())?;
    writeln!(writer, "property float x")?;
    writeln!(writer, "property float y")?;
    writeln!(writer, "property float z")?;

    if mesh.has_vertex_normals() {
        writeln!(writer, "property float nx")?;
        writeln!(writer, "property float ny")?;
        writeln!(writer, "property float nz")?;
    }

    if mesh.has_vertex_colors() {
        writeln!(writer, "property uchar red")?;
        writeln!(writer, "property uchar green")?;
        writeln!(writer, "property uchar blue")?;
        writeln!(writer, "property uchar alpha")?;
    }

    for prop in &attrib_props {
        match &prop.schema {
            AttribVertexPropertySchema::Float { name } => {
                writeln!(writer, "property float {name}")?;
            }
            AttribVertexPropertySchema::Int { name } => {
                writeln!(writer, "property int {name}")?;
            }
            AttribVertexPropertySchema::Vec3 { field_names, .. } => {
                for field_name in field_names {
                    writeln!(writer, "property float {field_name}")?;
                }
            }
        }
    }

    writeln!(writer, "element face {}", mesh.n_faces())?;
    writeln!(writer, "property list uchar int vertex_indices")?;
    writeln!(writer, "end_header")?;

    for v_idx in 0..mesh.n_vertices() {
        let vh = VertexHandle::new(v_idx as u32);
        let point = mesh
            .point(v_idx)
            .ok_or_else(|| invalid_data(format!("Vertex {v_idx} is missing a position")))?;
        write!(writer, "{} {} {}", point.x, point.y, point.z)?;

        if mesh.has_vertex_normals() {
            let normal = mesh.vertex_normal(vh).unwrap_or(Vec3::new(0.0, 0.0, 1.0));
            write!(writer, " {} {} {}", normal.x, normal.y, normal.z)?;
        }

        if mesh.has_vertex_colors() {
            let color = mesh
                .vertex_color(vh)
                .unwrap_or(Vec4::new(1.0, 1.0, 1.0, 1.0));
            let r = (color.x.clamp(0.0, 1.0) * 255.0) as u8;
            let g = (color.y.clamp(0.0, 1.0) * 255.0) as u8;
            let b = (color.z.clamp(0.0, 1.0) * 255.0) as u8;
            let a = (color.w.clamp(0.0, 1.0) * 255.0) as u8;
            write!(writer, " {} {} {} {}", r, g, b, a)?;
        }

        for prop in &attrib_props {
            match prop.values {
                VertexPropertyRef::Float { values, .. } => write!(writer, " {}", values[v_idx])?,
                VertexPropertyRef::Int { values, .. } => write!(writer, " {}", values[v_idx])?,
                VertexPropertyRef::Vec3 { values, .. } => {
                    let value = values[v_idx];
                    write!(writer, " {} {} {}", value.x, value.y, value.z)?;
                }
                VertexPropertyRef::Vec2 { .. } | VertexPropertyRef::Vec4 { .. } => {
                    unreachable!("unsupported property types are rejected earlier")
                }
            }
        }

        writeln!(writer)?;
    }

    for f_idx in 0..mesh.n_faces() {
        let vertices = attrib_face_vertices(mesh, FaceHandle::new(f_idx as u32))?;
        write!(writer, "{}", vertices.len())?;
        for vh in vertices {
            write!(writer, " {}", vh.idx_usize())?;
        }
        writeln!(writer)?;
    }

    writer.flush()?;
    Ok(())
}

/// Write PLY in binary format
fn write_ply_binary(mesh: &RustMesh, path: impl AsRef<Path>, big_endian: bool) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let has_normals = mesh.has_vertex_normals();
    let has_colors = mesh.has_vertex_colors();

    // Write header (always ASCII)
    let format_str = if big_endian {
        "binary_big_endian"
    } else {
        "binary_little_endian"
    };

    writeln!(writer, "ply")?;
    writeln!(writer, "format {} 1.0", format_str)?;
    writeln!(writer, "comment RustMesh PLY Export")?;
    writeln!(writer, "element vertex {}", mesh.n_vertices())?;
    writeln!(writer, "property float x")?;
    writeln!(writer, "property float y")?;
    writeln!(writer, "property float z")?;

    if has_normals {
        writeln!(writer, "property float nx")?;
        writeln!(writer, "property float ny")?;
        writeln!(writer, "property float nz")?;
    }

    if has_colors {
        writeln!(writer, "property uchar red")?;
        writeln!(writer, "property uchar green")?;
        writeln!(writer, "property uchar blue")?;
        writeln!(writer, "property uchar alpha")?;
    }

    writeln!(writer, "element face {}", mesh.n_faces())?;
    writeln!(writer, "property list uchar int vertex_indices")?;
    writeln!(writer, "end_header")?;

    // Write vertices (binary)
    for v_idx in 0..mesh.n_vertices() {
        if let Some(point) = mesh.point_by_index(v_idx) {
            if big_endian {
                writer.write_all(&point.x.to_be_bytes())?;
                writer.write_all(&point.y.to_be_bytes())?;
                writer.write_all(&point.z.to_be_bytes())?;
            } else {
                writer.write_all(&point.x.to_le_bytes())?;
                writer.write_all(&point.y.to_le_bytes())?;
                writer.write_all(&point.z.to_le_bytes())?;
            }

            if has_normals {
                if let Some(normal) = mesh.vertex_normal_by_index(v_idx) {
                    if big_endian {
                        writer.write_all(&normal.x.to_be_bytes())?;
                        writer.write_all(&normal.y.to_be_bytes())?;
                        writer.write_all(&normal.z.to_be_bytes())?;
                    } else {
                        writer.write_all(&normal.x.to_le_bytes())?;
                        writer.write_all(&normal.y.to_le_bytes())?;
                        writer.write_all(&normal.z.to_le_bytes())?;
                    }
                } else {
                    let zeros = [0.0f32; 3];
                    for z in zeros {
                        let bytes = if big_endian {
                            z.to_be_bytes()
                        } else {
                            z.to_le_bytes()
                        };
                        writer.write_all(&bytes)?;
                    }
                }
            }

            if has_colors {
                if let Some(color) = mesh.vertex_color_by_index(v_idx) {
                    let r = (color.x.clamp(0.0, 1.0) * 255.0) as u8;
                    let g = (color.y.clamp(0.0, 1.0) * 255.0) as u8;
                    let b = (color.z.clamp(0.0, 1.0) * 255.0) as u8;
                    let a = (color.w.clamp(0.0, 1.0) * 255.0) as u8;
                    writer.write_all(&[r, g, b, a])?;
                } else {
                    writer.write_all(&[255u8, 255, 255, 255])?;
                }
            }
        }
    }

    // Write faces (binary)
    for f_idx in 0..mesh.n_faces() {
        let face_handle = crate::FaceHandle::new(f_idx as u32);
        let vertices = mesh.face_vertices_vec(face_handle);

        if vertices.is_empty() {
            continue;
        }

        // Write vertex count
        writer.write_all(&[vertices.len() as u8])?;

        // Write indices
        for vh in vertices {
            let idx = vh.idx_usize() as i32;
            if big_endian {
                writer.write_all(&idx.to_be_bytes())?;
            } else {
                writer.write_all(&idx.to_le_bytes())?;
            }
        }
    }

    writer.flush()?;
    Ok(())
}

fn write_attrib_ply_binary(
    mesh: &AttribSoAKernel,
    path: impl AsRef<Path>,
    big_endian: bool,
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let attrib_props = build_attrib_vertex_property_writes(mesh)?;

    let format_str = if big_endian {
        "binary_big_endian"
    } else {
        "binary_little_endian"
    };

    writeln!(writer, "ply")?;
    writeln!(writer, "format {} 1.0", format_str)?;
    writeln!(writer, "comment RustMesh AttribSoAKernel PLY Export")?;
    for prop in &attrib_props {
        writeln!(writer, "comment {}", prop.schema.comment_text())?;
    }
    writeln!(writer, "element vertex {}", mesh.n_vertices())?;
    writeln!(writer, "property float x")?;
    writeln!(writer, "property float y")?;
    writeln!(writer, "property float z")?;

    if mesh.has_vertex_normals() {
        writeln!(writer, "property float nx")?;
        writeln!(writer, "property float ny")?;
        writeln!(writer, "property float nz")?;
    }

    if mesh.has_vertex_colors() {
        writeln!(writer, "property uchar red")?;
        writeln!(writer, "property uchar green")?;
        writeln!(writer, "property uchar blue")?;
        writeln!(writer, "property uchar alpha")?;
    }

    for prop in &attrib_props {
        match &prop.schema {
            AttribVertexPropertySchema::Float { name } => {
                writeln!(writer, "property float {name}")?;
            }
            AttribVertexPropertySchema::Int { name } => {
                writeln!(writer, "property int {name}")?;
            }
            AttribVertexPropertySchema::Vec3 { field_names, .. } => {
                for field_name in field_names {
                    writeln!(writer, "property float {field_name}")?;
                }
            }
        }
    }

    writeln!(writer, "element face {}", mesh.n_faces())?;
    writeln!(writer, "property list uchar int vertex_indices")?;
    writeln!(writer, "end_header")?;

    for v_idx in 0..mesh.n_vertices() {
        let vh = VertexHandle::new(v_idx as u32);
        let point = mesh
            .point(v_idx)
            .ok_or_else(|| invalid_data(format!("Vertex {v_idx} is missing a position")))?;

        let coords = [point.x, point.y, point.z];
        for coord in coords {
            let bytes = if big_endian {
                coord.to_be_bytes()
            } else {
                coord.to_le_bytes()
            };
            writer.write_all(&bytes)?;
        }

        if mesh.has_vertex_normals() {
            let normal = mesh.vertex_normal(vh).unwrap_or(Vec3::new(0.0, 0.0, 1.0));
            for coord in [normal.x, normal.y, normal.z] {
                let bytes = if big_endian {
                    coord.to_be_bytes()
                } else {
                    coord.to_le_bytes()
                };
                writer.write_all(&bytes)?;
            }
        }

        if mesh.has_vertex_colors() {
            let color = mesh
                .vertex_color(vh)
                .unwrap_or(Vec4::new(1.0, 1.0, 1.0, 1.0));
            let rgba = [
                (color.x.clamp(0.0, 1.0) * 255.0) as u8,
                (color.y.clamp(0.0, 1.0) * 255.0) as u8,
                (color.z.clamp(0.0, 1.0) * 255.0) as u8,
                (color.w.clamp(0.0, 1.0) * 255.0) as u8,
            ];
            writer.write_all(&rgba)?;
        }

        for prop in &attrib_props {
            match prop.values {
                VertexPropertyRef::Float { values, .. } => {
                    let bytes = if big_endian {
                        values[v_idx].to_be_bytes()
                    } else {
                        values[v_idx].to_le_bytes()
                    };
                    writer.write_all(&bytes)?;
                }
                VertexPropertyRef::Int { values, .. } => {
                    let bytes = if big_endian {
                        values[v_idx].to_be_bytes()
                    } else {
                        values[v_idx].to_le_bytes()
                    };
                    writer.write_all(&bytes)?;
                }
                VertexPropertyRef::Vec3 { values, .. } => {
                    let value = values[v_idx];
                    for coord in [value.x, value.y, value.z] {
                        let bytes = if big_endian {
                            coord.to_be_bytes()
                        } else {
                            coord.to_le_bytes()
                        };
                        writer.write_all(&bytes)?;
                    }
                }
                VertexPropertyRef::Vec2 { .. } | VertexPropertyRef::Vec4 { .. } => {
                    unreachable!("unsupported property types are rejected earlier")
                }
            }
        }
    }

    for f_idx in 0..mesh.n_faces() {
        let vertices = attrib_face_vertices(mesh, FaceHandle::new(f_idx as u32))?;
        writer.write_all(&[vertices.len() as u8])?;
        for vh in vertices {
            let idx = vh.idx_usize() as i32;
            let bytes = if big_endian {
                idx.to_be_bytes()
            } else {
                idx.to_le_bytes()
            };
            writer.write_all(&bytes)?;
        }
    }

    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_mesh() -> RustMesh {
        let mut mesh = RustMesh::new();
        let v0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(Point3::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(Point3::new(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]);
        mesh
    }

    fn create_attrib_test_mesh() -> AttribSoAKernel {
        let mut mesh = AttribSoAKernel::new();
        let v0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(Point3::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(Point3::new(0.0, 1.0, 0.0));

        mesh.request_vertex_normals();
        mesh.set_vertex_normal(v0, Vec3::new(0.0, 0.0, 1.0));
        mesh.set_vertex_normal(v1, Vec3::new(0.0, 0.0, 1.0));
        mesh.set_vertex_normal(v2, Vec3::new(0.0, 0.0, 1.0));

        mesh.request_vertex_colors();
        mesh.set_vertex_color(v0, Vec4::new(1.0, 0.0, 0.0, 1.0));
        mesh.set_vertex_color(v1, Vec4::new(0.0, 1.0, 0.0, 1.0));
        mesh.set_vertex_color(v2, Vec4::new(0.0, 0.0, 1.0, 1.0));

        let quality = mesh.add_vertex_property::<f32>("quality");
        let region = mesh.add_vertex_property::<i32>("region");
        let gradient = mesh.add_vertex_property::<Vec3>("gradient");

        mesh.set_vertex_property(quality, v0, 1.5);
        mesh.set_vertex_property(quality, v1, 2.5);
        mesh.set_vertex_property(quality, v2, 3.5);

        mesh.set_vertex_property(region, v0, 7);
        mesh.set_vertex_property(region, v1, 8);
        mesh.set_vertex_property(region, v2, 9);

        mesh.set_vertex_property(gradient, v0, Vec3::new(1.0, 2.0, 3.0));
        mesh.set_vertex_property(gradient, v1, Vec3::new(4.0, 5.0, 6.0));
        mesh.set_vertex_property(gradient, v2, Vec3::new(7.0, 8.0, 9.0));

        let e0 = mesh.add_edge(v0, v1);
        let e1 = mesh.add_edge(v1, v2);
        let e2 = mesh.add_edge(v2, v0);
        let f0 = mesh.add_face(Some(e0));

        mesh.set_face_handle(e0, f0);
        mesh.set_face_handle(e1, f0);
        mesh.set_face_handle(e2, f0);
        mesh.set_next_halfedge_handle(e0, e1);
        mesh.set_next_halfedge_handle(e1, e2);
        mesh.set_next_halfedge_handle(e2, e0);
        mesh.halfedge_mut(e0).unwrap().prev_halfedge_handle = Some(e2);
        mesh.halfedge_mut(e1).unwrap().prev_halfedge_handle = Some(e0);
        mesh.halfedge_mut(e2).unwrap().prev_halfedge_handle = Some(e1);

        mesh
    }

    #[test]
    fn test_ply_ascii_write() {
        use std::fs;

        let mesh = create_test_mesh();
        let path = "/tmp/test_mesh.ply";
        write_ply(&mesh, path, PlyFormat::Ascii).unwrap();

        // Verify file exists
        assert!(std::path::Path::new(path).exists());

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_ply_binary_write() {
        use std::fs;

        let mesh = create_test_mesh();
        let path = "/tmp/test_mesh_binary.ply";
        write_ply(&mesh, path, PlyFormat::BinaryLittleEndian).unwrap();

        assert!(std::path::Path::new(path).exists());

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_ply_ascii_roundtrip() {
        use std::fs;

        let original = create_test_mesh();
        let path = "/tmp/test_roundtrip.ply";

        // Write
        write_ply(&original, path, PlyFormat::Ascii).unwrap();

        // Read
        let loaded = read_ply(path).unwrap();

        // Verify
        assert_eq!(loaded.n_vertices(), original.n_vertices());
        assert_eq!(loaded.n_faces(), original.n_faces());

        // Clean up
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_ply_binary_roundtrip() {
        use std::fs;

        let original = create_test_mesh();
        let path = "/tmp/test_roundtrip_binary.ply";

        // Write
        write_ply(&original, path, PlyFormat::BinaryLittleEndian).unwrap();

        // Read
        let loaded = read_ply(path).unwrap();

        // Verify
        assert_eq!(loaded.n_vertices(), original.n_vertices());
        assert_eq!(loaded.n_faces(), original.n_faces());

        // Clean up
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_ply_with_normals() {
        use std::fs;

        let mut mesh = RustMesh::new();
        mesh.request_vertex_normals();

        let v0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(Point3::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(Point3::new(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]);

        mesh.set_vertex_normal_by_index(0, Vec3::new(0.0, 0.0, 1.0));
        mesh.set_vertex_normal_by_index(1, Vec3::new(0.0, 0.0, 1.0));
        mesh.set_vertex_normal_by_index(2, Vec3::new(0.0, 0.0, 1.0));

        let path = "/tmp/test_normals.ply";
        write_ply(&mesh, path, PlyFormat::Ascii).unwrap();

        let loaded = read_ply(path).unwrap();

        assert_eq!(loaded.n_vertices(), 3);
        assert!(loaded.has_vertex_normals());

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_ply_with_colors() {
        use std::fs;

        let mut mesh = RustMesh::new();
        mesh.request_vertex_colors();

        let v0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(Point3::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(Point3::new(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]);

        mesh.set_vertex_color_by_index(0, Vec4::new(1.0, 0.0, 0.0, 1.0));
        mesh.set_vertex_color_by_index(1, Vec4::new(0.0, 1.0, 0.0, 1.0));
        mesh.set_vertex_color_by_index(2, Vec4::new(0.0, 0.0, 1.0, 1.0));

        let path = "/tmp/test_colors.ply";
        write_ply(&mesh, path, PlyFormat::Ascii).unwrap();

        let loaded = read_ply(path).unwrap();

        assert_eq!(loaded.n_vertices(), 3);
        assert!(loaded.has_vertex_colors());

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_attrib_ply_ascii_roundtrip_preserves_supported_vertex_properties() {
        let original = create_attrib_test_mesh();
        let dir = tempdir().unwrap();
        let path = dir.path().join("attrib_ascii_roundtrip.ply");

        write_attrib_ply(&original, &path, PlyFormat::Ascii).unwrap();
        let loaded = read_attrib_ply(&path).unwrap();

        assert_eq!(loaded.n_vertices(), original.n_vertices());
        assert_eq!(loaded.n_faces(), original.n_faces());
        assert!(loaded.has_vertex_normals());
        assert!(loaded.has_vertex_colors());
        assert_eq!(
            loaded.vertex_normal(VertexHandle::new(0)),
            Some(Vec3::new(0.0, 0.0, 1.0))
        );

        let props = loaded.vertex_property_refs();
        assert_eq!(props.len(), 3);
        assert!(matches!(
            props[0],
            VertexPropertyRef::Float { name: "quality", values }
                if values == &[1.5f32, 2.5, 3.5]
        ));
        assert!(matches!(
            props[1],
            VertexPropertyRef::Int { name: "region", values }
                if values == &[7, 8, 9]
        ));
        assert!(matches!(
            props[2],
            VertexPropertyRef::Vec3 { name: "gradient", values }
                if values[0] == Vec3::new(1.0, 2.0, 3.0)
                    && values[1] == Vec3::new(4.0, 5.0, 6.0)
                    && values[2] == Vec3::new(7.0, 8.0, 9.0)
        ));
    }

    #[test]
    fn test_attrib_ply_binary_roundtrip_preserves_schema_and_values() {
        let original = create_attrib_test_mesh();
        let dir = tempdir().unwrap();
        let path = dir.path().join("attrib_binary_roundtrip.ply");

        write_attrib_ply(&original, &path, PlyFormat::BinaryLittleEndian).unwrap();
        let loaded = read_attrib_ply(&path).unwrap();

        assert_eq!(loaded.n_vertices(), 3);
        assert_eq!(loaded.n_faces(), 1);
        let props = loaded.vertex_property_refs();
        assert_eq!(props.len(), 3);
        assert!(matches!(
            props[0],
            VertexPropertyRef::Float { name: "quality", values }
                if values[1] == 2.5
        ));
        assert!(matches!(
            props[1],
            VertexPropertyRef::Int { name: "region", values }
                if values[2] == 9
        ));
        assert!(matches!(
            props[2],
            VertexPropertyRef::Vec3 { name: "gradient", values }
                if values[1] == Vec3::new(4.0, 5.0, 6.0)
        ));
    }

    #[test]
    fn test_attrib_ply_rejects_unsupported_vec4_dynamic_property() {
        let mut mesh = AttribSoAKernel::new();
        let vh = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let weights = mesh.add_vertex_property::<Vec4>("weights");
        mesh.set_vertex_property(weights, vh, Vec4::repeat(1.0));

        let dir = tempdir().unwrap();
        let path = dir.path().join("attrib_unsupported_vec4.ply");
        let err = write_attrib_ply(&mesh, &path, PlyFormat::Ascii).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("Vec4"));
    }
}
