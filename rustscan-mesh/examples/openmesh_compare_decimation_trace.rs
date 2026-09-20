mod openmesh_compare_common;

use openmesh_compare_common::{
    cleanup_paths, measure, mesh_digest, openmesh_root, print_duration_compare, print_header,
    print_mesh_digest, write_temp_off,
};
use rustscan_mesh::{generate_sphere, read_off, Decimater, DecimationTraceStep};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone)]
struct TraceStepRow {
    step: usize,
    removed: usize,
    kept: usize,
    boundary: bool,
    faces_removed: usize,
    priority: f64,
    active_faces_before: usize,
    active_faces_after: usize,
}

#[derive(Debug, Clone, Default)]
struct TraceRun {
    collapsed: usize,
    boundary_collapses: usize,
    interior_collapses: usize,
    final_vertices: usize,
    final_faces: usize,
    steps: Vec<TraceStepRow>,
}

fn main() {
    let trace_limit = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(40);

    print_header("RustMesh vs OpenMesh Decimation Trace");
    println!("Reference sources:");
    println!("  - Mirror/OpenMesh-11.0.0/src/OpenMesh/Tools/Decimater/DecimaterT_impl.hh");
    println!("  - Mirror/OpenMesh-11.0.0/src/OpenMesh/Tools/Decimater/BaseDecimaterT_impl.hh");
    println!("  - Mirror/OpenMesh-11.0.0/src/OpenMesh/Tools/Decimater/ModQuadricT.hh");

    let input = generate_sphere(1.0, 10, 10);
    let input_path = write_temp_off(&input, "decimate-trace-input").expect("write input OFF");
    let target_vertices = input.n_vertices() / 2;

    print_mesh_digest("Input mesh", mesh_digest(&input));
    println!(
        "Trace configuration: target_vertices={}, trace_limit={}",
        target_vertices, trace_limit
    );

    let (rust_time, rust_trace) =
        measure(|| run_rust_trace(&input_path, target_vertices, trace_limit));
    let rust_trace = rust_trace.expect("run RustMesh trace");
    print_summary("rustmesh", &rust_trace);

    let (openmesh_time, openmesh_trace) =
        measure(|| run_openmesh_trace(&input_path, target_vertices, trace_limit));
    let openmesh_trace = openmesh_trace.expect("run OpenMesh trace");
    print_summary("OpenMesh", &openmesh_trace);

    print_duration_compare("Trace pipeline", rust_time, Some(openmesh_time));

    print_header("Trace Comparison");
    let matching_prefix = rust_trace
        .steps
        .iter()
        .zip(openmesh_trace.steps.iter())
        .take_while(|(lhs, rhs)| same_signature(lhs, rhs))
        .count();
    println!(
        "Matching prefix on removed/kept/boundary/faces_removed: {} steps",
        matching_prefix
    );
    let matching_edge_prefix = rust_trace
        .steps
        .iter()
        .zip(openmesh_trace.steps.iter())
        .take_while(|(lhs, rhs)| same_undirected_edge(lhs, rhs))
        .count();
    println!(
        "Matching prefix on undirected edge + faces_removed: {} steps",
        matching_edge_prefix
    );
    if let Some((lhs, rhs)) = rust_trace
        .steps
        .iter()
        .zip(openmesh_trace.steps.iter())
        .find(|(lhs, rhs)| !same_signature(lhs, rhs))
    {
        println!(
            "First divergence at step {}: RustMesh {} vs OpenMesh {}",
            lhs.step,
            format_step(lhs),
            format_step(rhs)
        );
    }

    let rust_boundary_prefix = rust_trace.steps.iter().filter(|step| step.boundary).count();
    let openmesh_boundary_prefix = openmesh_trace
        .steps
        .iter()
        .filter(|step| step.boundary)
        .count();
    println!(
        "Boundary collapses in first {} traced steps: RustMesh={}, OpenMesh={}",
        trace_limit, rust_boundary_prefix, openmesh_boundary_prefix
    );
    if let Some(step) = first_boundary_mix_gap(&rust_trace.steps, &openmesh_trace.steps) {
        println!(
            "First cumulative boundary/interior mix gap within trace: step {}",
            step
        );
    }

    if let Some((lhs, rhs)) = rust_trace
        .steps
        .iter()
        .zip(openmesh_trace.steps.iter())
        .find(|(lhs, rhs)| lhs.active_faces_after != rhs.active_faces_after)
    {
        println!(
            "First active-face gap at step {}: RustMesh={} OpenMesh={}",
            lhs.step, lhs.active_faces_after, rhs.active_faces_after
        );
    } else {
        println!("No active-face gap inside the traced prefix.");
    }

    println!("\nStep-by-step trace:");
    for idx in 0..trace_limit
        .min(rust_trace.steps.len())
        .min(openmesh_trace.steps.len())
    {
        let rust_step = &rust_trace.steps[idx];
        let openmesh_step = &openmesh_trace.steps[idx];
        println!(
            "#{:02} RustMesh: {:<44} | OpenMesh: {:<44} | {}",
            rust_step.step,
            format_step(rust_step),
            format_step(openmesh_step),
            if same_signature(rust_step, openmesh_step) {
                "match"
            } else {
                "diff"
            }
        );
    }

    cleanup_paths(&[input_path.as_path()]);
}

fn run_rust_trace(
    input_path: &Path,
    target_vertices: usize,
    trace_limit: usize,
) -> io::Result<TraceRun> {
    let mut mesh = read_off(input_path)
        .map_err(|err| io::Error::new(io::ErrorKind::Other, err.to_string()))?;
    let mut decimater = Decimater::new(&mut mesh);
    let trace = decimater.decimate_to_with_trace(target_vertices, trace_limit);
    let boundary_collapses = decimater.boundary_collapses();
    let interior_collapses = decimater.interior_collapses();
    drop(decimater);
    mesh.garbage_collection();
    let digest = mesh_digest(&mesh);

    Ok(TraceRun {
        collapsed: trace.collapsed,
        boundary_collapses,
        interior_collapses,
        final_vertices: digest.vertices,
        final_faces: digest.faces,
        steps: trace.steps.into_iter().map(step_from_rust).collect(),
    })
}

fn step_from_rust(step: DecimationTraceStep) -> TraceStepRow {
    TraceStepRow {
        step: step.step,
        removed: step.v_removed.idx_usize(),
        kept: step.v_kept.idx_usize(),
        boundary: step.is_boundary,
        faces_removed: step.faces_removed as usize,
        priority: step.priority as f64,
        active_faces_before: step.active_faces_before,
        active_faces_after: step.active_faces_after,
    }
}

fn run_openmesh_trace(
    input_path: &Path,
    target_vertices: usize,
    trace_limit: usize,
) -> io::Result<TraceRun> {
    let cpp_path = PathBuf::from("/tmp").join("rustmesh-openmesh-decimation-trace.cpp");
    let bin_path = PathBuf::from("/tmp").join("rustmesh-openmesh-decimation-trace");
    fs::write(&cpp_path, openmesh_trace_cpp_source())?;

    let openmesh_root = openmesh_root();
    let include_src = openmesh_root.join("src");
    let include_build = openmesh_root.join("build/src");
    let lib_dir = openmesh_root.join("build/Build/lib");

    let compile_output = Command::new("c++")
        .arg("-O2")
        .arg("-std=c++17")
        .arg("-I")
        .arg(&include_src)
        .arg("-I")
        .arg(&include_build)
        .arg(&cpp_path)
        .arg("-L")
        .arg(&lib_dir)
        .arg("-lOpenMeshTools")
        .arg("-lOpenMeshCore")
        .arg("-o")
        .arg(&bin_path)
        .output()?;
    if !compile_output.status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!(
                "failed to compile OpenMesh decimation trace driver:\n{}\n{}",
                String::from_utf8_lossy(&compile_output.stdout),
                String::from_utf8_lossy(&compile_output.stderr)
            ),
        ));
    }

    let output = Command::new(&bin_path)
        .arg(input_path)
        .arg(target_vertices.to_string())
        .arg(trace_limit.to_string())
        .env("DYLD_LIBRARY_PATH", &lib_dir)
        .output()?;
    if !output.status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }

    parse_openmesh_trace(&String::from_utf8_lossy(&output.stdout))
}

fn parse_openmesh_trace(stdout: &str) -> io::Result<TraceRun> {
    let mut run = TraceRun::default();

    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("SUMMARY ") {
            for (key, value) in parse_kv_fields(rest) {
                match key {
                    "collapsed" => run.collapsed = value.parse().unwrap_or(0),
                    "boundary" => run.boundary_collapses = value.parse().unwrap_or(0),
                    "interior" => run.interior_collapses = value.parse().unwrap_or(0),
                    "final_vertices" => run.final_vertices = value.parse().unwrap_or(0),
                    "final_faces" => run.final_faces = value.parse().unwrap_or(0),
                    _ => {}
                }
            }
        } else if let Some(rest) = line.strip_prefix("TRACE ") {
            let mut step = TraceStepRow {
                step: 0,
                removed: 0,
                kept: 0,
                boundary: false,
                faces_removed: 0,
                priority: 0.0,
                active_faces_before: 0,
                active_faces_after: 0,
            };

            for (key, value) in parse_kv_fields(rest) {
                match key {
                    "step" => step.step = value.parse().unwrap_or(0),
                    "removed" => step.removed = value.parse().unwrap_or(0),
                    "kept" => step.kept = value.parse().unwrap_or(0),
                    "boundary" => step.boundary = value == "1",
                    "faces_removed" => step.faces_removed = value.parse().unwrap_or(0),
                    "priority" => step.priority = value.parse().unwrap_or(0.0),
                    "active_faces_before" => step.active_faces_before = value.parse().unwrap_or(0),
                    "active_faces_after" => step.active_faces_after = value.parse().unwrap_or(0),
                    _ => {}
                }
            }

            run.steps.push(step);
        }
    }

    if run.collapsed == 0 && run.steps.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "failed to parse OpenMesh decimation trace output",
        ));
    }

    Ok(run)
}

fn parse_kv_fields(line: &str) -> Vec<(&str, &str)> {
    line.split_whitespace()
        .filter_map(|field| field.split_once('='))
        .collect()
}

fn same_signature(lhs: &TraceStepRow, rhs: &TraceStepRow) -> bool {
    lhs.removed == rhs.removed
        && lhs.kept == rhs.kept
        && lhs.boundary == rhs.boundary
        && lhs.faces_removed == rhs.faces_removed
}

fn same_undirected_edge(lhs: &TraceStepRow, rhs: &TraceStepRow) -> bool {
    let lhs_edge = (lhs.removed.min(lhs.kept), lhs.removed.max(lhs.kept));
    let rhs_edge = (rhs.removed.min(rhs.kept), rhs.removed.max(rhs.kept));
    lhs_edge == rhs_edge && lhs.faces_removed == rhs.faces_removed
}

fn first_boundary_mix_gap(lhs: &[TraceStepRow], rhs: &[TraceStepRow]) -> Option<usize> {
    let mut lhs_boundary = 0usize;
    let mut rhs_boundary = 0usize;

    for (left, right) in lhs.iter().zip(rhs.iter()) {
        if left.boundary {
            lhs_boundary += 1;
        }
        if right.boundary {
            rhs_boundary += 1;
        }

        if lhs_boundary != rhs_boundary {
            return Some(left.step);
        }
    }

    None
}

fn format_step(step: &TraceStepRow) -> String {
    format!(
        "{}->{} {} faces={} prio={:.6} active={}=>{}",
        step.removed,
        step.kept,
        if step.boundary { "B" } else { "I" },
        step.faces_removed,
        step.priority,
        step.active_faces_before,
        step.active_faces_after
    )
}

fn print_summary(label: &str, run: &TraceRun) {
    println!(
        "{} summary: collapsed={}, boundary={}, interior={}, final V={}, final F={}",
        label,
        run.collapsed,
        run.boundary_collapses,
        run.interior_collapses,
        run.final_vertices,
        run.final_faces
    );
}

fn openmesh_trace_cpp_source() -> &'static str {
    r#"
#include <cfloat>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/Status.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/System/config.h>
#include <OpenMesh/Tools/Decimater/BaseDecimaterT.hh>
#include <OpenMesh/Tools/Decimater/CollapseInfoT.hh>
#include <OpenMesh/Tools/Decimater/ModQuadricT.hh>
#include <OpenMesh/Tools/Utils/HeapT.hh>

struct TraceTraits : public OpenMesh::DefaultTraits {
  VertexAttributes(OpenMesh::Attributes::Status);
  FaceAttributes(OpenMesh::Attributes::Status);
  EdgeAttributes(OpenMesh::Attributes::Status);
};

using Mesh = OpenMesh::TriMesh_ArrayKernelT<TraceTraits>;

template <typename MeshT>
class TraceDecimater : public OpenMesh::Decimater::BaseDecimaterT<MeshT> {
public:
  using Base = OpenMesh::Decimater::BaseDecimaterT<MeshT>;
  using CollapseInfo = typename Base::CollapseInfo;
  using VertexHandle = typename MeshT::VertexHandle;
  using HalfedgeHandle = typename MeshT::HalfedgeHandle;

  struct TraceStep {
    size_t step = 0;
    unsigned removed = 0;
    unsigned kept = 0;
    bool boundary = false;
    unsigned faces_removed = 0;
    float priority = 0.0f;
    size_t active_faces_before = 0;
    size_t active_faces_after = 0;
  };

  struct TraceResult {
    size_t collapsed = 0;
    size_t boundary_collapses = 0;
    size_t interior_collapses = 0;
    size_t final_vertices = 0;
    size_t final_faces = 0;
    std::vector<TraceStep> steps;
  };

  class HeapInterface {
  public:
    HeapInterface(
        MeshT& mesh,
        OpenMesh::VPropHandleT<float> prio,
        OpenMesh::VPropHandleT<int> pos)
        : mesh_(mesh), prio_(prio), pos_(pos) {}

    bool less(VertexHandle lhs, VertexHandle rhs) {
      return mesh_.property(prio_, lhs) < mesh_.property(prio_, rhs);
    }

    bool greater(VertexHandle lhs, VertexHandle rhs) {
      return mesh_.property(prio_, lhs) > mesh_.property(prio_, rhs);
    }

    int get_heap_position(VertexHandle vh) {
      return mesh_.property(pos_, vh);
    }

    void set_heap_position(VertexHandle vh, int pos) {
      mesh_.property(pos_, vh) = pos;
    }

  private:
    MeshT& mesh_;
    OpenMesh::VPropHandleT<float> prio_;
    OpenMesh::VPropHandleT<int> pos_;
  };

  using DeciHeap = OpenMesh::Utils::HeapT<VertexHandle, HeapInterface>;

  explicit TraceDecimater(MeshT& mesh)
      : Base(mesh), mesh_(mesh) {
    mesh_.add_property(collapse_target_);
    mesh_.add_property(priority_);
    mesh_.add_property(heap_position_);
  }

  ~TraceDecimater() override {
    mesh_.remove_property(collapse_target_);
    mesh_.remove_property(priority_);
    mesh_.remove_property(heap_position_);
  }

  using Base::add;
  using Base::initialize;
  using Base::module;

  TraceResult trace_to(size_t target_vertices, size_t trace_limit) {
    size_t n_collapses =
        target_vertices < mesh_.n_vertices() ? mesh_.n_vertices() - target_vertices : 0;
    return trace_decimate(n_collapses, trace_limit);
  }

private:
  void heap_vertex(VertexHandle vh) {
    float prio = 0.0f;
    float best_prio = FLT_MAX;
    HalfedgeHandle heh;
    HalfedgeHandle collapse_target;

    typename MeshT::VertexOHalfedgeIter voh_it(mesh_, vh);
    for (; voh_it.is_valid(); ++voh_it) {
      heh = *voh_it;
      CollapseInfo ci(mesh_, heh);
      if (this->is_collapse_legal(ci)) {
        prio = this->collapse_priority(ci);
        if (prio >= 0.0f && prio < best_prio) {
          best_prio = prio;
          collapse_target = heh;
        }
      }
    }

    if (collapse_target.is_valid()) {
      mesh_.property(collapse_target_, vh) = collapse_target;
      mesh_.property(priority_, vh) = best_prio;

      if (heap_->is_stored(vh)) {
        heap_->update(vh);
      } else {
        heap_->insert(vh);
      }
    } else {
      if (heap_->is_stored(vh)) {
        heap_->remove(vh);
      }
      mesh_.property(collapse_target_, vh) = collapse_target;
      mesh_.property(priority_, vh) = -1.0f;
    }
  }

  size_t active_faces() const {
    size_t count = 0;
    for (auto fh : mesh_.faces()) {
      if (!mesh_.status(fh).deleted()) {
        ++count;
      }
    }
    return count;
  }

  TraceResult trace_decimate(size_t n_collapses, size_t trace_limit) {
    TraceResult result;
    if (!this->is_initialized()) {
      return result;
    }

    using Support = std::vector<VertexHandle>;
    Support support;

    HeapInterface heap_interface(mesh_, priority_, heap_position_);
    heap_ = std::unique_ptr<DeciHeap>(new DeciHeap(heap_interface));
    heap_->reserve(mesh_.n_vertices());

    for (auto vh : mesh_.vertices()) {
      heap_->reset_heap_position(vh);
      if (!mesh_.status(vh).deleted()) {
        heap_vertex(vh);
      }
    }

    while (!heap_->empty() && result.collapsed < n_collapses) {
      VertexHandle vp = heap_->front();
      HalfedgeHandle v0v1 = mesh_.property(collapse_target_, vp);
      heap_->pop_front();

      CollapseInfo ci(mesh_, v0v1);
      if (!this->is_collapse_legal(ci)) {
        continue;
      }

      support.clear();
      for (typename MeshT::VertexVertexIter vv_it = mesh_.vv_iter(ci.v0); vv_it.is_valid(); ++vv_it) {
        support.push_back(*vv_it);
      }

      const bool is_boundary = mesh_.is_boundary(ci.v0v1) || mesh_.is_boundary(ci.v1v0);
      const unsigned faces_removed =
          (ci.fl.is_valid() ? 1u : 0u) + (ci.fr.is_valid() ? 1u : 0u);
      const float priority = this->collapse_priority(ci);
      const size_t active_faces_before = active_faces();

      this->preprocess_collapse(ci);
      mesh_.collapse(v0v1);
      ++result.collapsed;
      this->postprocess_collapse(ci);

      if (is_boundary) {
        ++result.boundary_collapses;
      } else {
        ++result.interior_collapses;
      }

      if (result.steps.size() < trace_limit) {
        TraceStep step;
        step.step = result.collapsed;
        step.removed = ci.v0.idx();
        step.kept = ci.v1.idx();
        step.boundary = is_boundary;
        step.faces_removed = faces_removed;
        step.priority = priority;
        step.active_faces_before = active_faces_before;
        step.active_faces_after = active_faces();
        result.steps.push_back(step);
      }

      for (auto vh : support) {
        if (!mesh_.status(vh).deleted()) {
          heap_vertex(vh);
        }
      }
    }

    heap_.reset();
    mesh_.garbage_collection();
    result.final_vertices = mesh_.n_vertices();
    result.final_faces = mesh_.n_faces();
    return result;
  }

private:
  MeshT& mesh_;
  std::unique_ptr<DeciHeap> heap_;
  OpenMesh::VPropHandleT<HalfedgeHandle> collapse_target_;
  OpenMesh::VPropHandleT<float> priority_;
  OpenMesh::VPropHandleT<int> heap_position_;
};

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: trace_driver <input.off> <target_vertices> <trace_limit>\n";
    return 1;
  }

  const std::string input_path = argv[1];
  const size_t target_vertices = static_cast<size_t>(std::strtoull(argv[2], nullptr, 10));
  const size_t trace_limit = static_cast<size_t>(std::strtoull(argv[3], nullptr, 10));

  Mesh mesh;
  if (!OpenMesh::IO::read_mesh(mesh, input_path)) {
    std::cerr << "failed to read mesh: " << input_path << "\n";
    return 1;
  }

  TraceDecimater<Mesh> decimater(mesh);
  OpenMesh::Decimater::ModQuadricT<Mesh>::Handle mod_quadric;
  decimater.add(mod_quadric);
  decimater.module(mod_quadric).set_binary(false);

  if (!decimater.initialize()) {
    std::cerr << "failed to initialize decimater\n";
    return 1;
  }

  const auto result = decimater.trace_to(target_vertices, trace_limit);
  std::cout << std::fixed << std::setprecision(9);
  std::cout << "SUMMARY"
            << " collapsed=" << result.collapsed
            << " boundary=" << result.boundary_collapses
            << " interior=" << result.interior_collapses
            << " final_vertices=" << result.final_vertices
            << " final_faces=" << result.final_faces
            << "\n";

  for (const auto& step : result.steps) {
    std::cout << "TRACE"
              << " step=" << step.step
              << " removed=" << step.removed
              << " kept=" << step.kept
              << " boundary=" << (step.boundary ? 1 : 0)
              << " faces_removed=" << step.faces_removed
              << " priority=" << step.priority
              << " active_faces_before=" << step.active_faces_before
              << " active_faces_after=" << step.active_faces_after
              << "\n";
  }

  return 0;
}
"#
}
