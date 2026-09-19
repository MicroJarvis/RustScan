# nalgebra 统一改造审查问题修复计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 nalgebra 统一改造审查中剩余的两个问题：RustMesh 的空间位置必须使用 `Point3<f32>`，以及 CI 必须可靠阻止新增 CPU 侧 glam 数学类型。

**Architecture:** 保留 RustMesh 的 SoA 数值存储和 GPU/IO 数组边界，但在 CPU 公共 API 中区分 `Point3<f32>`（位置）和 `Vector3<f32>`（方向、位移、法线）。新增一个独立的源码策略检查脚本，检查 Rust 导入、完全限定类型和 Cargo 依赖，并在 CI 中调用它；图形/GPU 明确边界通过 allowlist 保留。

**Tech Stack:** Rust 2021、`nalgebra 0.33`、Cargo workspace、Python 3（CI 已可用）、GitHub Actions。

**Spec:** `docs/nalgebra-unification-todo.md`、根目录 `AGENTS.md`。

## Global Constraints

- CPU 数学类型只能使用 `nalgebra`；位置使用 `nalgebra::Point3<T>`，方向/位移/法线使用 `nalgebra::Vector3<T>`。
- 矩阵索引为 `(row, column)`，向量是列向量，变换为 `p' = M * p`。
- 固定数组只能出现在 IO、序列化、checkpoint、GPU uniform 或 FFI 边界；边界函数必须明确布局和语义。
- 不得重新增加 `glam` 依赖，也不得在 CPU API 中引入 `glam::Vec*`、`glam::Mat*`、`glam::Quat`。
- **序列化决策：不兼容旧 glam 文件。** 当前 `SE3` 的对象格式（`rotation: {x,y,z,w}`、`translation: {x,y,z}`）是新的明确格式；不要增加旧 `[x,y,z,w]`/`[x,y,z]` 文件的反序列化兼容层。只需把测试名和文档写清楚这是有意的格式变更。
- 不要为了修复本计划而重写 SE3、GPU shader ABI 或 IO 文件格式。

## 当前审查证据

- `cargo check --workspace --all-targets`：通过。
- `cargo fmt --check`：通过。
- `cargo test -p rustscan-types --lib`：24/24 通过。
- 在 `RustMesh/` 工作目录执行 `cargo test --lib`：210/210 通过。
- 当前问题位置：
  - `RustMesh/src/lib.rs:47-50` 只有 `Vec3 = Vector3<f32>`，没有独立的 `Point3` 别名。
  - `RustMesh/src/Core/geometry.rs:6-8` 将 `Point`、`Vector`、`Normal` 全部别名为同一个 `Vec3`。
  - `RustMesh/src/Core/items.rs:13` 的 `Vertex.point` 是 `Vec3`。
  - `.github/workflows/ci.yml:35-40` 只匹配 `glam::Vec*` 等完全限定写法，无法发现 `use glam::{Vec3, Mat4}`、别名导入或 re-export。

## 文件范围与职责

### RustMesh 类型语义

- 修改：`RustMesh/src/lib.rs`
  - 保留 `Vec2`、`Vec3`、`Vec4` 作为 nalgebra 向量别名。
  - 新增明确的 `Point3`/`Point` 别名（推荐 `pub type Point3 = nalgebra::Point3<f32>; pub type Point = Point3;`）。
- 修改：`RustMesh/src/Core/geometry.rs`
  - `Point` 必须是 `Point3<f32>`，`Vector`/`Normal` 必须是 `Vector3<f32>`。
  - 修正点运算，不能依赖 `Point3 + Point3`：质心使用 `.coords` 求平均后再构造 `Point3`；点差使用 `p1 - p0`；点平移使用 `point + vector`。
- 修改：`RustMesh/src/Core/items.rs`
  - `Vertex.point`、`Vertex::new`、默认值使用 `Point3`。
- 修改：`RustMesh/src/Core/connectivity.rs`、`RustMesh/src/Core/soa_kernel.rs`、`RustMesh/src/Core/attrib_soa_kernel.rs`
  - 所有“顶点位置”参数和返回值改为 `Point3`。
  - 法线和一般向量属性继续使用 `Vector3`；不要把所有 `DynamicProperty::Vec3` 机械替换成点。
  - SoA 内部的 `x/y/z: Vec<f32>` 可继续保留；从数组重建公共位置值时构造 `Point3::new(x, y, z)`。
- 修改：`RustMesh/src/Tools/analysis.rs`、`RustMesh/src/Tools/smart_ranges.rs`、`RustMesh/src/Utils/quadric.rs`、`RustMesh/src/Utils/test_data.rs`
  - 按参数含义逐个迁移：包围盒、质心、顶点位置、测试网格顶点使用 `Point3`；法线、边方向和四元数/矩阵无关的向量计算使用 `Vector3`。
  - Quadric 的平面点和优化结果是位置，使用 `Point3`；平面法向量仍使用 `Vector3`。内部齐次矩阵计算可显式读取 `point.coords`。
- 修改：`RustMesh/src/Core/io/*.rs` 及所有 examples/tests 中受影响的调用点
  - OBJ/PLY/OFF/STL 等解析边界先从标量/数组构造 `Point3`，写出时通过 `.x/.y/.z` 或明确的数组转换。
  - `Vec3::new(...)` 只有在值表示方向、法线、颜色或通用向量时保留；顶点坐标改成 `Point3::new(...)`。

### CI 策略检查

- 新建：`scripts/check_cpu_glam.py`
  - 递归扫描受管控的 `*.rs` 和 crate `Cargo.toml`，跳过 `vendor/`、`target/`、`output/`、`.worktrees/`。
  - 检查以下实际代码模式：
    - `use glam::...`、`pub use glam::...`、`extern crate glam`；
    - 完全限定的 `glam::Vec*`、`glam::DVec*`、`glam::Mat*`、`glam::DMat*`、`glam::Quat`、`glam::DQuat`；
    - crate manifest 中新增的 `glam` 依赖。
  - 忽略 Rust 行注释和文档注释，避免历史说明文字触发误报；不要简单用一个只匹配 `glam::Vec` 的 grep 替代。
  - 支持一个小的显式 allowlist 文件（例如 `scripts/cpu-glam-allowlist.txt`），每行写仓库相对路径和允许原因；默认 allowlist 为空。当前仓库没有需要放行的 CPU glam 类型。
  - 出错时打印路径、行号、匹配内容和修复提示，返回非零退出码。
- 修改：`.github/workflows/ci.yml`
  - 将现有 `Forbid CPU glam types` 的 grep 步骤替换为 `python3 scripts/check_cpu_glam.py`。
  - 保留该步骤在 workspace check 之前运行，使策略违规能快速失败。
- 可选修改：`docs/nalgebra-unification-todo.md`
  - 将“增加静态检查或 CI grep”改为“CI 调用 `scripts/check_cpu_glam.py`”，记录检查覆盖导入、别名/re-export、完全限定类型和依赖声明。

### 序列化决策文档化

- 修改：`rustscan-types/src/pose.rs:433`
  - 将 `serde_preserves_glam_object_layout` 重命名为 `serde_uses_explicit_object_layout`（或同等清晰名称）。
  - 测试继续验证当前对象字段和 round-trip，不要加入旧数组格式测试或兼容解析器。
- 修改：`docs/nalgebra-unification-todo.md` 第 102-112 行附近
  - 明确序列化/checkpoint 使用显式字段格式，旧 glam 文件不在兼容范围内；这是一项有意的破坏性格式变更。

## 任务清单

### Task 1：建立 Point3/Vector3 语义边界

**目标：** 先完成 RustMesh 核心类型和几何函数的语义分离，再扩散到调用点。

- [x] **Step 1：补充失败测试。** 在 `RustMesh/src/Core/geometry.rs` 测试模块增加以下断言：

```rust
use nalgebra::{Point3, Vector3};

#[test]
fn point_and_vector_have_distinct_semantics() {
    let p0 = Point3::new(1.0, 2.0, 3.0);
    let p1 = Point3::new(3.0, 5.0, 7.0);
    let delta: Vector3<f32> = p1 - p0;
    assert_eq!(delta, Vector3::new(2.0, 3.0, 4.0));
    assert_eq!(p0 + delta, p1);
}

#[test]
fn centroid_returns_a_point() {
    let c = triangle_centroid(
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(3.0, 0.0, 0.0),
        Point3::new(0.0, 3.0, 0.0),
    );
    assert_eq!(c, Point3::new(1.0, 1.0, 0.0));
}
```

- [x] **Step 2：运行目标测试，确认它们在旧别名下不能作为语义验证。**

```bash
cargo test -p rustmesh core::geometry::tests::point_and_vector_have_distinct_semantics --lib
cargo test -p rustmesh core::geometry::tests::centroid_returns_a_point --lib
```

预期：在实现迁移前，测试无法按 `Point3` 签名编译或几何函数签名不匹配。

- [x] **Step 3：修改类型别名和几何实现。** 将 `Point` 改成 `Point3<f32>`，将 `Vector`/`Normal` 保持为 `Vector3<f32>`；重写质心、包围盒、点内测试等函数中依赖点加法的表达式。

- [x] **Step 4：运行几何测试。**

```bash
cargo test -p rustmesh core::geometry --lib
```

预期：新增测试和原有几何测试全部通过。

### Task 2：迁移 RustMesh 顶点位置 API

**目标：** 让所有公共顶点位置 API 使用 `Point3`，同时保持法线和通用向量属性为 `Vector3`。

- [x] **Step 1：按语义修改接口。** 至少覆盖 `Vertex.point`、`Vertex::new`、`RustMesh::add_vertex`、`point`、`point_unchecked`、`set_point`、`SoAKernel` 对应方法、连接性算法和包围盒结果。每次修改都检查调用方是“位置”还是“方向”，不要全局替换 `Vec3`。

- [x] **Step 2：修复算子和存储边界。**
  - 从 SoA 标量数组返回位置时构造 `Point3::new(...)`。
  - 点平均通过 `.coords` 完成，再用 `Point3::from(...)` 构造结果。
  - 点差保留为 `Vector3`。
  - IO 数组转换显式写出 `[point.x, point.y, point.z]`；不要把 `Point3` 直接当作 GPU/文件布局。

- [x] **Step 3：更新测试和 examples。** 将测试网格中的顶点构造改成 `Point3::new`，法线/方向保留 `Vector3::new`；增加一个非原点平移测试，证明点平移包含平移量而方向变换不包含平移量。

- [x] **Step 4：运行 RustMesh 全量测试。** 必须从 `RustMesh` 工作目录运行，避免相对 fixture 路径错误：

```bash
(cd RustMesh && cargo test)
```

预期：全部测试通过，且 `cargo check -p rustmesh --all-targets` 通过。

### Task 3：替换 CI glam 检查

**目标：** CI 能发现导入形式、别名/re-export、完全限定类型和新依赖，而不是只匹配一种文本写法。

- [x] **Step 1：先写脚本自测夹具。** 在 `scripts/check_cpu_glam.py` 中使用临时目录或内置字符串测试以下样例：

```rust
use glam::{Vec3, Mat4};
use glam as gm;
pub use glam::Quat;
let a: glam::Vec3 = glam::Vec3::ZERO;
let b: gm::Vec3 = gm::Vec3::ZERO;
```

每个样例都必须返回违规；注释中的 `glam::Vec3` 和普通字符串中的同样文本不得返回违规。

- [x] **Step 2：运行脚本自测并扫描当前仓库。**

```bash
python3 scripts/check_cpu_glam.py --self-test
python3 scripts/check_cpu_glam.py
```

预期：自测全部通过，当前仓库扫描返回 0；历史注释不应造成误报。

- [x] **Step 3：接入 GitHub Actions。** 替换 `.github/workflows/ci.yml` 中旧 grep 步骤，并在 CI 中执行同一条命令。

- [x] **Step 4：验证策略不会漏报。** 临时在一个非 allowlist Rust 文件中加入 `use glam::{Vec3, Mat4};`，确认脚本返回非零；随后删除临时文件并重新运行脚本，确认返回 0。

### Task 4：记录有意的序列化破坏性变更并做最终验收

**目标：** 避免后续工具错误地重新加入 glam 文件兼容层，并给出完整验证证据。

- [x] **Step 1：重命名 Serde 测试并更新 TODO 文档。** 只记录当前对象格式；明确旧 glam 数组格式不受支持。

- [x] **Step 2：运行格式、策略和 workspace 检查。**

```bash
cargo fmt --check
python3 scripts/check_cpu_glam.py
cargo check --workspace --all-targets
```

- [x] **Step 3：运行受影响测试。**

```bash
cargo test -p rustscan-types --lib
cargo test -p rustmesh --lib
(cd RustMesh && cargo test)
```

- [x] **Step 4：确认差异范围。**

```bash
git diff --check
rg -n "glam::(Vec|Quat|Mat|DVec|DQuat|DMat)|use glam|pub use glam|extern crate glam" --glob '*.rs' --glob 'Cargo.toml' .
```

最终报告必须说明：workspace 编译通过、RustMesh 全量测试通过、策略检查通过，以及旧 glam 序列化文件明确不兼容。

## 不要做的事情

- 不要为旧 glam JSON/checkpoint 增加 `untagged enum`、自定义 visitor 或双格式读取。
- 不要把 `Point3` 机械替换所有 `Vec3`；法线、颜色、方向、平移和动态向量属性仍应使用 `Vector3`。
- 不要把 SoA/GPU/IO 的固定数组误当成 CPU 数学公共 API；转换必须在边界处显式完成。
- 不要只扩大现有 grep 正则而跳过导入、别名和 manifest 检查。
