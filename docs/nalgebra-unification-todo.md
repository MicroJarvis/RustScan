# CPU `nalgebra` 统一改造 TODO

## 目标

将 RustScan 的 CPU 数学类型统一为 `nalgebra`：

- 矩阵：`Matrix*`、`SMatrix`、`DMatrix`
- 向量：`Vector2/3/4<T>`
- 空间点：`Point2/3<T>`
- 旋转：`UnitQuaternion<T>`；只有原始四元数运算才使用 `Quaternion<T>`

统一的是 CPU 领域 API 和内部实现，不要求 GPU、WGSL、C/C++ FFI 或 UI
框架共享同一个物理内存类型。

## 不在改造范围内的边界类型

以下类型继续保留，但只能出现在适配层：

- Burn `Tensor`
- WGSL `vec*` / `mat*`
- Eigen C/C++ 类型
- GPU uniform、序列化、checkpoint、FFI 使用的固定数组和扁平数组
- `egui::Vec2` 等 UI 尺寸类型
- 第三方图形接口明确要求的 `glam` 值

每个边界都必须有显式、可测试的转换函数，并在函数名或文档中说明
row-major/column-major、四元数分量顺序和位姿方向。

## 类型和语义约定

- 矩阵按 `(row, column)` 访问。
- 向量按列向量处理，变换为 `p' = M * p`。
- 位置使用 `Point3<T>`，方向/位移使用 `Vector3<T>`。
- 旋转使用 `UnitQuaternion<T>`，COLMAP `[w, x, y, z]` 与 glam
  `[x, y, z, w]` 的转换必须集中处理。
- 实时和渲染路径默认 `f32`；SfM、BA 和高精度求解保留 `f64`。
- 位姿名称或文档必须说明 `world_from_camera` 或 `camera_from_world`。

## 改造阶段

### 1. 建立公共类型基础

- [x] 在 workspace 依赖中统一 `nalgebra` 版本和 feature 配置。
- [x] 为 `rustscan-types` 增加 `nalgebra` 依赖；启用需要的 `serde`
      支持，确认 `Vector3` 和 `UnitQuaternion` 的序列化策略。
- [x] 在 `rustscan-types` 定义公共别名或公共 API，例如：
      `Vec3f = Vector3<f32>`、`Vec3d = Vector3<f64>`、
      `Rotation3f = UnitQuaternion<f32>`。
- [x] 明确 `Point3` 与 `Vector3` 的边界，禁止用裸数组区分空间点和方向。
- [x] 为公共类型增加非对称旋转、非零平移、点变换、向量变换、四元数
      round-trip 测试。

### 2. 合并 `SE3`

- [x] 将 `rustscan-types/src/pose.rs` 的内部字段改为
      `UnitQuaternion<f32>` + `Vector3<f32>`。
- [x] 将 `RustSLAM/src/core/pose.rs` 删除或改为对共享 `SE3` 的 re-export。
- [x] 统一 `new`、`from_rotation_translation`、`to_matrix`、旋转矩阵和
      平移访问器的语义。
- [x] 增加明确命名的转换函数：
      `to_row_major_array`、`to_column_major_array`、
      `from_row_major_array`、`from_column_major_array`。
- [x] 修复当前两份 `SE3` 的数组行列组织差异；禁止继续使用无标签的
      `to_matrix()` 作为跨 crate 协议。
- [x] 更新 RustGS、RustSFM、RustSLAM、RustViewer 和根 CLI 的 `SE3`
      导入路径。

`SE3::to_matrix()` 保留为已文档化的 row-major `[[f32; 4]; 4]` 兼容访问器，
等同于 `to_row_major_array()`。跨 crate 新代码应使用
`to_homogeneous_matrix()` 或具名 array 转换。

### 3. 迁移 RustSFM 和 RustSLAM CPU 几何代码

- [x] RustSFM：将剩余 `glam::Vec3`、`glam::Quat` 和 `glam::Mat*` 替换为
      `Vector*`、`Point*`、`UnitQuaternion` 和 `Matrix*`。
- [x] 优先处理 `geometry/`、`ba/`、`sfm/`、`io/` 和 mapper 中的公共函数
      参数与返回值。
- [x] 保留并复用已有的 `nalgebra::Vector*`、`UnitQuaternion`、`Matrix*`
      实现，避免重复转换。
- [x] RustSLAM：迁移 tracker、loop closing、fusion、camera、pose 和
      mesh/fusion 中的 CPU 数学值。
- [x] 将 `glam` 仅留在明确的渲染/GPU/第三方适配函数中。
- [x] 对 PnP、三角化、位姿图、旋转平均、BA、TSDF 和 mesh 几何补充
      非对称旋转与非零平移回归测试。

### 4. 迁移 RustMesh、RustViewer、RustGS 和 RustFF

- [x] RustMesh：迁移 mesh geometry、quadric、拓扑和属性存储中的
      `glam::Vec2/Vec3/Vec4`；数组只保留在 IO/GPU 边界。
- [x] RustViewer：迁移机器人、相机和场景几何中的 CPU `glam` 类型；保留
      `egui::Vec2` 作为 UI 尺寸类型。
- [x] RustGS：继续使用 `nalgebra` 作为 CPU 矩阵/向量/四元数类型；将
      `core/matrix.rs` 中的 `glam` 输入改为 `UnitQuaternion` + `Vector3`。
- [x] RustGS：在 shader/uniform 适配层将 `nalgebra` 显式转换为数组，保持
      Burn Tensor 和 WGSL ABI 不变。
- [x] RustFF：迁移 Procrustes、pose model 和 inference 结果中的 CPU
      `glam::Mat3/Mat4/Vec3`；手写 3×3 数组算法可保留为局部算法缓冲区，
      但公共 API 必须返回 `nalgebra` 类型。
- [x] 根 CLI 和跨 crate 命令统一使用共享 `nalgebra` 类型，避免重新引入
      `glam`。

### 5. 处理边界转换

- [x] GPU uniform：集中实现 `nalgebra` 到 `[[f32; 4]; 4]`、`[f32; 4]`
      等布局转换。
- [x] Eigen FFI：集中实现 `DMatrix`/`Matrix3` 与 row-major flat buffer
      的转换，标明 Eigen 的存储布局。
- [x] 序列化和 checkpoint：使用显式对象字段格式
      （例如 `rotation: {x,y,z,w}`、`translation: {x,y,z}`），不直接序列化
      `nalgebra` 内部存储布局。旧 glam 数组文件格式（`[x,y,z,w]` /
      `[x,y,z]`）明确不在兼容范围内；这是有意的破坏性格式变更，不要加
      双格式读取或兼容层。
- [x] 四元数 FFI/文件转换：集中处理 `[w,x,y,z]`、`[x,y,z,w]` 和
      `nalgebra::Quaternion::new(w, i, j, k)` 的顺序。
- [x] 每个转换函数至少有一个非对称旋转、平移和点变换 round-trip 测试。

### 6. 依赖和规范清理

- [x] `rustscan-types`、RustViewer、RustFF 等 CPU crate 直接声明所需的
      `nalgebra` 依赖，避免通过其他 crate 间接获得类型。
- [x] 在所有不再需要 `glam` 的 CPU crate 中移除 `glam` 依赖。
- [x] 保留 `glam` 时，在 Cargo 注释或模块文档中说明它属于哪个边界。
- [x] 更新根目录 `AGENTS.md` 与相关 crate README，保持规则一致。
- [x] CI 调用 `scripts/check_cpu_glam.py`，覆盖 `use glam` /
      `pub use glam` / `extern crate glam`、别名导入、完全限定
      `glam::{Vec,Mat,Quat,...}` 类型，以及 crate `Cargo.toml` 中的
      `glam` 依赖声明；图形/GPU 边界通过
      `scripts/cpu-glam-allowlist.txt` 显式放行。

## 验收标准

- [x] CPU 公共 API 中不再出现 `glam::Mat*`、`glam::Vec*`、`glam::Quat`。
- [x] 只有一个共享 `SE3` 实现。
- [x] `nalgebra` 是所有 CPU 数学 crate 的直接依赖，且版本一致。
- [x] GPU、WGSL、Eigen、序列化和 UI 边界仍能编译并通过 round-trip 测试。
- [x] 所有 workspace crate、examples 和 tests 通过 `cargo check --workspace`。
- [x] 运行受影响 crate 的单元测试和集成测试。
- [x] 重点验证非对称旋转、平移、点/向量区别、四元数顺序和矩阵布局。
- [x] `rg` 检查结果中，剩余 `glam` 使用均能对应到已记录的边界例外。

## 风险和注意事项

- `nalgebra` 与 `glam` 的构造函数、索引、矩阵布局和四元数分量顺序不同，
  不能做机械替换。
- `nalgebra` 类型不应直接通过 `bytemuck` 当作 GPU ABI；必须先转换成明确
  布局的 `#[repr(C)]` 结构或数组。
- `Point3` 和 `Vector3` 的替换可能改变算子行为，迁移后要重新检查加法、
  减法和变换语义。
- RustMesh、RustViewer 和 RustGS 的公共 API 变化可能影响 examples 和
  外部调用方，应分 crate、小批次迁移并在每批次后运行测试。
