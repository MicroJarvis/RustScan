# RustSLAM 框架设计与路线图

> Rust 实现的视觉 SLAM 库

---

## 一、项目概述

### 1.1 目标

打造一个高性能、高可用的纯 Rust 视觉 SLAM 库，支持：
- 单目/双目/RGB-D 视觉里程计
- 稀疏地图构建
- 回环检测
- 可选的稠密重建

### 1.2 技术选型

| 组件 | 技术方案 | 理由 |
|------|----------|------|
| **图像处理** | opencv-rust | OpenCV 官方绑定 |
| **数学库** | glam | SIMD 加速，高性能 |
| **深度学习** | tch-rs (PyTorch) | 模型丰富，成熟稳定 |
| **图优化** | g2o-rs | 通用优化器 |
| **并发** | rayon | 数据并行 |

---

## 二、模块架构

```
rustslam/
├── Cargo.toml
│
├── src/
│   ├── lib.rs
│   │
│   ├── core/                # 核心数据结构
│   │   ├── mod.rs
│   │   ├── frame.rs         # 帧
│   │   ├── keyframe.rs      # 关键帧
│   │   ├── map_point.rs     # 地图点
│   │   ├── map.rs           # 地图管理
│   │   ├── camera.rs        # 相机模型
│   │   └── pose.rs          # 位姿 (SE3)
│   │
│   ├── features/            # 特征提取
│   │   ├── mod.rs
│   │   ├── base.rs          # 特征接口
│   │   ├── orb.rs           # ORB 特征
│   │   ├── akaze.rs         # AKAZE 特征
│   │   ├── superpoint.rs    # SuperPoint
│   │   └── matcher.rs       # 特征匹配
│   │
│   ├── tracker/             # 视觉里程计
│   │   ├── mod.rs
│   │   ├── vo.rs            # 里程计主逻辑
│   │   ├── pnp.rs           # PnP 求解
│   │   ├── motion_model.rs  # 运动模型
│   │   └── initializer.rs   # 初始化
│   │
│   ├── mapping/             # 局部建图
│   │   ├── mod.rs
│   │   ├── triangulate.rs   # 三角化
│   │   ├── keyframe_culling.rs
│   │   └── local_ba.rs      # 局部 BA
│   │
│   ├── optimizer/           # 图优化
│   │   ├── mod.rs
│   │   ├── g2o.rs
│   │   └── bundle_adjustment.rs
│   │
│   ├── loop_closing/        # 回环检测
│   │   ├── mod.rs
│   │   ├── detector.rs
│   │   ├── vocabulary.rs    # BoW 词袋
│   │   ├── database.rs
│   │   └── geometric_verifier.rs
│   │
│   ├── fusion/             # 稠密融合 (可选)
│   │   ├── mod.rs
│   │   └── tsdf.rs
│   │
│   └── io/                 # IO
│       ├── mod.rs
│       └── trajectory.rs
│
└── examples/               # 示例
    ├── run_vo.rs
    └── run_slam.rs
```

---

## 三、模块详解

### 3.1 core (核心数据结构)

```rust
// pose.rs - 位姿表示
use glam::{Quat, Vec3, Mat4};

pub struct SE3 {
    rotation: Quat,      // 旋转
    translation: Vec3,   // 平移
}

impl SE3 {
    pub fn from_matrix(mat: &Mat4) -> Self;
    pub fn to_matrix(&self) -> Mat4;
    pub fn compose(&self, other: &SE3) -> SE3;
    pub fn inverse(&self) -> SE3;
}

// frame.rs - 帧
pub struct Frame {
    pub id: u64,
    pub timestamp: f64,
    pub image: GrayImage,
    pub depth: Option<DepthImage>,
    pub features: Vec<Feature>,
    pub pose: Option<SE3>,
}
```

### 3.2 features (特征提取)

```rust
// base.rs - 特征接口
pub trait FeatureExtractor {
    fn detect(&mut self, image: &GrayImage) -> Result<(Vec<KeyPoint>, Descriptors)>;
}

pub trait FeatureMatcher {
    fn match_features(&self, des1: &Descriptors, des2: &Descriptors) -> Result<Vec<Match>>;
}

// orb.rs - ORB 特征
pub struct OrbExtractor {
    inner: opencv::features::Orb,
}

impl FeatureExtractor for OrbExtractor {
    fn detect(&mut self, image: &GrayImage) -> Result<(Vec<KeyPoint>, Descriptors)> {
        // 调用 OpenCV ORB
    }
}
```

### 3.3 tracker (视觉里程计)

```rust
// vo.rs - 里程计主逻辑
pub struct VisualOdometry {
    extractor: Box<dyn FeatureExtractor>,
    matcher: Box<dyn FeatureMatcher>,
    pnp_solver: PnPSolver,
    motion_model: MotionModel,
}

impl VisualOdometry {
    pub fn estimate(&mut self, frame: &mut Frame) -> Result<SE3> {
        // 1. 提取特征
        // 2. 与上一帧匹配
        // 3. PnP 求解
        // 4. 更新运动模型
    }
}
```

### 3.4 mapping (局部建图)

```rust
// triangulate.rs - 三角化
pub fn triangulate_points(
    kp1: &KeyPoint, kp2: &KeyPoint,
    pose1: &SE3, pose2: &SE3,
    camera: &Camera,
) -> Option<Vec3> {
    // SVD 三角化
}

// local_ba.rs - 局部 BA
pub fn local_bundle_adjustment(
    keyframes: &[KeyFrame],
    points: &[MapPoint],
    optimizer: &mut G2oOptimizer,
) {
    // 添加顶点和边，执行优化
}
```

### 3.5 loop_closing (回环检测)

```rust
// vocabulary.rs - BoW 词袋
pub struct BoWVocabulary {
    words: Vec<BoWWord>,
    levels: u32,
}

// detector.rs - 回环检测器
pub struct LoopDetector {
    vocabulary: BoWVocabulary,
    database: KeyframeDatabase,
}
```

---

## 四、依赖配置

### Cargo.toml

```toml
[package]
name = "rustslam"
version = "0.1.0"
edition = "2021"

[dependencies]
# 图像处理
opencv = "0.9"

# 数学库 (SIMD 加速)
glam = "0.25"

# 深度学习
tch = "0.5"

# 图优化
g2o = "0.7"

# 并发
rayon = "1.8"

# 序列化
serde = { version = "1.0", features = ["derive"] }

# 日志
log = "0.4"
env_logger = "0.11"

# 错误处理
thiserror = "1.0"

[build-dependencies]
bindgen = "0.69"
```

---

## 五、路线图 (Roadmap)

### Phase 1: 基础框架 (v0.1) - 4周

| 周次 | 任务 | 交付物 |
|------|------|--------|
| **W1** | 搭建项目结构，core 数据结构 | `core/` 模块 |
| **W2** | ORB 特征提取 + 匹配 | `features/orb.rs` |
| **W3** | PnP 位姿求解，运动模型 | `tracker/` 基础 |
| **W4** | 简单视觉里程计 demo | `examples/run_vo` |

**里程碑**: 能跑通单目 VO

---

### Phase 2: 建图与优化 (v0.2) - 4周

| 周次 | 任务 | 交付物 |
|------|------|--------|
| **W5** | 三角化 + 地图点管理 | `mapping/triangulate.rs` |
| **W6** | 关键帧管理 + 局部 BA | `mapping/local_ba.rs` |
| **W7** | g2o 集成 + BA 优化 | `optimizer/` |
| **W8** | 完整 SLAM 流程集成 | `lib.rs` 主逻辑 |

**里程碑**: 完整稀疏 SLAM

---

### Phase 3: 回环检测 (v0.3) - 3周

| 周次 | 任务 | 交付物 |
|------|------|--------|
| **W9** | BoW 词袋实现 | `loop_closing/vocabulary.rs` |
| **W10** | 关键帧数据库 + 检索 | `loop_closing/database.rs` |
| **W11** | 几何验证 + Sim3 校正 | `loop_closing/geometric.rs` |

**里程碑**: 支持回环的完整 SLAM

---

### Phase 4: 高级功能 (v0.4) - 4周

| 周次 | 任务 | 交付物 |
|------|------|--------|
| **W12** | SuperPoint 特征 | `features/superpoint.rs` |
| **W13** | IMU 预积分 | `imu.rs` |
| **W14** | TSDF 稠密融合 | `fusion/tsdf.rs` |
| **W15** | 性能优化 + 文档 | 稳定版 |

**里程碑**: 高精度 SLAM

---

## 六、进度总览

```
v0.1 (W1-W4): 基础框架 + VO
    ├── W1: core 数据结构
    ├── W2: ORB 特征
    ├── W3: PnP + 运动模型
    └── W4: VO demo

v0.2 (W5-W8): 建图 + BA
    ├── W5: 三角化
    ├── W6: 关键帧管理
    ├── W7: g2o 集成
    └── W8: 完整 SLAM

v0.3 (W9-W11): 回环检测
    ├── W9: BoW 词袋
    ├── W10: 关键帧数据库
    └── W11: 几何验证

v0.4 (W12-W15): 高级功能
    ├── W12: SuperPoint
    ├── W13: IMU
    ├── W14: TSDF 融合
    └── W15: 优化
```

---

## 七、验证数据集

| 数据集 | 用途 | 传感器 |
|--------|------|--------|
| **KITTI** | 室外双目 | 双目 |
| **EuRoC** | 室内 + IMU | 单目+IMU |
| **TUM RGB-D** | 室内 RGB-D | RGB-D |
| **口腔扫描数据** | 目标场景 | 结构光/双目 |

---

## 八、参考

- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [pySLAM](https://github.com/luigifreda/pyslam)
- [OpenCV Rust](https://github.com/twistedfall/opencv-rust)
- [glam](https://github.com/bitshifter/glam-rs)
- [tch-rs](https://github.com/LaurentMazare/tch-rs)

---

*最后更新: 2026-02-13*
