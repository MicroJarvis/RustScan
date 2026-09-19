# RustGS 后续工作 TODO

**更新日期：** 2026-09-18

**代码依据：** 本轮在 `938fa8d` 工作树之上完成 R01～R12（证据见实验记录）。二进制归档：`output/rustgs-optimization/2026-09-18/bin/rustgs-r09-current`。

**使用方式：** 按下列顺序执行；编号为本次重排后的新编号。这里仅保留未完成工作，每项通过验收后从清单移除，证据写入独立实验记录。

[审核与 Home 实验记录](reviews/2026-09-18-rustgs-training-review-and-home-evidence.md)保留已有结果。旧技术方案已在计划审计中删除；当前范围、优先级及纠错要求以本文为准，不能按旧编号重复实施。仍在执行的 pipeline remediation 见 [RS-2026-002](agent/changes/RS-2026-002-rustgs-training-pipeline-remediation/)。

## 残留：跨场景验收缺口

### [ ] R09b 补齐 TUM 全视图与留出视图阶梯

**阻塞：** 本机无 `test_data/tum/rgbd_dataset_freiburg1_xyz`；`output/vksplat_rustgs_benchmark/datasets/tum_freiburg1_xyz_colmap/images` 仅含失效 symlink；官方与 mirror 下载返回 403/401/超时。

**要做：** 取得可复现 TUM COLMAP pack 后，按 500 → 3k → 10k → 30k 跑全视图与独立留出视图；对照 `09f9254` 与当前 revision；最差帧下降 >0.2 dB 或 fog/ghosting 即拒绝。Home 与 flowers2-24v 外部短跑已通过（见实验记录），不替代本项。

## 执行与完成规则

- R01～R12（除 TUM 阶梯）已记录证据；下一步仅 R09b。
- 每项记录代码 revision、验证命令、实际结果及失败原因。测试通过不等于跨场景质量已通过。
- 失败时保留候选和证据，通过显式实验配置或独立 checkout 返回基线；不回滚其他未提交工作。
- 后续实验记录写入独立文档与产物目录，本 TODO 不再追加已完成事项和逐次跑分流水账。
