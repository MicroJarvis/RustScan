# 数据目录约定

仓库中的数据按用途分为三类，避免把输入数据、实验记录和运行产物混在一起。

## `artifacts/inputs/`

只放可复用的输入数据和小型测试 fixture。当前 flowers2 的唯一原始图像入口是：

```text
artifacts/inputs/flowers2/images/
```

不要在 `artifacts/inputs/` 下创建运行输出、缓存、训练工作区或临时重建目录。大型数据按现有
`.gitignore` 规则保留在本机，不提交到 Git。

## `artifacts/runs/`

放所有本地生成的运行结果、数据库、日志、稀疏模型、训练产物和性能报告。历史的
flowers2 工作区已经统一移动到：

```text
artifacts/runs/legacy/flowers2/
```

这里保留原目录名，便于根据旧报告追溯；新实验应使用带日期或任务编号的独立目录，
不要继续使用 `artifacts/runs/legacy/flowers2/out*`。

## `artifacts/evidence/`

只保留体积较小、需要版本化并被文档引用的实验证据，例如 JSON、日志和复现脚本。
大型二进制快照、数据库和中间文件放到 `artifacts/runs/`，不要复制到这里。可再生的 GPU
sidecar 文件继续按 `.gitignore` 规则忽略。

## 去重规则

迁移后的 legacy JPG/PNG 按内容做了硬链接去重，数据库、日志和模型文本没有做硬链接。
这些图片视为只读输入；需要修改时先复制到新的运行目录，避免原地写入影响其他历史结果。

所有脚本和 Agent 新增数据时应遵守这三个边界：输入写 `artifacts/inputs/`，证据写
`artifacts/evidence/`，生成物写 `artifacts/runs/`。

`artifacts/runs/` 中少量既有示例快照继续由 Git 跟踪，以保留迁移前的仓库内容；
这不构成新增运行产物可以提交的先例。
