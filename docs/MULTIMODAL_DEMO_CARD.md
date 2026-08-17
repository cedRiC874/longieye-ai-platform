# LongiEye 全合成多模态 Demo Card

## 状态

- 版本：`v0.4.0`
- 阶段：`demo_multimodal_synthetic`
- 临床用途：`false`
- 图像来源：纯代码确定性生成，不读取、转换或仿制任何真实眼底图像
- 使用位置：仅离线 CLI、测试与本机基准；不进入公开 HTTP API

本组件展示图像合同、质量门、编码器接口、有界融合和安全回退等工程能力。它不是经过训练的视觉模型，不证明多模态方法能改善预测效果，也不得用于诊断、筛查或治疗决策。

## 固定合成图像

| 眼别 | 路径 | PNG SHA-256 | RGB 像素 SHA-256 |
| --- | --- | --- | --- |
| OD | `examples/synthetic_fundus/od.png` | `f232b6fc7a44b1c96d259cfced275c8bbfe84d3914234bae50515e7e1cd3e2dc` | `2c2e1da93031d9f9855886158bd3b1732e3bf8976bc62d2a8606d6a19d6ce9b5` |
| OS | `examples/synthetic_fundus/os.png` | `8b07a05783bd18a35f0581b325628268930b093267aba3b955e9abf64b64a16e` | `4ac1c9e10ba2119ae983d97ac6f9cb7683667f4ab05eeb6dd2b8e1f06e1a81a4` |

生成器只使用固定整数绘图规则，构造黑色背景、圆形视网膜、示意视盘、黄斑和血管，并写入可见的 `SYNTHETIC OD/OS` 水印。PNG 编码器仅生成固定的 128×128、RGB8、无交错、无元数据、filter 0 格式。重新生成不会使用时间、系统随机数、用户名、主机路径或网络资源。

公开工件策略通过扩展名与常见文件头默认拒绝 PNG/JPEG/TIFF/DICOM/NIfTI/PPM、SVG、常见相机 RAW、PDF 和归档容器等工件，仅允许上述两个精确路径与文件摘要，以及既有架构 SVG。该规则是仓库 guardrail，不是内容识别 DLP。

## 解码与质量合同

严格解码器只接受：

- 128×128 RGB8 canonical PNG；
- 精确的 `IHDR → IDAT → IEND` 三个 chunk；
- 有效 CRC、固定 scanline 大小、filter 0 且无尾随数据；
- 最大 512 KiB 的内存字节输入；核心解码器不接受路径、URL 或 Base64；
- 可选的调用方预期眼别与像素 SHA-256；未提供摘要时，解码结果会明确标记为未绑定来源。

质量门计算亮度、对比度、极端像素比例、有效视野覆盖率和相邻像素清晰度。这些阈值只表示“适合本工程 fixture”，不是临床图像质量标准，也不能识别任意噪声或真实眼底疾病。

离线融合器随后会把解码结果与可信内置 OD/OS registry 再次绑定。通过质量门后，图像以确定性 4×4 area pooling 转换为 32×32 RGB。任何眼别、摘要或预处理来源不一致都会整单拒绝，不会静默回退。

## 参考编码器与融合

`DeterministicFundusEncoder` 没有可学习参数，也没有训练数据。它输出五个可审阅统计量：平均亮度、亮度对比度、边缘能量、中心—周边差异和左右非对称度。

`StructuredAnchoredFusionAdapter` 先调用现有九维纵向结构化合成模型，再把图像统计转换为逐眼 logit residual。每眼 residual 的绝对值被限制在 `0.35` 以内；结果仍称为“合成演示分数”，不称为校准风险概率。

| 状态 | 行为 |
| --- | --- |
| 合成图像完整且质量通过 | 该眼使用结构化锚点 + 有界合成图像 residual |
| 图像缺失 | 该眼逐值精确回退到原结构化分数 |
| 质量拒绝 | 该眼逐值精确回退，并返回稳定原因码 |
| 编码器不可用 | 该眼逐值精确回退 |
| 本次单眼编码合同失败 | 失败眼回退，另一眼仍处理；随后图像组件锁定为 not-ready |
| 眼别、摘要、容器或预处理 provenance 冲突 | 整单 fail closed |

输出不包含文件路径、文件名、摘要、像素、embedding 或内部异常文本。

## API 与隐私边界

公开 `/predict` 仍只接受 Y1/Y2 结构化 JSON，并继续返回原有 `demo_synthetic` 响应。`images`、`image_path`、`image_url` 和 `image_base64` 都会作为额外字段被拒绝；仓库没有图像上传端点。

真实研究图像、公开医学数据集、DICOM、研究 checkpoint 和训练得到的图像 encoder 均不在 Sprint 3A 范围内。未来若要处理真实图像，必须另行建立授权、方向/眼别、解码隔离、数据保留、模型验证与 API 版本合同，不能重用本 Demo 的 synthetic provenance 作为许可。

## 可复现证据

```powershell
python scripts/generate_synthetic_fundus.py --check
python scripts/run_multimodal_demo.py --scenario both --human
python scripts/run_multimodal_demo.py --scenario missing-os --human
python scripts/run_multimodal_demo.py --scenario missing-both --human
python scripts/benchmark_multimodal.py
python -m pytest -q tests/test_imaging.py tests/test_fusion.py tests/test_multimodal_demo.py
```

基准只保存三种分支模式的聚合延迟、顺序吞吐量和分支计数，不保存图像、路径、像素、embedding、病例别名、逐例分数或 AUC。
