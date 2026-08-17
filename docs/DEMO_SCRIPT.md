# 一分钟作品集演示脚本

## 录制前准备

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

打开 `http://127.0.0.1:8000/docs`，并在另一个标签页准备好 `benchmarks/latest.md`。如果需要命令行画面，可运行：

```powershell
.\.venv\Scripts\python.exe scripts\run_demo.py --human
```

## 60秒中文解说

**0–10秒：** “LongiEye 将我的纵向近视研究整理成一个隐私安全的工程演示。公开服务不包含受试者数据，也不包含研究模型检查点。”

**10–25秒：** 展示 `/predict`。“请求包含两个相隔12个月的随访时间点。领域层会校验输入范围，并提取静态性别编码和八项纵向变化量；球镜等效值和身份标识被明确排除。”

**25–38秒：** 执行示例请求。“同一个请求 ID 会贯穿响应头、JSON 响应和结构化日志。所有输出都明确标记为合成演示结果，不可用于临床。”

**38–50秒：** 展示错误示例。“非法输入会返回稳定的错误码，但不会回显用户提交的值，从而降低敏感信息意外暴露的风险。”

**50–60秒：** 展示基准报告。“仓库包含可复现的 P50、P95、P99 延迟与内存基准、自动化测试、持续集成和模型卡。内部 adapter 合同可替换，但当前公开 API 会硬性拒绝研究 stage，避免把研究输出误标成合成分数。”

不要把合成 AUC 或接口返回的演示分数当作医疗证据。录制前确认画面中没有私有仓库路径、个人通知或其他敏感信息。

## Sprint 2 adapter 演示补充

Sprint 2 的演示重点是“受控接入边界”，不是声称已经转换真实模型，更不是展示真实患者分数。公开录屏只能使用合成 JSON 模型和测试运行时临时生成的合成 PyTorch fixture：

1. 展示 `RiskModelBackend` 如何让公开 Demo 与测试专用 adapter 接受同一特征契约。
2. 展示 manifest 中的 schema、feature-contract version、shape、dtype、三组黄金向量和 SHA-256 校验。
3. 展示 PyTorch CI 测试与 comparison builder 的 schema；当前没有真实研究 comparison report，也不展示不存在的性能数字。
4. 使用测试工件触发 checksum 或 shape 错误，展示 adapter fail closed。
5. 返回 `/health` 或 `/ready`，确认公开 API 仍报告 `demo_synthetic`，没有启用研究模式。
6. 展示[授权清单](RESEARCH_ARTIFACT_AUTHORIZATION.md)，说明包内 manifest 不能自我批准，真实加载还需要绑定全部哈希且位于包外的审批回执。

推荐解说：

> Sprint 2 将模型加载改造成可审计的 adapter 边界。公开 CI 只在临时目录生成合成 state dict，用它验证外部审批绑定、文件清单、特征顺序、shape、dtype、checksum 和黄金向量一致性；真实研究 checkpoint 默认未授权，也没有接入公开 API。这里的 SHA-256 只证明字节一致，包外 JSON 回执也不等于机构数字签名、公开发布授权或临床有效性。

只有 Phase B 获得明确演示许可后，才能在批准范围内展示真实研究模型的聚合证据；本地使用批准不等于录屏或公开发布批准。

## Sprint 3 全合成多模态演示补充

录制前先验证固定图像并准备两个终端：

```powershell
.\.venv\Scripts\python.exe scripts\generate_synthetic_fundus.py --check
.\.venv\Scripts\python.exe scripts\run_multimodal_demo.py --scenario both --human
```

推荐 45 秒补充解说：

**0–10秒：** 展示两张带水印的图片。“这两张 OD/OS 图不是患者影像，也不是下载的数据集，而是固定整数代码生成的 fundus-like fixture。仓库只允许这两个路径和摘要。”

**10–22秒：** 展示 `both` 输出。“严格解码器会检查 PNG chunk、CRC、尺寸、解压上限、眼别像素摘要和工程质控；参考 encoder 只计算五个可审阅统计量，没有训练 CNN。”

**22–34秒：** 运行单眼回退：

```powershell
.\.venv\Scripts\python.exe scripts\run_multimodal_demo.py --scenario missing-os --human
```

“OS 图像缺失后，只回退 OS，并且分数与原结构化模型逐值相同；OD 仍使用有界合成图像修正。回退原因是显式字段，不会静默退化。”

**34–45秒：** 返回 Swagger `/docs`。“多模态路径故意不进入 HTTP。公开请求仍只接收 Y1/Y2 JSON，图像、路径、URL 和 Base64 都被拒绝。这个 Sprint 展示的是多模态合同、质量门和优雅降级，不是医疗效果。”

不要把参考 encoder 称为“图像 AI 模型”，不要声称融合提升准确率，也不要把合成图像分数与论文指标放在一起。录屏只显示固定 fixture、聚合基准和安全标签。
