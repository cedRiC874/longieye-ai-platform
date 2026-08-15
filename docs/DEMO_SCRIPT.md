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
