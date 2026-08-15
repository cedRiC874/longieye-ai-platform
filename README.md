# LongiEye AI Platform

LongiEye 是一个面向作品集的、隐私优先的纵向近视风险建模工程示例。它把硕士论文中的纵向临床建模思路整理成清晰的领域模型、特征管道、可替换推理后端和 FastAPI 服务。

> 当前 `v0.2.0` 仅使用确定性生成的合成数据训练演示模型。输出不是医疗结论，不能用于诊断、筛查或治疗决策。

![LongiEye architecture](docs/assets/architecture.svg)

## 为什么值得放进作品集

- 研究依据可追溯：特征设计来自既有纵向多模态研究，但不包含原始受试者数据。
- 工程边界清楚：领域校验、特征提取、模型推理和 API 分层实现。
- 结果不夸大：所有返回值都标记为 `demo_synthetic`，真实研究指标与演示模型指标分开陈述。
- 可持续演进：后续可以替换为真实交叉验证模型、图像分支和监控组件。

## 当前功能

- Y1/Y2 两次随访数据校验。
- “静态性别编码 + 8项纵向变化量”特征提取。
- 纯 Python 合成数据和双眼逻辑回归演示模型训练。
- OD/OS 双目标风险演示接口。
- FastAPI `/health`、`/ready` 与 `/predict` 端点。
- 统一错误响应、端到端请求 ID 和隐私安全的 JSON 日志。
- P50/P95/P99、顺序吞吐量、进程 RSS 与 Python 峰值内存基准。
- 21项测试、锁定依赖，以及已配置的 Dockerfile 和 GitHub Actions CI。

## 快速开始

```powershell
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pytest -q
uvicorn app.main:app --reload
```

访问 `http://127.0.0.1:8000/docs`，或发送示例请求：

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/predict `
  -ContentType application/json `
  -InFile examples/request.json
```

也可以直接运行不依赖 Web 框架的命令行演示：

```powershell
python scripts/run_demo.py
```

仓库已包含可运行的合成模型 artifact；启动服务不需要重新训练。要验证模型生成的确定性，可运行：

```powershell
python scripts/train_demo_model.py
```

## 项目结构

```text
app/                 FastAPI 入口
configs/             可审阅的演示模型配置
docs/                架构、研究来源与后续路线
examples/            脱敏请求样例
scripts/             合成训练与本地演示脚本
src/longieye/        领域、特征、模型和服务代码
tests/               核心单元测试
```

## API 示例

请求包含两个固定相隔12个月的随访时间点。接口没有姓名、证件号、受试者编号或图像路径字段，任何额外字段都会被拒绝。可选 `case_id` 只允许非识别性别名，调用方不得在其中放入个人信息。

```json
{
  "case_id": "demo-001",
  "followup_months": 12,
  "y1": {
    "sex_code": 0,
    "height_cm": 145.0,
    "weight_kg": 38.0,
    "sbp_mmhg": 105.0,
    "dbp_mmhg": 68.0,
    "waist_cm": 62.0,
    "wears_glasses": 0,
    "axial_length_od_mm": 23.50,
    "axial_length_os_mm": 23.45
  },
  "y2": {
    "sex_code": 0,
    "height_cm": 151.0,
    "weight_kg": 43.0,
    "sbp_mmhg": 108.0,
    "dbp_mmhg": 70.0,
    "waist_cm": 65.0,
    "wears_glasses": 1,
    "axial_length_od_mm": 23.82,
    "axial_length_os_mm": 23.74
  }
}
```

## 隐私与用途边界

- 不提交真实临床表格、受试者标识、图像路径、模型权重或逐人预测。
- 演示模型只证明软件工程流程，不复现论文中的研究性能。
- 接入真实研究模型前，必须完成数据授权、外部验证、校准、偏倚分析和临床治理。

详细说明见 [研究来源](docs/RESEARCH_PROVENANCE.md) 与 [架构设计](docs/ARCHITECTURE.md)。

## Sprint 1 本机基准

基准环境为 Python 3.12.13 / Windows 11；结果来自同进程顺序调用，不包含网络、反向代理、容器或并发负载。

| 路径 | 迭代 | P50 | P95 | P99 | 顺序吞吐量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Core service | 5,000 | 0.008 ms | 0.009 ms | 0.013 ms | 114,052.4 req/s |
| In-process ASGI | 500 | 1.863 ms | 2.631 ms | 3.078 ms | 513.8 req/s |

完整的环境、内存定义和限制见 [基准报告](benchmarks/latest.md)。这些是合成模型的工程指标，不是医疗效果指标。

## 进一步阅读

- [模型卡](docs/MODEL_CARD.md)
- [运行手册](docs/OPERATIONS.md)
- [一分钟演示脚本](docs/DEMO_SCRIPT.md)
- [作品集路线](docs/ROADMAP.md)
