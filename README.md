# LongiEye AI Platform

> **公开队列上可复现的纵向结局验证管线**

LongiEye `v0.5.0` 将公开真实队列的数据追溯、参与者级交叉验证、校准、决策曲线、多重性校正和安全工件边界整合为一条可重复运行的 AI 验证流程。

[完整验证报告](docs/PUBLIC_COHORT_VALIDATION.md) · [聚合结果 JSON](benchmarks/public_cohort_validation.json) · [运行手册](docs/OPERATIONS.md)

## 核心结果

数据来自 Orinda Longitudinal Study of Myopia（OLSM）公开子集：618 名基线非近视儿童，使用初始检查的右眼眼部、行为和家族史变量，预测五年内新发近视；共 81 例发生结局。

| 模型 | AUC（95% CI） | Brier（95% CI） | 校准截距 | 校准斜率 |
| --- | ---: | ---: | ---: | ---: |
| 仅基线球镜等效值 | 0.859（0.820-0.900） | 0.083（0.069-0.099） | -0.004 | 1.002 |
| 眼部基线模型 | 0.864（0.825-0.902） | 0.083（0.068-0.099） | -0.009 | 0.992 |
| **完整 ridge 模型** | **0.872（0.830-0.911）** | **0.080（0.066-0.096）** | **-0.009** | **0.934** |

> **结论：**完整模型取得最高点估计，但三项配对 AUC 比较经 Holm 校正后均不显著，调整后 `p=1.0000`。项目保留这个阴性结果，而不是只展示最好看的 AUC。

| 校准曲线 | 探索性决策曲线（DCA） |
| --- | --- |
| ![公开队列校准曲线](docs/assets/public_cohort_calibration.svg) | ![公开队列探索性 DCA](docs/assets/public_cohort_dca.svg) |

### 验证设计

- 5 次重复嵌套 5×5 participant-level cross-fitting；预处理和 ridge 选择只在训练折完成。
- 每名参与者的 5 份 OOF 概率取平均，再计算 AUC、Brier、校准和 DCA。
- 2,000 次参与者 bootstrap 给出条件区间；不把它包装成整条训练流水线的重拟合区间。
- 10,000 次配对随机化检验产生原始 p 值，三项模型比较统一使用 Holm FWER 校正。
- 原始 TSV、参与者 ID、折分、逐人 OOF 和冻结参数均留在 Git 忽略的本地 `build/`；仓库只保存聚合结果和曲线。

### 一键复现

```powershell
python -m pip install -r requirements.lock
python -m pip install -r requirements.evaluation.lock
python -m pip install --no-deps -e .
python scripts/run_public_cohort_validation.py --bootstrap 2000
```

### 解释边界

- 这是只有 81 个事件的 pilot 内部验证，不是外部验证，也不是临床部署证据。
- 公开子集只有基线预测变量和五年纵向结局，不包含重复 Y1/Y2 预测变量，因此不是私有论文九维变化量模型的复现。
- DCA 对应假设性的“加强随访”动作，没有独立临床效用研究，只能视为探索性分析。
- `aplore3` 分发包的 GPL-3 已确认，但底层 OLSM 队列数据的再利用权未由本项目独立核验；商业使用或公开冻结参数前仍需人工权利审查。

## 为什么值得放进作品集

- **结果可复现：**数据 URL、commit、SHA-256、随机种子、验证配置和聚合产物均被固定。
- **统计流程完整：**不仅报告 AUC，还包含 Brier、校准截距/斜率、带区间的校准曲线、DCA 和多重性校正。
- **防止数据泄漏：**参与者级分折、折内预处理、唯一 OOF 预测和严格原始数据门禁都有自动化测试。
- **不夸大结果：**明确区分内部验证、外部验证、临床效用和软件工程正确性，并公开阴性模型比较。
- **工程可交付：**155 项测试、Python 3.10/3.12 评估 CI、锁定依赖、隔离 wheel 验证和公开工件扫描均已配置。

## 工程系统与安全边界

![LongiEye 系统架构图](docs/assets/architecture.svg)

公开 `/predict` 仍保留原有的合成 Y1/Y2 九维结构化合同，作为 API、错误边界、日志、Docker 和模型门禁的工程演示；真实 OLSM 验证不会静默替换不兼容的 HTTP 合同。所有输出均为非临床结果，不能用于诊断、筛查或治疗决策。

## 当前功能

- OLSM 公开真实队列的五年新发近视内部验证。
- 重复嵌套交叉验证、participant bootstrap、校准、探索性 DCA 与 Holm 多重性校正。
- 固定下载摘要、原始数据防误提交、聚合报告和确定性 SVG 曲线。
- Y1/Y2 两次随访领域校验与“静态性别编码 + 8 项纵向变化量”合成服务合同。
- FastAPI `/health`、`/ready` 与 `/predict`，统一错误响应、请求 ID 和隐私安全 JSON 日志。
- 研究模型 manifest、包外审批、SHA-256、权重结构和黄金向量 fail-closed 门禁。
- 严格 canonical PNG、合成眼底图质控、确定性图像编码与逐眼精确回退。
- P50/P95/P99、顺序吞吐量、进程 RSS 与 Python 峰值内存基准。
- 155 项测试、锁定依赖、Dockerfile 和 GitHub Actions CI。

## Sprint 3：全合成多模态扩展

Sprint 3A 已实现一条独立的离线多模态路径：现有九维纵向结构化模型提供锚点分数，两张代码生成的 OD/OS 合成眼底示意图经过完整性绑定、质量门和确定性统计编码器，再产生最大绝对值为 `0.35` 的逐眼 logit 修正。图像缺失或质控失败时，该眼逐值精确回退到原结构化结果。

| 全合成 OD fixture | 全合成 OS fixture |
| --- | --- |
| ![全合成 OD 眼底示意图](examples/synthetic_fundus/od.png) | ![全合成 OS 眼底示意图](examples/synthetic_fundus/os.png) |

两张图都由固定整数绘图规则生成并带有 `SYNTHETIC OD/OS` 水印，不来自患者或医学数据集。CI guardrail 通过扩展名、常见文件头和精确摘要拒绝其他影像或容器工件；它用于降低误提交风险，但不等同于内容识别 DLP。

多模态功能不会改变公开 `/predict`：HTTP 仍只接收结构化 Y1/Y2 JSON，`images`、图像路径、URL 和 Base64 字段都会被拒绝。离线 adapter 使用 `demo_multimodal_synthetic` stage，因此也无法被现有 `RiskPredictionService` 静默装入公开服务。详细边界见[全合成多模态 Demo Card](docs/MULTIMODAL_DEMO_CARD.md)。

## Sprint 2：研究模型适配边界

Sprint 2 的目标是证明“用于接入经授权研究模型”的 adapter、manifest、完整性校验和黄金向量自检流程可以运行。当前没有转换或部署真实研究 checkpoint；公开服务的非临床语义保持不变。

Phase A 的实现与 GitHub Actions 验证均已完成。Phase B 仍因真实工件未授权而关闭。

研究工件的默认状态是 **未授权**。公开仓库不得包含真实 checkpoint、预处理统计、受试者记录、拆分 ID、逐人预测或 OOF 输出。即使本地存在经过授权的研究工件，公开 `/predict` 仍只运行合成 JSON 模型；研究适配与比较在隔离的离线路径中完成。

| 组件 | 当前/目标状态 | 是否进入公开仓库 | 是否由公开 API 使用 |
| --- | --- | --- | --- |
| 合成 JSON 模型 | 当前启用、完全可复现 | 是 | 是 |
| `RiskModelBackend` 协议 | Phase A 公共适配基础 | 是 | 仅约束内部后端 |
| `ResearchModelAdapter` | Phase A 公共 adapter 与严格校验 | 是，不含真实权重 | 否 |
| 内部 `_TorchStateDictRuntime` | Phase A 惰性可选 PyTorch runtime；只能由已验证研究包构造，默认安装不要求 torch | 是 | 否 |
| 合成 PyTorch state dict | CI/pytest 运行时临时生成 | 否；仅生成逻辑与测试代码公开 | 否 |
| 真实研究工件 | Phase B，默认未授权 | **否** | **未启用** |
| 临床有效性与真实世界部署 | 未建立 | 不适用 | 不允许 |

### 作品集证据索引

| 能力主张 | 可审计证据 | 边界 |
| --- | --- | --- |
| 可复现合成服务 | [公开合成模型卡](docs/MODEL_CARD.md)、测试与基准报告 | 只证明工程链路 |
| 研究证据链设计 | [研究来源与隐私边界](docs/RESEARCH_PROVENANCE.md) | 私有 commit 与证据位置必须留在受控登记中 |
| 工件加载受控 | [研究工件授权清单](docs/RESEARCH_ARTIFACT_AUTHORIZATION.md) | 包不能自我批准；需包外审批策略与绑定回执 |
| 研究模型说明规范 | [研究模型卡模板](docs/RESEARCH_MODEL_CARD_TEMPLATE.md) | 未填完和未批准时不得发布指标或工件 |
| Adapter 合同与合成等价性 | 多黄金向量、checksum/shape/dtype/NaN 失败测试和 comparison builder | 合成 fixture 不能证明真实 checkpoint 已转换 |
| 已加载 adapter 计时能力 | 严格分区 comparison builder 的顺序 P50/P95/P99 | 通用 builder 本身不验证授权；标准 CLI 先通过包门禁。尚无真实 PyTorch 报告，不含冷加载、RSS 或生产 SLA |
| 全合成多模态工程 | canonical PNG、OD/OS 像素摘要、质量门、参考 encoder、逐眼回退测试和聚合基准 | 没有真实图像、CNN 训练或多模态效果主张 |
| 学术研究结果 | 固定 commit、聚合结果文件和论文表格的逐项映射 | 与合成 AUC、adapter 指标严格分开 |

## 快速开始

```powershell
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pytest -q
uvicorn app.main:app --reload
```

默认轻量环境不安装 PyTorch 或 NumPy；当前基线为 `141 passed, 6 skipped`。安装研究适配和公开队列依赖后的完整基线为 `155 passed`。GitHub 的独立 research-adapter 与 public-cohort-validation jobs 会要求对应测试真实运行，依赖缺失时不能假绿。

访问 `http://127.0.0.1:8000/docs` 查看中文接口说明，或发送示例请求。HTTP 路径和 JSON 字段名保留英文，以维持稳定的机器合同：

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/predict `
  -ContentType application/json `
  -InFile examples/request.json
```

也可以直接运行不依赖 Web 框架的命令行演示：

```powershell
python scripts/run_demo.py --human
```

默认不加 `--human` 时仍输出适合程序解析的原始 JSON。

运行全合成多模态演示与回退场景：

```powershell
python scripts/generate_synthetic_fundus.py --check
python scripts/run_multimodal_demo.py --scenario both --human
python scripts/run_multimodal_demo.py --scenario missing-os --human
python scripts/run_multimodal_demo.py --scenario missing-both --human
```

这条路径仅在仓库内离线运行，不会启动图像上传 API，也不会读取任意外部图片。

Sprint 2 的公开 manifest 模板可以在完全不读取 checkpoint 的情况下校验：

```powershell
python scripts/validate_research_manifest.py
python scripts/check_public_artifacts.py
```

独立的 GitHub Actions 任务会安装 `requirements.research.lock` 中的 CPU-only PyTorch 2.13.0，并在临时目录生成合成 state dict，验证 manifest、SHA-256、预处理、OD/OS 映射和失败路径。该依赖不会进入默认安装或 Docker 镜像。

如需在本机运行同一组可选测试：

```powershell
python -m pip install -r requirements.research.lock
$env:LONGIEYE_REQUIRE_TORCH = "1"
python -m pytest -q tests/test_model_contract.py tests/test_research_adapter.py tests/test_comparison.py
```

仓库已包含可运行的合成模型制品（artifact）；启动服务不需要重新训练。要验证模型生成的确定性，可运行：

```powershell
python scripts/train_demo_model.py
```

## 项目结构

```text
app/                 FastAPI 入口
configs/             可审阅的演示模型配置与未授权 research manifest 模板
docs/                架构、研究来源与后续路线
examples/            脱敏请求与两张固定全合成眼底 fixture
scripts/             合成训练、图像生成、工件检查、离线演示与基准脚本
src/longieye/        领域、特征、模型合同、图像/融合 adapter 和服务代码
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
- 公开队列原始行、participant ID、折分、OOF 预测与冻结模型参数仅保存在忽略的本地 `build/`；Git 只保存聚合指标和曲线。
- 不提交真实眼底图、任意栅格图像或医学影像；固定 fixture 必须与生成器和摘要 registry 一致。
- 演示模型只证明软件工程流程，不复现论文中的研究性能。
- 离线 Phase B parity 前必须完成所有权、隐私与工件使用授权。
- 任何公开或临床路径启用还必须另行完成外部验证、校准、偏倚分析和临床治理。

详细说明见 [研究来源](docs/RESEARCH_PROVENANCE.md) 与 [架构设计](docs/ARCHITECTURE.md)。

## Sprint 1 本机基准

基准环境为 Python 3.12.13 / Windows 11；结果来自同进程顺序调用，不包含网络、反向代理、容器或并发负载。

| 路径 | 迭代 | P50 | P95 | P99 | 顺序吞吐量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 核心服务（core_service） | 5,000 | 0.009 ms | 0.015 ms | 0.020 ms | 99,169.4 req/s |
| 进程内 ASGI（in_process_asgi） | 500 | 2.497 ms | 3.902 ms | 4.704 ms | 371.5 req/s |

完整的环境、内存定义和限制见 [基准报告](benchmarks/latest.md)。这些是合成模型的工程指标，不是医疗效果指标。

## Sprint 3 多模态回退基准

`benchmarks/multimodal_latest.md` 分别测量双眼合成图像、缺失 OS 和双眼缺失三种离线路径。计时包含质控、32×32 预处理、确定性统计编码和有界融合，不包含文件读取，也不保存图像、路径、embedding 或逐例分数。

完整结果见[全合成多模态基准](benchmarks/multimodal_latest.md)。这些数字只表示本机工程开销，不是多模态模型效果或临床性能。

## 许可证与第三方数据

除另有说明外，本仓库源代码和项目文档按 [GNU General Public License v3.0 only](LICENSE)（`GPL-3.0-only`）发布。分发修改版本时，需要遵守 GPLv3 对源代码提供、许可证保留和修改声明等要求。

本许可证只覆盖本仓库有权许可的内容，不会改变第三方依赖或数据集的原始条款。OLSM 原始参与者数据不进入 Git；`aplore3` 分发包的 GPL-3 已确认，但底层队列数据的独立再利用授权仍需在商业使用、冻结参数公开或临床扩展前另行核验。

## 进一步阅读

- [模型卡](docs/MODEL_CARD.md)
- [研究工件授权清单](docs/RESEARCH_ARTIFACT_AUTHORIZATION.md)
- [研究模型卡模板](docs/RESEARCH_MODEL_CARD_TEMPLATE.md)
- [全合成多模态 Demo Card](docs/MULTIMODAL_DEMO_CARD.md)
- [公开纵向队列验证报告](docs/PUBLIC_COHORT_VALIDATION.md)
- [运行手册](docs/OPERATIONS.md)
- [一分钟演示脚本](docs/DEMO_SCRIPT.md)
- [作品集路线](docs/ROADMAP.md)
