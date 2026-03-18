# Multi-Agent LangGraph Pipeline
- 本项目为vibe coding项目。

## 项目说明
本项目的核心任务是：基于 `data.pq` 中的特征预测目标列 `Y1`，并比较多模型效果后给出可复现实验结果。
本项目使用 LangGraph 实现一个多智能体机器学习工作流，包含：
- Agent A：数据读取、清洗、特征处理与切分
- Agent B：模型训练（SVM / XGBoost / MLP）
- Agent C：评估可视化与时序泄漏检测

## 时序切分与防泄漏设计
- 当识别到时间列时，训练/测试切分采用**严格时序分组切分**（`time_group_strict`）：
  - 同一时间戳不会同时出现在 train 和 test 两侧
  - 在满足严格时序前提下，尽量逼近目标比例（例如 80/20）
- 当无法识别时间列时，回退为随机切分（`random_split`）
- Agent C 会执行时序泄漏检测并输出报告，包含：
  - 是否检测到泄漏
  - 训练集最大时间与测试集最小时间
  - 重叠样本数

## 主要文件
- `code/multi_agent_langgraph_pipeline.py`：主流程入口
- `code/notebook_helpers.py`：Notebook 调用辅助（运行主流程、加载结果、指标表格化）
- `code/requirements.txt`：依赖列表
- `Q2.ipynb`：可一键运行的 Notebook，含完整执行输出
- `Q2.md`：Agent 系统设计文档（架构图、Prompt、工作流、异常处理）

## 安装依赖
```bash
pip install -r ./code/requirements.txt
```

如果 `xgboost` 下载过慢，可优先使用 conda 安装重依赖，再用 pip 补齐其余包。

## LLM Planner（默认 DeepSeek）
- 默认模型 `deepseek-chat`
- 启动时自动检测并决定是否启用 LLM Planner（无需 Y/n 交互）
- 程序会做预检：
  - API Key 是否存在
  - `openai` SDK 是否可导入
  - 网络是否可达对应 API 域名（443）
- 预检失败会自动回退到规则 Planner，不影响流程跑通
- Notebook 场景建议在安装依赖的同一个 cell 设置 `DEEPSEEK_API_KEY`

### 环境变量
- DeepSeek：`DEEPSEEK_API_KEY`

## 运行方式
```bash
python multi_agent_langgraph_pipeline.py \
  --data_path ./data.pq \
  --target_column Y1 \
  --output_dir ml_pipeline_outputs
```

## 常用参数
- `--test_size`：测试集比例（默认 0.2）
- `--max_features`：SelectKBest 选择特征数（默认 50）
- `--pca_components`：PCA 主成分数（默认 20）
- `--svm_max_iter`：SVM 迭代次数（默认 2000）
- `--mlp_epochs`：MLP 训练轮数（默认 20）
- `--xgb_n_estimators`：XGBoost 轮数（默认 200）
- `--xgb_log_every`：XGBoost 日志步长（默认 20）
- `--planner_model`：Planner 模型名（默认 `deepseek-chat`）
- `--planner_temperature`：Planner 温度（默认 `0.0`）
- `--planner_max_retries`：Planner 计划重试次数（默认 `1`）

## 输出产物
在 `output_dir` 下会生成：
- 数据概览文档：
  - `data_overview.md`（pandas 导出的数据总览，含样本预览、字段类型、缺失统计、描述统计）
- 数据概览图：
  - `target_distribution.png`
  - `missing_all_features.png`
  - `missing_ratio_hist.png`
- 评估图：
  - `*_confusion_matrix.png`
  - `*_roc_curve.png`
  - `model_comparison.png`（AUC / Precision / Recall / F1 对比）
- 报告与结果：
  - `langgraph_pipeline_report.md`（中文报告）
  - `langgraph_pipeline_result.json`（中文字段，指标字段保留英文）
  - 数据泄露检测结果（包含在报告与 JSON 中，字段为“时序泄漏检测”）
  - LLM 计划信息（包含在 JSON 中，字段为“LLM规划”）

### `langgraph_pipeline_report.md` 详细结构
报告会按以下大项输出：
- **(1) 数据与标签**
  - 数据路径、数据形状、目标列等基础信息
- **(2) LLM Planner**
  - 是否启用 LLM Planner、计划校验结果、最终计划内容
- **(3) 预处理决策（Agent A）**
  - 清洗与特征处理摘要（切分方式、缺失处理、缩放、特征筛选、PCA等）
- **(4) 模型指标（Agent B）**
  - 各模型的 AUC / Precision / Recall / F1 / Accuracy
- **(5) Model Evaluation + 数据泄漏检查（Agent C）**
  - 评估文件列表、是否有时间数据、是否检测到泄漏、边界时间与重叠样本
- **(6) 产出文件**
  - 所有评估图与对比图路径清单
- **(7) 运行日志**
  - 全流程关键日志（含 planner 决策与回退信息）

## 日志说明
- 启用 LLM 时会显示 `LLM Planner ...`
- 未启用或预检失败时会显示 `Rule Planner ...`
- Agent A 执行阶段会打印清洗决策摘要（split / impute / scaling / feature_select / pca）
