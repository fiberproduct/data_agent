# Multi-Agent LangGraph Pipeline Report

## (1) Data and Target
- Data Path: ../data.pq
- Data Shape: [81046, 321]
- Target Column: Y1

## (2) LLM Planner
- LLM Planner Enabled: True
- Plan Validation: {'valid': True, 'issues': []}
- Plan Content: {'split': {'method': 'time_group_strict', 'time_column': 'trade_date', 'train_ratio': 0.7, 'val_ratio': 0.15, 'test_ratio': 0.15, 'shuffle': False, 'stratify': False, 'test_size': 0.2}, 'categorical': {'method': 'none', 'columns': [], 'encoding': 'onehot', 'drop_first': True}, 'impute': {'method': 'iterative', 'max_iter': 10, 'initial_strategy': 'median', 'random_state': 42, 'strategy': 'median'}, 'scaling': {'method': 'robust', 'quantile_range': [25, 75], 'use_standard': True, 'use_minmax': True}, 'feature_select': {'method': 'variance_threshold', 'threshold': 0.01, 'after_impute': True, 'enabled': False, 'k': 50}, 'pca': {'method': 'optional', 'n_components': 0.95, 'whiten': False, 'apply_after_scaling': True, 'enabled': True}, 'plan_source': 'llm:deepseek:deepseek-chat'}

## (3) Preprocessing Decisions (Agent A)
- {'decision': 'plan_driven_preprocessing', 'plan_source': 'llm:deepseek:deepseek-chat', 'plan': {'split': {'method': 'time_group_strict', 'time_column': 'trade_date', 'train_ratio': 0.7, 'val_ratio': 0.15, 'test_ratio': 0.15, 'shuffle': False, 'stratify': False, 'test_size': 0.2}, 'categorical': {'method': 'none', 'columns': [], 'encoding': 'onehot', 'drop_first': True}, 'impute': {'method': 'iterative', 'max_iter': 10, 'initial_strategy': 'median', 'random_state': 42, 'strategy': 'median'}, 'scaling': {'method': 'robust', 'quantile_range': [25, 75], 'use_standard': True, 'use_minmax': True}, 'feature_select': {'method': 'variance_threshold', 'threshold': 0.01, 'after_impute': True, 'enabled': False, 'k': 50}, 'pca': {'method': 'optional', 'n_components': 0.95, 'whiten': False, 'apply_after_scaling': True, 'enabled': True}, 'plan_source': 'llm:deepseek:deepseek-chat'}, 'split_info': {'split_method': 'time_group_strict', 'target_train_ratio': 0.8, 'actual_train_ratio': 0.799965451718777, 'boundary_time': '2019-12-26 00:00:00', 'train_count': 64834, 'test_count': 16212}, 'select_k': 50, 'pca_components': 0, 'explained_variance_ratio': 0.9500874991576264, 'selected_feature_count': 381}

## (4) Model Metrics (Agent B)
- svm: AUC=0.9078, Precision=0.7548, Recall=0.5793, F1=0.5474, Accuracy=0.7606
- xgboost: AUC=0.9426, Precision=0.7605, Recall=0.7744, F1=0.7652, Accuracy=0.8164
- mlp: AUC=0.9408, Precision=0.7540, Recall=0.8144, F1=0.7796, Accuracy=0.8173

## (5) Model Evaluation and Data Leakage Check (Agent C)
- Model Evaluation Files Directory: ml_pipeline_outputs
- {'has_time_data': True, 'leakage_detected': False, 'max_train_time': '2019-12-25 00:00:00', 'min_test_time': '2019-12-26 00:00:00', 'overlap_count': 0, 'message': 'No leakage detected'}

## (6) Output Files
- ml_pipeline_outputs/svm_confusion_matrix.png
- ml_pipeline_outputs/svm_roc_curve.png
- ml_pipeline_outputs/xgboost_confusion_matrix.png
- ml_pipeline_outputs/xgboost_roc_curve.png
- ml_pipeline_outputs/mlp_confusion_matrix.png
- ml_pipeline_outputs/mlp_roc_curve.png
- ml_pipeline_outputs/model_comparison.png

## (7) Run Logs
- [2026-03-01T11:05:17] Read data started: loading parquet from ../data.pq
- [2026-03-01T11:05:17] Read parquet done: rows=81046, cols=321, x_features=300
- [2026-03-01T11:05:17] Output directory ready: ml_pipeline_outputs
- [2026-03-01T11:05:17] Generating target distribution figure...
- [2026-03-01T11:05:17] Target distribution figure done
- [2026-03-01T11:05:17] Building pandas data overview markdown...
- [2026-03-01T11:05:18] Data overview markdown done: data_overview.md
- [2026-03-01T11:05:18] Computing missing ratio over all X features...
- [2026-03-01T11:05:19] Missing ratio all-features bar chart done
- [2026-03-01T11:05:19] Generating missing ratio histogram...
- [2026-03-01T11:05:19] Missing ratio histogram done
- [2026-03-01T11:05:19] Skipping missingness correlation heatmap by design
- [2026-03-01T11:05:19] Read data complete: shape=(81046, 321), target=Y1
- [2026-03-01T11:05:19] LLM Planner started
- [2026-03-01T11:05:27] LLM Planner plan accepted (attempt=1)
- [2026-03-01T11:05:27] LLM Planner complete: source=llm:deepseek:deepseek-chat
- [2026-03-01T11:05:27] Agent A started
- [2026-03-01T11:05:27] Planner decision snapshot: {'split': {'method': 'time_group_strict', 'time_column': 'trade_date', 'train_ratio': 0.7, 'val_ratio': 0.15, 'test_ratio': 0.15, 'shuffle': False, 'stratify': False, 'test_size': 0.2}, 'categorical': {'method': 'none', 'columns': [], 'encoding': 'onehot', 'drop_first': True}, 'impute': {'method': 'iterative', 'max_iter': 10, 'initial_strategy': 'median', 'random_state': 42, 'strategy': 'median'}, 'scaling': {'method': 'robust', 'quantile_range': [25, 75], 'use_standard': True, 'use_minmax': True}, 'feature_select': {'method': 'variance_threshold', 'threshold': 0.01, 'after_impute': True, 'enabled': False, 'k': 50}, 'pca': {'method': 'optional', 'n_components': 0.95, 'whiten': False, 'apply_after_scaling': True, 'enabled': True}, 'plan_source': 'llm:deepseek:deepseek-chat'}
- [2026-03-01T11:05:27] Cleaning decision - split: method=time_group_strict, test_size=0.2
- [2026-03-01T11:05:27] Agent A strict time split: train=64834, test=16212, target_ratio=0.8000, actual_ratio=0.8000, boundary=2019-12-26 00:00:00
- [2026-03-01T11:05:27] Agent A split complete: train=(64834, 385), test=(16212, 385)
- [2026-03-01T11:05:27] Cleaning decision - impute: strategy=median
- [2026-03-01T11:05:32] Cleaning decision - scaling: standard=True, minmax=True
- [2026-03-01T11:05:32] Agent A variance filter: 385 -> 381
- [2026-03-01T11:05:32] Cleaning decision - feature_select: enabled=False, method=variance_threshold, k=50
- [2026-03-01T11:05:32] Agent A SelectKBest skipped by plan or k >= remaining features
- [2026-03-01T11:05:32] Cleaning decision - pca: enabled=True, n_components=0.95
- [2026-03-01T11:05:33] Agent A PCA complete: components=0.95
- [2026-03-01T11:05:33] Agent A complete: train=(64834, 102), test=(16212, 102), selected=381
- [2026-03-01T11:05:33] Agent B started
- [2026-03-01T11:05:33] Agent B device check: prefer=cuda, torch_cuda=True, cuml=False, xgboost=True
- [2026-03-01T11:05:33] Agent B data ready: X_train=(64834, 102), X_test=(16212, 102), classes=3
- [2026-03-01T11:05:33] Agent B training SVM... (iterative, max_iter=2000, log_every=100)
- [2026-03-01T11:05:33] Agent B SVM iter 1/2000
- [2026-03-01T11:05:40] Agent B SVM iter 100/2000
- [2026-03-01T11:05:46] Agent B SVM iter 200/2000
- [2026-03-01T11:05:53] Agent B SVM iter 300/2000
- [2026-03-01T11:06:00] Agent B SVM iter 400/2000
- [2026-03-01T11:06:06] Agent B SVM iter 500/2000
- [2026-03-01T11:06:13] Agent B SVM iter 600/2000
- [2026-03-01T11:06:20] Agent B SVM iter 700/2000
- [2026-03-01T11:06:27] Agent B SVM iter 800/2000
- [2026-03-01T11:06:33] Agent B SVM iter 900/2000
- [2026-03-01T11:06:40] Agent B SVM iter 1000/2000
- [2026-03-01T11:06:47] Agent B SVM iter 1100/2000
- [2026-03-01T11:06:54] Agent B SVM iter 1200/2000
- [2026-03-01T11:07:00] Agent B SVM iter 1300/2000
- [2026-03-01T11:07:07] Agent B SVM iter 1400/2000
- [2026-03-01T11:07:14] Agent B SVM iter 1500/2000
- [2026-03-01T11:07:20] Agent B SVM iter 1600/2000
- [2026-03-01T11:07:27] Agent B SVM iter 1700/2000
- [2026-03-01T11:07:34] Agent B SVM iter 1800/2000
- [2026-03-01T11:07:41] Agent B SVM iter 1900/2000
- [2026-03-01T11:07:47] Agent B SVM iter 2000/2000
- [2026-03-01T11:07:47] Agent B SVM done
- [2026-03-01T11:07:47] Agent B training XGBoost...
- [2026-03-01T11:07:47] Agent B XGBoost backend: GPU(cuda)
- [2026-03-01T11:07:50] Agent B XGBoost trained rounds=200, elapsed=2.20s
- [2026-03-01T11:07:50] Agent B XGBoost done
- [2026-03-01T11:07:50] Agent B training MLP...
- [2026-03-01T11:07:52] Agent B MLP epoch 1/20, loss=0.611496
- [2026-03-01T11:07:53] Agent B MLP epoch 2/20, loss=0.322023
- [2026-03-01T11:07:54] Agent B MLP epoch 3/20, loss=0.282485
- [2026-03-01T11:07:54] Agent B MLP epoch 4/20, loss=0.277312
- [2026-03-01T11:07:55] Agent B MLP epoch 5/20, loss=0.272877
- [2026-03-01T11:07:57] Agent B MLP epoch 6/20, loss=0.271302
- [2026-03-01T11:07:58] Agent B MLP epoch 7/20, loss=0.268327
- [2026-03-01T11:07:59] Agent B MLP epoch 8/20, loss=0.266453
- [2026-03-01T11:08:00] Agent B MLP epoch 9/20, loss=0.264905
- [2026-03-01T11:08:01] Agent B MLP epoch 10/20, loss=0.263677
- [2026-03-01T11:08:02] Agent B MLP epoch 11/20, loss=0.261307
- [2026-03-01T11:08:03] Agent B MLP epoch 12/20, loss=0.259531
- [2026-03-01T11:08:04] Agent B MLP epoch 13/20, loss=0.258855
- [2026-03-01T11:08:05] Agent B MLP epoch 14/20, loss=0.256650
- [2026-03-01T11:08:06] Agent B MLP epoch 15/20, loss=0.255447
- [2026-03-01T11:08:07] Agent B MLP epoch 16/20, loss=0.252899
- [2026-03-01T11:08:08] Agent B MLP epoch 17/20, loss=0.252143
- [2026-03-01T11:08:09] Agent B MLP epoch 18/20, loss=0.250391
- [2026-03-01T11:08:10] Agent B MLP epoch 19/20, loss=0.249911
- [2026-03-01T11:08:11] Agent B MLP epoch 20/20, loss=0.248544
- [2026-03-01T11:08:11] Agent B MLP done
- [2026-03-01T11:08:11] Agent B complete: models=['svm', 'xgboost', 'mlp'], best=mlp
- [2026-03-01T11:08:11] Agent C started
- [2026-03-01T11:08:11] Agent C plotting svm...
- [2026-03-01T11:08:11] Agent C plotting xgboost...
- [2026-03-01T11:08:12] Agent C plotting mlp...