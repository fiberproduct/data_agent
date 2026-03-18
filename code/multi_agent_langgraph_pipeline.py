from __future__ import annotations

import argparse
import json
import os
import socket
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, StateGraph
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, label_binarize
from sklearn.svm import SVC

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import xgboost as xgb
except Exception:
    xgb = None
    XGB_IMPORT_ERROR = traceback.format_exc()
else:
    XGB_IMPORT_ERROR = None

try:
    from cuml.svm import SVC as cuSVC
except Exception:
    cuSVC = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


class PipelineState(TypedDict, total=False):
    data_path: str
    target_column: str
    output_dir: str
    test_size: float
    random_state: int
    max_features: int
    pca_components: int
    xgb_n_estimators: int
    xgb_log_every: int
    use_llm_planner: bool

    logs: List[str]
    errors: List[str]

    raw_shape: List[int]
    feature_columns: List[str]
    time_column: Optional[str]
    data_profile: Dict[str, Any]
    preprocessing_plan: Dict[str, Any]
    plan_validation: Dict[str, Any]

    preprocessed: Dict[str, Any]
    training_results: Dict[str, Any]
    evaluation_results: Dict[str, Any]

    final_report_path: str
    final_json_path: str


@dataclass
class Config:
    data_path: str
    target_column: str
    output_dir: str
    test_size: float = 0.2
    random_state: int = 42
    max_features: int = 50
    pca_components: int = 20
    train_sample_max: int = 120000
    svm_max_iter: int = 2000
    mlp_epochs: int = 20
    xgb_n_estimators: int = 200
    xgb_log_every: int = 20
    planner_model: str = "deepseek-chat"
    planner_temperature: float = 0.0
    planner_max_retries: int = 1
    use_llm_planner: bool = True


def _append_log(state: PipelineState, message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    state.setdefault("logs", []).append(line)
    print(line, flush=True)


def _append_error(state: PipelineState, message: str) -> None:
    state.setdefault("errors", []).append(message)
    _append_log(state, f"ERROR: {message}")


def _find_time_column(df: pd.DataFrame) -> Optional[str]:
    priority = ["trade_date", "start_time", "end_time", "date", "time", "timestamp"]
    cols = [c for c in df.columns]
    for p in priority:
        for c in cols:
            if p == c.lower() or p in c.lower():
                return c
    return None


def _parse_time_series(df: pd.DataFrame, time_col: Optional[str]) -> Optional[pd.Series]:
    if not time_col or time_col not in df.columns:
        return None
    raw_col = df[time_col]
    if isinstance(raw_col, pd.DataFrame):
        raw_col = raw_col.iloc[:, 0]
    s = pd.to_datetime(raw_col, errors="coerce")
    if s.isna().all() and "trade_date" in df.columns and "start_time" in df.columns:
        merged = (
            df["trade_date"].astype(str).str.slice(0, 10)
            + " "
            + df["start_time"].astype(str)
        )
        s = pd.to_datetime(merged, errors="coerce")
    if s.isna().all():
        return None
    return s.ffill().bfill()


def _time_group_strict_split(
    dt_series: pd.Series,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    # Split by unique timestamps so one timestamp never appears in both train and test.
    order = np.argsort(dt_series.values)
    dt_sorted = dt_series.iloc[order].reset_index(drop=True)
    n = len(dt_sorted)
    target_train = max(1, int(n * (1 - test_size)))

    grp = dt_sorted.groupby(dt_sorted, sort=True).size().reset_index(name="cnt")
    grp["cum"] = grp["cnt"].cumsum()

    # Candidate A: first boundary where cumulative >= target_train
    cand_a_idx = int(np.searchsorted(grp["cum"].to_numpy(), target_train, side="left"))
    cand_a_idx = min(max(cand_a_idx, 0), len(grp) - 1)
    train_cnt_a = int(grp.iloc[cand_a_idx]["cum"])
    test_cnt_a = n - train_cnt_a

    # Candidate B: previous boundary, if exists
    if cand_a_idx > 0:
        train_cnt_b = int(grp.iloc[cand_a_idx - 1]["cum"])
        test_cnt_b = n - train_cnt_b
    else:
        train_cnt_b = 0
        test_cnt_b = n

    # Pick boundary with smaller ratio error and ensure both sets are non-empty.
    ratio_a_err = abs(train_cnt_a / n - (1 - test_size)) if test_cnt_a > 0 else 1e9
    ratio_b_err = abs(train_cnt_b / n - (1 - test_size)) if train_cnt_b > 0 and test_cnt_b > 0 else 1e9
    use_b = ratio_b_err < ratio_a_err
    split_train_count = train_cnt_b if use_b else train_cnt_a

    # Safety fallback
    split_train_count = min(max(split_train_count, 1), n - 1)
    train_sorted_idx = order[:split_train_count]
    test_sorted_idx = order[split_train_count:]

    info = {
        "split_method": "time_group_strict",
        "target_train_ratio": float(1 - test_size),
        "actual_train_ratio": float(split_train_count / n),
        "boundary_time": str(dt_sorted.iloc[split_train_count]),
        "train_count": int(len(train_sorted_idx)),
        "test_count": int(len(test_sorted_idx)),
    }
    return train_sorted_idx, test_sorted_idx, info


def _build_data_profile(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    time_column: Optional[str],
) -> Dict[str, Any]:
    dtype_counts = df.dtypes.astype(str).value_counts().to_dict()
    miss = df[feature_columns].isna().mean() if feature_columns else pd.Series(dtype=float)
    profile = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "target_column": target_column,
        "time_column": time_column,
        "feature_count": int(len(feature_columns)),
        "dtype_counts": {str(k): int(v) for k, v in dtype_counts.items()},
        "target_classes": sorted(pd.Series(df[target_column]).dropna().unique().tolist()) if target_column in df.columns else [],
        "target_distribution": (
            pd.Series(df[target_column]).value_counts(dropna=False).to_dict()
            if target_column in df.columns
            else {}
        ),
        "missing_ratio_summary": {
            "mean": float(miss.mean()) if len(miss) else 0.0,
            "max": float(miss.max()) if len(miss) else 0.0,
            "min": float(miss.min()) if len(miss) else 0.0,
        },
        "top_missing_features": (
            miss.sort_values(ascending=False).head(20).to_dict()
            if len(miss)
            else {}
        ),
    }
    return profile


def _default_preprocessing_plan(state: PipelineState) -> Dict[str, Any]:
    return {
        "plan_source": "default_rule",
        "split": {
            "method": "time_group_strict" if state.get("time_column") else "random_split",
            "test_size": float(state.get("test_size", 0.2)),
        },
        "categorical": {"encoding": "onehot", "drop_first": True},
        "impute": {"strategy": "median"},
        "scaling": {"use_standard": True, "use_minmax": True},
        "feature_select": {"enabled": True, "method": "selectkbest", "k": int(state.get("max_features", 50))},
        "pca": {"enabled": True, "n_components": int(state.get("pca_components", 20))},
    }


def _normalize_preprocessing_plan(plan: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
    p = dict(plan)

    split = dict(p.get("split", {}))
    split_method = str(split.get("method", "")).strip().lower()
    split_map = {
        "time_based": "time_group_strict",
        "time": "time_group_strict",
        "chronological": "time_group_strict",
        "time_group": "time_group_strict",
        "random": "random_split",
    }
    if split_method in split_map:
        split["method"] = split_map[split_method]
    if "test_size" not in split:
        split["test_size"] = float(state.get("test_size", 0.2))
    p["split"] = split

    fs = dict(p.get("feature_select", {}))
    fs_method = str(fs.get("method", "selectkbest")).strip().lower()
    fs_map = {
        "kbest": "selectkbest",
        "select_k_best": "selectkbest",
        "selectkbest": "selectkbest",
        "variance_threshold": "variance_threshold",
    }
    fs_method = fs_map.get(fs_method, fs_method)
    fs["method"] = fs_method
    if fs_method == "variance_threshold":
        # Pipeline already runs VarianceThreshold before SelectKBest.
        fs["enabled"] = False
    if "k" not in fs:
        fs["k"] = int(state.get("max_features", 50))
    p["feature_select"] = fs

    pca = dict(p.get("pca", {}))
    if "enabled" not in pca:
        pca["enabled"] = True
    if "n_components" not in pca:
        pca["n_components"] = int(state.get("pca_components", 20))
    p["pca"] = pca

    cat = dict(p.get("categorical", {}))
    if "encoding" not in cat:
        cat["encoding"] = "onehot"
    if "drop_first" not in cat:
        cat["drop_first"] = True
    p["categorical"] = cat

    impute = dict(p.get("impute", {}))
    if "strategy" not in impute:
        impute["strategy"] = "median"
    p["impute"] = impute

    scaling = dict(p.get("scaling", {}))
    if "use_standard" not in scaling:
        scaling["use_standard"] = True
    if "use_minmax" not in scaling:
        scaling["use_minmax"] = True
    p["scaling"] = scaling

    return p


def _validate_preprocessing_plan(plan: Dict[str, Any], state: PipelineState) -> tuple[bool, List[str]]:
    issues: List[str] = []
    if not isinstance(plan, dict):
        return False, ["plan must be a dict"]

    split = plan.get("split", {})
    method = split.get("method")
    if method not in {"time_group_strict", "random_split"}:
        issues.append(f"split.method invalid: {method}")
    ts = split.get("test_size", state.get("test_size", 0.2))
    try:
        ts_val = float(ts)
        if ts_val <= 0 or ts_val >= 1:
            issues.append(f"split.test_size out of range: {ts_val}")
    except Exception:
        issues.append(f"split.test_size invalid: {ts}")

    fs = plan.get("feature_select", {})
    if fs.get("enabled", True):
        if fs.get("method") not in {"selectkbest", "variance_threshold"}:
            issues.append(f"feature_select.method invalid: {fs.get('method')}")
        k = fs.get("k", state.get("max_features", 50))
        try:
            if int(k) <= 0:
                issues.append(f"feature_select.k invalid: {k}")
        except Exception:
            issues.append(f"feature_select.k invalid: {k}")

    pca = plan.get("pca", {})
    if pca.get("enabled", True):
        n_comp = pca.get("n_components", state.get("pca_components", 20))
        try:
            n_comp_f = float(n_comp)
            if 0 < n_comp_f < 1:
                pass  # explained variance ratio mode for PCA
            elif int(n_comp_f) <= 0:
                issues.append(f"pca.n_components invalid: {n_comp}")
        except Exception:
            issues.append(f"pca.n_components invalid: {n_comp}")

    return len(issues) == 0, issues


def _try_generate_llm_plan(state: PipelineState) -> Optional[Dict[str, Any]]:
    if not bool(state.get("use_llm_planner", True)):
        return None
    provider = "deepseek"
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI  # Optional runtime dependency
    except Exception:
        return None

    profile = state.get("data_profile", {})
    system_msg = (
        "You are an ML preprocessing planner. Return ONLY valid JSON with keys: "
        "split, categorical, impute, scaling, feature_select, pca. "
        "Do not include markdown or prose."
    )
    user_msg = (
        f"Target={state.get('target_column')}, time_column={state.get('time_column')}, "
        f"profile={json.dumps(profile, ensure_ascii=False)}. "
        f"Need robust preprocessing for classification with no temporal leakage."
    )

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    model = str(state.get("planner_model", "deepseek-chat"))
    temperature = float(state.get("planner_temperature", 0.0))
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    content = resp.choices[0].message.content or ""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        parsed["plan_source"] = f"llm:{provider}:{model}"
        return parsed
    return None


def node_agent_a_plan(state: PipelineState) -> PipelineState:
    try:
        planner_label = "LLM Planner" if bool(state.get("use_llm_planner", True)) else "Rule Planner"
        _append_log(state, f"{planner_label} started")
        if "preprocessed" not in state or "raw_df" not in state["preprocessed"]:
            raise ValueError("raw_df missing before planner")

        df = state["preprocessed"]["raw_df"]
        profile = _build_data_profile(
            df=df,
            target_column=state["target_column"],
            feature_columns=state.get("feature_columns", []),
            time_column=state.get("time_column"),
        )
        state["data_profile"] = profile

        plan = None
        max_retries = int(state.get("planner_max_retries", 1))
        for attempt in range(max_retries):
            plan = _try_generate_llm_plan(state)
            if plan is None:
                _append_log(state, "LLM Planner unavailable/no key, switching to Rule Planner")
                break
            plan = _normalize_preprocessing_plan(plan, state)
            ok, issues = _validate_preprocessing_plan(plan, state)
            if ok:
                _append_log(state, f"LLM Planner plan accepted (attempt={attempt + 1})")
                break
            _append_log(state, f"LLM Planner plan rejected: {issues}")
            plan = None

        if plan is None:
            plan = _default_preprocessing_plan(state)
            _append_log(state, "Rule Planner using default rule-based plan")

        plan = _normalize_preprocessing_plan(plan, state)
        ok, issues = _validate_preprocessing_plan(plan, state)
        state["plan_validation"] = {"valid": ok, "issues": issues}
        if not ok:
            _append_log(state, f"Rule Planner fallback due to validation issues: {issues}")
            plan = _default_preprocessing_plan(state)
            state["plan_validation"] = {"valid": True, "issues": ["fallback_to_default"]}

        state["preprocessing_plan"] = plan
        source = plan.get("plan_source", "unknown")
        _append_log(state, f"{'LLM Planner' if str(source).startswith('llm:') else 'Rule Planner'} complete: source={source}")
    except Exception as e:
        _append_error(state, f"agent_a_plan failed: {e}\n{traceback.format_exc()}")
        state["preprocessing_plan"] = _default_preprocessing_plan(state)
        state["plan_validation"] = {"valid": False, "issues": ["planner_exception_fallback_default"]}
    return state


def node_read_data(state: PipelineState) -> PipelineState:
    try:
        _append_log(state, f"Read data started: loading parquet from {state['data_path']}")
        df = pd.read_parquet(state["data_path"])
        state["raw_shape"] = [int(df.shape[0]), int(df.shape[1])]
        state["feature_columns"] = [c for c in df.columns if c.startswith("X")]
        state["time_column"] = _find_time_column(df)
        _append_log(
            state,
            f"Read parquet done: rows={df.shape[0]}, cols={df.shape[1]}, x_features={len(state['feature_columns'])}",
        )

        os.makedirs(state["output_dir"], exist_ok=True)
        _append_log(state, f"Output directory ready: {state['output_dir']}")

        _append_log(state, "Generating target distribution figure...")
        plt.figure(figsize=(8, 4))
        if state["target_column"] in df.columns:
            df[state["target_column"]].value_counts(dropna=False).sort_index().plot(kind="bar")
            plt.title(f"Target Distribution: {state['target_column']}")
            plt.tight_layout()
            plt.savefig(os.path.join(state["output_dir"], "target_distribution.png"), dpi=150)
        plt.close()
        _append_log(state, "Target distribution figure done")

        # Save a pandas-based data overview markdown for notebook/report references.
        _append_log(state, "Building pandas data overview markdown...")
        head_df = df.head(10)
        dtypes_df = df.dtypes.astype(str).rename("dtype").to_frame()
        missing_df = (
            df.isna()
            .sum()
            .rename("missing_count")
            .to_frame()
        )
        missing_df["missing_ratio"] = missing_df["missing_count"] / max(len(df), 1)
        missing_df = missing_df.sort_values("missing_ratio", ascending=False).head(30)
        numeric_desc = df.describe(include=[np.number]).T

        overview_lines = [
            "# 数据概况总览",
            "",
            "## 1) 数据基本信息",
            f"- 文件路径: {state['data_path']}",
            f"- 行数: {df.shape[0]}",
            f"- 列数: {df.shape[1]}",
            f"- 目标列: {state['target_column']}",
            f"- 时间列(自动识别): {state['time_column']}",
            "",
            "## 2) 前10行样本预览",
            "```text",
            head_df.to_string(max_cols=30),
            "```",
            "",
            "## 3) 字段类型统计",
            "```text",
            dtypes_df["dtype"].value_counts().to_string(),
            "```",
            "",
            "## 4) 字段类型明细（前100列）",
            "```text",
            dtypes_df.head(100).to_string(),
            "```",
            "",
            "## 5) 缺失值最多字段（Top 30）",
            "```text",
            missing_df.to_string(),
            "```",
        ]
        if not numeric_desc.empty:
            overview_lines.extend(
                [
                    "",
                    "## 6) 数值字段描述统计（前30列）",
                    "```text",
                    numeric_desc.head(30).to_string(),
                    "```",
                ]
            )
        with open(os.path.join(state["output_dir"], "data_overview.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(overview_lines))
        _append_log(state, "Data overview markdown done: data_overview.md")

        if state["feature_columns"]:
            _append_log(state, "Computing missing ratio over all X features...")
            miss_all = df[state["feature_columns"]].isna().mean()

            # All X features missing ratio (original feature order, not sorted).
            feature_cols = state["feature_columns"]
            y_vals = miss_all.reindex(feature_cols).to_numpy()
            x_idx = np.arange(len(feature_cols))
            plt.figure(figsize=(20, 7))
            plt.bar(x_idx, y_vals, color="#5c7cfa", width=0.9)
            step = max(1, len(feature_cols) // 30)
            tick_idx = x_idx[::step]
            tick_labels = [feature_cols[i] for i in tick_idx]
            plt.xticks(tick_idx, tick_labels, rotation=60, ha="right")
            plt.title("Missing Ratio for All X Features (Original Order)")
            plt.xlabel("Feature")
            plt.ylabel("Missing Ratio")
            plt.tight_layout()
            plt.savefig(os.path.join(state["output_dir"], "missing_all_features.png"), dpi=150)
            plt.close()
            _append_log(state, "Missing ratio all-features bar chart done")

            # Full missing-ratio distribution over all X features.
            _append_log(state, "Generating missing ratio histogram...")
            plt.figure(figsize=(8, 5))
            sns.histplot(miss_all, bins=20, kde=True, color="#457b9d")
            plt.title("Missing Ratio Distribution (All X Features)")
            plt.xlabel("Missing Ratio")
            plt.ylabel("Feature Count")
            plt.tight_layout()
            plt.savefig(os.path.join(state["output_dir"], "missing_ratio_hist.png"), dpi=150)
            plt.close()
            _append_log(state, "Missing ratio histogram done")

            _append_log(state, "Skipping missingness correlation heatmap by design")

        state["preprocessed"] = {"raw_df": df}
        _append_log(state, f"Read data complete: shape={df.shape}, target={state['target_column']}")
    except Exception as e:
        _append_error(state, f"read_data failed: {e}\n{traceback.format_exc()}")
    return state


def node_agent_a_preprocess(state: PipelineState) -> PipelineState:
    try:
        _append_log(state, "Agent A started")
        df = state["preprocessed"]["raw_df"]
        plan = state.get("preprocessing_plan", _default_preprocessing_plan(state))
        _append_log(state, f"Planner decision snapshot: {plan}")
        target_column = state["target_column"]
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not in dataframe")

        y = df[target_column].copy()
        X = df.drop(columns=[target_column]).copy()

        time_col = state.get("time_column")
        dt_series = _parse_time_series(df, time_col)
        if time_col and time_col in X.columns:
            X = X.drop(columns=[time_col])

        # Drop any datetime-like columns from model features; keep time only for split/leakage checks.
        datetime_cols = X.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
        if datetime_cols:
            X = X.drop(columns=datetime_cols)

        cat_cfg = plan.get("categorical", {})
        cat_encoding = str(cat_cfg.get("encoding", "onehot"))
        cat_drop_first = bool(cat_cfg.get("drop_first", True))
        cat_cols = X.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
        if cat_encoding == "onehot":
            X = pd.get_dummies(X, columns=cat_cols, drop_first=cat_drop_first)
        elif cat_encoding == "none":
            X = X.drop(columns=cat_cols, errors="ignore")
        else:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=cat_drop_first)
        X = X.apply(pd.to_numeric, errors="coerce")

        numeric_cols = X.columns.tolist()
        X = X.replace([np.inf, -np.inf], np.nan)

        split_info: Dict[str, Any]
        split_cfg = plan.get("split", {})
        split_method = split_cfg.get("method", "time_group_strict" if dt_series is not None else "random_split")
        split_test_size = float(split_cfg.get("test_size", state["test_size"]))
        _append_log(state, f"Cleaning decision - split: method={split_method}, test_size={split_test_size}")
        if split_method == "time_group_strict" and dt_series is not None:
            train_idx, test_idx, split_info = _time_group_strict_split(
                dt_series=dt_series,
                test_size=split_test_size,
            )
            _append_log(
                state,
                (
                    "Agent A strict time split: "
                    f"train={split_info['train_count']}, test={split_info['test_count']}, "
                    f"target_ratio={split_info['target_train_ratio']:.4f}, "
                    f"actual_ratio={split_info['actual_train_ratio']:.4f}, "
                    f"boundary={split_info['boundary_time']}"
                ),
            )
        else:
            train_idx, test_idx = train_test_split(
                np.arange(len(X)),
                test_size=split_test_size,
                random_state=state["random_state"],
                stratify=y if y.nunique() < 20 else None,
            )
            split_info = {
                "split_method": "random_split",
                "target_train_ratio": float(1 - split_test_size),
                "actual_train_ratio": float(len(train_idx) / len(X)),
                "boundary_time": None,
                "train_count": int(len(train_idx)),
                "test_count": int(len(test_idx)),
            }

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()
        _append_log(state, f"Agent A split complete: train={X_train.shape}, test={X_test.shape}")

        raw_impute_strategy = str(plan.get("impute", {}).get("strategy", "median")).strip().lower()
        valid_impute_strategies = {"mean", "median", "most_frequent", "constant"}
        if raw_impute_strategy not in valid_impute_strategies:
            _append_log(
                state,
                (
                    "Cleaning decision - impute strategy invalid for SimpleImputer: "
                    f"{raw_impute_strategy}. fallback=median"
                ),
            )
            impute_strategy = "median"
        else:
            impute_strategy = raw_impute_strategy
        _append_log(state, f"Cleaning decision - impute: strategy={impute_strategy}")
        imp = SimpleImputer(strategy=impute_strategy)
        scaler_std = StandardScaler()
        scaler_mm = MinMaxScaler()

        X_train_imp = imp.fit_transform(X_train)
        X_test_imp = imp.transform(X_test)

        scale_cfg = plan.get("scaling", {})
        use_standard = bool(scale_cfg.get("use_standard", True))
        use_minmax = bool(scale_cfg.get("use_minmax", True))
        _append_log(state, f"Cleaning decision - scaling: standard={use_standard}, minmax={use_minmax}")

        X_train_std = X_train_imp
        X_test_std = X_test_imp
        if use_standard:
            X_train_std = scaler_std.fit_transform(X_train_imp)
            X_test_std = scaler_std.transform(X_test_imp)

        X_train_mm = X_train_std
        X_test_mm = X_test_std
        if use_minmax:
            X_train_mm = scaler_mm.fit_transform(X_train_std)
            X_test_mm = scaler_mm.transform(X_test_std)

        # Remove zero-variance features first to avoid SelectKBest warnings and invalid scores.
        var_filter = VarianceThreshold(threshold=0.0)
        X_train_var = var_filter.fit_transform(X_train_mm)
        X_test_var = var_filter.transform(X_test_mm)
        var_support = var_filter.get_support(indices=True)
        numeric_cols_var = [numeric_cols[i] for i in var_support]
        _append_log(state, f"Agent A variance filter: {X_train_mm.shape[1]} -> {X_train_var.shape[1]}")

        fs_cfg = plan.get("feature_select", {})
        fs_enabled = bool(fs_cfg.get("enabled", True))
        fs_method = str(fs_cfg.get("method", "selectkbest")).lower()
        fs_k = int(fs_cfg.get("k", state["max_features"]))
        _append_log(state, f"Cleaning decision - feature_select: enabled={fs_enabled}, method={fs_method}, k={fs_k}")
        if fs_method == "variance_threshold":
            fs_enabled = False
        k = min(max(5, fs_k), X_train_var.shape[1])
        if X_train_var.shape[1] == 0:
            raise ValueError("All features were removed by variance filter.")
        if not fs_enabled or k >= X_train_var.shape[1]:
            X_train_sel = X_train_var
            X_test_sel = X_test_var
            selected_features = numeric_cols_var
            _append_log(state, "Agent A SelectKBest skipped by plan or k >= remaining features")
        else:
            selector = SelectKBest(score_func=f_classif, k=k)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                X_train_sel = selector.fit_transform(X_train_var, y_train)
                X_test_sel = selector.transform(X_test_var)
            selected_indices = selector.get_support(indices=True)
            selected_features = [numeric_cols_var[i] for i in selected_indices]
            _append_log(state, f"Agent A SelectKBest complete: k={k}")

        pca_cfg = plan.get("pca", {})
        pca_enabled = bool(pca_cfg.get("enabled", True))
        pca_comp_raw = pca_cfg.get("n_components", state["pca_components"])
        _append_log(state, f"Cleaning decision - pca: enabled={pca_enabled}, n_components={pca_comp_raw}")
        pca_n: Any
        try:
            pca_comp_f = float(pca_comp_raw)
            if 0 < pca_comp_f < 1:
                pca_n = pca_comp_f
            else:
                pca_n = min(max(2, int(pca_comp_f)), X_train_sel.shape[1])
        except Exception:
            pca_n = min(max(2, int(state["pca_components"])), X_train_sel.shape[1])
        if pca_enabled:
            pca = PCA(n_components=pca_n, random_state=state["random_state"])
            X_train_pca = pca.fit_transform(X_train_sel)
            X_test_pca = pca.transform(X_test_sel)
            explained_ratio = float(np.sum(pca.explained_variance_ratio_))
            _append_log(state, f"Agent A PCA complete: components={pca_n}")
        else:
            X_train_pca = X_train_sel
            X_test_pca = X_test_sel
            pca_n = int(X_train_sel.shape[1])
            explained_ratio = 1.0
            _append_log(state, "Agent A PCA skipped by plan")

        state["preprocessed"] = {
            "X_train": X_train_pca,
            "X_test": X_test_pca,
            "y_train": y_train.to_numpy(),
            "y_test": y_test.to_numpy(),
            "selected_features": selected_features,
            "time_train": dt_series.iloc[train_idx].astype(str).tolist() if dt_series is not None else None,
            "time_test": dt_series.iloc[test_idx].astype(str).tolist() if dt_series is not None else None,
            "summary": {
                "decision": "plan_driven_preprocessing",
                "plan_source": plan.get("plan_source", "unknown"),
                "plan": plan,
                "split_info": split_info,
                "select_k": int(k),
                "pca_components": int(pca_n),
                "explained_variance_ratio": explained_ratio,
                "selected_feature_count": len(selected_features),
            },
        }
        _append_log(
            state,
            f"Agent A complete: train={X_train_pca.shape}, test={X_test_pca.shape}, selected={len(selected_features)}",
        )
    except Exception as e:
        _append_error(state, f"agent_a_preprocess failed: {e}\n{traceback.format_exc()}")
    return state


class _MLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def _train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    num_classes: int,
    seed: int,
    epochs: int = 20,
    log_fn: Optional[Callable[[str], None]] = None,
):
    if torch is None:
        raise RuntimeError("torch is not installed")

    torch.manual_seed(seed)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=512, shuffle=True)
    model = _MLP(X_train.shape[1], num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batch_count += 1
        if log_fn is not None:
            avg_loss = total_loss / max(batch_count, 1)
            log_fn(f"Agent B MLP epoch {epoch + 1}/{epochs}, loss={avg_loss:.6f}")

    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        proba = torch.softmax(logits, dim=1).numpy()
        pred = np.argmax(proba, axis=1)
    return pred, proba


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, classes: np.ndarray) -> Dict[str, float]:
    avg = "binary" if len(classes) == 2 else "macro"
    if len(classes) == 2:
        auc_val = roc_auc_score(y_true, y_proba[:, 1]) if y_proba.ndim == 2 and y_proba.shape[1] > 1 else np.nan
    else:
        y_bin = label_binarize(y_true, classes=classes)
        auc_val = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")

    return {
        "auc": float(auc_val),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _decision_to_proba(decision: np.ndarray) -> np.ndarray:
    # Convert margin scores to probability-like outputs for AUC/ROC compatibility.
    dec = np.asarray(decision)
    if dec.ndim == 1:
        p1 = 1.0 / (1.0 + np.exp(-dec))
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T
    z = dec - np.max(dec, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.clip(np.sum(ez, axis=1, keepdims=True), 1e-12, None)


def _train_svm_with_progress(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    random_state: int,
    max_iter: int,
    log_fn: Callable[[str], None],
) -> tuple[np.ndarray, np.ndarray]:
    # Use SGD hinge loss as an iterative linear-SVM style optimizer so we can print progress.
    classes = np.unique(y_train)
    clf = SGDClassifier(
        loss="hinge",
        alpha=1e-4,
        learning_rate="optimal",
        random_state=random_state,
        fit_intercept=True,
    )
    for i in range(1, max_iter + 1):
        if i == 1:
            clf.partial_fit(X_train, y_train, classes=classes)
        else:
            clf.partial_fit(X_train, y_train)
        if i % 100 == 0 or i == 1 or i == max_iter:
            log_fn(f"Agent B SVM iter {i}/{max_iter}")

    y_pred = clf.predict(X_test)
    y_score = clf.decision_function(X_test)
    y_proba = _decision_to_proba(y_score)
    return y_pred, y_proba


def _detect_compute_device() -> Dict[str, Any]:
    has_torch_cuda = bool(torch is not None and torch.cuda.is_available())
    return {
        "prefer_gpu": has_torch_cuda,
        "device": "cuda" if has_torch_cuda else "cpu",
        "torch_cuda_available": has_torch_cuda,
        "cuml_available": cuSVC is not None,
        "xgboost_available": xgb is not None,
    }


def node_agent_b_train(state: PipelineState) -> PipelineState:
    try:
        _append_log(state, "Agent B started")
        d = state["preprocessed"]
        required_keys = ("X_train", "X_test", "y_train", "y_test")
        missing = [k for k in required_keys if k not in d]
        if missing:
            _append_error(
                state,
                f"agent_b_train skipped: missing preprocessed keys {missing}. Upstream step likely failed.",
            )
            return state
        X_train = np.asarray(d["X_train"], dtype=float)
        X_test = np.asarray(d["X_test"], dtype=float)
        y_train = np.asarray(d["y_train"])
        y_test = np.asarray(d["y_test"])
        classes = np.unique(np.concatenate([y_train, y_test]))
        device_info = _detect_compute_device()
        _append_log(
            state,
            (
                "Agent B device check: "
                f"prefer={device_info['device']}, "
                f"torch_cuda={device_info['torch_cuda_available']}, "
                f"cuml={device_info['cuml_available']}, "
                f"xgboost={device_info['xgboost_available']}"
            ),
        )
        _append_log(state, f"Agent B data ready: X_train={X_train.shape}, X_test={X_test.shape}, classes={len(classes)}")

        sample_max = int(state.get("train_sample_max", 120000))
        if len(X_train) > sample_max:
            rng = np.random.default_rng(state["random_state"])
            idx = rng.choice(len(X_train), size=sample_max, replace=False)
            X_train_fit = X_train[idx]
            y_train_fit = y_train[idx]
            _append_log(state, f"Agent B train sampling applied: {len(X_train)} -> {len(X_train_fit)}")
        else:
            X_train_fit = X_train
            y_train_fit = y_train

        results: Dict[str, Any] = {}

        svm_max_iter = int(state.get("svm_max_iter", 2000))
        _append_log(state, f"Agent B training SVM... (iterative, max_iter={svm_max_iter}, log_every=100)")
        if device_info["prefer_gpu"] and cuSVC is not None:
            _append_log(
                state,
                "Agent B SVM note: for per-100-iter logging, using iterative CPU solver instead of cuML GPU SVC.",
            )
        svm_backend = "sgd_hinge_cpu"
        svm_pred, svm_proba = _train_svm_with_progress(
            X_train_fit,
            y_train_fit,
            X_test,
            random_state=state["random_state"],
            max_iter=svm_max_iter,
            log_fn=lambda msg: _append_log(state, msg),
        )
        results["svm"] = {
            "metrics": _compute_metrics(y_test, svm_pred, svm_proba, classes),
            "predictions": svm_pred.tolist(),
            "probabilities": svm_proba.tolist(),
            "backend": svm_backend,
        }
        _append_log(state, "Agent B SVM done")

        if xgb is not None:
            _append_log(state, "Agent B training XGBoost...")
            objective = "binary:logistic" if len(classes) == 2 else "multi:softprob"
            eval_metric = "logloss" if len(classes) == 2 else "mlogloss"
            label_to_id = {c: i for i, c in enumerate(classes)}
            id_to_label = {i: c for c, i in label_to_id.items()}
            y_train_fit_xgb = np.array([label_to_id[v] for v in y_train_fit], dtype=int)
            base_kwargs = dict(
                n_estimators=int(state.get("xgb_n_estimators", 200)),
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=state["random_state"],
                objective=objective,
                eval_metric=eval_metric,
                n_jobs=-1,
                verbosity=0,
            )
            if len(classes) > 2:
                base_kwargs["num_class"] = int(len(classes))
            xgb_backend = "cpu"
            if device_info["prefer_gpu"]:
                try:
                    _append_log(state, "Agent B XGBoost backend: GPU(cuda)")
                    xgb_model = xgb.XGBClassifier(**base_kwargs, device="cuda", tree_method="hist")
                    t0 = time.perf_counter()
                    xgb_model.fit(
                        X_train_fit,
                        y_train_fit_xgb,
                        eval_set=[(X_train_fit, y_train_fit_xgb)],
                        verbose=max(1, int(state.get("xgb_log_every", 20))),
                    )
                    elapsed = time.perf_counter() - t0
                    xgb_backend = "cuda"
                except Exception as e:
                    _append_log(state, f"Agent B XGBoost GPU failed, fallback CPU: {e}")
                    xgb_model = xgb.XGBClassifier(**base_kwargs)
                    t0 = time.perf_counter()
                    xgb_model.fit(
                        X_train_fit,
                        y_train_fit_xgb,
                        eval_set=[(X_train_fit, y_train_fit_xgb)],
                        verbose=max(1, int(state.get("xgb_log_every", 20))),
                    )
                    elapsed = time.perf_counter() - t0
            else:
                _append_log(state, "Agent B XGBoost backend: CPU")
                xgb_model = xgb.XGBClassifier(**base_kwargs)
                t0 = time.perf_counter()
                xgb_model.fit(
                    X_train_fit,
                    y_train_fit_xgb,
                    eval_set=[(X_train_fit, y_train_fit_xgb)],
                    verbose=max(1, int(state.get("xgb_log_every", 20))),
                )
                elapsed = time.perf_counter() - t0
            xgb_pred_idx = xgb_model.predict(X_test).astype(int)
            xgb_proba = xgb_model.predict_proba(X_test)
            xgb_pred = np.array([id_to_label[i] for i in xgb_pred_idx])
            rounds = int(xgb_model.get_booster().num_boosted_rounds())
            _append_log(state, f"Agent B XGBoost trained rounds={rounds}, elapsed={elapsed:.2f}s")
            results["xgboost"] = {
                "metrics": _compute_metrics(y_test, xgb_pred, xgb_proba, classes),
                "predictions": xgb_pred.tolist(),
                "probabilities": xgb_proba.tolist(),
                "backend": xgb_backend,
            }
            _append_log(state, "Agent B XGBoost done")
        else:
            if XGB_IMPORT_ERROR:
                _append_log(state, f"Agent B XGBoost unavailable, fallback to tree model. import_error={XGB_IMPORT_ERROR.splitlines()[-1]}")
            _append_log(state, "Agent B training tree fallback...")
            rf = RandomForestClassifier(n_estimators=300, random_state=state["random_state"])
            rf.fit(X_train_fit, y_train_fit)
            rf_pred = rf.predict(X_test)
            rf_proba = rf.predict_proba(X_test)
            results["tree_fallback"] = {
                "metrics": _compute_metrics(y_test, rf_pred, rf_proba, classes),
                "predictions": rf_pred.tolist(),
                "probabilities": rf_proba.tolist(),
                "note": "xgboost missing, used random forest fallback",
            }
            _append_log(state, "Agent B tree fallback done")

        if torch is not None and len(classes) >= 2:
            _append_log(state, "Agent B training MLP...")
            label_to_id = {c: i for i, c in enumerate(classes)}
            id_to_label = {i: c for c, i in label_to_id.items()}
            y_train_idx = np.array([label_to_id[v] for v in y_train_fit])
            y_test_idx = np.array([label_to_id[v] for v in y_test])

            mlp_pred_idx, mlp_proba = _train_mlp(
                X_train_fit,
                y_train_idx,
                X_test,
                num_classes=len(classes),
                seed=state["random_state"],
                epochs=int(state.get("mlp_epochs", 20)),
                log_fn=lambda msg: _append_log(state, msg),
            )
            mlp_pred = np.array([id_to_label[i] for i in mlp_pred_idx])
            results["mlp"] = {
                "metrics": _compute_metrics(y_test_idx, mlp_pred_idx, mlp_proba, np.arange(len(classes))),
                "predictions": mlp_pred.tolist(),
                "probabilities": mlp_proba.tolist(),
            }
            _append_log(state, "Agent B MLP done")
        else:
            _append_log(state, "Torch unavailable, skip MLP model")

        best_model = max(results.items(), key=lambda kv: kv[1]["metrics"].get("f1", -1))[0]

        state["training_results"] = {
            "models": results,
            "best_model": best_model,
            "classes": classes.tolist(),
            "y_test": y_test.tolist(),
            "time_train": d.get("time_train"),
            "time_test": d.get("time_test"),
            "time_column": state.get("time_column"),
        }
        _append_log(state, f"Agent B complete: models={list(results.keys())}, best={best_model}")
    except Exception as e:
        _append_error(state, f"agent_b_train failed: {e}\n{traceback.format_exc()}")
    return state


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray, save_path: str) -> None:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_roc(y_true: np.ndarray, y_prob: np.ndarray, labels: np.ndarray, save_path: str) -> None:
    from sklearn.metrics import auc, roc_curve

    plt.figure(figsize=(7, 6))
    if len(labels) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1], pos_label=labels[1])
        plt.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.4f}")
    else:
        y_bin = label_binarize(y_true, classes=labels)
        for i, c in enumerate(labels):
            if i >= y_prob.shape[1]:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            plt.plot(fpr, tpr, label=f"class {c}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _check_temporal_leakage(time_train: Optional[List[str]], time_test: Optional[List[str]]) -> Dict[str, Any]:
    if not time_train or not time_test:
        return {
            "has_time_data": False,
            "leakage_detected": None,
            "message": "No time data found. Temporal leakage check skipped.",
        }

    tr = pd.to_datetime(pd.Series(time_train), errors="coerce").dropna()
    te = pd.to_datetime(pd.Series(time_test), errors="coerce").dropna()
    if tr.empty or te.empty:
        return {
            "has_time_data": False,
            "leakage_detected": None,
            "message": "Time parsing failed. Temporal leakage check skipped.",
        }

    max_train = tr.max()
    min_test = te.min()
    leakage = bool(max_train >= min_test)
    overlap_count = int((tr >= min_test).sum()) if leakage else 0

    return {
        "has_time_data": True,
        "leakage_detected": leakage,
        "max_train_time": str(max_train),
        "min_test_time": str(min_test),
        "overlap_count": overlap_count,
        "message": "Leakage detected" if leakage else "No leakage detected",
    }


def node_agent_c_evaluate(state: PipelineState) -> PipelineState:
    try:
        _append_log(state, "Agent C started")
        if "training_results" not in state:
            _append_error(
                state,
                "agent_c_evaluate skipped: training_results missing. Upstream training step likely failed.",
            )
            return state

        tr = state["training_results"]
        y_test = np.asarray(tr["y_test"])
        labels = np.asarray(tr["classes"])
        model_results = tr["models"]

        rows = []
        vis_files: List[str] = []

        for name, info in model_results.items():
            _append_log(state, f"Agent C plotting {name}...")
            y_pred = np.asarray(info["predictions"])
            y_prob = np.asarray(info["probabilities"])
            metrics = info["metrics"]

            cm_path = os.path.join(state["output_dir"], f"{name}_confusion_matrix.png")
            roc_path = os.path.join(state["output_dir"], f"{name}_roc_curve.png")
            _plot_confusion(y_test, y_pred, labels, cm_path)
            _plot_roc(y_test, y_prob, labels, roc_path)
            vis_files.extend([cm_path, roc_path])

            rows.append(
                {
                    "model": name,
                    "auc": metrics["auc"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "accuracy": metrics["accuracy"],
                }
            )

        compare_df = pd.DataFrame(rows).sort_values("f1", ascending=False)
        compare_path = os.path.join(state["output_dir"], "model_comparison.png")
        metric_order = ["auc", "precision", "recall", "f1"]
        compare_long = compare_df.melt(
            id_vars=["model"],
            value_vars=metric_order,
            var_name="metric",
            value_name="score",
        )
        plt.figure(figsize=(12, 5))
        sns.barplot(
            data=compare_long,
            x="model",
            y="score",
            hue="metric",
            hue_order=metric_order,
        )
        plt.ylim(0, 1)
        plt.title("Model Comparison (AUC/Precision/Recall/F1)")
        plt.tight_layout()
        plt.savefig(compare_path, dpi=150)
        plt.close()
        vis_files.append(compare_path)

        leakage = _check_temporal_leakage(tr.get("time_train"), tr.get("time_test"))
        json_path = os.path.join(state["output_dir"], "langgraph_pipeline_result.json")
        report_path = os.path.join(state["output_dir"], "langgraph_pipeline_report.md")

        model_rows = [
            {
                "model": r["model"],
                "auc": r["auc"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
                "accuracy": r["accuracy"],
            }
            for r in rows
        ]

        payload = {
            "summary": {
                "data_shape": state.get("raw_shape"),
                "target_column": state.get("target_column"),
                "best_model": tr.get("best_model"),
                "errors": state.get("errors", []),
            },
            "llm_planner": {
                "enabled": bool(state.get("use_llm_planner", True)),
                "plan": state.get("preprocessing_plan", {}),
                "validation": state.get("plan_validation", {}),
            },
            "preprocessing_summary": state.get("preprocessed", {}).get("summary", {}),
            "model_metrics": model_rows,
            "model_evaluation_and_temporal_leakage": {
                "model_evaluation_files_directory": state.get("output_dir"),
                "has_time_data": leakage.get("has_time_data"),
                "leakage_detected": leakage.get("leakage_detected"),
                "max_train_time": leakage.get("max_train_time"),
                "min_test_time": leakage.get("min_test_time"),
                "overlap_count": leakage.get("overlap_count"),
                "message": leakage.get("message"),
            },
            "visualization_files": vis_files,
            "run_logs": state.get("logs", []),
            # Compatibility alias (English-only)
            "temporal_leakage_check": leakage,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        report_lines = [
            "# Multi-Agent LangGraph Pipeline Report",
            "",
            "## (1) Data and Target",
            f"- Data Path: {state.get('data_path')}",
            f"- Data Shape: {state.get('raw_shape')}",
            f"- Target Column: {state.get('target_column')}",
            "",
            "## (2) LLM Planner",
            f"- LLM Planner Enabled: {bool(state.get('use_llm_planner', True))}",
            f"- Plan Validation: {state.get('plan_validation', {})}",
            f"- Plan Content: {state.get('preprocessing_plan', {})}",
            "",
            "## (3) Preprocessing Decisions (Agent A)",
            f"- {state.get('preprocessed', {}).get('summary', {})}",
            "",
            "## (4) Model Metrics (Agent B)",
        ]
        for r in rows:
            report_lines.append(
                f"- {r['model']}: AUC={r['auc']:.4f}, Precision={r['precision']:.4f}, Recall={r['recall']:.4f}, F1={r['f1']:.4f}, Accuracy={r['accuracy']:.4f}"
            )

        report_lines.extend(
            [
                "",
                "## (5) Model Evaluation and Data Leakage Check (Agent C)",
                f"- Model Evaluation Files Directory: {state.get('output_dir')}",
                f"- {leakage}",
                "",
                "## (6) Output Files",
            ]
        )
        report_lines.extend([f"- {x}" for x in vis_files])
        report_lines.extend(["", "## (7) Run Logs"])
        report_lines.extend([f"- {x}" for x in state.get("logs", [])])



        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        state["evaluation_results"] = payload
        state["final_json_path"] = json_path
        state["final_report_path"] = report_path
        _append_log(state, f"Agent C complete: report={report_path}")
    except Exception as e:
        _append_error(state, f"agent_c_evaluate failed: {e}\n{traceback.format_exc()}")
    return state


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("read_data", node_read_data)
    graph.add_node("agent_a_plan", node_agent_a_plan)
    graph.add_node("agent_a_preprocess", node_agent_a_preprocess)
    graph.add_node("agent_b_train", node_agent_b_train)
    graph.add_node("agent_c_evaluate", node_agent_c_evaluate)

    graph.set_entry_point("read_data")
    graph.add_edge("read_data", "agent_a_plan")
    graph.add_edge("agent_a_plan", "agent_a_preprocess")
    graph.add_edge("agent_a_preprocess", "agent_b_train")
    graph.add_edge("agent_b_train", "agent_c_evaluate")
    graph.add_edge("agent_c_evaluate", END)
    return graph.compile()


def run_pipeline(config: Config) -> PipelineState:
    graph = build_graph()
    init_state: PipelineState = {
        "data_path": config.data_path,
        "target_column": config.target_column,
        "output_dir": config.output_dir,
        "test_size": config.test_size,
        "random_state": config.random_state,
        "max_features": config.max_features,
        "pca_components": config.pca_components,
        "train_sample_max": config.train_sample_max,
        "svm_max_iter": config.svm_max_iter,
        "mlp_epochs": config.mlp_epochs,
        "xgb_n_estimators": config.xgb_n_estimators,
        "xgb_log_every": config.xgb_log_every,
        "planner_model": config.planner_model,
        "planner_temperature": config.planner_temperature,
        "planner_max_retries": config.planner_max_retries,
        "use_llm_planner": config.use_llm_planner,
        "logs": [],
        "errors": [],
    }
    return graph.invoke(init_state)


def _resolve_llm_planner_mode(planner_model: str = "deepseek-chat") -> tuple[bool, str]:
    env_name = "DEEPSEEK_API_KEY"
    has_key = bool(os.getenv(env_name))
    print(
        "This agent system can use the DeepSeek LLM for data-cleaning planning. "
        "Set DEEPSEEK_API_KEY before running (for notebook, set it in the install/setup cell). "
        "If key is missing or connection fails, the pipeline will auto-fallback to Rule Planner.",
        flush=True,
    )
    print(f"Startup check: scanning {env_name} for LLM planner.", flush=True)
    print(f"Detected {env_name}: {'yes' if has_key else 'no'}", flush=True)

    use_llm_planner, reason = _preflight_llm_planner(True, planner_model=planner_model)
    if use_llm_planner:
        print("Planner mode: LLM planner enabled (DeepSeek, preflight passed)", flush=True)
    else:
        print(f"Planner mode: fallback rule planner only (reason={reason})", flush=True)
    return use_llm_planner, reason


def _preflight_llm_planner(use_llm_planner: bool, planner_model: str = "deepseek-chat") -> tuple[bool, str]:
    if not use_llm_planner:
        return False, "user_selected_fallback"

    env_name, host = "DEEPSEEK_API_KEY", "api.deepseek.com"
    if not os.getenv(env_name):
        return False, f"{env_name} not found"

    try:
        from openai import OpenAI
    except Exception as e:
        return False, f"openai package unavailable: {e}"

    try:
        # Fast network reachability precheck to avoid long stack traces during planning call.
        with socket.create_connection((host, 443), timeout=5):
            pass
    except Exception as e:
        return False, f"network to {host}:443 unreachable: {e}"

    # Fast auth probe: if key is invalid, fallback before entering Agent A planning.
    try:
        client = OpenAI(api_key=os.getenv(env_name), base_url="https://api.deepseek.com")
        client.chat.completions.create(
            model=planner_model or "deepseek-chat",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            timeout=6,
        )
    except Exception as e:
        return False, f"auth probe failed: {e}"

    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph Multi-Agent ML Pipeline")
    parser.add_argument("--data_path", type=str, default="./data.pq")
    parser.add_argument("--target_column", type=str, default="Y1")
    parser.add_argument("--output_dir", type=str, default="ml_pipeline_outputs")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--max_features", type=int, default=50)
    parser.add_argument("--pca_components", type=int, default=20)
    parser.add_argument("--train_sample_max", type=int, default=120000)
    parser.add_argument("--svm_max_iter", type=int, default=2000)
    parser.add_argument("--mlp_epochs", type=int, default=20)
    parser.add_argument("--xgb_n_estimators", type=int, default=200)
    parser.add_argument("--xgb_log_every", type=int, default=20)
    parser.add_argument("--planner_model", type=str, default="deepseek-chat")
    parser.add_argument("--planner_temperature", type=float, default=0.0)
    parser.add_argument("--planner_max_retries", type=int, default=1)
    args = parser.parse_args()

    use_llm_planner, _ = _resolve_llm_planner_mode(planner_model=args.planner_model)

    cfg = Config(
        data_path=args.data_path,
        target_column=args.target_column,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        max_features=args.max_features,
        pca_components=args.pca_components,
        train_sample_max=args.train_sample_max,
        svm_max_iter=args.svm_max_iter,
        mlp_epochs=args.mlp_epochs,
        xgb_n_estimators=args.xgb_n_estimators,
        xgb_log_every=args.xgb_log_every,
        planner_model=args.planner_model,
        planner_temperature=args.planner_temperature,
        planner_max_retries=args.planner_max_retries,
        use_llm_planner=use_llm_planner,
    )

    state = run_pipeline(cfg)

    print("=== LangGraph Pipeline Finished ===")
    print(f"Errors: {len(state.get('errors', []))}")
    if state.get("final_json_path"):
        print(f"Result JSON: {state['final_json_path']}")
    if state.get("final_report_path"):
        print(f"Report MD : {state['final_report_path']}")

    if state.get("errors"):
        print("\\n".join(state["errors"]))


if __name__ == "__main__":
    main()
