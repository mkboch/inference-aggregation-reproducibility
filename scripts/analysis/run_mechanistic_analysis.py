#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path
import hashlib
import json
import math
import os
from datetime import datetime, timezone
import re
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("AGGREGATION_REPRO_ROOT", Path(__file__).resolve().parents[2]))
OUTCOME = ROOT / "results/final_outcome_analysis_20260830"
RAW = ROOT / "results/final_full_benchmark_20260817"
OUT = ROOT / "results/final_mechanistic_analysis_20260830"
PARSER = ROOT / "src/evaluation/parser.py"
SEED = 20260816
DATASETS = ["medqa", "medmcqa", "pubmedqa"]
METHODS = ["direct", "cot", "structured"]
ALL_METHODS = METHODS + ["fusion"]
PRIMARY_FALLBACK = ("direct", "cot", "structured")
VALID_ANSWERS = {"A", "B", "C", "D", "E"}
VALID_SPACES = {"VALID", "INCOMPLETE", "CONTRADICTED"}
MODEL_ORDER = ["qwen36", "gemma4_31b", "medgemma27b", "qwen25", "phi4mini"]


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(x: float) -> str:
    if math.isnan(x):
        return "NA"
    return f"{100*x:.2f}%"


def exact_json_contract(raw: str) -> tuple[bool, Any, str]:
    try:
        obj = json.loads((raw or "").strip())
    except Exception as exc:
        return False, None, type(exc).__name__
    ok = isinstance(obj, dict) and set(obj.keys()) == {"space_label", "answer", "confidence"}
    return ok, obj, ""


def simple_json_objects(raw: str) -> list[dict[str, Any]]:
    out = []
    for candidate in re.findall(r"\{[^{}]*\}", raw or "", flags=re.DOTALL):
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def first_line_answer(raw: str) -> str:
    lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    m = re.match(r"^([A-E])\s*[\.\):,-]?$", lines[0], flags=re.I)
    return m.group(1).upper() if m else ""


def regex_answer_source(raw: str) -> tuple[str, str]:
    patterns = [
        ("answer_key_regex", r'"answer"\s*:\s*"([A-E])"'),
        ("FINAL_ANSWER_regex", r"FINAL_ANSWER\s*[:=]\s*([A-E])\b"),
        ("final_answer_phrase", r"Final\s+answer\s*[:=]?\s*([A-E])\b"),
        ("answer_colon_phrase", r"\bAnswer\s*[:=]\s*([A-E])\b"),
        ("the_answer_is_phrase", r"\bthe\s+answer\s+is\s+([A-E])\b"),
        ("option_phrase", r"\bOption\s+([A-E])\b"),
    ]
    for name, pat in patterns:
        hit = re.search(pat, raw or "", flags=re.I)
        if hit:
            return name, hit.group(1).upper()
    fl = first_line_answer(raw)
    if fl:
        return "first_nonempty_line_answer", fl
    return "", ""


def clean_excerpt(raw: str, n: int = 1600) -> str:
    text = (raw or "").replace("\r\n", "\n")
    if len(text) > n:
        return text[:n] + "\n...[truncated excerpt]"
    return text


def classify_output_form(raw: str, recovered: bool) -> str:
    text = raw or ""
    stripped = text.strip()
    objs = simple_json_objects(text)
    if exact_json_contract(text)[0]:
        return "exact_top_level_json"
    if "```" in stripped and objs:
        return "markdown_fenced_or_wrapped_json"
    if stripped.lower().startswith("<unused94>thought"):
        return "medgemma_native_reasoning_trace_with_parser_answer" if recovered else "medgemma_native_reasoning_trace_without_parser_answer"
    if objs:
        before = text[: text.find("{")].strip()
        after = text[text.rfind("}") + 1 :].strip()
        if before and after:
            return "prose_before_and_after_embedded_json"
        if before:
            return "prose_before_embedded_json"
        if after:
            return "prose_after_embedded_json"
        return "embedded_json_not_exact"
    if "{" in text or "}" in text:
        return "malformed_or_nested_json_no_simple_object"
    if first_line_answer(text):
        return "bare_option_first_line"
    if regex_answer_source(text)[0]:
        return "prose_answer_pattern_no_json"
    return "free_text_no_parser_answer" if not recovered else "free_text_recovered_by_parser_pattern"


def classify_structured_row(row: pd.Series, dataset_spaces: dict[str, set[str]]) -> dict[str, Any]:
    raw = row["raw_output"] or ""
    exact = bool(row["structured_exact_json"])
    recovered = bool(row["parser_recovered"])
    full_valid, full_obj, full_error = exact_json_contract(raw)
    objs = simple_json_objects(raw)
    first_obj = objs[0] if objs else None
    obj = full_obj if isinstance(full_obj, dict) else first_obj
    keys = sorted(obj.keys()) if isinstance(obj, dict) else []
    answer_value = obj.get("answer") if isinstance(obj, dict) else None
    space_value = obj.get("space_label") if isinstance(obj, dict) else None
    conf_value = obj.get("confidence") if isinstance(obj, dict) else None
    answer_text = "" if answer_value is None else str(answer_value).strip()
    answer_upper = answer_text.upper()
    answer_is_valid_global = isinstance(answer_value, str) and bool(re.fullmatch(r"[A-E]", answer_upper))
    answer_in_dataset_space = answer_upper in dataset_spaces[row["dataset"]]
    conf_numeric = isinstance(conf_value, (int, float)) and not isinstance(conf_value, bool)
    conf_in_range = conf_numeric and 0.0 <= float(conf_value) <= 1.0
    regex_source, regex_answer = regex_answer_source(raw)
    if recovered:
        if isinstance(first_obj, dict) and str(first_obj.get("answer", "")).strip().upper() in VALID_ANSWERS and len(str(first_obj.get("answer", "")).strip()) == 1:
            recovery_source = "json_object_answer"
        elif regex_source:
            recovery_source = regex_source
        else:
            recovery_source = "parser_other"
        recovery_failure_reason = ""
    else:
        recovery_source = ""
        if exact:
            if "answer" not in keys:
                recovery_failure_reason = "exact_json_missing_answer_key"
            elif answer_value is None:
                recovery_failure_reason = "exact_json_answer_null"
            elif isinstance(answer_value, str) and answer_value.strip() == "":
                recovery_failure_reason = "exact_json_answer_empty_string"
            elif not isinstance(answer_value, str):
                recovery_failure_reason = "exact_json_answer_non_string"
            elif not answer_is_valid_global:
                recovery_failure_reason = "exact_json_answer_outside_A_to_E"
            else:
                recovery_failure_reason = "exact_json_parser_rejection_other"
        elif first_obj is not None:
            if "answer" not in keys:
                recovery_failure_reason = "embedded_json_missing_answer_key"
            elif answer_value is None:
                recovery_failure_reason = "embedded_json_answer_null"
            elif isinstance(answer_value, str) and answer_value.strip() == "":
                recovery_failure_reason = "embedded_json_answer_empty_string"
            elif not isinstance(answer_value, str):
                recovery_failure_reason = "embedded_json_answer_non_string"
            elif not answer_is_valid_global:
                recovery_failure_reason = "embedded_json_answer_outside_A_to_E"
            else:
                recovery_failure_reason = "embedded_json_parser_rejection_other"
        elif bool(row["length_truncation"]):
            recovery_failure_reason = "no_parser_answer_length_truncated"
        elif bool(row["reasoning_marker"]):
            recovery_failure_reason = "no_parser_answer_reasoning_trace"
        elif "{" in raw or "}" in raw:
            recovery_failure_reason = "malformed_json_no_parser_answer"
        else:
            recovery_failure_reason = "no_parser_accepted_answer"
    if exact and recovered:
        contract_recovery_class = "exact_contract_and_recovered"
    elif exact and not recovered:
        contract_recovery_class = "exact_contract_but_unrecovered"
    elif (not exact) and recovered:
        contract_recovery_class = "nonexact_but_recovered"
    else:
        contract_recovery_class = "nonexact_and_unrecovered"
    if answer_value is None:
        answer_value_class = "null_or_absent"
    elif isinstance(answer_value, str) and answer_value.strip() == "":
        answer_value_class = "empty_string"
    elif answer_is_valid_global:
        answer_value_class = "valid_A_to_E"
    elif isinstance(answer_value, str) and answer_upper in {"YES", "NO", "MAYBE"}:
        answer_value_class = "yes_no_maybe_text"
    elif isinstance(answer_value, str):
        answer_value_class = "nonempty_nonoption_string"
    else:
        answer_value_class = type(answer_value).__name__
    return {
        "model_key": row["model_key"],
        "dataset": row["dataset"],
        "source_index": int(row["source_index"]),
        "uid": row["uid"],
        "gold": row["gold"],
        "raw_output_path": row["raw_output_path"],
        "raw_output_line": row["raw_output_line"],
        "exact_json": exact,
        "parser_recovered": recovered,
        "correct": bool(row["correct"]),
        "parsed_answer": row["parsed_answer"],
        "finish_reason": row["finish_reason"],
        "output_tokens": row["output_tokens"],
        "length_truncation": bool(row["length_truncation"]),
        "severe_repetition": bool(row["severe_repetition"]),
        "reasoning_marker": bool(row["reasoning_marker"]),
        "contract_recovery_class": contract_recovery_class,
        "output_form": classify_output_form(raw, recovered),
        "recovery_source": recovery_source,
        "recovery_failure_reason": recovery_failure_reason,
        "full_json_parse_valid": isinstance(full_obj, dict),
        "full_json_contract_valid": full_valid,
        "full_json_error": full_error,
        "simple_json_object_count": len(objs),
        "expected_keys_present": set(keys) >= {"space_label", "answer", "confidence"},
        "missing_required_keys": "|".join(k for k in ["space_label", "answer", "confidence"] if k not in keys),
        "extra_keys": "|".join(k for k in keys if k not in {"space_label", "answer", "confidence"}),
        "json_keys": "|".join(keys),
        "answer_value_repr": repr(answer_value),
        "answer_value_class": answer_value_class,
        "answer_is_valid_A_to_E": answer_is_valid_global,
        "answer_in_observed_dataset_space": answer_in_dataset_space,
        "observed_dataset_answer_space": "".join(sorted(dataset_spaces[row["dataset"]])),
        "space_label_repr": repr(space_value),
        "space_label_valid": isinstance(space_value, str) and space_value.strip().upper() in VALID_SPACES,
        "confidence_repr": repr(conf_value),
        "confidence_numeric": conf_numeric,
        "confidence_in_range": conf_in_range,
        "regex_recovery_candidate_source": regex_source,
        "regex_recovery_candidate_answer": regex_answer,
        "raw_output_excerpt": clean_excerpt(raw),
    }


def summarize_counts(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        rows.append({
            **dict(zip(group_cols, key)),
            "N": len(g),
            "rate_within_group": math.nan,
            "exact_json_n": int(g["exact_json"].sum()),
            "parser_recovered_n": int(g["parser_recovered"].sum()),
            "correct_n": int(g["correct"].sum()),
            "truncation_n": int(g["length_truncation"].sum()),
            "reasoning_marker_n": int(g["reasoning_marker"].sum()),
        })
    out = pd.DataFrame(rows)
    parent_cols = [c for c in group_cols if c not in {"contract_recovery_class", "recovery_failure_reason", "output_form", "answer_value_class"}]
    if parent_cols and len(out):
        den = out.groupby(parent_cols)["N"].transform("sum")
        out["rate_within_group"] = out["N"] / den
    return out


def structured_analysis(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    s = df[df["method"] == "structured"].copy()
    dataset_spaces = {d: set(df[df["dataset"] == d]["gold"].dropna().astype(str)) for d in DATASETS}
    item = pd.DataFrame([classify_structured_row(r, dataset_spaces) for _, r in s.iterrows()])
    taxonomy = summarize_counts(item, ["model_key", "contract_recovery_class", "recovery_failure_reason", "output_form", "answer_value_class"])
    by_dataset = summarize_counts(item, ["model_key", "dataset", "contract_recovery_class", "recovery_failure_reason", "output_form", "answer_value_class"])
    examples = (
        item.sort_values(["model_key", "contract_recovery_class", "recovery_failure_reason", "dataset", "source_index", "uid"])
        .groupby(["model_key", "contract_recovery_class", "recovery_failure_reason"], dropna=False)
        .head(5)
        .copy()
    )
    qmask = (item["model_key"] == "qwen25") & item["exact_json"] & (~item["parser_recovered"])
    q = item[qmask].copy()
    q_parts = []
    for name, col in [
        ("rejection_reason", "recovery_failure_reason"),
        ("dataset", "dataset"),
        ("gold", "gold"),
        ("answer_value", "answer_value_repr"),
        ("space_label", "space_label_repr"),
        ("confidence_valid", "confidence_in_range"),
        ("json_parse_valid", "full_json_parse_valid"),
        ("required_keys_present", "expected_keys_present"),
        ("answer_value_class", "answer_value_class"),
    ]:
        vc = q[col].value_counts(dropna=False).reset_index()
        vc.columns = ["value", "N"]
        vc.insert(0, "analysis", name)
        vc["rate"] = vc["N"] / len(q) if len(q) else math.nan
        q_parts.append(vc)
    qwen25_deep = pd.concat(q_parts, ignore_index=True)
    gmask = (item["model_key"] == "gemma4_31b") & (~item["exact_json"]) & item["parser_recovered"]
    g = item[gmask].copy()
    g_parts = []
    for name, col in [
        ("output_form", "output_form"),
        ("recovery_source", "recovery_source"),
        ("answer_value_class", "answer_value_class"),
        ("json_object_count", "simple_json_object_count"),
        ("dataset", "dataset"),
    ]:
        vc = g[col].value_counts(dropna=False).reset_index()
        vc.columns = ["value", "N"]
        vc.insert(0, "analysis", name)
        vc["rate"] = vc["N"] / len(g) if len(g) else math.nan
        g_parts.append(vc)
    gemma4_deep = pd.concat(g_parts, ignore_index=True)
    return {
        "item": item,
        "taxonomy": taxonomy,
        "by_dataset": by_dataset,
        "examples": examples,
        "qwen25": qwen25_deep,
        "gemma4": gemma4_deep,
    }


def wide_from_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["model_key", "model_id", "revision", "condition", "dataset", "source_index", "uid", "gold"]
    for key, g in df.groupby(keys, sort=True):
        rec = dict(zip(keys, key))
        for _, r in g.iterrows():
            m = r["method"]
            rec[f"{m}_answer"] = r["parsed_answer"] if pd.notna(r["parsed_answer"]) else ""
            rec[f"{m}_recovered"] = bool(r["parser_recovered"])
            rec[f"{m}_correct"] = bool(r["correct"])
            rec[f"{m}_finish_reason"] = r["finish_reason"]
            rec[f"{m}_output_tokens"] = r["output_tokens"]
            rec[f"{m}_length_truncation"] = bool(r["length_truncation"])
            rec[f"{m}_severe_repetition"] = bool(r["severe_repetition"])
            rec[f"{m}_reasoning_marker"] = bool(r["reasoning_marker"])
            if m == "fusion":
                rec["fusion_resolution"] = r["fusion_resolution"]
                rec["fusion_fallback_used"] = bool(r["fusion_fallback_used"])
                rec["fusion_supporting_methods"] = r["fusion_supporting_methods"]
        rows.append(rec)
    wide = pd.DataFrame(rows)
    for m in ALL_METHODS:
        for suffix, default in [("answer", ""), ("recovered", False), ("correct", False)]:
            col = f"{m}_{suffix}"
            if col not in wide:
                wide[col] = default
    return wide


def classify_agreement(row: pd.Series) -> dict[str, Any]:
    answers = {m: row[f"{m}_answer"] if row[f"{m}_recovered"] else "" for m in METHODS}
    recovered = {m: bool(row[f"{m}_recovered"]) for m in METHODS}
    rec_methods = [m for m in METHODS if recovered[m]]
    missing = [m for m in METHODS if not recovered[m]]
    majority_methods: list[str] = []
    minority_method = ""
    majority_answer = ""
    minority_answer = ""
    if len(rec_methods) == 3:
        d, c, s = answers["direct"], answers["cot"], answers["structured"]
        if d == c == s:
            state = "unanimous"
        elif d == c and d != s:
            state = "direct_cot_majority"
            majority_methods, minority_method = ["direct", "cot"], "structured"
        elif d == s and d != c:
            state = "direct_structured_majority"
            majority_methods, minority_method = ["direct", "structured"], "cot"
        elif c == s and c != d:
            state = "cot_structured_majority"
            majority_methods, minority_method = ["cot", "structured"], "direct"
        else:
            state = "all_three_different"
    else:
        if len(rec_methods) == 2:
            relation = "recovered_pair_agree" if answers[rec_methods[0]] == answers[rec_methods[1]] else "recovered_pair_disagree"
        elif len(rec_methods) == 1:
            relation = "single_recovered"
        else:
            relation = "none_recovered"
        state = f"{len(missing)}_unrecovered_{'|'.join(missing)}_{relation}"
        if len(rec_methods) == 2 and relation == "recovered_pair_agree":
            majority_methods = rec_methods
    if majority_methods:
        majority_answer = answers[majority_methods[0]]
        if minority_method:
            minority_answer = answers[minority_method]
    return {
        "agreement_state": state,
        "generated_recovered_count": len(rec_methods),
        "unrecovered_methods": "|".join(missing),
        "strict_majority_item": bool(row.get("fusion_resolution") == "strict_majority"),
        "fallback_item": bool(row.get("fusion_fallback_used")),
        "majority_methods": "|".join(majority_methods),
        "minority_method": minority_method,
        "majority_answer": majority_answer,
        "minority_answer": minority_answer,
        "majority_correct": bool(majority_answer and majority_answer == row["gold"]),
        "minority_correct": bool(minority_method and row[f"{minority_method}_correct"]),
        "minority_uniquely_correct": bool(minority_method and row[f"{minority_method}_correct"] and not (majority_answer == row["gold"])),
    }


def boot_ci_diff(a: np.ndarray, b: np.ndarray, salt: int = 0, B: int = 2000) -> tuple[float, float]:
    if len(a) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(SEED + len(a) + salt)
    vals = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, len(a), len(a))
        vals[i] = float(np.mean(a[idx] - b[idx]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def summarize_state_group(scope: str, model: str, dataset: str, state: str, g: pd.DataFrame, denom: int) -> dict[str, Any]:
    return {
        "scope": scope,
        "model_key": model,
        "dataset": dataset,
        "agreement_state": state,
        "N": len(g),
        "fraction_of_items": len(g) / denom if denom else math.nan,
        "direct_accuracy": float(g["direct_correct"].mean()) if len(g) else math.nan,
        "cot_accuracy": float(g["cot_correct"].mean()) if len(g) else math.nan,
        "structured_accuracy": float(g["structured_correct"].mean()) if len(g) else math.nan,
        "fusion_accuracy": float(g["fusion_correct"].mean()) if len(g) else math.nan,
        "fusion_minus_cot_contribution": float((g["fusion_correct"].astype(int) - g["cot_correct"].astype(int)).sum() / denom) if denom else math.nan,
    }


def fusion_analysis(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    wide = wide_from_analysis(df)
    cls = pd.DataFrame([classify_agreement(r) for _, r in wide.iterrows()])
    item = pd.concat([wide.reset_index(drop=True), cls], axis=1)
    item["fusion_minus_cot_delta"] = item["fusion_correct"].astype(int) - item["cot_correct"].astype(int)
    item["all_four_recovered"] = item[[f"{m}_recovered" for m in ALL_METHODS]].all(axis=1)

    state_rows = []
    coalition_rows = []
    stratum_rows = []
    strict_rows = []
    dataset_rows = []
    for model, mg in item.groupby("model_key", sort=True):
        for dataset, dg in mg.groupby("dataset", sort=True):
            denom = len(dg)
            diff = float(dg["fusion_minus_cot_delta"].mean())
            top_neg = (
                dg.groupby("agreement_state")["fusion_minus_cot_delta"].sum()
                .sort_values()
                .head(1)
            )
            dataset_rows.append({
                "model_key": model,
                "dataset": dataset,
                "N": denom,
                "cot_accuracy": float(dg["cot_correct"].mean()),
                "fusion_accuracy": float(dg["fusion_correct"].mean()),
                "fusion_minus_cot": diff,
                "top_negative_state": top_neg.index[0] if len(top_neg) else "",
                "top_negative_state_contribution": float(top_neg.iloc[0] / denom) if len(top_neg) else math.nan,
            })
            for state, sg in dg.groupby("agreement_state", sort=True):
                rec = summarize_state_group("dataset", model, dataset, state, sg, denom)
                state_rows.append(rec)
                stratum_rows.append(rec)
            for bucket, bg in dg.groupby(dg["strict_majority_item"].map({True: "strict_majority", False: "fallback_or_no_majority"}), sort=True):
                strict_rows.append({
                    "scope": "dataset",
                    "model_key": model,
                    "dataset": dataset,
                    "bucket": bucket,
                    "N": len(bg),
                    "fraction_of_items": len(bg) / denom,
                    "cot_accuracy": float(bg["cot_correct"].mean()) if len(bg) else math.nan,
                    "fusion_accuracy": float(bg["fusion_correct"].mean()) if len(bg) else math.nan,
                    "fusion_minus_cot_contribution": float(bg["fusion_minus_cot_delta"].sum() / denom),
                })
            for state, sg in dg[dg["minority_method"] != ""].groupby("agreement_state", sort=True):
                maj = sg["majority_correct"].astype(float).to_numpy()
                mino = sg["minority_correct"].astype(float).to_numpy()
                lo, hi = boot_ci_diff(maj, mino, salt=sum(map(ord, model + dataset + state)))
                coalition_rows.append({
                    "scope": "dataset",
                    "model_key": model,
                    "dataset": dataset,
                    "agreement_state": state,
                    "majority_methods": sg["majority_methods"].iloc[0],
                    "minority_method": sg["minority_method"].iloc[0],
                    "N": len(sg),
                    "majority_answer_accuracy": float(maj.mean()),
                    "minority_answer_accuracy": float(mino.mean()),
                    "majority_minus_minority_accuracy": float(maj.mean() - mino.mean()),
                    "CI95_low": lo,
                    "CI95_high": hi,
                    "minority_uniquely_correct_n": int(sg["minority_uniquely_correct"].sum()),
                    "minority_uniquely_correct_rate": float(sg["minority_uniquely_correct"].mean()),
                    "majority_correct_n": int(sg["majority_correct"].sum()),
                    "all_answers_wrong_n": int((~sg["majority_correct"] & ~sg["minority_correct"]).sum()),
                    "fusion_minus_cot_contribution": float(sg["fusion_minus_cot_delta"].sum() / denom),
                })
        for state in sorted(mg["agreement_state"].unique()):
            parts = []
            for dataset in DATASETS:
                dg = mg[(mg["dataset"] == dataset) & (mg["agreement_state"] == state)]
                denom = len(mg[mg["dataset"] == dataset])
                parts.append(summarize_state_group("dataset_part", model, dataset, state, dg, denom))
            state_rows.append({
                "scope": "equal_weight_macro",
                "model_key": model,
                "dataset": "MACRO",
                "agreement_state": state,
                "N": int(sum(x["N"] for x in parts)),
                "fraction_of_items": float(np.mean([x["fraction_of_items"] for x in parts])),
                "direct_accuracy": float(mg[mg["agreement_state"] == state]["direct_correct"].mean()) if (mg["agreement_state"] == state).any() else math.nan,
                "cot_accuracy": float(mg[mg["agreement_state"] == state]["cot_correct"].mean()) if (mg["agreement_state"] == state).any() else math.nan,
                "structured_accuracy": float(mg[mg["agreement_state"] == state]["structured_correct"].mean()) if (mg["agreement_state"] == state).any() else math.nan,
                "fusion_accuracy": float(mg[mg["agreement_state"] == state]["fusion_correct"].mean()) if (mg["agreement_state"] == state).any() else math.nan,
                "fusion_minus_cot_contribution": float(np.mean([x["fusion_minus_cot_contribution"] for x in parts])),
            })
            stratum_rows.append(state_rows[-1])
        for bucket in ["strict_majority", "fallback_or_no_majority"]:
            vals = []
            n_sum = 0
            sub_all = []
            for dataset in DATASETS:
                dg0 = mg[mg["dataset"] == dataset]
                bg = dg0[dg0["strict_majority_item"].map({True: "strict_majority", False: "fallback_or_no_majority"}) == bucket]
                n_sum += len(bg)
                sub_all.append(bg)
                vals.append(float(bg["fusion_minus_cot_delta"].sum() / len(dg0)))
            sub = pd.concat(sub_all) if sub_all else pd.DataFrame()
            strict_rows.append({
                "scope": "equal_weight_macro",
                "model_key": model,
                "dataset": "MACRO",
                "bucket": bucket,
                "N": n_sum,
                "fraction_of_items": float(np.mean([len(mg[(mg["dataset"] == d) & (mg["strict_majority_item"].map({True: "strict_majority", False: "fallback_or_no_majority"}) == bucket)]) / len(mg[mg["dataset"] == d]) for d in DATASETS])),
                "cot_accuracy": float(sub["cot_correct"].mean()) if len(sub) else math.nan,
                "fusion_accuracy": float(sub["fusion_correct"].mean()) if len(sub) else math.nan,
                "fusion_minus_cot_contribution": float(np.mean(vals)),
            })

    joint = item[item["all_four_recovered"]].copy()
    joint_rows = []
    for model, mg in joint.groupby("model_key", sort=True):
        for dataset, dg in mg.groupby("dataset", sort=True):
            denom = len(dg)
            for state, sg in dg.groupby("agreement_state", sort=True):
                joint_rows.append(summarize_state_group("joint_recovered_dataset", model, dataset, state, sg, denom))
        for state in sorted(mg["agreement_state"].unique()):
            vals = []
            frac = []
            for dataset in DATASETS:
                dg0 = mg[mg["dataset"] == dataset]
                sg = dg0[dg0["agreement_state"] == state]
                vals.append(float(sg["fusion_minus_cot_delta"].sum() / len(dg0)) if len(dg0) else math.nan)
                frac.append(len(sg) / len(dg0) if len(dg0) else math.nan)
            sub = mg[mg["agreement_state"] == state]
            joint_rows.append({
                "scope": "joint_recovered_equal_weight_macro",
                "model_key": model,
                "dataset": "MACRO",
                "agreement_state": state,
                "N": len(sub),
                "fraction_of_items": float(np.nanmean(frac)),
                "direct_accuracy": float(sub["direct_correct"].mean()) if len(sub) else math.nan,
                "cot_accuracy": float(sub["cot_correct"].mean()) if len(sub) else math.nan,
                "structured_accuracy": float(sub["structured_correct"].mean()) if len(sub) else math.nan,
                "fusion_accuracy": float(sub["fusion_correct"].mean()) if len(sub) else math.nan,
                "fusion_minus_cot_contribution": float(np.nanmean(vals)),
            })

    minority = item[item["minority_method"] != ""].copy()
    minority_rows = []
    for cols in [["model_key"], ["model_key", "dataset"], ["model_key", "agreement_state"], ["model_key", "dataset", "agreement_state"]]:
        for key, g in minority.groupby(cols, sort=True):
            if not isinstance(key, tuple):
                key = (key,)
            minority_rows.append({
                **dict(zip(cols, key)),
                "N": len(g),
                "minority_uniquely_correct_n": int(g["minority_uniquely_correct"].sum()),
                "minority_uniquely_correct_rate": float(g["minority_uniquely_correct"].mean()),
                "majority_correct_n": int(g["majority_correct"].sum()),
                "majority_correct_rate": float(g["majority_correct"].mean()),
                "minority_method_counts": json.dumps(g["minority_method"].value_counts().to_dict(), sort_keys=True),
            })
    return {
        "item": item,
        "state_summary": pd.DataFrame(state_rows),
        "coalition": pd.DataFrame(coalition_rows),
        "minority": pd.DataFrame(minority_rows),
        "stratum": pd.DataFrame(stratum_rows),
        "strict": pd.DataFrame(strict_rows),
        "joint": pd.DataFrame(joint_rows),
        "dataset": pd.DataFrame(dataset_rows),
    }


def fallback_summary() -> pd.DataFrame:
    path = OUTCOME / "fallback_order_sensitivity.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows = []
    macro = df[df["scope"].eq("equal_weight_macro")].copy()
    for model, g in macro.groupby("model_key", sort=True):
        if not len(g):
            continue
        rows.append({
            "model_key": model,
            "fusion_macro_min": float(g["fusion_accuracy"].min()),
            "fusion_macro_max": float(g["fusion_accuracy"].max()),
            "fusion_macro_range": float(g["fusion_accuracy"].max() - g["fusion_accuracy"].min()),
            "best_order": g.sort_values("fusion_accuracy", ascending=False)["fallback_order"].iloc[0],
            "primary_order_accuracy": float(g[g["fallback_order"].eq("direct>cot>structured")]["fusion_accuracy"].iloc[0]),
        })
    return pd.DataFrame(rows)


def validate_inputs(df: pd.DataFrame) -> dict[str, Any]:
    observed = {
        "analysis_ready_exists": (OUTCOME / "analysis_ready_item_level.parquet").exists(),
        "analysis_ready_rows": len(df),
        "parser_sha256": sha_file(PARSER),
        "models": sorted(df["model_key"].unique()),
        "methods": sorted(df["method"].unique()),
        "row_type_counts": df["row_type"].value_counts().to_dict(),
    }
    local_protocol = json.loads((ROOT / 'results/final_protocol_freeze_20260817/final_prompt_parser_hashes.json').read_text())
    if observed["parser_sha256"] != local_protocol['parser']['sha256']:
        raise RuntimeError(f"Frozen parser mismatch: {observed['parser_sha256']}")
    if len(df) != 119120:
        raise RuntimeError(f"Unexpected analysis-ready rows: {len(df)}")
    if sorted(df["model_key"].unique()) != sorted(MODEL_ORDER):
        raise RuntimeError("Unexpected model set")
    counts = df.groupby(["model_key", "method"]).size()
    for model in MODEL_ORDER:
        for method in ALL_METHODS:
            if int(counts.get((model, method), 0)) != 5956:
                raise RuntimeError(f"Bad count for {model}/{method}: {counts.get((model, method), 0)}")
    dupes = int(df.duplicated(["model_key", "dataset", "source_index", "uid", "method"]).sum())
    observed["duplicate_model_item_method_keys"] = dupes
    if dupes:
        raise RuntimeError(f"Duplicate analysis keys: {dupes}")
    manifests = {}
    for path in sorted(RAW.glob("*_completion_manifest.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        out_path = ROOT / m["output_jsonl"]
        got = sha_file(out_path)
        if got != m["output_sha256"]:
            raise RuntimeError(f"Raw checksum mismatch for {out_path}")
        manifests[m["model_key"]] = {
            "manifest": str(path.relative_to(ROOT)),
            "output_jsonl": m["output_jsonl"],
            "output_sha256": got,
            "rows": m["rows"],
            "unique_keys": m["unique_keys"],
            "complete": m["complete"],
        }
    observed["raw_generation_manifests"] = manifests
    return observed


def md_table(df: pd.DataFrame, cols: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "(no rows)"
    d = df[cols].head(max_rows).copy()
    return d.to_markdown(index=False)


def write_docs(struct: dict[str, pd.DataFrame], fus: dict[str, pd.DataFrame], validation: dict[str, Any], fb: pd.DataFrame) -> None:
    item = struct["item"]
    contract = item.groupby("model_key").agg(N=("uid","size"), exact_json_n=("exact_json","sum"), recovered_n=("parser_recovered","sum"), correct_n=("correct","sum")).reset_index()
    failure = item[~item["parser_recovered"]].groupby(["model_key","recovery_failure_reason"]).size().reset_index(name="N")
    labels = item.groupby(["model_key","space_label_repr","answer_value_class"]).size().reset_index(name="N")
    text = "# Structured contract and recovery\n\nExactness is a JSON key-set check, not answer-value validation. Recovery uses the unchanged parser. Unrecovered outputs count as incorrect. SPACE is an additional task, not merely formatting.\n\n"
    text += contract.to_markdown(index=False) + "\n\n" + failure.to_markdown(index=False) + "\n\n" + labels.to_markdown(index=False)
    (OUT / "STRUCTURED_FAILURE_ANALYSIS.md").write_text(text + "\n")
    strict = fus["strict"][fus["strict"]["scope"].eq("equal_weight_macro")]
    lines = ["# Fusion mechanism", "", "Contributions use each full dataset denominator, then equal dataset weighting. Association/decomposition does not establish causality.", "", strict.to_markdown(index=False), ""]
    for model, group in strict.groupby("model_key"):
        negatives = group[group.fusion_minus_cot_contribution.lt(0)]
        if negatives.empty:
            lines.append(f"- {model}: neither stratum has a negative macro contribution.")
        else:
            row = negatives.loc[negatives.fusion_minus_cot_contribution.idxmin()]
            lines.append(f"- {model}: largest negative contribution is {row.bucket}: {row.fusion_minus_cot_contribution:+.8f}.")
    lines += ["", "## Six fallback orders", "", fb.to_markdown(index=False), "", "## Dataset decomposition", "", fus["dataset"].to_markdown(index=False)]
    for name in ["FUSION_MECHANISM_ANALYSIS.md", "FINAL_MECHANISTIC_SUMMARY.md"]:
        (OUT/name).write_text("\n".join(lines)+"\n")
    (OUT/"METHODS_AUDIT.md").write_text("# Local input validation\n\n"+json.dumps(validation,indent=2,default=str)+"\n")


def write_outputs(struct: dict[str, pd.DataFrame], fus: dict[str, pd.DataFrame], fb: pd.DataFrame) -> None:
    struct["taxonomy"].to_csv(OUT / "structured_failure_taxonomy.csv", index=False)
    struct["by_dataset"].to_csv(OUT / "structured_failure_taxonomy_by_dataset.csv", index=False)
    struct["item"].to_parquet(OUT / "structured_failure_taxonomy_item_level.parquet", index=False)
    struct["examples"].to_csv(OUT / "structured_failure_examples.csv", index=False)
    struct["qwen25"].to_csv(OUT / "qwen25_exact_but_unrecovered_analysis.csv", index=False)
    struct["gemma4"].to_csv(OUT / "gemma4_nonexact_but_recovered_analysis.csv", index=False)
    fus["item"].to_parquet(OUT / "fusion_agreement_state_item_level.parquet", index=False)
    fus["state_summary"].to_csv(OUT / "fusion_agreement_state_summary.csv", index=False)
    fus["coalition"].to_csv(OUT / "fusion_coalition_accuracy.csv", index=False)
    fus["minority"].to_csv(OUT / "fusion_minority_correct_analysis.csv", index=False)
    fus["stratum"].to_csv(OUT / "fusion_stratum_contribution.csv", index=False)
    fus["strict"].to_csv(OUT / "fusion_strict_majority_vs_fallback.csv", index=False)
    fus["joint"].to_csv(OUT / "fusion_joint_recovered_decomposition.csv", index=False)
    fus["dataset"].to_csv(OUT / "fusion_dataset_decomposition.csv", index=False)
    fb.to_csv(OUT / "fallback_order_range_summary.csv", index=False)


def write_checksums() -> None:
    rows = []
    for path in sorted(OUT.iterdir()):
        if path.name == "checksums.sha256" or not path.is_file():
            continue
        rows.append(f"{sha_file(path)}  {path.relative_to(ROOT)}")
    (OUT / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(OUTCOME / "analysis_ready_item_level.parquet")
    validation = validate_inputs(df)
    struct = structured_analysis(df)
    fus = fusion_analysis(df)
    fb = fallback_summary()
    write_outputs(struct, fus, fb)
    write_docs(struct, fus, validation, fb)
    manifest = {
        "status": "complete",
        "created_date": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(OUT),
        "new_llm_generations_performed": 0,
        "structured_rows_classified": int(len(struct["item"])),
        "fusion_item_rows_classified": int(len(fus["item"])),
        "parser_sha256": validation["parser_sha256"],
    }
    (OUT / "mechanistic_analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_checksums()
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
