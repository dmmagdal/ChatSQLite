#!/usr/bin/env python3
"""
evaluate.py - Quantitative Evaluation Framework for ChatSQLite Models

Evaluates and benchmarks LLM model combinations across the ChatSQLite architecture:
  1. Router Stage: Table selection Precision, Recall, and F1.
  2. SQL Generation & Self-Healing: AST validity, AST safety, Pass@1, Pass@K,
     Execution Accuracy (EX), and Healing Convergence Rate (HCR).
  3. Insights & Visualization: JSON conformance, column grounding, and numerical faithfulness.
  4. System Performance: Latency, attempt overhead, and Multi-Criteria Decision Analysis (CQI/Pareto).
"""

import argparse
from dataclasses import asdict, dataclass, field
import itertools
import json
import os
import random
import re
import sqlite3
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import requests
import sqlglot
from sqlglot import exp

from helper import (
    DB_PATH,
    get_connection,
    get_db_schema,
    get_filtered_db_schema,
    get_filtered_metadata,
    get_metadata,
    get_tables,
    is_read_only_query,
)
from workflow import extract_sql


# =====================================================================
# Data Classes
# =====================================================================

@dataclass
class BenchmarkCase:
    id: str
    query: str
    difficulty: str
    gold_tables: List[str]
    gold_sql: str
    expected_chart: Optional[Dict[str, Any]] = None


@dataclass
class RouterMetrics:
    predicted_tables: List[str]
    gold_tables: List[str]
    precision: float
    recall: float
    f1: float
    exact_match: bool
    under_selected: bool
    over_selected: bool
    latency_sec: float


@dataclass
class SQLMetrics:
    pass_at_1: bool
    pass_at_k: bool
    attempts_used: int
    ast_valid: bool
    ast_safe: bool
    execution_accuracy: bool
    healing_converged: bool
    final_sql: str
    last_error: str
    latency_sec: float


@dataclass
class InsightMetrics:
    json_valid: bool
    chart_type: str
    x_col: Optional[str]
    y_col: Optional[str]
    column_grounded: bool
    faithfulness_score: float
    raw_response: str
    latency_sec: float


@dataclass
class CaseResult:
    case_id: str
    difficulty: str
    router: Optional[RouterMetrics] = None
    sql: Optional[SQLMetrics] = None
    insight: Optional[InsightMetrics] = None
    total_latency_sec: float = 0.0


@dataclass
class ModelCombination:
    router_model: str
    sql_model: str
    chat_model: str

    def to_key(self) -> str:
        return f"router={self.router_model}|sql={self.sql_model}|chat={self.chat_model}"


@dataclass
class CombinationSummary:
    combination: ModelCombination
    total_cases: int
    router_precision: float
    router_recall: float
    router_f1: float
    router_exact_match_rate: float
    sql_ast_valid_rate: float
    sql_ast_safe_rate: float
    sql_pass_at_1_rate: float
    sql_pass_at_k_rate: float
    sql_execution_accuracy: float
    healing_convergence_rate: float
    mean_attempts: float
    insight_json_conformance_rate: float
    insight_column_grounding_rate: float
    insight_faithfulness_rate: float
    mean_latency_sec: float
    p90_latency_sec: float
    composite_quality_index: float = 0.0
    is_pareto_optimal: bool = False
    case_results: List[CaseResult] = field(default_factory=list)


# =====================================================================
# Quantitative Metrics Algorithms
# =====================================================================

def compute_router_metrics(
    predicted: List[str], gold: List[str], latency_sec: float = 0.0
) -> RouterMetrics:
    """
    Computes table selection Precision, Recall, and F1 score.
    """
    pred_set = {t.lower().strip() for t in predicted}
    gold_set = {t.lower().strip() for t in gold}

    if not pred_set and not gold_set:
        precision, recall, f1 = 1.0, 1.0, 1.0
    elif not pred_set or not gold_set:
        precision = 1.0 if not pred_set and not gold_set else 0.0
        recall = 1.0 if not pred_set and not gold_set else 0.0
        f1 = 0.0
    else:
        true_positives = len(pred_set & gold_set)
        precision = true_positives / len(pred_set)
        recall = true_positives / len(gold_set)
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

    exact_match = pred_set == gold_set
    under_selected = len(gold_set - pred_set) > 0
    over_selected = len(pred_set - gold_set) > 0

    return RouterMetrics(
        predicted_tables=sorted(list(pred_set)),
        gold_tables=sorted(list(gold_set)),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        exact_match=exact_match,
        under_selected=under_selected,
        over_selected=over_selected,
        latency_sec=round(latency_sec, 4),
    )


def normalize_value(val: Any) -> Any:
    """Normalize single cell value for comparison."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        # Normalize float precision
        return round(float(val), 4)
    if isinstance(val, str):
        return val.strip().lower()
    return val


def normalize_rows(
    rows: List[Tuple[Any, ...]], require_order: bool = False
) -> List[Tuple[Any, ...]]:
    """
    Normalizes SQL result rows by standardizing types, stripping whitespace,
    and sorting rows if order is not strictly mandated by the query.
    """
    norm = [tuple(normalize_value(x) for x in r) for r in rows]
    if not require_order:
        # Sort stably using representation for heterogeneous types
        norm.sort(key=lambda item: str(item))
    return norm


def compare_execution_results(
    pred_rows: List[Tuple[Any, ...]],
    gold_rows: List[Tuple[Any, ...]],
    gold_sql: str = "",
) -> bool:
    """
    Evaluates Execution Accuracy (EX) between predicted and gold queries.
    Determines if row ordering is required based on presence of ORDER BY in gold SQL.
    """
    if len(pred_rows) != len(gold_rows):
        return False

    require_order = bool(re.search(r"\bORDER\s+BY\b", gold_sql, re.IGNORECASE))
    norm_pred = normalize_rows(pred_rows, require_order=require_order)
    norm_gold = normalize_rows(gold_rows, require_order=require_order)

    return norm_pred == norm_gold


def compute_faithfulness(summary_text: str, result_rows: List[Tuple[Any, ...]]) -> float:
    """
    Measures the numerical faithfulness of LLM-generated summaries by
    verifying if cited numbers correspond to values in the executed result set.
    """
    if not summary_text:
        return 0.0

    # Extract all numbers from the summary (integers, floats, dollar amounts)
    # Strip currency signs and commas
    cleaned_text = re.sub(r"[$,%]", "", summary_text)
    cited_nums = set()
    for token in re.findall(r"-?\b\d+(?:\.\d+)?\b", cleaned_text):
        try:
            cited_nums.add(round(float(token), 2))
        except ValueError:
            pass

    if not cited_nums:
        return 1.0  # No numerical claims made; not unfaithful

    # Extract all numbers from result set
    truth_nums = set()
    for row in result_rows:
        for val in row:
            if isinstance(val, (int, float)):
                truth_nums.add(round(float(val), 2))
            elif isinstance(val, str):
                cleaned_val = re.sub(r"[$,%]", "", val)
                for match in re.findall(r"-?\b\d+(?:\.\d+)?\b", cleaned_val):
                    try:
                        truth_nums.add(round(float(match), 2))
                    except ValueError:
                        pass

    grounded_count = sum(1 for num in cited_nums if num in truth_nums)
    return round(grounded_count / len(cited_nums), 4)


def compute_cqi(
    summary: CombinationSummary,
    weights: Optional[Dict[str, float]] = None,
    norm_latency: float = 0.0,
) -> float:
    """
    Calculates Composite Quality Index (CQI) via Multi-Criteria Decision Analysis:
      CQI = w_ex * EX + w_heal * HCR + w_route * F1 + w_vis * VisScore - w_lat * NormLatency
    """
    w = {
        "ex": 0.40,
        "heal": 0.20,
        "route": 0.15,
        "vis": 0.10,
        "lat": 0.15,
    }
    if weights:
        w.update(weights)

    vis_score = (
        0.5 * summary.insight_json_conformance_rate
        + 0.5 * summary.insight_column_grounding_rate
    )

    cqi = (
        w["ex"] * summary.sql_execution_accuracy
        + w["heal"] * summary.healing_convergence_rate
        + w["route"] * summary.router_f1
        + w["vis"] * vis_score
        - w["lat"] * norm_latency
    )
    return round(max(0.0, min(1.0, cqi)), 4)


def identify_pareto_frontier(
    summaries: List[CombinationSummary],
) -> List[CombinationSummary]:
    """
    Identifies Pareto-optimal configurations based on (Accuracy vs. Latency).
    Config A dominates B if Accuracy(A) >= Accuracy(B) and Latency(A) <= Latency(B)
    with at least one strict inequality.
    """
    for a in summaries:
        dominated = False
        for b in summaries:
            if a == b:
                continue
            # Criterion 1: Accuracy (higher is better)
            # Criterion 2: Latency (lower is better)
            better_or_equal = (
                b.sql_execution_accuracy >= a.sql_execution_accuracy
                and b.mean_latency_sec <= a.mean_latency_sec
            )
            strictly_better = (
                b.sql_execution_accuracy > a.sql_execution_accuracy
                or b.mean_latency_sec < a.mean_latency_sec
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        a.is_pareto_optimal = not dominated
    return summaries


# =====================================================================
# LLM Client (Ollama & Mock)
# =====================================================================

class LLMClient:
    """Client for Ollama API with optional mock support for deterministic testing."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        mock: bool = False,
        mock_handler: Optional[Callable[[str, str, str], str]] = None,
    ):
        self.base_url = base_url
        self.mock = mock
        self.mock_handler = mock_handler

    def generate(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        timeout: int = 120,
    ) -> str:
        if self.mock:
            if self.mock_handler:
                return self.mock_handler(model, system_prompt, user_prompt)
            return self._default_mock(system_prompt, user_prompt)

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }

        resp = requests.post(
            f"{self.base_url}/api/chat/", json=payload, timeout=timeout
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def _default_mock(self, system_prompt: str, user_prompt: str) -> str:
        # Default mock heuristic for testing harness without Ollama
        if "Database Router" in system_prompt:
            return '["aux_presidents", "fec_cm", "fec_cn"]'
        if "Data Analyst Agent" in system_prompt or "executive summary" in user_prompt:
            return '{"summary": "Found 5 records.", "chart_type": "bar", "x_col": "CMTE_ST", "y_col": "committee_count", "title": "Chart"}'
        # SQL generator mock
        return "```sql\nSELECT * FROM aux_presidents LIMIT 5;\n```"


# =====================================================================
# Evaluator Engine
# =====================================================================

class BenchmarkRunner:
    """Runs test cases against specified model configurations and computes quantitative metrics."""

    def __init__(
        self,
        db_path: str = DB_PATH,
        client: Optional[LLMClient] = None,
        max_retries: int = 3,
    ):
        self.db_path = db_path
        self.client = client or LLMClient()
        self.max_retries = max_retries

    def load_suite(self, suite_path: str) -> List[BenchmarkCase]:
        with open(suite_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [BenchmarkCase(**item) for item in raw]

    def execute_query(
        self, query: str, max_limit: int = 100
    ) -> Tuple[List[Tuple[Any, ...]], List[str], str]:
        """
        Executes query read-only on sqlite3.
        Returns (rows, column_names, error_string).
        """
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            conn.execute("PRAGMA query_only = TRUE;")
            cursor = conn.execute(query)
            rows = cursor.fetchmany(max_limit)
            col_names = (
                [desc[0] for desc in cursor.description]
                if cursor.description
                else []
            )
            conn.close()
            return rows, col_names, ""
        except Exception as e:
            return [], [], str(e)

    # -----------------------------------------------------------------
    # Stage 1: Router Evaluation
    # -----------------------------------------------------------------
    def eval_router(
        self, model: str, case: BenchmarkCase, all_metadata: str
    ) -> RouterMetrics:
        all_tables = get_tables()
        system_prompt = f"""You are a Database Router.
Given a user's analytical question, identify which database tables from the list below are necessary to answer it.

{all_metadata}

Respond ONLY with a JSON array of string table names. Example: ["fec_cm", "fec_pas2"]"""

        user_prompt = f"User Request: {case.query}"

        start_t = time.time()
        try:
            raw_response = self.client.generate(model, system_prompt, user_prompt)
            match = re.search(r"\[.*?\]", raw_response, re.DOTALL)
            if match:
                selected_tables = json.loads(match.group(0))
                predicted = [t for t in selected_tables if t in all_tables]
            else:
                predicted = all_tables
        except Exception:
            predicted = all_tables
        latency = time.time() - start_t

        return compute_router_metrics(
            predicted=predicted, gold=case.gold_tables, latency_sec=latency
        )

    # -----------------------------------------------------------------
    # Stage 2: SQL Generation & Self-Healing Evaluation
    # -----------------------------------------------------------------
    def eval_sql(
        self,
        model: str,
        case: BenchmarkCase,
        schema_str: str,
        metadata_str: str,
        gold_rows: List[Tuple[Any, ...]],
    ) -> SQLMetrics:
        system_prompt = f"""You are a SQLite Data Analyst Agent.
Your job is to generate a SINGLE valid SQLite SELECT query based on the user's request.

CRITICAL RULES:
1. Output ONLY the raw SQL query inside a ```sql ``` code block.
2. NEVER issue INSERT, UPDATE, DELETE, DROP, ALTER, or PRAGMA statements.
3. Keep results limited to 100 rows unless explicitly performing aggregated SUM/COUNT queries.

TABLE METADATA:
{metadata_str}

DATABASE SCHEMA:
{schema_str}
"""
        current_prompt = f"User Request: {case.query}"

        pass_at_1 = False
        pass_at_k = False
        ast_valid = False
        ast_safe = False
        healing_converged = False
        final_sql = ""
        last_error = ""
        start_t = time.time()

        for attempt in range(1, self.max_retries + 1):
            raw_response = self.client.generate(model, system_prompt, current_prompt)

            try:
                extracted = extract_sql(raw_response)
                final_sql = extracted
            except ValueError as ve:
                last_error = str(ve)
                current_prompt = f"Previous response contained no valid SQL. Error: {last_error}\nPlease provide valid SQL in ```sql ```."
                continue

            # AST validation
            try:
                parsed = sqlglot.parse(final_sql, dialect="sqlite")
                ast_valid = bool(parsed and parsed[0] is not None)
            except Exception:
                ast_valid = False

            # AST safety check
            ast_safe = is_read_only_query(final_sql)
            if not ast_safe:
                last_error = "Security Exception: Query failed read-only AST check."
                current_prompt = f"Previous SQL: {final_sql}\nError: {last_error}\nPlease rewrite to use strictly SELECT statements."
                continue

            # Execution check
            pred_rows, _, exec_err = self.execute_query(final_sql)
            if exec_err:
                last_error = exec_err
                current_prompt = (
                    f"The SQL query you provided failed with the following error:\n"
                    f"SQL: {final_sql}\n"
                    f"SQLite Error: {exec_err}\n\n"
                    f"Please correct the query syntax or column references and try again."
                )
                continue

            # Execution succeeded
            last_error = ""
            if attempt == 1:
                pass_at_1 = True
            else:
                healing_converged = True
            pass_at_k = True
            break

        total_latency = time.time() - start_t

        # Execution Accuracy check vs Gold
        ex_acc = False
        if pass_at_k and not last_error:
            pred_rows, _, _ = self.execute_query(final_sql)
            ex_acc = compare_execution_results(
                pred_rows, gold_rows, gold_sql=case.gold_sql
            )

        return SQLMetrics(
            pass_at_1=pass_at_1,
            pass_at_k=pass_at_k,
            attempts_used=attempt,
            ast_valid=ast_valid,
            ast_safe=ast_safe,
            execution_accuracy=ex_acc,
            healing_converged=healing_converged,
            final_sql=final_sql,
            last_error=last_error,
            latency_sec=round(total_latency, 4),
        )

    # -----------------------------------------------------------------
    # Stage 3: Insights & Visualizations Evaluation
    # -----------------------------------------------------------------
    def eval_insight(
        self,
        model: str,
        case: BenchmarkCase,
        pred_rows: List[Tuple[Any, ...]],
        col_names: List[str],
    ) -> InsightMetrics:
        if not pred_rows or not col_names:
            return InsightMetrics(
                json_valid=False,
                chart_type="none",
                x_col=None,
                y_col=None,
                column_grounded=False,
                faithfulness_score=0.0,
                raw_response="No data to analyze.",
                latency_sec=0.0,
            )

        df = pd.DataFrame(pred_rows, columns=col_names)
        summary_prompt = f"""Analyze the dataset returned for the question: "{case.query}"
Data Sample (Top 5 Rows):
{df.head(5).to_json(orient='records')}

Task:
1. Write a concise 2-3 sentence executive summary.
2. Recommend a chart type ('bar', 'line', 'pie', 'scatter', or 'none') to visualize this result.
Respond in JSON format with keys: "summary", "chart_type", "x_col", "y_col", "title".
"""
        start_t = time.time()
        raw = self.client.generate(
            model,
            "You are a Data Analyst. Return valid JSON only.",
            summary_prompt,
        )
        latency = time.time() - start_t

        json_valid = False
        chart_type = "none"
        x_col = None
        y_col = None
        summary_text = ""
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                cfg = json.loads(match.group(0))
                json_valid = all(
                    k in cfg
                    for k in ["summary", "chart_type", "x_col", "y_col", "title"]
                )
                chart_type = cfg.get("chart_type", "none")
                x_col = cfg.get("x_col")
                y_col = cfg.get("y_col")
                summary_text = cfg.get("summary", "")
        except Exception:
            pass

        # Check column grounding
        grounded = True
        if chart_type != "none":
            grounded = bool(x_col in col_names and y_col in col_names)

        # Numerical faithfulness
        faithfulness = compute_faithfulness(summary_text, pred_rows)

        return InsightMetrics(
            json_valid=json_valid,
            chart_type=chart_type,
            x_col=x_col,
            y_col=y_col,
            column_grounded=grounded,
            faithfulness_score=faithfulness,
            raw_response=raw,
            latency_sec=round(latency, 4),
        )

    # -----------------------------------------------------------------
    # End-to-End Evaluation of a Model Combination
    # -----------------------------------------------------------------
    def evaluate_combination(
        self, combo: ModelCombination, suite: List[BenchmarkCase]
    ) -> CombinationSummary:
        case_results = []
        all_metadata = get_metadata()

        for case in suite:
            # 1. Gold query execution for baseline truth
            gold_rows, gold_cols, gold_err = self.execute_query(case.gold_sql)

            # 2. Router evaluation
            router_res = self.eval_router(
                model=combo.router_model, case=case, all_metadata=all_metadata
            )

            # Retrieve schema & metadata filtered by router selection
            filtered_schema = get_filtered_db_schema(router_res.predicted_tables)
            filtered_meta = get_filtered_metadata(router_res.predicted_tables)

            # 3. SQL generation & healing evaluation
            sql_res = self.eval_sql(
                model=combo.sql_model,
                case=case,
                schema_str=filtered_schema,
                metadata_str=filtered_meta,
                gold_rows=gold_rows,
            )

            # 4. Insight / Visualization evaluation
            pred_rows, pred_cols, _ = (
                self.execute_query(sql_res.final_sql)
                if sql_res.pass_at_k
                else ([], [], "")
            )
            insight_res = self.eval_insight(
                model=combo.chat_model,
                case=case,
                pred_rows=pred_rows,
                col_names=pred_cols,
            )

            total_lat = (
                router_res.latency_sec
                + sql_res.latency_sec
                + insight_res.latency_sec
            )
            case_results.append(
                CaseResult(
                    case_id=case.id,
                    difficulty=case.difficulty,
                    router=router_res,
                    sql=sql_res,
                    insight=insight_res,
                    total_latency_sec=round(total_lat, 4),
                )
            )

        # Aggregate Summary
        n = len(suite)
        router_p = sum(c.router.precision for c in case_results) / n
        router_r = sum(c.router.recall for c in case_results) / n
        router_f1 = sum(c.router.f1 for c in case_results) / n
        router_em = sum(1 for c in case_results if c.router.exact_match) / n

        sql_valid = sum(1 for c in case_results if c.sql.ast_valid) / n
        sql_safe = sum(1 for c in case_results if c.sql.ast_safe) / n
        sql_p1 = sum(1 for c in case_results if c.sql.pass_at_1) / n
        sql_pk = sum(1 for c in case_results if c.sql.pass_at_k) / n
        sql_ex = sum(1 for c in case_results if c.sql.execution_accuracy) / n

        # Healing convergence rate: queries that failed on attempt 1 but succeeded later
        failed_initially = [c for c in case_results if not c.sql.pass_at_1]
        healed = [c for c in failed_initially if c.sql.healing_converged]
        hcr = (len(healed) / len(failed_initially)) if failed_initially else 1.0

        mean_attempts = sum(c.sql.attempts_used for c in case_results) / n

        ins_json = sum(1 for c in case_results if c.insight.json_valid) / n
        ins_ground = sum(1 for c in case_results if c.insight.column_grounded) / n
        ins_faith = sum(c.insight.faithfulness_score for c in case_results) / n

        latencies = [c.total_latency_sec for c in case_results]
        mean_lat = sum(latencies) / n
        latencies_sorted = sorted(latencies)
        p90_idx = int(0.9 * n)
        p90_lat = (
            latencies_sorted[min(p90_idx, n - 1)] if latencies_sorted else 0.0
        )

        return CombinationSummary(
            combination=combo,
            total_cases=n,
            router_precision=round(router_p, 4),
            router_recall=round(router_r, 4),
            router_f1=round(router_f1, 4),
            router_exact_match_rate=round(router_em, 4),
            sql_ast_valid_rate=round(sql_valid, 4),
            sql_ast_safe_rate=round(sql_safe, 4),
            sql_pass_at_1_rate=round(sql_p1, 4),
            sql_pass_at_k_rate=round(sql_pk, 4),
            sql_execution_accuracy=round(sql_ex, 4),
            healing_convergence_rate=round(hcr, 4),
            mean_attempts=round(mean_attempts, 2),
            insight_json_conformance_rate=round(ins_json, 4),
            insight_column_grounding_rate=round(ins_ground, 4),
            insight_faithfulness_rate=round(ins_faith, 4),
            mean_latency_sec=round(mean_lat, 4),
            p90_latency_sec=round(p90_lat, 4),
            case_results=case_results,
        )


# =====================================================================
# Decision Analysis & Report Generator
# =====================================================================

def generate_combinations(
    matrix_data: Union[Dict[str, List[str]], List[Dict[str, str]]],
    strategy: str = "grid",
    sample_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[ModelCombination]:
    """
    Generates model combinations from either:
      1. A candidate pool dictionary:
         {"router_model": [...], "sql_model": [...], "chat_model": [...]}
         leveraging itertools.product to build the Cartesian product.
      2. An explicit list of combination mappings:
         [{"router_model": "...", "sql_model": "...", "chat_model": "..."}, ...]

    Supports sampling a subset with a deterministic random seed.
    """
    if isinstance(matrix_data, dict):
        routers = (
            matrix_data.get("router_model")
            or matrix_data.get("router")
            or ["qwen2.5-coder:1.5b"]
        )
        sqls = (
            matrix_data.get("sql_model")
            or matrix_data.get("sql")
            or ["qwen2.5-coder:1.5b"]
        )
        chats = (
            matrix_data.get("chat_model")
            or matrix_data.get("chat")
            or ["lfm2.5-thinking:1.2b"]
        )

        combos = [
            ModelCombination(router_model=r, sql_model=s, chat_model=c)
            for r, s, c in itertools.product(routers, sqls, chats)
        ]
    elif isinstance(matrix_data, list):
        combos = [ModelCombination(**item) for item in matrix_data]
    else:
        raise ValueError(f"Unsupported matrix data format: {type(matrix_data)}")

    if strategy == "sample" and sample_size and sample_size < len(combos):
        rng = random.Random(seed)
        combos = rng.sample(combos, sample_size)

    return combos


def search_combinations_with_restarts(
    matrix_data: Dict[str, List[str]],
    suite: List[BenchmarkCase],
    runner: BenchmarkRunner,
    restarts: int = 2,
    seed: int = 42,
    weights: Optional[Dict[str, float]] = None,
) -> List[CombinationSummary]:
    """
    Explores the model combination space via Coordinate Descent / Local Greedy Search
    with random restarts to find the optimal combination without testing all permutations.

    Algorithm:
      1. For each restart r = 0 .. restarts-1:
         - Seed random state with (seed + r * 1000).
         - Select a random initial state (router_0, sql_0, chat_0).
         - Evaluate state and compute its baseline score.
         - Iteratively vary one dimension at a time (Router -> SQL -> Chat).
         - Move to the best neighbor that improves the score.
         - Stop when no single dimension change improves score (local optimum).
      2. Cache all evaluations so no model combination is ever evaluated twice.
      3. Compute normalized scores and Pareto frontier over all visited combinations.
    """
    routers = (
        matrix_data.get("router_model")
        or matrix_data.get("router")
        or ["qwen2.5-coder:1.5b"]
    )
    sqls = (
        matrix_data.get("sql_model")
        or matrix_data.get("sql")
        or ["qwen2.5-coder:1.5b"]
    )
    chats = (
        matrix_data.get("chat_model")
        or matrix_data.get("chat")
        or ["lfm2.5-thinking:1.2b"]
    )

    evaluated_cache: Dict[str, CombinationSummary] = {}

    def get_or_eval(combo: ModelCombination) -> CombinationSummary:
        key = combo.to_key()
        if key not in evaluated_cache:
            evaluated_cache[key] = runner.evaluate_combination(combo, suite)
        return evaluated_cache[key]

    print(
        f"🔍 Starting Coordinate Descent exploration ({restarts} restarts, seed={seed})..."
    )

    for restart_idx in range(restarts):
        rng = random.Random(seed + restart_idx * 1000)
        curr_r = rng.choice(routers)
        curr_s = rng.choice(sqls)
        curr_c = rng.choice(chats)
        curr_combo = ModelCombination(curr_r, curr_s, curr_c)
        curr_summary = get_or_eval(curr_combo)
        curr_score = compute_cqi(curr_summary, weights=weights, norm_latency=0.0)

        print(
            f"\n🔄 Restart #{restart_idx + 1}/{restarts} starting at: "
            f"[{curr_r} | {curr_s} | {curr_c}] (Initial CQI: {curr_score:.4f})"
        )

        improved = True
        step = 0
        while improved and step < 10:
            step += 1
            improved = False

            # Coordinate 1: Router
            best_r_combo = curr_combo
            best_r_score = curr_score
            for r_cand in routers:
                if r_cand == curr_combo.router_model:
                    continue
                cand_combo = ModelCombination(
                    r_cand, curr_combo.sql_model, curr_combo.chat_model
                )
                cand_summary = get_or_eval(cand_combo)
                cand_score = compute_cqi(
                    cand_summary, weights=weights, norm_latency=0.0
                )
                if cand_score > best_r_score:
                    best_r_score = cand_score
                    best_r_combo = cand_combo

            if best_r_score > curr_score:
                print(
                    f"  ↗ Step {step} (Router): {curr_combo.router_model} -> {best_r_combo.router_model} (CQI: {best_r_score:.4f})"
                )
                curr_combo = best_r_combo
                curr_score = best_r_score
                improved = True

            # Coordinate 2: SQL Model
            best_s_combo = curr_combo
            best_s_score = curr_score
            for s_cand in sqls:
                if s_cand == curr_combo.sql_model:
                    continue
                cand_combo = ModelCombination(
                    curr_combo.router_model, s_cand, curr_combo.chat_model
                )
                cand_summary = get_or_eval(cand_combo)
                cand_score = compute_cqi(
                    cand_summary, weights=weights, norm_latency=0.0
                )
                if cand_score > best_s_score:
                    best_s_score = cand_score
                    best_s_combo = cand_combo

            if best_s_score > curr_score:
                print(
                    f"  ↗ Step {step} (SQL): {curr_combo.sql_model} -> {best_s_combo.sql_model} (CQI: {best_s_score:.4f})"
                )
                curr_combo = best_s_combo
                curr_score = best_s_score
                improved = True

            # Coordinate 3: Chat Model
            best_c_combo = curr_combo
            best_c_score = curr_score
            for c_cand in chats:
                if c_cand == curr_combo.chat_model:
                    continue
                cand_combo = ModelCombination(
                    curr_combo.router_model, curr_combo.sql_model, c_cand
                )
                cand_summary = get_or_eval(cand_combo)
                cand_score = compute_cqi(
                    cand_summary, weights=weights, norm_latency=0.0
                )
                if cand_score > best_c_score:
                    best_c_score = cand_score
                    best_c_combo = cand_combo

            if best_c_score > curr_score:
                print(
                    f"  ↗ Step {step} (Chat): {curr_combo.chat_model} -> {best_c_combo.chat_model} (CQI: {best_c_score:.4f})"
                )
                curr_combo = best_c_combo
                curr_score = best_c_score
                improved = True

        print(
            f"🏁 Restart #{restart_idx + 1} converged to local optimum: {curr_combo.to_key()} (CQI: {curr_score:.4f})"
        )

    all_summaries = list(evaluated_cache.values())

    # Re-normalize latencies across all explored combinations
    all_lats = [s.mean_latency_sec for s in all_summaries]
    min_lat, max_lat = min(all_lats), max(all_lats)
    lat_range = (max_lat - min_lat) if max_lat > min_lat else 1.0

    for s in all_summaries:
        norm_lat = (
            (s.mean_latency_sec - min_lat) / lat_range
            if max_lat > min_lat
            else 0.0
        )
        s.composite_quality_index = compute_cqi(
            s, weights=weights, norm_latency=norm_lat
        )

    identify_pareto_frontier(all_summaries)
    all_summaries.sort(key=lambda s: s.composite_quality_index, reverse=True)
    return all_summaries


def evaluate_matrix(
    combos: List[ModelCombination],
    suite: List[BenchmarkCase],
    runner: BenchmarkRunner,
    weights: Optional[Dict[str, float]] = None,
) -> List[CombinationSummary]:
    """
    Evaluates a matrix of model combinations, calculates min-max normalized latencies,
    computes Composite Quality Index (CQI), and flags Pareto-optimal combinations.
    """
    summaries = [runner.evaluate_combination(combo, suite) for combo in combos]

    # Normalize latencies across all combinations
    all_lats = [s.mean_latency_sec for s in summaries]
    min_lat, max_lat = min(all_lats), max(all_lats)
    lat_range = (max_lat - min_lat) if max_lat > min_lat else 1.0

    for s in summaries:
        norm_lat = (s.mean_latency_sec - min_lat) / lat_range if max_lat > min_lat else 0.0
        s.composite_quality_index = compute_cqi(s, weights=weights, norm_latency=norm_lat)

    identify_pareto_frontier(summaries)
    summaries.sort(key=lambda s: s.composite_quality_index, reverse=True)
    return summaries


def format_markdown_report(
    summaries: List[CombinationSummary], suite: List[BenchmarkCase]
) -> str:
    """Formats benchmark results into a clean, GitHub-compatible markdown report."""
    md = []
    md.append("# ChatSQLite Model Evaluation Benchmark Report\n")
    md.append(f"**Total Benchmark Test Cases:** {len(suite)}")
    md.append(f"**Date Evaluated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    if summaries:
        best = summaries[0]
        md.append("## 🏆 Best Configuration: Champion Overview\n")
        md.append(
            f"- **Top Model Combination:** `{best.combination.router_model}` (Router) + "
            f"`{best.combination.sql_model}` (SQL) + `{best.combination.chat_model}` (Chat)"
        )
        md.append(
            f"- **Composite Quality Index (CQI):** **{best.composite_quality_index:.4f}** / 1.0000"
        )
        md.append("- **Detailed Score Breakdown:**")
        md.append(
            f"  - **SQL Execution Accuracy (EX):** {best.sql_execution_accuracy*100:.1f}%"
        )
        md.append(
            f"  - **Pass@1 (Zero-Shot SQL):** {best.sql_pass_at_1_rate*100:.1f}%"
        )
        md.append(
            f"  - **Pass@K (After Self-Healing):** {best.sql_pass_at_k_rate*100:.1f}% "
            f"(Healing Convergence: {best.healing_convergence_rate*100:.1f}%, Mean Attempts: {best.mean_attempts:.2f})"
        )
        md.append(
            f"  - **Router Performance (F1):** {best.router_f1:.3f} "
            f"(Precision: {best.router_precision*100:.1f}%, Recall: {best.router_recall*100:.1f}%, Exact Match: {best.router_exact_match_rate*100:.1f}%)"
        )
        md.append(
            f"  - **Visualization & Insights:** JSON Conformance: {best.insight_json_conformance_rate*100:.1f}%, "
            f"Column Grounded: {best.insight_column_grounding_rate*100:.1f}%, Faithfulness: {best.insight_faithfulness_rate*100:.1f}%"
        )
        md.append(
            f"  - **Latency:** Mean: {best.mean_latency_sec:.2f}s, P90: {best.p90_latency_sec:.2f}s"
        )

        pareto_fast = min(
            (s for s in summaries if s.is_pareto_optimal),
            key=lambda s: s.mean_latency_sec,
            default=None,
        )
        if pareto_fast and pareto_fast != best:
            md.append(
                f"\n- **⚡ Low-Latency Pareto Alternative:** `{pareto_fast.combination.router_model}` / "
                f"`{pareto_fast.combination.sql_model}` / `{pareto_fast.combination.chat_model}`"
            )
            md.append(
                f"  - Mean Latency: **{pareto_fast.mean_latency_sec:.2f}s** | "
                f"EX Acc: **{pareto_fast.sql_execution_accuracy*100:.1f}%** | "
                f"CQI: **{pareto_fast.composite_quality_index:.4f}**"
            )
        md.append("\n---\n")

    md.append("## Executive Leaderboard (Ranked by Composite Quality Index)\n")
    md.append(
        "| Rank | Combination (Router / SQL / Chat) | CQI Score | EX Acc | Pass@1 | Pass@K | HCR | Router F1 | Latency (Mean) | Pareto Optimal? |"
    )
    md.append(
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    )

    for rank, s in enumerate(summaries, 1):
        combo_str = f"`{s.combination.router_model}` / `{s.combination.sql_model}` / `{s.combination.chat_model}`"
        pareto_badge = "🌟 **Yes**" if s.is_pareto_optimal else "No"
        md.append(
            f"| {rank} | {combo_str} | **{s.composite_quality_index:.4f}** | "
            f"{s.sql_execution_accuracy*100:.1f}% | {s.sql_pass_at_1_rate*100:.1f}% | "
            f"{s.sql_pass_at_k_rate*100:.1f}% | {s.healing_convergence_rate*100:.1f}% | "
            f"{s.router_f1:.2f} | {s.mean_latency_sec:.2f}s | {pareto_badge} |"
        )

    md.append("\n## Detailed Stage Breakdown\n")
    md.append("### 1. Text-to-SQL & Self-Healing Performance\n")
    md.append(
        "| Model Combination | AST Valid | AST Safe | Pass@1 | Pass@K | EX Acc | Mean Attempts | HCR |"
    )
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for s in summaries:
        c_name = f"`{s.combination.sql_model}`"
        md.append(
            f"| {c_name} | {s.sql_ast_valid_rate*100:.1f}% | {s.sql_ast_safe_rate*100:.1f}% | "
            f"{s.sql_pass_at_1_rate*100:.1f}% | {s.sql_pass_at_k_rate*100:.1f}% | "
            f"{s.sql_execution_accuracy*100:.1f}% | {s.mean_attempts:.2f} | {s.healing_convergence_rate*100:.1f}% |"
        )

    md.append("\n### 2. Router & Visualization Performance\n")
    md.append(
        "| Router Model | Router Precision | Router Recall | Router F1 | Chat Model | JSON Valid | Column Grounded | Faithfulness |"
    )
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for s in summaries:
        md.append(
            f"| `{s.combination.router_model}` | {s.router_precision*100:.1f}% | "
            f"{s.router_recall*100:.1f}% | {s.router_f1:.3f} | "
            f"`{s.combination.chat_model}` | {s.insight_json_conformance_rate*100:.1f}% | "
            f"{s.insight_column_grounding_rate*100:.1f}% | {s.insight_faithfulness_rate*100:.1f}% |"
        )

    return "\n".join(md)


# =====================================================================
# CLI Entry Point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Quantitative Model Evaluation Framework for ChatSQLite"
    )
    parser.add_argument(
        "--suite",
        default="eval_suite.json",
        help="Path to evaluation benchmark JSON suite.",
    )
    parser.add_argument(
        "--db",
        default=DB_PATH,
        help="Path to SQLite database file (default: fec_data.db).",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        help="Base URL for Ollama service.",
    )
    parser.add_argument(
        "--router-model",
        default="qwen2.5-coder:1.5b",
        help="Model name for Table Router.",
    )
    parser.add_argument(
        "--sql-model",
        default="qwen2.5-coder:1.5b",
        help="Model name for SQL Generator.",
    )
    parser.add_argument(
        "--chat-model",
        default="lfm2.5-thinking:1.2b",
        help="Model name for Insights / Chart generator.",
    )
    parser.add_argument(
        "--matrix",
        help="Optional JSON file defining candidate models per role (dict) or an array of combinations.",
    )
    parser.add_argument(
        "--strategy",
        choices=["grid", "sample", "explore"],
        default="grid",
        help="Search strategy: 'grid' (Cartesian product), 'sample' (random seeded subset), 'explore' (local search with restarts).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of combinations to randomly sample when strategy is 'sample'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling and restarts.",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=2,
        help="Number of random restarts for the 'explore' strategy.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Maximum self-healing SQL retry attempts.",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Path to save structured evaluation results.",
    )
    parser.add_argument(
        "--report",
        default="benchmark_report.md",
        help="Path to save Markdown summary report.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode (fast, deterministic testing without live Ollama).",
    )

    args = parser.parse_args()

    client = LLMClient(base_url=args.ollama_url, mock=args.mock)
    runner = BenchmarkRunner(
        db_path=args.db, client=client, max_retries=args.retries
    )
    suite = runner.load_suite(args.suite)

    summaries: List[CombinationSummary] = []

    if args.matrix:
        with open(args.matrix, "r", encoding="utf-8") as f:
            raw_matrix = json.load(f)

        if args.strategy == "explore" and isinstance(raw_matrix, dict):
            summaries = search_combinations_with_restarts(
                matrix_data=raw_matrix,
                suite=suite,
                runner=runner,
                restarts=args.restarts,
                seed=args.seed,
            )
        else:
            combos = generate_combinations(
                matrix_data=raw_matrix,
                strategy=args.strategy,
                sample_size=args.sample_size,
                seed=args.seed,
            )
            print(
                f"🚀 Running quantitative evaluation across {len(combos)} model combinations and {len(suite)} test cases (strategy={args.strategy})..."
            )
            summaries = evaluate_matrix(combos, suite, runner)
    else:
        combos = [
            ModelCombination(
                router_model=args.router_model,
                sql_model=args.sql_model,
                chat_model=args.chat_model,
            )
        ]
        print(
            f"🚀 Running quantitative evaluation for 1 model combination and {len(suite)} test cases..."
        )
        summaries = evaluate_matrix(combos, suite, runner)

    # Save JSON results
    serializable = []
    for s in summaries:
        data = asdict(s)
        serializable.append(data)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"✅ Raw metrics exported to {args.output}")

    # Generate Markdown Report
    report_md = format_markdown_report(summaries, suite)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"📊 Markdown Leaderboard exported to {args.report}")
    print("\n" + report_md)


if __name__ == "__main__":
    main()
