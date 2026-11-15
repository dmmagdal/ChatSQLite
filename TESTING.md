# Model Evaluation & Quantitative Benchmarking Guide (`TESTING.md`)

This document defines the quantitative evaluation framework for the **ChatSQLite** project. It outlines the mathematical formulations, benchmark schema, evaluation workflow, and decision-analysis algorithms used to measure and compare different LLM model combinations across the pipeline.

---

## 1. Executive Summary & Objective

ChatSQLite is a multi-agent system that translates natural language questions into secure SQLite queries, executes them against FEC election campaign datasets, and visualizes analytical insights.

Because different open-weight and proprietary models possess distinct strengths (e.g., small fast coders vs. reasoning models), no single model is optimal for all tasks. This evaluation framework provides an **empirical, reproducible, and quantitative method** to:
1. Benchmark individual agent roles in isolation (Router, SQL Generator, Insights/Visualizer).
2. Measure end-to-end multi-agent pipeline performance.
3. Quantify self-healing recovery efficiency under SQLite syntax and execution errors.
4. Rank model combinations using **Multi-Criteria Decision Analysis (MCDA)** and **Pareto Efficiency Frontiers** (Accuracy vs. Latency / Resource Cost).

---

## 2. Multi-Agent Role Architecture

The pipeline partitions cognitive responsibilities across three distinct sub-tasks:

```
                  ┌───────────────────────────────┐
                  │ User Natural Language Request │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │ Stage 1: Table Router                                  │
      │ (_select_relevant_tables in pipeline.py)               │
      │ Output: JSON array of required tables                  │
      └───────────────────────────┬────────────────────────────┘
                                  │ Filtered Schema & Metadata
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │ Stage 2: Text-to-SQL & Self-Healing Agent              │
      │ (workflow.py / pipeline.py)                            │
      │ Generates SQLite query -> AST Check -> Execute         │
      │ Feedback loop: Retries with SQLite error up to K times │
      └───────────────────────────┬────────────────────────────┘
                                  │ SQL Result Set (DataFrame / Tuples)
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │ Stage 3: Data Analyst & Visualization Agent            │
      │ (_agent_generate_visualizations_and_reports)           │
      │ Output: Executive summary + Chart configuration        │
      └────────────────────────────────────────────────────────┘
```

---

## 3. Benchmark Dataset Specification (`eval_suite.json`)

Evaluation is driven by a curated test suite $\mathcal{D} = \{ (q_i, \mathcal{T}^*_i, S^*_i, V^*_i) \}_{i=1}^N$ grounded in the actual schema and data of `fec_data.db` and `fec_metadata.db`.

### Benchmark Schema

```json
[
  {
    "id": "fec_03",
    "query": "What are the top 5 states with the most committees registered in the committee master table?",
    "difficulty": "moderate",
    "gold_tables": ["fec_cm"],
    "gold_sql": "SELECT CMTE_ST, count(*) AS committee_count FROM fec_cm WHERE CMTE_ST IS NOT NULL GROUP BY CMTE_ST ORDER BY committee_count DESC LIMIT 5;",
    "expected_chart": {
      "chart_type": "bar",
      "x_col": "CMTE_ST",
      "y_col": "committee_count",
      "title": "Top 5 States with Most Committees"
    }
  }
]
```

### Difficulty Stratification
- **Simple:** Single-table lookups, simple filters, direct aggregations (`aux_presidents`, `fec_cm`).
- **Moderate:** Aggregations with `GROUP BY`, `ORDER BY`, multi-column projections, and two-table joins (`fec_cn` joined with `fec_cm`).
- **Complex:** Three-table relationships, compound conditional logic, nested queries, or complex filtering.
- **Security / Guardrail:** Adversarial prompts containing prompt injection or DDL/DML statements (`DROP TABLE`, `DELETE`, `UPDATE`) to verify AST read-only safety filters.

---

## 4. Quantitative Metrics

### A. Stage 1: Table Router Performance
Measures the router's ability to select necessary tables without polluting the prompt context:

$$\text{Precision} = \frac{|\hat{\mathcal{T}} \cap \mathcal{T}^*|}{|\hat{\mathcal{T}}|}, \quad \text{Recall} = \frac{|\hat{\mathcal{T}} \cap \mathcal{T}^*|}{|\mathcal{T}^*|}, \quad F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

- **Exact Match Rate:** $\mathbb{I}(\hat{\mathcal{T}} \equiv \mathcal{T}^*)$
- **Under-Selection Rate:** Percentage of queries where $\text{Recall} < 1.0$ (fatal for downstream joins).
- **Over-Selection Rate:** Percentage of queries where $\text{Precision} < 1.0$ (context bloat).

### B. Stage 2: Text-to-SQL & Self-Healing
1. **AST Parse Rate (VSR):** Percentage of generated queries that produce a valid Abstract Syntax Tree via `sqlglot.parse(..., dialect="sqlite")`.
2. **AST Read-Only Safety Compliance (RSC):** Percentage of queries passing the `is_read_only_query()` security check.
3. **Pass@1:** Execution success on the zero-shot initial attempt.
4. **Pass@K:** Execution success within up to $K$ self-healing retries ($K=3$ default).
5. **Execution Accuracy ($\text{EX}$):**
   Comparing execution results $\hat{R}_i$ against gold execution $R^*_i$:
   $$\text{EX}(q_i) = \begin{cases} 1 & \text{if } \text{normalize}(\hat{R}_i) \equiv \text{normalize}(R^*_i) \\ 0 & \text{otherwise} \end{cases}$$
   - **Normalization Rules:**
     - Float rounding (tolerance to 4 decimal places).
     - String stripping and case-insensitive comparison.
     - Row-order invariant multiset comparison unless the gold query explicitly specifies `ORDER BY`.
6. **Healing Convergence Rate (HCR):**
   Measures how effectively the model fixes its own errors when provided SQLite runtime error feedback:
   $$\text{HCR} = \frac{|\text{Queries resolved on attempts } 2..K|}{|\text{Queries that failed attempt } 1|}$$
7. **Mean Repair Attempts ($\overline{A}$):** Average attempts required to reach a successful execution.

### C. Stage 3: Insights & Visualizations
1. **JSON Conformance Rate (JCR):** Percentage of responses adhering strictly to the schema:
   `{"summary": str, "chart_type": str, "x_col": str, "y_col": str, "title": str}`
2. **Column Grounding Accuracy (CGA):** Percentage of recommended chart dimensions (`x_col`, `y_col`) that physically exist in the executed query DataFrame columns.
3. **Numerical Faithfulness Score:** Measures factual consistency by calculating the proportion of numbers cited in the summary text that exist in the underlying SQL result set:
   $$\text{Faithfulness} = \frac{|\text{Numbers cited in summary matching result set values}|}{|\text{Total numbers cited in summary}|}$$

### D. Latency & System Efficiency
- **Mean & P90 Latency (s):** Tracked per stage and end-to-end.
- **Min-Max Normalized Latency ($\widetilde{\text{Lat}}$):** Used for composite scoring across tested combinations:
  $$\widetilde{\text{Lat}}(C) = \frac{\text{Lat}(C) - \text{Lat}_{\min}}{\text{Lat}_{\max} - \text{Lat}_{\min}}$$

---

## 5. Multi-Criteria Decision Analysis (MCDA)

### Composite Quality Index (CQI)
To rank model combinations $C = (M_{\text{router}}, M_{\text{sql}}, M_{\text{chat}})$, we compute a weighted utility score $S(C) \in [0, 1]$:

$$S(C) = w_{\text{ex}} \cdot \text{EX} + w_{\text{heal}} \cdot \text{HCR} + w_{\text{route}} \cdot F_1 + w_{\text{vis}} \cdot \text{VisScore} - w_{\text{lat}} \cdot \widetilde{\text{Lat}}$$

Where:
- $\text{VisScore} = 0.5 \cdot \text{JCR} + 0.5 \cdot \text{CGA}$
- Standard Default Weights:
  - $w_{\text{ex}} = 0.40$ (Execution Accuracy)
  - $w_{\text{heal}} = 0.20$ (Self-Healing Recovery)
  - $w_{\text{route}} = 0.15$ (Table Router $F_1$)
  - $w_{\text{vis}} = 0.10$ (Visualization Quality)
  - $w_{\text{lat}} = 0.15$ (Latency Penalty)

### Pareto Optimality Frontier
A model combination $A$ dominates combination $B$ if:
$$\text{EX}(A) \ge \text{EX}(B) \quad \text{and} \quad \text{Lat}(A) \le \text{Lat}(B)$$
with at least one strict inequality.

The **Pareto Frontier** identifies optimal trade-offs:
- **Best Lightweight / Edge Combination:** Lowest latency and VRAM footprint with acceptable accuracy.
- **Best Maximum Accuracy Combination:** Highest execution accuracy where latency is secondary.

---

## 6. How to Run the Evaluation Harness

### Environment Setup
Always activate the project conda environment before running benchmarks:
```bash
conda activate chat-sqlite
```

Verify your environment dependencies:
```bash
python -c "import langchain_ollama, langgraph, sqlglot, pandas, plotly; print('Ready!')"
```

### Running Unit Tests for the Evaluation Framework
```bash
python -m unittest test_evaluate.py
```

### Evaluating a Single Model Combination
```bash
python evaluate.py \
  --suite eval_suite.json \
  --router-model qwen2.5-coder:1.5b \
  --sql-model qwen2.5-coder:7b \
  --chat-model lfm2.5-thinking:1.2b \
  --output results.json \
  --report REPORT.md
```

### Evaluating a Matrix of Candidate Models (`eval_matrix.json`)

Instead of writing out every permutation by hand, define candidate models per role as lists inside a dictionary:

```json
{
  "router_model": [
    "qwen2.5-coder:1.5b",
    "phi4-mini-reasoning:3.8b"
  ],
  "sql_model": [
    "qwen2.5-coder:1.5b",
    "qwen2.5-coder:7b"
  ],
  "chat_model": [
    "lfm2.5-thinking:1.2b",
    "phi4-mini-reasoning:3.8b"
  ]
}
```

The evaluation runner uses `itertools.product` to automatically generate the Cartesian product of combinations.

#### Strategy 1: Full Cartesian Grid Search (`--strategy grid`)
Evaluates all $|\mathcal{R}| \times |\mathcal{S}| \times |\mathcal{C}|$ combinations (e.g., $2 \times 2 \times 2 = 8$ combos):
```bash
python evaluate.py \
  --suite eval_suite.json \
  --matrix eval_matrix.json \
  --strategy grid \
  --output matrix_results.json \
  --report MATRIX_REPORT.md
```

#### Strategy 2: Seeded Random Subsampling (`--strategy sample`)
Randomly draws $N$ combinations using a deterministic seed for fast, repeatable checks across large model pools:
```bash
python evaluate.py \
  --suite eval_suite.json \
  --matrix eval_matrix.json \
  --strategy sample \
  --sample-size 4 \
  --seed 42 \
  --output sample_results.json \
  --report SAMPLE_REPORT.md
```

#### Strategy 3: Guided Space Exploration with Restarts (`--strategy explore`)
Uses **Coordinate Descent / Local Greedy Search with Random Restarts** to discover the best combination without evaluating all permutations:
- Starts from a random candidate combination $(r_0, s_0, c_0)$ determined by `--seed`.
- Optimizes one coordinate at a time (Router $\to$ SQL $\to$ Chat), greedily moving to the neighbor with the highest Composite Quality Index ($\text{CQI}$).
- Upon reaching a local optimum, initiates a new restart from a different seeded starting point.
- Avoids redundant runs via an evaluation cache.
- Re-normalizes metrics and computes the Pareto frontier across all explored combinations.

```bash
python evaluate.py \
  --suite eval_suite.json \
  --matrix eval_matrix.json \
  --strategy explore \
  --restarts 3 \
  --seed 42 \
  --output explore_results.json \
  --report EXPLORE_REPORT.md
```

### Running in Fast Offline / Mock Mode (CI / Verification)
To test the evaluation pipeline without querying a live Ollama server:
```bash
python evaluate.py --mock --suite eval_suite.json --matrix eval_matrix.json --strategy grid
```

---

## 7. Sample Benchmark Output & Leaderboard

The benchmark exports both raw telemetry (`results.json`) and a formatted Markdown report (`REPORT.md`):

```markdown
# ChatSQLite Model Evaluation Benchmark Report

## 🏆 Best Configuration: Champion Overview

- **Top Model Combination:** `phi4-mini-reasoning:3.8b` (Router) + `qwen2.5-coder:7b` (SQL) + `lfm2.5-thinking:1.2b` (Chat)
- **Composite Quality Index (CQI):** **0.8420** / 1.0000
- **Detailed Score Breakdown:**
  - **SQL Execution Accuracy (EX):** 87.5%
  - **Pass@1 (Zero-Shot SQL):** 75.0%
  - **Pass@K (After Self-Healing):** 87.5% (Healing Convergence: 50.0%, Mean Attempts: 1.35)
  - **Router Performance (F1):** 0.940 (Precision: 92.0%, Recall: 96.0%, Exact Match: 80.0%)
  - **Visualization & Insights:** JSON Conformance: 100.0%, Column Grounded: 100.0%, Faithfulness: 95.0%
  - **Latency:** Mean: 2.15s, P90: 2.80s

- **⚡ Low-Latency Pareto Alternative:** `qwen2.5-coder:1.5b` / `qwen2.5-coder:1.5b` / `lfm2.5-thinking:1.2b`
  - Mean Latency: **0.85s** | EX Acc: **75.0%** | CQI: **0.7150**

---

## Executive Leaderboard (Ranked by Composite Quality Index)

| Rank | Combination (Router / SQL / Chat) | CQI Score | EX Acc | Pass@1 | Pass@K | HCR | Router F1 | Latency (Mean) | Pareto Optimal? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `phi4-mini:3.8b` / `qwen2.5-coder:7b` / `lfm2.5-thinking:1.2b` | **0.8420** | 87.5% | 75.0% | 87.5% | 50.0% | 0.94 | 2.15s | 🌟 **Yes** |
| 2 | `qwen2.5-coder:1.5b` / `qwen2.5-coder:1.5b` / `lfm2.5-thinking:1.2b` | **0.7150** | 75.0% | 62.5% | 75.0% | 33.3% | 0.88 | 0.85s | 🌟 **Yes** |
| 3 | `qwen2.5-coder:1.5b` / `qwen2.5-coder:7b` / `lfm2.5-thinking:1.2b` | **0.6980** | 75.0% | 62.5% | 75.0% | 33.3% | 0.88 | 1.95s | No |
```

---

## 8. Extending the Benchmark Suite

To add new queries to `eval_suite.json`:
1. Pick a natural language query covering candidate tables (`fec_cn`, `fec_cm`, `fec_pas2`, `aux_presidents`, `aux_congress`).
2. Write and verify the canonical SQLite query against `fec_data.db`.
3. Add the entry to `eval_suite.json` with the appropriate `difficulty` tag and `expected_chart` structure.
4. Re-run `python evaluate.py` to update the benchmark scores.
