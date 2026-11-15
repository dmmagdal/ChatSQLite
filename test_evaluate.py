import unittest
from unittest.mock import Mock, patch

import evaluate
from evaluate import (
    BenchmarkCase,
    BenchmarkRunner,
    CombinationSummary,
    LLMClient,
    ModelCombination,
    compare_execution_results,
    compute_cqi,
    compute_faithfulness,
    compute_router_metrics,
    generate_combinations,
    identify_pareto_frontier,
    normalize_rows,
    search_combinations_with_restarts,
)


class RouterMetricsTests(unittest.TestCase):
    def test_perfect_match(self):
        metrics = compute_router_metrics(
            predicted=["fec_cm", "fec_cn"],
            gold=["fec_cm", "fec_cn"],
            latency_sec=0.12,
        )
        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.recall, 1.0)
        self.assertEqual(metrics.f1, 1.0)
        self.assertTrue(metrics.exact_match)
        self.assertFalse(metrics.under_selected)
        self.assertFalse(metrics.over_selected)

    def test_over_selection(self):
        metrics = compute_router_metrics(
            predicted=["fec_cm", "fec_cn", "fec_pas2"],
            gold=["fec_cm"],
        )
        self.assertAlmostEqual(metrics.precision, 1 / 3, places=3)
        self.assertEqual(metrics.recall, 1.0)
        self.assertTrue(metrics.over_selected)
        self.assertFalse(metrics.under_selected)
        self.assertFalse(metrics.exact_match)

    def test_under_selection(self):
        metrics = compute_router_metrics(
            predicted=["fec_cm"],
            gold=["fec_cm", "fec_cn"],
        )
        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.recall, 0.5)
        self.assertFalse(metrics.over_selected)
        self.assertTrue(metrics.under_selected)
        self.assertFalse(metrics.exact_match)

    def test_empty_sets(self):
        empty_both = compute_router_metrics([], [])
        self.assertEqual(empty_both.f1, 1.0)

        empty_pred = compute_router_metrics([], ["fec_cm"])
        self.assertEqual(empty_pred.f1, 0.0)
        self.assertTrue(empty_pred.under_selected)


class ExecutionComparisonTests(unittest.TestCase):
    def test_unordered_comparison_matches_regardless_of_permutation(self):
        pred = [("DEM", 100), ("REP", 200)]
        gold = [("REP", 200), ("DEM", 100)]
        gold_sql = "SELECT party, count(*) FROM table GROUP BY party;"
        self.assertTrue(compare_execution_results(pred, gold, gold_sql=gold_sql))

    def test_ordered_comparison_requires_exact_sequence(self):
        pred = [("DEM", 100), ("REP", 200)]
        gold = [("REP", 200), ("DEM", 100)]
        gold_sql = "SELECT party, count(*) FROM table ORDER BY count DESC;"
        self.assertFalse(compare_execution_results(pred, gold, gold_sql=gold_sql))

    def test_float_and_whitespace_normalization(self):
        pred = [(" California ", 45.12341)]
        gold = [("california", 45.12344)]
        self.assertTrue(compare_execution_results(pred, gold))

    def test_mismatched_row_counts_fail(self):
        pred = [("CA", 10)]
        gold = [("CA", 10), ("NY", 20)]
        self.assertFalse(compare_execution_results(pred, gold))


class FaithfulnessTests(unittest.TestCase):
    def test_all_numbers_grounded(self):
        rows = [("California", 29303), ("DC", 27203)]
        summary = "There are 29,303 committees in California and 27,203 in DC."
        score = compute_faithfulness(summary, rows)
        self.assertEqual(score, 1.0)

    def test_hallucinated_numbers_reduce_score(self):
        rows = [("California", 29303)]
        summary = "There are 29,303 in CA and an estimated 99,999 overall."
        score = compute_faithfulness(summary, rows)
        self.assertEqual(score, 0.5)

    def test_no_numbers_in_summary(self):
        rows = [("California", 29303)]
        summary = "Committees are heavily concentrated in California."
        score = compute_faithfulness(summary, rows)
        self.assertEqual(score, 1.0)


class DecisionAnalysisTests(unittest.TestCase):
    def test_cqi_calculation(self):
        summary = CombinationSummary(
            combination=ModelCombination("m1", "m2", "m3"),
            total_cases=10,
            router_precision=0.8,
            router_recall=0.8,
            router_f1=0.8,
            router_exact_match_rate=0.7,
            sql_ast_valid_rate=1.0,
            sql_ast_safe_rate=1.0,
            sql_pass_at_1_rate=0.7,
            sql_pass_at_k_rate=0.9,
            sql_execution_accuracy=0.85,
            healing_convergence_rate=0.66,
            mean_attempts=1.4,
            insight_json_conformance_rate=0.9,
            insight_column_grounding_rate=0.9,
            insight_faithfulness_rate=0.95,
            mean_latency_sec=1.5,
            p90_latency_sec=2.0,
        )
        cqi = compute_cqi(summary, norm_latency=0.2)
        self.assertGreater(cqi, 0.0)
        self.assertLessEqual(cqi, 1.0)

    def test_pareto_dominance(self):
        combo_a = CombinationSummary(
            combination=ModelCombination("r1", "s1", "c1"),
            total_cases=5,
            router_precision=1.0, router_recall=1.0, router_f1=1.0, router_exact_match_rate=1.0,
            sql_ast_valid_rate=1.0, sql_ast_safe_rate=1.0, sql_pass_at_1_rate=0.9, sql_pass_at_k_rate=0.9,
            sql_execution_accuracy=0.90,  # High accuracy
            healing_convergence_rate=1.0, mean_attempts=1.0,
            insight_json_conformance_rate=1.0, insight_column_grounding_rate=1.0, insight_faithfulness_rate=1.0,
            mean_latency_sec=1.0,         # Low latency
            p90_latency_sec=1.2,
        )
        combo_b = CombinationSummary(
            combination=ModelCombination("r2", "s2", "c2"),
            total_cases=5,
            router_precision=1.0, router_recall=1.0, router_f1=1.0, router_exact_match_rate=1.0,
            sql_ast_valid_rate=1.0, sql_ast_safe_rate=1.0, sql_pass_at_1_rate=0.6, sql_pass_at_k_rate=0.6,
            sql_execution_accuracy=0.60,  # Lower accuracy
            healing_convergence_rate=0.5, mean_attempts=2.0,
            insight_json_conformance_rate=1.0, insight_column_grounding_rate=1.0, insight_faithfulness_rate=1.0,
            mean_latency_sec=3.0,         # Higher latency -> strictly dominated by combo_a
            p90_latency_sec=3.5,
        )
        frontier = identify_pareto_frontier([combo_a, combo_b])
        self.assertTrue(combo_a.is_pareto_optimal)
        self.assertFalse(combo_b.is_pareto_optimal)


class SelfHealingMockTests(unittest.TestCase):
    def test_self_healing_recovers_after_initial_error(self):
        call_count = 0

        def dynamic_mock(model, system_prompt, user_prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Syntax error on attempt 1
                return "```sql\nSELECT INVALID FROM\n```"
            # Fixed query on attempt 2
            return "```sql\nSELECT count(*) FROM aux_presidents;\n```"

        client = LLMClient(mock=True, mock_handler=dynamic_mock)
        runner = BenchmarkRunner(client=client, max_retries=3)

        case = BenchmarkCase(
            id="test_01",
            query="Count presidents",
            difficulty="simple",
            gold_tables=["aux_presidents"],
            gold_sql="SELECT count(*) FROM aux_presidents;",
        )

        gold_rows, _, _ = runner.execute_query(case.gold_sql)
        sql_res = runner.eval_sql(
            model="mock_model",
            case=case,
            schema_str="",
            metadata_str="",
            gold_rows=gold_rows,
        )

        self.assertFalse(sql_res.pass_at_1)
        self.assertTrue(sql_res.pass_at_k)
        self.assertTrue(sql_res.healing_converged)
        self.assertEqual(sql_res.attempts_used, 2)
        self.assertTrue(sql_res.execution_accuracy)


class CombinationGenerationTests(unittest.TestCase):
    def test_cartesian_product_from_dict(self):
        matrix_dict = {
            "router_model": ["r1", "r2"],
            "sql_model": ["s1", "s2", "s3"],
            "chat_model": ["c1", "c2"],
        }
        combos = generate_combinations(matrix_dict, strategy="grid")
        self.assertEqual(len(combos), 2 * 3 * 2)
        self.assertEqual(combos[0].router_model, "r1")
        self.assertEqual(combos[0].sql_model, "s1")
        self.assertEqual(combos[0].chat_model, "c1")

    def test_explicit_list_input(self):
        explicit = [
            {"router_model": "r1", "sql_model": "s1", "chat_model": "c1"},
            {"router_model": "r2", "sql_model": "s2", "chat_model": "c2"},
        ]
        combos = generate_combinations(explicit)
        self.assertEqual(len(combos), 2)
        self.assertEqual(combos[1].router_model, "r2")

    def test_seeded_subsampling(self):
        matrix_dict = {
            "router_model": ["r1", "r2", "r3"],
            "sql_model": ["s1", "s2", "s3"],
            "chat_model": ["c1", "c2", "c3"],
        }
        # 3 * 3 * 3 = 27 combos
        sample_a = generate_combinations(
            matrix_dict, strategy="sample", sample_size=5, seed=123
        )
        sample_b = generate_combinations(
            matrix_dict, strategy="sample", sample_size=5, seed=123
        )
        self.assertEqual(len(sample_a), 5)
        self.assertEqual(
            [c.to_key() for c in sample_a], [c.to_key() for c in sample_b]
        )

    def test_search_combinations_with_restarts(self):
        matrix_dict = {
            "router_model": ["r1", "r2"],
            "sql_model": ["s1", "s2"],
            "chat_model": ["c1", "c2"],
        }
        client = LLMClient(mock=True)
        runner = BenchmarkRunner(client=client)
        suite = [
            BenchmarkCase(
                id="test_01",
                query="Count presidents",
                difficulty="simple",
                gold_tables=["aux_presidents"],
                gold_sql="SELECT count(*) FROM aux_presidents;",
            )
        ]
        summaries = search_combinations_with_restarts(
            matrix_data=matrix_dict,
            suite=suite,
            runner=runner,
            restarts=2,
            seed=42,
        )
        self.assertGreater(len(summaries), 0)
        self.assertTrue(any(s.is_pareto_optimal for s in summaries))


if __name__ == "__main__":
    unittest.main()
