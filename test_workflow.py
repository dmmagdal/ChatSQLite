import sqlite3
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import workflow


class ExtractSqlTests(unittest.TestCase):
	def test_longest_candidate_ignores_thinking_and_other_code(self):
		text = """<think>
```sql
SELECT * FROM ignored_thinking_table;
```
</think>
```python
SELECT = "not SQL"
```
```SQL
SELECT 1;
```
~~~sqlite
SELECT name
FROM people
WHERE name = 'A;B';
~~~
"""
		self.assertEqual(
			workflow.extract_sql(text),
			"SELECT name\nFROM people\nWHERE name = 'A;B';",
		)

	def test_extracts_unfenced_multiline_sql_and_discards_prose(self):
		text = "Here is the query:\nSELECT 'a;b' AS value\nFROM people;\nHope this helps!"
		self.assertEqual(workflow.extract_sql(text), "SELECT 'a;b' AS value\nFROM people;")
		self.assertEqual(workflow.extract_sql("Query: SELECT 1; Done."), "SELECT 1;")

	def test_unlabelled_and_unclosed_fences(self):
		self.assertEqual(workflow.extract_sql("```\nSELECT 1\n```"), "SELECT 1")
		self.assertEqual(workflow.extract_sql("```\n-- comment\nSELECT 1\n```"), "-- comment\nSELECT 1")
		self.assertEqual(workflow.extract_sql("```sql\nSELECT * FROM"), "SELECT * FROM")

	def test_ignores_thinking_and_invalid_responses(self):
		for text in (
			"<think>SELECT 1;</think>\nNo query.",
			"<think>SELECT 1;",
			"SELECT 'unterminated",
			"SELECT count(*)\nFROM",
		):
			with self.subTest(text=text):
				with self.assertRaises(ValueError):
					workflow.extract_sql(text)

	def test_preserves_think_tags_in_sql_literals(self):
		self.assertEqual(workflow.extract_sql("```sql\nSELECT '<think>literal</think>';\n```"),
			"SELECT '<think>literal</think>';")


class WorkflowTests(unittest.TestCase):
	def test_stops_after_six_failures_with_error_answer(self):
		sql = Mock()
		sql.invoke.return_value = SimpleNamespace(content="```sql\nSELECT FROM;\n```")
		chat = Mock()

		with patch.object(workflow, "get_db_schema", return_value=""), \
			patch.object(workflow, "get_metadata_db_schema", return_value=""), \
			patch.object(workflow, "get_connection", side_effect=lambda: sqlite3.connect(":memory:")), \
			patch.object(workflow, "sql", sql), \
			patch.object(workflow, "chat", chat):
			result = workflow.app.invoke({"user_query": "test"})

		self.assertEqual(result["failure_count"], 6)
		self.assertEqual(sql.invoke.call_count, 6)
		self.assertIn(result["sql_error"], result["final_answer"])
		self.assertIn("6", result["final_answer"])
		chat.invoke.assert_not_called()

	def test_success_after_retry_generates_insights(self):
		sql = Mock()
		sql.invoke.side_effect = [
			SimpleNamespace(content="INVALID SQL"),
			SimpleNamespace(content="SELECT 1"),
		]
		chat = Mock()
		chat.invoke.return_value = SimpleNamespace(content="One result")

		with patch.object(workflow, "get_db_schema", return_value=""), \
			patch.object(workflow, "get_metadata_db_schema", return_value=""), \
			patch.object(workflow, "get_connection", side_effect=lambda: sqlite3.connect(":memory:")), \
			patch.object(workflow, "sql", sql), \
			patch.object(workflow, "chat", chat):
			result = workflow.app.invoke({"user_query": "test"})

		self.assertEqual(result["failure_count"], 1)
		self.assertEqual(result["sql_error"], "")
		self.assertEqual(result["final_answer"], "One result")
		self.assertEqual(sql.invoke.call_count, 2)

	def test_missing_sql_retries_without_executing_empty_query(self):
		sql = Mock()
		sql.invoke.side_effect = [
			SimpleNamespace(content="<think>SELECT 99;</think>\nNo query."),
			SimpleNamespace(content="SELECT 1;"),
		]
		chat = Mock()
		chat.invoke.return_value = SimpleNamespace(content="One result")
		connection = Mock(side_effect=lambda: sqlite3.connect(":memory:"))

		with patch.object(workflow, "get_db_schema", return_value=""), \
			patch.object(workflow, "get_metadata_db_schema", return_value=""), \
			patch.object(workflow, "get_connection", connection), \
			patch.object(workflow, "sql", sql), \
			patch.object(workflow, "chat", chat):
			result = workflow.app.invoke({"user_query": "test"})

		self.assertEqual(result["failure_count"], 1)
		self.assertEqual(result["final_answer"], "One result")
		self.assertEqual(connection.call_count, 1)
		self.assertIn("No SQL statement found", sql.invoke.call_args_list[1].args[0][0].content)

	def test_fenced_query_executes_without_markdown(self):
		sql = Mock()
		sql.invoke.return_value = SimpleNamespace(content="<think>reasoning</think>\n```sql\nSELECT 1;\n```")
		chat = Mock()
		chat.invoke.return_value = SimpleNamespace(content="One result")

		with patch.object(workflow, "get_db_schema", return_value=""), \
			patch.object(workflow, "get_metadata_db_schema", return_value=""), \
			patch.object(workflow, "get_connection", side_effect=lambda: sqlite3.connect(":memory:")), \
			patch.object(workflow, "sql", sql), \
			patch.object(workflow, "chat", chat):
			result = workflow.app.invoke({"user_query": "test"})

		self.assertEqual(result["sql_query"], "SELECT 1;")
		self.assertEqual(result["query_result"], [(1,)])
		self.assertEqual(result["final_answer"], "One result")
		self.assertEqual(sql.invoke.call_count, 1)


if __name__ == "__main__":
	unittest.main()
