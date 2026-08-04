# pipeline.py
# Implement the chat-to-SQL pipeline.
# Windows/MacOS/Linux
# Python 3.10


import json
import re
import sqlite3
from typing import List, Union, Generator, Iterator

import pandas as pd
import plotly.express as px
import plotly.io as pio
from pydantic import BaseModel, Field
import requests
import sqlglot
from sqlglot import exp

from helper import *


class Pipeline:
	class Valves(BaseModel):
		OLLAMA_BASE_URL: str = Field(
			default="http://localhost:11434",
			description="Base URL for the Ollama API.",
		)
		MODEL_NAME: str = Field(
			default="qwen2.5-coder:7b",
			description="Name of the model to use with the Ollama API.",
		)
		DB_PATH: str = Field(
			default="fec_data.db",
			description="Path to the SQLite database file.",
		)
		MAX_RETRIES: int = Field(
			default=3,
			description="Maximum number of retries for API requests.",
		)
		

	def __init__(self):
		self.name = "SQLite Self-Healing Multi-Agent Analytics"
		self.valves = self.Valves()


	async def on_startup(self) -> None:
		print(f"Starting up {self.name} and connecting to {self.valves.DB_PATH}...")


	async def on_shutdown(self) -> None:
		print(f"Stopped {self.name}.")


	def pipe(
		self, 
		user_message: str, 
		model_id: str, 
		messages: List[dict], 
		body: dict
	) -> Union[str, Generator, Iterator]:
		"""Main entry point called by Open WebUI."""
		try:
			# Get schema context
			# schema_str = get_db_schema()
			# metadata_schema_str = get_metadata_db_schema()
			# metadata_tables = get_metadata()

			selected_tables = self._select_relevant_tables(user_message)
			print(f"Selected tables for user request: {selected_tables}")
			print(f"all tables: {get_tables()}")

			filtered_schema = get_db_schema(selected_tables)
			filtered_metadata = get_metadata(selected_tables)

			# print(f"Database schema retrieved:\n{schema_str}")
			print(f"Filtered database schema:\n{filtered_schema}")
			print(f"Filtered metadata:\n{filtered_metadata}")

			# Agent 1 & 2: Generate and heal SQL
			# sql_query, df = self._agent_generate_and_heal_sql(user_message, schema_str, metadata_tables)
			sql_query, df = self._agent_generate_and_heal_sql(user_message, filtered_schema, filtered_metadata)

			# Agent 3: Generate Plotly charts and markdown reports
			return self._agent_generate_visualizations_and_reports(user_message, sql_query, df)

		except Exception as e:
			return f"❌ **Pipeline Execution Error:** {str(e)}"



	###################################################################
	# Internal utilities.
	###################################################################

	def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
		"""
		Call Ollama server to for chat completion requests.
		@param system_prompt: The system prompt for the Ollama API.
		@param user_prompt: The user prompt for the Ollama API.
		@return: The response from the Ollama API as a string.
		"""
		# Build out the payload to the Ollama API.
		payload = {
			"model": self.valves.MODEL_NAME,
			"messages": [
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			],
			"stream": False,
			"options": {
				"temperature": 0.1#1.0
			}
		}

		# Send the request to the Ollama API.
		response = requests.post(
			f"{self.valves.OLLAMA_BASE_URL}/api/chat/",
			json=payload,
			timeout=300,
		)
		if response.status_code != 200:
			print(f"Error response from Ollama API: {response.text}")

		# Raise an exception if the request failed and return the 
		# content of the response.
		response.raise_for_status()
		return response.json()["message"]["content"].strip()


	def _execute_read_only(self, query: str, max_limit: int = 1_000) -> pd.DataFrame:
		"""
		Execute SQL statement safely using read-only URI and PRAGMA settings.
		@param query: The SQL query string to execute.
		@return: A pandas DataFrame containing the query results.
		"""
		# Establish a read-only connection to the SQLite database.
		conn = sqlite3.connect(f"file:{self.valves.DB_PATH}?mode=ro", uri=True)
		# conn.execute("PRAGMA read_only = TRUE;")
		conn.execute("PRAGMA query_only = TRUE;")
		# conn.execute("PRAGMA foreign_keys = ON;")
		# conn.execute("PRAGMA journal_mode = WAL;")
		# conn.execute("PRAGMA synchronous = NORMAL;")

		# Try running the query and ensure the connection is closed afterwards.
		try:
			cursor = conn.execute(query)
			rows = cursor.fetchmany(max_limit)
			columns = [description[0] for description in cursor.description]

			return pd.DataFrame(rows, columns=columns)
		finally:
			conn.close()


	def _clean_sql(self, llm_response: str) -> str:
		"""
		Clean and normalize the given SQL query.
		@param llm_response: The raw response from the LLM containing 
			the SQL query.
		@return: The cleaned and normalized SQL query string.
		"""
		match = re.search(r"```(?:sql)?\s*(.*?)\s*```", llm_response, re.DOTALL | re.IGNORECASE)
		if match:
			return match.group(1).strip()
		return llm_response.strip().rstrip(";") + ";"


	###################################################################
	# Agentic stages.
	###################################################################

	def _select_relevant_tables(self, user_prompt: str) -> List[str]:
		"""
		Select relevant tables based on the user's prompt.
		@param user_prompt: The user's natural language request.
		@return: A list of relevant table names.
		"""
		all_tables = get_tables()
		all_metadata = get_metadata()

		system_prompt = f"""You are a Database Router.
Given a user's analytical question, identify which database tables from the list below are necessary to answer it.

{all_metadata}

Respond ONLY with a JSON array of string table names. Example: ["fec_cm", "fec_pas2"]"""

		raw_response = self._call_ollama(system_prompt, f"User Request: {user_prompt}")
		
		try:
			match = re.search(r"\[.*?\]", raw_response, re.DOTALL)
			if match:
				selected_tables = json.loads(match.group(0))
				# Filter to ensure only valid tables in the DB are returned
				valid_tables = [t for t in selected_tables if t in all_tables]
				if valid_tables:
					return valid_tables
	
		except Exception as e:
			print(f"Table routing fallback triggered: {e}")

		# Fallback to returning all tables if routing fails or yields empty results.
		return all_tables


	def _agent_generate_and_heal_sql(self, user_prompt: str, schema_str: str, metadata_schema_str: str) -> str:
		"""
		Generate SQL and handle self healing retries if syntax errors 
			or schema mismatches are returned by SQLite.
		@param user_prompt: The user's natural language request.
		@param schema_str: The database schema as a string.
		@return: A valid SQL query string that adheres to the schema 
			and rules.
		"""
		# NOTE
		# It is important that any additional context or schema information is included in the system prompt to ensure 
		# the LLM is aware of the relationship between tables and their columns in order to generate valid SQL queries.
		system_prompt = f"""You are a SQLite Data Analyst Agent.
Your job is to generate a SINGLE valid SQLite SELECT query based on the user's request.

CRITICAL RULES:
1. Output ONLY the raw SQL query inside a ```sql ``` code block.
2. NEVER issue INSERT, UPDATE, DELETE, DROP, ALTER, or PRAGMA statements.
3. Keep results limited to 100 rows unless explicitly performing aggregated SUM/COUNT queries.

TABLE METADATA:
{metadata_schema_str}

DATABASE SCHEMA:
{schema_str}
"""
		current_prompt = f"User Request: {user_prompt}"
		error_history = []

		for attempt in range(1, self.valves.MAX_RETRIES + 1):
			# Pass to ollama model to generate SQL query and clean the 
			# response.
			raw_response = self._call_ollama(system_prompt, current_prompt)
			sql_query = self._clean_sql(raw_response)
			print(sql_query)

			# Security Guardrail Check.
			if not is_read_only_query(sql_query):
				error_msg = "Security Exception: Query failed read-only AST check."
				current_prompt = f"Previous SQL: {sql_query}\nError: {error_msg}\nPlease rewrite to use strictly SELECT statements."
				print(current_prompt)
				continue

			# Execution Check.
			try:
				df = self._execute_read_only(sql_query)
				print(df.head())
				return sql_query, df
			except Exception as e:
				error_msg = str(e)
				error_history.append(f"Attempt {attempt} Error: {error_msg}")
				current_prompt = (
					f"The SQL query you provided failed with the following error:\n"
					f"SQL: {sql_query}\n"
					f"SQLite Error: {error_msg}\n\n"
					f"Please correct the query syntax or column references and try again."
				)
				print(current_prompt)

		raise RuntimeError(f"Self-healing SQL failed after {self.valves.MAX_RETRIES} attempts:\n" + "\n".join(error_history))


	def _agent_generate_visualizations_and_reports(self, user_prompt: str, sql_query: str, df: pd.DataFrame) -> str:
		if df.empty: 
			return f"### Query Executed\n```sql\n{sql_query}\n```\n\n**Result:** No matching records found."

		# Prepare markdown data,
		head_limit = 15
		display_df = df.head(head_limit)
		markdown_table = display_df.to_markdown(index=False)
		table_note = f"\n\n*Showing top {len(display_df)} of {len(df)} total rows.*" if len(df) > head_limit else ""

		# Have the LLM generate an executive summary of the dataset.
		summary_prompt = f"""Analyze the dataset returned for the question: "{user_prompt}"
Data Sample (Top 5 Rows):
{df.head(5).to_json(orient='records')}

Task:
1. Write a concise 2-3 sentence executive summary.
2. Recommend a chart type ('bar', 'line', 'pie', 'scatter', or 'none') to visualize this result.
Respond in JSON format with keys: "summary", "chart_type", "x_col", "y_col", "title".
"""
		summary_raw = self._call_ollama("You are a Data Analyst. Return valid JSON only.", summary_prompt)

		try:
			json_match = re.search(r"\{.*\}", summary_raw, re.DOTALL)
			chart_config = json.loads(json_match.group(0)) if json_match else {}
		except Exception:
			chart_config = {"summary": "Data retrieved successfully.", "chart_type": "none"}

		chart_html = ""
		chart_type = chart_config.get("chart_type", "none")
		x_col = chart_config.get("x_col")
		y_col = chart_config.get("y_col")
		chart_title = chart_config.get("title", f"{y_col} by {x_col}" if x_col and y_col else "Data Visualization")

		# Validate that selected columns exist in the DataFrame
		if chart_type != "none" and x_col in df.columns and y_col in df.columns:
			plot_df = df.head(20)  # Restrict plot points to keep UI responsive

			if chart_type == "bar":
				fig = px.bar(plot_df, x=x_col, y=y_col, title=chart_title, template="plotly_dark")
			elif chart_type == "line":
				fig = px.line(plot_df, x=x_col, y=y_col, title=chart_title, template="plotly_dark")
			elif chart_type == "pie":
				fig = px.pie(plot_df, names=x_col, values=y_col, title=chart_title, template="plotly_dark")
			elif chart_type == "scatter":
				fig = px.scatter(plot_df, x=x_col, y=y_col, title=chart_title, template="plotly_dark")
			else:
				fig = None

			if fig:
				# Custom dark styling to blend seamlessly into Open WebUI's dark mode
				fig.update_layout(
					paper_bgcolor="rgba(24, 24, 37, 1)",
					plot_bgcolor="rgba(24, 24, 37, 1)",
					font_color="#cdd6f4",
					margin=dict(l=40, r=40, t=50, b=40)
				)

				# Export to an HTML snippet. 
				# full_html=False creates a <div> block without <html>/<head> tags.
				# include_plotlyjs='cdn' fetches the JS engine externally so your response payload stays tiny.
				raw_plotly_html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
				
				# Wrap in markdown ```html block so Open WebUI renders it automatically
				chart_html = f"\n```html\n{raw_plotly_html}\n```\n"

		return f"""### 📊 Executive Summary
{chart_config.get('summary', '')}

{chart_html}

### 📋 Data Table
{markdown_table}{table_note}

<details>
<summary><b>🔍 View Generated SQL Query</b></summary>

```sql
{sql_query}
```
</details>
"""

