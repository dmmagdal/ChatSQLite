# pipeline.py
# Implement the chat-to-SQL pipeline.
# Windows/MacOS/Linux
# Python 3.10


import re
import sqlite3
from typing import List, Union, Generator, Iterator

import pandas as pd
from pydantic import BaseModel, Field
import requests
import sqlglot
from sqlglot import exp

from helper import get_db_schema, is_read_only_query


class Pipeline:
	class Valves(BaseModel):
		OLLAMA_BASE_URL: str = Field(
			default="",
			description="Base URL for the Ollama API.",
		)
		MODEL_NAME: str = Field(
			default="",
			description="Name of the model to use with the Ollama API.",
		)
		DB_PATH: str = Field(
			default="",
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
				"temperature": 1.0
			}
		}

		# Send the request to the Ollama API.
		response = requests.post(
			f"{self.valves.OLLAMA_BASE_URL}/v1/chat",
			json=payload,
			timeout=120,
		)

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
		conn.execute("PRAGMA read_only = TRUE;")
		# conn.execute("PRAGMA query_only = TRUE;")
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

	def _agent_generate_and_heal_sql(self, user_prompt: str, schema_str: str) -> str:
		"""
		Generate SQL and handle self healing retries if syntax errors 
			or schema mismatches are returned by SQLite.
		@param user_prompt: The user's natural language request.
		@param schema_str: The database schema as a string.
		@return: A valid SQL query string that adheres to the schema 
			and rules.
		"""
		system_prompt = f"""You are a SQLite Data Analyst Agent.
Your job is to generate a SINGLE valid SQLite SELECT query based on the user's request.

CRITICAL RULES:
1. Output ONLY the raw SQL query inside a ```sql ``` code block.
2. NEVER issue INSERT, UPDATE, DELETE, DROP, ALTER, or PRAGMA statements.
3. Keep results limited to 100 rows unless explicitly performing aggregated SUM/COUNT queries.

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

			# Security Guardrail Check.
			if not is_read_only_query(sql_query):
				error_msg = "Security Exception: Query failed read-only AST check."
				current_prompt = f"Previous SQL: {sql_query}\nError: {error_msg}\nPlease rewrite to use strictly SELECT statements."
				continue

			# Execution Check.
			try:
				df = self._execute_read_only(sql_query)
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

		raise RuntimeError(f"Self-healing SQL failed after {self.valves.MAX_RETRIES} attempts:\n" + "\n".join(error_history))


	def _agent_generate_visualizations_and_reports(self, user_prompt: str, sql_query: str, df: pd.DataFrame) -> str:
		if df.empty: 
			return f"### Query Executed\n```sql\n{sql_query}\n```\n\n**Result:** No matching records found."

		
		pass

