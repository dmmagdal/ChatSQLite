# workflow.py


import sqlite3
import re
import os
from typing import List, Dict, Any, Tuple, TypedDict, Annotated

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from sqlglot import parse, parse_one, transpile
from sqlglot.errors import ParseError, TokenError

from helper import *


# State.
class AgentState(TypedDict):
	"""
	Represents the state of the agent in the workflow.
	"""
	db_schema: str
	db_tables_meta_schema: str
	user_query: str
	sql_query: str
	sql_error: str
	# query_result: pd.DataFrame
	query_result: List[Tuple[str]]
	final_answer: str
	failure_count: int


# Initialize Ollama model(s).
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# chat = ChatOllama(model="phi4-mini-reasoning:3.8b", base_url=OLLAMA_URL, temperature=0.0)
chat = ChatOllama(model="lfm2.5-thinking:1.2b", base_url=OLLAMA_URL, temperature=0.0)
# sql = ChatOllama(model="qwen2.5-coder:1.5b", base_url=OLLAMA_URL)
sql = ChatOllama(model="qwen2.5-coder:1.5b", base_url=OLLAMA_URL, temperature=0.0)
coder = ChatOllama(model="qwen2.5-coder:1.5b", base_url=OLLAMA_URL)


def extract_sql(text: str) -> str:
	"""
	Return the longest SQL candidate from an LLM response.

	SQL-labelled and unlabelled Markdown fences are supported, as are
	unfenced statements starting on their own line or after a colon.
	Raises ValueError if no SQL candidate is found. Extraction does not
	validate SQL safety.
	"""
	text = re.sub(
		r"(?m)^[ \t]*<(think|thinking|analysis)\b[^>]*>.*?(?:</\1\s*>|\Z)",
		"",
		text,
		flags=re.IGNORECASE | re.DOTALL,
	)
	fence_start = re.compile(r"(?m)^[ \t]{0,3}(`{3,}|~{3,})[ \t]*([^\r\n]*)\r?\n")
	sql_start = re.compile(
		r"(?im)(?:^[ \t]*|(?<=:)[ \t]+)(?:SELECT|WITH|VALUES|INSERT|UPDATE|DELETE|REPLACE|"
		r"CREATE|ALTER|DROP|PRAGMA|EXPLAIN|VACUUM|ATTACH|DETACH|BEGIN|"
		r"COMMIT|ROLLBACK|SAVEPOINT|RELEASE|REINDEX|ANALYZE)\b"
	)
	sql_continuation = re.compile(
		r"(?i)^(?:FROM|WHERE|GROUP|HAVING|ORDER|LIMIT|OFFSET|JOIN|"
		r"LEFT|RIGHT|INNER|OUTER|CROSS|ON|UNION|INTERSECT|EXCEPT|"
		r"AND|OR|AS|RETURNING|VALUES|SET|INTO)\b|^[,)]"
	)
	candidates = []
	unfenced = []
	position = 0
	while match := fence_start.search(text, position):
		unfenced.append(text[position:match.start()])
		marker, info = match.group(1), match.group(2).strip().lower()
		fence_end = re.compile(
			r"(?m)^[ \t]{0,3}" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}[ \t]*(?:\r?$)"
		).search(text, match.end())
		end = fence_end.start() if fence_end else len(text)
		content = text[match.end():end].strip()
		language = info.split(maxsplit=1)[0] if info else ""
		without_comments = re.sub(r"\A(?:\s|--[^\n]*(?:\n|$)|/\*.*?\*/)*", "", content, flags=re.DOTALL)
		if content and (language in ("sql", "sqlite", "sqlite3") or (not info and sql_start.match(without_comments))):
			candidates.append(content)
		position = fence_end.end() if fence_end else len(text)
	unfenced.append(text[position:])

	# Only parse unfenced text: fenced SQL can still be sent to SQLite for
	# useful syntax errors and the existing self-healing retry.
	for section in unfenced:
		for match in sql_start.finditer(section):
			tail = section[match.start():]
			endpoints = {len(tail)}
			endpoints.update(m.end() for m in re.finditer(r";|\r?\n", tail))
			for end in sorted(endpoints, reverse=True):
				candidate = tail[:end].strip()
				if not candidate.endswith(";") and sql_continuation.match(tail[end:].lstrip()):
					continue
				try:
					statements = parse(candidate, dialect="sqlite")
				except (ParseError, TokenError):
					continue
				if len(statements) == 1 and statements[0] is not None:
					candidates.append(candidate)
					break

	if not candidates:
		raise ValueError("No SQL statement found in the model response.")

	return max(candidates, key=len)


#######################################################################
# Define nodes.
#######################################################################

def retrieve_metadata(state: AgentState) -> AgentState:
	"""
	Retrieves the database schema and updates the state.
	"""
	state["db_schema"] = get_db_schema()
	# state["db_tables_meta_schema"] = get_metadata_db_schema()
	state["db_tables_meta_schema"] = get_metadata()
	state["failure_count"] = state.get("failure_count", 0)
	return state


def generate_sql_query(state: AgentState) -> AgentState:
	"""
	Generates an SQL query based on the user's natural language query.
	"""
	# NOTE
	# It is important that any additional context or schema information is included in the system prompt to ensure 
	# the LLM is aware of the relationship between tables and their columns in order to generate valid SQL queries.
	prompt = f"""You are a SQLite Data Analyst Agent.
	Your job is to generate a SINGLE valid SQLite SELECT query based on the user's request.
	
	CRITICAL RULES:
	1. Output ONLY the raw SQL query inside a ```sql ``` code block.
	2. NEVER issue INSERT, UPDATE, DELETE, DROP, ALTER, or PRAGMA statements.
	3. Keep results limited to 100 rows unless explicitly performing aggregated SUM/COUNT queries.
	
	TABLE METADATA:
	{state['db_tables_meta_schema']}
	
	DATABASE SCHEMA:
	{state['db_schema']}
	"""

	if state.get("sql_error"):
		prompt += f"\n\nPREVIOUS SQL ERROR:\n{state['sql_error']}\n\n"
		# print("-" * 72)
		# print(prompt)
		# print(state["sql_query"])
		# print("-" * 72)
		state["failure_count"] += 1
		if state["failure_count"] > 5:
			state["final_answer"] = (
				f"Unable to execute the SQL query after {state['failure_count']} failures. "
				f"Last SQL error: {state['sql_error']}"
			)
			return state

	# Generate SQL query using Ollama model.
	response = sql.invoke(
		[
			SystemMessage(content=prompt), 
			HumanMessage(content=state['user_query'])
		]
	)
	
	# Update state with generated SQL query.
	try:
		state["sql_query"] = extract_sql(response.content)
	except ValueError as e:
		state["sql_query"] = ""
		state["sql_error"] = str(e)
	
	return state


def execute_sql_query(state: AgentState) -> AgentState:
	"""
	Executes the generated SQL query and updates the state with the results or any errors.
	"""
	try:
		# Establish connection to the database.
		conn = get_connection()
		
		# Execute the SQL query and fetch results.
		# state["query_result"] = pd.read_sql_query(state["sql_query"], conn)
		state["query_result"] = conn.execute(state["sql_query"]).fetchall()
		
		# Clear any previous SQL error.
		state["sql_error"] = ""
	except Exception as e:
		# Capture any SQL execution errors.
		state["sql_error"] = str(e)
	finally:
		# Close the database connection.
		conn.close()
	
	return state


def generate_insights(state: AgentState) -> AgentState:
	"""
	Generates insights based on the SQL query results.
	"""
	prompt = f"""You are a Data Analyst Agent.
	Your job is to generate insights based on the SQL query results and the user input.
	
	CRITICAL RULES:
	1. Output ONLY the insights in plain text.
	2. Do not include any SQL queries or code blocks.
	
	SQL QUERY RESULTS:
	{state['query_result']}
	
	USER INPUT/QUERY:
	{state['user_query']}
	"""

	# Generate insights using Ollama model.
	response = chat.invoke(
		[
			SystemMessage(content=prompt), 
			HumanMessage(content="Please provide insights based on the above SQL query results.")
		]
	)
	
	# Update state with generated insights.
	state["final_answer"] = response.content.strip()
	
	return state


#######################################################################
# Define router.
#######################################################################

def route_after_generation(state: AgentState) -> str:
	"""
	Stops retrying after more than five SQL failures.
	"""
	if state["failure_count"] > 5:
		return END
	if not state["sql_query"]:
		return "generate_sql_query"
	return "execute_sql_query"


def route_after_execution(state: AgentState) -> str:
	"""
	Routes the workflow based on the presence of SQL execution errors.
	"""
	if state["sql_error"]:
		return "generate_sql_query"  # If there's an error, go back to generating SQL query.
	else:
		return "generate_insights"  # If no error, proceed to generate insights.


#######################################################################
# Build graph.
#######################################################################

# Build the graph.
workflow = StateGraph(AgentState)

# Add the nodes.
workflow.add_node("retrieve_metadata", retrieve_metadata)
workflow.add_node("generate_sql_query", generate_sql_query)
workflow.add_node("execute_sql_query", execute_sql_query)
workflow.add_node("generate_insights", generate_insights	)

# Define flow (edges).
workflow.set_entry_point("retrieve_metadata")
workflow.add_edge("retrieve_metadata", "generate_sql_query")
workflow.add_conditional_edges(
	"generate_sql_query",
	route_after_generation,
	{
		"execute_sql_query": "execute_sql_query",
		"generate_sql_query": "generate_sql_query",
		END: END
	}
)
workflow.add_conditional_edges(
	"execute_sql_query", 
	route_after_execution,
	{
		"generate_sql_query": "generate_sql_query",
		"generate_insights": "generate_insights"
	}
)
workflow.add_edge("generate_insights", END)

# Compile the graph.
app = workflow.compile()

# Example Usage
if __name__ == "__main__":
	# initial_state = {"question": "What were the top 5 highest contributions?"}
	initial_state = {"user_query": "What were the top 5 highest contributions?"}
	# initial_state = {"user_query": "What were the top 5 highest contributions? Be sure to list the names and dollar amounts in a ranked list order."}
	# initial_state = {"user_query": "What were the top 5 highest contributions? Be sure to list the dollar amounts in a ranked list order."}
	# initial_state = {"user_query": "What were the top 5 highest contributions? Be sure to list the names in a ranked list order."}
	# initial_state = {"user_query": "What were the top 5 highest contributions? Be sure to list the information in a ranked list order."}
	result = app.invoke(initial_state)
	print(result["final_answer"])

