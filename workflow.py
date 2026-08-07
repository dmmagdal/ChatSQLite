# workflow.py


import sqlite3
from typing import List, Dict, Any, Tuple, TypedDict, Annotated

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
import pandas as pd
from sqlglot import parse_one, transpile

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


# Initialize Ollama model(s).
OLLAMA_URL = "http://localhost:11434"
chat = ChatOllama(model="phi4-mini-reasoning:3.8b", base_url=OLLAMA_URL, temperature=0.0)
sql = ChatOllama(model="qwen2.5-coder:1.5b", base_url=OLLAMA_URL)
coder = ChatOllama(model="qwen2.5-coder:1.5b", base_url=OLLAMA_URL)


#######################################################################
# Define nodes.
#######################################################################

def retrieve_metadata(state: AgentState) -> AgentState:
	"""
	Retrieves the database schema and updates the state.
	"""
	state["db_schema"] = get_db_schema()
	state["db_tables_meta_schema"] = get_metadata_db_schema()
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

	# Generate SQL query using Ollama model.
	response = sql.invoke(
		[
			SystemMessage(content=prompt), 
			HumanMessage(content=state['user_query'])
		]
	)
	
	# Update state with generated SQL query.
	state["sql_query"] = response.content.strip()
	
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
	
	USER INPUT:
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
workflow.add_edge("generate_sql_query", "execute_sql_query")
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
	result = app.invoke(initial_state)
	print(result["final_answer"])

