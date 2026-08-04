# workflow.py


import sqlite3
from typing import List, Dict, Any, TypedDict, Annotated

from langchain_ollama import ChatOllama
from langchain_core import SystemMessage, HumanMessage
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
    user_query: str
    sql_query: str
    sql_error: str
    query_result: pd.DataFrame
    final_answer: str


# Initialize Ollama model(s).
OLLAMA_URL = "http://localhost:11434"
chat = ChatOllama(model="llama2", base_url=OLLAMA_URL)
sql = ChatOllama(model="qwen2.5-coder:1.5b", base_url=OLLAMA_URL)
coder = ChatOllama(model="qwen2.5-coder:1.5b", base_url=OLLAMA_URL)


# Define nodes.
###########################