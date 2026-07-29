# helper.py
# Helper functions for database operations and data processing.
# Windows/MacOS/Linux
# Python 3.10


import sqlite3

import pandas as pd
import sqlglot
from sqlglot import exp


# Database configuration and global variables.
DB_PATH = "fec_data.db"


def get_connection() -> sqlite3.Connection:
	"""
	Establish and return a connection to the SQLite database.
	@param: takes no arguments.
	@return: sqlite3.Connection object.
	"""
	# Establish connection to database.
	conn = sqlite3.connect(DB_PATH)

	# Set SQLite pragmas for performance and reliability.
	conn.execute("PRAGMA page_size = 16384;")		# 16KB pages for higher compression.
	conn.execute("PRAGMA journal_mode = WAL;")		# Fast concurrent writes.
	conn.execute("PRAGMA synchronous = NORMAL;")	# Balance between performance and data safety.

	# Return connection.
	return conn


def get_db_schema() -> str:
	"""
	Retrieve the schemas of all tables in the database.
	@param: takes no arguments.
	@return: str containing the SQL schema of all tables.
	"""
	# Establish connection.
	conn = get_connection()

	# Retrieve the SQL schema for all tables in the database.
	schema = "\n".join(
		conn.execute("SELECT sql FROM sqlite_master WHERE type='table';").fetchall()
	)

	# Close connection.
	conn.close()

	# Return the schema.
	return schema


def is_read_only_query(query: str) -> bool:
	"""
	Determine if the given SQL query is read-only.
	@param query: SQL query string.
	@return: True if the query is read-only, False otherwise.
	"""
	try:
		# Parse the query. 
		parsed_statements = sqlglot.parse(query, dialect="sqlite")

		# If the parsed query is empty or invalid, return false.
		if not parsed_statements:
			return False

		# Define expression types that modify data or schema.
		write_types = (
			exp.Insert,
			exp.Update,
			exp.Delete,
			exp.Create,
			exp.Drop,
			exp.Alter,
			exp.Truncate,
			exp.Command,  # <- Covers things like PRAGMA writes, VACUUM, etc. if desired
		)

		# Iterate through each statement and validate it against the 
		# expression types.
		for statement in parsed_statements:
			# If the statement is None, return False.
			if statement is None:
				return False

			# If the statement is one of the write types, return False.
			if isinstance(statement, write_types):
				return False

			# Go down the AST to ensure no subqueries or CTEs contain 
			# mutations. Return false if the subquery contains any write 
			# types.
			for node in statement.find_all(Expression=True):
				if isinstance(node, write_types):
					return False

		# If none of the statements are write types, return True.
		return True
	except Exception:
		# If the query can't be parsed correctly, return false.
		return False


if __name__ == '__main__':
	###################################################################
	# Tests
	###################################################################

	# Get the schema.
	print(get_db_schema())

	# Test read-only query detection.
	print(is_read_only_query("SELECT * FROM fec_cm WHERE CMTE_ST = 'CA';")) 
	# Output: True
	print(is_read_only_query("UPDATE fec_cm SET CMTE_NM = 'Hacked' WHERE CMTE_ID = '123';")) 
	# Output: False
	print(is_read_only_query("SELECT * FROM fec_cm; DROP TABLE aux_congress;")) 
	# Output: False (Catches multi-statement injection attempts)

