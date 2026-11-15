# helper.py
# Helper functions for database operations and data processing.
# Windows/MacOS/Linux
# Python 3.10


import sqlite3
from typing import List

import sqlglot
from sqlglot import exp


# Database configuration and global variables.
DB_PATH = "fec_data.db"
DB_PATH_METADATA = "fec_metadata.db"


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
		result[0] 
		for result in conn.execute("SELECT sql FROM sqlite_master WHERE type='table';").fetchall()
		if isinstance(result, tuple) and len(result) == 1 and isinstance(result[0], str)
	)

	# Close connection.
	conn.close()

	# Return the schema.
	return schema


def get_tables() -> List[str]:
	"""
	Retrieve the list of table names from the database.
	@param: takes no arguments.
	@return: List of table names as strings.
	"""
	# Establish connection.
	conn = get_connection()

	# Retrieve the list of table names in the database.
	tables = [
		result[0] 
		for result in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
		if isinstance(result, tuple) and len(result) == 1 and isinstance(result[0], str)
	]

	# Close connection.
	conn.close()

	# Return the list of table names.
	return tables


def get_metadata_connection() -> sqlite3.Connection:
	"""
	Establish and return a connection to the metadata SQLite database.
	@param: takes no arguments.
	@return: sqlite3.Connection object.
	"""
	# Establish connection to database.
	conn = sqlite3.connect(DB_PATH_METADATA)

	# Set SQLite pragmas for performance and reliability.
	conn.execute("PRAGMA page_size = 16384;")		# 16KB pages for higher compression.
	conn.execute("PRAGMA journal_mode = WAL;")		# Fast concurrent writes.
	conn.execute("PRAGMA synchronous = NORMAL;")	# Balance between performance and data safety.

	# Return connection.
	return conn


def get_metadata_db_schema() -> str:
	"""
	Retrieve the schemas of all tables in the metadata database.
	@param: takes no arguments.
	@return: str containing the SQL schema of all tables.
	"""
	# Establish connection.
	conn = get_metadata_connection()

	# Retrieve the SQL schema for all tables in the database.
	schema = "\n".join(
		result[0] 
		for result in conn.execute("SELECT sql FROM sqlite_master WHERE type='table';").fetchall()
		if isinstance(result, tuple) and len(result) == 1 and isinstance(result[0], str)
	)

	# Close connection.
	conn.close()

	# Return the schema.
	return schema


def get_metadata() -> str:
	"""
	Retrieve the metadata from the metadata database.
	@param: takes no arguments.
	@return: str containing the metadata of all tables.
	"""
	# Establish connection.
	conn = get_metadata_connection()

	# Get the list of tables in the metadata database.
	tables = get_metadata_tables()

	# Initialise the metadata string to accumulate results.
	metadata = ""

	# Iterate through each table and retrieve its metadata.
	for table in tables:
		# Retrieve the metadata for all tables in the database.
		# print(f"Metadata table found: {table}")
		metadata += f"\n--- Metadata for table: {table.rstrip('_table')} ---\n"
		metadata += "\n".join(
			"|".join(str(item) for item in result)
			for result in conn.execute(f"SELECT * FROM {table};").fetchall()
			if isinstance(result, tuple)
		)
		metadata += "\n"	# Add a newline between tables for clarity.

	# Close connection.
	conn.close()

	# Return the metadata.
	return metadata


def get_metadata_tables() -> List[str]:
	"""
	Retrieve the list of table names from the metadata database.
	@param: takes no arguments.
	@return: List of table names as strings.
	"""
	# Establish connection.
	conn = get_metadata_connection()

	# Retrieve the list of table names in the database.
	tables = [
		result[0] 
		for result in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
		if isinstance(result, tuple) and len(result) == 1 and isinstance(result[0], str)
	]

	# Close connection.
	conn.close()

	# Return the list of table names.
	return tables


def get_filtered_db_schema(tables: List[str]) -> str:
	"""
	Retrieve the schemas of specified tables in the database.
	@param tables: List of table names to retrieve schemas for.
	@return: str containing the SQL schema of specified tables.
	"""
	if not tables:
		return get_db_schema()

	# Establish connection.
	conn = get_connection()

	# Retrieve the SQL schema for specified tables in the database.
	placeholders = ','.join('?' for _ in tables)
	schema = "\n".join(
		result[0] 
		for result in conn.execute(
			f"SELECT sql FROM sqlite_master WHERE type='table' AND name IN ({placeholders});", 
			tables
		).fetchall()
		if isinstance(result, tuple) and len(result) == 1 and isinstance(result[0], str)
	)

	# Close connection.
	conn.close()

	# Return the schema.
	return schema


TABLE_METADATA_MAP = {
	"fec_cn": "all_candidates_table",
	"fec_cm": "committee_master_table",
	"fec_pas2": "committee_contributions_table",
	"fec_oth": "committee_contributions_table",
}


def get_filtered_metadata(tables: List[str]) -> str:
	"""
	Retrieve the metadata for specified tables in the metadata database.
	@param tables: List of table names to retrieve metadata for.
	@return: str containing the metadata of specified tables.
	"""
	if not tables:
		return ""

	# Establish connection.
	conn = get_metadata_connection()
	all_metadata_tables = get_metadata_tables()

	# Initialise the metadata string to accumulate results.
	metadata = ""
	seen_meta_tables = set()

	# Iterate through each specified table and retrieve its metadata.
	for table in tables:
		target_meta = TABLE_METADATA_MAP.get(table)
		if not target_meta:
			candidate = f"{table}_table"
			if candidate in all_metadata_tables:
				target_meta = candidate
			elif table in all_metadata_tables:
				target_meta = table

		if target_meta and target_meta not in seen_meta_tables:
			seen_meta_tables.add(target_meta)
			metadata += f"\n--- Metadata for table: {table} ({target_meta}) ---\n"
			metadata += "\n".join(
				"|".join(str(item) for item in result)
				for result in conn.execute(f"SELECT * FROM {target_meta};").fetchall()
				if isinstance(result, tuple)
			)
			metadata += "\n"  # Add a newline between tables for clarity.

	# Close connection.
	conn.close()

	# Return the metadata.
	return metadata


def is_read_only_query(query: str) -> bool:
	"""
	Determine if the given SQL query is read-only.
	@param query: SQL query string.
	@return: True if the query is read-only, False otherwise.
	"""
	try:
		# Parse the query. 
		parsed_statements = sqlglot.parse(query, dialect="sqlite")
		print(parsed_statements)

		# If the parsed query is empty or invalid, return false.
		if not parsed_statements:
			return False

		# Explicitly allow statement types that are read-only at the 
		# root level.
		allowed_root_types = (
			exp.Select,
			exp.Union,
			exp.Expression,  # Base wrapper for standard expressions
		)

		# Define expression types that modify data or schema.
		write_types = (
			exp.Insert,
			exp.Update,
			exp.Delete,
			exp.Create,
			exp.Drop,
			exp.Alter,
			# exp.Truncate,
			exp.Command,  # <- Covers things like PRAGMA writes, VACUUM, etc. if desired
		)

		# Iterate through each statement and validate it against the 
		# expression types.
		for statement in parsed_statements:
			# If the statement is None, return False.
			if statement is None:
				return False

			# If the statement is one of the write types, return False.
			# if isinstance(statement, write_types):
			if not isinstance(statement, allowed_root_types):# or isinstance(statement, exp.Command):
				print(f"Disallowed root statement type detected: {type(statement).__name__}")
				return False

			# Go down the AST to ensure no subqueries or CTEs contain 
			# mutations. Return false if the subquery contains any write 
			# types.
			# for node in statement.find_all(Expression=True):
			for node in statement.walk():
				if isinstance(node, write_types):
					print(f"Disallowed subquery/CTE statement type detected: {type(node).__name__}")
					return False

		# If none of the statements are write types, return True.
		return True
	except Exception as e:
		# If the query can't be parsed correctly, return false.
		print(f"Query parsing failed for: {query}\nError: {e}")
		return False


if __name__ == '__main__':
	###################################################################
	# Tests
	###################################################################

	# Get the schema and database metadata.
	print(get_db_schema())
	print(get_tables())
	print(get_metadata_db_schema())
	print(get_metadata_tables())
	print(get_metadata())
	print(len(get_metadata().splitlines()))
	print(len(get_metadata()))

	# # Test read-only query detection.
	# print(is_read_only_query("SELECT * FROM fec_cm WHERE CMTE_ST = 'CA';")) 
	# # Output: True
	# print(is_read_only_query("UPDATE fec_cm SET CMTE_NM = 'Hacked' WHERE CMTE_ID = '123';")) 
	# # Output: False
	# print(is_read_only_query("SELECT * FROM fec_cm; DROP TABLE aux_congress;")) 
	# # Output: False (Catches multi-statement injection attempts)

