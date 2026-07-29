# download_data.py
# Download and merge the raw fec data with the candidate information to 
# get a complete picture of the election results.


from datetime import datetime
import io
import os
import re
import sqlite3
from typing import List, Optional
import zipfile

from bs4 import BeautifulSoup
import pandas as pd
import requests
from tqdm import tqdm


# Database configuration and global variables.
DB_PATH = "fec_data.db"
AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
SUFFIXES = {"JR", "SR", "II", "III", "IV"}
FILE_HEADERS = {
	# 'indiv': ['CMTE_ID', 'AMNDT_IND', 'RPT_TP', 'TRANSACTION_PGI', 'IMAGE_NUM', 'TRANSACTION_TP', 'ENTITY_TP', 'NAME', 'CITY', 'STATE', 'ZIP_CODE', 'EMPLOYER', 'OCCUPATION', 'TRANSACTION_DT', 'TRANSACTION_AMT', 'OTHER_ID', 'TRAN_ID', 'FILE_NUM', 'MEMO_CD', 'MEMO_TEXT', 'SUB_ID'],
	'cm': ['CMTE_ID', 'CMTE_NM', 'TRES_NM', 'CMTE_ST1', 'CMTE_ST2', 'CMTE_CITY', 'CMTE_ST', 'CMTE_ZIP', 'CMTE_DSGN', 'CMTE_TP', 'CMTE_PTY_AFFILIATION', 'CMTE_FILING_FREQ', 'ORG_TP', 'CONNECTED_ORG_NM', 'CAND_ID'],
	'cn': ['CAND_ID', 'CAND_NAME', 'CAND_PTY_AFFILIATION', 'CAND_ELECTION_YR', 'CAND_OFFICE_ST', 'CAND_OFFICE', 'CAND_OFFICE_DIST', 'CAND_ICI', 'CAND_STATUS', 'CAND_PCC', 'CAND_ST1', 'CAND_ST2', 'CAND_CITY', 'CAND_ST', 'CAND_ZIP'],
	'pas2': ['CMTE_ID', 'AMNDT_IND', 'RPT_TP', 'TRANSACTION_PGI', 'IMAGE_NUM', 'TRANSACTION_TP', 'ENTITY_TP', 'NAME', 'CITY', 'STATE', 'ZIP_CODE', 'EMPLOYER', 'OCCUPATION', 'TRANSACTION_DT', 'TRANSACTION_AMT', 'OTHER_ID', 'CAND_ID', 'TRAN_ID', 'FILE_NUM', 'MEMO_CD', 'MEMO_TEXT', 'SUB_ID'],
	'oth': ['CMTE_ID', 'AMNDT_IND', 'RPT_TP', 'TRANSACTION_PGI', 'IMAGE_NUM', 'TRANSACTION_TP', 'ENTITY_TP', 'NAME', 'CITY', 'STATE', 'ZIP_CODE', 'EMPLOYER', 'OCCUPATION', 'TRANSACTION_DT', 'TRANSACTION_AMT', 'OTHER_ID', 'TRAN_ID', 'FILE_NUM', 'MEMO_CD', 'MEMO_TEXT', 'SUB_ID'],
	'oppexp': ['CMTE_ID', 'AMNDT_IND', 'RPT_TP', 'LINE_NUM', 'FORM_TP_CD', 'SCHED_TP_CD', 'NAME', 'CITY', 'STATE', 'ZIP_CODE', 'TRANSACTION_DT', 'TRANSACTION_AMT', 'TRANSACTION_PGI', 'PURPOSE', 'CATEGORY', 'CATEGORY_DESC', 'MEMO_CD', 'MEMO_TEXT', 'ENTITY_TP', 'SUB_ID', 'FILE_NUM', 'TRAN_ID', 'BACK_REF_TRAN_ID']
}


#######################################################################
# SQLite database setup and utility functions.
#######################################################################

def db_connection() -> sqlite3.Connection:
	"""
	Connect to database and return the connection object.
	@param: takes no arguments
	@return: sqlite3.Connection object
	"""
	# Establish connection to database.
	conn = sqlite3.connect(DB_PATH)

	# Set SQLite pragmas for performance and reliability.
	conn.execute("PRAGMA page_size = 16384;")		# 16KB pages for higher compression.
	conn.execute("PRAGMA journal_mode = WAL;")		# Fast concurrent writes.
	conn.execute("PRAGMA synchronous = NORMAL;")	# Balance between performance and data safety.

	# Return connection.
	return conn


def init_manifest_table(conn: sqlite3.Connection) -> None:
	"""
	Initialize the manifest table in the databaset (if it does not 
		already exist).
	@param: conn - sqlite3.Connection object representing the database 
		connection.
	@return: None
	"""
	# Create the tracking table to monitor completed downloads/
	# ingestion.
	with conn:
		conn.execute("""
			CREATE TABLE IF NOT EXISTS download_manifest (
				file_identifier TEXT PRIMARY KEY,
				file_type TEXT NOT NULL,
				data_year INTEGER,
				records_ingested INTEGER NOT NULL,
				status TEXT NOT NULL,
				updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
			);
		""")


def is_already_ingested(conn: sqlite3.Connection, file_identifier: str) -> bool:
	"""
	Check if a file has already been ingested.
	@param: conn - sqlite3.Connection object representing the database 
		connection.
	@param: file_identifier - str representing the unique identifier of 
		the file.
	@return: True if the file has already been ingested, False otherwise.
	"""
	cursor = conn.cursor()
	cursor.execute(
		"SELECT 1 FROM download_manifest WHERE file_identifier = ? AND status = 'COMPLETED';", 
		(file_identifier,)
	)
	return cursor.fetchone() is not None


def mark_manifest_status(
		conn: sqlite3.Connection, 
		file_identifier: str, 
		f_type: str, 
		year: Optional[int], 
		records: int, 
		status: str
) -> None:
	"""
	Log/updates the status of an ingestion step.
	@param: conn - sqlite3.Connection object representing the database 
		connection.
	@param: file_identifier - str representing the unique identifier of 
		the file.
	@param: f_type - str representing the file type.
	@param: year - Optional[int] representing the data year.
	@param: records - int representing the number of records ingested.
	@param: status - str representing the current status of the 
		ingestion step.
	@return: None
	"""
	with conn:
		conn.execute("""
			INSERT INTO download_manifest (file_identifier, file_type, data_year, records_ingested, status)
			VALUES (?, ?, ?, ?, ?)
			ON CONFLICT(file_identifier) DO UPDATE SET
				records_ingested = excluded.records_ingested,
				status = excluded.status,
				updated_at = CURRENT_TIMESTAMP;
		""", (file_identifier, f_type, year, records, status))


#######################################################################
# Download presidential & congressional records and clean the data.
#######################################################################

def to_fec_format(name: str) -> str:
	"""
	Convert a name to the FEC standard format: 
		"LAST, FIRST MIDDLE SUFFIX". Handles suffixes like "JR", "SR", 
		"II", "III", "IV". Returns the name as is if it is NaN.
	@param: name - str representing the name to be converted.
	@return: str representing the name in FEC standard format.
	"""
	# If the name is NaN, return it as is.
	if pd.isna(name):
		return name
	
	# Standardize the name to uppercase and split it into parts.
	parts = name.strip().upper().split()
	
	# If there are no parts, return the name as is.
	if len(parts) == 0:
		return name
	
	# Handle suffix. If the last part of the name is a suffix, remove 
	# it from the parts list and store it separately.
	suffix = ""
	if parts[-1] in SUFFIXES:
		suffix = parts.pop()
	
	if len(parts) == 1:
		return parts[0]
	
	first = parts[0]
	last = parts[-1]
	middle = " ".join(parts[1:-1])
	
	# Rebuild the name.
	if middle:
		base = f"{last}, {first} {middle}"
	else:
		base = f"{last}, {first}"
	
	if suffix:
		base = f"{base} {suffix}"
	
	# Return the name.
	return base.replace("\"\"", "\"")


def fetch_and_ingest_presidents_wikipedia(conn: sqlite3.Connection) -> pd.DataFrame:
	"""
	Download president data and store to database.
	@param: conn - sqlite3.Connection object representing the database connection.
	@return: pd.DataFrame containing the president data.
	"""
	# Check if the data has already been ingested.
	file_id = "wiki_presidents"
	if is_already_ingested(conn, file_id):
		print("Skipping Wikipedia Presidents - already ingested.")
		return pd.read_sql("SELECT * FROM aux_presidents;", conn)

	# Download the ordered list of presidents from Wikipedia.
	print("Fetching Presidents table from Wikipedia...")
	url = "https://en.wikipedia.org/wiki/List_of_presidents_of_the_United_States"
	response = requests.get(url, headers={"User-Agent": AGENT})

	# Validate the response code.
	response.raise_for_status()

	# Parse the HTML content using BeautifulSoup and isolate the table.
	soup = BeautifulSoup(response.text, 'lxml')
	table = soup.find('table', {'class': 'wikitable'})

	# Convert the table data to a pandas DataFrame.
	df = pd.read_html(str(table))[0]

	# Clean data. Special cases for Joe Biden & Bill Clinton.
	regex_pattern = r'(.*?)\s*\((?:b\.\s*)?(\d{4})(?:[–-](\d{4}))?\)'
	df[['Name', 'Birth', 'Death']] = df['Name (birth–death)'].str.extract(regex_pattern)
	df['Name'] = df['Name'].str.strip()
	df.loc[df["Name"] == "Bill Clinton", "Name"] = "William Jefferson Clinton"
	df.loc[df["Name"] == "Joe Biden", "Name"] = "Joseph R. Biden"
	df['Name_FEC'] = df['Name'].apply(to_fec_format)
	df['Election'] = (
		pd.to_numeric(df['Election'].str.extract(r'(\d{4})')[0])
		.fillna(0)
		.astype(int)
	)

	# Store the cleaned data into the database and update the manifest.
	with conn:
		df.to_sql(
			"aux_presidents", conn, if_exists="replace", index=False
		)
		mark_manifest_status(
			conn, file_id, "presidents", None, len(df), "COMPLETED"
		)

	# Return the dataframe.
	return df


def fetch_and_ingest_congress(conn: sqlite3.Connection) -> pd.DataFrame:
	"""
	Download congressional legislator data and store to database.
	@param: conn - sqlite3.Connection object representing the database connection.
	@return: pd.DataFrame containing the congressional legislator data.
	"""
	# Check if the data has already been ingested.
	file_id = "github_congress"
	if is_already_ingested(conn, file_id):
		print("Skipping Historical Congress Data - already ingested.")
		return pd.read_sql("SELECT * FROM aux_congress;", conn)
	
	# Download the ordered list of presidents from unitedstates.github.io.
	print("Fetching Historical Congress Data...")
	url = "https://unitedstates.github.io/congress-legislators/legislators-historical.json"
	response = requests.get(url, headers={"User-Agent": AGENT})

	# Validate the response code.
	response.raise_for_status()

	# Convert the response to JSON and then to a pandas DataFrame.
	data = response.json()
	df_raw = pd.json_normalize(data)

	# Expand the 'terms' column into separate rows and normalize the 
	# nested JSON structure.
	congress_expanded = df_raw.explode('terms').reset_index(drop=True)
	terms_columns = pd.json_normalize(congress_expanded['terms'])
	final_df = pd.concat(
		[congress_expanded.drop(columns='terms'), terms_columns], 
		axis=1
	)

	# Create full name and FEC-formatted candidate name, and extract 
	# start and end years from the terms.
	final_df["full_name"] = final_df[["name.first", "name.last"]]\
		.fillna("")\
		.agg(" ".join, axis=1)\
		.str\
		.strip()
	final_df["candidate_fec"] = final_df["full_name"].apply(to_fec_format)
	final_df["start_year"] = pd.to_datetime(final_df["start"], errors='coerce').dt.year - 1
	final_df["end_year"] = pd.to_datetime(final_df["end"], errors='coerce').dt.year - 1

	# NOTE/TODO:
	# Update this later to be more robust for handling columns 
	# containing lists or dicts before ingestion. For the moment, we
	# just need the data to be loaded to the database for the POC. In
	# a real production environment, we would need to handle these 
	# cases properly, so that the data is clean before ingestion.
	for col in final_df.columns:
		if final_df[col].dtype == "object":
			# Convert all columns containing lists or dicts to strings 
			# before ingestion.
			final_df[col] = final_df[col].apply(
				lambda x: str(x) if isinstance(x, (list, dict)) else x
			)

	with conn:
		final_df.to_sql(
			"aux_congress", conn, if_exists="replace", index=False
		)
		mark_manifest_status(
			conn, file_id, "congress", None, len(final_df), "COMPLETED"
		)

	# Return the dataframe.
	return final_df


#######################################################################
# Download FEC bulk data records and clean the data.
#######################################################################

def download_and_ingest_fec_bulk_data(
		conn: sqlite3.Connection, 
		years: List[int], 
		file_types: List[str]
) -> None:
	"""
	Streams FEC bulk zip files, reads chunks, downcasts, and writes to 
		database.
	@param: conn - sqlite3.Connection object representing the database 
		connection.
	@param: years - List[int] representing the years of the FEC data to 
		download.
	@param: file_types - List[str] representing the types of FEC files 
		to download.
	@return: None
	"""
	# Iterate through the years.
	for year in years:
		# Isolate the last two digits of the year for constructing the 
		# FEC bulk download URL.
		yy = str(year)[-2:]

		# Iterate through the file types.
		for f_type in file_types:
			file_id = f"fec_{f_type}_{year}"

			# Skip the file if it has already been successfully ingested.
			if is_already_ingested(conn, file_id):
				print(f"Skipping {file_id} - already successfully ingested.")
				continue

			# Build the URL for the FEC bulk download.
			url = f"https://www.fec.gov/files/bulk-downloads/{year}/{f_type}{yy}.zip"
			print(f"Processing {file_id} from {url}...")
			
			try:
				# Fetch the FEC bulk zip file from the constructed URL.
				response = requests.get(url, timeout=60)
				response.raise_for_status()

				# Open the zip file and locate the target text file within it.
				with zipfile.ZipFile(io.BytesIO(response.content)) as z:
					target_file = next(
						(
							f for f in z.namelist() 
							if not f.startswith('by_date/') and f.endswith('.txt')
						), 
						z.namelist()[0]
					)

					total_rows = 0
					table_name = f"fec_{f_type}"

					# Prepare the table name for ingestion.
					with z.open(target_file) as f:
						text_wrapper = io.TextIOWrapper(f, encoding="ISO-8859-1", errors="replace")
						reader = pd.read_csv(
							text_wrapper,
							sep='|',
							names=FILE_HEADERS.get(f_type),
							index_col=False,
							chunksize=100_000,
							low_memory=False,
							engine="c",
							on_bad_lines='skip'
						)

						# Wrap ingestion in explicit transaction.
						with conn:
							for chunk in reader:
								chunk["data_year"] = year
								
								# Downcast numeric types for size 
								# optimization.
								for col in chunk.select_dtypes(include=["int64"]).columns:
									chunk[col] = pd.to_numeric(chunk[col], downcast="integer")
								for col in chunk.select_dtypes(include=["float64"]).columns:
									chunk[col] = pd.to_numeric(chunk[col], downcast="float")

								chunk.to_sql(
									name=table_name,
									con=conn,
									if_exists="append",
									chunksize=10_000,
									index=False,
									method="multi"
								)
								total_rows += len(chunk)

							# Record completion in manifest within same 
							# transaction.
							mark_manifest_status(conn, file_id, f_type, year, total_rows, "COMPLETED")

				print(f"\tSuccessfully ingested {total_rows:,} rows into table '{table_name}'.")

			except Exception as e:
				print(f"\tERROR processing {file_id}: {e}")
				mark_manifest_status(conn, file_id, f_type, year, 0, f"FAILED: {str(e)[:100]}")

	pass


def main():
	# Establish database connection and initialize manifest table.
	conn = db_connection()
	init_manifest_table(conn)

	# Download and ingest candidate sources.
	_ = fetch_and_ingest_presidents_wikipedia(conn)
	_ = fetch_and_ingest_congress(conn)

	# Download and ingest bulk FEC data.
	required_file_types = [
		'cm', 
		# 'indiv', 
		'oth', 
		'pas2', 
		'cn'
	]
	current_year = int(datetime.now().year)
	all_years = [year for year in range(1980, current_year, 2)]
	download_and_ingest_fec_bulk_data(
		conn, years=all_years, file_types=required_file_types
	)

	# Reclaim and optimize.
	print("Optimizing database file structure with VACUUM...")
	conn.execute("VACUUM;")
	conn.close()
	print("Ingestion complete. All datasets stored inside SQLite.")

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()