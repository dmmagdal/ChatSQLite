# download_metadata.py
# Download the metadata around the fec information. This will allow for better prompting and context for the LLM to 
# generate valid SQL queries.
# Windows/MacOS/Linux
# Python 3.10


# import json
import sqlite3

from bs4 import BeautifulSoup
import pandas as pd
import requests


# Database configuration and global variables.
DB_PATH = "fec_metadata.db"
AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
URLS = {
	"all_candidates": "https://www.fec.gov/campaign-finance-data/all-candidates-file-description/",
	"committee_master": "https://www.fec.gov/campaign-finance-data/committee-master-file-description/",
	"committee_contributions": "https://www.fec.gov/campaign-finance-data/contributions-committees-candidates-file-description/",
	"report_types": "https://www.fec.gov/campaign-finance-data/report-type-code-descriptions/",
	"transaction_types": "https://www.fec.gov/campaign-finance-data/transaction-type-code-descriptions/",
	"committee_types": "https://www.fec.gov/campaign-finance-data/committee-type-code-descriptions/",
	"party_codes": "https://www.fec.gov/campaign-finance-data/party-code-descriptions/"
}


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


def main():
	# Initialize the extracted data as a JSON dictionary.
	extracted_data = {}

	# Establish connection to the database.
	conn = db_connection()

	# Iterate through each url and fetch the metadata.
	for key, url in URLS.items():
		# Send a GET request to the URL and check for a successful response.
		print(f"Fetching {key}...")
		resp = requests.get(url, headers={"User-Agent": AGENT})
		if resp.status_code != 200:
			print(f"Failed to fetch {url}")
			continue

		# Parse the HTML content.
		soup = BeautifulSoup(resp.content, "html.parser")

		# Save the page content to a local HTML file for reference.
		with open(f"{key}.html", "w", encoding="utf-8") as f:
			f.write(soup.prettify())

		# Extract the table data from the page.
		tables = soup.find_all("table")

		# table_data = []
		for table in tables:
			# Load the table into a DataFrame.
			df = pd.read_html(str(table))[0]

			# Fix the dataframe to have the first row as header and 
			# remove it from the data.
			df.columns = df.iloc[0]
			df = df[1:].reset_index(drop=True)

			# Standardize column names.
			df.columns = [
				col.lower().replace(" ", "_") for col in df.columns
			]

			# Save the table to the database.
			with conn:
				df.to_sql(
					f"{key}_table", conn, if_exists="replace", index=False
				)

	# 	extracted_data[key] = table_data

	# # Save extracted dictionary
	# with open("raw_fec_docs.json", "w") as f:
	# 	json.dump(extracted_data, f, indent=2)

	# Close the connection to the database.
	conn.close()

	# Exit the program.
	exit(0)


if __name__ == "__main__":
	main()