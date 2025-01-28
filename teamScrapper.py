import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

# Define the base folder for Team Stats
BASE_FOLDER = "Team Stats"

# URL for Team Stats
TEAM_STATS_URL = "https://fbref.com/en/comps/9/Premier-League-Stats"

# Create base folder if it doesn't exist
os.makedirs(BASE_FOLDER, exist_ok=True)

def scrape_team_stats():
    print(f"Scraping data from {TEAM_STATS_URL}...")

    # Set up headers to mimic a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # Request the page content
    response = requests.get(TEAM_STATS_URL, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch the page. Status code: {response.status_code}")
        return

    # Parse the HTML content
    soup = BeautifulSoup(response.content, "html.parser")

    # Find all tables on the page
    tables = soup.find_all("table")
    if not tables:
        print("No tables found on the page.")
        return

    # Loop through each table and process it
    for table in tables:
        # Get the table ID (used to name the folder and file)
        table_id = table.get("id")
        if not table_id:
            continue  # Skip tables without an ID

        print(f"Processing table: {table_id}")

        # Create a subfolder for this table
        table_folder = os.path.join(BASE_FOLDER, table_id)
        os.makedirs(table_folder, exist_ok=True)

        # Extract headers
        headers = [th.text.strip() for th in table.find("thead").find_all("th")]

        # Extract rows
        rows = []
        for tr in table.find("tbody").find_all("tr"):
            row_data = []
            for td in tr.find_all(["td", "th"]):
                # Handle links inside table cells
                if td.find("a"):
                    row_data.append(td.find("a").text.strip())
                else:
                    row_data.append(td.text.strip())
            if row_data:  # Only add non-empty rows
                rows.append(row_data)

        # Ensure all rows have the same number of columns as headers
        if rows and headers:
            max_columns = len(headers)
            for i in range(len(rows)):
                if len(rows[i]) < max_columns:
                    rows[i] += [""] * (max_columns - len(rows[i]))  # Pad missing columns

            # Create a DataFrame
            df = pd.DataFrame(rows, columns=headers)

            # Generate a dated filename
            date_str = datetime.now().strftime("%Y-%m-%d")
            filename = os.path.join(table_folder, f"{table_id}_{date_str}.csv")

            # Save to CSV
            df.to_csv(filename, index=False)
            print(f"Saved data for table '{table_id}' to {filename}.")
        else:
            print(f"Table '{table_id}' has no valid data or headers.")

# Run the scraper
scrape_team_stats()
