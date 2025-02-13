import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

# Define the base folder for Player Stats
BASE_FOLDER = "Player Stats"

# Define the URLs and corresponding folder names
PLAYER_STATS_URLS = {
    "Standard Stats": "https://fbref.com/en/comps/9/stats/Premier-League-Stats",
    "Advanced GK Stats": "https://fbref.com/en/comps/9/keepersadv/Premier-League-Stats",
    "Shooting": "https://fbref.com/en/comps/9/shooting/Premier-League-Stats",
    "Passing": "https://fbref.com/en/comps/9/passing/Premier-League-Stats",
    "Goal and Shot Creation": "https://fbref.com/en/comps/9/gca/Premier-League-Stats",
    "Possession": "https://fbref.com/en/comps/9/possession/Premier-League-Stats",
    "Miscellaneous Stats": "https://fbref.com/en/comps/9/misc/Premier-League-Stats",
}

# Create base folder if it doesn't exist
os.makedirs(BASE_FOLDER, exist_ok=True)

# Set up Selenium
def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode (no GUI)
    driver = webdriver.Chrome(options=options)
    return driver

def scrape_player_stats():
    driver = setup_driver()

    for stat_name, url in PLAYER_STATS_URLS.items():
        print(f"Scraping data for {stat_name}...")

        # Create subfolder for the stat
        stat_folder = os.path.join(BASE_FOLDER, stat_name)
        os.makedirs(stat_folder, exist_ok=True)

        # Load the page
        driver.get(url)

        # Get the page source and parse it
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Find all tables on the page
        tables = soup.find_all("table")
        print(f"Found {len(tables)} tables for {stat_name}.")

        # Ensure we target the correct tables
        squad_table = None
        player_table = None

        for table in tables:
            headers = [th.text.strip() for th in table.find("thead").find_all("th")]

            if "Squad" in headers and not squad_table:
                squad_table = table  # Assign the squad table
            elif "Player" in headers and not player_table:
                player_table = table  # Assign the player table
        if squad_table:
            save_table_to_csv(squad_table, stat_folder, stat_name, "Squad")
        else:
            print(f"Squad table not found for {stat_name}.")

        if player_table:
            save_table_to_csv(player_table, stat_folder, stat_name, "Player")
        else:
            print(f"Player table not found for {stat_name}.")

    driver.quit()


def save_table_to_csv(table, folder, stat_name, table_type):
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
        filename = os.path.join(folder, f"{table_type}_{stat_name.replace(' ', '_')}_{date_str}.csv")

        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"Saved {table_type} data for '{stat_name}' to {filename}.")
    else:
        print(f"No valid data found in the {table_type} table for {stat_name}.")


# Run the scraper
scrape_player_stats()
