import os
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from datetime import datetime

# Set up Selenium driver (ensure you have the correct WebDriver)
driver = webdriver.Chrome()

# URLs for player stats
player_stats_links = {
    "Standard Stats": "https://fbref.com/en/comps/9/stats/Premier-League-Stats",
    "Advanced GK Stats": "https://fbref.com/en/comps/9/keepersadv/Premier-League-Stats",
    "Shooting": "https://fbref.com/en/comps/9/shooting/Premier-League-Stats",
    "Passing": "https://fbref.com/en/comps/9/passing/Premier-League-Stats",
    "Goal and Shot Creation": "https://fbref.com/en/comps/9/gca/Premier-League-Stats",
    "Possession": "https://fbref.com/en/comps/9/possession/Premier-League-Stats",
    "Miscellaneous Stats": "https://fbref.com/en/comps/9/misc/Premier-League-Stats"
}

# Base directory to save CSV files
base_dir = "Player Stats"

def scrape_player_stats():
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Loop through each player stats page
    for stat_name, url in player_stats_links.items():
        print(f"Scraping data for {stat_name}...")
        driver.get(url)

        # Wait up to 10 seconds for at least one table to appear
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table"))
            )
        except TimeoutException:
            print(f"Timed out waiting for table to load on {url}. Skipping.")
            continue

        # Now that at least one table is present, parse the page
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Find all tables on the page
        tables = soup.find_all("table")
        print(f"Found {len(tables)} tables for {stat_name}.")

        # Directory for this stat type
        stat_dir = os.path.join(base_dir, stat_name)
        if not os.path.exists(stat_dir):
            os.makedirs(stat_dir)

        # Loop through tables and extract data
        for i, table in enumerate(tables, start=1):
            thead = table.find("thead")
            tbody = table.find("tbody")

            # --- HEADERS ---
            if thead is not None:
                headers = [th.get_text(strip=True) for th in thead.find_all("th")]
            else:
                # Fallback: use first row of <tbody> if available
                print(f"No <thead> found for table {i}. Attempting fallback.")
                if tbody and tbody.find("tr"):
                    first_row = tbody.find("tr")
                    headers = [td.get_text(strip=True) for td in first_row.find_all("td")]
                else:
                    print(f"Skipping table {i} due to missing headers.")
                    continue

            # --- ROWS ---
            if not tbody:
                print(f"Skipping table {i} since <tbody> is missing.")
                continue

            rows_data = []
            for tr in tbody.find_all("tr"):
                row_cells = [td.get_text(strip=True) for td in tr.find_all("td")]
                # You could filter out completely empty rows if needed:
                if any(cell != '' for cell in row_cells):
                    rows_data.append(row_cells)

            if not rows_data:
                print(f"Skipping table {i} as it contains no data.")
                continue

            # Ensure header-row length consistency
            min_len = min(len(headers), len(rows_data[0]))
            headers = headers[:min_len]
            rows_data = [row[:min_len] for row in rows_data]

            # Save the table as a CSV file with the current date
            current_date = datetime.now().strftime("%Y-%m-%d")
            filename = f"{stat_name}_table_{i}_{current_date}.csv"
            file_path = os.path.join(stat_dir, filename)

            # Create and save DataFrame
            try:
                df = pd.DataFrame(rows_data, columns=headers)
                df.to_csv(file_path, index=False)
                print(f"Saved table {i} for {stat_name} as {filename}.")
            except Exception as e:
                print(f"Error creating DataFrame for table {i}: {e}")

    driver.quit()

# Run the scraper
scrape_player_stats()
