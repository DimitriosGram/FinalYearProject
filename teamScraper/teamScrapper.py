import os
import boto3
import pandas as pd
import time
import re
from bs4 import Comment
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

def handler(event, context):
    chrome_options = webdriver.ChromeOptions()
    chrome_options.binary_location = os.environ.get("CHROME_BIN", "/usr/bin/google-chrome-stable")
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--single-process")
    chrome_options.add_argument("--disable-background-networking")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-sync")
    chrome_options.add_argument("--metrics-recording-only")
    chrome_options.add_argument("--safebrowsing-disable-auto-update")
    chrome_options.add_argument("--disable-default-apps")
    chrome_options.add_argument("--no-first-run")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--disable-translate")
    chrome_options.add_argument("--window-size=1280,800")

    service = Service("/usr/bin/chromedriver")
    driver = webdriver.Chrome(service=service, options=chrome_options)

    player_stats_links = {
        "Standard Stats": "https://fbref.com/en/comps/9/stats/Premier-League-Stats",
        "Advanced GK Stats": "https://fbref.com/en/comps/9/keepersadv/Premier-League-Stats",
        "Shooting": "https://fbref.com/en/comps/9/shooting/Premier-League-Stats",
        "Passing": "https://fbref.com/en/comps/9/passing/Premier-League-Stats",
        "Goal and Shot Creation": "https://fbref.com/en/comps/9/gca/Premier-League-Stats",
        "Possession": "https://fbref.com/en/comps/9/possession/Premier-League-Stats",
        "Miscellaneous Stats": "https://fbref.com/en/comps/9/misc/Premier-League-Stats"
    }

    base_dir = "/tmp/Player Stats"
    print(f"Creating directory: {base_dir}")
    scrape_player_stats(driver, player_stats_links, base_dir)
    driver.quit()

    return {
        "statusCode": 200,
        "body": "Scraping and upload to S3 completed successfully!"
    }

def scrape_player_stats(driver, player_stats_links, base_dir):
    import re
    from bs4 import BeautifulSoup, Comment

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    s3 = boto3.client('s3')
    bucket_name = "finalyearproject2025"
    s3_prefix = "teamData"

    for stat_name, url in player_stats_links.items():
        print(f"Scraping data for {stat_name}...")
        time.sleep(1)
        driver.get(url)

        # Try clicking the "Squads" tab to trigger rendering
        try:
            squads_tab = WebDriverWait(driver, 25).until(
                EC.element_to_be_clickable((By.LINK_TEXT, "Squads"))
            )
            squads_tab.click()
            print("Clicked 'Squads' tab.")
        except Exception as e:
            print(f"Could not click 'Squads' tab: {e}")

        # Allow time for JavaScript to render the comment-embedded tables
        time.sleep(5)

        soup = BeautifulSoup(driver.page_source, "lxml")

        # Extract tables from within HTML comments
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        tables = []
        for c in comments:
            if "<table" in c:
                comment_soup = BeautifulSoup(c, "lxml")
                tables.extend(comment_soup.find_all("table"))

        print(f"Found {len(tables)} tables for {stat_name} (including inside comments).")

        stat_dir = os.path.join(base_dir, stat_name)
        os.makedirs(stat_dir, exist_ok=True)

        for i, table in enumerate(tables, start=1):
            thead = table.find("thead")
            tbody = table.find("tbody")

            table_id = table.get("id", "") or "_".join(table.get("class", [])) or f"table{i}"
            table_id = re.sub(r"[^\w\-]", "_", table_id)[:30]

            # Extract headers
            if thead:
                headers = [th.get_text(strip=True) for th in thead.find_all("th")]
            else:
                print(f"No <thead> found for table {i}. Attempting fallback.")
                if tbody and tbody.find("tr"):
                    first_row = tbody.find("tr")
                    headers = [td.get_text(strip=True) for td in first_row.find_all(["td", "th"])]
                else:
                    print(f"Skipping table {i} due to missing headers.")
                    continue

            if not tbody:
                print(f"Skipping table {i} since <tbody> is missing.")
                continue

            rows_data = []
            for tr in tbody.find_all("tr"):
                row_cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if any(cell != '' for cell in row_cells):
                    rows_data.append(row_cells)

            if not rows_data:
                print(f"Skipping table {i} as it contains no data.")
                continue

            # Align headers with rows
            min_len = min(len(headers), len(rows_data[0]))
            headers = headers[:min_len]
            rows_data = [row[:min_len] for row in rows_data]

            current_date = datetime.now().strftime("%Y-%m-%d")
            filename = f"{stat_name.replace(' ', '_')}_{table_id}_{current_date}.csv"
            file_path = os.path.join(stat_dir, filename)

            try:
                df = pd.DataFrame(rows_data, columns=headers)
                df.to_csv(file_path, index=False)
                print(f"Saved table {i} for {stat_name} as {filename}.")

                s3_key = f"{s3_prefix}/{stat_name}/{filename}"
                s3.upload_file(file_path, bucket_name, s3_key)
                print(f"Uploaded {file_path} to s3://{bucket_name}/{s3_key}")

            except Exception as e:
                print(f"Error processing table {i} ({table_id}): {e}")