import boto3
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import Comment
import time
import pandas as pd
from datetime import datetime
from io import StringIO

# === S3 Configuration ===
s3 = boto3.client("s3")
S3_BUCKET = "finalyearproject2025"
S3_PREFIX = "playerData/"

# === FBref URLs ===
PLAYER_STATS_URLS = {
    "Standard Stats": "https://fbref.com/en/comps/9/stats/Premier-League-Stats",
    "Advanced GK Stats": "https://fbref.com/en/comps/9/keepersadv/Premier-League-Stats",
    "Shooting": "https://fbref.com/en/comps/9/shooting/Premier-League-Stats",
    "Passing": "https://fbref.com/en/comps/9/passing/Premier-League-Stats",
    "Goal and Shot Creation": "https://fbref.com/en/comps/9/gca/Premier-League-Stats",
    "Possession": "https://fbref.com/en/comps/9/possession/Premier-League-Stats",
    "Miscellaneous Stats": "https://fbref.com/en/comps/9/misc/Premier-League-Stats",
}

# === Selenium Setup ===
def setup_driver():
    chrome_options = Options()
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
    return driver

# === Upload to S3 ===
def upload_df_to_s3(df, stat_name, table_type):
    clean_stat_name = stat_name.replace(" ", "_")
    date_str = datetime.now().strftime("%Y-%m-%d")
    key = f"{S3_PREFIX}{stat_name}/{table_type}_{clean_stat_name}_{date_str}.csv"

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=csv_buffer.getvalue())
    print(f"✅ Uploaded to s3://{S3_BUCKET}/{key}")

# === Parse and Save Table ===
def save_table_to_s3(table, stat_name, table_type):
    header_rows = table.find("thead").find_all("tr")
    headers = [th.text.strip() for th in header_rows[-1].find_all("th")]

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        if tr.get("class") and "thead" in tr.get("class"):
            continue
        row_data = []
        for td in tr.find_all(["td", "th"]):
            if td.find("a"):
                row_data.append(td.find("a").text.strip())
            else:
                row_data.append(td.text.strip())
        if row_data:
            # Pad to match headers
            row_data += [""] * (len(headers) - len(row_data))
            rows.append(row_data)

    if rows:
        df = pd.DataFrame(rows, columns=headers)
        upload_df_to_s3(df, stat_name, table_type)
    else:
        print(f"⚠️ No valid rows for {stat_name} - {table_type}")

# === Main Scraper ===
def scrape_player_stats():
    driver = setup_driver()
    for stat_name, url in PLAYER_STATS_URLS.items():
        print(f"🔎 Scraping {stat_name} from {url}")
        driver.get(url)
        try:
            squads_tab = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.LINK_TEXT, "Squads"))
            )
            squads_tab.click()
            print("🖱️ Clicked 'Squads' tab to trigger rendering")
        except Exception as e:
            print(f"⚠️ Could not click 'Squads' tab: {e}")

        time.sleep(15)  # Wait for JS to inject comment tables

        # Sleep instead of WebDriverWait (like teamScrapper does)
        
        time.sleep(15)  # give JS time to render

        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Look inside comment blocks (FBref hides tables in comments)
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        tables = []
        for c in comments:
            comment_soup = BeautifulSoup(c, "html.parser")
            tables.extend(comment_soup.find_all("table"))

        print(f"🔢 Found {len(tables)} total tables in {stat_name}")

        squad_table = None
        player_table = None
        for table in tables:
            headers = [th.text.strip() for th in table.find("thead").find_all("th")]
            if "Squad" in headers and not squad_table:
                squad_table = table
            elif "Player" in headers and not player_table:
                player_table = table

        if squad_table:
            save_table_to_s3(squad_table, stat_name, "Squad")
        if player_table:
            save_table_to_s3(player_table, stat_name, "Player")

    driver.quit()


# === Lambda Entry Point ===
def lambda_handler(event, context):
    scrape_player_stats()
    return {"statusCode": 200, "body": "✅ FBref player stats scraped and uploaded to S3."}
