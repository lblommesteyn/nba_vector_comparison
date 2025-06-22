import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
from datetime import datetime
import os
from io import StringIO

def scrape_per_game_stats(year=2024):
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
    
    # Add headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        soup = BeautifulSoup(response.content, "html.parser")
        
        table = soup.find('table', {'id': 'per_game_stats'})
        if not table:
            print(f"  × Could not find stats table for {year}")
            return None
            
        # Use StringIO to avoid deprecation warning
        df = pd.read_html(StringIO(str(table)))[0]
        
        # Clean up the data
        df = df[df['Player'] != 'Player']  # Remove header rows
        df = df[df['Player'].notnull()]  # Remove rows with null player names
        
        # Add season information
        season = f"{year-1}-{str(year)[-2:]}"
        df['Season'] = season
        
        # Add year to player name
        df['Player'] = df['Player'].apply(lambda x: f"{x} ({season})")
        
        # Convert numeric columns to float
        numeric_cols = df.columns.drop(['Player', 'Pos', 'Team', 'Season', 'Awards'])
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        return df
        
    except Exception as e:
        print(f"  × Error scraping {year}: {str(e)}")
        return None

def save_as_json(df, filename="player_stats.json"):
    if df is None or df.empty:
        print("  × No data to save")
        return
    
    # Format: { "LeBron James (2020-21)": [PTS, AST, REB, ...] }
    player_dict = {}
    
    for _, row in df.iterrows():
        # Get player name with season
        player_key = row["Player"]
        
        # Drop non-numeric columns
        drop_cols = ['Player', 'Pos', 'Team', 'Season', 'Awards']
        drop_cols = [col for col in drop_cols if col in df.columns]
        
        player_dict[player_key] = row.drop(drop_cols).tolist()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True) if os.path.dirname(filename) else None
    
    with open(filename, "w") as f:
        json.dump(player_dict, f, indent=2)

def save_as_csv(df, filename="player_stats.csv"):
    if df is None or df.empty:
        print("  × No data to save")
        return
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True) if os.path.dirname(filename) else None
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    # Create a directory for storing yearly data
    output_dir = "nba_stats_by_year"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # List to store all DataFrames for combined data
    all_players_dfs = []
    total_players = 0
    
    # Loop through years from 2025 back to 1995
    years = list(range(2025, 1994, -1))
    
    for year in years:
        season = f"{year-1}-{str(year)[-2:]}"
        print(f"Scraping data for the {season} NBA season...")
        
        stats_df = scrape_per_game_stats(year)
        
        if stats_df is not None and not stats_df.empty:
            # Save individual year data
            save_as_csv(stats_df, filename=f"{output_dir}/player_stats_{year}.csv")
            save_as_json(stats_df, filename=f"{output_dir}/player_stats_{year}.json")
            
            # Add to combined data
            all_players_dfs.append(stats_df)
            total_players += len(stats_df)
            
            print(f"  ✓ Scraped {len(stats_df)} players for {season}")
            
            # Add a small delay to avoid hitting the website too frequently
            time.sleep(2)  # Increased delay to be more polite
        else:
            print(f"  × Failed to scrape data for {season}")
    
    # Combine all years' data and save
    if all_players_dfs:
        try:
            combined_df = pd.concat(all_players_dfs, ignore_index=True)
            save_as_csv(combined_df, filename="all_players_stats.csv")
            save_as_json(combined_df, filename="all_players_stats.json")
            
            print(f"\nSuccessfully scraped and saved data for {len(all_players_dfs)} seasons and {total_players} player-seasons.")
            print(f"Combined data saved to all_players_stats.csv and all_players_stats.json")
        except Exception as e:
            print(f"Error saving combined data: {str(e)}")
    else:
        print("\nNo data was successfully scraped. Please check your internet connection and try again.")