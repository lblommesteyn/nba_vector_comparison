import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
from datetime import datetime
import os
from io import StringIO

def scrape_comprehensive_stats(year=2024):
    """
    Scrape comprehensive NBA stats including:
    - Per game stats
    - Advanced stats 
    - Shooting stats
    - Player biographical info (height, weight, position, age, etc.)
    """
    print(f"Scraping comprehensive data for {year}...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    season = f"{year-1}-{str(year)[-2:]}"
    all_data = {}
    
    # 1. Per Game Stats (includes basic player info)
    print(f"  → Scraping per game stats...")
    per_game_df = scrape_table(f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html", 
                              'per_game_stats', headers, year)
    
    # 2. Advanced Stats
    print(f"  → Scraping advanced stats...")
    advanced_df = scrape_table(f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html", 
                              'advanced', headers, year)
    
    # 3. Shooting Stats
    print(f"  → Scraping shooting stats...")
    shooting_df = scrape_table(f"https://www.basketball-reference.com/leagues/NBA_{year}_shooting.html", 
                              'shooting', headers, year)
    
    # 4. Per 100 Possessions
    print(f"  → Scraping per 100 possession stats...")
    per100_df = scrape_table(f"https://www.basketball-reference.com/leagues/NBA_{year}_per_poss.html", 
                            'per_poss', headers, year)
    
    # 5. Player biographical data (height, weight, etc.)
    print(f"  → Scraping player biographical data...")
    bio_df = scrape_player_bio_data(f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html", 
                                   headers, year)
    
    if not any([per_game_df is not None, advanced_df is not None, shooting_df is not None, per100_df is not None]):
        print(f"  × No data found for {year}")
        return None
    
    # Merge all dataframes
    print(f"  → Merging datasets...")
    merged_df = merge_player_data(per_game_df, advanced_df, shooting_df, per100_df, bio_df, season)
    
    return merged_df

def scrape_table(url, table_id, headers, year):
    """Scrape a specific table from basketball-reference"""
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        
        # First try the expected table ID
        table = soup.find('table', {'id': table_id})
        if table:
            df = pd.read_html(StringIO(str(table)))[0]
        else:
            # If not found, look for any tables and print their IDs for debugging
            all_tables = soup.find_all('table')
            print(f"    × Could not find {table_id} table. Available tables:")
            for i, t in enumerate(all_tables[:5]):  # Show first 5 tables
                table_id_attr = t.get('id', f'no-id-{i}')
                print(f"      - {table_id_attr}")
            
            # Try to use the first table with stats data
            if all_tables:
                print(f"    → Trying first available table...")
                df = pd.read_html(StringIO(str(all_tables[0])))[0]
            else:
                return None
        
        # Clean up the data
        df = df[df['Player'] != 'Player']  # Remove header rows
        df = df[df['Player'].notnull()]  # Remove rows with null player names
        
        # Handle multi-level columns (flatten them)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
        
        # Clean column names
        df.columns = [col.replace('Unnamed: ', '').replace('_level_0', '') for col in df.columns]
        
        return df
        
    except Exception as e:
        print(f"    × Error scraping {table_id}: {str(e)}")
        return None

def scrape_player_bio_data(url, headers, year):
    """Scrape player biographical data including height, weight, position"""
    try:
        # First try to get data from the roster page which has height/weight
        roster_url = f"https://www.basketball-reference.com/leagues/NBA_{year}.html"
        response = requests.get(roster_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Look for team roster links to get comprehensive player data
        team_links = soup.find_all('a', href=True)
        team_urls = []
        for link in team_links:
            href = link['href']
            if f'/teams/' in href and f'/{year}.html' in href:
                team_urls.append(f"https://www.basketball-reference.com{href}")
        
        all_player_data = []
        
        # If we can't get team roster data, fall back to per-game stats
        if not team_urls:
            print(f"    → Using per-game stats for bio data...")
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Find the per game stats table which contains player info
            table = soup.find('table', {'id': 'per_game_stats'})
            if not table:
                # Try alternative table IDs
                table = soup.find('table')
            
            if table:
                df = pd.read_html(StringIO(str(table)))[0]
            else:
                print(f"    × Could not find player stats table for bio data")
                return None
            
            # Clean up the data
            df = df[df['Player'] != 'Player']  # Remove header rows
            df = df[df['Player'].notnull()]  # Remove rows with null player names
            
            # Handle multi-level columns (flatten them)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(col).strip() for col in df.columns.values]
            
            # Clean column names
            df.columns = [col.replace('Unnamed: ', '').replace('_level_0', '') for col in df.columns]
            
            # Select available biographical columns
            available_cols = df.columns.tolist()
            bio_cols = ['Player']
            
            # Add position if available
            if 'Pos' in available_cols:
                bio_cols.append('Pos')
            elif 'Position' in available_cols:
                bio_cols.append('Position')
                df = df.rename(columns={'Position': 'Pos'})
            
            # Add age if available
            if 'Age' in available_cols:
                bio_cols.append('Age')
            
            # Add team if available
            if 'Tm' in available_cols:
                bio_cols.append('Tm')
            elif 'Team' in available_cols:
                bio_cols.append('Team')
                df = df.rename(columns={'Team': 'Tm'})
            
            # Create a bio dataframe with available columns
            bio_df = df[bio_cols].copy()
            
        else:
            print(f"    → Scraping from {len(team_urls[:5])} team roster pages for detailed bio data...")
            
            # Sample a few teams to get player bio data (to avoid too many requests)
            for i, team_url in enumerate(team_urls[:5]):  # Limit to first 5 teams
                try:
                    time.sleep(1)  # Be respectful
                    response = requests.get(team_url, headers=headers)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.content, "html.parser")
                    
                    # Look for roster table
                    roster_table = soup.find('table', {'id': 'roster'})
                    if roster_table:
                        team_df = pd.read_html(StringIO(str(roster_table)))[0]
                        
                        # Clean up the data
                        if 'Player' in team_df.columns:
                            team_df = team_df[team_df['Player'] != 'Player']
                            team_df = team_df[team_df['Player'].notnull()]
                            all_player_data.append(team_df)
                            
                except Exception as e:
                    print(f"    × Error scraping team {i+1}: {str(e)}")
                    continue
            
            if all_player_data:
                # Combine all team data
                bio_df = pd.concat(all_player_data, ignore_index=True)
                bio_df = bio_df.drop_duplicates(subset=['Player'], keep='first')
                
                # Process height and weight if available
                if 'Ht' in bio_df.columns:
                    # Convert height to inches (e.g., "6-8" -> 80)
                    bio_df['Height_inches'] = bio_df['Ht'].apply(convert_height_to_inches)
                
                if 'Wt' in bio_df.columns:
                    # Convert weight to numeric
                    bio_df['Weight_lbs'] = pd.to_numeric(bio_df['Wt'], errors='coerce')
                
                # Calculate BMI if we have height and weight
                if 'Height_inches' in bio_df.columns and 'Weight_lbs' in bio_df.columns:
                    bio_df['BMI'] = (bio_df['Weight_lbs'] * 703) / (bio_df['Height_inches'] ** 2)
                
            else:
                print(f"    × Could not get roster data, falling back to basic bio data")
                return None
        
        # Process position data to create position categories
        if 'Pos' in bio_df.columns:
            bio_df['Primary_Pos'] = bio_df['Pos'].str.split('-').str[0]  # Get primary position
            
            # Create position group features
            bio_df['Is_Guard'] = bio_df['Primary_Pos'].isin(['PG', 'SG']).astype(int)
            bio_df['Is_Forward'] = bio_df['Primary_Pos'].isin(['SF', 'PF']).astype(int)
            bio_df['Is_Center'] = bio_df['Primary_Pos'].isin(['C']).astype(int)
            
            # Create more granular position features
            bio_df['Is_PG'] = (bio_df['Primary_Pos'] == 'PG').astype(int)
            bio_df['Is_SG'] = (bio_df['Primary_Pos'] == 'SG').astype(int)
            bio_df['Is_SF'] = (bio_df['Primary_Pos'] == 'SF').astype(int)
            bio_df['Is_PF'] = (bio_df['Primary_Pos'] == 'PF').astype(int)
            bio_df['Is_C'] = (bio_df['Primary_Pos'] == 'C').astype(int)
        
        print(f"    → Extracted bio data for {len(bio_df)} players with columns: {bio_df.columns.tolist()}")
        return bio_df
        
    except Exception as e:
        print(f"    × Error scraping player bio data: {str(e)}")
        return None

def convert_height_to_inches(height_str):
    """Convert height string like '6-8' to inches (80)"""
    try:
        if pd.isna(height_str) or height_str == '':
            return None
        
        if '-' in str(height_str):
            feet, inches = str(height_str).split('-')
            return int(feet) * 12 + int(inches)
        else:
            return None
    except:
        return None

def merge_player_data(per_game_df, advanced_df, shooting_df, per100_df, bio_df, season):
    """Merge all player data sources into a comprehensive dataset"""
    
    # Start with per_game as base (most complete player list)
    if per_game_df is not None:
        merged = per_game_df.copy()
        base_key = 'Player'
    else:
        print("  × No base dataset available")
        return None
    
    # Add season info
    merged['Season'] = season
    merged['Player_Season'] = merged['Player'].apply(lambda x: f"{x} ({season})")
    
    # Merge advanced stats
    if advanced_df is not None:
        print(f"    → Merging advanced stats ({len(advanced_df)} players)")
        # Select key advanced metrics
        adv_cols = ['Player', 'PER', 'TS%', 'USG%', 'OWS', 'DWS', 'WS', 'BPM', 'VORP']
        adv_cols = [col for col in adv_cols if col in advanced_df.columns]
        
        if len(adv_cols) > 1:  # More than just Player column
            merged = merged.merge(advanced_df[adv_cols], on='Player', how='left', suffixes=('', '_adv'))
    
    # Merge shooting stats
    if shooting_df is not None:
        print(f"    → Merging shooting stats ({len(shooting_df)} players)")
        # Select key shooting metrics
        shoot_cols = ['Player', '3P%', '2P%', 'FT%', '3PA', '2PA', 'FTA']
        # Handle potential column name variations
        for col in list(shoot_cols):
            if col not in shooting_df.columns:
                # Try common variations
                variations = [col.replace('%', ' %'), col.replace('P', 'PT'), f"{col}_"]
                for var in variations:
                    if var in shooting_df.columns:
                        shoot_cols[shoot_cols.index(col)] = var
                        break
        
        shoot_cols = [col for col in shoot_cols if col in shooting_df.columns]
        if len(shoot_cols) > 1:
            merged = merged.merge(shooting_df[shoot_cols], on='Player', how='left', suffixes=('', '_shoot'))
    
    # Merge per 100 possession stats
    if per100_df is not None:
        print(f"    → Merging per 100 possession stats ({len(per100_df)} players)")
        # Select key per-100 metrics (avoid duplicates with per-game)
        per100_cols = ['Player', 'ORtg', 'DRtg']  # Offensive/Defensive Rating
        per100_cols = [col for col in per100_cols if col in per100_df.columns]
        
        if len(per100_cols) > 1:
            merged = merged.merge(per100_df[per100_cols], on='Player', how='left', suffixes=('', '_per100'))
    
    # Merge biographical data
    if bio_df is not None:
        print(f"    → Merging biographical data ({len(bio_df)} players)")
        bio_cols = ['Player', 'Pos', 'Age', 'Tm', 'Primary_Pos', 'Is_Guard', 'Is_Forward', 'Is_Center', 'Is_PG', 'Is_SG', 'Is_SF', 'Is_PF', 'Is_C', 'Height_inches', 'Weight_lbs', 'BMI']
        bio_cols = [col for col in bio_cols if col in bio_df.columns]
        
        if len(bio_cols) > 1:
            merged = merged.merge(bio_df[bio_cols], on='Player', how='left', suffixes=('', '_bio'))
    
    # Clean up and prepare final dataset
    print(f"    → Final dataset: {len(merged)} players, {len(merged.columns)} features")
    
    # Convert Player_Season to be the main identifier
    merged['Player'] = merged['Player_Season']
    merged = merged.drop(['Player_Season'], axis=1)
    
    # Convert numeric columns
    numeric_cols = []
    for col in merged.columns:
        if col not in ['Player', 'Pos', 'Team', 'Season']:
            try:
                merged[col] = pd.to_numeric(merged[col], errors='coerce')
                numeric_cols.append(col)
            except:
                pass
    
    print(f"    → Converted {len(numeric_cols)} numeric columns")
    
    # Fill NaN values with 0 for numeric columns
    merged[numeric_cols] = merged[numeric_cols].fillna(0)
    
    return merged

def save_enhanced_data(df, filename="data/enhanced_players_stats.json"):
    """Save the enhanced dataset"""
    if df is None or df.empty:
        print("  × No data to save")
        return
    
    # Create the data structure: { "Player (Season)": [stat1, stat2, ...] }
    player_dict = {}
    
    # Get feature columns (exclude non-numeric)
    feature_cols = []
    for col in df.columns:
        if col not in ['Player', 'Pos', 'Team', 'Season'] and df[col].dtype in ['int64', 'float64']:
            feature_cols.append(col)
    
    print(f"  → Using {len(feature_cols)} features: {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")
    
    for _, row in df.iterrows():
        player_key = row["Player"]
        player_dict[player_key] = row[feature_cols].tolist()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, "w") as f:
        json.dump(player_dict, f, indent=2)
    
    print(f"  ✓ Saved {len(player_dict)} players to {filename}")
    
    # Also save feature names for reference
    feature_file = filename.replace('.json', '_features.json')
    with open(feature_file, "w") as f:
        json.dump(feature_cols, f, indent=2)
    
    print(f"  ✓ Saved feature list to {feature_file}")

if __name__ == "__main__":
    print("Enhanced NBA Stats Scraper - 30 Year Edition")
    print("=" * 50)
    
    # Create output directory
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # Checkpoint file to resume scraping if interrupted
    checkpoint_file = "data/scraper_checkpoint.json"
    
    # Scrape data from the last 30 years
    current_year = datetime.now().year
    start_year = current_year - 30
    years_to_scrape = list(range(current_year, start_year, -1))
    
    print(f"Scraping data from {start_year} to {current_year} ({len(years_to_scrape)} seasons)")
    
    # Check if we have a checkpoint to resume from
    completed_years = []
    all_data = []
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                completed_years = checkpoint.get('completed_years', [])
                print(f"Resuming from checkpoint. Already completed: {len(completed_years)} seasons.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    # Filter out years we've already processed
    years_remaining = [year for year in years_to_scrape if year not in completed_years]
    
    print(f"Processing {len(years_remaining)} remaining seasons...")
    
    # Track success and failures
    success_count = 0
    failure_count = 0
    
    # Process each year
    for i, year in enumerate(years_remaining):
        try:
            print(f"\nProcessing {year} season... ({i+1}/{len(years_remaining)})")
            df = scrape_comprehensive_stats(year)
            
            if df is not None and not df.empty:
                all_data.append(df)
                completed_years.append(year)
                success_count += 1
                print(f"  ✓ Successfully processed {year}: {len(df)} players")
                
                # Save checkpoint after each successful year
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        'completed_years': completed_years,
                        'last_updated': datetime.now().isoformat(),
                        'success_count': success_count,
                        'failure_count': failure_count
                    }, f)
            else:
                print(f"  × No data found for {year}")
                failure_count += 1
        except Exception as e:
            print(f"  × Error processing {year}: {str(e)}")
            failure_count += 1
        
        # Be more respectful to the server when scraping many years
        delay = 3 + (i % 5)  # Variable delay between 3-7 seconds to appear more human-like
        print(f"  → Waiting {delay} seconds before next request...")
        time.sleep(delay)
    
    # Try to load previously saved data if it exists
    existing_data_file = "data/enhanced_players_stats.json"
    existing_features_file = "data/enhanced_players_stats_features.json"
    existing_data = None
    
    if os.path.exists(existing_data_file) and os.path.exists(existing_features_file):
        try:
            print("\nLoading previously saved data...")
            with open(existing_data_file, 'r') as f:
                existing_data = json.load(f)
            with open(existing_features_file, 'r') as f:
                existing_features = json.load(f)
            print(f"  ✓ Loaded {len(existing_data)} existing player-seasons")
        except Exception as e:
            print(f"  × Error loading existing data: {str(e)}")
    
    if all_data:
        # Combine all years
        print(f"\nCombining data from {len(all_data)} seasons...")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # If we have existing data and new data has the same features, merge them
        if existing_data is not None:
            print("Merging with existing data...")
            # Convert existing data back to DataFrame for processing
            existing_df = pd.DataFrame()
            for player, stats in existing_data.items():
                row = pd.Series([player] + stats, index=['Player'] + existing_features)
                existing_df = pd.concat([existing_df, pd.DataFrame([row])], ignore_index=True)
            
            # Check for duplicate players in the new data
            existing_players = set(existing_df['Player'])
            new_players = set(combined_df['Player'])
            duplicates = existing_players.intersection(new_players)
            
            if duplicates:
                print(f"  • Found {len(duplicates)} duplicate players - using newest data")
                # Remove duplicates from existing data
                existing_df = existing_df[~existing_df['Player'].isin(duplicates)]
            
            # Ensure columns match before concatenating
            common_cols = list(set(existing_df.columns).intersection(set(combined_df.columns)))
            if len(common_cols) > 1:  # At least Player and some stats
                print(f"  • Merging on {len(common_cols)} common columns")
                final_df = pd.concat([existing_df[common_cols], combined_df[common_cols]], ignore_index=True)
                print(f"  ✓ Successfully merged datasets")
            else:
                print("  × Column mismatch between datasets, using only new data")
                final_df = combined_df
        else:
            final_df = combined_df
        
        # Calculate statistics
        total_players = len(final_df)
        unique_players = len(final_df['Player'].str.extract(r'(.+)\s+\(')
                             .dropna().iloc[:, 0].unique())
        num_features = len([col for col in final_df.columns 
                           if col not in ['Player', 'Pos', 'Team', 'Season']])
        years_covered = sorted(final_df['Player'].str.extract(r'\((\d{4})')
                               .dropna().iloc[:, 0].unique().astype(int))
        
        print(f"\nFinal dataset statistics:")
        print(f"  • Total player-seasons: {total_players}")
        print(f"  • Unique players: {unique_players}")
        print(f"  • Features: {num_features}")
        print(f"  • Years covered: {min(years_covered)}-{max(years_covered)} ({len(years_covered)} seasons)")
        
        # Save the enhanced dataset
        save_enhanced_data(final_df)
        
        # Generate detailed report
        print("\n✓ Enhanced 30-year NBA dataset complete!")
        print("\nThe comprehensive dataset includes:")
        print("  • Basic per-game stats (PTS, REB, AST, etc.)")
        print("  • Advanced metrics (PER, TS%, BPM, VORP, etc.)")
        print("  • Shooting efficiency (3P%, 2P%, FT%)")
        print("  • Pace-adjusted stats (ORtg, DRtg)")
        print("  • Usage and efficiency metrics")
        print("  • Player biographical data (position, age, team)")
        print("  • Physical metrics (height, weight, BMI)")
        print("  • Position indicators (guard/forward/center flags)")
        
        print("\nScraping summary:")
        print(f"  • Successfully processed: {success_count} seasons")
        print(f"  • Failed to process: {failure_count} seasons")
        print(f"  • Total seasons in dataset: {len(years_covered)}")
        
    else:
        print("\n× No new data was successfully scraped")
        
        # If we have existing data, remind the user
        if existing_data is not None:
            print(f"  • Your existing dataset with {len(existing_data)} player-seasons is still available")
            print(f"  • Run the app with the existing data to continue using it")
