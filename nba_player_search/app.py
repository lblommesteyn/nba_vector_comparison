import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pandas as pd
import re
from difflib import get_close_matches, SequenceMatcher
from typing import Union, List, Dict, Optional, Any

app = Flask(__name__)

# Load the player data with error handling for the larger dataset
try:
    print("Loading enhanced 30-year NBA dataset...")
    with open('data/enhanced_players_stats.json', 'r') as f:
        players_data = json.load(f)

    # Load feature names from the separate features file
    with open('data/enhanced_players_stats_features.json', 'r') as f:
        feature_names = json.load(f)
        
    print(f"Successfully loaded data for {len(players_data)} player-seasons spanning 30 years")
except Exception as e:
    print(f"Error loading player data: {str(e)}")
    print("Falling back to default data location...")
    # Try alternate location as fallback
    try:
        with open('../data/enhanced_players_stats.json', 'r') as f:
            players_data = json.load(f)

        with open('../data/enhanced_players_stats_features.json', 'r') as f:
            feature_names = json.load(f)
            
        print(f"Successfully loaded data from alternate location: {len(players_data)} player-seasons")
    except Exception as e:
        print(f"Could not load 30-year dataset, falling back to original dataset...")
        # Fall back to original dataset as last resort
        try:
            with open('data/players_stats.json', 'r') as f:
                players_data = json.load(f)

            with open('data/players_stats_features.json', 'r') as f:
                feature_names = json.load(f)
                
            print(f"Loaded original dataset with {len(players_data)} player-seasons")
        except Exception as e:
            print(f"Fatal error: Could not load any player data: {str(e)}")
            exit(1)

print(f"Loaded dataset with {len(feature_names)} features:")
print(f"Features: {feature_names[:10]}{'...' if len(feature_names) > 10 else ''}")

# Debug: Print first few items from players_data
print("First 3 items from players_data:")
for i, (player_name, stats_array) in enumerate(players_data.items()):
    if i >= 3:  # Only print first 3 for brevity
        break
    print(f"  {player_name}: {len(stats_array)} features")

# Initialize variables that will be used later
skipped = 0
df_rows = []
feature_names = []

# Standard NBA stat columns we expect to find
standard_feature_names = [
    'Rk', 'Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%',
    'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS',
    'PER', 'TS%', 'USG%', 'BPM', 'VORP', 'WS', 'ORtg', 'DRtg', 'Height_inches', 'Weight_lbs', 'BMI',
    'Is_Guard', 'Is_Forward', 'Is_Center', 'Is_PG', 'Is_SG', 'Is_SF', 'Is_PF', 'Is_C'
]

# Create a DataFrame from the loaded data
try:
    print("Creating DataFrame from player data...")
    # Check if the data is already in DataFrame format or needs conversion
    if isinstance(players_data, list):
        print("Data is in list format, converting to DataFrame...")
        df = pd.DataFrame(players_data)
        print(f"Successfully created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    else:
        # Track progress for large datasets
        total_players = len(players_data)
        progress_interval = max(1, total_players // 10)  # Show progress at 10% intervals
        
        df_rows = []
        # Get feature names from the first player's data
        first_player = next(iter(players_data.values()))
        
        # Map numeric indices to standard feature names if possible
        if len(first_player) <= len(standard_feature_names):
            feature_names = standard_feature_names[:len(first_player)]
        else:
            # If we have more features than expected, use numeric names
            feature_names = [f'stat_{i}' for i in range(len(first_player))]
        
        for i, (player_name, stats) in enumerate(players_data.items()):
            if i % progress_interval == 0:
                print(f"Processing player data: {i}/{total_players} ({i/total_players*100:.1f}%)")
            
            # Create a row with player name and stats
            row = [player_name] + stats
            df_rows.append(row)
        
        # Create DataFrame with player name and stats columns
        df = pd.DataFrame(df_rows, columns=['Player'] + feature_names)
        print(f"Successfully created DataFrame with {len(df)} rows and {len(df.columns)} columns")
except Exception as e:
    print(f"Error creating DataFrame: {str(e)}")

if skipped > 0:
    print(f"Total players skipped due to data mismatches: {skipped}")

if not df_rows:
    print("Error: No valid data rows found!")
    exit(1)
    
print(f"Successfully processed {len(df_rows)} player-seasons")

# Convert all column names to strings and clean them
df.columns = [str(col).strip() for col in df.columns]

print(f"DataFrame created with {len(df)} rows and {len(df.columns)} columns")
print("Available columns:", df.columns.tolist()[:10], "..." if len(df.columns) > 10 else "")

# Get the number of features from the first row (excluding the player name)
num_features = len(df_rows[0]) - 1 if df_rows else 0

# Extract year from player name (format: "Player Name (2023-24)")
print("\nExtracting year and base name from player names...")
df['Year'] = df['Player'].str.extract(r'\((\d{4}-\d{2})\)')
df['BaseName'] = df['Player'].str.replace(r'\s*\(\d{4}-\d{2}\)', '', regex=True).str.strip()

# Debug: Print DataFrame info
print("\nDataFrame info:")
print(f"Total rows: {len(df)}")
print("Columns:", df.columns.tolist())
print("Sample player names:", df['Player'].head(3).tolist())
print("Sample base names:", df['BaseName'].head(3).tolist())
print("Sample years:", df['Year'].head(3).tolist())

# Get the feature columns (all numeric columns except 'Player', 'Year', 'BaseName')
# Focus on style-defining metrics that capture player "flavor" and archetype
style_features = [
    # Shooting style and efficiency
    'FG%', '3P%', '2P%', 'eFG%', 'TS%', 'FT%',
    
    # Shot distribution (style indicators)
    '3P_Rate', '2P_Rate', 'FT_Rate',
    
    # Per-minute normalized stats (playing style independent of minutes)
    'PTS_per_min', 'AST_per_min', 'TRB_per_min', 'STL_per_min', 'BLK_per_min', 'TOV_per_min',
    
    # Efficiency ratios
    'AST_TO_ratio', 'STL_PF_ratio',
    
    # Advanced metrics (impact and efficiency)
    'PER', 'TS%', 'USG%', 'BPM', 'VORP', 'WS', 'ORtg', 'DRtg',
    
    # Physical and positional features (NEW)
    'Height_inches', 'Weight_lbs', 'BMI',
    'Is_Guard', 'Is_Forward', 'Is_Center',
    'Is_PG', 'Is_SG', 'Is_SF', 'Is_PF', 'Is_C',
    
    # Basic stats (fallback if advanced metrics aren't available)
    'PTS', 'AST', 'TRB', 'STL', 'BLK', 'TOV', 'PF',
    'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA',
    'MP', 'G', 'GS'
]

# Create ratio features that better capture playing style
print("\nCreating style-defining ratio features...")

# Check which base columns are available for calculations
required_cols = ['FGA', '3PA', '2PA', 'FTA', 'MP', 'PTS', 'AST', 'TRB', 'STL', 'BLK', 'TOV', 'PF']
available_cols = [col for col in required_cols if col in df.columns]
print(f"Available base columns for calculations: {available_cols}")

# Convert base columns to numeric first
for col in available_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Shot attempt ratios (style indicators) - only if base columns exist
if 'FGA' in df.columns and '3PA' in df.columns:
    df['3P_Rate'] = df['3PA'] / (df['FGA'] + 0.001)  # 3-point attempt rate
if 'FGA' in df.columns and '2PA' in df.columns:
    df['2P_Rate'] = df['2PA'] / (df['FGA'] + 0.001)  # 2-point attempt rate
if 'FGA' in df.columns and 'FTA' in df.columns:
    df['FT_Rate'] = df['FTA'] / (df['FGA'] + 0.001)  # Free throw attempt rate

# Per-minute rates (normalize for playing time) - only if MP exists
if 'MP' in df.columns:
    if 'PTS' in df.columns:
        df['PTS_per_min'] = df['PTS'] / (df['MP'] + 0.001)
    if 'AST' in df.columns:
        df['AST_per_min'] = df['AST'] / (df['MP'] + 0.001)
    if 'TRB' in df.columns:
        df['TRB_per_min'] = df['TRB'] / (df['MP'] + 0.001)
    if 'STL' in df.columns:
        df['STL_per_min'] = df['STL'] / (df['MP'] + 0.001)
    if 'BLK' in df.columns:
        df['BLK_per_min'] = df['BLK'] / (df['MP'] + 0.001)
    if 'TOV' in df.columns:
        df['TOV_per_min'] = df['TOV'] / (df['MP'] + 0.001)

# Efficiency ratios - only if base columns exist
if 'AST' in df.columns and 'TOV' in df.columns:
    df['AST_TO_ratio'] = df['AST'] / (df['TOV'] + 0.001)  # Assist to turnover ratio
if 'STL' in df.columns and 'PF' in df.columns:
    df['STL_PF_ratio'] = df['STL'] / (df['PF'] + 0.001)  # Steal to foul ratio

# List of calculated features that might have been created
calculated_features = [
    '3P_Rate', '2P_Rate', 'FT_Rate',
    'PTS_per_min', 'AST_per_min', 'TRB_per_min', 'STL_per_min', 'BLK_per_min', 'TOV_per_min',
    'AST_TO_ratio', 'STL_PF_ratio'
]

# Only include calculated features that were actually created
available_calculated_features = [col for col in calculated_features if col in df.columns]
print(f"Created calculated features: {available_calculated_features}")

# Use available style features, prioritizing ratios and efficiency metrics
available_style_features = [col for col in style_features if col in df.columns]
feature_columns = available_style_features + available_calculated_features

# Remove duplicates while preserving order
seen = set()
feature_columns = [col for col in feature_columns if not (col in seen or seen.add(col))]

# Remove any features with too many missing values (>50% missing)
feature_columns = [col for col in feature_columns if df[col].notna().sum() > len(df) * 0.5]

# If we have numeric column names and no features, use all numeric columns as features
if len(feature_columns) < 5:
    print("Warning: Not enough named features found. Using all numeric columns as features.")
    # Get all columns that could be converted to numeric
    numeric_cols = []
    for col in df.columns:
        if col not in ['Player', 'Year', 'BaseName']:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                numeric_cols.append(col)
            except:
                pass
    feature_columns = numeric_cols

# Ensure we have at least some features
if len(feature_columns) < 5:
    print("Warning: Still not enough features. Using basic stats as features.")
    # Create some basic stats columns if they don't exist
    basic_stats = ['PTS', 'AST', 'TRB', 'STL', 'BLK']
    for i, stat in enumerate(basic_stats):
        if stat not in df.columns and str(i) in df.columns:
            df[stat] = df[str(i)]
            feature_columns.append(stat)

print(f"\nFinal feature selection: {len(feature_columns)} features")
print(f"Available style features: {len(available_style_features)}")
print(f"Available calculated features: {len(available_calculated_features)}")

print("\n=== FEATURE CATEGORIES ===")
# Categorize features for better display
shooting_features = [f for f in feature_columns if any(x in f for x in ['%', 'TS', 'eFG'])]
physical_features = [f for f in feature_columns if any(x in f for x in ['Height', 'Weight', 'BMI'])]
position_features = [f for f in feature_columns if f.startswith('Is_')]
advanced_features = [f for f in feature_columns if any(x in f for x in ['PER', 'BPM', 'VORP', 'WS', 'USG', 'Rtg'])]
ratio_features = [f for f in feature_columns if 'per_min' in f or 'ratio' in f or 'Rate' in f]
other_features = [f for f in feature_columns if f not in shooting_features + physical_features + position_features + advanced_features + ratio_features]

if shooting_features:
    print(f"📊 Shooting & Efficiency ({len(shooting_features)}): {', '.join(shooting_features)}")
if physical_features:
    print(f"📏 Physical Attributes ({len(physical_features)}): {', '.join(physical_features)}")
if position_features:
    print(f"🏀 Position Indicators ({len(position_features)}): {', '.join(position_features)}")
if advanced_features:
    print(f"📈 Advanced Metrics ({len(advanced_features)}): {', '.join(advanced_features)}")
if ratio_features:
    print(f"⚖️  Ratio & Per-Minute ({len(ratio_features)}): {', '.join(ratio_features)}")
if other_features:
    print(f"📋 Other Features ({len(other_features)}): {', '.join(other_features)}")

print("=" * 50)

# Convert feature columns to numeric
print("\nConverting feature columns to numeric...")
for col in feature_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill any remaining NaN values with 0 (or another appropriate value)
print("Filling missing values...")
df[feature_columns] = df[feature_columns].fillna(0)

# Standardize the features once
print("Standardizing features...")
try:
    scaler = StandardScaler()
    player_features = scaler.fit_transform(df[feature_columns])
    print(f"Successfully standardized features with shape: {player_features.shape}")
    
    # Pre-compute unit vectors for cosine similarity (so dot product = cosine)
    print("Computing unit vectors...")
    feature_norms = np.linalg.norm(player_features, axis=1, keepdims=True)
    # Guard against zero division
    feature_norms[feature_norms == 0] = 1.0
    unit_vectors = player_features / feature_norms
    print(f"Successfully computed {len(unit_vectors)} unit vectors")
    
    # Verify unit vectors are valid
    if np.isnan(unit_vectors).any():
        print("WARNING: NaN values detected in unit vectors. Replacing with zeros.")
        unit_vectors = np.nan_to_num(unit_vectors)
    
    # Verify all unit vectors have norm ≈ 1
    norms = np.linalg.norm(unit_vectors, axis=1)
    if not np.allclose(norms, np.ones_like(norms), rtol=1e-5, atol=1e-5):
        print("WARNING: Some unit vectors do not have norm 1. Renormalizing.")
        unit_vectors = unit_vectors / np.linalg.norm(unit_vectors, axis=1, keepdims=True)
        
except Exception as e:
    print(f"Error standardizing features: {e}")
    # Fallback: use raw features if standardization fails
    print("Using raw features as fallback...")
    player_features = df[feature_columns].values
    
    # Normalize to unit vectors
    feature_norms = np.linalg.norm(player_features, axis=1, keepdims=True)
    feature_norms[feature_norms == 0] = 1.0
    unit_vectors = player_features / feature_norms

# Build quick lookup maps
name_to_idx = {p: i for i, p in enumerate(df['Player'])}
year_to_indices: dict[str, list[int]] = {}
for idx, yr in enumerate(df['Year']):
    year_to_indices.setdefault(yr, []).append(idx)


def find_similar_players(player_name: str, n: int = 5, filtered_df: Optional[pd.DataFrame] = None):
    """Return top *n* players most similar to *player_name*.

    Uses pre-computed unit vectors so similarity calculation is a fast
    dot-product.  If *filtered_df* is supplied (e.g. season-specific slice),
    only those rows are considered.
    """
    # Ensure the player exists
    player_idx = name_to_idx.get(player_name)
    if player_idx is None:
        return []

    # Determine candidate indices
    if filtered_df is not None:
        candidate_idx = filtered_df.index.to_numpy(dtype=int)
    else:
        candidate_idx = np.arange(len(df), dtype=int)

    # Remove the selected player from candidates
    candidate_idx = candidate_idx[candidate_idx != player_idx]
    if candidate_idx.size == 0:
        return []

    # Cosine similarity via dot-product because vectors are unit-normalised
    sims = unit_vectors[candidate_idx] @ unit_vectors[player_idx]

    # Top-n indices (descending similarity)
    top_order = sims.argsort()[-n:][::-1]
    top_indices = candidate_idx[top_order]

    results = [
        {
            "name": df.iloc[idx]["Player"],
            "similarity": float(sims[top_order[i]])
        }
        for i, idx in enumerate(top_indices)
    ]
    return results



@app.route('/analogy')
def analogy():
    a = request.args.get('a', '').strip()
    b = request.args.get('b', '').strip()
    c = request.args.get('c', '').strip()
    year_a = request.args.get('year_a', '').strip()
    year_b = request.args.get('year_b', '').strip()
    direction = request.args.get('direction', 'a-b').strip()  # 'a-b' or 'b-a'
    exclude_input_players = request.args.get('exclude_input_players', 'true').lower() == 'true'
    exclude_same_team = request.args.get('exclude_same_team', 'false').lower() == 'true'
    
    # New parameters for historical comparisons
    decade_filter = request.args.get('decade', '').strip()  # Filter by decade (e.g., '1990s')
    era_adjust = request.args.get('era_adjust', 'true').lower() == 'true'  # Adjust for era differences
    min_year = request.args.get('min_year', '').strip()  # Minimum year to include
    max_year = request.args.get('max_year', '').strip()  # Maximum year to include
    
    try:
        k = min(max(int(request.args.get('top', 5)), 1), 10)
    except ValueError:
        k = 5
    
    if not (a and b):
        return jsonify({'error': 'Both players A and B are required'}), 400
    
    # Filter dataframes by year/decade if specified
    df_a = df
    df_b = df
    search_df = df  # DataFrame for searching results
    
    if year_a:
        df_a = df[df['Year'] == year_a]
        if df_a.empty:
            return jsonify({'error': f'No data found for player A in year {year_a}'}), 404
    
    if year_b:
        df_b = df[df['Year'] == year_b]
        if df_b.empty:
            return jsonify({'error': f'No data found for player B in year {year_b}'}), 404
            
    # Apply decade filter if specified
    if decade_filter:
        print(f"Applying decade filter: {decade_filter}")
        if decade_filter == '1990s':
            # Filter for the 1990s
            search_df = df[df['Year'].str.contains('199', na=False)]
        elif decade_filter == '2000s':
            # Filter for the 2000s
            search_df = df[df['Year'].str.contains('200', na=False)]
        elif decade_filter == '2010s':
            # Filter for the 2010s
            search_df = df[df['Year'].str.contains('201', na=False)]
        elif decade_filter == '2020s':
            # Filter for the 2020s
            search_df = df[df['Year'].str.contains('202', na=False)]
        else:
            # Try to extract decade start from filter string
            decade_start = decade_filter[:3]  # e.g., '199' from '1990s'
            if decade_start.isdigit():
                # Filter for the specified decade
                search_df = df[df['Year'].str.contains(decade_start, na=False)]
        
        if search_df.empty:
            return jsonify({'error': f'No data found for decade {decade_filter}'}), 404
        print(f"Filtered to {len(search_df)} player-seasons from {decade_filter}")
    
    # Apply min/max year filters if specified
    if min_year and min_year.isdigit():
        print(f"Filtering for years >= {min_year}")
        try:
            # First try to extract year from 'Year' column (format: '2023-24')
            year_extracted = search_df['Year'].str.extract(r'(\d{4})', expand=False)
            if not year_extracted.empty:
                search_df = search_df[year_extracted.astype(int, errors='ignore') >= int(min_year)]
            else:
                # Fallback: just check if the year string contains the min_year
                search_df = search_df[search_df['Year'].str.contains(min_year, na=False)]
                
            if search_df.empty:
                return jsonify({'error': f'No data found after year {min_year}'}), 404
            print(f"Found {len(search_df)} player-seasons after year {min_year}")
        except Exception as e:
            print(f"Error filtering by min_year: {e}")
            # Fallback to simple string comparison
            search_df = search_df[search_df['Year'].str.contains(min_year, na=False)]
    
    if max_year and max_year.isdigit():
        print(f"Filtering for years <= {max_year}")
        try:
            # First try to extract year from 'Year' column (format: '2023-24')
            year_extracted = search_df['Year'].str.extract(r'(\d{4})', expand=False)
            if not year_extracted.empty:
                search_df = search_df[year_extracted.astype(int, errors='ignore') <= int(max_year)]
            else:
                # Simple filtering based on string comparison
                # This is less accurate but better than nothing
                search_df = search_df[search_df['Year'].astype(str).str[:4].astype(int, errors='ignore') <= int(max_year)]
                
            if search_df.empty:
                return jsonify({'error': f'No data found before year {max_year}'}), 404
            print(f"Found {len(search_df)} player-seasons before year {max_year}")
        except Exception as e:
            print(f"Error filtering by max_year: {e}")
            # Fallback to simple string comparison
            search_df = search_df[search_df['Year'].str.contains(max_year[:2], na=False)]
    
    # Find player indices
    try:
        # Look for player A in the filtered dataframe
        player_a_matches = df_a[df_a['Player'] == a]
        if player_a_matches.empty:
            return jsonify({'error': f'Player A "{a}" not found' + (f' in year {year_a}' if year_a else '')}), 404
        
        # Look for player B in the filtered dataframe
        player_b_matches = df_b[df_b['Player'] == b]
        if player_b_matches.empty:
            return jsonify({'error': f'Player B "{b}" not found' + (f' in year {year_b}' if year_b else '')}), 404
        
        # Get the indices from the original dataframe
        ia = player_a_matches.index[0]
        ib = player_b_matches.index[0]
        
    except (KeyError, IndexError):
        return jsonify({'error': 'One or more players not found'}), 404
    
    # Get team information for filtering
    team_a = df.iloc[ia]['Player'].split('(')[0].strip() if exclude_same_team else None
    team_b = df.iloc[ib]['Player'].split('(')[0].strip() if exclude_same_team else None
    
    # Handle direction toggle - swap A and B if needed
    if direction == 'b-a':
        a, b = b, a
        ia, ib = ib, ia
        year_a, year_b = year_b, year_a
        team_a, team_b = team_b, team_a
    
    # Get base names (without years) for input players to exclude all their seasons
    base_name_a = df.iloc[ia]['BaseName'] if exclude_input_players else None
    base_name_b = df.iloc[ib]['BaseName'] if exclude_input_players else None
    
    print(f"Debug: exclude_input_players={exclude_input_players}, exclude_same_team={exclude_same_team}")
    print(f"Debug: Player A index={ia}, Player B index={ib}")
    print(f"Debug: base_name_a={base_name_a}, base_name_b={base_name_b}")
    print(f"Debug: team_a={team_a}, team_b={team_b}")
    
    # Enhanced analogy algorithm for better "flavor" matching:
    # "Who is the LeBron of guards?" should find elite guards with LeBron-like dominance
    # "Who is the Steph Curry of big men?" should find big men with Curry-like shooting/efficiency
    
    # Step 1: Analyze the relationship between A and B
    vec_a = unit_vectors[ia]
    vec_b = unit_vectors[ib]
    
    # Calculate the "style difference" vector from B to A
    style_difference = vec_a - vec_b
    
    # Step 2: For each candidate, evaluate how well they embody A's style relative to B's archetype
    candidate_scores = []
    
    # Use the filtered search_df indices instead of all indices
    search_indices = search_df.index.tolist()
    total_candidates = len(search_indices)
    print(f"Evaluating {total_candidates} candidates after filtering")
    
    # Process candidates in batches for large datasets
    batch_size = min(5000, total_candidates)
    processed = 0
    
    for candidate_idx in search_indices:
        candidate_player = df.iloc[candidate_idx]['Player']
        candidate_base_name = df.iloc[candidate_idx]['BaseName']
        candidate_year = df.iloc[candidate_idx]['Year']
        
        # Skip if this is any season of an input player and we want to exclude them
        if exclude_input_players and (candidate_base_name == base_name_a or candidate_base_name == base_name_b):
            continue
            
        # Skip if this is same team and we want to exclude same team
        if exclude_same_team and (candidate_base_name == team_a or candidate_base_name == team_b):
            continue
            
        # Track progress for large datasets
        processed += 1
        if processed % batch_size == 0:
            print(f"  → Processed {processed}/{total_candidates} candidates ({processed/total_candidates*100:.1f}%)")
        
        vec_candidate = unit_vectors[candidate_idx]
        
        # Method 1: Enhanced Style Transfer Score
        # How well does this candidate represent A's style applied to B's context?
        # Use a weighted style difference to better capture the essence of A's style
        
        # Calculate style importance weights based on variance across the dataset
        # Features with higher variance are more distinctive for a player's style
        style_importance = np.var(player_features, axis=0)
        style_importance = style_importance / np.sum(style_importance)  # Normalize
        
        # Apply weighted style difference
        weighted_style_diff = style_difference * style_importance
        projected_style = vec_b + weighted_style_diff  # What B would look like with A's style
        
        # Normalize projected style vector
        projected_style_norm = np.linalg.norm(projected_style)
        if projected_style_norm > 0:
            projected_style = projected_style / projected_style_norm
            
        style_transfer_score = np.dot(vec_candidate, projected_style)
        
        # Method 2: Archetype Similarity
        # How similar is the candidate to A in terms of playing style?
        archetype_similarity = np.dot(vec_candidate, vec_a)
        
        # Method 3: Context Similarity  
        # How similar is the candidate's context to B's context?
        context_similarity = np.dot(vec_candidate, vec_b)
        
        # Method 4: Enhanced Relative Position Score
        # Does the candidate occupy a similar relative position to A as A does to B?
        # Use cosine similarity for more robust comparison
        candidate_to_b_diff = vec_candidate - vec_b
        a_to_b_diff = vec_a - vec_b
        
        # Normalize the difference vectors
        c_b_norm = np.linalg.norm(candidate_to_b_diff)
        a_b_norm = np.linalg.norm(a_to_b_diff)
        
        if c_b_norm > 0 and a_b_norm > 0:
            # Use cosine similarity between the difference vectors
            relative_position_score = np.dot(candidate_to_b_diff, a_to_b_diff) / (c_b_norm * a_b_norm)
        else:
            relative_position_score = 0.0
        
        # Method 5: Physical Similarity Score (NEW)
        # Consider physical attributes for better archetype matching
        physical_similarity = 0.0
        physical_weight = 0.0
        
        # Height similarity
        if 'Height_inches' in df.columns:
            height_a = df.iloc[ia]['Height_inches']
            height_b = df.iloc[ib]['Height_inches'] 
            height_candidate = df.iloc[candidate_idx]['Height_inches']
            
            if pd.notna(height_a) and pd.notna(height_b) and pd.notna(height_candidate):
                # Calculate how similar candidate's height relationship to B is compared to A's relationship to B
                height_diff_ab = abs(height_a - height_b)
                height_diff_cb = abs(height_candidate - height_b)
                height_similarity = 1.0 - min(abs(height_diff_ab - height_diff_cb) / 12.0, 1.0)  # Normalize by 12 inches
                physical_similarity += height_similarity * 0.4
                physical_weight += 0.4
        
        # Enhanced Position and Role Similarity
        position_similarity = 0.0
        role_similarity = 0.0
        
        if all(col in df.columns for col in ['Is_Guard', 'Is_Forward', 'Is_Center', 'Is_PG', 'Is_SG', 'Is_SF', 'Is_PF', 'Is_C']):
            # Get position group vectors (Guard, Forward, Center)
            pos_group_a = [df.iloc[ia]['Is_Guard'], df.iloc[ia]['Is_Forward'], df.iloc[ia]['Is_Center']]
            pos_group_b = [df.iloc[ib]['Is_Guard'], df.iloc[ib]['Is_Forward'], df.iloc[ib]['Is_Center']]
            pos_group_candidate = [df.iloc[candidate_idx]['Is_Guard'], df.iloc[candidate_idx]['Is_Forward'], df.iloc[candidate_idx]['Is_Center']]
            
            # Get specific position vectors (PG, SG, SF, PF, C)
            specific_pos_a = [df.iloc[ia]['Is_PG'], df.iloc[ia]['Is_SG'], df.iloc[ia]['Is_SF'], df.iloc[ia]['Is_PF'], df.iloc[ia]['Is_C']]
            specific_pos_b = [df.iloc[ib]['Is_PG'], df.iloc[ib]['Is_SG'], df.iloc[ib]['Is_SF'], df.iloc[ib]['Is_PF'], df.iloc[ib]['Is_C']]
            specific_pos_candidate = [
                df.iloc[candidate_idx]['Is_PG'], 
                df.iloc[candidate_idx]['Is_SG'], 
                df.iloc[candidate_idx]['Is_SF'], 
                df.iloc[candidate_idx]['Is_PF'], 
                df.iloc[candidate_idx]['Is_C']
            ]
            
            # Calculate position group similarity (Guard/Forward/Center) with jaccard similarity
            # This better handles multi-position players
            sum_pos_group = sum(p1 and p2 for p1, p2 in zip(pos_group_candidate, pos_group_b))
            union_pos_group = sum(p1 or p2 for p1, p2 in zip(pos_group_candidate, pos_group_b))
            pos_group_sim = sum_pos_group / max(1, union_pos_group)
            
            # Calculate specific position similarity with weighted matching
            # Primary positions get higher weight
            primary_pos_candidate = np.argmax(specific_pos_candidate) if max(specific_pos_candidate) > 0 else -1
            primary_pos_b = np.argmax(specific_pos_b) if max(specific_pos_b) > 0 else -1
            
            # Base similarity from dot product
            specific_pos_sim = np.dot(specific_pos_candidate, specific_pos_b)
            
            # Boost if primary positions match
            if primary_pos_candidate == primary_pos_b and primary_pos_candidate >= 0:
                specific_pos_sim = 0.7 + (specific_pos_sim * 0.3)  # At least 0.7 if primary positions match
            
            # Role preservation: How well does candidate maintain A's role while matching B's position
            # Calculate role vector from A (what makes A special)
            role_vector_a = vec_a - np.mean(unit_vectors, axis=0)  # A's deviation from average player
            role_vector_candidate = vec_candidate - np.mean(unit_vectors, axis=0)  # Candidate's deviation
            
            # How similar is candidate's role to A's role
            role_similarity = np.dot(role_vector_candidate, role_vector_a)
            role_similarity = (role_similarity + 1) / 2  # Rescale from [-1,1] to [0,1]
            
            # Position similarity combines position matching with role preservation
            position_similarity = (specific_pos_sim * 0.4) + (pos_group_sim * 0.3) + (role_similarity * 0.3)
            
            # Apply a position-based filter: if the candidate is in a completely different position group, heavily penalize
            if max(pos_group_candidate) > 0 and max(pos_group_b) > 0 and np.argmax(pos_group_candidate) != np.argmax(pos_group_b):
                position_similarity *= 0.3  # 70% penalty for crossing position groups
                
            physical_similarity += position_similarity * 0.6
            physical_weight += 0.6
        
        # Normalize physical similarity
        if physical_weight > 0:
            physical_similarity = physical_similarity / physical_weight
        else:
            physical_similarity = 0.5  # Neutral if no physical data
        
        # Combine scores with enhanced position/role awareness
        # Handle direction toggle
        if direction == 'b-a':
            # Swap A and B for the query description
            a, b = b, a
            year_a, year_b = year_b, year_a
        
        # Dynamic weights based on data availability and query type
        # Removed position from weights as requested
        base_weights = {
            'style_transfer': 0.35,  # Increased for better style matching
            'archetype': 0.30,      # Increased importance
            'physical': 0.20,       # Maintained for balance
            'relative': 0.15,       # Maintained
            'context': 0.10         # Maintained for context relevance
        }
        
        # Position-based weight adjustments removed as requested
        # No position-specific adjustments needed anymore
        
        # Calculate base score components - position removed as requested
        components = {
            'style_transfer': style_transfer_score,
            'archetype': archetype_similarity,
            'physical': physical_similarity,
            'relative': relative_position_score,
            'context': context_similarity
        }
        
        # Position-based scaling removed as requested
        # No position-specific adjustments needed anymore
        
        # Calculate final score - position-based scaling removed
        analogy_score = 0
        total_weight = 0
        
        # Track individual component scores for analysis
        component_scores = {}
        
        for component, weight in base_weights.items():
            # No position-based scaling anymore
            scaled_weight = weight
            component_score = components[component] * scaled_weight
            analogy_score += component_score
            total_weight += scaled_weight
            component_scores[component] = {
                'raw': float(components[component]),
                'weight': float(scaled_weight),
                'weighted': float(component_score)
            }
            
        # Normalize the score
        if total_weight > 0:
            analogy_score /= total_weight
            
        # Store candidate with all score components
        candidate_scores.append((candidate_idx, analogy_score, component_scores))
    
    # Sort by analogy score and get top k
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates = candidate_scores[:k]
    
    # Determine query description based on direction
    if direction == 'b-a':
        query_desc = f"Who is the {b.split('(')[0].strip()} of {a.split('(')[0].strip()}?"
    else:  # default 'a-b'
        query_desc = f"Who is the {a.split('(')[0].strip()} of {b.split('(')[0].strip()}?"
    
    results = []
    for idx, score, components in top_candidates:
        player_data = df.iloc[idx]
        # Safely get position data with defaults
        position_data = {
            'primary': player_data.get('Pos', 'N/A'),
            'is_guard': bool(player_data.get('Is_Guard', 0) or 
                          ('PG' in str(player_data.get('Pos', '')) or 
                           'SG' in str(player_data.get('Pos', '')))),
            'is_forward': bool(player_data.get('Is_Forward', 0) or 
                            ('SF' in str(player_data.get('Pos', '')) or 
                             'PF' in str(player_data.get('Pos', '')))),
            'is_center': bool(player_data.get('Is_Center', 0) or 
                           'C' in str(player_data.get('Pos', '')))
        }
        
        # Estimate height/weight from other stats if not available
        height = player_data.get('Height_inches', 
                               player_data.get('height', 0) or 
                               (78 if position_data['is_center'] else 76))
        
        weight = player_data.get('Weight_lbs', 
                               player_data.get('weight', 0) or 
                               (240 if position_data['is_center'] else 220))
        
        # Calculate BMI if not available
        bmi = player_data.get('BMI', 0)
        if not bmi and height > 0:
            bmi = (float(weight) / (float(height) ** 2)) * 703  # Convert to imperial BMI
        
        results.append({
            'name': player_data['Player'],
            'similarity': float(score),
            'season': player_data.get('Year', 'N/A'),
            'scores': components,
            'position': position_data
            # Removed physical information (height, weight, BMI) as requested
        })
    
    # Calculate era statistics if era adjustment was enabled
    era_stats = {}
    if era_adjust:
        # Always include the 2020s decade regardless of results
        eras = {'2020s'}
        
        # Add any other eras from results
        for r in results:
            if 'season' in r and r['season']:
                # Extract the first 4 digits from the season string
                year_match = re.search(r'\d{4}', str(r['season']))
                if year_match:
                    year = year_match.group(0)
                    decade = year[:3] + '0s'  # Convert year to decade (e.g., 2020s)
                    eras.add(decade)
        
        # Calculate average stats by era
        print(f"Calculating stats for eras: {eras}")
        for era in eras:
            # Find all players from this decade
            decade_prefix = era[:3]  # e.g., '202' for 2020s
            decade_players = df[df['Year'].astype(str).str.contains(decade_prefix, na=False)]
            count = len(decade_players)
            
            # Calculate average points per game for this decade
            if 'PTS' in df.columns and count > 0:
                avg_pts = decade_players['PTS'].mean()
            else:
                avg_pts = 0
            
            # Calculate average pace for this decade (if available)
            if 'Pace' in df.columns and count > 0:
                avg_pace = decade_players['Pace'].mean()
            else:
                avg_pace = 0
            
            era_stats[era] = {
                'avg_pts': float(avg_pts),
                'avg_pace': float(avg_pace),
                'count': count
            }
            
            print(f"Era stats for {era}: {count} players, {avg_pts:.1f} pts, {avg_pace:.1f} pace")
    
    response = {
        'a': a,
        'b': b,
        'year_a': year_a,
        'year_b': year_b,
        'direction': direction,
        'query': query_desc,
        'filters': {
            'exclude_input_players': exclude_input_players,
            'exclude_same_team': exclude_same_team,
            'decade': decade_filter,
            'min_year': min_year,
            'max_year': max_year,
            'era_adjust': era_adjust
        },
        'score_weights': base_weights,
        'era_stats': era_stats,
        'results': results
    }
    
    return jsonify(response)

@app.route('/search_players')
def search_players():
    query = request.args.get('q', '').lower().strip()
    year = request.args.get('year', '')
    
    if not query:
        return jsonify([])
    
    # Filter by year if specified
    filtered = df
    if year:
        # Handle both exact year match and year range formats (e.g., '2023-24')
        filtered = df[df['Year'].str.contains(year, na=False)]
        
        # If no results, try to match just the first part of the year
        if filtered.empty and year.isdigit():
            filtered = df[df['Year'].str.startswith(year, na=False)]
    
    # Get unique player names and perform fuzzy matching
    names_series = filtered['Player'].str.lower()
    scores = names_series.apply(lambda name: SequenceMatcher(None, query, name).ratio())
    
    # Add scores and filter
    filtered = filtered.assign(_score=scores)
    top = filtered[filtered['_score'] > 0].sort_values('_score', ascending=False).head(20)
    
    print(f"Search for '{query}' returned {len(top)} results")
    if len(top) > 0:
        print(f"Top results: {top['Player'].head(3).tolist()}")
    
    return jsonify(top['Player'].tolist())

@app.route('/')
def index():
    # Get unique years for the year filter
    years = sorted(df['Year'].dropna().unique(), reverse=True)
    
    # Get decades for decade filter
    decades = []
    for year in years:
        if year and len(year) >= 4:
            decade = year[:3] + '0s'
            if decade not in decades:
                decades.append(decade)
    decades = sorted(decades, reverse=True)
    
    # Get stats about the dataset
    total_players = len(df)
    unique_players = len(df['BaseName'].unique())
    year_range = f"{min(years) if years else 'N/A'} - {max(years) if years else 'N/A'}"
    
    return render_template('index.html', 
                           years=years, 
                           decades=decades,
                           stats={
                               'total_players': total_players,
                               'unique_players': unique_players,
                               'year_range': year_range,
                               'features': len(feature_columns)
                           })

@app.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html')

@app.route('/find_similar', methods=['GET'])
def find_similar():
    try:
        player_name = request.args.get('player', '').strip()
        
        # Validate player name
        if not player_name:
            return jsonify({'error': 'Player name is required'}), 400
        
        # Get and validate number of results
        try:
            n = min(max(1, int(request.args.get('n', 5))), 20)  # Clamp between 1 and 20
        except (TypeError, ValueError):
            return jsonify({'error': 'Invalid number of results requested'}), 400
        
        year = request.args.get('year', '').strip()
        
        # Filter by year if specified
        filtered_df = df
        if year:
            filtered_df = filtered_df[filtered_df['Year'] == year]
        
        # Find the exact player match
        player_matches = filtered_df[filtered_df['Player'] == player_name]
        if player_matches.empty:
            return jsonify({'error': f'Player "{player_name}" not found' + (f' in {year}' if year else '')}), 404
        
        # Get similar players
        similar_players = find_similar_players(player_name, n, filtered_df)
        
        if not similar_players:
            return jsonify({
                'player': player_name,
                'similar_players': [],
                'message': 'No similar players found. Try adjusting your search criteria.'
            })
        
        return jsonify({
            'player': player_name,
            'similar_players': similar_players,
            'total_players_searched': len(filtered_df)
        })
        
    except Exception as e:
        error_msg = f'Error finding similar players: {str(e)}'
        print(error_msg)
        return jsonify({'error': 'An unexpected error occurred'}), 500

def get_players():
    """Get the list of all players in the dataset."""
    return df['Player'].tolist()

def get_analogy(a, b, c=None, direction='a-b', top=5):
    """Get player analogy results.
    
    Args:
        a: Player A name
        b: Player B name
        c: Optional Player C name for more specific analogy
        direction: 'a-b' or 'b-a' for analogy direction
        top: Number of results to return
        
    Returns:
        List of dicts with player and similarity score
    """
    # Convert to request args format for the existing analogy function
    args = {
        'a': a,
        'b': b,
        'c': c,
        'direction': direction,
        'top': str(top)
    }
    
    # Call the existing analogy function
    with app.test_request_context('/analogy', query_string=args):
        response = analogy()
        
    # Parse the response
    if isinstance(response, str):
        # If it's a string, it's an error message
        return {'error': response}
    
    # Convert the response to a list of dicts
    results = []
    for item in response.get('results', []):
        results.append({
            'player': item['player'],
            'season': item.get('season', ''),
            'similarity': item.get('similarity', 0),
            'scores': {
                'style_transfer': item.get('style_transfer_score', 0),
                'archetype': item.get('archetype_similarity', 0),
                'physical': item.get('physical_similarity', 0),
                'position': item.get('position_similarity', 0),
                'context': item.get('context_similarity', 0)
            }
        })
    
    return results

# Entry point for local development
if __name__ == '__main__':
    app.run(debug=True)
    
# For Vercel serverless deployment
app.config['JSON_SORT_KEYS'] = False  # Preserve order of JSON keys
