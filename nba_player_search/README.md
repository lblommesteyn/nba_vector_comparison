# NBA Player Analogy Tool

A web application that finds analogous NBA players across a 30-year dataset (1995-2023) based on playing style, statistical profiles, and contextual similarities.

## Features

- **Comprehensive Dataset**: 30 years of NBA data (1995-2023) with 59,833 player-seasons and 2,698 unique players
- **Advanced Analogy Algorithm**: Finds players with similar styles in different contexts
- **Multiple Similarity Components**:
  - Style Transfer (35%): Projects player A's style onto player B's context
  - Archetype Similarity (30%): Finds players similar to player A's style
  - Physical Similarity (20%): Considers physical attributes
  - Relative Position (15%): Maintains player relationships
  - Context Similarity (10%): Considers similarity to player B's context
- **Filtering Options**:
  - Decade-based filtering (1990s, 2000s, 2010s, 2020s)
  - Custom year range filtering
  - Option to exclude input players
  - Option to exclude same-team players
- **Era Adjustment**: Accounts for different playing styles across decades

## Technical Details

- **Backend**: Flask, Pandas, NumPy, scikit-learn
- **Frontend**: Bootstrap 5, jQuery, Select2
- **Data Processing**: 51+ statistical features including advanced metrics
- **Algorithm**: Cosine similarity on unit-normalized feature vectors

## Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`

## Usage

1. Select two players to compare
2. Optionally set filters for decades, year range, and other options
3. Click "Find Analogies" to see results
4. View detailed component scores for each result

## Deployment

This application is configured for deployment on Render.com using the provided `render.yaml` configuration file.
