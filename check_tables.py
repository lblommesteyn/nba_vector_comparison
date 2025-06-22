import requests
from bs4 import BeautifulSoup

def check_table_ids(url, page_name):
    """Check what table IDs are available on a Basketball Reference page"""
    print(f"\nChecking {page_name}: {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        
        tables = soup.find_all('table')
        print(f"Found {len(tables)} tables:")
        
        for i, table in enumerate(tables):
            table_id = table.get('id', f'no-id-{i}')
            table_class = table.get('class', [])
            print(f"  {i+1}. ID: '{table_id}' | Class: {table_class}")
            
            # Show first few column headers
            headers_row = table.find('tr')
            if headers_row:
                headers = [th.get_text().strip() for th in headers_row.find_all(['th', 'td'])[:8]]
                print(f"     Headers: {headers}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    year = 2024
    
    # Check different stat pages
    pages = [
        (f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html", "Per Game Stats"),
        (f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html", "Advanced Stats"),
        (f"https://www.basketball-reference.com/leagues/NBA_{year}_shooting.html", "Shooting Stats"),
        (f"https://www.basketball-reference.com/leagues/NBA_{year}_per_poss.html", "Per 100 Possessions"),
    ]
    
    for url, name in pages:
        check_table_ids(url, name)
        print("-" * 60)
