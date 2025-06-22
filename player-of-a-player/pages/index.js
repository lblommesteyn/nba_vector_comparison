import { useState, useEffect } from 'react';

export default function Home() {
  const [players, setPlayers] = useState([]);
  const [playerA, setPlayerA] = useState('');
  const [playerB, setPlayerB] = useState('');
  const [playerC, setPlayerC] = useState('');
  const [direction, setDirection] = useState('a-b');
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    // Fetch the list of players from the backend
    fetch('/api/players')
      .then(res => res.json())
      .then(data => setPlayers(data))
      .catch(err => setError('Failed to load players'));
  }, []);

  const handleSubmit = async () => {
    if (!playerA || !playerB) {
      setError('Please select at least two players');
      return;
    }

    setLoading(true);
    setError('');
    setResult('');
    
    try {
      // Extract base names (without year) for the API
      const baseA = playerA.split(' (')[0];
      const baseB = playerB.split(' (')[0];
      const baseC = playerC ? playerC.split(' (')[0] : '';
      
      // Build the query parameters
      const params = new URLSearchParams({
        a: baseA,
        b: baseB,
        direction,
        top: 5
      });
      
      if (baseC) {
        params.append('c', baseC);
      }

      const response = await fetch(`/api/analogy?${params}`);
      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
        return;
      }
      
      if (!data.results || data.results.length === 0) {
        setError('No results found. Try different players.');
        return;
      }

      // Format the result based on the direction
      const topResult = data.results[0];
      const playerName = topResult.player;
      const season = topResult.season || '';
      
      // Show the result in the format based on the selected direction
      if (direction === 'a-b') {
        // For "Who is the A of B?", show "The A of B is [result]"
        setResult(`The ${baseA} of ${baseB} is ${playerName}${season ? ` (${season})` : ''}`);
      } else {
        // For "Who is the B of A?", show "The B of A is [result]"
        setResult(`The ${baseB} of ${baseA} is ${playerName}${season ? ` (${season})` : ''}`);
      }
    } catch (err) {
      setError('Error fetching analogy. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '2rem', fontFamily: 'sans-serif', maxWidth: '800px', margin: '0 auto' }}>
      <h1>🏀 NBA Player Analogy</h1>
      <p>
        Find player analogies like "Who is the LeBron James of centers?"
      </p>

      <div style={{ margin: '2rem 0' }}>
        <div style={{ marginBottom: '2rem' }}>
          <div style={{ marginBottom: '0.5rem', fontWeight: '500' }}>Direction:</div>
          <div style={{ display: 'flex', gap: '1.5rem', marginBottom: '1rem' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <input
                type="radio"
                name="direction"
                value="a-b"
                checked={direction === 'a-b'}
                onChange={() => setDirection('a-b')}
                style={{ width: '1.2em', height: '1.2em' }}
              />
              <span>Who is the <strong>A</strong> of <strong>B</strong>?</span>
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <input
                type="radio"
                name="direction"
                value="b-a"
                checked={direction === 'b-a'}
                onChange={() => setDirection('b-a')}
                style={{ width: '1.2em', height: '1.2em' }}
              />
              <span>Who is the <strong>B</strong> of <strong>A</strong>?</span>
            </label>
          </div>
          <div style={{ color: '#666', fontSize: '0.9rem' }}>
            Example: "Who is the LeBron James of centers?" vs "Who is the center of LeBron James?"
          </div>
        </div>

        <div style={{ display: 'grid', gap: '1rem', marginBottom: '1rem' }}>
          <div>
            <label style={{ display: 'block', marginBottom: '0.5rem' }}>
              {direction === 'a-b' ? 'Player A (e.g., LeBron James)' : 'Player B (e.g., centers)'}
            </label>
            <select
              value={playerA}
              onChange={(e) => setPlayerA(e.target.value)}
              style={{ width: '100%', padding: '0.5rem' }}
            >
              <option value="">Select a player</option>
              {players.map((player) => (
                <option key={player} value={player}>
                  {player}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '0.5rem' }}>
              {direction === 'a-b' ? 'Player B (e.g., centers)' : 'Player A (e.g., LeBron James)'}
            </label>
            <select
              value={playerB}
              onChange={(e) => setPlayerB(e.target.value)}
              style={{ width: '100%', padding: '0.5rem' }}
            >
              <option value="">Select a player</option>
              {players.map((player) => (
                <option key={player} value={player}>
                  {player}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '0.5rem' }}>
              Optional: Player C (for more specific analogy)
            </label>
            <select
              value={playerC}
              onChange={(e) => setPlayerC(e.target.value)}
              style={{ width: '100%', padding: '0.5rem' }}
            >
              <option value="">Select a player (optional)</option>
              {players.map((player) => (
                <option key={player} value={player}>
                  {player}
                </option>
              ))}
            </select>
          </div>
        </div>

        <button
          onClick={handleSubmit}
          disabled={loading || !playerA || !playerB}
          style={{
            padding: '0.75rem 1.5rem',
            backgroundColor: loading || !playerA || !playerB ? '#ccc' : '#0070f3',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: loading || !playerA || !playerB ? 'not-allowed' : 'pointer',
            fontSize: '1rem',
            marginTop: '1rem',
          }}
        >
          {loading ? 'Finding...' : 'Find Analogy'}
        </button>
      </div>

      {error && (
        <div style={{ color: '#e53e3e', margin: '1rem 0', padding: '0.75rem', backgroundColor: '#fff5f5', borderRadius: '4px' }}>
          {error}
        </div>
      )}

      {result && !error && (
        <div style={{ marginTop: '2rem', padding: '1.5rem', backgroundColor: '#f7fafc', borderRadius: '8px' }}>
          <h2 style={{ margin: '0 0 1rem 0' }}>Result</h2>
          <p style={{ fontSize: '1.25rem', margin: 0 }}>{result}</p>
        </div>
      )}
    </div>
  );
}
