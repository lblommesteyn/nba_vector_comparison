import { getAnalogy } from '../../../nba_player_search/app';

export default async function handler(req, res) {
  const { a, b, c, direction = 'a-b', top = 5 } = req.query;

  if (!a || !b) {
    return res.status(400).json({ error: 'Missing required parameters: a and b' });
  }

  try {
    // Call the backend's analogy function with the provided parameters
    const result = await getAnalogy(a, b, c, direction, parseInt(top));
    res.status(200).json(result);
  } catch (error) {
    console.error('Error in analogy API:', error);
    res.status(500).json({ error: 'Failed to compute analogy', details: error.message });
  }
}
