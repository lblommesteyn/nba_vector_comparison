import { getPlayers } from '../../../nba_player_search/app';

export default function handler(req, res) {
  try {
    const players = getPlayers();
    res.status(200).json(players);
  } catch (error) {
    console.error('Error fetching players:', error);
    res.status(500).json({ error: 'Failed to fetch players' });
  }
}
