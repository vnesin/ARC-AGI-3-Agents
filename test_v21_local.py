#!/usr/bin/env python3
"""Quick test of v21 agent improvements."""
import sys
import logging
sys.path.insert(0, '/home/vova-spark/ARC-AGI-3-Agents')

logging.basicConfig(level=logging.INFO, format='%(message)s')

from rewind_agent_v21 import _scan_actions, _bfs, _replay_path
import importlib.util
from pathlib import Path

def test_game(game_id, level=0, timeout=60):
    """Test BFS on a game."""
    print(f"\n{'='*70}")
    print(f"Testing {game_id} Level {level} with v21 improvements")
    print('='*70)
    
    # Load game
    spec = importlib.util.spec_from_file_location(game_id, f'environment_files/{game_id}/{game_id}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    GameCls = getattr(mod, game_id.capitalize())
    
    # Scan actions
    prioritize = game_id in ['r11l', 'su15', 'tn36']
    actions = _scan_actions(GameCls, level, timeout=15, prioritize_clicks=prioritize)
    print(f"Found {len(actions)} actions")
    
    # Try BFS with v21 tuning
    print(f"Running BFS with game-specific tuning...")
    sol = _bfs(GameCls, actions, level, max_states=500000, timeout=timeout, game_id=game_id)
    
    if sol:
        print(f"\n✓ SOLUTION FOUND: {len(sol)} actions")
        # Verify
        g, r = _replay_path(GameCls, sol, level)
        if r and r.levels_completed > 0:
            print(f"✓ VERIFIED: levels_completed = {r.levels_completed}")
            return True
        else:
            print(f"✗ Solution doesn't complete level")
            return False
    else:
        print(f"\n✗ No solution found")
        return False

# Test the 5 unsolved games - just level 0 of each with short timeouts
games = [
    ('wa30', 60),  # Sokoban - needs longer
    ('bp35', 45),
    ('dc22', 45),
    ('r11l', 30),  # Click-only
    ('su15', 30),
]

results = {}
for game_id, timeout in games:
    try:
        solved = test_game(game_id, level=0, timeout=timeout)
        results[game_id] = solved
    except Exception as e:
        print(f"\nERROR testing {game_id}: {e}")
        import traceback
        traceback.print_exc()
        results[game_id] = False

print(f"\n{'='*70}")
print("SUMMARY")
print('='*70)
for game_id, solved in results.items():
    status = "✓ SOLVED" if solved else "✗ FAILED"
    print(f"{game_id:10s} {status}")
print(f"\nTotal solved: {sum(results.values())}/{ len(results)}")
