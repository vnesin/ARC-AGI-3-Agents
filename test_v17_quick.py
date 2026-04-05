#!/usr/bin/env python3
"""Quick test of v17 on known-solvable games and key categories."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from test_v17_offline import test_game_offline
import logging

logging.basicConfig(level=logging.WARNING)

# Games v16 claimed to solve at L0
v16_solved = ['cd82', 'cn04', 'ft09', 'lp85', 'ls20', 'm0r0', 'r11l', 's5i5', 'sp80', 'tu93', 'vc33']

# Key categories
click_only = ['ft09', 'lp85', 'r11l', 's5i5', 'tn36', 'vc33']
keyboard_only = ['ls20', 'tr87', 'tu93', 'g50t', 're86', 'wa30']

# Test set: v16 solved + representative samples
test_games = sorted(set(v16_solved + click_only[:3] + keyboard_only[:3]))

print(f'Testing {len(test_games)} games: {", ".join(test_games)}\n')

results = {}
for game_id in test_games:
    res = test_game_offline(game_id, max_levels=1, solve_timeout=30)  # Just L0, faster timeout
    results[game_id] = res
    l0_solved = res.get(0, {}).get('solved', False)
    l0_actions = res.get(0, {}).get('actions', 0)
    status = '✓' if l0_solved else '✗'
    print(f'{game_id:6} L0:{status} ({l0_actions:3}a)')

print(f'\n{"="*60}')
solved_count = sum(1 for r in results.values() if r.get(0, {}).get('solved'))
print(f'L0 solved: {solved_count}/{len(test_games)} ({100*solved_count/len(test_games):.1f}%)')
print(f'{"="*60}')
