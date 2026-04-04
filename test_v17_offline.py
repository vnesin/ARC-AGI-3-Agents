#!/usr/bin/env python3
"""Offline test of RewindAgentV17 across all 25 games."""
import importlib.util
import logging
import os
import sys
import time

import numpy as np
from arcengine import ActionInput, GameAction, GameState

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Import the BFS functions directly
sys.path.insert(0, '.')
from agents.templates.rewind_v17 import _scan_effective_actions, _bfs, _iddfs, _state_hash, _replay_path

GAMES = sorted(os.listdir('environment_files'))
RESULTS = {}

for game in GAMES:
    gpath = f'environment_files/{game}/{game}.py'
    if not os.path.exists(gpath):
        continue
    class_name = game[0].upper() + game[1:]
    
    print(f'\n{"="*60}')
    print(f'GAME: {game}')
    print(f'{"="*60}')
    
    try:
        spec = importlib.util.spec_from_file_location(f'g_{game}', gpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = getattr(mod, class_name)
    except Exception as e:
        print(f'  FAILED to load: {e}')
        RESULTS[game] = {'levels': 0, 'error': str(e)}
        continue
    
    levels_solved = 0
    total_actions = 0
    game_start = time.time()
    
    for level in range(10):  # Try up to 10 levels
        if time.time() - game_start > 300:  # 5 min max per game
            print(f'  Game timeout at level {level}')
            break
            
        try:
            g = cls()
            if hasattr(g, 'set_level'):
                g.set_level(level)
            
            r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            if not r.frame:
                print(f'  L{level}: No frame after reset')
                break
            
            f0 = np.array(r.frame[-1])
            bg = int(np.bincount(f0.flatten(), minlength=16).argmax())
            
            # Scan actions
            t0 = time.time()
            actions = _scan_effective_actions(g, f0, bg, timeout=15.0)
            scan_time = time.time() - t0
            
            if not actions:
                print(f'  L{level}: No effective actions ({scan_time:.1f}s scan)')
                break
            
            branching = len(actions)
            print(f'  L{level}: {branching} actions ({scan_time:.1f}s scan)', end=' → ')
            
            # BFS (path-replay)
            t0 = time.time()
            if branching <= 30:
                sol = _bfs(cls, actions, level_idx=level, max_states=50000, timeout=120.0)
            elif branching <= 100:
                sol = _bfs(cls, actions, level_idx=level, max_states=10000, timeout=60.0)
                if sol is None:
                    sol = _iddfs(cls, actions, level_idx=level, max_depth=15, timeout=60.0)
            else:
                sol = _iddfs(cls, actions, level_idx=level, max_depth=10, timeout=120.0)
            
            solve_time = time.time() - t0
            
            if sol:
                levels_solved += 1
                total_actions += len(sol)
                print(f'SOLVED in {len(sol)} actions ({solve_time:.1f}s)')
            else:
                print(f'FAILED ({solve_time:.1f}s)')
                break  # Can't progress without solving current level
                
        except Exception as e:
            print(f'  L{level}: ERROR: {e}')
            break
    
    elapsed = time.time() - game_start
    RESULTS[game] = {'levels': levels_solved, 'actions': total_actions, 'time': elapsed}
    print(f'  → {levels_solved} levels solved in {elapsed:.1f}s')

# Summary
print(f'\n{"="*60}')
print(f'SUMMARY')
print(f'{"="*60}')
total_levels = 0
games_with_l0 = 0
for game in sorted(RESULTS):
    r = RESULTS[game]
    levels = r.get('levels', 0)
    total_levels += levels
    if levels > 0:
        games_with_l0 += 1
    status = f'L0-L{levels-1}' if levels > 0 else 'UNSOLVED'
    extra = f' ({r.get("actions", 0)} actions, {r.get("time", 0):.1f}s)' if levels > 0 else ''
    err = f' ERROR: {r.get("error", "")}' if 'error' in r else ''
    print(f'  {game:8s}: {status:12s}{extra}{err}')

print(f'\nTotal: {games_with_l0}/25 games with L0+, {total_levels} total levels')
