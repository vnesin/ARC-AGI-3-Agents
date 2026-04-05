#!/usr/bin/env python3
"""Quick test of v20 solver to see which games are actually solved."""
import copy as _cp
import hashlib
import importlib.util
import logging
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from arcengine import ActionInput, GameAction, GameState

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

GAMES = [
    'ar25', 'bp35', 'cd82', 'cn04', 'dc22', 'ft09', 'g50t', 'ka59',
    'lf52', 'lp85', 'ls20', 'm0r0', 'r11l', 're86', 's5i5', 'sb26',
    'sc25', 'sk48', 'sp80', 'su15', 'tn36', 'tr87', 'tu93', 'vc33', 'wa30'
]

def _state_hash(game, frame):
    """Hash combining frame + hidden state."""
    parts = [frame.tobytes()]
    if hasattr(game, '_get_hidden_state'):
        try:
            hs = game._get_hidden_state()
            if hs is not None:
                parts.append(np.asarray(hs).tobytes())
        except:
            pass
    return hashlib.md5(b''.join(parts)).hexdigest()[:16]

def _replay_path(cls, path, level=0):
    """Replay action sequence from scratch."""
    g = cls()
    if hasattr(g, 'set_level'):
        try:
            g.set_level(level)
        except:
            return None, None
    r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    for a, d in path:
        ai = (ActionInput(id=GameAction.from_id(a), data=d)
              if d else ActionInput(id=GameAction.from_id(a)))
        r = g.perform_action(ai, raw=True)
    return g, r

def _scan_actions(cls, level=0, timeout=10.0):
    """Discover effective actions."""
    g = cls()
    if hasattr(g, 'set_level'):
        try:
            g.set_level(level)
        except:
            return []
    r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    if not r.frame:
        return []
    f0 = np.array(r.frame[-1])
    has_hidden = hasattr(g, '_get_hidden_state')
    avail = g._available_actions
    actions = []
    t0 = time.time()

    # Non-click actions
    for a in [x for x in avail if x != 6]:
        actions.append((a, None))

    # Click actions - scan ALL cells
    if 6 in avail:
        seen_fx = set()
        for y in range(0, 64, 2):
            if time.time() - t0 > timeout:
                break
            for x in range(0, 64, 2):
                if time.time() - t0 > timeout:
                    break
                gc = _cp.deepcopy(g)
                try:
                    r2 = gc.perform_action(
                        ActionInput(id=GameAction.ACTION6,
                                    data={'x': int(x), 'y': int(y), 'game_id': 'scan'}),
                        raw=True)
                    if r2.frame:
                        f1 = np.array(r2.frame[-1])
                        changes = np.any(f0 != f1)
                        if not changes and has_hidden:
                            try:
                                changes = not np.array_equal(
                                    np.asarray(g._get_hidden_state()),
                                    np.asarray(gc._get_hidden_state()))
                            except:
                                pass
                        if changes:
                            eh = hashlib.md5(f1.tobytes()).hexdigest()[:12]
                            if eh not in seen_fx:
                                seen_fx.add(eh)
                                actions.append((6, {'x': int(x), 'y': int(y), 'game_id': 'bfs'}))
                                if r2.levels_completed > 0:
                                    return [(6, {'x': int(x), 'y': int(y), 'game_id': 'bfs'})]
                except:
                    pass

    return actions

def _bfs(cls, actions, level=0, max_states=100000, timeout=25.0):
    """Path-replay BFS with hidden state dedup."""
    g, r = _replay_path(cls, [], level)
    if g is None or not r.frame:
        return None
    f0 = np.array(r.frame[-1])
    h0 = _state_hash(g, f0)
    t0 = time.time()
    visited = {h0}
    queue = deque([[]])
    states = 0
    b = len(actions)

    if b <= 4:
        max_depth = 40
    elif b <= 10:
        max_depth = 25
    elif b <= 30:
        max_depth = 12
    else:
        max_depth = 8

    while queue and states < max_states and (time.time() - t0) < timeout:
        path = queue.popleft()
        states += 1
        if len(path) >= max_depth:
            continue

        for act_id, data in actions:
            cand = path + [(act_id, data)]
            try:
                g2, r2 = _replay_path(cls, cand, level)
            except:
                continue
            if g2 is None or not r2.frame:
                continue
            f = np.array(r2.frame[-1])

            if r2.levels_completed > 0 or r2.state == GameState.WIN:
                return cand

            if r2.state == GameState.GAME_OVER:
                continue
            h = _state_hash(g2, f)
            if h in visited:
                continue
            visited.add(h)
            queue.append(cand)

    return None

def _solve_tr87(cls, level=0):
    """Game-specific solver for tr87."""
    import copy as _cp
    
    def _parse_seq(seq_names, rules, max_depth=5):
        current = seq_names[:]
        for _ in range(max_depth):
            new_seq = []
            i = 0
            changed = False
            while i < len(current):
                matched = False
                for lhs, rhs in sorted(rules, key=lambda r: len(r[0]), reverse=True):
                    ln = [s.name for s in lhs]
                    if i+len(ln) <= len(current) and current[i:i+len(ln)] == ln:
                        new_seq.extend([s.name for s in rhs])
                        i += len(ln)
                        matched = True
                        changed = True
                        break
                if not matched:
                    new_seq.append(current[i])
                    i += 1
            current = new_seq
            if not changed:
                break
        return current
    
    g = cls()
    if hasattr(g, 'set_level'):
        try:
            g.set_level(level)
        except:
            return None
    g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    
    if not all(hasattr(g, a) for a in ['ztgmtnnufb', 'cifzvbcuwqe', 'zvojhrjxxm', 'zdwrfusvmx']):
        return None
    
    n_elem = len(g.ztgmtnnufb)
    n_vals = len(g.zdwrfusvmx)
    
    seq_names = [s.name for s in g.zvojhrjxxm]
    targets = _parse_seq(seq_names, g.cifzvbcuwqe)
    
    if len(targets) != n_elem:
        return None
    
    rotations = []
    for ei in range(n_elem):
        gc = _cp.deepcopy(g)
        ci = gc.qvtymdcqear_index
        while ci != ei:
            gc.perform_action(ActionInput(id=GameAction.ACTION4), raw=True)
            ci = (ci + 1) % n_elem
        found = None
        for rot in range(n_vals):
            if gc.ztgmtnnufb[ei].name == targets[ei]:
                found = rot
                break
            gc.perform_action(ActionInput(id=GameAction.ACTION1), raw=True)
        if found is None:
            return None
        rotations.append(found)
    
    path = []
    ci = g.qvtymdcqear_index
    for ei, rot in enumerate(rotations):
        while ci != ei:
            path.append((4, None))
            ci = (ci + 1) % n_elem
        for _ in range(rot):
            path.append((1, None))
    
    return path

def test_game(game_id, max_levels=20):
    """Test a single game and return solved levels."""
    path = Path(f'environment_files/{game_id}/{game_id}.py')
    if not path.exists():
        return 0, "No source file"
    
    class_name = game_id[0].upper() + game_id[1:]
    try:
        spec = importlib.util.spec_from_file_location(f'g_{game_id}', str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = getattr(mod, class_name)
    except Exception as e:
        return 0, f"Load error: {e}"
    
    solved_levels = 0
    game_start = time.time()
    
    for level in range(max_levels):
        if time.time() - game_start > 300:  # 5min per game
            return solved_levels, f"Game timeout at L{level}"
        
        # Try tr87 solver first
        sol = _solve_tr87(cls, level)
        
        if not sol:
            # Try BFS
            try:
                actions = _scan_actions(cls, level, timeout=10)
                if not actions:
                    return solved_levels, f"No actions at L{level}"
                sol = _bfs(cls, actions, level, max_states=100000, timeout=25)
            except Exception as e:
                return solved_levels, f"Error at L{level}: {e}"
        
        if sol:
            solved_levels += 1
        else:
            return solved_levels, f"Unsolved at L{level}"
    
    return solved_levels, "Max levels reached"

def main():
    results = {}
    total_levels = 0
    games_solved = 0
    
    for game_id in GAMES:
        print(f"\nTesting {game_id}...", end=' ')
        levels, msg = test_game(game_id, max_levels=20)
        results[game_id] = (levels, msg)
        total_levels += levels
        if levels > 0:
            games_solved += 1
        print(f"{levels} levels ({msg})")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for game_id in GAMES:
        levels, msg = results[game_id]
        status = "✓" if levels > 0 else "✗"
        print(f"{status} {game_id:6s}: {levels:2d} levels - {msg}")
    
    print(f"\n{games_solved}/25 games, {total_levels} total levels")

if __name__ == '__main__':
    main()
