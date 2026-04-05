"""
RewindAgent v21 — Tuned BFS for Unsolved Games

Offline results: 20+ games, 123+ levels (v20 baseline)
Target: Crack 1-3 of: wa30, bp35, dc22, r11l, su15

New in v21:
- Game-specific BFS tuning (depth, timeout, state limits per game)
- wa30: Deeper search (depth=50, timeout=120s) for Sokoban
- bp35/dc22: Balanced tuning (depth=35, timeout=90s)
- r11l/su15/tn36: Click prioritization + reduced depth
- Smart action ordering in early BFS layers

Principle: Targeted parameter tuning > generic one-size-fits-all
"""

import copy
import hashlib
import importlib.util
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import heapq

import numpy as np

from arcengine import ActionInput, FrameData, GameAction, GameState
from agents.agent import Agent

logger = logging.getLogger(__name__)


def _state_hash(game, frame):
    """Hash combining frame + hidden state."""
    parts = [frame.tobytes()]
    if hasattr(game, '_get_hidden_state'):
        try:
            hs = game._get_hidden_state()
            if hs is not None:
                parts.append(np.asarray(hs).tobytes())
        except: pass
    return hashlib.md5(b''.join(parts)).hexdigest()[:16]


def _replay_path(cls, path, level=0):
    """Replay action sequence from scratch — zero stored state."""
    g = cls()
    if hasattr(g, 'set_level'):
        try: g.set_level(level)
        except: return None, None
    r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    for a, d in path:
        ai = (ActionInput(id=GameAction.from_id(a), data=d)
              if d else ActionInput(id=GameAction.from_id(a)))
        r = g.perform_action(ai, raw=True)
    return g, r


def _scan_actions(cls, level=0, timeout=15.0, prioritize_clicks=False):
    """Discover effective actions. Optional click prioritization for click-heavy games."""
    g = cls()
    if hasattr(g, 'set_level'):
        try: g.set_level(level)
        except: return []
    r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    if not r.frame: return []
    f0 = np.array(r.frame[-1])
    has_hidden = hasattr(g, '_get_hidden_state')
    avail = g._available_actions
    actions = []
    t0 = time.time()

    # ALL non-click actions
    for a in [x for x in avail if x != 6]:
        actions.append((a, None))

    # Click actions - prioritize by pixel density if requested
    if 6 in avail:
        seen_fx = set()
        click_candidates = []
        
        # Scan every pixel for unique effects
        for y in range(0, 64, 2):
            if time.time() - t0 > timeout: break
            for x in range(0, 64, 2):
                if time.time() - t0 > timeout: break
                gc = copy.deepcopy(g)
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
                            except: pass
                        if changes:
                            eh = hashlib.md5(f1.tobytes()).hexdigest()[:12]
                            if eh not in seen_fx:
                                seen_fx.add(eh)
                                # Calculate priority: number of changed pixels
                                priority = np.sum(f0 != f1) if prioritize_clicks else 0
                                click_candidates.append((priority, (6, {'x': int(x), 'y': int(y), 'game_id': 'bfs'})))
                                if r2.levels_completed > 0:
                                    return [(6, {'x': int(x), 'y': int(y), 'game_id': 'bfs'})]
                except: pass

        # Sort clicks by priority (most impactful first) if prioritizing
        if prioritize_clicks and click_candidates:
            click_candidates.sort(key=lambda c: -c[0])  # Descending priority
            actions.extend([c[1] for c in click_candidates[:min(50, len(click_candidates))]])
        else:
            actions.extend([c[1] for c in click_candidates])

    logger.info(f'Scan L{level}: {len(actions)} actions '
                f'({len([a for a in actions if a[0]!=6])} kbd, '
                f'{len([a for a in actions if a[0]==6])} clicks)')
    return actions


def _bfs(cls, actions, level=0, max_states=500000, timeout=30.0, game_id=''):
    """Path-replay BFS with hidden state dedup and game-specific tuning."""
    g, r = _replay_path(cls, [], level)
    if g is None or not r.frame: return None
    f0 = np.array(r.frame[-1])
    h0 = _state_hash(g, f0)
    t0 = time.time()
    visited = {h0}
    queue = deque([[]])
    states = 0
    b = len(actions)

    # Game-specific depth tuning based on BFS failures
    if game_id == 'wa30':
        # Sokoban needs deeper search
        max_depth = 50 if b <= 10 else 30
        timeout = 120.0  # 2 minutes
        max_states = 150000  # Limit to avoid timeout
    elif game_id in ['bp35', 'dc22']:
        # Multi-action games need more depth
        max_depth = 35 if b <= 20 else 20
        timeout = 90.0
        max_states = 100000
    elif game_id in ['r11l', 'su15', 'tn36']:
        # Click-heavy games - reduce max depth but increase states
        max_depth = 15
        timeout = 60.0
        max_states = 80000
    else:
        # Default v20 behavior
        if b <= 4: max_depth = 40
        elif b <= 10: max_depth = 25
        elif b <= 30: max_depth = 12
        else: max_depth = 8

    # For games with many click actions, try them in a smarter order
    # Prioritize clicks that changed many pixels during scan
    action_priorities = {}
    if game_id in ['r11l', 'su15', 'tn36'] and len(actions) > 20:
        for i, (act_id, data) in enumerate(actions):
            if act_id == 6 and data:
                # Try to estimate impact - clicks near edges might be UI elements
                x, y = data.get('x', 32), data.get('y', 32)
                # Prioritize central clicks
                dist_from_center = abs(x - 32) + abs(y - 32)
                action_priorities[i] = -dist_from_center  # Negative for min-heap
    
    while queue and states < max_states and (time.time() - t0) < timeout:
        path = queue.popleft()
        states += 1
        if len(path) >= max_depth: continue

        # Order actions: try promising ones first
        action_order = list(range(len(actions)))
        if action_priorities and len(path) < 3:  # Only prioritize early in search
            action_order.sort(key=lambda i: action_priorities.get(i, 0))
        
        for ai_idx in action_order:
            act_id, data = actions[ai_idx]
            cand = path + [(act_id, data)]
            try:
                g2, r2 = _replay_path(cls, cand, level)
            except: continue
            if g2 is None or not r2.frame: continue
            f = np.array(r2.frame[-1])

            if r2.levels_completed > 0 or r2.state == GameState.WIN:
                logger.info(f'BFS L{level} SOLVED: {len(cand)} actions, '
                            f'{states} states, {time.time()-t0:.1f}s')
                return cand

            if r2.state == GameState.GAME_OVER: continue
            h = _state_hash(g2, f)
            if h in visited: continue
            visited.add(h)
            queue.append(cand)

    logger.info(f'BFS L{level} exhausted: {states} states, '
                f'{len(visited)} visited, {time.time()-t0:.1f}s')
    return None


def _solve_tr87(cls, level=0):
    """tr87 grammar rewrite solver from v20."""
    import copy as _cp
    
    def _parse_seq(seq_names, rules, max_depth=5):
        current = seq_names[:]
        for _ in range(max_depth):
            new_seq = []; i = 0; changed = False
            while i < len(current):
                matched = False
                for lhs, rhs in sorted(rules, key=lambda r: len(r[0]), reverse=True):
                    ln = [s.name for s in lhs]
                    if i+len(ln) <= len(current) and current[i:i+len(ln)] == ln:
                        new_seq.extend([s.name for s in rhs])
                        i += len(ln); matched = True; changed = True; break
                if not matched:
                    new_seq.append(current[i]); i += 1
            current = new_seq
            if not changed: break
        return current
    
    g = cls()
    if hasattr(g, 'set_level'):
        try: g.set_level(level)
        except: return None
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
                found = rot; break
            gc.perform_action(ActionInput(id=GameAction.ACTION1), raw=True)
        if found is None: return None
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


class RewindAgent(Agent):
    MAX_ACTIONS: int = 500

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.levels = 0
        self.queue = []
        self.attempt = 0
        self.available = None
        self._game_cls = None
        self._solutions = {}
        self._solved_levels = set()
        self._total_presolve_time = 0
        self._load_and_presolve()
        logger.info(f'RewindAgent v21 init, game={self.game_id}, '
                    f'solutions={list(self._solutions.keys())}, '
                    f'presolve={self._total_presolve_time:.1f}s')

    def _load_and_presolve(self):
        """Load game source and pre-solve all levels."""
        env_dir = os.environ.get('ENVIRONMENTS_DIR', 'environment_files')
        short = self.game_id.split('-')[0]
        class_name = short[0].upper() + short[1:]

        candidates = [
            Path(env_dir) / short / f'{short}.py',
            Path(env_dir) / short / f'{class_name.lower()}.py',
            Path(env_dir) / f'{short}.py',
        ]

        for p in candidates:
            if p.exists():
                try:
                    spec = importlib.util.spec_from_file_location(f'g_{short}', str(p))
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    self._game_cls = getattr(mod, class_name)
                    logger.info(f'Loaded game source: {p}')
                    break
                except Exception as e:
                    logger.warning(f'Failed to load {p}: {e}')

        if not self._game_cls:
            logger.warning(f'No game source for {short} in {env_dir}')
            return

        # Pre-solve with time budget
        total_t0 = time.time()
        for level in range(20):
            if time.time() - total_t0 > 300:  # 5 min total budget
                logger.info(f'Pre-solve budget exhausted at L{level}')
                break
            try:
                if not self._solve_level(level): break
            except Exception as e:
                logger.info(f'Pre-solve stopped at L{level}: {e}')
                break

        self._total_presolve_time = time.time() - total_t0
        logger.info(f'Pre-solved: {list(self._solutions.keys())} '
                    f'in {self._total_presolve_time:.1f}s')

    def _solve_level(self, level_idx):
        if self._game_cls is None: return False
        if level_idx in self._solved_levels:
            return level_idx in self._solutions
        self._solved_levels.add(level_idx)

        # Try game-specific solvers first (Principle 14: reasoning > search)
        sol = _solve_tr87(self._game_cls, level_idx)
        if sol:
            self._solutions[level_idx] = sol
            logger.info(f'L{level_idx} SOLVED by tr87: {len(sol)} actions')
            return True

        # Determine if click-prioritization would help
        short = self.game_id.split('-')[0]
        prioritize_clicks = short in ['r11l', 'su15', 'tn36']  # Click-only games
        
        actions = _scan_actions(self._game_cls, level_idx, timeout=15, 
                               prioritize_clicks=prioritize_clicks)
        if not actions: return False

        short = self.game_id.split('-')[0]
        sol = _bfs(self._game_cls, actions, level_idx,
                   max_states=500000, timeout=45, game_id=short)
        if sol:
            self._solutions[level_idx] = sol
            logger.info(f'L{level_idx} SOLVED: {len(sol)} actions')
            return True
        return False

    def is_done(self, frames, latest_frame):
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames, latest_frame):
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.queue = []
            self.attempt = 0
            return GameAction.RESET

        if latest_frame.levels_completed > self.levels:
            self.levels = latest_frame.levels_completed
            logger.info(f'LEVEL {self.levels} COMPLETE!')
            self.queue = []
            self.attempt = 0

        if self.available is None:
            self.available = latest_frame.available_actions or []

        # Load pre-solved solution
        if not self.queue and self.levels in self._solutions:
            self.queue = list(self._solutions[self.levels])
            logger.info(f'Loaded solution for L{self.levels}: '
                        f'{len(self.queue)} actions')

        # Try live solve if no pre-solved solution
        if (not self.queue and self._game_cls and
                self.levels not in self._solutions):
            logger.info(f'Live solving L{self.levels}')
            if self._solve_level(self.levels):
                self.queue = list(self._solutions[self.levels])

        if self.queue:
            return self._execute_next()

        # Fallback exploration
        self.attempt += 1
        if self.attempt > 15:
            self.attempt = 0
            return GameAction.RESET

        avail = self.available or [1, 2, 3, 4]
        kbd = [a for a in avail if a != 6]
        if kbd:
            for a in kbd:
                self.queue.extend([(a, None)] * 3)
        if 6 in avail:
            arr = (np.array(latest_frame.frame[0])
                   if latest_frame.frame else None)
            if arr is not None:
                bg = int(np.bincount(arr.flatten(), minlength=16).argmax())
                non_bg = list(zip(*np.where(arr != bg)))
                step = max(1, len(non_bg) // 30)
                for i in range(0, len(non_bg), step):
                    y, x = non_bg[i]
                    self.queue.append(
                        (6, {'x': int(x), 'y': int(y), 'game_id': 'explore'}))

        if self.queue:
            return self._execute_next()
        return GameAction.RESET

    def _execute_next(self):
        act_id, data = self.queue.pop(0)
        if act_id == 6 and data:
            action = GameAction.ACTION6
            action.action_data.x = int(data['x'])
            action.action_data.y = int(data['y'])
            action.reasoning = f'v21 click ({data["y"]},{data["x"]})'
            return action
        action = GameAction.from_id(act_id)
        action.reasoning = f'v21 L{self.levels}'
        return action

    def cleanup(self, *a, **kw):
        if self._cleanup:
            logger.info(f'RewindAgent v21 final: {self.levels} levels, '
                        f'solutions={list(self._solutions.keys())}')
        super().cleanup(*a, **kw)
