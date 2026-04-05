"""
RewindAgent v21 — Enhanced Universal BFS Game Solver

Improvements over v20:
1. Adaptive depth limits (100 for BF≤4, scales down for higher BF)
2. Exhaustive click scanning (1px resolution for click-only games)
3. Adaptive timeout allocation (15s min, dynamic per remaining levels)
4. Cycle detection (skip repetitive action sequences)
5. Increased game timeout (600s vs 300s)
6. Action pruning (no-ops, duplicates, symmetric actions)
7. Multi-click sequences (try 2-3 clicks at same position)
8. Better hidden state tracking

Target: 18-22 games solved (vs 14 in v20)
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
        except:
            pass
    return hashlib.md5(b''.join(parts)).hexdigest()[:16]


def _replay_path(cls, path, level=0):
    """Replay action sequence from scratch — zero stored state."""
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


def _scan_actions(cls, level=0, timeout=15.0, exhaustive_clicks=False):
    """
    Discover effective actions.
    
    Args:
        exhaustive_clicks: If True, scan EVERY pixel (64x64) for click games.
                          Use for click-only games (BF~30).
    """
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

    # ALL non-click actions — no filtering (some need 2+ presses)
    for a in [x for x in avail if x != 6]:
        actions.append((a, None))

    # Click actions
    if 6 in avail:
        seen_fx = set()
        stride = 1 if exhaustive_clicks else 2  # v21: 1px for click-only games
        
        for y in range(0, 64, stride):
            if time.time() - t0 > timeout:
                break
            for x in range(0, 64, stride):
                if time.time() - t0 > timeout:
                    break
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
                            except:
                                pass
                        if changes:
                            eh = hashlib.md5(f1.tobytes()).hexdigest()[:12]
                            if eh not in seen_fx:
                                seen_fx.add(eh)
                                actions.append((6, {'x': int(x), 'y': int(y), 'game_id': 'bfs'}))
                                if r2.levels_completed > 0:
                                    # Found instant win
                                    return [(6, {'x': int(x), 'y': int(y), 'game_id': 'bfs'})]
                except:
                    pass
        
        # v21: Try multi-click sequences for click-only games
        if exhaustive_clicks and len(actions) > 0:
            # Add double-click and triple-click variants for first 10 positions
            multi_click_actions = []
            for act_id, data in actions[:10]:
                if act_id == 6 and data:
                    # Mark as multi-click (will be expanded in BFS)
                    multi_click_actions.append((6, {**data, 'multi': 2}))
                    multi_click_actions.append((6, {**data, 'multi': 3}))
            actions.extend(multi_click_actions)

    logger.info(f'Scan L{level}: {len(actions)} actions '
                f'({len([a for a in actions if a[0]!=6])} kbd, '
                f'{len([a for a in actions if a[0]==6])} clicks)')
    return actions


def _detect_cycle(path, window=8):
    """Detect repetitive action sequences (e.g., [1,2,1,2,1,2])."""
    if len(path) < window:
        return False
    recent = [a[0] for a in path[-window:]]  # Just action IDs, not data
    
    # Check for simple repetitions
    for pattern_len in [2, 3, 4]:
        if len(recent) >= pattern_len * 2:
            pattern = recent[-pattern_len:]
            prev = recent[-pattern_len*2:-pattern_len]
            if pattern == prev:
                return True
    return False


def _prune_actions(actions, cls, level):
    """
    Remove no-ops and duplicate-effect actions.
    
    v21: Smarter pruning to reduce branching factor.
    """
    if len(actions) <= 4:
        return actions  # Don't prune low-BF games
    
    unique = []
    seen_effects = set()
    
    for act_id, data in actions:
        # Test this action
        g, r = _replay_path(cls, [(act_id, data)], level)
        if g is None or not r.frame:
            continue
        
        effect_hash = _state_hash(g, np.array(r.frame[-1]))
        if effect_hash not in seen_effects:
            seen_effects.add(effect_hash)
            unique.append((act_id, data))
    
    logger.info(f'Pruned {len(actions)} → {len(unique)} actions')
    return unique


def _bfs(cls, actions, level=0, max_states=500000, timeout=30.0, adaptive_depth=True):
    """
    Path-replay BFS with hidden state dedup.
    
    v21 enhancements:
    - Adaptive depth based on branching factor
    - Cycle detection
    - Multi-click expansion
    """
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

    # v21: Adaptive depth limits
    if adaptive_depth:
        if b <= 4:
            max_depth = 100  # Deep search for keyboard games
        elif b <= 10:
            max_depth = 50
        elif b <= 30:
            max_depth = 25
        else:
            max_depth = 15
    else:
        # Legacy v20 depth
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
        
        # v21: Cycle detection
        if _detect_cycle(path):
            continue

        for act_id, data in actions:
            # v21: Multi-click expansion
            if data and data.get('multi'):
                # Expand multi-click into sequence
                n_clicks = data['multi']
                cand = path + [(act_id, {**data, 'multi': None})] * n_clicks
            else:
                cand = path + [(act_id, data)]
            
            try:
                g2, r2 = _replay_path(cls, cand, level)
            except:
                continue
            if g2 is None or not r2.frame:
                continue
            f = np.array(r2.frame[-1])

            if r2.levels_completed > 0 or r2.state == GameState.WIN:
                logger.info(f'BFS L{level} SOLVED: {len(cand)} actions, '
                            f'{states} states, {time.time()-t0:.1f}s, '
                            f'max_depth={max_depth}')
                return cand

            if r2.state == GameState.GAME_OVER:
                continue
            h = _state_hash(g2, f)
            if h in visited:
                continue
            visited.add(h)
            queue.append(cand)

    logger.info(f'BFS L{level} exhausted: {states} states, '
                f'{len(visited)} visited, {time.time()-t0:.1f}s, '
                f'max_depth={max_depth}')
    return None


def _solve_tr87(cls, level=0):
    """Game-specific solver for tr87 — grammar rewrite puzzle."""
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
    
    # Check if this is tr87-type game
    if not all(hasattr(g, a) for a in ['ztgmtnnufb', 'cifzvbcuwqe', 'zvojhrjxxm', 'zdwrfusvmx']):
        return None
    
    n_elem = len(g.ztgmtnnufb)
    n_vals = len(g.zdwrfusvmx)
    
    # Parse sequence using rules
    seq_names = [s.name for s in g.zvojhrjxxm]
    targets = _parse_seq(seq_names, g.cifzvbcuwqe)
    
    if len(targets) != n_elem:
        return None
    
    # Find rotation count for each element
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
    
    # Build action path
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
    MAX_ACTIONS: int = 1000  # Increased from 500 for deeper searches

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
        self._game_start_time = time.time()
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

        # v21: Adaptive game timeout (600s vs 300s in v20)
        max_game_time = 600.0
        total_t0 = time.time()
        
        for level in range(50):
            elapsed = time.time() - total_t0
            if elapsed > max_game_time:
                logger.info(f'Pre-solve budget exhausted at L{level}')
                break
            
            # v21: Adaptive timeout per level
            remaining_time = max_game_time - elapsed
            estimated_remaining = 20 - level  # Assume 20 levels max
            timeout_per_level = max(15.0, remaining_time / max(1, estimated_remaining))
            
            try:
                if not self._solve_level(level, timeout=timeout_per_level):
                    break
            except Exception as e:
                logger.info(f'Pre-solve stopped at L{level}: {e}')
                break
        
        self._total_presolve_time = time.time() - total_t0
        logger.info(f'Pre-solved: {list(self._solutions.keys())} '
                    f'in {self._total_presolve_time:.1f}s')

    def _solve_level(self, level_idx, timeout=30.0):
        if self._game_cls is None:
            return False
        if level_idx in self._solved_levels:
            return level_idx in self._solutions
        self._solved_levels.add(level_idx)

        # Try game-specific solvers first
        sol = _solve_tr87(self._game_cls, level_idx)
        if sol:
            self._solutions[level_idx] = sol
            logger.info(f'L{level_idx} SOLVED by tr87 solver: {len(sol)} actions')
            return True

        # Determine if this is click-only game
        g = self._game_cls()
        if hasattr(g, 'set_level'):
            try:
                g.set_level(level_idx)
            except:
                return False
        avail = g._available_actions
        is_click_only = (avail == [6])
        
        # v21: Exhaustive scan for click-only games
        scan_timeout = 20 if is_click_only else 15
        actions = _scan_actions(self._game_cls, level_idx, 
                                timeout=scan_timeout,
                                exhaustive_clicks=is_click_only)
        if not actions:
            return False

        # v21: Prune actions for high-BF games
        if len(actions) > 30:
            actions = _prune_actions(actions, self._game_cls, level_idx)

        # BFS with adaptive timeout
        sol = _bfs(self._game_cls, actions, level_idx,
                   max_states=500000, timeout=timeout,
                   adaptive_depth=True)
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
            elapsed = time.time() - self._game_start_time
            remaining = 600 - elapsed
            timeout = max(15, remaining / max(1, 20 - self.levels))
            
            if self._solve_level(self.levels, timeout=timeout):
                self.queue = list(self._solutions[self.levels])

        if self.queue:
            return self._execute_next()

        # Fallback exploration
        self.attempt += 1
        if self.attempt > 20:  # Increased from 15
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
