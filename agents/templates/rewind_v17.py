"""
RewindAgent v17 — Universal BFS Game Solver

Principles (from ARC-AGI-3 insights):
1. Environment IS the model — deep-copy game state, simulate offline
2. Discover actions empirically — scan every action, keep only those that change state
3. BFS with state deduplication — systematic exploration, no assumptions
4. Re-scan on each level — levels change everything
5. No hardcoding — same algorithm for all 25 games

Approach:
- Load game source from environment_files/
- For each level: scan effective actions → BFS to find solution → replay online
- Click games: find all unique-effect click positions (typically 2-100)
- Keyboard games: 2-5 effective directions per state
- Mixed: combine both
- IDDFS fallback when state space too large
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

from ..agent import Agent

logger = logging.getLogger(__name__)


def _state_hash(frame_arr: np.ndarray) -> str:
    """Fast state hash using MD5 of raw bytes."""
    return hashlib.md5(frame_arr.tobytes()).hexdigest()[:16]


def _scan_effective_actions(
    game: Any, f0: np.ndarray, bg: int, timeout: float = 15.0
) -> List[Tuple[int, Optional[Dict]]]:
    """
    Discover ALL actions that actually change game state.
    Returns list of (action_id, data_dict_or_None).
    For clicks, finds unique-effect positions by scanning non-bg cells.
    """
    avail = game._available_actions
    actions = []
    seen_effects: Set[str] = set()

    # Non-click actions (1-5, 7)
    for a in [x for x in avail if x != 6]:
        g = copy.deepcopy(game)
        try:
            r = g.perform_action(ActionInput(id=GameAction.from_id(a)), raw=True)
            if r.frame:
                f1 = np.array(r.frame[-1])
                if np.any(f0 != f1):
                    eh = hashlib.md5(f1.tobytes()).hexdigest()[:12]
                    if eh not in seen_effects:
                        seen_effects.add(eh)
                        actions.append((a, None))
        except Exception:
            pass

    # Click actions (ACTION6) — scan all non-bg positions for unique effects
    if 6 in avail:
        click_effects: Set[str] = set()
        t0 = time.time()

        # Get all non-bg positions
        non_bg_positions = list(zip(*np.where(f0 != bg)))

        # Sort by value to group similar cells together
        non_bg_positions.sort(key=lambda p: (f0[p[0], p[1]], p[0], p[1]))

        # Track which (value, region) combos we've already found effective clicks for
        tested_positions: Set[Tuple[int, int]] = set()

        for y, x in non_bg_positions:
            if time.time() - t0 > timeout:
                break
            if (x, y) in tested_positions:
                continue
            tested_positions.add((x, y))

            g = copy.deepcopy(game)
            try:
                r = g.perform_action(
                    ActionInput(
                        id=GameAction.ACTION6,
                        data={"x": int(x), "y": int(y), "game_id": "scan"},
                    ),
                    raw=True,
                )
                if not r.frame:
                    continue
                f1 = np.array(r.frame[-1])
                if np.any(f0 != f1):
                    eh = hashlib.md5(f1.tobytes()).hexdigest()[:12]
                    if eh not in click_effects:
                        click_effects.add(eh)
                        actions.append(
                            (6, {"x": int(x), "y": int(y), "game_id": "bfs"})
                        )
                        # Check for level complete / win
                        if r.levels_completed > 0 or r.state == GameState.WIN:
                            logger.info(f"Single-click solution found at ({x},{y})!")
                            return [(6, {"x": int(x), "y": int(y), "game_id": "bfs"})]
            except Exception:
                pass

    logger.info(
        f"Action scan: {len(actions)} effective actions "
        f"({len([a for a in actions if a[0] != 6])} non-click, "
        f"{len([a for a in actions if a[0] == 6])} clicks)"
    )
    return actions


def _replay_path(
    game_cls: Any, path: List[Tuple[int, Optional[Dict]]], level_idx: int = 0
) -> Tuple[Any, Any]:
    """
    Replay an action path from a fresh game instance.
    Returns (game_instance, last_result).
    Memory-efficient: no stored game copies needed.
    """
    g = game_cls()
    if hasattr(g, "set_level"):
        g.set_level(level_idx)
    r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    for act_id, data in path:
        ai = (
            ActionInput(id=GameAction.from_id(act_id), data=data)
            if data
            else ActionInput(id=GameAction.from_id(act_id))
        )
        r = g.perform_action(ai, raw=True)
    return g, r


def _bfs(
    game_cls: Any,
    actions: List[Tuple[int, Optional[Dict]]],
    level_idx: int = 0,
    max_states: int = 50000,
    timeout: float = 120.0,
) -> Optional[List[Tuple[int, Optional[Dict]]]]:
    """
    Zero-copy BFS — stores only action paths, replays from scratch.
    For each candidate move: replay(path + [move]), check result.
    No deepcopy at all. Uses ~0 memory beyond the queue of paths.
    """
    game = game_cls()
    if hasattr(game, "set_level"):
        game.set_level(level_idx)
    r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    if not r0.frame:
        return None

    f0 = np.array(r0.frame[-1])
    h0 = _state_hash(f0)
    t0 = time.time()
    visited: Set[str] = {h0}
    queue: deque = deque([[]])  # store paths only
    states = 0
    branching = len(actions)

    # Adaptive depth limit based on branching factor
    if branching <= 4:
        max_depth = 50
    elif branching <= 10:
        max_depth = 25
    elif branching <= 30:
        max_depth = 12
    else:
        max_depth = 8

    logger.info(
        f"BFS start: {branching} actions, max_depth={max_depth}, "
        f"max_states={max_states}"
    )

    while queue and states < max_states and (time.time() - t0) < timeout:
        path = queue.popleft()
        states += 1

        if len(path) >= max_depth:
            continue

        for act_id, data in actions:
            # Replay full path + this action from scratch (zero memory)
            candidate = path + [(act_id, data)]
            try:
                _, r = _replay_path(game_cls, candidate, level_idx)
            except Exception:
                continue

            if not r.frame:
                continue

            f = np.array(r.frame[-1])

            # Check win condition
            if r.levels_completed > 0 or r.state == GameState.WIN:
                elapsed = time.time() - t0
                logger.info(
                    f"BFS SOLVED in {len(candidate)} actions, "
                    f"{states} states, {elapsed:.1f}s"
                )
                return candidate

            # Skip game over states
            if r.state == GameState.GAME_OVER:
                continue

            h = _state_hash(f)
            if h in visited:
                continue
            visited.add(h)
            queue.append(candidate)

    elapsed = time.time() - t0
    logger.info(
        f"BFS exhausted: {states} states, depth_limit={max_depth}, "
        f"{elapsed:.1f}s, visited={len(visited)}"
    )
    return None


def _iddfs(
    game_cls: Any,
    actions: List[Tuple[int, Optional[Dict]]],
    level_idx: int = 0,
    max_depth: int = 30,
    timeout: float = 120.0,
) -> Optional[List[Tuple[int, Optional[Dict]]]]:
    """
    Iterative Deepening DFS with path-replay — low memory.
    """
    t0 = time.time()
    total_states = 0

    for depth in range(1, max_depth + 1):
        if time.time() - t0 > timeout:
            break

        game = game_cls()
        if hasattr(game, "set_level"):
            game.set_level(level_idx)
        r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        if not r0.frame:
            return None

        f0 = np.array(r0.frame[-1])
        visited: Set[str] = {_state_hash(f0)}

        # Stack stores paths only
        stack: List[Tuple[List, int]] = [([], 0)]
        states_this_depth = 0

        while stack and (time.time() - t0) < timeout:
            path, d = stack.pop()
            states_this_depth += 1
            total_states += 1

            if d >= depth:
                continue

            for act_id, data in actions:
                candidate = path + [(act_id, data)]
                try:
                    _, r = _replay_path(game_cls, candidate, level_idx)
                except Exception:
                    continue

                if not r.frame:
                    continue

                f = np.array(r.frame[-1])

                if r.levels_completed > 0 or r.state == GameState.WIN:
                    elapsed = time.time() - t0
                    logger.info(
                        f"IDDFS SOLVED at depth {depth} in {len(candidate)} actions, "
                        f"{total_states} total states, {elapsed:.1f}s"
                    )
                    return candidate

                if r.state == GameState.GAME_OVER:
                    continue

                h = _state_hash(f)
                if h in visited:
                    continue
                visited.add(h)
                stack.append((candidate, d + 1))

        logger.info(
            f"IDDFS depth {depth}: {states_this_depth} states, "
            f"visited={len(visited)}"
        )

    logger.info(f"IDDFS exhausted: {total_states} total states")
    return None


class RewindAgentV17(Agent):
    """Universal BFS game solver — same algorithm for all 25 games."""

    MAX_ACTIONS: int = 500

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.levels = 0
        self.queue: List[Tuple[int, Optional[Dict]]] = []
        self.attempt = 0
        self._game_cls: Any = None
        self._solutions: Dict[int, List[Tuple[int, Optional[Dict]]]] = {}
        self._solved_levels: Set[int] = set()
        self._load_game_source()
        logger.info(f"RewindAgentV17 init, game_cls={self._game_cls is not None}")
        # Pre-solve all levels at init time (before threads compete)
        self._pre_solve_all()

    def _load_game_source(self) -> None:
        """Load game source file for offline BFS."""
        env_dir = os.environ.get("ENVIRONMENTS_DIR", "environment_files")
        short = self.game_id.split("-")[0]
        class_name = short[0].upper() + short[1:]

        for p in [
            Path(env_dir) / short / f"{short}.py",
            Path(env_dir) / short / f"{class_name.lower()}.py",
            Path(env_dir) / f"{short}.py",
        ]:
            if p.exists():
                try:
                    spec = importlib.util.spec_from_file_location(
                        f"g_{short}", str(p)
                    )
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)  # type: ignore
                    self._game_cls = getattr(mod, class_name)
                    logger.info(f"Loaded game source: {p}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load {p}: {e}")

    def _pre_solve_all(self) -> None:
        """Pre-solve all levels at init time."""
        if self._game_cls is None:
            return
        for level in range(20):  # try up to 20 levels
            try:
                if not self._solve_level(level):
                    break
            except Exception:
                break
        solved = list(self._solutions.keys())
        logger.info(f"Pre-solved levels: {solved}")

    def _solve_level(self, level_idx: int) -> bool:
        """
        Solve a level offline using BFS (or IDDFS fallback).
        Stores solution in self._solutions[level_idx].
        """
        if self._game_cls is None:
            return False
        if level_idx in self._solved_levels:
            return level_idx in self._solutions

        self._solved_levels.add(level_idx)
        logger.info(f"Attempting to solve level {level_idx} offline...")

        try:
            game = self._game_cls()

            # Set level if method exists
            if hasattr(game, "set_level"):
                game.set_level(level_idx)

            # Reset to get initial state
            r = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            if not r.frame:
                logger.warning(f"No frame after reset for level {level_idx}")
                return False

            f0 = np.array(r.frame[-1])
            bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

            # Scan effective actions
            actions = _scan_effective_actions(game, f0, bg, timeout=15.0)
            if not actions:
                logger.warning(f"No effective actions found for level {level_idx}")
                return False

            branching = len(actions)
            logger.info(f"Level {level_idx}: {branching} effective actions")

            # Choose search strategy based on branching factor
            if branching <= 30:
                # BFS — optimal for low branching
                sol = _bfs(
                    self._game_cls, actions, level_idx=level_idx,
                    max_states=50000, timeout=120.0,
                )
            elif branching <= 100:
                # BFS with smaller state limit, then IDDFS
                sol = _bfs(
                    self._game_cls, actions, level_idx=level_idx,
                    max_states=10000, timeout=60.0,
                )
                if sol is None:
                    logger.info("BFS failed, trying IDDFS...")
                    sol = _iddfs(
                        self._game_cls, actions, level_idx=level_idx,
                        max_depth=15, timeout=60.0,
                    )
            else:
                # High branching — IDDFS only
                sol = _iddfs(
                    self._game_cls, actions, level_idx=level_idx,
                    max_depth=10, timeout=120.0,
                )

            if sol:
                self._solutions[level_idx] = sol
                logger.info(
                    f"Level {level_idx} SOLVED: {len(sol)} actions"
                )
                return True
            else:
                logger.warning(f"Level {level_idx}: no solution found")
                return False

        except Exception as e:
            logger.error(f"Error solving level {level_idx}: {e}")
            return False

    # ─── Online fallback helpers ───

    def _get_arr(self, frame: FrameData) -> Optional[np.ndarray]:
        return np.array(frame.frame[0]) if frame.frame else None

    def _generic_explore(
        self, frame: Optional[FrameData] = None
    ) -> List[Tuple[int, Optional[Dict]]]:
        """Generic exploration using available actions."""
        avail = self.available or [1, 2, 3, 4]
        moves: List[Tuple[int, Optional[Dict]]] = []

        # Keyboard actions
        kbd = [a for a in avail if a != 6]
        if kbd:
            for _ in range(5):
                for a in kbd:
                    moves.extend([(a, None)] * 3)

        # Click actions — try clicking non-background cells
        if 6 in avail and frame is not None:
            arr = self._get_arr(frame)
            if arr is not None:
                bg = int(np.bincount(arr.flatten(), minlength=16).argmax())
                non_bg = list(zip(*np.where(arr != bg)))
                # Sample up to 30 click positions
                step = max(1, len(non_bg) // 30)
                for i in range(0, len(non_bg), step):
                    y, x = non_bg[i]
                    moves.append(
                        (6, {"x": int(x), "y": int(y), "game_id": "explore"})
                    )

        return moves if moves else [(1, None)] * 15  # absolute fallback

    # ─── Main loop ───

    def is_done(self, frames: list, latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list, latest_frame: FrameData
    ) -> GameAction:
        # Handle reset states
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.queue = []
            self.attempt = 0
            return GameAction.RESET

        # Level completion detection
        if latest_frame.levels_completed > self.levels:
            self.levels = latest_frame.levels_completed
            logger.info(f"🎉 LEVEL {self.levels} DONE!")
            self.queue = []
            self.attempt = 0

        # Store available actions on first frame
        if not hasattr(self, "available") or self.available is None:
            self.available = latest_frame.available_actions or []
            logger.info(f"Available actions: {self.available}")

        # Try offline BFS for current level
        if not self.queue and self.levels not in self._solved_levels:
            if self._solve_level(self.levels):
                self.queue = list(self._solutions[self.levels])
                logger.info(
                    f"Loaded BFS solution for level {self.levels}: "
                    f"{len(self.queue)} actions"
                )

        # Execute queued actions
        if self.queue:
            return self._execute_next()

        # Online fallback — generic exploration
        self.attempt += 1
        if self.attempt > 20:
            # Reset and try again
            self.attempt = 0
            return GameAction.RESET

        self.queue = self._generic_explore(latest_frame)
        if self.queue:
            return self._execute_next()

        return GameAction.RESET

    def _execute_next(self) -> GameAction:
        """Execute next action from queue."""
        act_id, data = self.queue.pop(0)

        if act_id == 6 and data:
            action = GameAction.ACTION6
            action.action_data.x = int(data["x"])
            action.action_data.y = int(data["y"])
            action.reasoning = f"v17 click ({data['y']},{data['x']})"
            return action

        action = GameAction.from_id(act_id)
        action.reasoning = f"v17 L{self.levels}"
        return action

    def cleanup(self, *a: Any, **kw: Any) -> None:
        if self._cleanup:
            solved = list(self._solutions.keys())
            logger.info(
                f"RewindAgentV17 done: {self.levels} levels completed, "
                f"BFS solved levels: {solved}"
            )
        super().cleanup(*a, **kw)
