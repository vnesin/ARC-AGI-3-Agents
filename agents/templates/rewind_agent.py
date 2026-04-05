"""
RewindAgent v14 — Universal game solver.

Strategy: detect available actions, identify game elements, use appropriate solver.
- Keyboard games (UDLR): BFS navigation toward targets
- Click games (ACTION6): launch blocks + click targets
- Mixed: try both approaches
- Pattern matching: detect lock/key patterns and match them via rotator
"""

import logging
import numpy as np
from collections import deque
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent

logger = logging.getLogger()


class RewindAgent(Agent):
    MAX_ACTIONS: int = 500

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.levels = 0
        self.queue: list = []
        self.phase = 'INIT'
        self.attempt = 0
        self.game_type = None  # 'keyboard', 'click', 'mixed'
        self.available = None
        # ls20 specific
        self.door_pat = None
        self.door_pos = None
        # vc33 specific
        self.vc33_phase = 'launch'
        logger.info("RewindAgent v14 init")

    # ========== DETECTION ==========
    def _detect_type(self, frame):
        """Detect game type from available actions."""
        aa = frame.available_actions or []
        has_click = 6 in aa
        has_kbd = any(a in aa for a in [1, 2, 3, 4])
        if has_click and not has_kbd:
            return 'click'
        elif has_kbd and not has_click:
            return 'keyboard'
        elif has_click and has_kbd:
            return 'mixed'
        else:
            return 'keyboard'  # default

    def _get_arr(self, frame):
        if frame.frame:
            return np.array(frame.frame[0])
        return None

    # ========== PLAYER DETECTION (value 12 = player in many games) ==========
    def _find_player(self, arr):
        p = np.where(arr[:52] == 12)
        if len(p[0]) > 0:
            return (int(round(p[0].mean())), int(round(p[1].mean())))
        return None

    # ========== KEYBOARD SOLVER ==========
    def _get_key(self, arr):
        """Get 3x3 key pattern from status bar (rows 55+)."""
        try:
            return tuple(
                tuple(1 if int(arr[55 + r * 2, 3 + c * 2]) == 9 else 0 for c in range(3))
                for r in range(3)
            )
        except (IndexError, ValueError):
            return None

    def _find_rotator(self, arr):
        g = arr[:52]
        rs, cs = [], []
        for v in [0, 1]:
            l = np.where(g == v)
            rs.extend(l[0].tolist())
            cs.extend(l[1].tolist())
        return (int(round(np.mean(rs))), int(round(np.mean(cs)))) if rs else None

    def _find_door(self, arr):
        g = arr[:52]
        nines = np.where(g == 9)
        if len(nines[0]) == 0:
            return None, None
        d9 = []
        for r, c in zip(nines[0].tolist(), nines[1].tolist()):
            if r > 45:
                continue
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 52 and 0 <= nc < 64 and int(g[nr, nc]) == 5:
                        d9.append((r, c))
                        break
                else:
                    continue
                break
        if not d9:
            return None, None
        rn = min(r for r, c in d9); rx = max(r for r, c in d9)
        cn = min(c for r, c in d9); cx = max(c for r, c in d9)
        cr, cc = (rn + rx) // 2, (cn + cx) // 2
        pat = tuple(
            tuple(1 if int(g[cr + dr, cc + dc]) == 9 else 0 for dc in [-1, 0, 1])
            for dr in [-1, 0, 1]
        )
        return pat, (cr, cc)

    def _build_clearspace(self, arr):
        wall = (arr[:52] == 4)
        cs = np.zeros((52, 64), dtype=bool)
        for r in range(2, 50):
            for c in range(2, 62):
                if not np.any(wall[r - 2:r + 3, c - 2:c + 3]):
                    cs[r, c] = True
        return cs

    def _sim_move(self, pos, d, cs):
        dr, dc = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}[d]
        r, c = pos
        for _ in range(5):
            nr, nc = r + dr, c + dc
            if nr < 2 or nr >= 50 or nc < 2 or nc >= 62 or not cs[nr, nc]:
                break
            r, c = nr, nc
        return (r, c)

    def _bfs_to(self, cs, start, goal, max_depth=30):
        vis = {start}
        q = deque([(start, [])])
        best_path, best_dist = [], abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        while q:
            p, path = q.popleft()
            d = abs(p[0] - goal[0]) + abs(p[1] - goal[1])
            if d < 8:
                return path
            if d < best_dist:
                best_path, best_dist = path, d
            if len(path) >= max_depth:
                continue
            for di in 'UDLR':
                np2 = self._sim_move(p, di, cs)
                if np2 != p and np2 not in vis:
                    vis.add(np2)
                    q.append((np2, path + [di]))
        return best_path

    def _keyboard_plan(self, arr):
        """Plan for keyboard-based games (ls20-like)."""
        player = self._find_player(arr)
        rot = self._find_rotator(arr)
        dp, dc = self._find_door(arr)
        key = self._get_key(arr)

        # Near door? Try to enter
        if dc and player and abs(player[0] - dc[0]) < 6 and abs(player[1] - dc[1]) < 6:
            return [('D', None)] * 3

        self.door_pat = dp
        self.door_pos = dc

        if not all([player, rot, dp]):
            # No pattern elements found — try generic exploration
            return self._generic_explore()

        cs = self._build_clearspace(arr)

        if dp == key:
            # Key matches door — navigate to door
            path = self._bfs_to(cs, player, dc)
            return [(d, None) for d in path] + [('U', None)] * 3 + [('D', None)] * 3 + [('L', None)] * 3 + [('R', None)] * 3
        else:
            # Navigate to rotator then to door
            path_to_rot = self._bfs_to(cs, player, rot)
            pos = player
            for m in path_to_rot:
                pos = self._sim_move(pos, m, cs)
            path_to_door = self._bfs_to(cs, pos, dc)
            return [(d, None) for d in path_to_rot + path_to_door] + [('U', None)] * 3 + [('D', None)] * 3 + [('L', None)] * 3 + [('R', None)] * 3

    # ========== CLICK SOLVER ==========
    def _click_plan(self, arr):
        """Plan for click-based games (vc33-like)."""
        actions = []

        # Find launchable blocks (value 4, bottom half)
        bottom4 = sorted(set(
            (int(r), int(c)) for r, c in zip(*np.where(arr == 4)) if r >= 40
        ))

        # Find targets (value 9, top half)
        nines = sorted(set(
            (int(r), int(c)) for r, c in zip(*np.where(arr == 9)) if r < 50
        ))

        # Launch all blocks first
        for r, c in bottom4:
            actions.append(('A6', {'x': c, 'y': r}))

        # Then click ALL targets
        for r, c in nines:
            actions.append(('A6', {'x': c, 'y': r}))

        # If no specific targets, try clicking all non-background cells
        if not actions:
            for val in [9, 5, 4, 11, 8, 10]:
                locs = np.where(arr == val)
                for r, c in zip(locs[0].tolist()[:20], locs[1].tolist()[:20]):
                    actions.append(('A6', {'x': int(c), 'y': int(r)}))
                if len(actions) > 50:
                    break

        return actions if actions else [('A6', {'x': 32, 'y': 32})]

    # ========== GENERIC EXPLORATION ==========
    def _generic_explore(self):
        """Generic exploration when game type is unknown."""
        moves = []
        # Try systematic: spiral outward
        for _ in range(3):
            moves.extend([('R', None)] * 5)
            moves.extend([('D', None)] * 5)
            moves.extend([('L', None)] * 5)
            moves.extend([('U', None)] * 5)
        return moves

    # ========== MIXED SOLVER ==========
    def _mixed_plan(self, arr, frame):
        """Plan for mixed keyboard+click games."""
        aa = frame.available_actions or []

        # Try keyboard first (navigate), then click
        plan = []
        if any(a in aa for a in [1, 2, 3, 4]):
            plan.extend(self._keyboard_plan(arr)[:20])
        if 6 in aa:
            plan.extend(self._click_plan(arr)[:30])
        if 5 in aa:
            # ACTION5 (interact) — sprinkle it in
            plan.append(('5', None))

        return plan if plan else self._generic_explore()

    # ========== MAIN LOOP ==========
    def is_done(self, frames, latest_frame):
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames, latest_frame):
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.phase = 'PLAN'
            self.queue = []
            self.attempt = 0
            return GameAction.RESET

        # Level completion
        if latest_frame.levels_completed > self.levels:
            self.levels = latest_frame.levels_completed
            logger.info(f"🎉 LEVEL {self.levels} DONE!")
            self.phase = 'PLAN'
            self.queue = []
            self.attempt = 0

        # Detect game type on first frame
        if self.game_type is None:
            self.game_type = self._detect_type(latest_frame)
            self.available = latest_frame.available_actions or []
            logger.info(f"Game type: {self.game_type}, available: {self.available}")

        arr = self._get_arr(latest_frame)
        if arr is None:
            return GameAction.RESET

        # Check key match for keyboard games (ls20-like)
        if self.game_type in ('keyboard', 'mixed') and self.door_pat:
            key = self._get_key(arr)
            if key == self.door_pat and self.phase == 'CROSSING':
                player = self._find_player(arr)
                _, dc = self._find_door(arr)
                if dc and player:
                    cs = self._build_clearspace(arr)
                    path = self._bfs_to(cs, player, dc)
                    self.queue = [(d, None) for d in path] + [('U', None)] * 3 + [('D', None)] * 3 + [('L', None)] * 3 + [('R', None)] * 3
                    self.phase = 'TO_DOOR'

        # Execute queued actions
        if self.queue:
            return self._execute_next()

        # Generate new plan
        self.attempt += 1
        if self.attempt > 10:
            # Give up on this approach, try generic
            self.queue = self._generic_explore()
        elif self.game_type == 'keyboard':
            self.queue = self._keyboard_plan(arr)
            self.phase = 'CROSSING'
        elif self.game_type == 'click':
            self.queue = self._click_plan(arr)
            self.phase = 'CLICKING'
        elif self.game_type == 'mixed':
            self.queue = self._mixed_plan(arr, latest_frame)
            self.phase = 'MIXED'
        else:
            self.queue = self._generic_explore()

        if self.queue:
            return self._execute_next()
        return GameAction.ACTION1

    def _execute_next(self):
        act, data = self.queue.pop(0)

        if act == 'A6' and data:
            action = GameAction.ACTION6
            action.action_data.x = int(data['x'])
            action.action_data.y = int(data['y'])
            action.reasoning = f"v14 click ({data['y']},{data['x']})"
            return action
        elif act == '5':
            action = GameAction.ACTION5
            action.reasoning = "v14 interact"
            return action

        am = {
            'U': GameAction.ACTION1, 'D': GameAction.ACTION2,
            'L': GameAction.ACTION3, 'R': GameAction.ACTION4
        }
        action = am.get(act, GameAction.ACTION1)
        action.reasoning = f"v14 {self.game_type} {self.phase}"
        return action

    def cleanup(self, *a, **kw):
        if self._cleanup:
            logger.info(f"RewindAgent v14 final: {self.levels}/7")
        super().cleanup(*a, **kw)
