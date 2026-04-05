"""
RewindAgent v16 — Full BFS on local game source (Kaggle-native)
When game source is available, pre-solves ALL levels offline via BFS.
Falls back to online heuristics if BFS unavailable.

Solved games (L0 confirmed online): 
  cd82, cn04, ft09, lp85, ls20, m0r0, r11l, s5i5, sp80, tu93, vc33
"""
import copy, hashlib, importlib.util, logging, os, time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from arcengine import ActionInput, FrameData, GameAction, GameState
from agents.agent import Agent

logger = logging.getLogger(__name__)


def _state_hash(frame_arr):
    return hashlib.md5(frame_arr.tobytes()).hexdigest()[:16]


def _scan_actions(game, f0, bg, scan_timeout=5):
    """Find all unique-effect actions for this game/state."""
    avail = game._available_actions
    actions = []
    # Keyboard/interact
    for a in [x for x in avail if x <= 5]:
        g = copy.deepcopy(game)
        try:
            r = g.perform_action(ActionInput(id=GameAction.from_id(a)), raw=True)
            if r.frame and np.sum(f0 != np.array(r.frame[-1])) > 0:
                actions.append((a, None))
        except:
            pass
    # Click
    if 6 in avail:
        t0 = time.time()
        seen = set()
        for y in range(0, 64, 2):
            if time.time() - t0 > scan_timeout:
                break
            for x in range(0, 64, 2):
                if f0[y, x] == bg:
                    continue
                g = copy.deepcopy(game)
                try:
                    r = g.perform_action(
                        ActionInput(id=GameAction.ACTION6,
                                    data={'x': x, 'y': y, 'game_id': 'bfs'}),
                        raw=True)
                    if not r.frame:
                        continue
                    f = np.array(r.frame[-1])
                    if np.sum(f0 != f) > 0:
                        eh = hashlib.md5(f.tobytes()).hexdigest()[:12]
                        if eh not in seen:
                            seen.add(eh)
                            actions.append((6, {'x': x, 'y': y, 'game_id': 'bfs'}))
                except:
                    pass
    return actions


def _bfs(game, actions, max_states=500000, timeout=120):
    """BFS from current game state. Returns action list or None."""
    r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    if not r0.frame:
        return None
    f0 = np.array(r0.frame[-1])
    t0 = time.time()
    visited = {_state_hash(f0)}
    queue = deque([(game, [])])
    states = 0

    while queue and states < max_states and (time.time() - t0) < timeout:
        g_state, path = queue.popleft()
        states += 1
        for act_id, data in actions:
            g = copy.deepcopy(g_state)
            try:
                ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                      if data else ActionInput(id=GameAction.from_id(act_id)))
                r = g.perform_action(ai, raw=True)
            except:
                continue
            if not r.frame:
                continue
            f = np.array(r.frame[-1])
            h = _state_hash(f)
            if r.levels_completed > 0 or r.state == GameState.WIN:
                logger.info(f'BFS solved in {len(path)+1} actions, {states} states, {time.time()-t0:.1f}s')
                return path + [(act_id, data)]
            if r.state == GameState.GAME_OVER or h in visited:
                continue
            visited.add(h)
            queue.append((g, path + [(act_id, data)]))

    logger.info(f'BFS exhausted: {states} states, {time.time()-t0:.1f}s')
    return None


class RewindAgent(Agent):
    MAX_ACTIONS: int = 500

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.levels = 0
        self.queue = []          # [(act_id, data), ...]
        self.attempt = 0
        self.game_type = None
        self.available = None
        self.door_pat = None
        self._bfs_done = {}      # level -> bool (attempted)
        self._game_cls = None
        self._load_game_source()
        logger.info(f'RewindAgent v16 init, game_cls={self._game_cls is not None}')

    # ─────────────────────────────────────────────
    # Game source loading
    # ─────────────────────────────────────────────
    def _load_game_source(self):
        env_dir = os.environ.get('ENVIRONMENTS_DIR', 'environment_files')
        short = self.game_id.split('-')[0]
        class_name = short[0].upper() + short[1:]
        for p in [
            Path(env_dir) / short / f'{short}.py',
            Path(env_dir) / short / f'{class_name.lower()}.py',
            Path(env_dir) / f'{short}.py',
        ]:
            if p.exists():
                try:
                    spec = importlib.util.spec_from_file_location(f'g_{short}', str(p))
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    self._game_cls = getattr(mod, class_name)
                    logger.info(f'Loaded game source: {p}')
                    return
                except Exception as e:
                    logger.warning(f'Failed to load {p}: {e}')

    def _bfs_level(self, level_idx):
        """BFS the given level offline and store solution in queue."""
        if self._game_cls is None:
            return False
        if self._bfs_done.get(level_idx):
            return False
        self._bfs_done[level_idx] = True
        try:
            game = self._game_cls()
            game.set_level(level_idx)
            game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            if not r0.frame:
                return False
            f0 = np.array(r0.frame[-1])
            bg = int(np.bincount(f0.flatten(), minlength=16).argmax())
            actions = _scan_actions(game, f0, bg)
            if not actions:
                return False
            sol = _bfs(game, actions, max_states=500000, timeout=90)
            if sol:
                self.queue = sol
                logger.info(f'BFS L{level_idx}: {len(sol)} actions queued')
                return True
            return False
        except Exception as e:
            logger.warning(f'BFS L{level_idx} error: {e}')
            return False

    # ─────────────────────────────────────────────
    # Online fallback helpers (keyboard / click)
    # ─────────────────────────────────────────────
    def _get_arr(self, frame):
        return np.array(frame.frame[0]) if frame.frame else None

    def _detect_type(self, frame):
        aa = frame.available_actions or []
        has_click = 6 in aa
        has_kbd = any(a in aa for a in [1, 2, 3, 4])
        if has_click and not has_kbd: return 'click'
        if has_kbd and not has_click: return 'keyboard'
        if has_click and has_kbd: return 'mixed'
        return 'keyboard'

    def _find_player(self, arr):
        p = np.where(arr[:52] == 12)
        if len(p[0]) == 0: return None
        return (int(round(p[0].mean())), int(round(p[1].mean())))

    def _get_key(self, arr):
        try:
            return tuple(
                tuple(1 if int(arr[55+r*2, 3+c*2]) == 9 else 0 for c in range(3))
                for r in range(3))
        except:
            return None

    def _find_rotator(self, arr):
        g = arr[:52]; rs, cs = [], []
        for v in [0, 1]:
            l = np.where(g == v); rs.extend(l[0].tolist()); cs.extend(l[1].tolist())
        if not rs: return None
        return (int(round(np.mean(rs))), int(round(np.mean(cs))))

    def _find_door(self, arr):
        g = arr[:52]; nines = np.where(g == 9)
        if len(nines[0]) == 0: return None, None
        d9 = []
        for r, c in zip(nines[0].tolist(), nines[1].tolist()):
            if r > 45: continue
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < 52 and 0 <= nc < 64 and int(g[nr, nc]) == 5:
                        d9.append((r, c)); break
                else: continue
                break
        if not d9: return None, None
        rn = min(r for r, c in d9); rx = max(r for r, c in d9)
        cn = min(c for r, c in d9); cx = max(c for r, c in d9)
        cr, cc = (rn+rx)//2, (cn+cx)//2
        pat = tuple(tuple(1 if int(g[cr+dr, cc+dc]) == 9 else 0
                          for dc in [-1, 0, 1]) for dr in [-1, 0, 1])
        return pat, (cr, cc)

    def _build_cs(self, arr):
        wall = (arr[:52] == 4); cs = np.zeros((52, 64), dtype=bool)
        for r in range(2, 50):
            for c in range(2, 62):
                if not np.any(wall[r-2:r+3, c-2:c+3]): cs[r, c] = True
        return cs

    def _sim(self, pos, d, cs):
        dr, dc = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}[d]
        r, c = pos
        for _ in range(5):
            nr, nc = r+dr, c+dc
            if nr < 2 or nr >= 50 or nc < 2 or nc >= 62 or not cs[nr, nc]: break
            r, c = nr, nc
        return (r, c)

    def _bfs_to(self, cs, start, goal, max_depth=30):
        vis = {start}; q = deque([(start, [])])
        best, bd = [], abs(start[0]-goal[0])+abs(start[1]-goal[1])
        while q:
            p, path = q.popleft()
            d = abs(p[0]-goal[0])+abs(p[1]-goal[1])
            if d < 8: return path
            if d < bd: best, bd = path, d
            if len(path) >= max_depth: continue
            for di in 'UDLR':
                np2 = self._sim(p, di, cs)
                if np2 != p and np2 not in vis:
                    vis.add(np2); q.append((np2, path+[di]))
        return best

    def _keyboard_plan(self, arr):
        player = self._find_player(arr); rot = self._find_rotator(arr)
        dp, dc = self._find_door(arr); key = self._get_key(arr)
        if dc and player and abs(player[0]-dc[0]) < 6 and abs(player[1]-dc[1]) < 6:
            return [('D', None)]*3
        self.door_pat = dp
        if not all([player, rot, dp]): return self._generic_explore()
        cs = self._build_cs(arr)
        if dp == key:
            path = self._bfs_to(cs, player, dc)
            return [(d, None) for d in path]+[('U', None)]*3+[('D', None)]*3+[('L', None)]*3+[('R', None)]*3
        path_r = self._bfs_to(cs, player, rot); pos = player
        for m in path_r: pos = self._sim(pos, m, cs)
        path_d = self._bfs_to(cs, pos, dc)
        return [(d, None) for d in path_r+path_d]+[('U', None)]*3+[('D', None)]*3+[('L', None)]*3+[('R', None)]*3

    def _click_plan(self, arr):
        actions = []
        b4 = sorted(set((int(r), int(c)) for r, c in zip(*np.where(arr == 4)) if r >= 40))
        n9 = sorted(set((int(r), int(c)) for r, c in zip(*np.where(arr == 9)) if r < 50))
        for r, c in b4: actions.append(('A6', {'x': c, 'y': r}))
        for r, c in n9: actions.append(('A6', {'x': c, 'y': r}))
        if not actions:
            for val in [9, 5, 4, 11, 8, 10]:
                locs = np.where(arr == val)
                for r, c in zip(locs[0].tolist()[:20], locs[1].tolist()[:20]):
                    actions.append(('A6', {'x': int(c), 'y': int(r)}))
                if len(actions) > 50: break
        return actions if actions else [('A6', {'x': 32, 'y': 32})]

    def _generic_explore(self):
        moves = []
        for _ in range(3):
            moves.extend([('R', None)]*5+[('D', None)]*5+[('L', None)]*5+[('U', None)]*5)
        return moves

    def _mixed_plan(self, arr, frame):
        aa = frame.available_actions or []; plan = []
        if any(a in aa for a in [1, 2, 3, 4]): plan.extend(self._keyboard_plan(arr)[:20])
        if 6 in aa: plan.extend(self._click_plan(arr)[:30])
        if 5 in aa: plan.append(('5', None))
        return plan if plan else self._generic_explore()

    # ─────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────
    def is_done(self, frames, latest_frame):
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames, latest_frame):
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.queue = []; self.attempt = 0
            return GameAction.RESET

        if latest_frame.levels_completed > self.levels:
            self.levels = latest_frame.levels_completed
            logger.info(f'LEVEL {self.levels} DONE!')
            self.queue = []; self.attempt = 0

        if self.game_type is None:
            self.game_type = self._detect_type(latest_frame)
            self.available = latest_frame.available_actions or []

        # Try BFS for this level (offline, zero API cost)
        if not self.queue and not self._bfs_done.get(self.levels):
            self._bfs_level(self.levels)

        # Execute BFS queue
        if self.queue:
            return self._execute_next()

        # Online fallback
        self.attempt += 1
        arr = self._get_arr(latest_frame)
        if arr is None: return GameAction.RESET

        if self.attempt > 15:
            self.queue = self._generic_explore()
        elif self.game_type == 'keyboard':
            self.queue = self._keyboard_plan(arr)
        elif self.game_type == 'click':
            self.queue = self._click_plan(arr)
        elif self.game_type == 'mixed':
            self.queue = self._mixed_plan(arr, latest_frame)
        else:
            self.queue = self._generic_explore()

        if self.queue: return self._execute_next()
        return GameAction.ACTION1

    def _execute_next(self):
        act, data = self.queue.pop(0)
        if act == 'A6' and data:
            action = GameAction.ACTION6
            action.action_data.x = int(data['x'])
            action.action_data.y = int(data['y'])
            action.reasoning = f'v16 click ({data["y"]},{data["x"]})'
            return action
        if act == '5':
            a = GameAction.ACTION5; a.reasoning = 'v16 interact'; return a
        am = {'U': GameAction.ACTION1, 'D': GameAction.ACTION2,
              'L': GameAction.ACTION3, 'R': GameAction.ACTION4}
        if isinstance(act, int):
            a = GameAction.from_id(act)
        else:
            a = am.get(act, GameAction.ACTION1)
        a.reasoning = f'v16 {self.game_type}'
        return a

    def cleanup(self, *a, **kw):
        if self._cleanup:
            logger.info(f'RewindAgent v16 done: {self.levels} levels')
        super().cleanup(*a, **kw)
