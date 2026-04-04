#!/usr/bin/env python3
"""Standalone v17 BFS test — no agents package import to avoid memory bloat."""
import copy
import gc
import hashlib
import importlib.util
import os
import sys
import time
from collections import deque

import numpy as np
from arcengine import ActionInput, GameAction, GameState


def state_hash(f):
    return hashlib.md5(f.tobytes()).hexdigest()[:16]


def replay_path(game_cls, path, level_idx=0):
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


def scan_actions(game, f0, bg, timeout=15.0):
    avail = game._available_actions
    actions = []
    seen = set()

    for a in [x for x in avail if x != 6]:
        g = copy.deepcopy(game)
        try:
            r = g.perform_action(ActionInput(id=GameAction.from_id(a)), raw=True)
            if r.frame:
                f1 = np.array(r.frame[-1])
                if np.any(f0 != f1):
                    eh = hashlib.md5(f1.tobytes()).hexdigest()[:12]
                    if eh not in seen:
                        seen.add(eh)
                        actions.append((a, None))
        except:
            pass
        del g

    if 6 in avail:
        click_fx = set()
        t0 = time.time()
        positions = list(zip(*np.where(f0 != bg)))
        positions.sort(key=lambda p: (f0[p[0], p[1]], p[0], p[1]))
        tested = set()
        for y, x in positions:
            if time.time() - t0 > timeout:
                break
            if (x, y) in tested:
                continue
            tested.add((x, y))
            g = copy.deepcopy(game)
            try:
                r = g.perform_action(
                    ActionInput(id=GameAction.ACTION6, data={"x": int(x), "y": int(y), "game_id": "scan"}),
                    raw=True,
                )
                if r.frame:
                    f1 = np.array(r.frame[-1])
                    if np.any(f0 != f1):
                        eh = hashlib.md5(f1.tobytes()).hexdigest()[:12]
                        if eh not in click_fx:
                            click_fx.add(eh)
                            actions.append((6, {"x": int(x), "y": int(y), "game_id": "bfs"}))
                            if r.levels_completed > 0:
                                return [(6, {"x": int(x), "y": int(y), "game_id": "bfs"})]
            except:
                pass
            del g

    return actions


def bfs(game_cls, actions, level_idx=0, max_states=5000, timeout=120.0):
    game = game_cls()
    if hasattr(game, "set_level"):
        game.set_level(level_idx)
    r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    if not r0.frame:
        return None
    del game

    f0 = np.array(r0.frame[-1])
    t0 = time.time()
    visited = {state_hash(f0)}
    queue = deque([[]])
    states = 0
    branching = len(actions)

    if branching <= 4:
        max_depth = 50
    elif branching <= 10:
        max_depth = 25
    elif branching <= 30:
        max_depth = 12
    else:
        max_depth = 8

    while queue and states < max_states and (time.time() - t0) < timeout:
        path = queue.popleft()
        states += 1
        if len(path) >= max_depth:
            continue

        for act_id, data in actions:
            candidate = path + [(act_id, data)]
            try:
                _, r = replay_path(game_cls, candidate, level_idx)
            except:
                continue
            if not r.frame:
                continue
            f = np.array(r.frame[-1])
            if r.levels_completed > 0 or r.state == GameState.WIN:
                return candidate
            if r.state == GameState.GAME_OVER:
                continue
            h = state_hash(f)
            if h in visited:
                continue
            visited.add(h)
            queue.append(candidate)

    return None


# Main
games = sorted(os.listdir("environment_files"))
results = {}

for game in games:
    gpath = f"environment_files/{game}/{game}.py"
    if not os.path.exists(gpath):
        continue
    class_name = game[0].upper() + game[1:]

    print(f"\n=== {game} ===", flush=True)

    try:
        spec = importlib.util.spec_from_file_location(f"g_{game}", gpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = getattr(mod, class_name)

        levels_solved = 0
        game_start = time.time()

        for level in range(10):
            if time.time() - game_start > 180:
                print(f"  Timeout L{level}", flush=True)
                break

            g = cls()
            if hasattr(g, "set_level"):
                g.set_level(level)
            r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            if not r.frame:
                break

            f0 = np.array(r.frame[-1])
            bg = int(np.bincount(f0.flatten(), minlength=16).argmax())
            actions = scan_actions(g, f0, bg, timeout=10)
            del g

            if not actions:
                print(f"  L{level}: no actions", flush=True)
                break

            b = len(actions)
            sol = bfs(cls, actions, level_idx=level, max_states=5000, timeout=120)

            if sol:
                levels_solved += 1
                print(f"  L{level}: SOLVED ({len(sol)} act, {b}b)", flush=True)
            else:
                print(f"  L{level}: FAILED ({b}b)", flush=True)
                break

            gc.collect()

        results[game] = levels_solved
        # Unload module
        del mod, cls
        gc.collect()

    except Exception as e:
        print(f"  ERROR: {str(e)[:80]}", flush=True)
        results[game] = 0

print(f"\n{'='*50}", flush=True)
print("SUMMARY", flush=True)
print(f"{'='*50}", flush=True)
total = 0
for g in sorted(results):
    lvl = results[g]
    total += lvl
    s = f"L0-L{lvl-1}" if lvl > 0 else "-"
    print(f"  {g:8s}: {s}")
gs = sum(1 for v in results.values() if v > 0)
print(f"\n{gs}/{len(results)} games, {total} levels")
