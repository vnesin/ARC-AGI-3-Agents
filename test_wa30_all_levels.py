"""Test wa30 solver across all 9 levels sequentially."""
import importlib.util, numpy as np, itertools
from arcengine import ActionInput, GameAction
from collections import deque

STEP = 4

spec = importlib.util.spec_from_file_location('wa30', 'environment_files/wa30/wa30.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def bfs_path(start, goal, blocked):
    if start == goal:
        return [start]
    q = deque([(start, [start])])
    visited = {start}
    while q:
        pos, path = q.popleft()
        for dx, dy in [(0,-STEP),(0,STEP),(-STEP,0),(STEP,0)]:
            npos = (pos[0]+dx, pos[1]+dy)
            if npos in visited or npos in blocked or npos[0]<0 or npos[0]>=64 or npos[1]<0 or npos[1]>=64:
                continue
            visited.add(npos)
            new_path = path + [npos]
            if npos == goal:
                return new_path
            q.append((npos, new_path))
    return None

def bfs_path_with_cargo(start, goal, cargo_offset, blocked):
    if start == goal:
        return [start]
    q = deque([(start, [start])])
    visited = {start}
    while q:
        pos, path = q.popleft()
        for dx, dy in [(0,-STEP),(0,STEP),(-STEP,0),(STEP,0)]:
            npos = (pos[0]+dx, pos[1]+dy)
            cargo_pos = (npos[0]+cargo_offset[0], npos[1]+cargo_offset[1])
            if npos in visited:
                continue
            if npos[0]<0 or npos[0]>=64 or npos[1]<0 or npos[1]>=64:
                continue
            if cargo_pos[0]<0 or cargo_pos[0]>=64 or cargo_pos[1]<0 or cargo_pos[1]>=64:
                continue
            cur_cargo = (pos[0]+cargo_offset[0], pos[1]+cargo_offset[1])
            if npos in blocked and npos != cur_cargo:
                continue
            if cargo_pos in blocked and cargo_pos != pos:
                continue
            visited.add(npos)
            new_path = path + [npos]
            if npos == goal:
                return new_path
            q.append((npos, new_path))
    return None

def path_to_actions(path):
    actions = []
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        if dy < 0: actions.append(1)
        elif dy > 0: actions.append(2)
        elif dx < 0: actions.append(3)
        elif dx > 0: actions.append(4)
    return actions

def plan_grab_drag(player_pos, target_pos, goal_pos, blocked):
    approach_configs = [
        (0, -STEP, (0, -1)),
        (0, STEP, (0, 1)),
        (-STEP, 0, (-1, 0)),
        (STEP, 0, (1, 0)),
    ]
    best = None
    best_len = float('inf')
    for ox, oy, facing in approach_configs:
        grab_pos = (target_pos[0] - ox, target_pos[1] - oy)
        if grab_pos in blocked and grab_pos != player_pos:
            continue
        if grab_pos[0] < 0 or grab_pos[0] >= 64 or grab_pos[1] < 0 or grab_pos[1] >= 64:
            continue
        temp_blocked = blocked - {target_pos}
        nav_path = bfs_path(player_pos, grab_pos, temp_blocked)
        if nav_path is None:
            continue
        offset = (target_pos[0] - grab_pos[0], target_pos[1] - grab_pos[1])
        release_pos = (goal_pos[0] - offset[0], goal_pos[1] - offset[1])
        if release_pos[0] < 0 or release_pos[0] >= 64 or release_pos[1] < 0 or release_pos[1] >= 64:
            continue
        if release_pos in temp_blocked:
            continue
        drag_path = bfs_path_with_cargo(grab_pos, release_pos, offset, temp_blocked)
        if drag_path is None:
            continue
        nav_actions = path_to_actions(nav_path)
        if len(nav_path) >= 2:
            last_dx = nav_path[-1][0] - nav_path[-2][0]
            last_dy = nav_path[-1][1] - nav_path[-2][1]
            needed = offset
            if (last_dx, last_dy) != (needed[0]//abs(needed[0]) if needed[0] else 0,
                                       needed[1]//abs(needed[1]) if needed[1] else 0):
                away = (grab_pos[0] - offset[0], grab_pos[1] - offset[1])
                if away not in blocked and 0 <= away[0] < 64 and 0 <= away[1] < 64:
                    nav_actions.extend(path_to_actions([grab_pos, away]))
                    nav_actions.extend(path_to_actions([away, grab_pos]))
        elif len(nav_path) == 1:
            away = (grab_pos[0] - offset[0], grab_pos[1] - offset[1])
            if away not in blocked and 0 <= away[0] < 64 and 0 <= away[1] < 64:
                nav_actions.extend(path_to_actions([grab_pos, away]))
                nav_actions.extend(path_to_actions([away, grab_pos]))
        nav_actions.append(5)
        drag_actions = path_to_actions(drag_path)
        drag_actions.append(5)
        total = nav_actions + drag_actions
        if len(total) < best_len:
            best_len = len(total)
            best = (total, release_pos)
    return best

def solve_current_level(g):
    """Solve whatever level the game is currently on."""
    player = g.current_level.get_sprites_by_tag('wbmdvjhthc')[0]
    targets = g.current_level.get_sprites_by_tag('geezpjgiyd')
    target_positions = [(t.x, t.y) for t in targets]
    
    # Find valid goal positions
    goal_positions = []
    for x in range(0, 64, STEP):
        for y in range(0, 64, STEP):
            if g.shbxbhnhjc((x, y)):
                goal_positions.append((x, y))
    
    if len(goal_positions) < len(target_positions):
        return None
    
    blocked = set(g.pkbufziase)
    
    # Try all permutations of goals and target order
    for goal_perm in itertools.permutations(range(len(goal_positions)), len(target_positions)):
        selected_goals = [goal_positions[i] for i in goal_perm]
        for target_order in itertools.permutations(range(len(target_positions))):
            actions = []
            cur_player = (player.x, player.y)
            cur_blocked = set(blocked)
            success = True
            for i in range(len(target_order)):
                ti = target_order[i]
                tpos = target_positions[ti]
                gpos = selected_goals[i]
                result = plan_grab_drag(cur_player, tpos, gpos, cur_blocked)
                if result is None:
                    success = False
                    break
                step_actions, new_player = result
                actions.extend(step_actions)
                cur_player = new_player
                cur_blocked.discard(tpos)
                cur_blocked.add(gpos)
            if success:
                return actions
    return None

# Run through all levels
g = mod.Wa30()
r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
total_levels = 0

for level_attempt in range(9):
    lvl_idx = g._current_level_index
    print(f'\n=== Level {lvl_idx} ===')
    player = g.current_level.get_sprites_by_tag('wbmdvjhthc')[0]
    targets = g.current_level.get_sprites_by_tag('geezpjgiyd')
    print(f'Player: ({player.x},{player.y}), Targets: {[(t.x,t.y) for t in targets]}, Steps: {g.kuncbnslnm.current_steps}')
    
    actions = solve_current_level(g)
    if actions is None:
        print(f'FAILED to solve level {lvl_idx}')
        break
    
    print(f'Solution: {len(actions)} actions')
    
    prev_level = g._current_level_index
    for i, a in enumerate(actions):
        ai = ActionInput(id=GameAction.from_id(a))
        r = g.perform_action(ai, raw=True)
        if g._current_level_index != prev_level:
            print(f'Level advanced to {g._current_level_index} after action {i+1}!')
            total_levels += 1
            break
    else:
        # Check win
        print(f'Actions exhausted. Won={g.ymzfopzgbq()}, Steps left={g.kuncbnslnm.current_steps}')
        if g.ymzfopzgbq():
            # Need one more step for the game to process win
            r = g.perform_action(ActionInput(id=GameAction.from_id(1)), raw=True)
            if g._current_level_index != prev_level:
                print(f'Level advanced after extra step!')
                total_levels += 1

print(f'\n=== TOTAL: {total_levels}/9 levels completed ===')
