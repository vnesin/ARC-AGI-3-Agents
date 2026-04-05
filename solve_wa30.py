"""Planner-based solver for wa30 (grab-and-drag puzzle)."""
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

def solve_level(level_idx):
    g = mod.Wa30()
    if level_idx > 0:
        g.set_level(level_idx)
    r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    
    player = g.current_level.get_sprites_by_tag('wbmdvjhthc')[0]
    targets = g.current_level.get_sprites_by_tag('geezpjgiyd')
    holes = g.current_level.get_sprites_by_tag('fsjjayjoeg')
    
    target_positions = [(t.x, t.y) for t in targets]
    
    # Find valid goal positions within hole regions
    goal_positions = []
    for x in range(0, 64, STEP):
        for y in range(0, 64, STEP):
            if g.shbxbhnhjc((x, y)):
                goal_positions.append((x, y))
    
    print(f'L{level_idx}: player=({player.x},{player.y}) targets={target_positions} goals={goal_positions} steps={g.kuncbnslnm.current_steps}')
    
    if len(goal_positions) < len(target_positions):
        print(f'  Not enough goal positions ({len(goal_positions)}) for targets ({len(target_positions)})')
        return None
    
    # Try all assignments of targets to goals
    best_actions = None
    best_len = float('inf')
    
    for goal_perm in itertools.permutations(range(len(goal_positions)), len(target_positions)):
        selected_goals = [goal_positions[i] for i in goal_perm]
        
        for target_perm in itertools.permutations(range(len(target_positions))):
            actions = try_plan(g, player, target_positions, selected_goals, target_perm)
            if actions is not None and len(actions) < best_len:
                best_len = len(actions)
                best_actions = actions
    
    return best_actions

def try_plan(g, player, target_positions, goal_positions, order):
    all_actions = []
    current_player = (player.x, player.y)
    blocked = set(g.pkbufziase)
    
    for i in range(len(order)):
        target_idx = order[i]
        tpos = target_positions[target_idx]
        gpos = goal_positions[i]
        
        result = plan_grab_drag(current_player, tpos, gpos, blocked)
        if result is None:
            return None
        
        actions, new_player_pos = result
        all_actions.extend(actions)
        current_player = new_player_pos
        blocked.discard(tpos)
        blocked.add(gpos)
    
    return all_actions

def plan_grab_drag(player_pos, target_pos, goal_pos, blocked):
    """Plan: navigate to target, grab, drag to goal, release."""
    # Try 4 approach directions
    # Direction from player to target after positioning
    approach_configs = [
        # (offset_x, offset_y, facing_action_to_set_rotation)
        (0, -STEP, (0, -1)),   # target above player, face UP
        (0, STEP, (0, 1)),     # target below player, face DOWN  
        (-STEP, 0, (-1, 0)),   # target left of player, face LEFT
        (STEP, 0, (1, 0)),     # target right of player, face RIGHT
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
        
        # For dragging, both player and target move. Check target path too.
        drag_blocked = temp_blocked.copy()
        drag_path = bfs_path_with_cargo(grab_pos, release_pos, offset, drag_blocked)
        if drag_path is None:
            continue
        
        nav_actions = path_to_actions(nav_path)
        
        # Ensure facing correct direction before grab
        if len(nav_path) >= 2:
            last_dx = nav_path[-1][0] - nav_path[-2][0]
            last_dy = nav_path[-1][1] - nav_path[-2][1]
            if (last_dx, last_dy) != (offset[0]//abs(offset[0]) if offset[0] else 0,
                                       offset[1]//abs(offset[1]) if offset[1] else 0):
                # Need to face correctly. Step away and back.
                away = (grab_pos[0] - offset[0], grab_pos[1] - offset[1])
                if away not in blocked and 0 <= away[0] < 64 and 0 <= away[1] < 64:
                    nav_actions.extend(path_to_actions([grab_pos, away]))
                    nav_actions.extend(path_to_actions([away, grab_pos]))
        elif len(nav_path) == 1:
            # Already at grab_pos, need to set facing
            away = (grab_pos[0] - offset[0], grab_pos[1] - offset[1])
            if away not in blocked and 0 <= away[0] < 64 and 0 <= away[1] < 64:
                nav_actions.extend(path_to_actions([grab_pos, away]))
                nav_actions.extend(path_to_actions([away, grab_pos]))
        
        nav_actions.append(5)  # grab
        drag_actions = path_to_actions(drag_path)
        drag_actions.append(5)  # release
        
        total = nav_actions + drag_actions
        if len(total) < best_len:
            best_len = len(total)
            best = (total, release_pos)
    
    return best

def bfs_path_with_cargo(start, goal, cargo_offset, blocked):
    """BFS where both player and cargo (at player+offset) must be free."""
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
            # Both positions must be free (except current player/cargo positions)
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

def test_solution(level_idx, actions):
    g = mod.Wa30()
    if level_idx > 0:
        g.set_level(level_idx)
    r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    
    for i, a in enumerate(actions):
        ai = ActionInput(id=GameAction.from_id(a))
        r = g.perform_action(ai, raw=True)
        if hasattr(r, 'levels_completed') and r.levels_completed and r.levels_completed > 0:
            print(f'  L{level_idx} SOLVED in {i+1} actions!')
            return True
        if g.kuncbnslnm.current_steps <= 0:
            print(f'  L{level_idx} OUT OF STEPS after {i+1} actions')
            return False
    
    targets = g.current_level.get_sprites_by_tag('geezpjgiyd')
    won = g.ymzfopzgbq()
    print(f'  L{level_idx} end: won={won} steps_left={g.kuncbnslnm.current_steps}')
    for t in targets:
        print(f'    target ({t.x},{t.y}) on_hole={g.shbxbhnhjc((t.x,t.y))} connected={t in g.zmqreragji}')
    return False

# Test all levels
for lvl in range(9):
    actions = solve_level(lvl)
    if actions:
        print(f'Solution found: {len(actions)} actions')
        test_solution(lvl, actions)
    else:
        print(f'No solution found for level {lvl}')
