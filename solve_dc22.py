"""Solver for dc22 — grid puzzle with button toggles."""
import importlib.util, copy, time
from arcengine import ActionInput, GameAction
from collections import deque

spec = importlib.util.spec_from_file_location('dc22', 'environment_files/dc22/dc22.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def step_anim(game):
    for _ in range(30):
        if not (game.uuehztercxf or game.jnmawhhrfhh or game.fgxfjbqnmgt):
            break
        game.perform_action(ActionInput(id=GameAction.from_id(1)), raw=True)

def button_state(game):
    states = []
    for s in game.current_level.get_sprites():
        if 'wbze' in (s.tags or []):
            states.append((s.name, str(s._interaction)))
    return tuple(sorted(states))

def find_buttons(game):
    """Find clickable buttons and compute their display coordinates."""
    buttons = []
    jpugs = game.current_level.get_sprites_by_tag('jpug')
    for j in jpugs:
        tag = next((t for t in j.tags if len(t) == 1), None)
        if tag:
            # Convert grid coords to display coords
            grid_x = j.x + j.width // 2
            grid_y = j.y + j.height // 2
            # Find display y that maps to grid y
            for dy in range(64):
                result = game.camera.display_to_grid(grid_x, dy)
                if result and result[1] == grid_y:
                    buttons.append((tag, grid_x, dy))
                    break
            else:
                # Fallback: try offset
                buttons.append((tag, grid_x, grid_y + 10))
    return buttons

def solve_dc22_level(game, timeout=120):
    """BFS with movement + button clicks."""
    buttons = find_buttons(game)
    print(f'  Buttons: {buttons}')
    
    t0 = time.time()
    queue = deque()
    visited = set()
    s0 = (game.fdvakicpimr.x, game.fdvakicpimr.y, button_state(game))
    visited.add(s0)
    queue.append((copy.deepcopy(game), []))
    
    click_actions = []
    for tag, cx, cy in buttons:
        click_actions.append((tag, ActionInput(id=GameAction.ACTION6.value, data={'x': cx, 'y': cy})))
    
    count = 0
    while queue and time.time() - t0 < timeout:
        gm, path = queue.popleft()
        count += 1
        
        if count % 1000 == 0:
            print(f'  {count} states, {len(visited)} visited, depth={len(path)}, {time.time()-t0:.0f}s')
        
        actions = [
            (1, ActionInput(id=GameAction.from_id(1))),
            (2, ActionInput(id=GameAction.from_id(2))),
            (3, ActionInput(id=GameAction.from_id(3))),
            (4, ActionInput(id=GameAction.from_id(4))),
        ] + click_actions
        
        for label, action in actions:
            g2 = copy.deepcopy(gm)
            g2.perform_action(action, raw=True)
            step_anim(g2)
            
            if g2._current_level_index > gm._current_level_index:
                return path + [label], count
            
            if g2.fdvakicpimr.x == g2.bqxa.x and g2.fdvakicpimr.y == g2.bqxa.y:
                return path + [label], count
            
            s = (g2.fdvakicpimr.x, g2.fdvakicpimr.y, button_state(g2))
            if s not in visited:
                visited.add(s)
                queue.append((g2, path + [label]))
    
    return None, count

# Run all levels
g = mod.Dc22()
r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
total_solved = 0

for attempt in range(6):
    lvl = g._current_level_index
    print(f'\n=== Level {lvl} ===')
    print(f'  Player: ({g.fdvakicpimr.x},{g.fdvakicpimr.y}), Goal: ({g.bqxa.x},{g.bqxa.y})')
    
    prev_level = lvl
    solution, states = solve_dc22_level(g, timeout=180)
    
    if solution:
        print(f'  SOLVED in {len(solution)} moves ({states} states): {solution}')
        # Execute the solution
        g_fresh = copy.deepcopy(g)
        buttons = find_buttons(g)
        for move in solution:
            if isinstance(move, int):
                g.perform_action(ActionInput(id=GameAction.from_id(move)), raw=True)
            else:
                # Button click
                for tag, cx, cy in buttons:
                    if tag == move:
                        g.perform_action(ActionInput(id=GameAction.ACTION6.value, data={'x': cx, 'y': cy}), raw=True)
                        break
            step_anim(g)
        
        if g._current_level_index > prev_level:
            total_solved += 1
            print(f'  Level advanced to {g._current_level_index}!')
        else:
            print(f'  Solution executed but level did not advance')
            break
    else:
        print(f'  FAILED ({states} states explored)')
        break

print(f'\n=== TOTAL: {total_solved}/6 levels solved ===')
