"""Solver for su15 — vacuum-pull fruit merge puzzle."""
import importlib.util, math
from arcengine import ActionInput, GameAction

spec = importlib.util.spec_from_file_location('su15', 'environment_files/su15/su15.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def solve_su15_all():
    g = mod.Su15()
    g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    solved = 0
    
    for attempt in range(10):
        lvl = g._current_level_index
        if lvl >= len(mod.levels):
            break
        
        goal_zones = g.rqdsgrklq
        goal_req = g.reqbygadvzmjired
        fruits = g.hmeulfxgy
        enemies = g.peiiyyzum
        
        print(f'\nL{lvl}: {len(fruits)} fruits, {len(enemies)} enemies, goal={goal_req}')
        if goal_zones:
            gz = goal_zones[0]
            gx, gy = gz.x + gz.width//2, gz.y + gz.height//2
            print(f'  Goal zone: ({gz.x},{gz.y}) {gz.width}x{gz.height} center=({gx},{gy})')
        else:
            print(f'  No goal zone!')
            break
        
        for f in fruits:
            print(f'  Fruit: ({f.x},{f.y}) level={g.amnmgwpkeb.get(f,0)}')
        
        # Strategy: drag each fruit toward the goal zone
        prev_level = g._current_level_index
        radius = g.qjlubdgly
        
        for click_num in range(30):
            if not g.hmeulfxgy:
                print(f'  No fruits left!')
                break
            
            # Find fruit closest to goal that needs to be in goal
            best_fruit = None
            best_dist = float('inf')
            for f in g.hmeulfxgy:
                fx = f.x + f.width//2
                fy = f.y + f.height//2
                d = math.sqrt((gx-fx)**2 + (gy-fy)**2)
                if d < best_dist:
                    best_dist = d
                    best_fruit = f
            
            if not best_fruit:
                break
            
            fx = best_fruit.x + best_fruit.width//2
            fy = best_fruit.y + best_fruit.height//2
            
            dx = gx - fx
            dy = gy - fy
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < 3:
                # Fruit is at goal, try next fruit or check win
                break
            
            # Click toward goal within radius
            step = min(radius - 1, dist)
            cx = int(fx + dx/dist * step)
            cy = int(fy + dy/dist * step)
            cx = max(0, min(63, cx))
            cy = max(10, min(63, cy))
            
            g.perform_action(ActionInput(id=GameAction.ACTION6.value, data={'x': cx, 'y': cy}), raw=True)
            for _ in range(50):
                if not g.anibpvotxtvdating:
                    break
                g.perform_action(ActionInput(id=GameAction.ACTION6.value, data={'x': 0, 'y': 0}), raw=True)
            
            if g._current_level_index > prev_level:
                print(f'  SOLVED in {click_num+1} clicks!')
                solved += 1
                break
            
            if g.kouxmshyjy():
                for _ in range(20):
                    g.perform_action(ActionInput(id=GameAction.ACTION6.value, data={'x': 0, 'y': 0}), raw=True)
                if g._current_level_index > prev_level:
                    print(f'  SOLVED after win check!')
                    solved += 1
                    break
        
        if g._current_level_index == prev_level:
            # Try ACTION7 (undo) to reset and retry with different strategy
            print(f'  NOT SOLVED after {click_num+1} clicks')
            break
    
    print(f'\n=== TOTAL: {solved}/{len(mod.levels)} levels ===')

solve_su15_all()
