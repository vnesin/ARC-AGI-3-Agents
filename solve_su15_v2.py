"""Optimised solver for su15 — merge-then-drag strategy."""
import importlib.util, math
from arcengine import ActionInput, GameAction

spec = importlib.util.spec_from_file_location('su15', 'environment_files/su15/su15.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def drag_toward(game, fruit, tx, ty, max_clicks=20):
    radius = game.qjlubdgly
    for _ in range(max_clicks):
        fx, fy = fruit.x+fruit.width//2, fruit.y+fruit.height//2
        dx, dy = tx-fx, ty-fy
        dist = math.sqrt(dx*dx+dy*dy)
        if dist < 4: return True
        step = min(radius-1, dist)
        cx = int(fx + dx/dist*step)
        cy = int(fy + dy/dist*step)
        cx, cy = max(0,min(63,cx)), max(10,min(63,cy))
        game.perform_action(ActionInput(id=GameAction.ACTION6.value, data={'x':cx,'y':cy}), raw=True)
        while game.anibpvotxtvdating:
            game.perform_action(ActionInput(id=GameAction.ACTION6.value, data={'x':0,'y':0}), raw=True)
        if game._current_level_index != game.level_index:
            return True
    return False

def closest_same_pair(fruits, game):
    best_d, best = float('inf'), None
    for i in range(len(fruits)):
        for j in range(i+1, len(fruits)):
            if game.amnmgwpkeb.get(fruits[i],0) != game.amnmgwpkeb.get(fruits[j],0): continue
            d = math.sqrt((fruits[i].x-fruits[j].x)**2 + (fruits[i].y-fruits[j].y)**2)
            if d < best_d: best_d, best = d, (i,j)
    return best, best_d

def solve_level(g):
    prev = g._current_level_index
    radius = g.qjlubdgly
    goal_req = g.reqbygadvzmjired
    gz = g.rqdsgrklq[0] if g.rqdsgrklq else None
    if not gz: return False
    gx, gy = gz.x+gz.width//2, gz.y+gz.height//2
    
    # Phase 1: merge fruits until we have the right level
    for merge_round in range(30):
        fruits = g.hmeulfxgy
        if not fruits: break
        if g._current_level_index != prev: return True
        
        pair, dist = closest_same_pair(fruits, g)
        if not pair:
            # No same-level pairs — check if we have the goal fruit
            break
        
        fi, fj = fruits[pair[0]], fruits[pair[1]]
        if dist > radius:
            drag_toward(g, fi, fj.x+fj.width//2, fj.y+fj.height//2)
        else:
            mid_x = (fi.x+fj.x)//2 + fi.width//2
            mid_y = (fi.y+fj.y)//2 + fi.height//2
            g.perform_action(ActionInput(id=GameAction.ACTION6.value, data={'x':mid_x,'y':mid_y}), raw=True)
            while g.anibpvotxtvdating:
                g.perform_action(ActionInput(id=GameAction.ACTION6.value, data={'x':0,'y':0}), raw=True)
        if g._current_level_index != prev: return True
    
    # Phase 2: drag highest-level fruit to goal
    fruits = g.hmeulfxgy
    if not fruits: return g._current_level_index != prev
    
    # Find highest level fruit
    best_fruit = max(fruits, key=lambda f: g.amnmgwpkeb.get(f, 0))
    drag_toward(g, best_fruit, gx, gy)
    
    # Check win
    for _ in range(20):
        g.perform_action(ActionInput(id=GameAction.ACTION6.value, data={'x':0,'y':0}), raw=True)
        if g._current_level_index != prev: return True
    
    return g._current_level_index != prev

# Run all levels
g = mod.Su15()
g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
solved = 0

for attempt in range(len(mod.levels)):
    lvl = g._current_level_index
    print(f'\nL{lvl}: {len(g.hmeulfxgy)} fruits, {len(g.peiiyyzum)} enemies, goal={g.reqbygadvzmjired}')
    
    if solve_level(g):
        solved += 1
        print(f'  SOLVED! Now on level {g._current_level_index}')
    else:
        levels = [g.amnmgwpkeb.get(f,0) for f in g.hmeulfxgy]
        print(f'  FAILED. Fruits: {levels}')
        break

print(f'\n=== TOTAL: {solved}/{len(mod.levels)} levels ===')
