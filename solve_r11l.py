"""Analytical solver for r11l — body-follows-legs centroid puzzle."""
import importlib.util, numpy as np
from arcengine import ActionInput, GameAction

spec = importlib.util.spec_from_file_location('r11l', 'environment_files/r11l/r11l.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def solve_r11l_level(g):
    """Solve current level by moving legs so body centroid overlaps target."""
    brdck = g.brdck
    actions_taken = []
    
    for name, data in brdck.items():
        body = data['kignw']
        target = data.get('xwdrv')
        legs = data['mdpcc']
        
        if not target or not legs:
            continue
        
        # Body pos = centroid_of_leg_centers - body_size/2
        # For collision: body rect must overlap target rect
        # Best: put body center = target center
        # body_center = centroid_of_leg_centers
        # So we want: mean(leg_center_x) = target_center_x, same for y
        # With N legs, easiest: put ALL legs at the same center point
        # That way centroid = that point, body = that point - body_size/2
        
        tcx = target.x + target.width // 2
        tcy = target.y + target.height // 2
        
        n = len(legs)
        target_positions = []
        # Place legs spread around target center so centroid = target center
        # Spread horizontally by ±offset to avoid overlap
        for i in range(n):
            # Offset from center: leg i gets offset (i - (n-1)/2) * spread
            spread = 6  # pixels between leg centers
            offset_x = int((i - (n-1)/2) * spread)
            cx = tcx + offset_x
            cy = tcy
            lx = cx - legs[i].width // 2
            ly = cy - legs[i].height // 2
            lx = max(0, min(58, lx))
            ly = max(0, min(58, ly))
            target_positions.append((lx, ly))
        
        for i, leg in enumerate(legs):
            tx, ty = target_positions[i]
            
            # Click on leg to select it
            click_x = leg.x + leg.width // 2
            click_y = leg.y + leg.height // 2
            a = ActionInput(id=GameAction.ACTION6.value, data={'x': click_x, 'y': click_y})
            r = g.perform_action(a, raw=True)
            actions_taken.append(('select', click_x, click_y))
            
            # Click at position where we want leg center to end up
            move_x = tx + leg.width // 2
            move_y = ty + leg.height // 2
            # Clamp to grid
            move_x = max(2, min(61, move_x))
            move_y = max(2, min(61, move_y))
            a = ActionInput(id=GameAction.ACTION6.value, data={'x': move_x, 'y': move_y})
            r = g.perform_action(a, raw=True)
            actions_taken.append(('move', move_x, move_y))
            
            # Animation may complete instantly (gfwuu=1) or need stepping
            for _ in range(50):
                if not g.bmtib:
                    break
                a = ActionInput(id=GameAction.ACTION6.value, data={'x': 0, 'y': 0})
                r = g.perform_action(a, raw=True)
    
    return actions_taken

# Test across all levels
g = mod.R11l()
r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
total_levels = len(mod.levels)
solved = 0

for attempt in range(total_levels):
    lvl = g._current_level_index
    print(f'\n=== Level {lvl} ===')
    
    # Print level info
    for name, data in g.brdck.items():
        body = data['kignw']
        target = data.get('xwdrv')
        legs = data['mdpcc']
        print(f'  Body {name}: pos=({body.x if body else "?"},{body.y if body else "?"}) '
              f'target=({target.x if target else "?"},{target.y if target else "?"}) '
              f'legs={len(legs)}')
    
    prev_level = g._current_level_index
    actions = solve_r11l_level(g)
    
    # Check if level advanced
    if g._current_level_index != prev_level:
        print(f'  SOLVED! Advanced to level {g._current_level_index} ({len(actions)} actions)')
        solved += 1
    else:
        # Check body-target collision
        for name, data in g.brdck.items():
            body = data['kignw']
            target = data.get('xwdrv')
            if body and target:
                print(f'  Body ({body.x},{body.y}) Target ({target.x},{target.y}) Collides={body.collides_with(target)}')
        print(f'  NOT SOLVED. Actions used: {g._action_count}')
        break

print(f'\n=== TOTAL: {solved}/{total_levels} levels ===')
