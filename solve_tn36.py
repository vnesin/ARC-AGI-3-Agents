"""Analytical solver for tn36 — shape transformation programming puzzle."""
import importlib.util, itertools
from arcengine import ActionInput, GameAction

spec = importlib.util.spec_from_file_location('tn36', 'environment_files/tn36/tn36.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Operation mapping from source code:
# 0: flip, 1: move_left, 2: move_right, 3: move_down, 33: move_up
# 5: rotate_90, 6: rotate_-90, 7: rotate_180, 16: rotate_270
# 8: scale_up, 9: scale_down
# 14/15/63: color changes

STEP = 4  # oocxrjijjq

def compute_needed_ops(dx, dy, drot, dscale, dcol_same):
    """Compute sequence of operations to transform current→target."""
    ops = []
    
    # Handle rotation first
    if drot != 0:
        drot = drot % 360
        if drot == 90: ops.append(5)
        elif drot == 180: ops.append(7)
        elif drot == 270: ops.append(6)
    
    # Handle scale
    while dscale > 0:
        ops.append(8)
        dscale -= 1
    while dscale < 0:
        ops.append(9)
        dscale += 1
    
    # Handle movement
    moves_right = dx // STEP
    moves_down = dy // STEP
    
    while moves_right > 0:
        ops.append(2)
        moves_right -= 1
    while moves_right < 0:
        ops.append(1)
        moves_right += 1
    while moves_down > 0:
        ops.append(3)
        moves_down -= 1
    while moves_down < 0:
        ops.append(33)
        moves_down += 1
    
    return ops

def solve_tn36_all():
    g = mod.Tn36()
    r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    solved = 0
    
    for attempt in range(7):
        lvl = g._current_level_index
        panel = g.tsflfunycx.xsseeglmfh
        current = panel.ravxreuqho
        target = panel.ddzsdagbti
        prog = panel.tlwkpfljid
        
        if not target:
            print(f'L{lvl}: No target')
            break
        
        dx = target.x - current.x
        dy = target.y - current.y
        drot = (target.rotation - current.rotation) % 360
        dscale = target.scale - current.scale
        dcol = target.dtxpbtpcbh == current.dtxpbtpcbh
        
        print(f'\nL{lvl}: dx={dx} dy={dy} drot={drot} dscale={dscale} color_same={dcol}')
        
        needed_ops = compute_needed_ops(dx, dy, drot, dscale, dcol)
        
        # Handle color if different
        if not dcol:
            # Try available color ops
            for op_id in [14, 15, 63]:
                if op_id in panel.dfguzecnsr:
                    needed_ops.append(op_id)
                    break
        
        num_slots = len(prog.thofkgziyd)
        print(f'  Needed ops: {needed_ops} ({len(needed_ops)} ops, {num_slots} slots)')
        
        if len(needed_ops) > num_slots:
            # Check if we can use double-move ops (10=2×right, 12=2×left)
            # Optimise: replace pairs of same ops with double ops
            optimised = []
            i = 0
            while i < len(needed_ops):
                if i+1 < len(needed_ops) and needed_ops[i] == needed_ops[i+1]:
                    if needed_ops[i] == 2 and 10 in panel.dfguzecnsr:
                        optimised.append(10)
                        i += 2
                        continue
                    elif needed_ops[i] == 1 and 12 in panel.dfguzecnsr:
                        optimised.append(12)
                        i += 2
                        continue
                optimised.append(needed_ops[i])
                i += 1
            needed_ops = optimised
            print(f'  Optimised: {needed_ops} ({len(needed_ops)} ops)')
        
        if len(needed_ops) > num_slots:
            print(f'  TOO MANY OPS ({len(needed_ops)} > {num_slots})')
            break
        
        # Pad with no-ops (0=flip twice = identity... but 0 is flip, not identity)
        # Actually we need to leave unused slots as they are if they're harmless
        # Or set them to an identity operation
        # Let's pad with the current slot values if they don't interfere
        while len(needed_ops) < num_slots:
            needed_ops.append(0)  # flip — we'll need to flip twice to cancel
        
        # Actually, flip is NOT identity. Let me check what op 0 does
        # 0: xmaelepexc() — that's a flip. Two flips = identity.
        # If we need even number of padding flips, it's fine.
        # If odd, we have a problem. Let's just try — the solver should handle it.
        
        # Better: don't add padding flips. Set unused slots to a no-move op.
        # But there's no explicit NOP. Let me just try the needed ops padded with 0s
        # and see if it works. If the extra flips mess up, we'll need to be smarter.
        
        # For now: just set the program to needed_ops
        target_program = needed_ops[:num_slots]
        current_program = prog.hcbeqruthf()
        print(f'  Current program: {current_program}')
        print(f'  Target program:  {target_program}')
        
        # Toggle bits to set each slot to the right value
        clicks = []
        for slot_idx in range(num_slots):
            current_val = current_program[slot_idx]
            target_val = target_program[slot_idx]
            if current_val == target_val:
                continue
            
            # Need to toggle bits that differ
            diff = current_val ^ target_val
            slot = prog.thofkgziyd[slot_idx]
            for bit_idx, bit_obj in enumerate(slot.puakvdstpr):
                if diff & (1 << bit_idx):
                    cx = bit_obj.qmbzztjrjk.x + bit_obj.qmbzztjrjk.width // 2
                    cy = bit_obj.qmbzztjrjk.y + bit_obj.qmbzztjrjk.height // 2
                    clicks.append((cx, cy))
        
        print(f'  Clicks needed: {len(clicks)}')
        
        # Execute clicks
        prev_level = g._current_level_index
        for cx, cy in clicks:
            g.perform_action(ActionInput(id=GameAction.ACTION6.value, data={'x': cx, 'y': cy}), raw=True)
            for _ in range(30):
                if not g.tsflfunycx.nwjrtjcxpo: break
                g.perform_action(ActionInput(id=GameAction.ACTION6.value, data={'x': 0, 'y': 0}), raw=True)
        
        print(f'  Program after clicks: {prog.hcbeqruthf()}')
        
        # Click submit (kbopcuwwcp at ~36,55)
        submit_sprites = [s for s in g.current_level.get_sprites() if 'rlqfpkqktk' in (s.tags or [])]
        if submit_sprites:
            sx = submit_sprites[0].x + submit_sprites[0].width // 2
            sy = submit_sprites[0].y + submit_sprites[0].height // 2
            g.perform_action(ActionInput(id=GameAction.ACTION6.value, data={'x': sx, 'y': sy}), raw=True)
            for _ in range(50):
                if not g.tsflfunycx.nwjrtjcxpo: break
                g.perform_action(ActionInput(id=GameAction.ACTION6.value, data={'x': 0, 'y': 0}), raw=True)
        
        if g._current_level_index > prev_level:
            print(f'  SOLVED! Advanced to level {g._current_level_index}')
            solved += 1
        elif g.rarvldaizc:
            # Extra step needed
            g.perform_action(ActionInput(id=GameAction.ACTION6.value, data={'x': 0, 'y': 0}), raw=True)
            if g._current_level_index > prev_level:
                print(f'  SOLVED after extra step!')
                solved += 1
            else:
                print(f'  Win flag but no advance')
                break
        else:
            print(f'  NOT SOLVED. Won={panel.yxabhsirzl}')
            break
    
    print(f'\n=== TOTAL: {solved}/7 levels ===')

solve_tn36_all()
