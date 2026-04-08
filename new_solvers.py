"""
New solvers for ARC-AGI-3 unsolved games: cn04, sk48
Based on patterns from RewindAgent v27 (kaggle_v27.ipynb)
"""

import copy
import hashlib
import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from arcengine import ActionInput, GameAction, GameState

logger = logging.getLogger(__name__)


def _state_hash(game, frame):
    """Hash combining frame + hidden state."""
    parts = [frame.tobytes()]
    if hasattr(game, '_get_hidden_state'):
        try:
            hs = game._get_hidden_state()
            if hs is not None:
                parts.append(np.asarray(hs).tobytes())
        except:
            pass
    return hashlib.md5(b''.join(parts)).hexdigest()[:16]


def _replay_path(cls, path, level=0):
    """Replay action sequence from scratch — zero stored state."""
    g = cls()
    if hasattr(g, 'set_level'):
        try:
            g.set_level(level)
        except:
            return None, None
    r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    for a, d in path:
        ai = (ActionInput(id=GameAction.from_id(a), data=d)
              if d else ActionInput(id=GameAction.from_id(a)))
        r = g.perform_action(ai, raw=True)
    return g, r


# ============= CN04 SOLVER (Puzzle piece overlap matching) =============
def _solve_cn04(cls, level_idx=0):
    """
    CN04: Puzzle piece matching game.
    - Click sprites to select them
    - Move with arrows (1-4), rotate with 5
    - Win when all color-8 pixels are overlapped (marked as color-3)
    
    Strategy: BFS on (selected_sprite, positions, rotations)
    """
    try:
        g = cls()
        if hasattr(g, 'set_level') and level_idx > 0:
            try:
                g.set_level(level_idx)
            except:
                pass
        r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        
        if not r.frame:
            return None
        
        # Get all clickable sprites (those with sys_click tag or just all sprites)
        sprites = g.current_level.get_sprites()
        if not sprites:
            return None
        
        # Record initial state
        recorded = []
        
        # Helper: get sprite at position
        def get_sprite_at(x, y):
            return g.current_level.get_sprite_at(x, y, ignore_collidable=True)
        
        # Helper: click sprite
        def click_sprite(game, sprite):
            cx = sprite.x + sprite.width // 2
            cy = sprite.y + sprite.height // 2
            game.perform_action(
                ActionInput(id=GameAction.ACTION6, data={'x': cx, 'y': cy}),
                raw=True
            )
        
        # BFS state: (game_copy, action_path)
        queue = deque([(copy.deepcopy(g), [])])
        visited = set()
        
        # Create state key from game
        def state_key(game):
            sprites_state = []
            for s in game.current_level.get_sprites():
                sprites_state.append((s.name, s.x, s.y, s.rotation))
            # Include selected sprite
            selected = game.weqid.name if hasattr(game, 'weqid') and game.weqid else None
            return (tuple(sorted(sprites_state)), selected)
        
        initial_key = state_key(g)
        visited.add(initial_key)
        
        max_depth = 50
        max_states = 100000
        timeout = 60.0
        t0 = time.time()
        states = 0
        
        while queue and states < max_states and (time.time() - t0) < timeout:
            g_state, path = queue.popleft()
            states += 1
            
            if len(path) >= max_depth:
                continue
            
            # Check win condition
            if hasattr(g_state, 'exlcvhdjsf') and g_state.exlcvhdjsf():
                logger.info(f'CN04 L{level_idx} SOLVED: {len(path)} actions, {states} states')
                return path if path else None
            
            # Try all possible actions
            actions_to_try = []
            
            # Try clicking each sprite to select/deselect
            for sprite in g_state.current_level.get_sprites():
                cx = sprite.x + sprite.width // 2
                cy = sprite.y + sprite.height // 2
                actions_to_try.append((6, {'x': cx, 'y': cy}))
            
            # If a sprite is selected, try moving and rotating
            if hasattr(g_state, 'weqid') and g_state.weqid:
                # Try 4 movement directions
                for move_id in [1, 2, 3, 4]:
                    actions_to_try.append((move_id, None))
                # Try rotation
                actions_to_try.append((5, None))
            
            for action_id, data in actions_to_try:
                g2 = copy.deepcopy(g_state)
                
                if data:
                    r2 = g2.perform_action(
                        ActionInput(id=GameAction.ACTION6, data=data),
                        raw=True
                    )
                else:
                    r2 = g2.perform_action(
                        ActionInput(id=GameAction.from_id(action_id)),
                        raw=True
                    )
                
                # Check win
                if r2.levels_completed > 0 or r2.state == GameState.WIN:
                    final_path = path + [(action_id, data)]
                    logger.info(f'CN04 L{level_idx} SOLVED: {len(final_path)} actions')
                    return final_path
                
                if r2.state == GameState.GAME_OVER:
                    continue
                
                # Check if already visited
                key = state_key(g2)
                if key not in visited:
                    visited.add(key)
                    queue.append((g2, path + [(action_id, data)]))
        
        logger.info(f'CN04 L{level_idx} exhausted: {states} states, {len(visited)} visited')
        return None
        
    except Exception as e:
        logger.warning(f'cn04 solver error: {e}')
        return None


# ============= SK48 SOLVER (Snake/train builder with sim-BFS) =============
def _solve_sk48(cls, level_idx=0):
    """
    SK48: Snake/train builder puzzle.
    - Click segments to attach to snake heads
    - Move snakes around
    - Match segment colors to targets
    
    Strategy: Sim-BFS with frame hashing and dense click scanning
    """
    try:
        g = cls()
        if level_idx > 0:
            try:
                g.set_level(level_idx)
            except:
                return None
        r0 = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        
        if not r0.frame:
            return None
        
        actions = g._available_actions
        t0 = time.time()
        
        def frame_hash(r):
            if r.frame:
                return hashlib.md5(np.array(r.frame[-1]).tobytes()).hexdigest()[:16]
            return None
        
        # Scan for unique click effects
        click_targets = []
        if 6 in actions and r0.frame:
            frame = np.array(r0.frame[-1])
            seen_fx = set()
            
            # Scan on 2px grid for efficiency
            for y in range(0, 64, 2):
                for x in range(0, 64, 2):
                    if time.time() - t0 > 15:
                        break
                    
                    g_test = copy.deepcopy(g)
                    r_test = g_test.perform_action(
                        ActionInput(id=GameAction.ACTION6, data={'x': x, 'y': y, 'game_id': 'scan'}),
                        raw=True
                    )
                    
                    if r_test.levels_completed > 0:
                        return [(6, {'x': x, 'y': y, 'game_id': 'bfs'})]
                    
                    if r_test.frame:
                        fh = hashlib.md5(np.array(r_test.frame[-1]).tobytes()).hexdigest()[:12]
                        if fh not in seen_fx:
                            seen_fx.add(fh)
                            click_targets.append((x, y))
                
                if time.time() - t0 > 15:
                    break
        
        logger.info(f'SK48 L{level_idx}: scanned {len(click_targets)} unique click targets')
        
        # BFS with frame hashing
        h0 = frame_hash(r0)
        if not h0:
            return None
        
        queue = deque([(copy.deepcopy(g), [])])
        visited = {h0}
        states = 0
        
        max_depth = 40
        max_states = 300000
        timeout = 60.0
        
        # Build action list
        all_actions = [(a, None) for a in actions if a not in [6, 7]]  # Exclude click and undo for now
        for x, y in click_targets[:50]:  # Limit to top 50 clicks
            all_actions.append((6, {'x': x, 'y': y, 'game_id': 'bfs'}))
        
        logger.info(f'SK48 L{level_idx}: {len(all_actions)} actions ({len([a for a in all_actions if a[0]!=6])} kbd, {len([a for a in all_actions if a[0]==6])} clicks)')
        
        while queue and states < max_states and (time.time() - t0) < timeout:
            g_state, path = queue.popleft()
            states += 1
            
            if len(path) >= max_depth:
                continue
            
            for action_id, data in all_actions:
                g2 = copy.deepcopy(g_state)
                
                if data:
                    r = g2.perform_action(
                        ActionInput(id=GameAction.ACTION6, data=data),
                        raw=True
                    )
                else:
                    r = g2.perform_action(
                        ActionInput(id=GameAction.from_id(action_id)),
                        raw=True
                    )
                
                if r.levels_completed > 0 or r.state == GameState.WIN:
                    final_path = path + [(action_id, data)]
                    logger.info(f'SK48 L{level_idx} SOLVED: {len(final_path)} actions, {states} states')
                    return final_path
                
                if r.state == GameState.GAME_OVER:
                    continue
                
                h = frame_hash(r)
                if h and h not in visited:
                    visited.add(h)
                    queue.append((copy.deepcopy(g2), path + [(action_id, data)]))
        
        logger.info(f'SK48 L{level_idx} exhausted: {states} states, {len(visited)} visited')
        return None
        
    except Exception as e:
        logger.warning(f'sk48 solver error: {e}')
        return None


# ============= SB26 SOLVER (Drag-and-drop puzzle builder with sim-BFS) =============
def _solve_sb26(cls, level_idx=0):
    """
    SB26: Drag-and-drop puzzle builder.
    Pure click-based interface - no movement keys.
    Very high branching factor.
    
    Strategy: Sim-BFS with aggressive pruning and timeout
    """
    try:
        g = cls()
        if level_idx > 0:
            try:
                g.set_level(level_idx)
            except:
                return None
        r0 = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        
        if not r0.frame:
            return None
        
        actions = g._available_actions
        t0 = time.time()
        
        def frame_hash(r):
            if r.frame:
                return hashlib.md5(np.array(r.frame[-1]).tobytes()).hexdigest()[:16]
            return None
        
        # Scan for unique click effects (more aggressive than sk48)
        click_targets = []
        if 6 in actions and r0.frame:
            frame = np.array(r0.frame[-1])
            seen_fx = set()
            
            # Scan on 3px grid for speed
            for y in range(0, 64, 3):
                for x in range(0, 64, 3):
                    if time.time() - t0 > 10:
                        break
                    
                    g_test = copy.deepcopy(g)
                    r_test = g_test.perform_action(
                        ActionInput(id=GameAction.ACTION6, data={'x': x, 'y': y, 'game_id': 'scan'}),
                        raw=True
                    )
                    
                    if r_test.levels_completed > 0:
                        return [(6, {'x': x, 'y': y, 'game_id': 'bfs'})]
                    
                    if r_test.frame:
                        fh = hashlib.md5(np.array(r_test.frame[-1]).tobytes()).hexdigest()[:12]
                        if fh not in seen_fx:
                            seen_fx.add(fh)
                            click_targets.append((x, y))
                
                if time.time() - t0 > 10:
                    break
        
        logger.info(f'SB26 L{level_idx}: scanned {len(click_targets)} unique click targets')
        
        # BFS with frame hashing
        h0 = frame_hash(r0)
        if not h0:
            return None
        
        queue = deque([(copy.deepcopy(g), [])])
        visited = {h0}
        states = 0
        
        max_depth = 25  # Shorter depth for high branching
        max_states = 200000
        timeout = 45.0
        
        # Build action list - prioritize non-click actions
        all_actions = [(a, None) for a in actions if a not in [6]]
        for x, y in click_targets[:30]:  # Limit to top 30 clicks
            all_actions.append((6, {'x': x, 'y': y, 'game_id': 'bfs'}))
        
        logger.info(f'SB26 L{level_idx}: {len(all_actions)} actions')
        
        while queue and states < max_states and (time.time() - t0) < timeout:
            g_state, path = queue.popleft()
            states += 1
            
            if len(path) >= max_depth:
                continue
            
            for action_id, data in all_actions:
                g2 = copy.deepcopy(g_state)
                
                if data:
                    r = g2.perform_action(
                        ActionInput(id=GameAction.ACTION6, data=data),
                        raw=True
                    )
                else:
                    r = g2.perform_action(
                        ActionInput(id=GameAction.from_id(action_id)),
                        raw=True
                    )
                
                if r.levels_completed > 0 or r.state == GameState.WIN:
                    final_path = path + [(action_id, data)]
                    logger.info(f'SB26 L{level_idx} SOLVED: {len(final_path)} actions, {states} states')
                    return final_path
                
                if r.state == GameState.GAME_OVER:
                    continue
                
                h = frame_hash(r)
                if h and h not in visited:
                    visited.add(h)
                    queue.append((copy.deepcopy(g2), path + [(action_id, data)]))
        
        logger.info(f'SB26 L{level_idx} exhausted: {states} states, {len(visited)} visited')
        return None
        
    except Exception as e:
        logger.warning(f'sb26 solver error: {e}')
        return None
