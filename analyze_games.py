#!/usr/bin/env python3
"""Quick analysis script to understand unsolved game mechanics."""
import sys
import importlib.util
from pathlib import Path

def load_game(game_id):
    """Dynamically load game class."""
    game_path = Path(f"environment_files/{game_id}/{game_id}.py")
    spec = importlib.util.spec_from_file_location(game_id, game_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, game_id.capitalize())()

def analyze_game(game_id):
    """Extract key info about a game."""
    print(f"\n{'='*60}")
    print(f"Game: {game_id}")
    print('='*60)
    
    g = load_game(game_id)
    
    # Basic info
    print(f"Available actions: {g._available_actions}")
    print(f"Number of levels: {len(g._levels)}")
    
    # Reset to level 0
    from arcengine import ActionInput, GameAction
    if hasattr(g, 'set_level'):
        try:
            g.set_level(0)
        except:
            pass
    g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    
    # Check for hidden state
    if hasattr(g, '_get_hidden_state'):
        try:
            hs = g._get_hidden_state()
            print(f"Hidden state: {type(hs)} - {hs if len(str(hs)) < 100 else str(hs)[:100]+'...'}")
        except Exception as e:
            print(f"Hidden state ERROR: {e}")
    else:
        print("No hidden state method")
    
    # Look for win condition methods
    methods = [m for m in dir(g) if not m.startswith('_') and callable(getattr(g, m))]
    likely_win = [m for m in methods if 'win' in m.lower() or 'complete' in m.lower() or 'check' in m.lower()]
    if likely_win:
        print(f"Possible win condition methods: {likely_win}")
    
    # Try to find step counter or scoring
    if hasattr(g, 'current_level'):
        level_data = g.current_level.get_all_data()
        print(f"Level data keys: {list(level_data.keys())}")
        
        # Get sprite tags
        all_sprites = g.current_level.get_sprites()
        all_tags = set()
        for s in all_sprites:
            if hasattr(s, 'tags'):
                all_tags.update(s.tags)
        if all_tags:
            print(f"Sprite tags: {sorted(all_tags)}")
    
    print()

if __name__ == "__main__":
    games = ['wa30', 'bp35', 'dc22', 'r11l', 'su15', 'tn36']
    for game_id in games:
        try:
            analyze_game(game_id)
        except Exception as e:
            print(f"\nERROR analyzing {game_id}: {e}")
            import traceback
            traceback.print_exc()
