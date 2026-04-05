#!/usr/bin/env python3
"""
Systematic analysis of all 25 ARC-AGI-3 games.
For each game, determine:
- Action count (available actions)
- State space size estimate
- Hidden state variables
- Why BFS might fail
- Proposed solver strategies
"""

import importlib.util
import inspect
from pathlib import Path
import ast
import re

GAMES = [
    'ar25', 'bp35', 'cd82', 'cn04', 'dc22', 'ft09', 'g50t', 'ka59', 
    'lf52', 'lp85', 'ls20', 'm0r0', 'r11l', 're86', 's5i5', 'sb26', 
    'sc25', 'sk48', 'sp80', 'su15', 'tn36', 'tr87', 'tu93', 'vc33', 'wa30'
]

def load_game_class(game_id):
    """Load game class from source file."""
    path = Path(f'environment_files/{game_id}/{game_id}.py')
    if not path.exists():
        return None
    
    spec = importlib.util.spec_from_file_location(f'g_{game_id}', str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    class_name = game_id[0].upper() + game_id[1:]
    return getattr(mod, class_name)

def count_state_variables(source_code):
    """Count instance variables that represent state."""
    tree = ast.parse(source_code)
    state_vars = set()
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                    if target.value.id == 'self':
                        state_vars.add(target.attr)
    
    return state_vars

def estimate_state_space(cls, source_code):
    """Estimate state space size based on variables and grid."""
    state_vars = count_state_variables(source_code)
    
    # Filter out methods and constants
    state_vars = {v for v in state_vars if not v.startswith('_') or v == '_available_actions'}
    
    # Count hidden state variables (not part of visible frame)
    hidden_vars = set()
    if hasattr(cls, '_get_hidden_state'):
        try:
            g = cls()
            g.set_level(0) if hasattr(g, 'set_level') else None
            hs = g._get_hidden_state()
            if hs is not None:
                if isinstance(hs, (list, tuple)):
                    hidden_vars = {f"hidden_{i}" for i in range(len(hs))}
                else:
                    hidden_vars = {'hidden_state'}
        except:
            pass
    
    # Grid contributes 64x64x16 possible states (but realistically much less)
    # Focus on hidden state variables which are the main complexity
    
    return {
        'total_vars': len(state_vars),
        'hidden_vars': hidden_vars,
        'has_hidden': len(hidden_vars) > 0
    }

def analyze_game(game_id):
    """Comprehensive analysis of a single game."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {game_id.upper()}")
    print(f"{'='*70}")
    
    cls = load_game_class(game_id)
    if not cls:
        print(f"ERROR: Could not load {game_id}")
        return None
    
    path = Path(f'environment_files/{game_id}/{game_id}.py')
    source = path.read_text()
    
    # Instantiate to check available actions
    try:
        g = cls()
        if hasattr(g, 'set_level'):
            try:
                g.set_level(0)
            except:
                pass
        
        avail = g._available_actions if hasattr(g, '_available_actions') else []
        
        # State space analysis
        state_info = estimate_state_space(cls, source)
        
        # Check for special patterns
        has_grammar = 'cifzvbcuwqe' in source  # Grammar rules (like tr87)
        has_sequence = 'zvojhrjxxm' in source  # Sequence parsing
        has_rotation = 'qvtymdcqear_index' in source  # Rotation index
        has_complex_logic = len(source) > 5000  # Large source = complex
        
        # Count levels
        num_levels = 0
        if hasattr(g, 'set_level'):
            for i in range(50):
                try:
                    g.set_level(i)
                    num_levels = i + 1
                except:
                    break
        
        result = {
            'game_id': game_id,
            'actions': avail,
            'action_count': len(avail),
            'num_levels': num_levels,
            'state_vars': state_info['total_vars'],
            'hidden_vars': state_info['hidden_vars'],
            'has_hidden': state_info['has_hidden'],
            'has_grammar': has_grammar,
            'has_sequence': has_sequence,
            'has_rotation': has_rotation,
            'complex_logic': has_complex_logic,
            'source_lines': len(source.split('\n'))
        }
        
        print(f"Actions: {avail} (count: {len(avail)})")
        print(f"Levels: {num_levels}")
        print(f"State variables: {state_info['total_vars']}")
        print(f"Hidden state: {state_info['has_hidden']} - {state_info['hidden_vars']}")
        print(f"Grammar rules: {has_grammar}")
        print(f"Sequence parsing: {has_sequence}")
        print(f"Rotation index: {has_rotation}")
        print(f"Complex logic: {has_complex_logic} ({len(source.split('\n'))} lines)")
        
        # Branching factor estimate
        bf = len(avail)
        if 6 in avail:  # Click action
            bf = len([a for a in avail if a != 6]) + 30  # ~30 unique click positions
        
        print(f"\nBranching factor: ~{bf}")
        print(f"State space (depth 10): ~{bf**10:.2e}")
        print(f"State space (depth 20): ~{bf**20:.2e}")
        
        return result
        
    except Exception as e:
        print(f"ERROR analyzing {game_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Analyze all games and generate report."""
    results = []
    
    for game_id in GAMES:
        result = analyze_game(game_id)
        if result:
            results.append(result)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Sort by action count (lowest branching factor first)
    results.sort(key=lambda x: x['action_count'])
    
    print("\nGames by action count (easiest first):")
    for r in results:
        hidden = "✓" if r['has_hidden'] else "✗"
        grammar = "✓" if r['has_grammar'] else "✗"
        print(f"{r['game_id']:6s} | Actions: {r['action_count']:2d} | Levels: {r['num_levels']:2d} | "
              f"Hidden: {hidden} | Grammar: {grammar} | Lines: {r['source_lines']:4d}")
    
    print("\n\nQuick wins (low branching factor, no hidden state):")
    quick_wins = [r for r in results if r['action_count'] <= 4 and not r['has_hidden']]
    for r in quick_wins:
        print(f"  {r['game_id']:6s} - {r['action_count']} actions, {r['num_levels']} levels")
    
    print("\n\nHidden state games (need proper hashing):")
    hidden_games = [r for r in results if r['has_hidden']]
    for r in hidden_games:
        print(f"  {r['game_id']:6s} - {r['action_count']} actions, vars: {r['hidden_vars']}")
    
    print("\n\nGrammar games (need special solver):")
    grammar_games = [r for r in results if r['has_grammar']]
    for r in grammar_games:
        print(f"  {r['game_id']:6s} - {r['action_count']} actions")

if __name__ == '__main__':
    main()
