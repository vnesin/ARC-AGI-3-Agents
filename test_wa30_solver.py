#!/usr/bin/env python3
"""Test wa30 Sokoban solver."""
import sys
sys.path.insert(0, '/home/vova-spark/ARC-AGI-3-Agents')

from rewind_agent_v21 import _solve_wa30_sokoban, _replay_path
import importlib.util
from pathlib import Path

# Load wa30
spec = importlib.util.spec_from_file_location('wa30', 'environment_files/wa30/wa30.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
Wa30 = mod.Wa30

for level in range(3):  # Test first 3 levels
    print(f"\n{'='*60}")
    print(f"Testing wa30 Level {level}")
    print('='*60)
    
    sol = _solve_wa30_sokoban(Wa30, level)
    if sol:
        print(f"SOLUTION FOUND: {len(sol)} actions")
        print(f"Actions: {sol[:20]}..." if len(sol) > 20 else f"Actions: {sol}")
        
        # Verify it works
        g, r = _replay_path(Wa30, sol, level)
        if g and hasattr(g, 'ymzfopzgbq'):
            if g.ymzfopzgbq():
                print("✓ VERIFIED: Solution reaches win condition!")
            else:
                print("✗ FAILED: Solution doesn't reach win condition")
                print(f"  levels_completed: {r.levels_completed if r else 'N/A'}")
    else:
        print("No solution found")
