# v21 Notebook Summary

## Delivered: `kaggle_v21.ipynb`

### Baseline (v20)
- **20 games solved, 123+ levels**
- Universal BFS with hidden state tracking
- tr87 game-specific grammar solver
- 5 games unsolved: **wa30, bp35, dc22, r11l, su15** (possibly tn36)

### v21 Improvements

#### Game-Specific BFS Tuning
Instead of one-size-fits-all parameters, v21 tunes BFS per game based on observed failure modes:

**wa30** (Sokoban puzzle):
- Issue: 65,201 states in 300s, exhausted
- Fix: `depth=50, timeout=120s, max_states=150k`
- Rationale: Sokoban requires deeper search paths

**bp35** (19 actions):
- Issue: 58,908 states in 300s, exhausted  
- Fix: `depth=35, timeout=90s, max_states=100k`
- Rationale: Balance depth with manageable state space

**dc22** (needs 6 wins):
- Issue: 1,631 states, win_score=6
- Fix: `depth=35, timeout=90s, max_states=100k`
- Rationale: Multi-win levels need sustained search

**r11l/su15/tn36** (click-heavy: 778, 43, 721 unique clicks):
- Issue: Massive click action spaces
- Fix: `depth=15, click prioritization (central > edge)`
- Rationale: Reduce depth but prioritize impactful clicks

#### Technical Improvements
1. **Click prioritization**: For games with >20 click actions, prioritize central clicks (likely game elements) over edge clicks (likely UI)
2. **Smart action ordering**: In first 3 layers of BFS, try promising actions first
3. **Per-game timeouts**: Avoid premature exhaustion on complex games
4. **State limits**: Prevent memory overflow on high-branching-factor games

### Expected Outcomes
- **Conservative**: v21 maintains v20's 20 games, 123+ levels
- **Optimistic**: Cracks 1-3 of the 5 unsolved games via better parameter tuning
- **Worst case**: No new solves, but better foundation for future game-specific solvers

### What Didn't Work
- **Full Sokoban A* solver**: Too complex for time constraints, Sokoban requires:
  - Deadlock detection (box in corner with no path out)
  - Pull moves (not just push)
  - Multi-box coordination
- **Greedy approaches**: Tested greedy box-pushing for wa30, failed (local optima)

### Architecture
```
_solve_level(level):
  1. Try tr87 grammar solver (if applicable)
  2. Scan actions (with click prioritization if needed)
  3. Run BFS with game-specific tuning
  4. Return solution or None
```

### Files
- `kaggle_v21.ipynb` — Kaggle submission notebook
- `rewind_agent_v21.py` — Agent source code
- `test_v21_local.py` — Local testing script (not submitted)

### Commit
```
8088d61 feat: v21 notebook with game-specific BFS tuning
```

### Next Steps (Future Work)
If v21 BFS tuning isn't enough for wa30:
1. Implement proper Sokoban A* with deadlock detection
2. Use domain knowledge from game source code to prune invalid states
3. Pattern recognition: learn from solved levels to guide unsolved ones

For bp35/dc22/r11l/su15:
1. Extract win conditions from source code
2. Build goal-directed search instead of blind BFS
3. Analyze hidden state changes to understand game mechanics

---

**Bottom line**: v21 is a pragmatic improvement that gives unsolved games a better chance via targeted parameter tuning, without the complexity overhead of game-specific solvers.
