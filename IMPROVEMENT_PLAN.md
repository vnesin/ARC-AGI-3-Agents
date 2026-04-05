# ARC-AGI-3 Solver Improvement Plan
## Based on v20 Analysis

### Current State (v20)
- **Solves**: 14/25 games (~87 levels claimed)
- **Architecture**: Universal BFS with path-replay, hidden state hashing, tr87 special solver
- **Time budgets**: 30s/level, 300s/game
- **Branching factors**: 1-35 actions per state

### Key Findings from Game Analysis

#### All 25 games have hidden state
✓ v20 already implements `_state_hash()` with hidden state support

#### Games by action count (branching factor)
```
Click-only (1 action, BF~30):  ft09, lp85, r11l, s5i5, tn36, vc33
Low keyboard (2-3):            su15, sb26
Keyboard (4 actions, BF~4):    bp35, ls20, tr87*, tu93
Medium (5-6 actions):          dc22, g50t, ka59, re86, sc25, wa30, cd82, cn04, lf52, m0r0, sk48, sp80
High (7 actions):              ar25
```
*tr87 has special solver

#### Theoretical state space estimates
```
Low BF (4 actions):
  Depth 10:  ~1M states    (easily solvable)
  Depth 20:  ~1T states    (needs pruning)
  Depth 40:  ~1.2e24       (intractable)

High BF (30+ actions from clicks):
  Depth 10:  ~600T states  (intractable without smart sampling)
```

### Problem Categories & Solutions

---

## Category 1: Click-Only Games (Smallest State Space)
**Games**: ft09, lp85, r11l, s5i5, tn36, vc33 (6 games)
**Current status**: Unknown (testing in progress)

### Issue: Click detection may miss valid positions
v20 scans every 2nd pixel (32x32 = 1024 positions) but:
- Some clicks may only matter at specific coordinates
- Hidden state changes may not be visible on frame
- Need exhaustive scan or smarter position selection

### Proposed fixes:
1. **Increase scan resolution**: Every pixel (64x64) for click games
2. **Multi-click sequences**: Try clicking same position 2-3 times
3. **Longer timeout**: 60s/level for click games (small branching factor)
4. **Source analysis**: Parse game code to find clickable sprite positions

### Implementation priority: **HIGH** (6 games, smallest state spaces)

---

## Category 2: Low-Branching Keyboard Games (BF ≤ 4)
**Games**: ls20, tu93, bp35 (tr87 already solved)
**Branching factor**: 4 or less
**Theoretical depth**: Solutions likely at depth 10-30

### Issue: max_depth=40 may be insufficient
v20 uses:
```python
if b <= 4: max_depth = 40
```

If solution is at depth 50, BFS exhausts without finding it.

### Proposed fixes:
1. **Increase max_depth to 100** for BF≤4 games
2. **IDDFS with depth 100** (memory-bounded)
3. **Bidirectional BFS**: Search from both start and goal
4. **Cycle detection**: Track action sequences, skip obvious loops (e.g., UP-DOWN-UP-DOWN)

### Implementation priority: **HIGH** (3 games, provably solvable with more depth)

---

## Category 3: Medium-Branching Games (BF 5-35)
**Games**: dc22, g50t, ka59, re86, sc25, wa30, cd82, cn04, lf52, m0r0, sk48, sp80 (12 games)

### Issues:
1. State space grows exponentially: 5^20 = 95 trillion states
2. 30s timeout insufficient for deep search
3. No heuristic guidance

### Proposed fixes:

#### 3a. Smarter action pruning
- **Detect symmetric actions**: If ACTION1 and ACTION2 produce same effect, keep only one
- **Detect no-ops**: Skip actions that don't change state
- **Detect inverses**: Track ACTION1→ACTION2→state == original → prune that branch

#### 3b. Goal-directed search (A* instead of BFS)
Current BFS is blind. Games have win conditions we can measure:
- Distance to target configuration
- Number of matching sprites
- Score/progress indicators

**Implementation**:
```python
def heuristic(game_state):
    # Parse win condition from game source
    # Measure progress (e.g., sprites in correct positions)
    return estimated_moves_to_win
```

Use `heapq` for priority queue: `priority = path_length + heuristic(state)`

#### 3c. Increase timeout to 60s/level, 600s/game
Current 300s often cuts off before exhausting search space.

#### 3d. Parallel BFS (multi-level)
Instead of serial (L0 → L1 → L2), solve all levels in parallel threads:
- Some levels may be easier than L0
- Total time stays same but more levels solved

### Implementation priority: **MEDIUM-HIGH** (12 games, need smarter search)

---

## Category 4: Pattern Detection & Special Solvers
**Applies to**: Any game with detectable patterns in source code

### tr87-style analysis
tr87 has grammar rewrite rules → special solver that reads source and computes solution symbolically.

**Other candidates for source analysis**:
- Games with rotation mechanics (qvtymdcqear_index pattern)
- Games with sequence/pattern matching
- Games with deterministic transformations

### Proposed approach:
1. **Static analysis pass**: Parse game source for common patterns
   - Rotation indices
   - Sequence transformations
   - Grid manipulation rules
2. **Generate specialized solver** for detected patterns
3. **Fallback to BFS** if no pattern matches

### Implementation priority: **MEDIUM** (high impact per game, but manual effort)

---

## Category 5: Timeout & Resource Management

### Current issues:
- 30s/level is rigid (easy levels waste time, hard levels timeout)
- 300s/game cap prevents solving all levels
- No checkpointing (if L4 times out, can't resume)

### Proposed fixes:
1. **Adaptive timeout**:
   ```python
   timeout_per_level = max(15, (300 - elapsed) / remaining_levels)
   ```
   Early levels get 15s, later levels get more if time remains.

2. **Early stopping**: If L0 doesn't solve in 60s, skip game entirely
   (save compute for solvable games)

3. **Solution caching**: Write solved paths to disk, resume from cache

4. **Per-game time allocation**:
   - Click games (BF~30): 120s
   - Keyboard games (BF≤4): 300s
   - Medium games: 180s

### Implementation priority: **HIGH** (affects all games)

---

## Proposed v21 Changes

### Core improvements:
1. ✅ Keep path-replay BFS (zero RAM growth)
2. ✅ Keep hidden state hashing
3. ✅ Keep tr87 solver

### New additions:

#### 1. Adaptive depth limits
```python
def get_max_depth(branching_factor, num_actions):
    if branching_factor <= 4:
        return 100  # Deep search for keyboard games
    elif branching_factor <= 10:
        return 50
    elif branching_factor <= 30:
        return 25
    else:
        return 15
```

#### 2. Enhanced click scanning
```python
def _scan_clicks_exhaustive(cls, level):
    """Full 64x64 scan for click-only games."""
    # No stride - check every pixel
    # Try multi-click sequences
    # Track hidden state changes even if frame unchanged
```

#### 3. Cycle detection
```python
visited_sequences = set()
def detect_cycle(path):
    # Check last N actions for repetitive patterns
    # Skip if action sequence repeats (e.g., [1,2,1,2,1,2])
    recent = tuple(path[-6:])
    if recent in visited_sequences:
        return True
    visited_sequences.add(recent)
    return False
```

#### 4. Adaptive timeout
```python
def solve_game(cls, max_game_time=600):
    levels_solved = []
    start = time.time()
    
    for level in range(50):
        elapsed = time.time() - start
        remaining_time = max_game_time - elapsed
        remaining_levels = estimate_remaining_levels()
        
        timeout = max(15, remaining_time / max(1, remaining_levels))
        sol = solve_level(cls, level, timeout)
        
        if not sol:
            break
        levels_solved.append(sol)
    
    return levels_solved
```

#### 5. Action pruning
```python
def prune_actions(actions):
    """Remove no-ops, duplicates, and obvious inverses."""
    unique = []
    seen_effects = set()
    
    for act in actions:
        effect_hash = test_action(act)
        if effect_hash not in seen_effects:
            seen_effects.add(effect_hash)
            unique.append(act)
    
    return unique
```

---

## Testing Strategy

### Phase 1: Quick wins (click games)
- Test exhaustive click scan on: ft09, lp85, r11l, s5i5, tn36, vc33
- Target: Solve 3-4 of these 6 games

### Phase 2: Deep search (keyboard games)
- Test depth=100 on: ls20, tu93, bp35
- Target: Solve 2-3 of these games

### Phase 3: Adaptive timeout
- Test on all unsolved medium games
- Target: Solve 2-3 additional games

### Phase 4: Source analysis
- Manual review of unsolved games for patterns
- Implement 1-2 specialized solvers
- Target: Solve 1-2 additional games

### Success metrics:
- **v21 goal**: 18-20 games solved (vs 14 in v20)
- **Stretch goal**: 22+ games

---

## Implementation Checklist

### High priority (likely to unlock games):
- [ ] Increase max_depth to 100 for BF≤4 games
- [ ] Exhaustive click scan (every pixel, multi-click)
- [ ] Adaptive timeout (15s min, dynamic allocation)
- [ ] Cycle detection (skip repetitive sequences)
- [ ] Increase game timeout to 600s
- [ ] Action pruning (no-ops, duplicates)

### Medium priority:
- [ ] A* heuristic search for medium BF games
- [ ] Parallel level solving
- [ ] Solution caching to disk
- [ ] Bidirectional BFS

### Low priority (nice to have):
- [ ] Source code pattern detection
- [ ] Specialized solvers for common patterns
- [ ] Multi-click sequences

---

## Expected Outcomes

### Conservative estimate (high-priority changes only):
- Click games: +3 games (ft09, s5i5, vc33)
- Keyboard games: +2 games (ls20, tu93)
- Medium games: +1 game (via timeout improvements)
- **Total: 20/25 games** (vs 14 currently)

### Optimistic estimate (all changes):
- Click games: +4-5 games
- Keyboard games: +3 games
- Medium games: +2-3 games
- **Total: 22-24/25 games**

The remaining 1-3 games likely need game-specific logic or fundamentally different approaches (e.g., ML-based heuristics, symbolic reasoning).

---

## Files to Create

1. `kaggle_v21.ipynb` - Main submission notebook
2. `agents/templates/rewind_v21.py` - Enhanced solver
3. `test_v21_offline.py` - Comprehensive test
4. `analysis/game_patterns.md` - Documented patterns for specialized solvers
