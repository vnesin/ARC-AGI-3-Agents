# ARC-AGI-3 Unsolved Games Analysis

Analysis of 9 unsolved games, prioritized by file size and complexity.

---

## CN04 (690 lines) ⭐ PRIORITY 1

### Mechanics
- **Grid**: 20x20 grid
- **Actions**: Movement (1-4), Rotate (5), Click (6)
- **Core mechanic**: Puzzle piece matching game
  - Multiple sprites with color `8` pixels (overlap markers)
  - Click to select/deselect sprites
  - Move sprites with arrow keys
  - Rotate sprites with action 5
  - Win when ALL `8` pixels overlap (none exposed)

### Win Condition
`exlcvhdjsf()` checks all original pixels marked as `8` are now marked as `3` (overlapped)

### Levels
5 levels total, increasing difficulty

### Solver Approach
**Custom analytical solver recommended**
- State: (selected_sprite, positions, rotations)
- BFS search space: select sprite → move/rotate → check overlap
- Small search space due to discrete positions/rotations
- Can pre-compute valid overlap configurations

### Implementation Complexity: ⭐⭐ MEDIUM
Sim-BFS would work but analytical solver is more efficient

---

## SK48 (984 lines) ⭐ PRIORITY 2

### Mechanics
- **Grid**: 64x64 grid
- **Actions**: Movement (1-4), Pause/Action5 (not clear), Click (6), Undo (7)
- **Core mechanic**: Snake/train builder puzzle
  - "Head" sprites (tagged `epdquznwmq`) with colored center windows
  - Extension segments (tagged `elmjchdqcn`) in various colors
  - Click on segments to attach them to snake heads
  - Move snakes around the grid
  - Match segment colors to target patterns (tagged `qtjqovumxf`)

### Win Condition
`gvtmoopqgy()` checks all segments match their target colors

### Levels
8 levels total

### Solver Approach
**Sim-BFS with click-heavy action scanning**
- Action space: 4 movements + ~50-100 click targets
- State: (head_positions, attached_segments, segment_colors)
- Branching factor is high but finite
- Frame hashing should work (visual state captures everything)

### Implementation Complexity: ⭐⭐⭐ MEDIUM-HIGH
Sim-BFS viable, but needs good click scanning. Analytical solver would need complex segment attachment logic.

---

## SB26 (1146 lines) ⭐ PRIORITY 3

### Mechanics
- **Grid**: 64x64 grid
- **Actions**: Action5, Click (6), Action7
- **Core mechanic**: Drag-and-drop puzzle builder
  - Draggable board segments (tagged `pkpgflvjel`) 
  - Clickable grid cells (tagged `lngftsryyw`)
  - Place/arrange segments to form patterns
  - No traditional movement keys - pure click-based

### Win Condition
Checks segment arrangement matches target configuration

### Levels
8 levels total

### Solver Approach
**Sim-BFS with dense click scanning**
- Pure click-based: scan 64x64 grid for valid clicks
- State captured entirely by frame (segment positions)
- Very high branching factor (~hundreds of clicks per state)
- May need depth limit and heuristics

### Implementation Complexity: ⭐⭐⭐⭐ HIGH
Sim-BFS is only option. Large action space. May timeout without good pruning.

---

## KA59 (1680 lines) ⭐ PRIORITY 4

### Mechanics
- **Grid**: 64x64 grid
- **Actions**: Complex action set (need to inspect available_actions)
- **Core mechanic**: [Requires deeper analysis - file too large for quick scan]
  - Multiple sprite types with complex interactions
  - Likely multi-stage puzzle

### Win Condition
[Needs detailed inspection]

### Levels
7 levels total

### Solver Approach
**Requires detailed analysis**
- Large codebase suggests complex mechanics
- Defer until simpler games are solved

### Implementation Complexity: ⭐⭐⭐⭐⭐ VERY HIGH
Postpone - focus on cn04, sk48, sb26 first

---

## Remaining Games (Not Prioritized Yet)

### BP35 (4565 lines)
Too complex - defer

### G50T (2855 lines)
Medium complexity - defer

### LF52 (5872 lines)
Too complex - defer

### RE86 (2158 lines)
Medium complexity - defer

### SC25 (2713 lines)
Medium complexity - defer

---

## Recommended Implementation Order

1. **CN04** - Write custom analytical solver
   - Pattern: Select → Move/Rotate → Check overlap
   - Likely solvable with small BFS on discrete state space
   
2. **SK48** - Sim-BFS with good click scanning
   - Scan grid, filter for sprites with `sys_click` tag
   - Use frame hashing
   - May need depth limit (30-40 actions)

3. **SB26** - Sim-BFS with aggressive pruning
   - Dense click scanning required
   - Frame hashing critical
   - Very high branching - may need heuristics or give up after timeout

4. **KA59** and beyond - Only if time permits

---

## Key Patterns from Existing Solvers

### Pattern 1: Sim-BFS (cd82, sp80, m0r0, lp85, s5i5, ft09, vc33, ar25)
```python
def _solve_sim_bfs(cls, level_idx, max_states=300000, max_depth=40, timeout=45.0)
```
- Frame hashing for dedup
- Scan clicks on 2px grid for unique effects
- Works when visual state == game state

### Pattern 2: Analytical solvers (dc22, wa30, r11l, su15, tn36, tr87, tu93, ls20)
- Read game state directly (sprites, positions, attributes)
- Compute solution algebraically or with domain-specific BFS
- Much faster than sim-BFS when applicable

### Pattern 3: Hybrid (dc22)
- BFS on abstract state (player position + button states)
- Deepcopy game objects, not frames
- Good when hidden state matters but is accessible

---

## Next Steps

1. Implement `_solve_cn04()` - custom analytical solver
2. Implement `_solve_sk48()` - sim-BFS
3. Test on offline game instances
4. Tune timeouts and depth limits
5. Move to sb26 if time permits
