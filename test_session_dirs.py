#!/usr/bin/env python3
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents.templates.claude_agent.claude_recorder import ClaudeCodeRecorder


def test_session_directory_creation():
    print("\n=== Test 1: Initial temp directory creation ===")
    
    recorder1 = ClaudeCodeRecorder(
        game_id="test-game",
        agent_name="testagent"
    )
    
    print(f"✓ Created directory: {recorder1.output_dir}")
    assert "temp_" in str(recorder1.output_dir), "Should have temp_ in directory name"
    assert recorder1.output_dir.exists(), "Directory should exist"
    
    recorder1.save_step(
        step=1,
        prompt="Test prompt",
        messages=[],
        parsed_action={"action": 1, "reasoning": "test"},
        total_cost_usd=0.0
    )
    
    step_file = recorder1.output_dir / "step_001.json"
    assert step_file.exists(), "Step file should exist"
    print(f"✓ Created step file: {step_file}")
    
    cleanup_test_dirs()
    print("✓ Test 1 passed!\n")


def test_session_id_update():
    print("=== Test 2: Session ID update and directory move ===")
    
    recorder = ClaudeCodeRecorder(
        game_id="test-game",
        agent_name="testagent"
    )
    
    temp_dir = str(recorder.output_dir)
    print(f"✓ Initial temp directory: {temp_dir}")
    
    recorder.save_step(
        step=1,
        prompt="Test prompt",
        messages=[],
        parsed_action={"action": 1, "reasoning": "test"},
        total_cost_usd=0.0
    )
    
    test_session_id = "test-session-id-1234567890"
    recorder.update_session_id(test_session_id)
    
    new_dir = str(recorder.output_dir)
    print(f"✓ New directory after session_id: {new_dir}")
    
    assert test_session_id in new_dir, "Session ID should be in directory name"
    assert not Path(temp_dir).exists(), "Old temp directory should not exist"
    assert recorder.output_dir.exists(), "New directory should exist"
    
    step_file = recorder.output_dir / "step_001.json"
    assert step_file.exists(), "Step file should be moved to new directory"
    print(f"✓ Step file preserved: {step_file}")
    
    cleanup_test_dirs()
    print("✓ Test 2 passed!\n")


def test_multiple_concurrent_sessions():
    print("=== Test 3: Multiple concurrent sessions (no collision) ===")
    
    recorder1 = ClaudeCodeRecorder(
        game_id="test-game",
        agent_name="testagent"
    )
    recorder2 = ClaudeCodeRecorder(
        game_id="test-game",
        agent_name="testagent"
    )
    
    dir1 = str(recorder1.output_dir)
    dir2 = str(recorder2.output_dir)
    
    print(f"✓ Session 1 directory: {dir1}")
    print(f"✓ Session 2 directory: {dir2}")
    
    assert dir1 != dir2, "Directories should be different"
    assert recorder1.output_dir.exists(), "Directory 1 should exist"
    assert recorder2.output_dir.exists(), "Directory 2 should exist"
    
    recorder1.save_step(1, "prompt1", [], {"action": 1}, 0.0)
    recorder2.save_step(1, "prompt2", [], {"action": 2}, 0.0)
    
    recorder1.update_session_id("session-id-aaaaaa")
    recorder2.update_session_id("session-id-bbbbbb")
    
    assert "session-id-aaaaaa" in str(recorder1.output_dir)
    assert "session-id-bbbbbb" in str(recorder2.output_dir)
    assert recorder1.output_dir != recorder2.output_dir
    
    print(f"✓ Final session 1 directory: {recorder1.output_dir}")
    print(f"✓ Final session 2 directory: {recorder2.output_dir}")
    
#    cleanup_test_dirs()
    print("✓ Test 3 passed!\n")


def test_with_session_id_from_start():
    print("=== Test 4: Recorder created with session_id ===")
    
    session_id = "known-session-id-123"
    recorder = ClaudeCodeRecorder(
        game_id="test-game",
        agent_name="testagent",
        session_id=session_id
    )
    
    assert "temp_" not in str(recorder.output_dir), "Should not have temp_ in name"
    assert session_id in str(recorder.output_dir), "Should have session_id in name"
    print(f"✓ Directory with session_id: {recorder.output_dir}")
    
    print("✓ Test 4 passed!\n")


def cleanup_test_dirs():
    recordings_dir = Path("recordings")
    if recordings_dir.exists():
        for item in recordings_dir.glob("test-game_*"):
            if item.is_dir():
                shutil.rmtree(item)


if __name__ == "__main__":
    cleanup_test_dirs()
    
    try:
#        test_session_directory_creation()
#        test_session_id_update()
        test_multiple_concurrent_sessions()
#        test_with_session_id_from_start()
        
        print("="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        cleanup_test_dirs()
