"""GAAL v3 — CLI entry point for running arena tests."""
import sys, os, json, yaml
from pathlib import Path

# Ensure project root is on sys.path for absolute imports
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from core.orchestrator import GAALOrchestrator


def run_arena(goal: str, mode: str = "lite", max_loops: int = 4, config_path: str = None):
    """Run a GAAL v3 arena with the given goal.
    
    Args:
        goal: The goal/task to run.
        mode: Arena mode ('lite', 'hard', 'super').
        max_loops: Max arena loops.
        config_path: Path to config YAML (default: config/gaal_v3.yaml).
    """
    config_path = config_path or str(ROOT / "config" / "gaal_v3.yaml")
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    cfg["gaal"]["goal"] = goal
    cfg["gaal"]["mode"] = mode
    cfg["gaal"]["max_loops"] = max_loops
    
    orc = GAALOrchestrator(config=cfg)
    result = orc.run(goal=goal)
    return result


if __name__ == "__main__":
    goal = sys.argv[1] if len(sys.argv) > 1 else "设计一个简单的文件备份系统"
    mode = sys.argv[2] if len(sys.argv) > 2 else "lite"
    
    print(f"GAAL v3 Arena — Mode: {mode}")
    print(f"Goal: {goal}")
    print("=" * 60)
    
    result = run_arena(goal=goal, mode=mode)
    
    print(f"\nExecution: {len(result.get('history', []))} nodes completed")
    print(f"Score: {result.get('final_score', 'N/A')} / 10")
    print(f"Passed: {result.get('passed', False)}")
    print(f"Loops: {result.get('loops_completed', 0)} / {result.get('max_loops', 4)}")
    print(f"Proposals: {result.get('total_proposals', 0)}")
    print(f"Eliminations: {len(result.get('eliminations', []))}")
    print(f"Winner: {result.get('winner', 'N/A')}")
