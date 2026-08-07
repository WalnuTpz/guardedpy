"""Run the three offline continuous-agent mechanism scenarios."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardedpy.mechanism_demo import run_all_scenarios


for result in run_all_scenarios():
    print(f"{result.name} status={result.status}")
