#!/usr/bin/env python
"""Run tests with 5-second timeout per file."""

import subprocess
import sys
from pathlib import Path
import os

os.chdir(Path(__file__).parent)

test_dir = Path("tests")
test_files = sorted(test_dir.glob("test_*.py"))

# Add subdirectory tests
for subdir in test_dir.iterdir():
    if subdir.is_dir() and subdir.name != "__pycache__":
        test_files.extend(sorted(subdir.glob("test_*.py")))

test_files.sort()

hanging = []
passed = []
failed = []

for test_file in test_files:
    rel_path = str(test_file)
    print(f"Testing {rel_path}...", end=" ", flush=True)

    try:
        result = subprocess.run(
            ["python", "-m", "pytest", rel_path, "-q", "--tb=no"],
            timeout=5,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("✓ PASSED")
            passed.append(rel_path)
        else:
            print(f"✗ FAILED")
            print(f"  {result.stdout}")
            failed.append(rel_path)
    except subprocess.TimeoutExpired:
        print("✗ TIMEOUT (>5s)")
        hanging.append(rel_path)

print("\n" + "=" * 60)
print(f"PASSED: {len(passed)}")
for f in passed:
    print(f"  ✓ {f}")

if failed:
    print(f"\nFAILED: {len(failed)}")
    for f in failed:
        print(f"  ✗ {f}")

if hanging:
    print(f"\nHANGING (>5s): {len(hanging)}")
    for f in hanging:
        print(f"  ⏱ {f}")
