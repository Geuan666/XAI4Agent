#!/usr/bin/env python3

# Test the tri function
import sys
sys.path.append('/root/autodl-tmp/xai/exp/real/projects/humaneval_130')
from main import tri

# Test the provided example
result = tri(3)
print(f"tri(3) = {result}")

# Expected: [1, 3, 2, 8]
expected = [1, 3, 2, 8]
if result == expected:
    print("✓ Test passed!")
else:
    print(f"✗ Test failed! Expected {expected}, got {result}")

# Test a few more cases
print(f"tri(0) = {tri(0)}")
print(f"tri(1) = {tri(1)}")
print(f"tri(2) = {tri(2)}")