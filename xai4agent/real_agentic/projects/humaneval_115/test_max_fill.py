from main import max_fill

# Test Example 1
grid1 = [[0,0,1,0], [0,1,0,0], [1,1,1,1]]
capacity1 = 1
result1 = max_fill(grid1, capacity1)
print(f"Example 1: {result1}")  # Should output 6

# Test Example 2
grid2 = [[0,0,1,1], [0,0,0,0], [1,1,1,1], [0,1,1,1]]
capacity2 = 2
result2 = max_fill(grid2, capacity2)
print(f"Example 2: {result2}")  # Should output 5

# Test Example 3
grid3 = [[0,0,0], [0,0,0]]
capacity3 = 5
result3 = max_fill(grid3, capacity3)
print(f"Example 3: {result3}")  # Should output 0

print("All tests completed.")