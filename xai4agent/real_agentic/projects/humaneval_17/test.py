from main import parse_music

# Test with the provided example
result = parse_music('o o| .| o| o| .| .| .| .| o o')
expected = [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]

print("Result:", result)
print("Expected:", expected)
print("Match:", result == expected)