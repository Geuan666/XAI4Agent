from main import how_many_times

# Test cases from the docstring
print(how_many_times('', 'a'))      # Should output: 0
print(how_many_times('aaa', 'a'))   # Should output: 3
print(how_many_times('aaaa', 'aa')) # Should output: 3

# Additional test cases
print(how_many_times('abababab', 'aba'))  # Should output: 3
print(how_many_times('hello world', 'll'))  # Should output: 1