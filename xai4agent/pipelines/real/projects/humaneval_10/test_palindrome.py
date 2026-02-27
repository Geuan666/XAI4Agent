from main import make_palindrome

# Test the examples from the docstring
print("Testing make_palindrome function:")
print(f"make_palindrome('') = '{make_palindrome('')}'")
print(f"make_palindrome('cat') = '{make_palindrome('cat')}'")
print(f"make_palindrome('cata') = '{make_palindrome('cata')}'")

# Additional tests
print(f"make_palindrome('abc') = '{make_palindrome('abc')}'")
print(f"make_palindrome('aba') = '{make_palindrome('aba')}'")