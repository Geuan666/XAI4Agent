from main import histogram

# Test cases from docstring
print("Testing histogram function:")
print("histogram('a b c'):", histogram('a b c'))
print("histogram('a b b a'):", histogram('a b b a'))
print("histogram('a b c a b'):", histogram('a b c a b'))
print("histogram('b b b b a'):", histogram('b b b b a'))
print("histogram(''):", histogram(''))

# Additional test cases
print("Additional tests:")
print("histogram('a a a'):", histogram('a a a'))
print("histogram('x y z'):", histogram('x y z'))