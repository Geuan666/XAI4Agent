from main import cycpattern_check

# Test cases from the docstring
print(cycpattern_check("abcd","abd"))    # Expected: False
print(cycpattern_check("hello","ell"))   # Expected: True
print(cycpattern_check("whassup","psus")) # Expected: False
print(cycpattern_check("abab","baa"))    # Expected: True
print(cycpattern_check("efef","eeff"))   # Expected: False
print(cycpattern_check("himenss","simen")) # Expected: True