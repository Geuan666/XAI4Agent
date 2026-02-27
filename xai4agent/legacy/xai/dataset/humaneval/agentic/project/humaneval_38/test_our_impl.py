# Test script to validate our implementation
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our main module
from main import encode_cyclic, decode_cyclic

def test_decode_cyclic():
    """Test our decode_cyclic implementation"""
    test_cases = [
        ("abc", "bca"),
        ("abcd", "bcad"), 
        ("hello world", "elho lorwld"),
        ("a", "a"),
        ("ab", "ab"),
        ("abcde", "bcade"),
        ("abcdef", "bcaefd")
    ]
    
    print("Testing encode/decode cycles:")
    for original, expected_encoded in test_cases:
        encoded = encode_cyclic(original)
        decoded = decode_cyclic(encoded)
        success = original == decoded
        print(f"Original: {original!r}")
        print(f"Expected encoded: {expected_encoded!r}")
        print(f"Actual encoded: {encoded!r}")
        print(f"Decoded: {decoded!r}")
        print(f"Success: {success}")
        print("-" * 40)
        
        if not success:
            print("FAILED!")
            sys.exit(1)
    
    # Test random strings like the original test does
    from random import randint, choice
    import string
    
    letters = string.ascii_lowercase
    print("Testing with random strings...")
    for _ in range(5):  # Reduced from 100 for speed
        test_str = ''.join(choice(letters) for i in range(randint(10, 20)))
        encoded = encode_cyclic(test_str)
        decoded = decode_cyclic(encoded)
        success = test_str == decoded
        print(f"Random test - Original: {test_str[:10]}..., Encoded: {encoded[:10]}..., Decoded: {decoded[:10]}..., Success: {success}")
        if not success:
            print("FAILED!")
            sys.exit(1)
    
    print("\n✓ ALL TESTS PASSED!")

if __name__ == "__main__":
    test_decode_cyclic()