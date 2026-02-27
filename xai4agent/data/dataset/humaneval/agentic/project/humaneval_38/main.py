def encode_cyclic(s: str):
    """
    returns encoded string by cycling groups of three characters.
    """
    # split string to groups. Each of length 3.
    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]
    # cycle elements in each group. Unless group has fewer elements than 3.
    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]
    return "".join(groups)


def decode_cyclic(s: str):
    """
    takes as input string encoded with encode_cyclic function. Returns decoded string.
    """
    # split string to groups. Each of length 3.
    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]
    # cycle elements in each group back. Unless group has fewer elements than 3.
    groups = [(group[-1] + group[:-1]) if len(group) == 3 else group for group in groups]
    return "".join(groups)


# Test function to verify implementation
def test_decode_cyclic():
    # Simple test cases
    test_cases = [
        ("abc", "bca"),
        ("abcd", "bcad"),
        ("hello world", "elho lorwld"),
        ("a", "a"),
        ("ab", "ab"),
        ("abcde", "bcade"),
        ("abcdef", "bcaefd")
    ]

    for original, expected_encoded in test_cases:
        encoded = encode_cyclic(original)
        decoded = decode_cyclic(encoded)
        assert original == decoded, f"Failed for {original}: got {decoded}"

    print("All manual tests passed!")
