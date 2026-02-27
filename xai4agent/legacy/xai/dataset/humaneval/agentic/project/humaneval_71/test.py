def check(candidate):

    # Check some simple cases
    assert candidate(3, 4, 5) == 6.00, "This prints if this assert fails 1 (good for debugging!)"
    assert candidate(1, 2, 10) == -1
    assert candidate(4, 8, 5) == 8.18
    assert candidate(2, 2, 2) == 1.73
    assert candidate(1, 2, 3) == -1
    assert candidate(10, 5, 7) == 16.25
    assert candidate(2, 6, 3) == -1

    # Check some edge cases that are easy to work out by hand.
    assert candidate(1, 1, 1) == 0.43, "This prints if this assert fails 2 (also good for debugging!)"
    assert candidate(2, 2, 10) == -1

if __name__ == "__main__":
    import sys
    import importlib.util
    import traceback

    spec = importlib.util.spec_from_file_location("main", "main.py")
    main_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_module)

    func_name = "triangle_area"

    if hasattr(main_module, func_name):
        func = getattr(main_module, func_name)
        try:
            check(func)
            print("\n✓ SUCCESS: All tests passed!")
        except AssertionError:
            print("\n✗ FAILURE: Test failed")
            traceback.print_exc()
            sys.exit(1)
    else:
        print(f"Error: Function '{func_name}' not found in main.py")
        sys.exit(1)

