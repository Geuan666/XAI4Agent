def check(candidate):

    # Check some simple cases
    assert True, "This prints if this assert fails 1 (good for debugging!)"
    assert candidate(5, 6, 10) == [11, 4], "Error"
    assert candidate(4, 8, 9) == [12, 1], "Error"
    assert candidate(1, 10, 10) == [11, 0], "Error"
    assert candidate(2, 11, 5) == [7, 0], "Error"

    # Check some edge cases that are easy to work out by hand.
    assert True, "This prints if this assert fails 2 (also good for debugging!)"
    assert candidate(4, 5, 7) == [9, 2], "Error"
    assert candidate(4, 5, 1) == [5, 0], "Error"

if __name__ == "__main__":
    import sys
    import importlib.util
    import traceback

    spec = importlib.util.spec_from_file_location("main", "main.py")
    main_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_module)

    func_name = "eat"

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

