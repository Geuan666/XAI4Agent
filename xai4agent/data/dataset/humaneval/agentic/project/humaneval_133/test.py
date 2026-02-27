def check(candidate):

    # Check some simple cases
    assert candidate([1,2,3])==14, "This prints if this assert fails 1 (good for debugging!)"
    assert candidate([1.0,2,3])==14, "This prints if this assert fails 1 (good for debugging!)"
    assert candidate([1,3,5,7])==84, "This prints if this assert fails 1 (good for debugging!)"
    assert candidate([1.4,4.2,0])==29, "This prints if this assert fails 1 (good for debugging!)"
    assert candidate([-2.4,1,1])==6, "This prints if this assert fails 1 (good for debugging!)"

    assert candidate([100,1,15,2])==10230, "This prints if this assert fails 1 (good for debugging!)"
    assert candidate([10000,10000])==200000000, "This prints if this assert fails 1 (good for debugging!)"
    assert candidate([-1.4,4.6,6.3])==75, "This prints if this assert fails 1 (good for debugging!)"
    assert candidate([-1.4,17.9,18.9,19.9])==1086, "This prints if this assert fails 1 (good for debugging!)"


    # Check some edge cases that are easy to work out by hand.
    assert candidate([0])==0, "This prints if this assert fails 2 (also good for debugging!)"
    assert candidate([-1])==1, "This prints if this assert fails 2 (also good for debugging!)"
    assert candidate([-1,1,0])==2, "This prints if this assert fails 2 (also good for debugging!)"

if __name__ == "__main__":
    import sys
    import importlib.util
    import traceback

    spec = importlib.util.spec_from_file_location("main", "main.py")
    main_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_module)

    func_name = "sum_squares"

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

