#!/usr/bin/env python3
"""Run all tests without requiring pytest. Exit non-zero on any failure."""

import importlib
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODULES = ["tests.test_core", "tests.test_io_roundtrip", "tests.test_antispoof"]


def main():
    total = passed = 0
    for modname in MODULES:
        mod = importlib.import_module(modname)
        fns = [v for k, v in sorted(vars(mod).items()) if k.startswith("test_") and callable(v)]
        print(f"\n=== {modname} ({len(fns)} tests) ===")
        for fn in fns:
            total += 1
            try:
                fn()
                passed += 1
                print(f"  PASS {fn.__name__}")
            except Exception:
                print(f"  FAIL {fn.__name__}")
                traceback.print_exc()
    print(f"\n{'=' * 40}\n{passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
