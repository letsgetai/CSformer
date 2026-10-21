"""CSformer command-line entry point (delegates to run_longExp)."""
import runpy

if __name__ == "__main__":
    runpy.run_module("run_longExp", run_name="__main__")
