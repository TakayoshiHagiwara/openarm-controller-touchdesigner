"""Allow ``python -m evaluation.grip`` execution."""

from evaluation.grip.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
