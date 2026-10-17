"""Write requirements.txt with the EXACT versions installed in this environment.

Run it once, inside the virtual environment where `pytest` and
`python -m src.run_all` both work, so the pins are versions that were actually
tested together:

    python scripts/pin_requirements.py            # rewrites requirements.txt
    python scripts/pin_requirements.py --check    # exit 1 if file != environment
"""
from __future__ import annotations

import argparse
import sys
from importlib import metadata
from pathlib import Path

# Direct dependencies only (what src/ and tests/ import).
PACKAGES = ("numpy", "pandas", "matplotlib", "scikit-learn", "pytest")


def render() -> str:
    lines = [
        f"# Pinned from a working environment (Python {sys.version.split()[0]}).",
        "# Recreate with:  pip install -r requirements.txt",
        "# scikit-learn is used only for baselines and tests, never as a core model.",
    ]
    lines += [f"{name}=={metadata.version(name)}" for name in PACKAGES]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="only compare, do not write")
    parser.add_argument("--path", default="requirements.txt")
    args = parser.parse_args()
    target = Path(args.path)
    content = render()
    if args.check:
        current = target.read_text(encoding="utf-8").splitlines()
        wanted = [line for line in content.splitlines() if not line.startswith("#")]
        found = [line.strip() for line in current if line.strip() and not line.startswith("#")]
        if sorted(found) != sorted(wanted):
            print("requirements.txt does not match the installed versions")
            return 1
        print("requirements.txt matches the installed versions")
        return 0
    target.write_text(content, encoding="utf-8")
    print(f"wrote {target}:\n{content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
