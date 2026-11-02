#!/usr/bin/env python3
"""Roll dice from the command line, using notation like 2d6+1."""

import argparse
import random
import re

DICE_PATTERN = re.compile(r"^(\d*)d(\d+)([+-]\d+)?$", re.IGNORECASE)


def parse(notation):
    """Turn a string like '2d6+1' into (count, sides, modifier)."""
    match = DICE_PATTERN.match(notation.strip())
    if not match:
        raise ValueError(f"Can't read {notation!r}. Try something like 2d6 or d20+3.")

    count = int(match.group(1) or 1)
    sides = int(match.group(2))
    modifier = int(match.group(3) or 0)

    if count < 1 or sides < 1:
        raise ValueError("You need at least one die with at least one side.")
    return count, sides, modifier


def roll(notation, rng=random):
    """Roll the dice and return (individual rolls, total)."""
    count, sides, modifier = parse(notation)
    rolls = [rng.randint(1, sides) for _ in range(count)]
    return rolls, sum(rolls) + modifier


def main():
    parser = argparse.ArgumentParser(description="Roll dice, e.g. 2d6+1")
    parser.add_argument(
        "notation", nargs="?", default="1d6", help="dice to roll (default: 1d6)"
    )
    args = parser.parse_args()

    try:
        rolls, total = roll(args.notation)
    except ValueError as error:
        parser.error(str(error))

    print(f"{args.notation}: {rolls} = {total}")


if __name__ == "__main__":
    main()
