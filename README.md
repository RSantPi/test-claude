# test-claude

A small repo for trying out working with Claude on GitHub.

## Dice roller

`roll.py` rolls dice using notation like `2d6+1`. It only needs Python 3.

```bash
python3 roll.py          # rolls 1d6
python3 roll.py 2d6+1    # rolls two six-sided dice and adds 1
python3 roll.py d20-2    # rolls one twenty-sided die and subtracts 2
```

Run the tests with:

```bash
python3 -m unittest
```
