import random
import unittest

from roll import parse, roll


class ParseTests(unittest.TestCase):
    def test_reads_count_sides_and_modifier(self):
        self.assertEqual(parse("2d6+1"), (2, 6, 1))

    def test_count_defaults_to_one(self):
        self.assertEqual(parse("d20"), (1, 20, 0))

    def test_reads_negative_modifier(self):
        self.assertEqual(parse("3d8-2"), (3, 8, -2))

    def test_rejects_bad_input(self):
        for bad in ("", "banana", "2d", "0d6", "2d0"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse(bad)


class RollTests(unittest.TestCase):
    def test_total_includes_modifier(self):
        rolls, total = roll("4d6+2", rng=random.Random(42))
        self.assertEqual(len(rolls), 4)
        self.assertEqual(total, sum(rolls) + 2)

    def test_rolls_stay_in_range(self):
        rolls, _ = roll("1000d6", rng=random.Random(42))
        self.assertTrue(all(1 <= value <= 6 for value in rolls))


if __name__ == "__main__":
    unittest.main()
