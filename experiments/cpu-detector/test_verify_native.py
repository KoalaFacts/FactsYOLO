# SPDX-License-Identifier: Apache-2.0
"""Guard real-artifact C/C# inputs against omitted target classes."""
import unittest

from verify_native import NATIVE_CASES, select_native_cases


class NativeReferenceSelectionTest(unittest.TestCase):
    @staticmethod
    def first_classes(count):
        # Independent LCG replay of the Rust generator's draw counts.
        # This inspects ground truth only, never model predictions.
        state = 909

        def draw():
            nonlocal state
            state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            return state >> 32

        classes = []
        for i in range(count):
            klass = None if i % 5 == 0 else draw() % 2
            for _ in range(4 + 64 * 64 * 3 + (4 if klass is not None else 0)):
                draw()  # x/y/w/h, RGB noise, optional rectangle colour
            classes.append(klass)
        return classes

    def test_six_sample_prefix_omits_class_zero_but_eight_cover_all(self):
        classes = self.first_classes(NATIVE_CASES)
        self.assertEqual(classes, [None, 1, 1, 1, 1, None, 1, 0])
        self.assertNotIn(0, classes[:6])
        rows = [{'index': i, 'expected_class': klass} for i, klass in enumerate(classes)]
        self.assertEqual(select_native_cases(rows), rows)

    def test_missing_reordered_and_forged_class_coverage_fails(self):
        rows = [{'index': i, 'expected_class': klass}
                for i, klass in enumerate(self.first_classes(NATIVE_CASES))]
        with self.assertRaises(ValueError):
            select_native_cases(rows[:6])
        for key, value in [('expected_class', 1), ('index', 99),
                           ('expected_class', True), ('expected_class', 3)]:
            bad = [dict(row) for row in rows]
            bad[-1][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                select_native_cases(bad)


if __name__ == '__main__':
    unittest.main()
