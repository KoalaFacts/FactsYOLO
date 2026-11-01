# SPDX-License-Identifier: Apache-2.0
"""Independent ground-truth selection checks for actual trained TD1 consumers."""
import unittest

from verify_native import NATIVE_CASES, select_native_cases


# Pre-registered from the untrained seed-909 generator, not model predictions.
SPREAD_INDICES = [*range(8), 12, 31, 47, 63, 80, 97, 111, 125]
SPREAD_CLASSES = [None, 1, 1, 1, 1, None, 1, 0,
                  0, 1, 1, 0, None, 0, 1, None]


class NativeReferenceSelectionTest(unittest.TestCase):
    @staticmethod
    def first_classes(count=128):
        # Independent LCG replay of Rust's generator and its draw counts.
        # Only ground-truth labels are observed; no learned weights are used.
        state = 909

        def draw():
            nonlocal state
            state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            return state >> 32

        classes = []
        for i in range(count):
            klass = None if i % 5 == 0 else draw() % 2
            for _ in range(4 + 64 * 64 * 3 + (4 if klass is not None else 0)):
                draw()  # geometry, RGB noise, optional rectangle colour
            classes.append(klass)
        return classes

    @classmethod
    def reference_rows(cls):
        return [{'index': i, 'expected_class': klass}
                for i, klass in enumerate(cls.first_classes())]

    def test_trained_consumers_select_nonprefix_samples_across_holdout(self):
        classes = self.first_classes()
        self.assertEqual(classes[:8], [None, 1, 1, 1, 1, None, 1, 0])
        self.assertNotIn(0, classes[:6])
        self.assertEqual(NATIVE_CASES, 16)
        selected = select_native_cases(self.reference_rows())
        self.assertEqual([row['index'] for row in selected], SPREAD_INDICES)
        self.assertEqual([row['expected_class'] for row in selected], SPREAD_CLASSES)
        self.assertEqual(set(row['expected_class'] for row in selected), {None, 0, 1})

    def test_missing_reordered_or_forged_spread_cases_fail_closed(self):
        rows = self.reference_rows()
        for count in (8, 125):
            with self.subTest(truncated=count), self.assertRaises(ValueError):
                select_native_cases(rows[:count])

        for index, field, value in (
            (12, 'index', 11),
            (97, 'index', True),
            (97, 'expected_class', True),
            (97, 'expected_class', 3),
            (97, 'expected_class', 1),  # otherwise valid but forged label
            (125, 'expected_class', 0),  # falsely converts background to target
        ):
            bad = [dict(row) for row in rows]
            bad[index][field] = value
            with self.subTest(index=index, field=field, value=value), self.assertRaises(ValueError):
                select_native_cases(bad)


if __name__ == '__main__':
    unittest.main()
