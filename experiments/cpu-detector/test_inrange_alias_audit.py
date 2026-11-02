# SPDX-License-Identifier: Apache-2.0
"""In-range counterbalance to the deliberately out-of-range 4px witness."""

import unittest

from inrange_alias_audit import axis_overlap, box_iou, inrange_alias_report, tile_counts
from spatial_alias_audit import block_features


def render(box):
    pixels = bytearray(bytes((7, 7, 7)) * (64 * 64))
    for y in range(box[1], box[3]):
        for x in range(box[0], box[2]):
            pixels[(y * 64 + x) * 3:(y * 64 + x) * 3 + 3] = (230, 20, 20)
    return bytes(pixels)


class InrangeAliasAuditTests(unittest.TestCase):
    def test_axis_and_tile_geometry_against_direct_pixel_counts(self):
        for box in [(2, 2, 14, 14), (10, 8, 22, 24), (8, 10, 24, 22),
                    (41, 41, 63, 63), (7, 19, 29, 31)]:
            pixels = render(box)
            got = tile_counts(box)
            self.assertEqual(len(got), 64)
            self.assertEqual(sum(got), (box[2] - box[0]) * (box[3] - box[1]))
            direct = [sum(pixels[(y*64+x)*3] == 230
                          for y in range(by*8, by*8+8)
                          for x in range(bx*8, bx*8+8))
                      for by in range(8) for bx in range(8)]
            self.assertEqual(list(got), direct)

    def test_inrange_geometry_collision_is_real_not_forced_iou50_failure(self):
        a, b = (10, 8, 22, 24), (8, 10, 24, 22)
        self.assertNotEqual(a, b)
        self.assertEqual(tile_counts(a), tile_counts(b))
        self.assertNotEqual(render(a), render(b))
        self.assertEqual(block_features(render(a), 8), block_features(render(b), 8))
        self.assertNotEqual(block_features(render(a), 4), block_features(render(b), 4))
        self.assertAlmostEqual(box_iou(a, b), 0.6)
        # Choosing one of the true boxes meets IoU50 for BOTH samples.
        self.assertGreaterEqual(box_iou(a, b), 0.5)
        self.assertEqual(box_iou(a, a), 1.0)

    def test_exhaustive_registered_rectangle_range(self):
        r = inrange_alias_report()
        self.assertEqual(r['total_rectangles'], 40 * 40 * 11 * 11)
        self.assertEqual(r['unique_tile_count_signatures'], 193543)
        self.assertEqual(r['ambiguous_groups'], 57)
        self.assertEqual(r['colliding_pairs'], 57)
        self.assertEqual(r['max_group_size'], 2)
        self.assertEqual(r['minimum_pairwise_iou'], 0.6)
        self.assertTrue(r['every_colliding_pair_has_a_shared_iou50_box'])
        self.assertEqual(r['minimum_pair'], [[10, 8, 22, 24], [8, 10, 24, 22]])
        self.assertFalse(r['training_executed'])
        self.assertFalse(r['final_holdout_evaluated'])
        self.assertFalse(r['detector_accuracy_claimed'])

    def test_guard_rejects_geometry_outside_registration(self):
        with self.assertRaises(ValueError):
            axis_overlap(1, 12)
        with self.assertRaises(ValueError):
            axis_overlap(2, 11)
        with self.assertRaises(ValueError):
            axis_overlap(42, 12)
        with self.assertRaises(ValueError):
            axis_overlap(2, 23)


if __name__ == '__main__':
    unittest.main()
