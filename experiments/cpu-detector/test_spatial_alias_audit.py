# SPDX-License-Identifier: Apache-2.0
"""Independently check the TD1 spatial-aliasing negative control."""
import hashlib
import struct
import unittest

from spatial_alias_audit import (
    block_features, candidate_pair, iou, spatial_alias_report,
)


class SpatialAliasAuditTests(unittest.TestCase):
    def test_two_different_ground_truth_boxes_have_identical_td1_inputs(self):
        (left, left_box), (right, right_box) = candidate_pair()
        self.assertNotEqual(left, right)
        self.assertNotEqual(left_box, right_box)
        self.assertEqual(len(left), 64 * 64 * 3)
        self.assertEqual(block_features(left, 8), block_features(right, 8))
        self.assertNotEqual(block_features(left, 4), block_features(right, 4))
        self.assertEqual(len(block_features(left, 8)), 192)
        self.assertEqual(len(block_features(left, 4)), 768)

    def test_packed_f32_features_match_independent_per_block_sum(self):
        (pixels, _), _ = candidate_pair()
        actual = block_features(pixels, 8)
        for y, x, channel in [(0, 0, 0), (0, 0, 1), (0, 1, 2), (7, 7, 0)]:
            expected_sum = sum(
                pixels[(row * 64 + col) * 3 + channel]
                for row in range(y * 8, y * 8 + 8)
                for col in range(x * 8, x * 8 + 8)
            )
            expected = struct.pack('<f', expected_sum / (64 * 255))
            actual_value = struct.pack('<f', actual[(y * 8 + x) * 3 + channel])
            self.assertEqual(actual_value, expected)

    def test_iou50_is_incompatible_for_opposite_corner_targets(self):
        (_, a), (_, b) = candidate_pair()
        self.assertEqual(iou(a, b), 0.0)
        self.assertEqual(iou(a, a), 1.0)
        self.assertEqual(iou(a, b), 0.0)
        # Unlike adjacent objects, no one predicted box can have IoU >= .5
        # with both diagonally opposite equal-area boxes.
        report = spatial_alias_report()
        self.assertEqual(report['common_iou50_possible'], False)
        self.assertEqual(report['max_correct_class_iou50_recall_for_pair'], 0.5)
        self.assertEqual(report['minimum_cover_area'], 64)
        self.assertEqual(report['sum_of_target_areas'], 32)
        # Negative control: the adjacent boxes can BOTH be hit by their union.
        self.assertEqual(iou((0, 0, 8, 4), (0, 0, 4, 4)), 0.5)
        self.assertEqual(iou((0, 0, 8, 4), (4, 0, 8, 4)), 0.5)

    def test_invalid_inputs_are_rejected(self):
        (pixels, _), _ = candidate_pair()
        for bad_size in (0, 3, 5, 16):
            with self.subTest(size=bad_size), self.assertRaises(ValueError):
                block_features(pixels, bad_size)
        with self.assertRaises(ValueError):
            block_features(pixels[:-1], 8)
        with self.assertRaises(ValueError):
            iou((0, 0, 0, 4), (0, 0, 4, 4))
        with self.assertRaises(ValueError):
            iou((0, 0, float('nan'), 4), (0, 0, 4, 4))

    def test_frozen_provenance_and_no_quality_claim(self):
        report = spatial_alias_report()
        self.assertEqual(report['status'], 'constructed-spatial-alias-negative-control')
        self.assertFalse(report['training_executed'])
        self.assertFalse(report['final_holdout_evaluated'])
        self.assertFalse(report['detector_accuracy_claimed'])
        self.assertEqual(report['current_feature_count'], 192)
        self.assertEqual(report['candidate_feature_count'], 768)
        self.assertEqual(report['current_parameter_count'], 12807)
        self.assertEqual(report['candidate_parameter_count'], 49671)
        self.assertEqual(report['additional_f32_parameter_bytes'], 147456)
        (a, _), (b, _) = candidate_pair()
        self.assertEqual(report['pixel_sha256'], [hashlib.sha256(a).hexdigest(), hashlib.sha256(b).hexdigest()])


if __name__ == '__main__':
    unittest.main()
