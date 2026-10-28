# SPDX-License-Identifier: Apache-2.0
"""Geometric negative controls for the development-only TD1 diagnostic."""
import unittest
from localization_decomposition import decompose


class DecompositionTests(unittest.TestCase):
    def test_center_offset_vs_size_mismatch_have_distinct_counterfactuals(self):
        rows = [
            {"class": 0, "corners": (4, 4, 16, 16), "area": 144, "index": 7},
            {"class": 1, "corners": (8, 8, 28, 28), "area": 400, "index": 8},
        ]
        # Offset only: fixing centers gives IoU 1, fixing sizes does not.
        shifted = {"class": 0, "corners": [12, 4, 24, 16]}
        # Size only: fixing sizes gives IoU 1; correcting centers cannot.
        oversized = {"class": 1, "corners": [3, 3, 33, 33]}
        result = decompose(rows, [shifted, oversized])
        self.assertEqual(len(result["small"]["low_iou_correct_class_cases"]), 1)
        case = result["small"]["low_iou_correct_class_cases"][0]
        self.assertEqual(case["sample_index"], 7)
        self.assertLess(case["iou"], 0.5)
        self.assertEqual(case["iou_if_center_aligned"], 1)
        self.assertLess(case["iou_if_size_aligned"], 0.5)
        self.assertEqual(result["larger"]["mean_iou_if_size_aligned"], 1)
        self.assertLess(result["larger"]["mean_iou_if_center_aligned"], 1)

    def test_missing_wrong_class_and_background_do_not_pollute_localization_means(self):
        rows = [
            {"class": 0, "corners": (2, 2, 14, 14), "area": 144},
            {"class": 1, "corners": (2, 2, 14, 14), "area": 144},
            {"class": None, "corners": (2, 2, 14, 14), "area": 144},
            {"class": 0, "corners": (2, 2, 24, 24), "area": 484},
        ]
        r = decompose(rows, [None, {"class": 0, "corners": [2,2,14,14]},
                             {"class": 1, "corners": [2,2,14,14]},
                             {"class": 0, "corners": [2,2,24,24]}])
        self.assertEqual(r["small"]["missing_detections"], 1)
        self.assertEqual(r["small"]["wrong_class_detections"], 1)
        self.assertIsNone(r["small"]["mean_iou_correct_class_detected"])
        self.assertEqual(r["larger"]["mean_iou_correct_class_detected"], 1)
        self.assertEqual(r["larger"]["mean_abs_center_error_px"], [0,0])

    def test_perfect_predictions_are_lossless(self):
        rows = [{"class": 0, "corners": (3, 5, 18, 20), "area": 225}]
        r = decompose(rows, [{"class": 0, "corners": [3,5,18,20]}])["small"]
        self.assertEqual(r["mean_iou_correct_class_detected"], 1)
        self.assertEqual(r["mean_iou_if_center_aligned"], 1)
        self.assertEqual(r["mean_iou_if_size_aligned"], 1)
        self.assertEqual(r["low_iou_correct_class_cases"], [])

    def test_rejects_invalid_geometry_or_area_and_mismatched_rows(self):
        row = {"class": 0, "corners": (4,4,16,16), "area": 144}
        for pred in [{"class": 0, "corners": [4,4,16,float('nan')]},
                     {"class": 0, "corners": [4,4,65,16]},
                     {"class": True, "corners": [4,4,16,16]},
                     {"class": 0, "corners": [4,4,4,16]}]:
            with self.subTest(pred=pred), self.assertRaises(ValueError):
                decompose([row], [pred])
        with self.assertRaises(ValueError):
            decompose([row], [])
        with self.assertRaises(ValueError):
            decompose([{**row, "area": 143}], [None])


if __name__ == '__main__':
    unittest.main()
