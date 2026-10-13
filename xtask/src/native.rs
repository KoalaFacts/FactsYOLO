// SPDX-License-Identifier: Apache-2.0
//! Task 6 RED checkpoint. These deliberately incomplete gates must not be merged.
#![allow(dead_code)]

pub(super) fn run(_exports_only: bool) -> Result<(), String> {
    Err("C conformance execution is not implemented".to_owned())
}

fn compare_layout(_expected: &str, _actual: &str) -> Result<(), String> {
    Ok(())
}

fn validate_symbols(_expected: &str, _actual: &str) -> Result<(), String> {
    Ok(())
}

fn exported_names(_output: &str) -> String {
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    const LAYOUT: &str = "P pointer 8\nT fy_detection 24 4\nF fy_detection score 16\n";

    #[test]
    fn layout_is_order_independent_and_accepts_crlf() {
        let reordered = "F fy_detection score 16\r\nP pointer 8\r\nT fy_detection 24 4\r\n";
        assert!(compare_layout(LAYOUT, reordered).is_ok());
    }

    #[test]
    fn changed_size_alignment_or_offset_is_rejected() {
        for (from, to) in [("24 4", "28 4"), ("24 4", "24 8"), ("score 16", "score 20")] {
            assert!(compare_layout(LAYOUT, &LAYOUT.replace(from, to)).is_err());
        }
    }

    #[test]
    fn empty_missing_and_extra_layout_records_are_rejected() {
        assert!(compare_layout("", "").is_err());
        assert!(compare_layout(LAYOUT, "P pointer 8\n").is_err());
        assert!(compare_layout(LAYOUT, &(LAYOUT.to_owned() + "F fy_detection fake 0\n")).is_err());
    }

    #[test]
    fn duplicate_and_malformed_layout_records_are_rejected() {
        assert!(compare_layout(LAYOUT, &(LAYOUT.to_owned() + "P pointer 8\n")).is_err());
        assert!(compare_layout("P pointer banana\n", "P pointer banana\n").is_err());
        assert!(compare_layout("unexpected output", "unexpected output").is_err());
    }

    #[test]
    fn exports_require_exact_nonempty_unique_sets() {
        let expected = "fy_abi_query\nfy_detect\n";
        assert!(validate_symbols(expected, "fy_detect\nfy_abi_query\n").is_ok());
        assert!(validate_symbols(expected, "fy_abi_query\n").is_err());
        assert!(validate_symbols(expected, &(expected.to_owned() + "fy_test_model_create\n")).is_err());
        assert!(validate_symbols(expected, &(expected.to_owned() + "fy_detect\n")).is_err());
        assert!(validate_symbols("", "").is_err());
    }

    #[test]
    fn symbol_readers_handle_nm_and_dumpbin_without_debug_string_matches() {
        assert_eq!(
            exported_names("0000000000001c00 T fy_detect\n0000 T other\n"),
            "fy_detect\n"
        );
        assert_eq!(
            exported_names("ordinal hint RVA name\n 1 0 00001000 fy_abi_query\n 2 1 00002000 fy_detect\n"),
            "fy_abi_query\nfy_detect\n"
        );
    }
}
