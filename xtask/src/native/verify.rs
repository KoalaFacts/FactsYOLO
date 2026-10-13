// SPDX-License-Identifier: Apache-2.0
//! Strict comparisons of compiler-produced layouts and defined export names.
use std::collections::{BTreeMap, BTreeSet};

fn layout(text: &str) -> Result<BTreeMap<String, Vec<u64>>, String> {
    if text.is_empty() || text.len() > 65536 {
        return Err("empty or excessive layout output".to_owned());
    }
    let mut entries = BTreeMap::new();
    for line in text.lines() {
        let parts: Vec<_> = line.split_whitespace().collect();
        let (key, numbers) = match parts.as_slice() {
            ["P", "pointer", value] => ("P pointer".to_owned(), vec![*value]),
            ["T", name, size, align] => (format!("T {name}"), vec![*size, *align]),
            ["F", name, field, offset] => (format!("F {name} {field}"), vec![*offset]),
            ["K", name, value] => (format!("K {name}"), vec![*value]),
            _ => return Err(format!("malformed layout record: {line}")),
        };
        let values = numbers
            .into_iter()
            .map(|value| value.parse::<u64>().map_err(|error| error.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        if entries.insert(key, values).is_some() {
            return Err("duplicate layout key".to_owned());
        }
    }
    if !entries.contains_key("P pointer") {
        return Err("layout omits pointer width".to_owned());
    }
    Ok(entries)
}

pub(super) fn compare_layout(expected: &str, actual: &str) -> Result<(), String> {
    let expected = layout(expected)?;
    let actual = layout(actual)?;
    if expected != actual {
        return Err(format!(
            "layout mismatch: expected {expected:?}, actual {actual:?}"
        ));
    }
    Ok(())
}

fn symbols(text: &str) -> Result<BTreeSet<&str>, String> {
    let mut values = BTreeSet::new();
    for line in text.lines() {
        let name = line.trim();
        if !name.starts_with("fy_")
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
            || !values.insert(name)
        {
            return Err(format!("invalid or duplicate export: {name}"));
        }
    }
    if values.is_empty() {
        return Err("empty export set".to_owned());
    }
    Ok(values)
}

pub(super) fn validate_symbols(expected: &str, actual: &str) -> Result<(), String> {
    let expected = symbols(expected)?;
    let actual = symbols(actual)?;
    if expected != actual {
        return Err(format!(
            "export mismatch: expected {expected:?}, actual {actual:?}"
        ));
    }
    Ok(())
}

pub(super) fn exported_names(output: &str) -> String {
    // Input is nm --defined-only or dumpbin /exports, never strings/debug data.
    let mut names = Vec::new();
    for line in output.lines() {
        let parts: Vec<_> = line.split_whitespace().collect();
        let candidate = match parts.as_slice() {
            [address, kind, name]
                if address.bytes().all(|b| b.is_ascii_hexdigit()) && *kind == "T" =>
            {
                Some(*name)
            }
            [ordinal, hint, rva, name, ..]
                if ordinal.parse::<u32>().is_ok()
                    && hint.bytes().all(|b| b.is_ascii_hexdigit())
                    && rva.bytes().all(|b| b.is_ascii_hexdigit()) =>
            {
                Some(*name)
            }
            _ => None,
        };
        if let Some(name) = candidate.filter(|name| name.starts_with("fy_")) {
            names.push(name);
        }
    }
    names.sort_unstable();
    if names.is_empty() {
        String::new()
    } else {
        names.join("\n") + "\n"
    }
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
        let extra = LAYOUT.to_owned() + "F fy_detection fake 0\n";
        assert!(compare_layout(LAYOUT, &extra).is_err());
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
        let leaked = expected.to_owned() + "fy_test_model_create\n";
        assert!(validate_symbols(expected, &leaked).is_err());
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
            exported_names(
                "ordinal hint RVA name\n 1 0 00001000 fy_abi_query\n 2 1 00002000 fy_detect\n"
            ),
            "fy_abi_query\nfy_detect\n"
        );
        assert!(exported_names("debug strings fy_detect\n").is_empty());
    }
}
