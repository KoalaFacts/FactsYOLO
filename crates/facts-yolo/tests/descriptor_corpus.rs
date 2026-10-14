// SPDX-License-Identifier: Apache-2.0
//! Fixed-seed allocated malformed records; also interpreted by the Miri gate.
use facts_yolo::{ffi::types::*, fy_abi_query};
use std::ptr::null_mut;

#[test]
fn ten_thousand_backed_abi_descriptors_preserve_rejected_storage() {
    #[repr(C)]
    struct Extended {
        base: FyAbiInfo,
        tail: [u8; 16],
    }
    #[repr(C, align(8))]
    struct Prefix([u32; 2]);
    #[repr(align(8))]
    struct Misaligned([u8; 128]);
    let mut seed = 0xFAC7_0008_2026_u64;
    let mut covered = [0; 8];
    for case in 0..10_000 {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Each malformed class gets coverage; seed varies header bits/sentinels.
        let kind = case % 8;
        covered[kind] += 1;
        let marker = (seed >> 32) as u32;
        let mut out = Extended {
            base: FyAbiInfo {
                abi_major: marker,
                ..Default::default()
            },
            tail: [0xA5; 16],
        };
        let expected = match kind {
            0 => {
                out.base.struct_version = 2 + marker % 1000;
                FY_UNSUPPORTED_ABI
            }
            1 => {
                out.base.required_flags = 1_u32 << (marker % 32);
                FY_UNSUPPORTED_FEATURE
            }
            2 => {
                out.base.struct_size = 8 + marker % (size_of::<FyAbiInfo>() as u32 - 8);
                FY_INVALID_ARGUMENT
            }
            3 => {
                let mut prefix = Prefix([8, marker]);
                assert_eq!(
                    unsafe { fy_abi_query(prefix.0.as_mut_ptr().cast(), null_mut()) },
                    FY_INVALID_ARGUMENT
                );
                assert_eq!(prefix.0, [8, marker]);
                continue;
            }
            4 => {
                let mut allocation = Misaligned([0xA5; 128]);
                let pointer = unsafe { allocation.0.as_mut_ptr().add(1).cast() };
                assert_eq!(
                    unsafe { fy_abi_query(pointer, null_mut()) },
                    FY_INVALID_ARGUMENT
                );
                assert_eq!(allocation.0, [0xA5; 128]);
                continue;
            }
            5 => {
                out.base.struct_size = size_of::<Extended>() as u32;
                FY_OK
            }
            6 => {
                let mut text = [0xA5_u8; 16];
                let mut diag = FyDiagnostic {
                    struct_size: size_of::<FyDiagnostic>() as u32,
                    struct_version: 2,
                    required_flags: 0,
                    data: text.as_mut_ptr().cast(),
                    capacity: text.len(),
                    required_bytes: 77,
                };
                assert_eq!(
                    unsafe { fy_abi_query(&mut out.base, &mut diag) },
                    FY_UNSUPPORTED_ABI
                );
                assert_eq!(out.base.abi_major, marker);
                assert_eq!(diag.required_bytes, 77);
                assert_eq!(text, [0xA5; 16]);
                continue;
            }
            _ => FY_OK,
        };
        // Every pointer above and below derives from live initialized backing
        // storage. Prefix-only cases never advertise a nonexistent full record.
        let header = (
            out.base.struct_size,
            out.base.struct_version,
            out.base.required_flags,
        );
        assert_eq!(
            unsafe { fy_abi_query(&mut out.base, null_mut()) },
            expected,
            "case={case}, seed={seed:#x}"
        );
        assert_eq!(out.tail, [0xA5; 16]);
        assert_eq!(
            (
                out.base.struct_size,
                out.base.struct_version,
                out.base.required_flags
            ),
            header
        );
        assert_eq!(
            out.base.abi_major,
            if expected == FY_OK { 1 } else { marker }
        );
    }
    assert_eq!(covered, [1250; 8]);
}
