// SPDX-License-Identifier: Apache-2.0
use facts_yolo::{ffi::types::*, fy_abi_query};
use std::mem::MaybeUninit;
use std::ptr::{addr_of_mut, null_mut};

fn diagnostic(buffer: &mut [u8]) -> FyDiagnostic {
    FyDiagnostic {
        struct_size: size_of::<FyDiagnostic>() as u32,
        struct_version: FY_STRUCT_VERSION,
        required_flags: 0,
        data: buffer.as_mut_ptr().cast(),
        capacity: buffer.len(),
        required_bytes: usize::MAX,
    }
}

#[test]
fn abi_query_reports_1_0() {
    let mut out = FyAbiInfo::default();
    assert_eq!(unsafe { fy_abi_query(&mut out, null_mut()) }, FY_OK);
    assert_eq!((out.abi_major, out.abi_minor), (1, 0));
    let end = out.build_id.iter().position(|b| *b == 0).unwrap();
    assert!(end > 0 && end < 64);
    assert!(out.build_id[..end].is_ascii());
}

#[test]
fn null_output_is_invalid() {
    assert_eq!(unsafe { fy_abi_query(null_mut(), null_mut()) }, FY_INVALID_ARGUMENT);
}

#[test]
fn prefix_only_allocation_rejected() {
    let mut prefix = [8_u32, 1_u32];
    let before = prefix;
    assert_eq!(
        unsafe { fy_abi_query(prefix.as_mut_ptr().cast(), null_mut()) },
        FY_INVALID_ARGUMENT
    );
    assert_eq!(prefix, before);
}

#[test]
fn unknown_version_rejected_without_mutating_output() {
    let mut out = FyAbiInfo { struct_version: 2, abi_major: 77, ..Default::default() };
    assert_eq!(unsafe { fy_abi_query(&mut out, null_mut()) }, FY_UNSUPPORTED_ABI);
    assert_eq!(out.abi_major, 77);
    assert_eq!(out.struct_version, 2);
}

#[test]
fn unknown_required_flag_rejected() {
    let mut out = FyAbiInfo { required_flags: 1, abi_major: 77, ..Default::default() };
    assert_eq!(unsafe { fy_abi_query(&mut out, null_mut()) }, FY_UNSUPPORTED_FEATURE);
    assert_eq!(out.abi_major, 77);
}

#[test]
fn larger_tail_unchanged() {
    #[repr(C)]
    struct Extended { base: FyAbiInfo, tail: [u8; 16] }
    let mut extended = Extended { base: FyAbiInfo::default(), tail: [0xA5; 16] };
    extended.base.struct_size = size_of::<Extended>() as u32;
    assert_eq!(unsafe { fy_abi_query(&mut extended.base, null_mut()) }, FY_OK);
    assert_eq!(extended.base.struct_size, size_of::<Extended>() as u32);
    assert_eq!(extended.tail, [0xA5; 16]);
}

#[test]
fn only_output_header_needs_initialization() {
    let mut out = MaybeUninit::<FyAbiInfo>::uninit();
    let p = out.as_mut_ptr();
    unsafe {
        addr_of_mut!((*p).struct_size).write(size_of::<FyAbiInfo>() as u32);
        addr_of_mut!((*p).struct_version).write(1);
        addr_of_mut!((*p).required_flags).write(0);
        assert_eq!(fy_abi_query(p, null_mut()), FY_OK);
        assert_eq!(out.assume_init().abi_major, 1);
    }
}

#[test]
fn misaligned_output_rejected_before_reading_header() {
    #[repr(align(8))]
    struct Aligned([u8; 128]);
    let mut backing = Aligned([0xA5; 128]);
    let p = unsafe { backing.0.as_mut_ptr().add(1).cast() };
    assert_eq!(unsafe { fy_abi_query(p, null_mut()) }, FY_INVALID_ARGUMENT);
    assert_eq!(backing.0, [0xA5; 128]);
}

#[test]
fn bad_diagnostic_precedes_output_validation() {
    let mut bytes = [0xA5; 16];
    let mut diag = diagnostic(&mut bytes);
    diag.struct_version = 99;
    let mut out = FyAbiInfo { abi_major: 77, ..Default::default() };
    assert_eq!(unsafe { fy_abi_query(&mut out, &mut diag) }, FY_UNSUPPORTED_ABI);
    assert_eq!(out.abi_major, 77);
    assert_eq!(diag.required_bytes, usize::MAX);
    assert_eq!(bytes, [0xA5; 16]);
}

#[test]
fn diagnostic_prefix_only_is_rejected() {
    #[repr(C, align(8))]
    struct Prefix([u32; 2]);
    let mut prefix = Prefix([8, 1]);
    let mut out = FyAbiInfo { abi_major: 77, ..Default::default() };
    assert_eq!(
        unsafe { fy_abi_query(&mut out, prefix.0.as_mut_ptr().cast()) },
        FY_INVALID_ARGUMENT
    );
    assert_eq!(out.abi_major, 77);
}

#[test]
fn success_clears_diagnostic() {
    let mut bytes = [0xA5; 8];
    let mut diag = diagnostic(&mut bytes);
    let mut out = FyAbiInfo::default();
    assert_eq!(unsafe { fy_abi_query(&mut out, &mut diag) }, FY_OK);
    assert_eq!(diag.required_bytes, 0);
    assert_eq!(bytes[0], 0);
    assert_eq!(bytes[1..], [0xA5; 7]);
}

#[test]
fn null_positive_capacity_diagnostic_rejected() {
    let mut diag = diagnostic(&mut []);
    diag.data = null_mut();
    diag.capacity = 1;
    let mut out = FyAbiInfo { abi_major: 77, ..Default::default() };
    assert_eq!(unsafe { fy_abi_query(&mut out, &mut diag) }, FY_INVALID_ARGUMENT);
    assert_eq!(out.abi_major, 77);
}

#[test]
fn oversized_diagnostic_capacity_rejected() {
    let mut buffer = [0xA5; 8];
    let mut diag = diagnostic(&mut buffer);
    diag.capacity = usize::MAX;
    let mut out = FyAbiInfo::default();
    assert_eq!(unsafe { fy_abi_query(&mut out, &mut diag) }, FY_INVALID_ARGUMENT);
    assert_eq!(buffer, [0xA5; 8]);
}

#[test]
fn diagnostic_zero_capacity_reports_required_bytes() {
    let mut diag = diagnostic(&mut []);
    diag.data = null_mut();
    let mut out = FyAbiInfo { struct_version: 2, ..Default::default() };
    assert_eq!(unsafe { fy_abi_query(&mut out, &mut diag) }, FY_UNSUPPORTED_ABI);
    assert_eq!(diag.required_bytes, "unsupported descriptor version".len());
}

#[test]
fn diagnostic_shortage_preserves_primary_status_and_canary() {
    for cap in [1, 2, 5, 31] {
        let mut bytes = [0xA5; 64];
        let mut diag = diagnostic(&mut bytes[..cap]);
        let mut out = FyAbiInfo { struct_version: 2, ..Default::default() };
        assert_eq!(unsafe { fy_abi_query(&mut out, &mut diag) }, FY_UNSUPPORTED_ABI);
        let message = b"unsupported descriptor version";
        let count = (cap - 1).min(message.len());
        assert_eq!(diag.required_bytes, message.len());
        assert_eq!(&bytes[..count], &message[..count]);
        assert_eq!(bytes[count], 0);
        assert!(bytes[cap..].iter().all(|b| *b == 0xA5));
    }
}

#[test]
fn diagnostic_required_count_need_not_be_initialized() {
    let mut bytes = [0xA5; 8];
    let mut diag = MaybeUninit::<FyDiagnostic>::uninit();
    let p = diag.as_mut_ptr();
    let mut out = FyAbiInfo::default();
    unsafe {
        addr_of_mut!((*p).struct_size).write(size_of::<FyDiagnostic>() as u32);
        addr_of_mut!((*p).struct_version).write(1);
        addr_of_mut!((*p).required_flags).write(0);
        addr_of_mut!((*p).data).write(bytes.as_mut_ptr().cast());
        addr_of_mut!((*p).capacity).write(bytes.len());
        assert_eq!(fy_abi_query(&mut out, p), FY_OK);
        assert_eq!(diag.assume_init().required_bytes, 0);
    }
}

#[test]
fn diagnostics_are_per_call() {
    let workers: Vec<_> = (0..16).map(|index| std::thread::spawn(move || {
        for _ in 0..100 {
            let mut bytes = [0xA5; 64];
            let mut diag = diagnostic(&mut bytes);
            let mut out = FyAbiInfo::default();
            let (expected, message) = if index % 2 == 0 {
                out.struct_version = 9;
                (FY_UNSUPPORTED_ABI, "unsupported descriptor version")
            } else {
                out.required_flags = 1;
                (FY_UNSUPPORTED_FEATURE, "unsupported required flags")
            };
            assert_eq!(unsafe { fy_abi_query(&mut out, &mut diag) }, expected);
            assert_eq!(&bytes[..diag.required_bytes], message.as_bytes());
            assert_eq!(bytes[diag.required_bytes], 0);
        }
    })).collect();
    for worker in workers { worker.join().unwrap(); }
}
