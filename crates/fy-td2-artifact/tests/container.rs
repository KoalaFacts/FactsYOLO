// SPDX-License-Identifier: Apache-2.0
use fy_td2_artifact::*;

const LIMITS: Limits = Limits {
    max_artifact_bytes: 1 << 20,
    max_parser_steps: 1 << 20,
};

fn layer(kind: u32, offset: u64, count: u64) -> LayerRecord {
    LayerRecord {
        kind,
        activation: 1,
        fields: [3, 16, 3, 3, 2, 1, 1, 0, 0, 0],
        parameter_offset: offset,
        parameter_count: count,
    }
}

fn sample_layers() -> Vec<LayerRecord> {
    vec![layer(1, 0, 5), layer(2, 5, 0), layer(3, 5, 3)]
}

fn sample_parameters() -> Vec<f32> {
    vec![0.5, -1.25, 3.0, 0.0, -0.0, 1e-30, 2.5, -7.0]
}

fn sample() -> Vec<u8> {
    encode(
        TASK_T1_DETECTION,
        320,
        82,
        &sample_layers(),
        &sample_parameters(),
    )
    .unwrap()
}

/// Re-seal a mutated artifact so only the intended defect remains.
fn reseal(mut bytes: Vec<u8>) -> Vec<u8> {
    use sha2::{Digest, Sha256};
    let at = bytes.len() - DIGEST_BYTES;
    let digest: [u8; 32] = Sha256::digest(&bytes[..at]).into();
    bytes[at..].copy_from_slice(&digest);
    bytes
}

fn put_u32(bytes: &mut [u8], at: usize, v: u32) {
    bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], at: usize, v: u64) {
    bytes[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn valid_artifact_round_trips_every_field() {
    let bytes = sample();
    // 64 header + 3*64 table = 256, already aligned; 8 params; 32 digest.
    assert_eq!(parameter_region_offset(3), 256);
    assert_eq!(bytes.len(), 256 + 8 * 4 + 32);
    let a = parse(&bytes, LIMITS).unwrap();
    assert_eq!(
        a.header,
        Header {
            task_kind: 1,
            input_size: 320,
            class_count: 82,
            layer_count: 3,
            parameter_count: 8,
            total_bytes: bytes.len() as u64,
        }
    );
    assert_eq!(a.layers, sample_layers());
    assert_eq!(
        a.parameters.iter().map(|p| p.to_bits()).collect::<Vec<_>>(),
        sample_parameters()
            .iter()
            .map(|p| p.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(&a.model_id[..], &bytes[bytes.len() - 32..]);
}

#[test]
fn byte_layout_is_frozen() {
    // Golden digest of the sample: any change to field order, widths,
    // padding or alignment changes it. Recomputed only with a spec change.
    let bytes = sample();
    assert_eq!(&bytes[..8], b"FYTDv002");
    assert_eq!(&bytes[8..12], &2u32.to_le_bytes());
    assert_eq!(&bytes[40..48], &(bytes.len() as u64).to_le_bytes());
    assert_eq!(
        hex(&parse(&bytes, LIMITS).unwrap().model_id),
        GOLDEN_SAMPLE_DIGEST
    );
}

const GOLDEN_SAMPLE_DIGEST: &str =
    "7a3491d2f7e2728bf4f46e7f677ccc75cebc8c8518c219cf37014458cc60451f";

#[test]
fn digest_is_sha256_known_answer() {
    use sha2::{Digest, Sha256};
    // FIPS 180-2 "abc" vector: confirms the hash in use is SHA-256.
    assert_eq!(
        hex(&Sha256::digest(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn parameter_region_is_always_simd_aligned() {
    for n in [1, 2, 3, 7, 64, MAX_LAYERS] {
        assert_eq!(parameter_region_offset(n) % PARAMETER_ALIGNMENT, 0, "{n}");
    }
    let bytes = encode(TASK_T1_DETECTION, 64, 1, &[layer(1, 0, 2)], &[1.0, 2.0]).unwrap();
    assert_eq!(&bytes[128..132], &1.0f32.to_le_bytes());
}

#[test]
fn caller_limits_are_enforced_before_any_work() {
    let bytes = sample();
    let n = bytes.len();
    let exact = Limits {
        max_artifact_bytes: n,
        max_parser_steps: n,
    };
    parse(&bytes, exact).unwrap();
    assert_eq!(
        parse(
            &bytes,
            Limits {
                max_artifact_bytes: n - 1,
                ..exact
            }
        ),
        Err(Error::TooLarge)
    );
    assert_eq!(
        parse(
            &bytes,
            Limits {
                max_parser_steps: n - 1,
                ..exact
            }
        ),
        Err(Error::StepLimit)
    );
}

#[test]
fn truncated_and_extended_inputs_are_rejected() {
    let bytes = sample();
    assert_eq!(parse(&[], LIMITS), Err(Error::Truncated));
    assert_eq!(parse(&bytes[..95], LIMITS), Err(Error::Truncated));
    // Any other length disagrees with the declared total.
    assert_eq!(
        parse(&bytes[..bytes.len() - 1], LIMITS),
        Err(Error::LengthMismatch)
    );
    let mut longer = bytes.clone();
    longer.push(0);
    assert_eq!(parse(&longer, LIMITS), Err(Error::LengthMismatch));
}

#[test]
fn header_field_rejections() {
    let base = sample();
    let case = |at: usize, v: u32, want: Error| {
        let mut b = base.clone();
        put_u32(&mut b, at, v);
        assert_eq!(
            parse(&reseal(b), LIMITS),
            Err(want),
            "offset {at} value {v}"
        );
    };
    case(0, u32::from_le_bytes(*b"FYTE"), Error::BadMagic);
    case(8, 1, Error::UnsupportedVersion);
    case(8, 3, Error::UnsupportedVersion);
    case(12, 0, Error::UnsupportedTask);
    case(12, 2, Error::UnsupportedTask);
    case(16, 0, Error::BadInputSize);
    case(16, 321, Error::BadInputSize);
    case(16, 1056, Error::BadInputSize);
    case(20, 0, Error::BadClassCount);
    case(20, 121, Error::BadClassCount);
    case(24, 0, Error::BadLayerCount);
    case(24, MAX_LAYERS + 1, Error::BadLayerCount);
    case(28, 1, Error::ReservedNonZero);
    case(60, 1, Error::ReservedNonZero);
    // Boundaries that are accepted.
    for (at, v) in [(16, 32), (16, 1024), (20, 1), (20, 120)] {
        let mut b = base.clone();
        put_u32(&mut b, at, v);
        parse(&reseal(b), LIMITS).unwrap();
    }
}

#[test]
fn declared_counts_must_match_the_layout() {
    let base = sample();
    // Layer count claims 4 records: the layout sum no longer matches.
    let mut b = base.clone();
    put_u32(&mut b, 24, 4);
    assert_eq!(parse(&reseal(b), LIMITS), Err(Error::LengthMismatch));
    // Parameter count 9 or a huge value that would overflow length arithmetic.
    for v in [9u64, u64::MAX / 2, u64::MAX] {
        let mut b = base.clone();
        put_u64(&mut b, 32, v);
        assert_eq!(parse(&reseal(b), LIMITS), Err(Error::LengthMismatch), "{v}");
    }
    let mut b = base.clone();
    put_u64(&mut b, 40, base.len() as u64 + 4);
    assert_eq!(parse(&reseal(b), LIMITS), Err(Error::LengthMismatch));
}

#[test]
fn layer_records_must_be_nonzero_kind_and_contiguous() {
    let params = sample_parameters();
    let enc = |layers: &[LayerRecord]| encode(TASK_T1_DETECTION, 320, 82, layers, &params).unwrap();
    assert_eq!(
        parse(
            &enc(&[layer(1, 0, 5), layer(0, 5, 0), layer(3, 5, 3)]),
            LIMITS
        ),
        Err(Error::BadLayerKind)
    );
    // Gap, overlap, wrong start, short and long coverage, overflow.
    for layers in [
        [layer(1, 0, 4), layer(2, 5, 0), layer(3, 5, 3)],
        [layer(1, 0, 5), layer(2, 4, 1), layer(3, 5, 3)],
        [layer(1, 1, 5), layer(2, 6, 0), layer(3, 6, 2)],
        [layer(1, 0, 5), layer(2, 5, 0), layer(3, 5, 2)],
        [layer(1, 0, 5), layer(2, 5, 0), layer(3, 5, 4)],
        [layer(1, 0, u64::MAX), layer(2, u64::MAX, 2), layer(3, 1, 7)],
        // Same counts and total as the valid table, offsets permuted or repeated:
        // only the per-record offset check can reject these.
        [layer(1, 3, 5), layer(2, 0, 0), layer(3, 0, 3)],
        [layer(1, 0, 5), layer(2, 0, 0), layer(3, 5, 3)],
    ] {
        assert_eq!(
            parse(&enc(&layers), LIMITS),
            Err(Error::ParameterLayout),
            "{layers:?}"
        );
    }
}

#[test]
fn nonfinite_parameters_are_rejected() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut params = sample_parameters();
        params[6] = bad;
        let bytes = encode(TASK_T1_DETECTION, 320, 82, &sample_layers(), &params).unwrap();
        assert_eq!(parse(&bytes, LIMITS), Err(Error::NonFiniteParameter));
    }
    // Subnormals and signed zero are finite and kept bit-exactly.
    let mut params = sample_parameters();
    params[0] = f32::from_bits(1);
    let bytes = encode(TASK_T1_DETECTION, 320, 82, &sample_layers(), &params).unwrap();
    assert_eq!(parse(&bytes, LIMITS).unwrap().parameters[0].to_bits(), 1);
}

#[test]
fn any_single_bit_flip_is_rejected() {
    let bytes = sample();
    for i in 0..bytes.len() {
        for bit in 0..8 {
            let mut b = bytes.clone();
            b[i] ^= 1 << bit;
            assert!(
                parse(&b, LIMITS).is_err(),
                "flip byte {i} bit {bit} accepted"
            );
        }
    }
}

#[test]
fn digest_mismatch_is_reported_when_only_the_trailer_differs() {
    let mut b = sample();
    let last = b.len() - 1;
    b[last] ^= 0x80;
    assert_eq!(parse(&b, LIMITS), Err(Error::DigestMismatch));
    // A changed parameter without resealing is also a digest mismatch.
    let mut b = sample();
    b[256] ^= 1;
    assert_eq!(parse(&b, LIMITS), Err(Error::DigestMismatch));
}
