// SPDX-License-Identifier: Apache-2.0
// An artifact-backed inference model has no mutable weights after validation.
use fy_tiny_model::{InferenceModel, Model, PARAMETERS};
use sha2::{Digest, Sha256};
use std::sync::Arc;

fn image(seed: usize) -> Vec<u8> {
    (0_usize..64 * 64 * 3)
        .map(|i| (i.wrapping_mul(31).wrapping_add(seed * 17) % 256) as u8)
        .collect()
}

#[test]
fn validated_inference_preserves_exact_mutable_reference_results() {
    let mut mutable = Model::initialized(404);
    mutable.weights[PARAMETERS - 6] = 8.0;
    let frozen = InferenceModel::from_bytes(&mutable.to_bytes().unwrap()).unwrap();
    let features = [0.25; 192];
    assert_eq!(
        frozen.forward(&features).unwrap(),
        mutable.forward(&features).unwrap()
    );
    for seed in 0..8 {
        let pixels = image(seed);
        let reference = mutable.detect(&pixels).unwrap();
        let result = frozen
            .detect_with_rgb(|x, y| {
                let at = (y * 64 + x) * 3;
                Some([pixels[at], pixels[at + 1], pixels[at + 2]])
            })
            .unwrap();
        assert_eq!(result, reference, "image {seed}");
    }

    // A later mutation of the public training/reference Model must not change
    // the independent validated artifact state or bypass the reference guard.
    let unchanged = frozen.forward(&features).unwrap();
    mutable.weights[0] = f32::NAN;
    assert_eq!(
        mutable.forward(&features).unwrap_err(),
        "invalid TD1 parameters"
    );
    assert_eq!(frozen.forward(&features).unwrap(), unchanged);
    assert_eq!(
        frozen.forward(&[f32::NAN; 192]).unwrap_err(),
        "invalid TD1 features"
    );
    assert!(frozen.detect_with_rgb(|_, _| None).is_err());
}

#[test]
fn validated_inference_preserves_background_suppression_and_is_shareable() {
    let mut reference = Model::initialized(404);
    reference.weights[PARAMETERS - 7] = 16.0;
    let frozen = Arc::new(InferenceModel::from_bytes(&reference.to_bytes().unwrap()).unwrap());
    let pixels = image(17);
    assert!(reference.detect(&pixels).unwrap().is_none());
    let expected = reference.forward(&[0.0; 192]).unwrap();
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let state = Arc::clone(&frozen);
            std::thread::spawn(move || state.forward(&[0.0; 192]).unwrap())
        })
        .collect();
    for handle in handles {
        assert_eq!(handle.join().unwrap(), expected);
    }
    assert!(frozen
        .detect_with_rgb(|x, y| {
            let at = (y * 64 + x) * 3;
            Some([pixels[at], pixels[at + 1], pixels[at + 2]])
        })
        .unwrap()
        .is_none());
}

#[test]
fn validated_inference_rejects_rehashed_invalid_weights_at_load() {
    let bytes = Model::initialized(404).to_bytes().unwrap();
    let mut invalid = bytes.clone();
    invalid[56..60].copy_from_slice(&f32::INFINITY.to_le_bytes());
    let digest = Sha256::new()
        .chain_update(&invalid[..24])
        .chain_update(&invalid[56..])
        .finalize();
    invalid[24..56].copy_from_slice(&digest);
    assert!(InferenceModel::from_bytes(&invalid).is_err());
    let mut corrupt = bytes.clone();
    corrupt[56] ^= 0x01;
    assert!(InferenceModel::from_bytes(&corrupt).is_err());
    assert!(InferenceModel::from_bytes(&bytes[..bytes.len() - 1]).is_err());
}
