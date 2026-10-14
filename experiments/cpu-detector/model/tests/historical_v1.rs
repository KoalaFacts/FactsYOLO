// SPDX-License-Identifier: Apache-2.0
//! Synthetic compatibility vectors: no trained weights or natural-image claims.
#[path = "../../../../tests/compat/td1_v1_vectors.rs"]
mod historical;
use fy_tiny_model::Model;
use sha2::{Digest, Sha256};

#[test]
fn historical_v1_bytes_and_parameter_bits_survive_current_reader_writer() {
    for vector in historical::VECTORS {
        let bytes = vector.bytes();
        // Even semantic rejection vectors have valid checksums. A checksum
        // failure must not hide a broken header/numeric validation check.
        let digest = Sha256::new()
            .chain_update(&bytes[..24])
            .chain_update(&bytes[56..])
            .finalize();
        assert_eq!(&bytes[24..56], &digest[..], "{}", vector.name);
        let loaded = Model::from_bytes(&bytes);
        assert_eq!(loaded.is_ok(), vector.accepted, "{}", vector.name);
        if let Ok(model) = loaded {
            assert_eq!(model.weights.len(), 12_807);
            for (weight, expected) in model.weights.iter().zip(bytes[56..].chunks_exact(4)) {
                assert_eq!(weight.to_bits().to_le_bytes(), expected, "{}", vector.name);
            }
            assert_eq!(model.to_bytes().unwrap(), bytes, "{}", vector.name);
        }
    }
}

#[test]
fn historical_v1_predictions_preserve_channels_classes_and_half_score() {
    let red: Vec<u8> = (0..4096).flat_map(|_| [255, 0, 0]).collect();
    let black = vec![0; 12_288];
    let blue: Vec<u8> = (0..4096).flat_map(|_| [0, 0, 255]).collect();
    let zero = Model::from_bytes(&historical::vector("signed-zero").bytes()).unwrap();
    assert!(zero.detect(&red).unwrap().is_none());
    for (name, class) in [("red-gate", 0), ("class-one-gate", 1)] {
        let model = Model::from_bytes(&historical::vector(name).bytes()).unwrap();
        let detection = model.detect(&red).unwrap().unwrap();
        assert_eq!(detection.class, class);
        assert_eq!(detection.corners, [16.0, 16.0, 48.0, 48.0]);
        // Analytical softmax of logits [0,10,0], captured independently.
        assert!((detection.score - 0.999_909_2).abs() < 0.000_001);
        assert!(model.detect(&black).unwrap().is_none());
        assert!(model.detect(&blue).unwrap().is_none());
    }
    let half = Model::from_bytes(&historical::vector("half-score").bytes()).unwrap();
    let detection = half.detect(&black).unwrap().unwrap();
    assert_eq!(detection.class, 0);
    assert_eq!(detection.score, 0.5);
    assert_eq!(detection.corners, [16.0, 16.0, 48.0, 48.0]);
}

#[test]
fn historical_v1_rejects_every_short_prefix_and_trailing_payload() {
    let bytes = historical::vector("red-gate").bytes();
    for length in 0..51_284 {
        assert!(
            Model::from_bytes(&bytes[..length]).is_err(),
            "length {length}"
        );
    }
    for extra in [1, 4, 64] {
        let mut trailing = bytes.clone();
        trailing.resize(51_284 + extra, 0);
        assert!(Model::from_bytes(&trailing).is_err());
    }
    for index in [24, 55, 56, 51_283] {
        let mut corrupt = bytes.clone();
        corrupt[index] ^= 1;
        assert!(Model::from_bytes(&corrupt).is_err(), "byte {index}");
    }
}
