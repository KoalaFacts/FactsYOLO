// SPDX-License-Identifier: Apache-2.0
//! Frozen synthetic wire vectors anchored to PR13 a58f6c44beb6b7c0c35452da2d3057de41b6ccb9.
//! No current model constants, serializer, decoder, RNG or trained weights are used.
//! Literal sparse F32 bit patterns; fixed digests captured with Python hashlib.
//! Do not regenerate expectations to accommodate a change to the current format.

pub struct Vector {
    pub name: &'static str,
    pub accepted: bool,
    header_patches: &'static [(usize, u8)],
    parameter_bits: &'static [(usize, u32)],
    digest: &'static str,
}
impl Vector {
    pub fn bytes(&self) -> Vec<u8> {
        let mut bytes = vec![0_u8; 51_284];
        bytes[..24].copy_from_slice(&[
            70, 89, 84, 68, 118, 48, 48, 49, 1, 0, 0, 0, 7, 50, 0, 0, 28, 200, 0, 0, 0, 0, 0, 0,
        ]);
        for &(offset, value) in self.header_patches {
            bytes[offset] = value;
        }
        for &(index, bits) in self.parameter_bits {
            bytes[56 + index * 4..60 + index * 4].copy_from_slice(&bits.to_le_bytes());
        }
        for (index, pair) in self.digest.as_bytes().chunks_exact(2).enumerate() {
            let digit = |c: u8| match c {
                b'0'..=b'9' => c - b'0',
                b'a'..=b'f' => c - b'a' + 10,
                _ => panic!("invalid frozen hex"),
            };
            bytes[24 + index] = digit(pair[0]) * 16 + digit(pair[1]);
        }
        bytes
    }
}
pub fn vector(name: &str) -> &'static Vector {
    VECTORS
        .iter()
        .find(|v| v.name == name)
        .expect("known historical vector")
}
pub const VECTORS: &[Vector] = &[
    Vector {
        name: "signed-zero",
        accepted: true,
        header_patches: &[],
        parameter_bits: &[(0, 0x80000000)],
        digest: "c6d4c8020353f6205a53397e116849da47f5d7fe1ed0d1804f441a4cf5fbe0e9",
    },
    Vector {
        name: "red-gate",
        accepted: true,
        header_patches: &[],
        parameter_bits: &[(0, 0x3f800000), (12353, 0x41200000), (12806, 0x80000000)],
        digest: "e4ff39bfab7910dade7b1d1661f782cfe14979e4f97c341de24591bfdf793e6b",
    },
    Vector {
        name: "class-one-gate",
        accepted: true,
        header_patches: &[],
        parameter_bits: &[(0, 0x3f800000), (12354, 0x41200000)],
        digest: "8e6b784676c94585698110af13fa5aa4f3afa291bc851bca7bdd793e9c352c8e",
    },
    Vector {
        name: "half-score",
        accepted: true,
        header_patches: &[],
        parameter_bits: &[(12800, 0xc61c4000)],
        digest: "43c59dc97cb0834db6437da6cc8672fdaec29bc004c57848ebbd6a7dbe4fb576",
    },
    Vector {
        name: "upper-endpoint",
        accepted: true,
        header_patches: &[],
        parameter_bits: &[(0, 0x461c4000)],
        digest: "ed8e871e0ff629e3167b1ae8c865a7da76c5ae4fe8163956b8782e55bcb144c9",
    },
    Vector {
        name: "lower-endpoint",
        accepted: true,
        header_patches: &[],
        parameter_bits: &[(0, 0xc61c4000)],
        digest: "e1bc6f7e143da5a228a139af2715aa7751166a0369a4f070e7e3936b074f3ac1",
    },
    Vector {
        name: "nan",
        accepted: false,
        header_patches: &[],
        parameter_bits: &[(0, 0x7fc00000)],
        digest: "ce06606db6acd1691c0f8bd09cc4fc1186bae467f193e87fc2f2f1c29b1a2f66",
    },
    Vector {
        name: "positive-infinity",
        accepted: false,
        header_patches: &[],
        parameter_bits: &[(0, 0x7f800000)],
        digest: "d7d8cd7db96e2f4aa81d638101bc065d4e372abe15cb2cf0bac6d112c2e2f8cd",
    },
    Vector {
        name: "negative-infinity",
        accepted: false,
        header_patches: &[],
        parameter_bits: &[(0, 0xff800000)],
        digest: "e0f1cdf892d223ff2170ff36d3836c655892b96dbc81196823fb27d5a1039968",
    },
    Vector {
        name: "above-upper",
        accepted: false,
        header_patches: &[],
        parameter_bits: &[(0, 0x461c4001)],
        digest: "e1a518671bfb9b4c7a96219ab1663d8b086499c4103d4c0b6aea40fa7001fe89",
    },
    Vector {
        name: "below-lower",
        accepted: false,
        header_patches: &[],
        parameter_bits: &[(0, 0xc61c4001)],
        digest: "655d537c79a29d0cf4c75d3943fd49ae8a78ad9c892d21a80d17724ceb3e2518",
    },
    Vector {
        name: "schema-two",
        accepted: false,
        header_patches: &[(7, 50)],
        parameter_bits: &[],
        digest: "95759156e3800b0987ee860e6dd8ef447566a108d72d8762a96778e9606ceb2b",
    },
    Vector {
        name: "architecture-two",
        accepted: false,
        header_patches: &[(8, 2)],
        parameter_bits: &[],
        digest: "7740593801a66560df57792dc7a88ddac72afa225207841da8c572cb631e97eb",
    },
    Vector {
        name: "wrong-count",
        accepted: false,
        header_patches: &[(12, 6)],
        parameter_bits: &[],
        digest: "a59be707d9590d6cc3db451b92c6434b3149f1da12884dc8045cd39740bf94b8",
    },
    Vector {
        name: "wrong-payload-length",
        accepted: false,
        header_patches: &[(16, 27)],
        parameter_bits: &[],
        digest: "13cbf3b91740ac47dc09ec0ce350e7524f7c96d43d24aefda963080ee71006b8",
    },
    Vector {
        name: "big-endian-architecture",
        accepted: false,
        header_patches: &[(8, 0), (11, 1)],
        parameter_bits: &[],
        digest: "d57ac41f4f95435d99174e44dc57ae51d4fa8876d29c6514a26c510535a4a2a5",
    },
    Vector {
        name: "max-payload-length",
        accepted: false,
        header_patches: &[
            (16, 255),
            (17, 255),
            (18, 255),
            (19, 255),
            (20, 255),
            (21, 255),
            (22, 255),
            (23, 255),
        ],
        parameter_bits: &[],
        digest: "16c9268af09b075d5c9d2ab88c51e02515be5dddb9de44f81e76ffb6a4ac7255",
    },
];
