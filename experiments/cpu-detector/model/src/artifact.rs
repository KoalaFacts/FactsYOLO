// SPDX-License-Identifier: Apache-2.0
use crate::{Model, PARAMETERS};
use sha2::{Digest, Sha256};
const SIZE: usize = 56 + PARAMETERS * 4;
impl Model {
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        self.validate()?;
        let mut bytes = Vec::with_capacity(SIZE);
        bytes.extend_from_slice(b"FYTDv001");
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&(PARAMETERS as u32).to_le_bytes());
        bytes.extend_from_slice(&((PARAMETERS * 4) as u64).to_le_bytes());
        bytes.extend_from_slice(&[0; 32]);
        for w in &self.weights {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        let hash = Sha256::new()
            .chain_update(&bytes[..24])
            .chain_update(&bytes[56..])
            .finalize();
        bytes[24..56].copy_from_slice(&hash);
        Ok(bytes)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        // Validate the complete fixed layout before slicing or allocating.
        if bytes.len() != SIZE {
            return Err("invalid artifact length".into());
        }
        if &bytes[..8] != b"FYTDv001"
            || bytes[8..12] != 1_u32.to_le_bytes()
            || bytes[12..16] != (PARAMETERS as u32).to_le_bytes()
            || bytes[16..24] != ((PARAMETERS * 4) as u64).to_le_bytes()
        {
            return Err("unsupported artifact header".into());
        }
        let hash = Sha256::new()
            .chain_update(&bytes[..24])
            .chain_update(&bytes[56..])
            .finalize();
        if bytes[24..56] != hash[..] {
            return Err("artifact checksum mismatch".into());
        }
        let weights = bytes[56..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let model = Self { weights };
        model.validate()?;
        Ok(model)
    }
}
