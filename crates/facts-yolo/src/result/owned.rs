// SPDX-License-Identifier: Apache-2.0
//! Completed host result: no image, session, model or device references.
use super::Finalized;
use crate::ffi::types::*;

pub(crate) struct OwnedResult {
    pub(crate) detections: Vec<FyDetection>,
    pub(crate) info: FyResultInfo,
    #[cfg(feature = "contract-fixtures")]
    _live: crate::fixtures::Live,
}

impl OwnedResult {
    pub(crate) fn new(
        finalized: Finalized,
        source: (u32, u32),
        execution: [u32; 5],
        model_id: [u8; 32],
    ) -> Self {
        let info = FyResultInfo {
            struct_size: size_of::<FyResultInfo>() as u32,
            struct_version: FY_STRUCT_VERSION,
            required_flags: 0,
            source_width: source.0,
            source_height: source.1,
            count: finalized.detections.len(),
            result_flags: if finalized.truncated { FY_RESULT_TRUNCATED } else { 0 },
            backend_id: execution[0],
            device_kind: execution[1],
            device_index: execution[2],
            precision: execution[3],
            execution_flags: execution[4],
            model_id,
        };
        Self {
            detections: finalized.detections,
            info,
            #[cfg(feature = "contract-fixtures")]
            _live: crate::fixtures::Live::new(3),
        }
    }
}
