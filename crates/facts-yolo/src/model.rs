// SPDX-License-Identifier: Apache-2.0
//! Immutable optional TD1 model state and isolated private contract fixtures.
#[cfg(any(feature = "contract-fixtures", feature = "td1"))]
use crate::error::INVALID;
#[cfg(feature = "contract-fixtures")]
use crate::fixtures;
use crate::{Error, image::ImageView};
use crate::{
    execution::Failure,
    result::{Candidate, ImageTransform},
    session::UNSUPPORTED,
};

pub(crate) struct Model {
    backend_id: u32,
    #[cfg(feature = "td1")]
    tiny: Option<fy_tiny_model::InferenceModel>,
    #[cfg(feature = "td1")]
    identity: [u8; 32],
    #[cfg(feature = "contract-fixtures")]
    fixture_id: u32,
    #[cfg(feature = "contract-fixtures")]
    _live: fixtures::Live,
}

impl Model {
    #[cfg(feature = "td1")]
    pub(crate) fn open(bytes: &[u8], parser_steps: usize) -> Result<Self, Error> {
        use crate::ffi::types::*;
        let unsupported = Error {
            status: FY_UNSUPPORTED_MODEL,
            message: "unsupported or invalid TD1 artifact",
        };
        // Fixed-size recognition bounds hashing and parameter allocation.
        if bytes.len() != 51284 || !bytes.starts_with(b"FYTDv001") {
            return Err(unsupported);
        }
        if parser_steps < bytes.len() {
            return Err(Error {
                status: FY_LIMIT_EXCEEDED,
                message: "TD1 parser byte-work limit exceeded",
            });
        }
        let tiny = fy_tiny_model::InferenceModel::from_bytes(bytes).map_err(|_| unsupported)?;
        let mut identity = [0; 32];
        identity.copy_from_slice(&bytes[24..56]);
        Ok(Self {
            backend_id: FY_BACKEND_TD1_CPU,
            tiny: Some(tiny),
            identity,
            #[cfg(feature = "contract-fixtures")]
            fixture_id: 0,
            #[cfg(feature = "contract-fixtures")]
            _live: fixtures::Live::new(1),
        })
    }
    pub(crate) fn validate_image(&self, _image: &ImageView<'_>) -> Result<(), Error> {
        #[cfg(feature = "td1")]
        if self.tiny.is_some() {
            return if _image.width() == 64 && _image.height() == 64 {
                Ok(())
            } else {
                Err(INVALID)
            };
        }
        #[cfg(feature = "contract-fixtures")]
        if _image.width() != 2 || _image.height() != 2 {
            return Err(INVALID);
        }
        Ok(())
    }

    #[cfg(feature = "contract-fixtures")]
    pub(crate) fn fixture(fixture_id: u32) -> Result<Self, Error> {
        if !(fixtures::BASIC..=fixtures::PAYLOAD_DROP_PANIC).contains(&fixture_id) {
            return Err(INVALID);
        }
        Ok(Self {
            backend_id: fixtures::FY_BACKEND_TEST_FIXTURE,
            #[cfg(feature = "td1")]
            tiny: None,
            #[cfg(feature = "td1")]
            identity: [0; 32],
            fixture_id,
            _live: fixtures::Live::new(1),
        })
    }

    pub(crate) fn backend_id(&self) -> u32 {
        self.backend_id
    }

    pub(crate) fn identity(&self) -> [u8; 32] {
        #[cfg(feature = "td1")]
        if self.tiny.is_some() {
            return self.identity;
        }
        #[cfg(feature = "contract-fixtures")]
        {
            let mut id = [0; 32];
            id[0] = self.fixture_id as u8;
            id
        }
        #[cfg(not(feature = "contract-fixtures"))]
        {
            [0; 32]
        }
    }

    #[cfg(feature = "contract-fixtures")]
    pub(crate) fn is_blocking(&self) -> bool {
        self.fixture_id == fixtures::BLOCKING
    }

    pub(crate) fn execute(
        &self,
        _image: &ImageView<'_>,
    ) -> Result<(Vec<Candidate>, ImageTransform), Failure> {
        #[cfg(feature = "td1")]
        if let Some(model) = &self.tiny {
            // The checked view has already been validated as 64x64. Reuse the
            // portable TD1 preprocessing directly from its borrowed pixels,
            // avoiding per-detection compact-RGB and feature Vec allocations.
            let detection = model
                .detect_with_rgb(|x, y| _image.rgb_at(x as u32, y as u32))
                .map_err(|_| {
                    Failure::Fatal(Error {
                        status: crate::ffi::types::FY_MODEL_OUTPUT_INVALID,
                        message: "invalid TD1 output or pixel view",
                    })
                })?;
            let candidates = detection
                .into_iter()
                .map(|d| Candidate {
                    x_min: d.corners[0],
                    y_min: d.corners[1],
                    x_max: d.corners[2],
                    y_max: d.corners[3],
                    score: d.score,
                    class_id: d.class as u32,
                    source_index: 0,
                })
                .collect();
            return Ok((
                candidates,
                ImageTransform {
                    scale_x: 1.0,
                    scale_y: 1.0,
                    pad_x: 0.0,
                    pad_y: 0.0,
                },
            ));
        }
        #[cfg(feature = "contract-fixtures")]
        {
            let values: &[Candidate] = match self.fixture_id {
                fixtures::BASIC | fixtures::BLOCKING => &fixtures::BASIC_CANDIDATES,
                fixtures::EMPTY => &[],
                fixtures::INVALID_OUTPUT => &fixtures::INVALID_CANDIDATES,
                fixtures::PANIC => {
                    fixtures::record_execution();
                    panic!("private-fixture execution panic");
                }
                fixtures::PAYLOAD_DROP_PANIC => {
                    fixtures::record_execution();
                    std::panic::panic_any(PayloadDropPanic);
                }
                // No operation has begun; this specific path guarantees reuse.
                _ => return Err(Failure::Reusable(UNSUPPORTED)),
            };
            fixtures::record_execution();
            Ok((values.to_vec(), fixtures::TRANSFORM))
        }
        #[cfg(not(feature = "contract-fixtures"))]
        {
            // Normal M1 still has no model format/backend and cannot execute.
            Err(Failure::Reusable(UNSUPPORTED))
        }
    }
}

#[cfg(feature = "contract-fixtures")]
struct PayloadDropPanic;

#[cfg(feature = "contract-fixtures")]
impl Drop for PayloadDropPanic {
    fn drop(&mut self) {
        panic!("private-fixture payload destructor panic");
    }
}
