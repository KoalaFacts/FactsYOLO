// SPDX-License-Identifier: Apache-2.0
//! Immutable model state. Only the feature-gated test factory constructs it.
#[cfg(feature = "contract-fixtures")]
use crate::{Error, error::INVALID, fixtures};
use crate::{
    execution::Failure,
    result::{Candidate, ImageTransform},
    session::UNSUPPORTED,
};

pub(crate) struct Model {
    backend_id: u32,
    #[cfg(feature = "contract-fixtures")]
    fixture_id: u32,
    #[cfg(feature = "contract-fixtures")]
    _live: fixtures::Live,
}

impl Model {
    #[cfg(feature = "contract-fixtures")]
    pub(crate) fn fixture(fixture_id: u32) -> Result<Self, Error> {
        if !(fixtures::BASIC..=fixtures::PAYLOAD_DROP_PANIC).contains(&fixture_id) {
            return Err(INVALID);
        }
        Ok(Self {
            backend_id: fixtures::FY_BACKEND_TEST_FIXTURE,
            fixture_id,
            _live: fixtures::Live::new(1),
        })
    }

    pub(crate) fn backend_id(&self) -> u32 {
        self.backend_id
    }

    pub(crate) fn identity(&self) -> [u8; 32] {
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

    pub(crate) fn execute(&self) -> Result<(&'static [Candidate], ImageTransform), Failure> {
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
            Ok((values, fixtures::TRANSFORM))
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
