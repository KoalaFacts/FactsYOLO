// SPDX-License-Identifier: Apache-2.0
//! Immutable model state. Only the feature-gated test factory constructs it.
#[cfg(feature = "contract-fixtures")]
use crate::{Error, error::INVALID, fixtures};

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
        #[cfg(feature = "contract-fixtures")]
        debug_assert!((fixtures::BASIC..=fixtures::PAYLOAD_DROP_PANIC).contains(&self.fixture_id));
        self.backend_id
    }
}
