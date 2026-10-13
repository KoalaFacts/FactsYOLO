// SPDX-License-Identifier: Apache-2.0
//! Test-only latch, owned per session rather than per shared model.
use crate::{Error, ffi::types::FY_INTERNAL_ERROR};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

const CONTROL: Error = Error {
    status: FY_INTERNAL_ERROR,
    message: "test execution control did not complete",
};

struct State {
    running: bool,
    permit: bool,
}

pub(super) struct Latch {
    state: Mutex<State>,
    changed: Condvar,
}

impl Latch {
    pub(super) fn new() -> Self {
        Self {
            state: Mutex::new(State {
                running: false,
                permit: false,
            }),
            changed: Condvar::new(),
        }
    }

    pub(super) fn block(&self) -> Result<(), Error> {
        let mut state = self.state.lock().map_err(|_| CONTROL)?;
        state.running = true;
        self.changed.notify_all();
        // Fail-safe for a broken test harness only, not an inference deadline.
        let (mut state, _) = self
            .changed
            .wait_timeout_while(state, Duration::from_secs(30), |state| !state.permit)
            .map_err(|_| CONTROL)?;
        let permitted = state.permit;
        state.running = false;
        state.permit = false;
        self.changed.notify_all();
        if permitted { Ok(()) } else { Err(CONTROL) }
    }

    pub(super) fn wait_running(&self, timeout_ms: u32) -> Result<(), Error> {
        let state = self.state.lock().map_err(|_| CONTROL)?;
        let (state, _) = self
            .changed
            .wait_timeout_while(
                state,
                Duration::from_millis(u64::from(timeout_ms)),
                |state| !state.running,
            )
            .map_err(|_| CONTROL)?;
        if state.running { Ok(()) } else { Err(CONTROL) }
    }

    pub(super) fn resume(&self) -> Result<(), Error> {
        // A single pre-entry permit is allowed for reliable cleanup after a
        // failed wait. Repeated resumes coalesce; one call consumes the permit.
        let mut state = self.state.lock().map_err(|_| CONTROL)?;
        state.permit = true;
        self.changed.notify_all();
        Ok(())
    }
}
