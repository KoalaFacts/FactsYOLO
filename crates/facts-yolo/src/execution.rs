// SPDX-License-Identifier: Apache-2.0
//! Admission and explicit state validity, independent of advisory mutex poisoning.
use crate::{Error, ffi::types::*};
use std::sync::{Mutex, MutexGuard, TryLockError};

/// Reusable is an adapter guarantee, not a classification inferred from a status.
pub(crate) enum Failure {
    Reusable(Error),
    Fatal(Error),
}

const BUSY: Error = Error {
    status: FY_BUSY,
    message: "session is already executing",
};
const POISONED: Error = Error {
    status: FY_POISONED,
    message: "session execution state is poisoned",
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Ready,
    Running,
    Poisoned,
}

pub(crate) struct Gate {
    state: Mutex<State>,
}

/// Every admitted path poisons unless a valid completion explicitly restores Ready.
/// The lock remains held during cleanup/finalization as well as backend execution.
struct Admission<'a> {
    state: MutexGuard<'a, State>,
}

impl Drop for Admission<'_> {
    fn drop(&mut self) {
        if *self.state == State::Running {
            *self.state = State::Poisoned;
        }
    }
}

impl Gate {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(State::Ready),
        }
    }

    pub(crate) fn run<T>(
        &self,
        operation: impl FnOnce() -> Result<T, Failure>,
    ) -> Result<T, Error> {
        let state = self.state.try_lock().map_err(|error| match error {
            TryLockError::WouldBlock => BUSY,
            TryLockError::Poisoned(_) => POISONED,
        })?;
        if *state != State::Ready {
            return Err(POISONED);
        }
        let mut admission = Admission { state };
        *admission.state = State::Running;
        match operation() {
            Ok(value) => {
                *admission.state = State::Ready;
                Ok(value)
            }
            Err(Failure::Reusable(error)) => {
                *admission.state = State::Ready;
                Err(error)
            }
            Err(Failure::Fatal(error)) => Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const BACKEND: Error = Error {
        status: FY_BACKEND_ERROR,
        message: "controlled backend failure",
    };

    #[test]
    fn success_restores_ready_and_nested_admission_is_busy() {
        let gate = Gate::new();
        let value = gate.run(|| {
            assert_eq!(gate.run(|| Ok(())).unwrap_err().status, FY_BUSY);
            Ok(7)
        });
        assert_eq!(value.unwrap(), 7);
        assert_eq!(gate.run(|| Ok(8)).unwrap(), 8);
    }

    #[test]
    fn explicit_adapter_recovery_restores_ready() {
        let gate = Gate::new();
        let error = gate.run::<()>(|| Err(Failure::Reusable(BACKEND)));
        assert_eq!(error.unwrap_err().status, FY_BACKEND_ERROR);
        assert_eq!(gate.run(|| Ok(9)).unwrap(), 9);
    }

    #[test]
    fn fatal_fault_poisoning_is_independent_of_mutex_unwinding() {
        let gate = Gate::new();
        let error = gate.run::<()>(|| Err(Failure::Fatal(BACKEND)));
        assert_eq!(error.unwrap_err().status, FY_BACKEND_ERROR);
        let error = gate.run(|| {
            panic!("poisoned execution must not be admitted");
            #[allow(unreachable_code)]
            Ok(())
        });
        assert_eq!(error.unwrap_err().status, FY_POISONED);
    }

    #[test]
    fn unwinding_marks_admitted_state_poisoned() {
        let gate = Gate::new();
        let result =
            std::panic::catch_unwind(|| gate.run::<()>(|| panic!("controlled gate panic")));
        assert!(result.is_err());
        assert_eq!(gate.run(|| Ok(())).unwrap_err().status, FY_POISONED);
    }
}
