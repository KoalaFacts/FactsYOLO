// SPDX-License-Identifier: Apache-2.0
//! RED checkpoint for the private execution-state contract.
#![allow(dead_code)]
use crate::{Error, ffi::types::*};

pub(crate) enum Failure {
    Reusable(Error),
    Fatal(Error),
}

pub(crate) struct Gate;

impl Gate {
    pub(crate) fn new() -> Self {
        Self
    }

    pub(crate) fn run<T>(
        &self,
        _operation: impl FnOnce() -> Result<T, Failure>,
    ) -> Result<T, Error> {
        Err(Error {
            status: FY_INTERNAL_ERROR,
            message: "execution gate is not implemented",
        })
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
        let result = std::panic::catch_unwind(|| {
            gate.run::<()>(|| panic!("controlled gate panic"))
        });
        assert!(result.is_err());
        assert_eq!(gate.run(|| Ok(())).unwrap_err().status, FY_POISONED);
    }
}
