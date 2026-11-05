// SPDX-License-Identifier: Apache-2.0
//! Reduced-size checks of both adapters; the full predeclared gates run in
//! the `fyp-compare gates` command (see docs/testing/td2-framework-comparison.md).
use fyp_compare::frameworks;
use fyp_compare::gates::{
    forward_error, g3_ratio, g5_ratio, loss_step, reference_gradients, relative_l2, train,
    G2_RELATIVE, G4_RELATIVE,
};
use fyp_core::data::batch;
use fyp_core::net::{initial_parameters, PARAMETER_COUNT, SEED};
use fyp_core::sample_plan::gradient_samples;

#[test]
fn both_forwards_match_the_f64_reference() {
    let p = initial_parameters();
    let samples = batch(SEED, 0, 2);
    for fw in frameworks() {
        let e = forward_error(fw.as_ref(), &p, &samples).unwrap();
        assert!(e <= G2_RELATIVE, "{}: relative error {e}", fw.name());
    }
}

#[test]
fn both_gradients_match_reference_finite_differences_on_every_tensor() {
    let p = initial_parameters();
    let samples = batch(SEED, 0, 2);
    // First predeclared sample of each tensor (16 parameters).
    let plan = gradient_samples();
    let mut picks = vec![plan[0]];
    for w in plan.windows(2) {
        if fyp_core::net::tensors()
            .iter()
            .any(|t| t.offset <= w[1] && w[0] < t.offset)
        {
            picks.push(w[1]);
        }
    }
    assert_eq!(picks.len(), 16);
    let fd = reference_gradients(&p, &samples, &picks);
    for fw in frameworks() {
        let (_, g) = loss_step(fw.as_ref(), &p, &samples).unwrap();
        assert_eq!(g.len(), PARAMETER_COUNT);
        for (i, f) in picks.iter().zip(&fd) {
            let r = g3_ratio(f64::from(g[*i]), *f);
            assert!(
                r <= 1.0,
                "{} parameter {i}: {} vs fd {f} (ratio {r})",
                fw.name(),
                g[*i]
            );
        }
    }
}

#[test]
fn burn_and_candle_gradients_agree() {
    let p = initial_parameters();
    let samples = batch(SEED, 0, 2);
    let g: Vec<Vec<f32>> = frameworks()
        .iter()
        .map(|fw| loss_step(fw.as_ref(), &p, &samples).unwrap().1)
        .collect();
    let d = relative_l2(&g[0], &g[1]);
    assert!(d <= G4_RELATIVE, "relative L2 {d}");
}

#[test]
fn short_training_is_finite_deterministic_and_reduces_loss() {
    for fw in frameworks() {
        let a = train(fw.as_ref(), 40, 4).unwrap();
        let b = train(fw.as_ref(), 40, 4).unwrap();
        assert_eq!(a, b, "{} is not deterministic", fw.name());
        assert!(
            g5_ratio(&a) < 1.0,
            "{} did not reduce loss: {a:?}",
            fw.name()
        );
    }
}

#[test]
fn malformed_inputs_are_rejected() {
    for fw in frameworks() {
        assert!(fw.head(&[0.0; 3], &[0.0; 3], 1).is_err());
        let p = initial_parameters();
        assert!(fw.head(&p, &[0.0; 7], 1).is_err());
        assert!(fw
            .step(&p, &vec![0.0; 3 * 64 * 64], 1, &mut |_| vec![0.0; 3])
            .is_err());
    }
}
