// SPDX-License-Identifier: Apache-2.0
use fy_cpu_evaluate::backend;
use fy_tiny_model::*;
fn check(name: &str) {
    let b = backend(name).unwrap();
    let mut m = Model {
        weights: vec![0.01; PARAMETERS],
    };
    for v in &mut m.weights[W1..B1] {
        *v = 0.1;
    }
    let input = (0..2 * INPUTS)
        .map(|i| (i % 17) as f32 / 20.0)
        .collect::<Vec<_>>();
    let expected = input
        .as_chunks::<INPUTS>()
        .0
        .iter()
        .flat_map(|x| m.forward(x).unwrap())
        .collect::<Vec<_>>();
    let output = b.forward(&m.weights, &input, 2).unwrap();
    assert_eq!(output.len(), 14);
    for (a, e) in output.iter().zip(&expected) {
        assert!(
            (a - e).abs() <= 1e-4 + 1e-4 * e.abs(),
            "{name} prediction {a} != {e}"
        );
    }
    let upstream = (0..14)
        .map(|i| [0.1, -0.2, 0.1, 0.25, -0.1, 0.05, 0.2][i % 7])
        .collect::<Vec<_>>();
    let gradient = b.backward(&m.weights, &input, 2, &upstream).unwrap();
    assert_eq!(gradient.len(), PARAMETERS);
    assert!(gradient.iter().any(|x| x.abs() > 0.01));
    for i in [
        0,
        63,
        64,
        W1 - 1,
        W1,
        B1 - 1,
        B1,
        W2 - 1,
        W2,
        PARAMETERS - 1,
    ] {
        let original = m.weights[i];
        m.weights[i] = original + 1e-3;
        let p = input
            .as_chunks::<INPUTS>()
            .0
            .iter()
            .flat_map(|x| m.forward(x).unwrap())
            .zip(&upstream)
            .map(|(x, g)| x * g)
            .sum::<f32>();
        m.weights[i] = original - 1e-3;
        let n = input
            .as_chunks::<INPUTS>()
            .0
            .iter()
            .flat_map(|x| m.forward(x).unwrap())
            .zip(&upstream)
            .map(|(x, g)| x * g)
            .sum::<f32>();
        m.weights[i] = original;
        let fd = (p - n) / 0.002;
        assert!(
            (gradient[i] - fd).abs() <= 2e-3 + 2e-2 * fd.abs(),
            "{name} gradient{i}:{} vs{fd}",
            gradient[i]
        );
    }
    assert!(b.forward(&[], &input, 2).is_err());
    assert!(b.forward(&m.weights, &[], 2).is_err());
    assert!(b.backward(&m.weights, &input, 2, &[]).is_err());
}
#[test]
fn burn_executes_real_forward_and_autodiff() {
    check("burn");
}
#[test]
fn candle_executes_real_forward_and_autodiff() {
    check("candle");
}
#[test]
fn candidate_parameter_gradients_agree() {
    let m = Model::initialized(404);
    let input = vec![0.2; INPUTS * 4];
    let upstream = vec![0.1; 7 * 4];
    let a = backend("burn")
        .unwrap()
        .backward(&m.weights, &input, 4, &upstream)
        .unwrap();
    let b = backend("candle")
        .unwrap()
        .backward(&m.weights, &input, 4, &upstream)
        .unwrap();
    assert_eq!(a.len(), PARAMETERS);
    assert_eq!(b.len(), PARAMETERS);
    for (i, (a, b)) in a.into_iter().zip(b).enumerate() {
        assert!(
            (a - b).abs() <= 2e-5 + 3e-4 * b.abs(),
            "gradient{i}: {a} vs{b}"
        );
    }
}
