// SPDX-License-Identifier: Apache-2.0
pub fn adam_step(
    weights: &mut [f32],
    gradient: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    step: usize,
) -> Result<(), String> {
    if step == 0
        || step > 2000
        || weights.is_empty()
        || [gradient.len(), m.len(), v.len()]
            .iter()
            .any(|n| *n != weights.len())
        || weights
            .iter()
            .chain(gradient)
            .chain(m.iter())
            .chain(v.iter())
            .any(|x| !x.is_finite())
        || v.iter().any(|x| *x < 0.0)
    {
        return Err("invalid Adam state/gradient/step".into());
    }
    let a = 1.0 - 0.9_f32.powi(step as i32);
    let b = 1.0 - 0.999_f32.powi(step as i32);
    // Validate the complete next state before modifying any caller-owned slice.
    let next = (0..weights.len())
        .map(|i| {
            let nm = 0.9 * m[i] + 0.1 * gradient[i];
            let nv = 0.999 * v[i] + 0.001 * gradient[i] * gradient[i];
            let nw = weights[i] - 0.003 * (nm / a) / ((nv / b).sqrt() + 1e-8);
            (nw, nm, nv)
        })
        .collect::<Vec<_>>();
    if next
        .iter()
        .any(|(w, m, v)| !w.is_finite() || w.abs() > 10000.0 || !m.is_finite() || !v.is_finite())
    {
        return Err("nonfinite/out-of-range Adam update".into());
    }
    for (i, (nw, nm, nv)) in next.into_iter().enumerate() {
        weights[i] = nw;
        m[i] = nm;
        v[i] = nv;
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn adam_changes_real_parameters_with_the_pinned_update_and_rejects_nan() {
        let mut weights = [1.0, -1.0];
        let mut m = [0.0; 2];
        let mut v = [0.0; 2];
        adam_step(&mut weights, &[2.0, -4.0], &mut m, &mut v, 1).unwrap();
        assert!((weights[0] - 0.997).abs() < 1e-6);
        assert!((weights[1] + 0.997).abs() < 1e-6);
        let before = (weights, m, v);
        assert!(adam_step(&mut weights, &[f32::NAN, 0.0], &mut m, &mut v, 2).is_err());
        assert_eq!((weights, m, v), before);
        assert!(adam_step(&mut weights, &[f32::MAX, 0.0], &mut m, &mut v, 2).is_err());
        assert_eq!((weights, m, v), before);
    }
}
