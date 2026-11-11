// SPDX-License-Identifier: Apache-2.0
//! First real TD2 Stage B CPU autodiff comparison against the frozen Python oracle.
//!
//! This uses the existing evaluation-only Burn/Candle Cargo lock. It is NOT a
//! deployed model, a framework decision, or a TD1 training change.
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
struct Inputs {
    x: Vec<f64>,
    conv: Vec<f64>,
    depthwise: Vec<f64>,
    gamma: Vec<f64>,
    beta: Vec<f64>,
    target: Vec<f64>,
}

#[derive(Deserialize)]
struct Oracle {
    inputs: Inputs,
}

impl Inputs {
    fn verify(&self) -> Result<(), String> {
        for (name, data, expected) in [
            ("x", &self.x, 32),
            ("conv", &self.conv, 36),
            ("depthwise", &self.depthwise, 18),
            ("gamma", &self.gamma, 2),
            ("beta", &self.beta, 2),
            ("target", &self.target, 32),
        ] {
            let invalid_value = data
                .iter()
                .any(|v| !v.is_finite() || !(*v as f32).is_finite());
            if data.len() != expected || invalid_value {
                return Err(format!(
                    "{name}: invalid length, nonfinite value or f32 overflow"
                ));
            }
        }
        Ok(())
    }

    fn f32(&self, name: &str) -> Vec<f32> {
        let data = match name {
            "x" => &self.x,
            "conv" => &self.conv,
            "depthwise" => &self.depthwise,
            "gamma" => &self.gamma,
            "beta" => &self.beta,
            "target" => &self.target,
            _ => unreachable!("fixed fixture field"),
        };
        data.iter().map(|v| *v as f32).collect()
    }
}

fn candle(f: &Inputs) -> candle_core::Result<Value> {
    use candle_core::{DType, Device, Tensor, Var};

    let device = Device::Cpu;
    let x = Var::from_slice(&f.f32("x"), (1, 2, 4, 4), &device)?;
    let conv = Var::from_slice(&f.f32("conv"), (2, 2, 3, 3), &device)?;
    let dw = Var::from_slice(&f.f32("depthwise"), (2, 1, 3, 3), &device)?;
    let gamma = Var::from_slice(&f.f32("gamma"), (1, 2, 1, 1), &device)?;
    let beta = Var::from_slice(&f.f32("beta"), (1, 2, 1, 1), &device)?;
    let target = Tensor::from_slice(&f.f32("target"), (1, 2, 4, 4), &device)?;

    // Explicit train-mode BN: population variance over four spatial values,
    // never Candle's unbiased variance helper or running inference statistics.
    let y = x.conv2d(&conv, 0, 1, 1, 1)?;
    let z = y.conv2d(&dw, 1, 1, 1, 2)?;
    let mean = z.reshape((2, 4))?.mean_keepdim(1)?.reshape((1, 2, 1, 1))?;
    let centered = z.broadcast_sub(&mean)?;
    let var = centered
        .sqr()?
        .reshape((2, 4))?
        .mean_keepdim(1)?
        .reshape((1, 2, 1, 1))?;
    let denom = (var + 1e-5)?.sqrt()?;
    let bn = centered
        .broadcast_div(&denom)?
        .broadcast_mul(&gamma)?
        .broadcast_add(&beta)?;
    let sigmoid_denom = ((&bn * -1.0)?.exp()? + 1.0)?;
    let sigmoid = Tensor::ones((1, 2, 2, 2), DType::F32, &device)?.div(&sigmoid_denom)?;
    let output = bn.mul(&sigmoid)?.upsample_nearest2d(4, 4)?;
    let residual = output.sub(&target)?;
    let loss = residual.sqr()?.mean_all()?;

    let values = output.flatten_all()?.to_vec1::<f32>()?;
    let scalar = loss.to_scalar::<f32>()?;
    let back = loss.backward()?;
    let grad = |parameter: &Var| -> candle_core::Result<Vec<f32>> {
        back.get(parameter)
            .ok_or_else(|| candle_core::Error::Msg("missing Candle gradient".into()))?
            .flatten_all()?
            .to_vec1::<f32>()
    };
    Ok(json!({
        "backend": "candle-cpu-0.11.0",
        "outputs": values,
        "loss": scalar,
        "gradients": {
            "x": grad(&x)?,
            "conv": grad(&conv)?,
            "depthwise": grad(&dw)?,
            "gamma": grad(&gamma)?,
            "beta": grad(&beta)?
        }
    }))
}

fn burn(f: &Inputs) -> Result<Value, String> {
    use burn::tensor::{
        activation::silu,
        module::{conv2d, interpolate},
        ops::{ConvOptions, InterpolateMode, InterpolateOptions},
        Device, Tensor, TensorData,
    };
    let device = Device::flex().autodiff();
    let variable = |field: &str, shape: [usize; 4]| {
        Tensor::<4>::from_data(TensorData::new(f.f32(field), shape), &device).require_grad()
    };
    let x = variable("x", [1, 2, 4, 4]);
    let conv = variable("conv", [2, 2, 3, 3]);
    let dw = variable("depthwise", [2, 1, 3, 3]);
    let gamma = variable("gamma", [1, 2, 1, 1]);
    let beta = variable("beta", [1, 2, 1, 1]);
    let target = Tensor::<4>::from_data(TensorData::new(f.f32("target"), [1, 2, 4, 4]), &device);

    let y = conv2d(
        x.clone(),
        conv.clone(),
        None,
        ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
    );
    let z = conv2d(
        y,
        dw.clone(),
        None,
        ConvOptions::new([1, 1], [1, 1], [1, 1], 2),
    );
    let mean = z.clone().mean_dim(3).mean_dim(2);
    let centered = z - mean;
    let var = (centered.clone() * centered.clone())
        .mean_dim(3)
        .mean_dim(2);
    let bn = centered / (var + 1e-5).sqrt() * gamma.clone() + beta.clone();
    let output = interpolate(
        silu(bn),
        InterpolateOptions::new(InterpolateMode::Nearest).with_output_size([4, 4]),
    );
    let residual = output.clone() - target;
    let loss = (residual.clone() * residual).mean();

    let values = output
        .to_data()
        .try_to_vec::<f32>()
        .map_err(|e| format!("Burn output conversion: {e:?}"))?;
    let scalar: f32 = loss.clone().into_scalar();
    let gradients = loss.backward();
    let grad = |parameter: &Tensor<4>| -> Result<Vec<f32>, String> {
        parameter
            .grad(&gradients)
            .ok_or_else(|| "missing Burn gradient".to_owned())?
            .to_data()
            .try_to_vec::<f32>()
            .map_err(|e| format!("Burn gradient conversion: {e:?}"))
    };
    Ok(json!({
        "backend": "burn-flex-0.22.0",
        "outputs": values,
        "loss": scalar,
        "gradients": {
            "x": grad(&x)?,
            "conv": grad(&conv)?,
            "depthwise": grad(&dw)?,
            "gamma": grad(&gamma)?,
            "beta": grad(&beta)?
        }
    }))
}

fn run() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let [backend, oracle_path] = args.as_slice() else {
        return Err("usage: td2-proxy <burn|candle> <frozen-oracle.json>".into());
    };
    let payload =
        std::fs::read_to_string(oracle_path).map_err(|e| format!("read frozen reference: {e}"))?;
    let fixture: Oracle =
        serde_json::from_str(&payload).map_err(|e| format!("parse frozen reference: {e}"))?;
    fixture.inputs.verify()?;
    let result = match backend.as_str() {
        "burn" => burn(&fixture.inputs)?,
        "candle" => candle(&fixture.inputs).map_err(|e| e.to_string())?,
        _ => return Err("backend must be burn or candle".into()),
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?
    );
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("td2-proxy: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::Inputs;

    #[test]
    fn reject_finite_f64_inputs_that_overflow_f32() {
        let mut inputs = Inputs {
            x: vec![0.0; 32],
            conv: vec![0.0; 36],
            depthwise: vec![0.0; 18],
            gamma: vec![0.0; 2],
            beta: vec![0.0; 2],
            target: vec![0.0; 32],
        };
        assert!(inputs.verify().is_ok());

        // A JSON number may be finite as f64 but become +inf/-inf in the
        // f32 tensor consumed by both backends. Reject before graph creation.
        inputs.x[0] = f64::MAX;
        assert!(inputs.verify().unwrap_err().contains("f32 overflow"));
        inputs.x[0] = -f64::MAX;
        assert!(inputs.verify().is_err());

        // The largest representable finite f32 must remain accepted.
        inputs.x[0] = f64::from(f32::MAX);
        assert!(inputs.verify().is_ok());
    }
}
