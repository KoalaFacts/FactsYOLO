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

fn candle_forward(
    x: &candle_core::Tensor,
    conv: &candle_core::Tensor,
    dw: &candle_core::Tensor,
    gamma: &candle_core::Tensor,
    beta: &candle_core::Tensor,
    target: &candle_core::Tensor,
) -> candle_core::Result<(candle_core::Tensor, candle_core::Tensor)> {
    use candle_core::{DType, Tensor};
    let device = x.device();
    // Explicit train-mode BN: population variance over four spatial values,
    // never Candle's unbiased variance helper or running inference statistics.
    let y = x.conv2d(conv, 0, 1, 1, 1)?;
    let z = y.conv2d(dw, 1, 1, 1, 2)?;
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
        .broadcast_mul(gamma)?
        .broadcast_add(beta)?;
    let sigmoid_denom = ((&bn * -1.0)?.exp()? + 1.0)?;
    let sigmoid = Tensor::ones((1, 2, 2, 2), DType::F32, device)?.div(&sigmoid_denom)?;
    let output = bn.mul(&sigmoid)?.upsample_nearest2d(4, 4)?;
    let residual = output.sub(target)?;
    let loss = residual.sqr()?.mean_all()?;

    Ok((output, loss))
}

fn candle(f: &Inputs) -> candle_core::Result<Value> {
    use candle_core::{Device, Tensor, Var};

    let device = Device::Cpu;
    let x = Var::from_slice(&f.f32("x"), (1, 2, 4, 4), &device)?;
    let conv = Var::from_slice(&f.f32("conv"), (2, 2, 3, 3), &device)?;
    let dw = Var::from_slice(&f.f32("depthwise"), (2, 1, 3, 3), &device)?;
    let gamma = Var::from_slice(&f.f32("gamma"), (1, 2, 1, 1), &device)?;
    let beta = Var::from_slice(&f.f32("beta"), (1, 2, 1, 1), &device)?;
    let target = Tensor::from_slice(&f.f32("target"), (1, 2, 4, 4), &device)?;

    let (output, loss) = candle_forward(&x, &conv, &dw, &gamma, &beta, &target)?;
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

fn burn_forward(
    x: &burn::tensor::Tensor<4>,
    conv: &burn::tensor::Tensor<4>,
    dw: &burn::tensor::Tensor<4>,
    gamma: &burn::tensor::Tensor<4>,
    beta: &burn::tensor::Tensor<4>,
    target: &burn::tensor::Tensor<4>,
) -> (burn::tensor::Tensor<4>, burn::tensor::Tensor<1>) {
    use burn::tensor::{
        activation::silu,
        module::{conv2d, interpolate},
        ops::{ConvOptions, InterpolateMode, InterpolateOptions},
    };
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
    let residual = output.clone() - target.clone();
    let loss = (residual.clone() * residual).mean();

    (output, loss)
}

fn burn(f: &Inputs) -> Result<Value, String> {
    use burn::tensor::{Device, Tensor, TensorData};
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

    let (output, loss) = burn_forward(&x, &conv, &dw, &gamma, &beta, &target);
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

/// Validated in-process SGD settings: plain SGD, no momentum or weight decay.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Sgd {
    steps: usize,
    learning_rate: f32,
}

impl Sgd {
    const MAX_STEPS: usize = 100;

    fn parse(steps: &str, learning_rate: &str) -> Result<Self, String> {
        let steps: usize = steps
            .parse()
            .map_err(|_| "steps must be an integer".to_owned())?;
        let learning_rate: f32 = learning_rate
            .parse()
            .map_err(|_| "learning rate must be a number".to_owned())?;
        if !(1..=Self::MAX_STEPS).contains(&steps) {
            return Err(format!("steps must be in 1..={}", Self::MAX_STEPS));
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 || learning_rate > 1.0 {
            return Err("learning rate must be finite and in (0, 1]".into());
        }
        Ok(Self {
            steps,
            learning_rate,
        })
    }
}

fn training_report(
    backend: &str,
    sgd: Sgd,
    losses: Vec<f32>,
    finals: [(&str, Vec<f32>); 4],
    seconds: f64,
) -> Value {
    let mut parameters = serde_json::Map::new();
    for (name, values) in finals {
        parameters.insert(name.to_owned(), json!(values));
    }
    json!({
        "backend": backend,
        "mode": "in-process-tensor-sgd",
        "steps": sgd.steps,
        "learning_rate": sgd.learning_rate,
        "losses": losses,
        "final_parameters": parameters,
        "train_loop_seconds": seconds,
        "framework_optimizer_api": false,
        "framework_throughput": Value::Null,
        "peak_rss_bytes": Value::Null
    })
}

/// Runs `sgd.steps` updates entirely on Candle tensors: forward, autodiff,
/// then `w <- w - lr * grad` with Candle ops. Image and target stay frozen.
fn candle_train(f: &Inputs, sgd: Sgd) -> candle_core::Result<Value> {
    use candle_core::{Device, Tensor, Var};
    let device = Device::Cpu;
    let x = Tensor::from_slice(&f.f32("x"), (1, 2, 4, 4), &device)?;
    let target = Tensor::from_slice(&f.f32("target"), (1, 2, 4, 4), &device)?;
    let conv = Var::from_slice(&f.f32("conv"), (2, 2, 3, 3), &device)?;
    let dw = Var::from_slice(&f.f32("depthwise"), (2, 1, 3, 3), &device)?;
    let gamma = Var::from_slice(&f.f32("gamma"), (1, 2, 1, 1), &device)?;
    let beta = Var::from_slice(&f.f32("beta"), (1, 2, 1, 1), &device)?;
    let lr = f64::from(sgd.learning_rate);
    let mut losses = Vec::with_capacity(sgd.steps + 1);
    let start = std::time::Instant::now();
    for step in 0..=sgd.steps {
        let (_, loss) = candle_forward(&x, &conv, &dw, &gamma, &beta, &target)?;
        losses.push(loss.to_scalar::<f32>()?);
        if step == sgd.steps {
            break;
        }
        let grads = loss.backward()?;
        for parameter in [&conv, &dw, &gamma, &beta] {
            let grad = grads
                .get(parameter)
                .ok_or_else(|| candle_core::Error::Msg("missing Candle gradient".into()))?;
            parameter.set(&parameter.as_tensor().sub(&grad.affine(lr, 0.0)?)?)?;
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    let flat = |v: &Var| v.as_tensor().flatten_all()?.to_vec1::<f32>();
    Ok(training_report(
        "candle-cpu-0.11.0",
        sgd,
        losses,
        [
            ("conv", flat(&conv)?),
            ("depthwise", flat(&dw)?),
            ("gamma", flat(&gamma)?),
            ("beta", flat(&beta)?),
        ],
        seconds,
    ))
}

/// Runs `sgd.steps` updates entirely on Burn tensors: forward, autodiff,
/// then `w <- w - lr * grad` on the inner backend. Image and target stay frozen.
fn burn_train(f: &Inputs, sgd: Sgd) -> Result<Value, String> {
    use burn::tensor::{Device, Tensor, TensorData};
    let device = Device::flex().autodiff();
    let constant =
        |field: &str| Tensor::<4>::from_data(TensorData::new(f.f32(field), [1, 2, 4, 4]), &device);
    let variable = |field: &str, shape: [usize; 4]| {
        Tensor::<4>::from_data(TensorData::new(f.f32(field), shape), &device).require_grad()
    };
    let x = constant("x");
    let target = constant("target");
    let mut params = [
        variable("conv", [2, 2, 3, 3]),
        variable("depthwise", [2, 1, 3, 3]),
        variable("gamma", [1, 2, 1, 1]),
        variable("beta", [1, 2, 1, 1]),
    ];
    let mut losses = Vec::with_capacity(sgd.steps + 1);
    let start = std::time::Instant::now();
    for step in 0..=sgd.steps {
        let [conv, dw, gamma, beta] = &params;
        let (_, loss) = burn_forward(&x, conv, dw, gamma, beta, &target);
        losses.push(loss.clone().into_scalar::<f32>());
        if step == sgd.steps {
            break;
        }
        let grads = loss.backward();
        let mut next = Vec::with_capacity(4);
        for parameter in &params {
            let grad = parameter
                .grad(&grads)
                .ok_or_else(|| "missing Burn gradient".to_owned())?;
            let updated = parameter.clone().inner() - grad * sgd.learning_rate;
            next.push(Tensor::from_inner(updated).require_grad());
        }
        params = next
            .try_into()
            .map_err(|_| "Burn parameter count changed".to_owned())?;
    }
    let seconds = start.elapsed().as_secs_f64();
    let flat = |t: &Tensor<4>| -> Result<Vec<f32>, String> {
        t.to_data()
            .try_to_vec::<f32>()
            .map_err(|e| format!("Burn parameter conversion: {e:?}"))
    };
    Ok(training_report(
        "burn-flex-0.22.0",
        sgd,
        losses,
        [
            ("conv", flat(&params[0])?),
            ("depthwise", flat(&params[1])?),
            ("gamma", flat(&params[2])?),
            ("beta", flat(&params[3])?),
        ],
        seconds,
    ))
}

fn run() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let (backend, oracle_path, sgd) = match args.as_slice() {
        [backend, path] => (backend, path, None),
        [backend, path, flag, steps, lr] if flag == "--sgd" => {
            (backend, path, Some(Sgd::parse(steps, lr)?))
        }
        _ => {
            return Err(
                "usage: td2-proxy <burn|candle> <frozen-oracle.json> [--sgd <steps> <lr>]".into(),
            )
        }
    };
    let payload =
        std::fs::read_to_string(oracle_path).map_err(|e| format!("read frozen reference: {e}"))?;
    let fixture: Oracle =
        serde_json::from_str(&payload).map_err(|e| format!("parse frozen reference: {e}"))?;
    fixture.inputs.verify()?;
    let result = match (backend.as_str(), sgd) {
        ("burn", None) => burn(&fixture.inputs)?,
        ("candle", None) => candle(&fixture.inputs).map_err(|e| e.to_string())?,
        ("burn", Some(sgd)) => burn_train(&fixture.inputs, sgd)?,
        ("candle", Some(sgd)) => candle_train(&fixture.inputs, sgd).map_err(|e| e.to_string())?,
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
    use super::{Inputs, Sgd};

    #[test]
    fn sgd_settings_are_bounded() {
        assert_eq!(
            Sgd::parse("20", "0.002"),
            Ok(Sgd {
                steps: 20,
                learning_rate: 0.002
            })
        );
        for (steps, lr) in [
            ("0", "0.002"),
            ("101", "0.002"),
            ("-1", "0.002"),
            ("2.5", "0.002"),
            ("20", "0"),
            ("20", "-0.1"),
            ("20", "1.5"),
            ("20", "NaN"),
            ("20", "inf"),
            ("20", "x"),
        ] {
            assert!(Sgd::parse(steps, lr).is_err(), "{steps} {lr}");
        }
        assert!(Sgd::parse("100", "1").is_ok());
    }

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
