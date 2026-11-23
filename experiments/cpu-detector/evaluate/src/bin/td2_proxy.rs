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

/// The frozen D2b2a optimizer recipe (`experiments/td2-targets/src/recipe.rs`,
/// spec section 6), copied as constants: AdamW with decoupled weight decay on
/// convolution weights only, global L2 gradient clip, warmup then cosine.
mod recipe {
    pub const BASE_LR: f64 = 2e-3;
    pub const MIN_LR_FRACTION: f64 = 0.05;
    pub const MIN_TOTAL_STEPS: usize = 20;
    pub const BETA1: f64 = 0.9;
    pub const BETA2: f64 = 0.999;
    pub const EPSILON: f64 = 1e-8;
    pub const WEIGHT_DECAY: f64 = 0.05;
    pub const GRAD_CLIP_NORM: f64 = 10.0;
    /// Parameters that are convolution weights (decayed); the batch-norm
    /// scale and shift (`gamma`, `beta`) get no decay.
    pub const DECAYED: [bool; 4] = [true, true, false, false];

    /// Learning rate of zero-based step `step` in a run of `total` steps,
    /// the same formula as `td2_targets::recipe::lr_at`.
    pub fn lr_at(step: usize, total: usize) -> f64 {
        let w = (total / 20).max(1);
        if step < w {
            return BASE_LR * (step + 1) as f64 / w as f64;
        }
        let progress = (step - w) as f64 / (total - 1 - w) as f64;
        let floor = BASE_LR * MIN_LR_FRACTION;
        floor + (BASE_LR - floor) * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
    }

    /// Factor applied to every gradient so the global L2 norm is at most
    /// `clip_norm` (the frozen recipe uses `GRAD_CLIP_NORM`); 1 when the norm
    /// is already within it.
    pub fn clip_scale(norm: f64, clip_norm: f64) -> f64 {
        if norm > clip_norm {
            clip_norm / norm
        } else {
            1.0
        }
    }

    /// Bias corrections `1 - beta^t` for one-based step `t`.
    pub fn bias_corrections(t: usize) -> (f64, f64) {
        let t = i32::try_from(t).unwrap_or(i32::MAX);
        (1.0 - BETA1.powi(t), 1.0 - BETA2.powi(t))
    }
}

/// Validated settings for the in-process AdamW recipe run.
#[derive(Clone, Copy, Debug, PartialEq)]
struct AdamW {
    steps: usize,
    /// Global gradient-norm clip threshold. `GRAD_CLIP_NORM` (the frozen
    /// recipe) unless a lower probe value is given to exercise the active
    /// clip path, which the frozen threshold never reaches on this proxy.
    clip_norm: f64,
}

impl AdamW {
    fn parse(steps: &str, clip_norm: Option<&str>) -> Result<Self, String> {
        let steps: usize = steps
            .parse()
            .map_err(|_| "steps must be an integer".to_owned())?;
        if !(recipe::MIN_TOTAL_STEPS..=Sgd::MAX_STEPS).contains(&steps) {
            return Err(format!(
                "AdamW steps must be in {}..={}",
                recipe::MIN_TOTAL_STEPS,
                Sgd::MAX_STEPS
            ));
        }
        let clip_norm = match clip_norm {
            None => recipe::GRAD_CLIP_NORM,
            Some(text) => {
                let value: f64 = text
                    .parse()
                    .map_err(|_| "clip norm must be a number".to_owned())?;
                if !(value.is_finite() && value > 0.0 && value <= recipe::GRAD_CLIP_NORM) {
                    return Err(format!(
                        "clip norm must be in (0, {}]",
                        recipe::GRAD_CLIP_NORM
                    ));
                }
                value
            }
        };
        Ok(Self { steps, clip_norm })
    }
}

fn adamw_report(
    backend: &str,
    adamw: AdamW,
    losses: Vec<f32>,
    finals: [(&str, Vec<f32>); 4],
    gradient_norms: Vec<f64>,
    rates: Vec<f64>,
    seconds: f64,
) -> Value {
    let mut parameters = serde_json::Map::new();
    for (name, values) in finals {
        parameters.insert(name.to_owned(), json!(values));
    }
    json!({
        "backend": backend,
        "mode": "in-process-tensor-adamw-recipe",
        "steps": adamw.steps,
        "grad_clip_norm": adamw.clip_norm,
        "learning_rates": rates,
        "gradient_norms": gradient_norms,
        "losses": losses,
        "final_parameters": parameters,
        "train_loop_seconds": seconds,
        "framework_optimizer_api": false,
        "framework_throughput": Value::Null,
        "peak_rss_bytes": Value::Null
    })
}

/// Runs the frozen AdamW recipe for `adamw.steps` steps on Candle tensors:
/// forward, autodiff, global clip, decoupled decay and the AdamW moments all
/// stay on Candle tensors. Image and target stay frozen.
fn candle_adamw(f: &Inputs, adamw: AdamW) -> candle_core::Result<Value> {
    use candle_core::{Device, Tensor, Var};
    let device = Device::Cpu;
    let x = Tensor::from_slice(&f.f32("x"), (1, 2, 4, 4), &device)?;
    let target = Tensor::from_slice(&f.f32("target"), (1, 2, 4, 4), &device)?;
    let params = [
        Var::from_slice(&f.f32("conv"), (2, 2, 3, 3), &device)?,
        Var::from_slice(&f.f32("depthwise"), (2, 1, 3, 3), &device)?,
        Var::from_slice(&f.f32("gamma"), (1, 2, 1, 1), &device)?,
        Var::from_slice(&f.f32("beta"), (1, 2, 1, 1), &device)?,
    ];
    let mut m: Vec<Tensor> = params
        .iter()
        .map(|p| p.as_tensor().zeros_like())
        .collect::<candle_core::Result<_>>()?;
    let mut v = m.clone();
    let mut losses = Vec::with_capacity(adamw.steps + 1);
    let mut norms = Vec::with_capacity(adamw.steps);
    let mut rates = Vec::with_capacity(adamw.steps);
    let start = std::time::Instant::now();
    for step in 0..=adamw.steps {
        let [conv, dw, gamma, beta] = &params;
        let (_, loss) = candle_forward(&x, conv, dw, gamma, beta, &target)?;
        losses.push(loss.to_scalar::<f32>()?);
        if step == adamw.steps {
            break;
        }
        let grads = loss.backward()?;
        let mut raw = Vec::with_capacity(4);
        let mut sum_sq = 0.0f64;
        for parameter in &params {
            let grad = grads
                .get(parameter)
                .ok_or_else(|| candle_core::Error::Msg("missing Candle gradient".into()))?
                .clone();
            sum_sq += f64::from(grad.sqr()?.sum_all()?.to_scalar::<f32>()?);
            raw.push(grad);
        }
        let norm = sum_sq.sqrt();
        norms.push(norm);
        let scale = recipe::clip_scale(norm, adamw.clip_norm);
        let lr = recipe::lr_at(step, adamw.steps);
        rates.push(lr);
        let (c1, c2) = recipe::bias_corrections(step + 1);
        for (i, parameter) in params.iter().enumerate() {
            let g = raw[i].affine(scale, 0.0)?;
            m[i] = m[i]
                .affine(recipe::BETA1, 0.0)?
                .add(&g.affine(1.0 - recipe::BETA1, 0.0)?)?;
            v[i] = v[i]
                .affine(recipe::BETA2, 0.0)?
                .add(&g.sqr()?.affine(1.0 - recipe::BETA2, 0.0)?)?;
            let mut w = parameter.as_tensor().clone();
            if recipe::DECAYED[i] {
                w = w.affine(1.0 - lr * recipe::WEIGHT_DECAY, 0.0)?;
            }
            let denom = v[i]
                .affine(1.0 / c2, 0.0)?
                .sqrt()?
                .affine(1.0, recipe::EPSILON)?;
            let update = m[i].affine(lr / c1, 0.0)?.div(&denom)?;
            parameter.set(&w.sub(&update)?)?;
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    let flat = |v: &Var| v.as_tensor().flatten_all()?.to_vec1::<f32>();
    Ok(adamw_report(
        "candle-cpu-0.11.0",
        adamw,
        losses,
        [
            ("conv", flat(&params[0])?),
            ("depthwise", flat(&params[1])?),
            ("gamma", flat(&params[2])?),
            ("beta", flat(&params[3])?),
        ],
        norms,
        rates,
        seconds,
    ))
}

/// Runs the frozen AdamW recipe for `adamw.steps` steps on Burn tensors:
/// autodiff on the autodiff backend, clip, decay and moments on the inner
/// backend. Image and target stay frozen.
fn burn_adamw(f: &Inputs, adamw: AdamW) -> Result<Value, String> {
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
    let mut m: Vec<_> = params
        .iter()
        .map(|p| p.clone().inner().zeros_like())
        .collect();
    let mut v = m.clone();
    let mut losses = Vec::with_capacity(adamw.steps + 1);
    let mut norms = Vec::with_capacity(adamw.steps);
    let mut rates = Vec::with_capacity(adamw.steps);
    let start = std::time::Instant::now();
    for step in 0..=adamw.steps {
        let [conv, dw, gamma, beta] = &params;
        let (_, loss) = burn_forward(&x, conv, dw, gamma, beta, &target);
        losses.push(loss.clone().into_scalar::<f32>());
        if step == adamw.steps {
            break;
        }
        let grads = loss.backward();
        let mut raw = Vec::with_capacity(4);
        let mut sum_sq = 0.0f64;
        for parameter in &params {
            let grad = parameter
                .grad(&grads)
                .ok_or_else(|| "missing Burn gradient".to_owned())?;
            sum_sq += f64::from(grad.clone().powi_scalar(2).sum().into_scalar::<f32>());
            raw.push(grad);
        }
        let norm = sum_sq.sqrt();
        norms.push(norm);
        let scale = recipe::clip_scale(norm, adamw.clip_norm) as f32;
        let lr = recipe::lr_at(step, adamw.steps);
        rates.push(lr);
        let (c1, c2) = recipe::bias_corrections(step + 1);
        let mut next = Vec::with_capacity(4);
        for (i, parameter) in params.iter().enumerate() {
            let g = raw[i].clone() * scale;
            m[i] = m[i].clone() * recipe::BETA1 as f32 + g.clone() * (1.0 - recipe::BETA1) as f32;
            v[i] = v[i].clone() * recipe::BETA2 as f32
                + g.powi_scalar(2) * (1.0 - recipe::BETA2) as f32;
            let mut w = parameter.clone().inner();
            if recipe::DECAYED[i] {
                w = w * (1.0 - lr * recipe::WEIGHT_DECAY) as f32;
            }
            let denom = (v[i].clone() * (1.0 / c2) as f32).sqrt() + recipe::EPSILON as f32;
            let update = m[i].clone() * (lr / c1) as f32 / denom;
            next.push(Tensor::from_inner(w - update).require_grad());
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
    Ok(adamw_report(
        "burn-flex-0.22.0",
        adamw,
        losses,
        [
            ("conv", flat(&params[0])?),
            ("depthwise", flat(&params[1])?),
            ("gamma", flat(&params[2])?),
            ("beta", flat(&params[3])?),
        ],
        norms,
        rates,
        seconds,
    ))
}

/// Stage B4: CPU throughput and peak memory of the same graph (3×3 conv,
/// depthwise 3×3, train-mode BN, SiLU, nearest 2× upsample, MSE) at a fixed,
/// predeclared size, trained with in-process tensor SGD on each framework.
/// Inputs, target and initialization come from one SplitMix64 stream, so both
/// backends start from identical values.
mod bench {
    pub const BATCH: usize = 8;
    pub const CHANNELS: usize = 16;
    /// Spatial size after the unpadded 3×3 convolution; the input is S + 2.
    pub const SIZE: usize = 64;
    pub const WARMUP_STEPS: usize = 2;
    pub const LEARNING_RATE: f32 = 0.01;
    pub const MAX_STEPS: usize = 200;
    const SEED: u64 = 0x7464_3262_3462_656e;

    pub fn parse(steps: &str) -> Result<usize, String> {
        let steps: usize = steps
            .parse()
            .map_err(|_| "bench steps must be an integer".to_owned())?;
        if !(1..=MAX_STEPS).contains(&steps) {
            return Err(format!("bench steps must be in 1..={MAX_STEPS}"));
        }
        Ok(steps)
    }

    pub struct Data {
        pub x: Vec<f32>,
        pub target: Vec<f32>,
        pub conv: Vec<f32>,
        pub depthwise: Vec<f32>,
        pub gamma: Vec<f32>,
        pub beta: Vec<f32>,
    }

    pub fn input_shape() -> [usize; 4] {
        [BATCH, CHANNELS, SIZE + 2, SIZE + 2]
    }

    pub fn output_shape() -> [usize; 4] {
        [BATCH, CHANNELS, 2 * SIZE, 2 * SIZE]
    }

    fn uniform(state: &mut u64, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = *state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                z ^= z >> 31;
                let u = (z >> 40) as f32 / (1u64 << 24) as f32;
                (2.0 * u - 1.0) * scale
            })
            .collect()
    }

    pub fn data() -> Data {
        let mut state = SEED;
        let count = |s: [usize; 4]| s.iter().product::<usize>();
        let fan_in = (9 * CHANNELS) as f32;
        Data {
            x: uniform(&mut state, count(input_shape()), 1.0),
            target: uniform(&mut state, count(output_shape()), 0.5),
            conv: uniform(&mut state, CHANNELS * CHANNELS * 9, 1.0 / fan_in.sqrt()),
            depthwise: uniform(&mut state, CHANNELS * 9, 1.0 / 3.0),
            gamma: vec![1.0; CHANNELS],
            beta: vec![0.0; CHANNELS],
        }
    }

    /// Peak resident set size of this process (Linux `VmHWM`), or `None`
    /// where the kernel does not report it.
    pub fn peak_rss_bytes() -> Option<u64> {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
        let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        kib.checked_mul(1024)
    }

    pub fn report(
        backend: &str,
        steps: usize,
        losses: Vec<f32>,
        step_seconds: Vec<f64>,
    ) -> Result<serde_json::Value, String> {
        let (Some(first), Some(last)) = (losses.first(), losses.last()) else {
            return Err("bench recorded no loss".into());
        };
        if losses.iter().any(|l| !l.is_finite()) || last >= first {
            return Err(format!(
                "bench loss must stay finite and fall: {first} -> {last}"
            ));
        }
        let mut sorted = step_seconds.clone();
        sorted.sort_by(f64::total_cmp);
        let measured: f64 = step_seconds.iter().skip(WARMUP_STEPS).sum();
        let timed = step_seconds.len().saturating_sub(WARMUP_STEPS);
        let per_second = if measured > 0.0 {
            timed as f64 / measured
        } else {
            0.0
        };
        Ok(serde_json::json!({
            "backend": backend,
            "mode": "b4-cpu-throughput",
            "shape": {
                "batch": BATCH,
                "channels": CHANNELS,
                "input": input_shape(),
                "output": output_shape()
            },
            "steps": steps,
            "warmup_steps": WARMUP_STEPS,
            "learning_rate": LEARNING_RATE,
            "losses": losses,
            "step_seconds": step_seconds,
            "median_step_seconds": sorted.get(sorted.len() / 2).copied(),
            "steps_per_second": per_second,
            "images_per_second": per_second * BATCH as f64,
            "rayon_num_threads": std::env::var("RAYON_NUM_THREADS").ok(),
            "peak_rss_bytes": peak_rss_bytes(),
            "framework_selected": false
        }))
    }
}

fn candle_bench(steps: usize) -> candle_core::Result<Value> {
    use candle_core::{DType, Device, Tensor, Var};
    let device = Device::Cpu;
    let d = bench::data();
    let c = bench::CHANNELS;
    let x = Tensor::from_slice(&d.x, bench::input_shape().to_vec(), &device)?;
    let target = Tensor::from_slice(&d.target, bench::output_shape().to_vec(), &device)?;
    let conv = Var::from_slice(&d.conv, (c, c, 3, 3), &device)?;
    let dw = Var::from_slice(&d.depthwise, (c, 1, 3, 3), &device)?;
    let gamma = Var::from_slice(&d.gamma, (1, c, 1, 1), &device)?;
    let beta = Var::from_slice(&d.beta, (1, c, 1, 1), &device)?;
    let lr = f64::from(bench::LEARNING_RATE);
    let total = steps + bench::WARMUP_STEPS;
    let mut losses = Vec::with_capacity(total);
    let mut times = Vec::with_capacity(total);
    for _ in 0..total {
        let start = std::time::Instant::now();
        let y = x.conv2d(&conv, 0, 1, 1, 1)?;
        let z = y.conv2d(&dw, 1, 1, 1, c)?;
        let mean = z.mean_keepdim((0, 2, 3))?;
        let centered = z.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim((0, 2, 3))?;
        let bn = centered
            .broadcast_div(&(var + 1e-5)?.sqrt()?)?
            .broadcast_mul(&gamma)?
            .broadcast_add(&beta)?;
        let silu = bn.mul(&candle_nn_free_sigmoid(&bn)?)?;
        let out = silu.upsample_nearest2d(2 * bench::SIZE, 2 * bench::SIZE)?;
        let loss = out.sub(&target)?.sqr()?.mean_all()?;
        let grads = loss.backward()?;
        for parameter in [&conv, &dw, &gamma, &beta] {
            let grad = grads
                .get(parameter)
                .ok_or_else(|| candle_core::Error::Msg("missing Candle gradient".into()))?;
            parameter.set(&parameter.as_tensor().sub(&grad.affine(lr, 0.0)?)?)?;
        }
        losses.push(loss.to_dtype(DType::F32)?.to_scalar::<f32>()?);
        times.push(start.elapsed().as_secs_f64());
    }
    bench::report("candle-cpu-0.11.0", steps, losses, times).map_err(candle_core::Error::Msg)
}

/// Sigmoid from core ops only (as in `candle_forward`), so no `candle-nn`.
fn candle_nn_free_sigmoid(t: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
    ((t * -1.0)?.exp()? + 1.0)?.recip()
}

fn burn_bench(steps: usize) -> Result<Value, String> {
    use burn::tensor::{
        activation::silu,
        module::{conv2d, interpolate},
        ops::{ConvOptions, InterpolateMode, InterpolateOptions},
        Device, Tensor, TensorData,
    };
    let device = Device::flex().autodiff();
    let d = bench::data();
    let c = bench::CHANNELS;
    let x = Tensor::<4>::from_data(TensorData::new(d.x, bench::input_shape()), &device);
    let target = Tensor::<4>::from_data(TensorData::new(d.target, bench::output_shape()), &device);
    let variable = |v: Vec<f32>, shape: [usize; 4]| {
        Tensor::<4>::from_data(TensorData::new(v, shape), &device).require_grad()
    };
    let mut params = [
        variable(d.conv, [c, c, 3, 3]),
        variable(d.depthwise, [c, 1, 3, 3]),
        variable(d.gamma, [1, c, 1, 1]),
        variable(d.beta, [1, c, 1, 1]),
    ];
    let total = steps + bench::WARMUP_STEPS;
    let mut losses = Vec::with_capacity(total);
    let mut times = Vec::with_capacity(total);
    for _ in 0..total {
        let start = std::time::Instant::now();
        let [conv, dw, gamma, beta] = &params;
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
            ConvOptions::new([1, 1], [1, 1], [1, 1], c),
        );
        let mean = z.clone().mean_dim(3).mean_dim(2).mean_dim(0);
        let centered = z - mean;
        let var = (centered.clone() * centered.clone())
            .mean_dim(3)
            .mean_dim(2)
            .mean_dim(0);
        let bn = centered / (var + 1e-5).sqrt() * gamma.clone() + beta.clone();
        let out = interpolate(
            silu(bn),
            InterpolateOptions::new(InterpolateMode::Nearest)
                .with_output_size([2 * bench::SIZE, 2 * bench::SIZE]),
        );
        let residual = out - target.clone();
        let loss = (residual.clone() * residual).mean();
        let grads = loss.backward();
        let mut next = Vec::with_capacity(4);
        for parameter in &params {
            let grad = parameter
                .grad(&grads)
                .ok_or_else(|| "missing Burn gradient".to_owned())?;
            let updated = parameter.clone().inner() - grad * bench::LEARNING_RATE;
            next.push(Tensor::from_inner(updated).require_grad());
        }
        params = next
            .try_into()
            .map_err(|_| "Burn parameter count changed".to_owned())?;
        losses.push(loss.into_scalar::<f32>());
        times.push(start.elapsed().as_secs_f64());
    }
    bench::report("burn-flex-0.22.0", steps, losses, times)
}

fn run() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    enum Mode {
        Gradients,
        Sgd(Sgd),
        AdamW(AdamW),
    }
    if let [backend, flag, steps] = args.as_slice() {
        if flag == "--bench" {
            let steps = bench::parse(steps)?;
            let result = match backend.as_str() {
                "burn" => burn_bench(steps)?,
                "candle" => candle_bench(steps).map_err(|e| e.to_string())?,
                _ => return Err("backend must be burn or candle".into()),
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?
            );
            return Ok(());
        }
    }
    let (backend, oracle_path, mode) = match args.as_slice() {
        [backend, path] => (backend, path, Mode::Gradients),
        [backend, path, flag, steps, lr] if flag == "--sgd" => {
            (backend, path, Mode::Sgd(Sgd::parse(steps, lr)?))
        }
        [backend, path, flag, steps] if flag == "--adamw-recipe" => {
            (backend, path, Mode::AdamW(AdamW::parse(steps, None)?))
        }
        [backend, path, flag, steps, clip] if flag == "--adamw-recipe" => {
            (backend, path, Mode::AdamW(AdamW::parse(steps, Some(clip))?))
        }
        _ => {
            return Err("usage: td2-proxy <burn|candle> <frozen-oracle.json> \
                 [--sgd <steps> <lr> | --adamw-recipe <steps> [<clip-norm>]] | \
                 td2-proxy <burn|candle> --bench <steps>"
                .into())
        }
    };
    let payload =
        std::fs::read_to_string(oracle_path).map_err(|e| format!("read frozen reference: {e}"))?;
    let fixture: Oracle =
        serde_json::from_str(&payload).map_err(|e| format!("parse frozen reference: {e}"))?;
    fixture.inputs.verify()?;
    let result = match (backend.as_str(), mode) {
        ("burn", Mode::Gradients) => burn(&fixture.inputs)?,
        ("candle", Mode::Gradients) => candle(&fixture.inputs).map_err(|e| e.to_string())?,
        ("burn", Mode::Sgd(sgd)) => burn_train(&fixture.inputs, sgd)?,
        ("candle", Mode::Sgd(sgd)) => {
            candle_train(&fixture.inputs, sgd).map_err(|e| e.to_string())?
        }
        ("burn", Mode::AdamW(adamw)) => burn_adamw(&fixture.inputs, adamw)?,
        ("candle", Mode::AdamW(adamw)) => {
            candle_adamw(&fixture.inputs, adamw).map_err(|e| e.to_string())?
        }
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
    use super::{bench, recipe, AdamW, Inputs, Sgd};

    #[test]
    fn adamw_recipe_schedule_matches_d2b2a() {
        assert_eq!(
            AdamW::parse("20", None),
            Ok(AdamW {
                steps: 20,
                clip_norm: 10.0
            })
        );
        assert_eq!(
            AdamW::parse("20", Some("0.5")).map(|a| a.clip_norm),
            Ok(0.5)
        );
        for steps in ["19", "0", "101", "x", "-20"] {
            assert!(AdamW::parse(steps, None).is_err(), "{steps}");
        }
        for clip in ["0", "-1", "10.5", "NaN", "inf", "x"] {
            assert!(AdamW::parse("20", Some(clip)).is_err(), "{clip}");
        }
        // total 20: warmup w = 1, so step 0 and step 1 are both at the peak
        // and the last step is at 5% of it (spec section 6).
        assert_eq!(recipe::lr_at(0, 20), 2e-3);
        assert_eq!(recipe::lr_at(1, 20), 2e-3);
        assert!((recipe::lr_at(19, 20) - 1e-4).abs() < 1e-18);
        // total 40: w = 2, step 0 at half the peak.
        assert_eq!(recipe::lr_at(0, 40), 1e-3);
        assert_eq!(recipe::lr_at(2, 40), 2e-3);
        assert_eq!(recipe::clip_scale(5.0, recipe::GRAD_CLIP_NORM), 1.0);
        assert_eq!(recipe::clip_scale(20.0, recipe::GRAD_CLIP_NORM), 0.5);
        assert_eq!(recipe::clip_scale(0.9, 0.5), 0.5 / 0.9);
        let (c1, c2) = recipe::bias_corrections(1);
        assert!((c1 - 0.1).abs() < 1e-15 && (c2 - 0.001).abs() < 1e-15);
    }

    #[test]
    fn bench_settings_and_data_are_fixed() {
        assert_eq!(bench::parse("20"), Ok(20));
        for steps in ["0", "201", "-1", "x", "2.5"] {
            assert!(bench::parse(steps).is_err(), "{steps}");
        }
        let (a, b) = (bench::data(), bench::data());
        assert_eq!(a.x, b.x);
        assert_eq!(a.conv, b.conv);
        assert_eq!(a.x.len(), 8 * 16 * 66 * 66);
        assert_eq!(a.target.len(), 8 * 16 * 128 * 128);
        assert_eq!(a.conv.len(), 16 * 16 * 9);
        assert!(a.x.iter().all(|v| (-1.0..1.0).contains(v)));
        let bound = 1.0 / 144f32.sqrt();
        assert!(a.conv.iter().all(|v| v.abs() <= bound));
        assert_ne!(a.x[0], a.x[1]);
        let ok = bench::report("t", 1, vec![0.5, 0.4, 0.3], vec![1.0; 3]).unwrap();
        assert_eq!(ok["steps_per_second"], 1.0);
        assert!(bench::report("t", 1, vec![0.5, 0.5, 0.5], vec![1.0; 3]).is_err());
        assert!(bench::report("t", 1, vec![0.5, f32::NAN, 0.3], vec![1.0; 3]).is_err());
        assert!(bench::report("t", 1, vec![], vec![]).is_err());
    }

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
