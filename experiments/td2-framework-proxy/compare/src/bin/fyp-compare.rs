// SPDX-License-Identifier: Apache-2.0
//! Stage B measurements. `gates` prints the predeclared gate results; `bench`
//! prints throughput and peak RSS for one framework and batch size. Output is
//! scalar JSON on stdout; nothing is written to disk.
use fyp_compare::gates::{
    forward_error, g3_ratio, g5_ratio, loss_step, reference_gradients, relative_l2, train, BATCH,
    G2_RELATIVE, G4_RELATIVE, G5_RATIO, STEPS,
};
use fyp_compare::{frameworks, Framework};
use fyp_core::data::batch;
use fyp_core::net::{initial_parameters, parameter_bytes, SEED};
use fyp_core::sample_plan::gradient_samples;
use fyp_core::sgd::Sgd;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::time::Instant;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn gates() -> Result<Value, String> {
    let p = initial_parameters();
    let plan = gradient_samples();
    let g3_samples = batch(SEED, 0, BATCH);
    let fd = reference_gradients(&p, &g3_samples, &plan);
    let mut results = Vec::new();
    let mut step0 = Vec::new();
    for fw in frameworks() {
        let fw: &dyn Framework = fw.as_ref();
        let g2: Vec<f64> = (0..4)
            .map(|b| forward_error(fw, &p, &batch(SEED, 1000 + b, BATCH)))
            .collect::<Result<_, _>>()?;
        let (_, g) = loss_step(fw, &p, &g3_samples)?;
        let ratios: Vec<f64> = plan
            .iter()
            .zip(&fd)
            .map(|(i, f)| g3_ratio(f64::from(g[*i]), *f))
            .collect();
        let worst = ratios.iter().cloned().fold(0.0, f64::max);
        step0.push(g);
        let start = Instant::now();
        let first = train(fw, STEPS, BATCH);
        let train_seconds = start.elapsed().as_secs_f64();
        let second = train(fw, STEPS, BATCH);
        let (g5, g6, history) = match (&first, &second) {
            (Ok(a), Ok(b)) => (Some(g5_ratio(a)), a == b, a.clone()),
            _ => (None, false, Vec::new()),
        };
        let g2_max = g2.iter().cloned().fold(0.0, f64::max);
        results.push(json!({
            "framework": fw.name(),
            "g1_all_operations_supported": first.is_ok(),
            "g2_max_relative_error": g2_max,
            "g2_pass": g2_max <= G2_RELATIVE,
            "g3_samples": ratios.len(),
            "g3_worst_ratio": worst,
            "g3_failures": ratios.iter().filter(|r| **r > 1.0).count(),
            "g3_pass": worst <= 1.0,
            "g5_ratio": g5,
            "g5_pass": g5.is_some_and(|r| r <= G5_RATIO),
            "g6_bitwise_identical": g6,
            "train_error": first.err(),
            "train_seconds_300_steps": train_seconds,
            "loss_at": history.iter().enumerate()
                .filter(|(i, _)| [0, 99, 199, 299].contains(i))
                .map(|(i, l)| json!({"step": i + 1, "loss": l})).collect::<Vec<_>>(),
        }));
        eprintln!("{} done", fw.name());
    }
    let g4 = relative_l2(&step0[0], &step0[1]);
    Ok(json!({
        "seed": SEED,
        "batch": BATCH,
        "steps": STEPS,
        "initial_parameters_sha256": hex(&Sha256::digest(parameter_bytes(&p))),
        "candidates": results,
        "g4_relative_l2": g4,
        "g4_pass": g4 <= G4_RELATIVE,
    }))
}

/// Peak resident set (VmHWM) in KiB from /proc; Linux only, read-only.
fn peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

fn bench(name: &str, n: usize) -> Result<Value, String> {
    let all = frameworks();
    let fw = all
        .iter()
        .find(|f| f.name().starts_with(name))
        .ok_or(format!("unknown framework {name}"))?;
    let mut p = initial_parameters();
    let mut opt = Sgd::new(p.len());
    let mut step = 0u64;
    let mut run = |count: usize| -> Result<f64, String> {
        let start = Instant::now();
        for _ in 0..count {
            let samples = batch(SEED, step, n);
            let (_, g) = loss_step(fw.as_ref(), &p, &samples)?;
            opt.step(&mut p, &g);
            step += 1;
        }
        Ok((count * n) as f64 / start.elapsed().as_secs_f64())
    };
    run(10)?;
    let mut windows: Vec<f64> = (0..5).map(|_| run(20)).collect::<Result<_, _>>()?;
    windows.sort_by(f64::total_cmp);
    Ok(json!({
        "framework": fw.name(),
        "batch": n,
        "steps": step,
        "threads": std::env::var("RAYON_NUM_THREADS").ok(),
        "images_per_second_median": windows[2],
        "images_per_second_min": windows[0],
        "images_per_second_max": windows[4],
        "peak_rss_kib": peak_rss_kib(),
    }))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["gates"] => gates(),
        ["bench", name, n] => n
            .parse()
            .map_err(|e| format!("batch: {e}"))
            .and_then(|n| bench(name, n)),
        _ => Err("usage: fyp-compare gates | fyp-compare bench <burn|candle> <batch>".into()),
    };
    match result {
        Ok(v) => println!("{}", serde_json::to_string_pretty(&v).expect("serialise")),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
