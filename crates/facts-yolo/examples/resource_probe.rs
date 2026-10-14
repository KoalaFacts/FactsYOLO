// SPDX-License-Identifier: Apache-2.0
//! Isolated development executable; no process-global allocator in the library.
#[path = "resource_probe/allocator.rs"]
mod allocator;
#[path = "resource_probe/memory.rs"]
mod memory;
#[path = "resource_probe/workload.rs"]
mod workload;
use facts_yolo::fixtures::resource_counts;

#[global_allocator]
static ALLOCATOR: allocator::CountedSystem = allocator::CountedSystem;

fn median(mut values: [u64; 10]) -> u64 {
    values.sort_unstable();
    values[4] + (values[5] - values[4]) / 2
}

fn main() -> Result<(), String> {
    println!(
        "fixture=basic-v1; Rust System allocation bytes exclude allocator metadata and foreign allocations"
    );
    println!(
        "batch\tmodel_handles\tmodel_states\tsessions\tresults\trust_live_bytes\trss_bytes\tprivate_bytes"
    );
    // Prove byte accounting observes actual backing allocations independently of
    // object counters, including reallocations, and returns on destruction.
    let before = allocator::live_bytes();
    let mut probe = vec![0_u8; 4096];
    std::hint::black_box(&probe);
    assert_eq!(allocator::live_bytes(), before + probe.capacity());
    probe.reserve_exact(4096);
    std::hint::black_box(&probe);
    assert_eq!(allocator::live_bytes(), before + probe.capacity());
    drop(probe);
    assert_eq!(allocator::live_bytes(), before);
    for _ in 0..1000 {
        workload::cycle();
    }
    memory::sample()?;
    let baseline = allocator::live_bytes();
    assert_eq!(resource_counts(), [0; 4]);
    let mut samples = [memory::Sample::default(); 100];
    for (batch, sample) in samples.iter_mut().enumerate() {
        for _ in 0..100 {
            workload::cycle();
        }
        let objects = resource_counts();
        let bytes = allocator::live_bytes();
        if objects != [0; 4] || bytes != baseline {
            return Err(format!(
                "resource baseline failed at batch {batch}: objects={objects:?}, bytes={bytes}, baseline={baseline}"
            ));
        }
        *sample = memory::sample()?;
        assert_eq!(
            allocator::live_bytes(),
            baseline,
            "sampling retained Rust allocations"
        );
        println!(
            "{}\t0\t0\t0\t0\t{bytes}\t{}\t{}",
            batch + 1,
            sample.resident,
            sample.private
        );
    }
    for (name, select) in [
        ("rss", (|s: memory::Sample| s.resident) as fn(_) -> _),
        ("private", (|s: memory::Sample| s.private) as fn(_) -> _),
    ] {
        let first = median(std::array::from_fn(|i| select(samples[i])));
        let last = median(std::array::from_fn(|i| select(samples[90 + i])));
        println!("{name}: first_median={first}, last_median={last}, budget=16777216");
        if last.saturating_sub(first) > 16 * 1024 * 1024 {
            return Err(format!("{name} plateau budget exceeded"));
        }
    }
    println!(
        "resources: 1000 warmup, 10000 measured, 100 samples; live objects and Rust allocation bytes returned to baseline"
    );
    Ok(())
}
