// SPDX-License-Identifier: Apache-2.0
#[derive(Clone, Copy, Default)]
pub struct Sample {
    pub resident: u64,
    pub private: u64,
}

#[cfg(target_os = "linux")]
pub fn sample() -> Result<Sample, String> {
    // smaps_rollup provides current RSS/private bytes, not cumulative high-water
    // marks. Sampling allocations are dropped before returning to the caller.
    let text = std::fs::read_to_string("/proc/self/smaps_rollup")
        .map_err(|error| format!("required current-memory sample unavailable: {error}"))?;
    let field = |key: &str| -> Result<u64, String> {
        let line = text
            .lines()
            .find(|line| line.starts_with(key))
            .ok_or_else(|| format!("missing smaps_rollup field {key}"))?;
        let mut words = line.split_whitespace();
        words.next();
        let value: u64 = words
            .next()
            .ok_or("missing memory value")?
            .parse()
            .map_err(|_| "invalid memory value")?;
        if words.next() != Some("kB") || words.next().is_some() {
            return Err("unexpected memory sample units".to_owned());
        }
        value
            .checked_mul(1024)
            .ok_or_else(|| "memory sample overflow".to_owned())
    };
    let huge = if text
        .lines()
        .any(|line| line.starts_with("Private_Hugetlb:"))
    {
        field("Private_Hugetlb:")?
    } else {
        0
    };
    let private = field("Private_Clean:")?
        .checked_add(field("Private_Dirty:")?)
        .and_then(|value| value.checked_add(huge))
        .ok_or("private memory sample overflow")?;
    Ok(Sample {
        resident: field("Rss:")?,
        private,
    })
}

#[cfg(windows)]
pub fn sample() -> Result<Sample, String> {
    use std::ffi::c_void;
    #[repr(C)]
    #[derive(Default)]
    struct Counters {
        size: u32,
        faults: u32,
        peak_working: usize,
        working: usize,
        peak_paged: usize,
        paged: usize,
        peak_nonpaged: usize,
        nonpaged: usize,
        pagefile: usize,
        peak_pagefile: usize,
        private: usize,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn K32GetProcessMemoryInfo(process: *mut c_void, counters: *mut Counters, size: u32)
        -> i32;
    }
    let size = size_of::<Counters>() as u32;
    let mut counters = Counters {
        size,
        ..Default::default()
    };
    // SAFETY: the pseudo handle represents this process; initialized writable
    // PROCESS_MEMORY_COUNTERS_EX storage has the advertised natural C layout.
    if unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, size) } == 0 {
        return Err(format!(
            "required current-memory sample unavailable: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(Sample {
        resident: counters.working as u64,
        private: counters.private as u64,
    })
}

#[cfg(not(any(target_os = "linux", windows)))]
pub fn sample() -> Result<Sample, String> {
    Err("resource samples require Linux or Windows".to_owned())
}
