// SPDX-License-Identifier: Apache-2.0
use crate::Rng;
#[derive(Clone)]
pub struct Sample {
    pub pixels: Vec<u8>,
    pub class: Option<usize>,
    pub box_xywh: [f32; 4],
}

pub fn dataset(seed: u64, count: usize) -> Vec<Sample> {
    let mut rng = Rng::new(seed);
    (0..count)
        .map(|i| {
            let class = if i % 5 == 0 {
                None
            } else {
                Some((rng.next_u32() % 2) as usize)
            };
            let x = 2 + rng.next_u32() % 40;
            let y = 2 + rng.next_u32() % 40;
            let w = 12 + rng.next_u32() % 11;
            let h = 12 + rng.next_u32() % 11;
            let mut pixels = (0..64 * 64 * 3)
                .map(|_| (rng.next_u32() % 12) as u8)
                .collect::<Vec<_>>();
            if let Some(class) = class {
                let mut color = [0; 3];
                for v in &mut color {
                    *v = (rng.next_u32() % 60) as u8;
                }
                color[class * 2] = 180 + (rng.next_u32() % 76) as u8;
                for py in y..y + h {
                    for px in x..x + w {
                        let index = ((py * 64 + px) * 3) as usize;
                        pixels[index..index + 3].copy_from_slice(&color);
                    }
                }
            }
            Sample {
                pixels,
                class,
                box_xywh: [
                    (x as f32 + w as f32 / 2.0) / 64.0,
                    (y as f32 + h as f32 / 2.0) / 64.0,
                    w as f32 / 64.0,
                    h as f32 / 64.0,
                ],
            }
        })
        .collect()
}

pub fn features(pixels: &[u8]) -> Result<Vec<f32>, String> {
    if pixels.len() != 64 * 64 * 3 {
        return Err("TD1 requires exactly 64x64 RGB bytes".into());
    }
    let mut output = vec![0.0; 192];
    for gy in 0..8 {
        for gx in 0..8 {
            for c in 0..3 {
                let mut sum = 0_u32;
                for y in gy * 8..gy * 8 + 8 {
                    for x in gx * 8..gx * 8 + 8 {
                        sum += u32::from(pixels[(y * 64 + x) * 3 + c]);
                    }
                }
                output[(gy * 8 + gx) * 3 + c] = sum as f32 / (64.0 * 255.0);
            }
        }
    }
    Ok(output)
}
