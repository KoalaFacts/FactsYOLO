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

// Shared preprocessing for packed training inputs and checked borrowed native
// image views. Each 8x8 block uses exact integer accumulation before converting
// to f32, preserving TD1's historical channel order and normalization.
pub(crate) fn features_from_rgb(
    mut rgb_at: impl FnMut(usize, usize) -> Option<[u8; 3]>,
) -> Result<[f32; 192], String> {
    let mut output = [0.0; 192];
    for gy in 0..8 {
        for gx in 0..8 {
            let mut sums = [0_u32; 3];
            for y in gy * 8..gy * 8 + 8 {
                for x in gx * 8..gx * 8 + 8 {
                    let rgb = rgb_at(x, y).ok_or("incomplete TD1 RGB view")?;
                    for c in 0..3 {
                        sums[c] += u32::from(rgb[c]);
                    }
                }
            }
            for c in 0..3 {
                output[(gy * 8 + gx) * 3 + c] = sums[c] as f32 / (64.0 * 255.0);
            }
        }
    }
    Ok(output)
}

pub fn features(pixels: &[u8]) -> Result<Vec<f32>, String> {
    if pixels.len() != 64 * 64 * 3 {
        return Err("TD1 requires exactly 64x64 RGB bytes".into());
    }
    Ok(features_from_rgb(|x, y| {
        let at = (y * 64 + x) * 3;
        Some([pixels[at], pixels[at + 1], pixels[at + 2]])
    })?
    .to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borrowed_rgb_features_are_bitwise_identical_and_reject_missing_pixels() {
        let pixels: Vec<u8> = (0..64 * 64 * 3)
            .map(|i| (i.wrapping_mul(31).wrapping_add(17) % 256) as u8)
            .collect();
        let compact = features(&pixels).unwrap();
        let borrowed = features_from_rgb(|x, y| {
            let at = (y * 64 + x) * 3;
            Some([pixels[at], pixels[at + 1], pixels[at + 2]])
        })
        .unwrap();
        assert_eq!(compact, borrowed);
        assert!(features_from_rgb(|x, y| {
            if x == 63 && y == 63 {
                None
            } else {
                Some([0, 0, 0])
            }
        })
        .is_err());
    }
}
