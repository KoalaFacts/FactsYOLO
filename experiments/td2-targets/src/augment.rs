// SPDX-License-Identifier: Apache-2.0
//! Frozen TD2 flip and photometric augmentation reference (stage D2b2b1).
//!
//! Fixed before any training so augmentation is not tuned against the
//! holdout (design section 6). Scale jitter, blur and JPEG-style compression
//! follow in D2b2b2. Ideas only: random horizontal flip and colour jitter as
//! widely used since [AlexNet](https://papers.nips.cc/paper/4824) and
//! [SSD](https://arxiv.org/abs/1512.02325); BT.601 luma weights. No code was
//! ported.
//!
//! One augmentation is drawn per image from a 64-bit seed with SplitMix64
//! (Steele, Lea and Flood, OOPSLA 2014), in this fixed draw order: flip,
//! low-light, low-light gain, brightness, contrast, saturation, grayscale.
//! Every draw is made even when its operation is off, so one seed always
//! yields the same parameters.
//! A probability test is `u < p` and a range value is `lo + (hi − lo) · u`,
//! with `u = (next >> 11) / 2^53` and every operation in F64 (so `hi − lo`
//! is rounded before the multiply).
//!
//! Pixels are RGB8, row-major, three bytes per pixel. Operations run in the
//! order flip, low-light gain, brightness, contrast, saturation on
//! F64 channel values (grayscale replaces the saturation step with factor 0,
//! it is not applied after it); each operation clamps to [0, 255] and the final value
//! is rounded half away from zero to u8.

use crate::GtBox;

/// Probability of a horizontal flip.
pub const FLIP_PROBABILITY: f64 = 0.5;
/// Probability of a low-light gain (simulated night or under-exposure).
pub const LOW_LIGHT_PROBABILITY: f64 = 0.2;
/// Multiplicative low-light gain range, [lo, hi].
pub const LOW_LIGHT_GAIN: (f64, f64) = (0.15, 0.5);
/// Multiplicative brightness factor range, [lo, hi].
pub const BRIGHTNESS: (f64, f64) = (0.75, 1.25);
/// Contrast factor range around the image's mean luma, [lo, hi].
pub const CONTRAST: (f64, f64) = (0.75, 1.25);
/// Saturation factor range (0 is gray, 1 is unchanged), [lo, hi].
pub const SATURATION: (f64, f64) = (0.6, 1.4);
/// Probability of converting to grayscale.
pub const GRAYSCALE_PROBABILITY: f64 = 0.1;
/// BT.601 luma weights for R, G, B.
pub const LUMA: [f64; 3] = [0.299, 0.587, 0.114];

/// Parameters of one augmentation, drawn by [`sample`] or set by hand.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    pub flip: bool,
    /// `None` when the low-light operation is off.
    pub low_light_gain: Option<f64>,
    pub brightness: f64,
    pub contrast: f64,
    pub saturation: f64,
    pub grayscale: bool,
}

impl Params {
    /// Parameters that leave the image and boxes unchanged.
    pub const IDENTITY: Self = Self {
        flip: false,
        low_light_gain: None,
        brightness: 1.0,
        contrast: 1.0,
        saturation: 1.0,
        grayscale: false,
    };
}

/// An RGB8 image, row-major, `width * height * 3` bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum AugmentError {
    /// Zero width or height, or a pixel buffer that is not `width * height * 3`.
    Shape,
    /// A parameter is nonfinite or outside its frozen range.
    Param,
    /// A box is nonfinite, inverted or not inside the image.
    Box { index: usize },
}

/// Draws the augmentation of one image from `seed`.
pub fn sample(seed: u64) -> Params {
    let mut rng = Rng(seed);
    let flip = rng.unit() < FLIP_PROBABILITY;
    let low_light = rng.unit() < LOW_LIGHT_PROBABILITY;
    let gain = rng.range(LOW_LIGHT_GAIN);
    let brightness = rng.range(BRIGHTNESS);
    let contrast = rng.range(CONTRAST);
    let saturation = rng.range(SATURATION);
    let grayscale = rng.unit() < GRAYSCALE_PROBABILITY;
    Params {
        flip,
        low_light_gain: low_light.then_some(gain),
        brightness,
        contrast,
        saturation,
        grayscale,
    }
}

/// Applies `params` to `image` and its boxes (pixel coordinates, edges in
/// `[0, width] × [0, height]`). Photometric operations leave boxes unchanged;
/// a flip maps `[l, r]` to `[width − r, width − l]`.
pub fn apply(
    image: &Image,
    boxes: &[GtBox],
    params: &Params,
) -> Result<(Image, Vec<GtBox>), AugmentError> {
    let pixels = image
        .width
        .checked_mul(image.height)
        .filter(|&n| n > 0)
        .ok_or(AugmentError::Shape)?;
    if pixels.checked_mul(3) != Some(image.rgb.len()) {
        return Err(AugmentError::Shape);
    }
    check_params(params)?;
    let (w, h) = (image.width as f64, image.height as f64);
    for (index, b) in boxes.iter().enumerate() {
        let ok = [b.left, b.top, b.right, b.bottom]
            .iter()
            .all(|v| v.is_finite())
            && 0.0 <= b.left
            && b.left < b.right
            && b.right <= w
            && 0.0 <= b.top
            && b.top < b.bottom
            && b.bottom <= h;
        if !ok {
            return Err(AugmentError::Box { index });
        }
    }

    let mut px: Vec<[f64; 3]> = image
        .rgb
        .chunks_exact(3)
        .map(|c| [f64::from(c[0]), f64::from(c[1]), f64::from(c[2])])
        .collect();
    let mut out_boxes = boxes.to_vec();
    if params.flip {
        for row in px.chunks_exact_mut(image.width) {
            row.reverse();
        }
        for b in &mut out_boxes {
            (b.left, b.right) = (w - b.right, w - b.left);
        }
    }
    if let Some(gain) = params.low_light_gain {
        scale(&mut px, gain);
    }
    scale(&mut px, params.brightness);
    // Contrast pivots on the mean luma of the image after brightness.
    let mean = px.iter().map(luma).sum::<f64>() / pixels as f64;
    for p in &mut px {
        for c in p.iter_mut() {
            *c = clamp(mean + (*c - mean) * params.contrast);
        }
    }
    for p in &mut px {
        let y = luma(p);
        let s = if params.grayscale {
            0.0
        } else {
            params.saturation
        };
        for c in p.iter_mut() {
            *c = clamp(y + (*c - y) * s);
        }
    }
    let rgb = px.iter().flat_map(|p| p.map(|c| c.round() as u8)).collect();
    Ok((
        Image {
            width: image.width,
            height: image.height,
            rgb,
        },
        out_boxes,
    ))
}

fn check_params(p: &Params) -> Result<(), AugmentError> {
    // Ranges are closed: `lo + (hi − lo) · u` can round up to `hi`. Every
    // range except the gain contains the identity value 1.0. NaN fails.
    let in_range = |v: f64, (lo, hi): (f64, f64)| lo <= v && v <= hi;
    if p.low_light_gain.is_none_or(|g| in_range(g, LOW_LIGHT_GAIN))
        && in_range(p.brightness, BRIGHTNESS)
        && in_range(p.contrast, CONTRAST)
        && in_range(p.saturation, SATURATION)
    {
        Ok(())
    } else {
        Err(AugmentError::Param)
    }
}

fn luma(p: &[f64; 3]) -> f64 {
    LUMA[0] * p[0] + LUMA[1] * p[1] + LUMA[2] * p[2]
}

fn clamp(v: f64) -> f64 {
    v.clamp(0.0, 255.0)
}

fn scale(px: &mut [[f64; 3]], factor: f64) {
    for p in px {
        for c in p.iter_mut() {
            *c = clamp(*c * factor);
        }
    }
}

/// SplitMix64 (Steele, Lea and Flood, "Fast splittable pseudorandom number
/// generators", OOPSLA 2014).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1) with 53 bits.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn range(&mut self, (lo, hi): (f64, f64)) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}
