// SPDX-License-Identifier: Apache-2.0
//! Frozen TD2 flip, photometric (stage D2b2b1) and scale-jitter (D2b2b2)
//! augmentation reference.
//!
//! Fixed before any training so augmentation is not tuned against the
//! holdout (design section 6). Blur and JPEG-style compression follow in
//! D2b2b3. Ideas only: random horizontal flip and colour jitter as
//! widely used since [AlexNet](https://papers.nips.cc/paper/4824) and
//! [SSD](https://arxiv.org/abs/1512.02325) (also its random crop and zoom-out
//! for scale jitter); BT.601 luma weights. No code was
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
    let pixels = check_input(image, boxes)?;
    check_params(params)?;
    let w = image.width as f64;
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

fn check_input(image: &Image, boxes: &[GtBox]) -> Result<usize, AugmentError> {
    let pixels = image
        .width
        .checked_mul(image.height)
        .filter(|&n| n > 0)
        .ok_or(AugmentError::Shape)?;
    if pixels.checked_mul(3) != Some(image.rgb.len()) {
        return Err(AugmentError::Shape);
    }
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

    Ok(pixels)
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

/// Probability of a scale jitter.
pub const SCALE_PROBABILITY: f64 = 0.5;
/// Content scale factor range, [lo, hi].
pub const SCALE: (f64, f64) = (0.5, 1.5);
/// Fill value for canvas pixels not covered by the scaled content.
pub const PAD_RGB: [u8; 3] = [0, 0, 0];
/// A box survives a scale jitter only if its clipped width and height are at
/// least this many pixels ...
pub const MIN_BOX_SIDE: f64 = 2.0;
/// ... and its clipped area is at least this fraction of its scaled area.
pub const MIN_VISIBLE_FRACTION: f64 = 0.25;

/// Parameters of one scale jitter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScaleParams {
    pub scale: f64,
    /// Placement draws in [0, 1]; see [`apply_scale`].
    pub shift_x: f64,
    pub shift_y: f64,
}

/// Draws the scale jitter of one image from the same `seed` as [`sample`]:
/// draws 8 to 11 of its SplitMix64 stream (apply, scale, shift x, shift y),
/// so the photometric parameters are unchanged. `None` means no jitter.
pub fn sample_scale(seed: u64) -> Option<ScaleParams> {
    let mut rng = Rng(seed);
    for _ in 0..7 {
        rng.next_u64();
    }
    let on = rng.unit() < SCALE_PROBABILITY;
    let scale = rng.range(SCALE);
    let shift_x = rng.unit();
    let shift_y = rng.unit();
    on.then_some(ScaleParams {
        scale,
        shift_x,
        shift_y,
    })
}

/// Rescales the image content by `params.scale` on a canvas of the original
/// size, then crops or pads it.
///
/// Content size `cw = max(1, round(W · scale))` (same for height). With
/// `d = |W − cw|` and `k = min(d, ⌊shift_x · (d + 1)⌋)`, the content's left
/// edge is at `−k` when it is at least as wide as the canvas (crop) and at
/// `+k` otherwise (pad with [`PAD_RGB`]). Pixels are sampled bilinearly with
/// half-pixel centres (`src = (x + 0.5) · W / cw − 0.5`, clamped to the image)
/// and rounded half away from zero. Box edges map as
/// `x · cw / W + offset`, are clipped to the canvas, and a box is dropped when
/// its clipped width or height is below [`MIN_BOX_SIDE`] or its clipped area
/// is below [`MIN_VISIBLE_FRACTION`] of its scaled area. Returned boxes keep
/// their input order.
pub fn apply_scale(
    image: &Image,
    boxes: &[GtBox],
    params: &ScaleParams,
) -> Result<(Image, Vec<GtBox>), AugmentError> {
    check_input(image, boxes)?;
    let unit_ok = |u: f64| (0.0..=1.0).contains(&u);
    if !((SCALE.0..=SCALE.1).contains(&params.scale)
        && unit_ok(params.shift_x)
        && unit_ok(params.shift_y))
    {
        return Err(AugmentError::Param);
    }
    let (w, h) = (image.width, image.height);
    let content = |n: usize| ((n as f64 * params.scale).round() as usize).max(1);
    let (cw, ch) = (content(w), content(h));
    let offset = |canvas: usize, size: usize, u: f64| -> f64 {
        let d = canvas.abs_diff(size);
        let k = ((u * (d + 1) as f64).floor() as usize).min(d) as f64;
        if size >= canvas { -k } else { k }
    };
    let (ox, oy) = (offset(w, cw, params.shift_x), offset(h, ch, params.shift_y));
    let (fx, fy) = (w as f64 / cw as f64, h as f64 / ch as f64);

    let at = |x: usize, y: usize, c: usize| f64::from(image.rgb[(y * w + x) * 3 + c]);
    let mut rgb = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let (cx, cy) = (x as f64 - ox, y as f64 - oy);
            if cx < 0.0 || cy < 0.0 || cx >= cw as f64 || cy >= ch as f64 {
                rgb.extend_from_slice(&PAD_RGB);
                continue;
            }
            let sx = ((cx + 0.5) * fx - 0.5).clamp(0.0, (w - 1) as f64);
            let sy = ((cy + 0.5) * fy - 0.5).clamp(0.0, (h - 1) as f64);
            let (x0, y0) = (sx.floor() as usize, sy.floor() as usize);
            let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
            let (ax, ay) = (sx - x0 as f64, sy - y0 as f64);
            for c in 0..3 {
                let top = at(x0, y0, c) * (1.0 - ax) + at(x1, y0, c) * ax;
                let bottom = at(x0, y1, c) * (1.0 - ax) + at(x1, y1, c) * ax;
                rgb.push(clamp(top * (1.0 - ay) + bottom * ay).round() as u8);
            }
        }
    }

    let (sw, sh) = (cw as f64 / w as f64, ch as f64 / h as f64);
    let (canvas_w, canvas_h) = (w as f64, h as f64);
    let kept = boxes
        .iter()
        .filter_map(|b| {
            let (l, r) = (b.left * sw + ox, b.right * sw + ox);
            let (t, bt) = (b.top * sh + oy, b.bottom * sh + oy);
            let scaled_area = (r - l) * (bt - t);
            let (cl, cr) = (l.clamp(0.0, canvas_w), r.clamp(0.0, canvas_w));
            let (ct, cb) = (t.clamp(0.0, canvas_h), bt.clamp(0.0, canvas_h));
            let visible = cr - cl >= MIN_BOX_SIDE
                && cb - ct >= MIN_BOX_SIDE
                && (cr - cl) * (cb - ct) >= MIN_VISIBLE_FRACTION * scaled_area;
            visible.then_some(GtBox {
                left: cl,
                top: ct,
                right: cr,
                bottom: cb,
                class: b.class,
            })
        })
        .collect();
    Ok((
        Image {
            width: w,
            height: h,
            rgb,
        },
        kept,
    ))
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
