// SPDX-License-Identifier: Apache-2.0
//! Rectified plate crops: the recognizer's input.

use crate::{RenderError, Sample, apply, homography};

/// A plate resampled to an upright rectangle.
#[derive(Clone, Debug, PartialEq)]
pub struct Crop {
    pub width: usize,
    pub height: usize,
    /// Row-major RGB8 pixels, `width * height * 3` bytes.
    pub rgb: Vec<u8>,
    /// Character cells in crop pixels (left, top, right, bottom), in
    /// `Sample::chars` order.
    pub chars: Vec<[f64; 4]>,
}

/// Resamples the plate of `sample` to `width × height` pixels by inverse
/// perspective mapping and bilinear interpolation (edge-clamped). The plate
/// quad maps onto the whole crop, so text lines become level.
pub fn rectify(sample: &Sample, width: usize, height: usize) -> Result<Crop, RenderError> {
    let sizes = 1..=4096;
    if !sizes.contains(&width) || !sizes.contains(&height) {
        return Err(RenderError::InvalidCrop(
            "crop sides must be 1..=4096 pixels",
        ));
    }
    if sample.width == 0
        || sample.height == 0
        || sample.rgb.len() != sample.width * sample.height * 3
    {
        return Err(RenderError::InvalidCrop(
            "sample pixels do not match its size",
        ));
    }
    let (w, h) = (width as f64, height as f64);
    let rect = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
    let to_frame = homography(&rect, &sample.plate_quad);
    let to_crop = homography(&sample.plate_quad, &rect);

    let texel = |x: isize, y: isize, c: usize| -> f64 {
        let x = x.clamp(0, sample.width as isize - 1) as usize;
        let y = y.clamp(0, sample.height as isize - 1) as usize;
        f64::from(sample.rgb[(y * sample.width + x) * 3 + c])
    };
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let [fx, fy] = apply(&to_frame, x as f64 + 0.5, y as f64 + 0.5);
            let (sx, sy) = (fx - 0.5, fy - 0.5);
            let (x0, y0) = (sx.floor(), sy.floor());
            let (tx, ty) = (sx - x0, sy - y0);
            let (x0, y0) = (x0 as isize, y0 as isize);
            for c in 0..3 {
                let top = texel(x0, y0, c) * (1.0 - tx) + texel(x0 + 1, y0, c) * tx;
                let bottom = texel(x0, y0 + 1, c) * (1.0 - tx) + texel(x0 + 1, y0 + 1, c) * tx;
                rgb.push((top * (1.0 - ty) + bottom * ty + 0.5) as u8);
            }
        }
    }

    let chars = sample
        .chars
        .iter()
        .map(|label| {
            let corners = label.quad.map(|[x, y]| apply(&to_crop, x, y));
            let xs = corners.map(|p| p[0]);
            let ys = corners.map(|p| p[1]);
            [
                xs.iter().copied().fold(f64::INFINITY, f64::min),
                ys.iter().copied().fold(f64::INFINITY, f64::min),
                xs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                ys.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            ]
        })
        .collect();
    Ok(Crop {
        width,
        height,
        rgb,
        chars,
    })
}
