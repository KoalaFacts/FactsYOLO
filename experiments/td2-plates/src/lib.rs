// SPDX-License-Identifier: Apache-2.0
//! Original synthetic licence-plate renderer for T2 (stage H) training data.
//!
//! Renders a plate carrying caller-chosen text into a cluttered RGB8 scene,
//! with per-character labels (cell quad and axis-aligned box) and the plate
//! quad. Augmentations: perspective, box blur and additive noise.
//!
//! Provenance: the glyphs are an original stroke font defined in this file as
//! polylines on a 4 × 6 grid. No font file, plate photograph, external image
//! or dataset is an input, so no font or data licence applies. Text formats
//! are caller-supplied patterns ([`Format`]); this crate makes no claim that a
//! pattern matches any jurisdiction's registration rules, and real plate
//! layouts (spec section 8, T2 row) still need their own record.
//!
//! Determinism: output depends only on the inputs and the seed. Rendering
//! uses integer arithmetic and IEEE-754 add, subtract, multiply, divide and
//! square root (no trigonometric or other library functions), so the same
//! inputs give identical bytes on every platform.
//!
//! Development tooling only: never linked into the deployed library, no I/O,
//! no dependencies.

use std::fmt;

/// Characters the stroke font can draw, in label order.
pub const GLYPHS: &str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-";

/// Smallest rendered character height in pixels.
const MIN_CHAR_PX: f64 = 16.0;
/// Stroke width as a fraction of the character height.
const STROKE: f64 = 0.14;
/// Gap between character cells, in character heights.
const GAP: f64 = 0.18;
/// Width of a space, in character heights.
const SPACE: f64 = 0.45;
/// Horizontal and vertical plate margins, in character heights.
const MARGIN_X: f64 = 0.35;
const MARGIN_Y: f64 = 0.3;
/// Share of the frame the plate may fill at most.
const FILL: f64 = 0.95;

type Polyline = &'static [(u8, u8)];

const OCTAGON: Polyline = &[
    (1, 0),
    (3, 0),
    (4, 1),
    (4, 5),
    (3, 6),
    (1, 6),
    (0, 5),
    (0, 1),
    (1, 0),
];
const P_BOWL: Polyline = &[(0, 6), (0, 0), (3, 0), (4, 1), (4, 2), (3, 3), (0, 3)];

/// Stroke polylines on a grid with x in 0..=4 (right) and y in 0..=6 (down).
fn strokes(glyph: char) -> Option<&'static [Polyline]> {
    Some(match glyph {
        '0' => &[OCTAGON, &[(4, 1), (0, 5)]],
        '1' => &[&[(0, 2), (2, 0), (2, 6)], &[(1, 6), (3, 6)]],
        '2' => &[&[(0, 1), (1, 0), (3, 0), (4, 1), (4, 2), (0, 6), (4, 6)]],
        '3' => &[&[
            (0, 0),
            (4, 0),
            (2, 3),
            (3, 3),
            (4, 4),
            (4, 5),
            (3, 6),
            (1, 6),
            (0, 5),
        ]],
        '4' => &[&[(3, 6), (3, 0), (0, 4), (4, 4)]],
        '5' => &[&[
            (4, 0),
            (0, 0),
            (0, 3),
            (3, 3),
            (4, 4),
            (4, 5),
            (3, 6),
            (0, 6),
        ]],
        '6' => &[&[
            (3, 0),
            (1, 0),
            (0, 1),
            (0, 5),
            (1, 6),
            (3, 6),
            (4, 5),
            (4, 4),
            (3, 3),
            (0, 3),
        ]],
        '7' => &[&[(0, 0), (4, 0), (1, 6)]],
        '8' => &[
            &[
                (1, 3),
                (0, 2),
                (0, 1),
                (1, 0),
                (3, 0),
                (4, 1),
                (4, 2),
                (3, 3),
                (1, 3),
            ],
            &[
                (1, 3),
                (0, 4),
                (0, 5),
                (1, 6),
                (3, 6),
                (4, 5),
                (4, 4),
                (3, 3),
            ],
        ],
        '9' => &[&[
            (4, 3),
            (1, 3),
            (0, 2),
            (0, 1),
            (1, 0),
            (3, 0),
            (4, 1),
            (4, 5),
            (3, 6),
            (1, 6),
        ]],
        'A' => &[&[(0, 6), (0, 2), (2, 0), (4, 2), (4, 6)], &[(0, 4), (4, 4)]],
        'B' => &[P_BOWL, &[(3, 3), (4, 4), (4, 5), (3, 6), (0, 6)]],
        'C' => &[&[
            (4, 1),
            (3, 0),
            (1, 0),
            (0, 1),
            (0, 5),
            (1, 6),
            (3, 6),
            (4, 5),
        ]],
        'D' => &[&[(0, 0), (2, 0), (4, 2), (4, 4), (2, 6), (0, 6), (0, 0)]],
        'E' => &[&[(4, 0), (0, 0), (0, 6), (4, 6)], &[(0, 3), (3, 3)]],
        'F' => &[&[(4, 0), (0, 0), (0, 6)], &[(0, 3), (3, 3)]],
        'G' => &[&[
            (4, 1),
            (3, 0),
            (1, 0),
            (0, 1),
            (0, 5),
            (1, 6),
            (3, 6),
            (4, 5),
            (4, 3),
            (2, 3),
        ]],
        'H' => &[&[(0, 0), (0, 6)], &[(4, 0), (4, 6)], &[(0, 3), (4, 3)]],
        'I' => &[&[(1, 0), (3, 0)], &[(2, 0), (2, 6)], &[(1, 6), (3, 6)]],
        'J' => &[&[(1, 0), (4, 0)], &[(3, 0), (3, 5), (2, 6), (1, 6), (0, 5)]],
        'K' => &[&[(0, 0), (0, 6)], &[(4, 0), (0, 4)], &[(1, 3), (4, 6)]],
        'L' => &[&[(0, 0), (0, 6), (4, 6)]],
        'M' => &[&[(0, 6), (0, 0), (2, 3), (4, 0), (4, 6)]],
        'N' => &[&[(0, 6), (0, 0), (4, 6), (4, 0)]],
        'O' => &[OCTAGON],
        'P' => &[P_BOWL],
        'Q' => &[OCTAGON, &[(2, 3), (4, 6)]],
        'R' => &[P_BOWL, &[(2, 3), (4, 6)]],
        'S' => &[&[
            (4, 1),
            (3, 0),
            (1, 0),
            (0, 1),
            (0, 2),
            (1, 3),
            (3, 3),
            (4, 4),
            (4, 5),
            (3, 6),
            (1, 6),
            (0, 5),
        ]],
        'T' => &[&[(0, 0), (4, 0)], &[(2, 0), (2, 6)]],
        'U' => &[&[(0, 0), (0, 5), (1, 6), (3, 6), (4, 5), (4, 0)]],
        'V' => &[&[(0, 0), (2, 6), (4, 0)]],
        'W' => &[&[(0, 0), (1, 6), (2, 3), (3, 6), (4, 0)]],
        'X' => &[&[(0, 0), (4, 6)], &[(4, 0), (0, 6)]],
        'Y' => &[&[(0, 0), (2, 3), (4, 0)], &[(2, 3), (2, 6)]],
        'Z' => &[&[(0, 0), (4, 0), (0, 6), (4, 6)]],
        '-' => &[&[(1, 3), (3, 3)]],
        _ => return None,
    })
}

/// Character cell width in character heights.
fn cell_width() -> f64 {
    4.0 * grid_step() + STROKE
}

fn grid_step() -> f64 {
    (1.0 - STROKE) / 6.0
}

/// True when the point (x, y), in character heights inside a cell, lies on a
/// stroke of `glyph`.
fn on_stroke(lines: &[Polyline], x: f64, y: f64) -> bool {
    let step = grid_step();
    let half = STROKE / 2.0;
    let limit = half * half;
    let point = |(gx, gy): (u8, u8)| (f64::from(gx) * step + half, f64::from(gy) * step + half);
    lines.iter().any(|line| {
        line.windows(2).any(|pair| {
            let (ax, ay) = point(pair[0]);
            let (bx, by) = point(pair[1]);
            let (dx, dy) = (bx - ax, by - ay);
            let t = (((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
            let (ex, ey) = (x - (ax + t * dx), y - (ay + t * dy));
            ex * ex + ey * ey <= limit
        })
    })
}

/// Rasterizes one glyph into a `height × (height * 2 / 3)` mask (row-major,
/// pixel centres sampled). `None` for a character outside [`GLYPHS`].
pub fn glyph_mask(glyph: char, height: usize) -> Option<Vec<bool>> {
    let lines = strokes(glyph)?;
    let width = height * 2 / 3;
    let mut mask = Vec::with_capacity(width * height);
    for row in 0..height {
        for column in 0..width {
            let x = (column as f64 + 0.5) / width as f64 * cell_width();
            let y = (row as f64 + 0.5) / height as f64;
            mask.push(on_stroke(lines, x, y));
        }
    }
    Some(mask)
}

/// Why a render or format was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum RenderError {
    /// The text has no drawable character.
    EmptyText,
    /// The text holds a character outside [`GLYPHS`] and space.
    UnknownGlyph(char),
    /// Characters would render below the minimum height. `max` is the
    /// longest text without spaces that fits this scene.
    TextTooLong { glyphs: usize, max: usize },
    /// A scene parameter is out of range.
    InvalidScene(&'static str),
    /// A format pattern or letter pool is invalid.
    InvalidFormat(&'static str),
}

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyText => write!(f, "text has no drawable character"),
            Self::UnknownGlyph(glyph) => write!(f, "character {glyph:?} has no glyph"),
            Self::TextTooLong { glyphs, max } => {
                write!(
                    f,
                    "{glyphs} characters do not fit; at most {max} fit this scene"
                )
            }
            Self::InvalidScene(why) => write!(f, "invalid scene: {why}"),
            Self::InvalidFormat(why) => write!(f, "invalid format: {why}"),
        }
    }
}

impl std::error::Error for RenderError {}

/// Frame and augmentation settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scene {
    /// Frame width in pixels, 16..=4096 and at least `height`.
    pub width: usize,
    /// Frame height in pixels, 16..=4096.
    pub height: usize,
    /// Largest corner displacement as a share of the plate height, 0..=0.3.
    pub max_perspective: f64,
    /// Largest box-blur radius in pixels, 0..=3.
    pub max_blur: u32,
    /// Largest additive noise per channel, in 8-bit levels.
    pub noise_amplitude: u8,
    /// Smallest luma difference between plate and text colours, 0..=0.9,
    /// with luma = 0.2126 R + 0.7152 G + 0.0722 B on encoded 8-bit values / 255.
    pub min_contrast: f64,
}

impl Scene {
    fn validate(&self) -> Result<(), RenderError> {
        let size = 16..=4096;
        if !size.contains(&self.width) || !size.contains(&self.height) {
            return Err(RenderError::InvalidScene(
                "frame sides must be 16..=4096 pixels",
            ));
        }
        if self.width < self.height {
            return Err(RenderError::InvalidScene(
                "frame must be at least as wide as it is tall",
            ));
        }
        if !(0.0..=0.3).contains(&self.max_perspective) {
            return Err(RenderError::InvalidScene(
                "max_perspective must be finite and 0..=0.3",
            ));
        }
        if self.max_blur > 3 {
            return Err(RenderError::InvalidScene("max_blur must be 0..=3"));
        }
        if !(0.0..=0.9).contains(&self.min_contrast) {
            return Err(RenderError::InvalidScene(
                "min_contrast must be finite and 0..=0.9",
            ));
        }
        Ok(())
    }
}

/// One character label in frame pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct CharLabel {
    pub glyph: char,
    /// Character cell corners, clockwise from top-left.
    pub quad: [[f64; 2]; 4],
    /// Axis-aligned bounds of `quad`: left, top, right, bottom.
    pub bbox: [f64; 4],
}

/// A rendered frame and its labels.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub width: usize,
    pub height: usize,
    /// Row-major RGB8 pixels, `width * height * 3` bytes.
    pub rgb: Vec<u8>,
    /// The rendered text, spaces included.
    pub text: String,
    /// Plate corners, clockwise from top-left.
    pub plate_quad: [[f64; 2]; 4],
    /// One label per non-space character, left to right.
    pub chars: Vec<CharLabel>,
    /// Plate and text colours before blur and noise.
    pub plate_rgb: [u8; 3],
    pub text_rgb: [u8; 3],
}

/// SplitMix64 (Steele, Lea and Flood, "Fast splittable pseudorandom number
/// generators", OOPSLA 2014).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

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

    fn range(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.unit()
    }

    /// Uniform in 0..bound (bound > 0), by rejection so it is unbiased.
    fn below(&mut self, bound: u64) -> u64 {
        let zone = u64::MAX - u64::MAX % bound;
        loop {
            let value = self.next_u64();
            if value < zone {
                return value % bound;
            }
        }
    }

    fn colour(&mut self) -> [u8; 3] {
        [
            self.below(256) as u8,
            self.below(256) as u8,
            self.below(256) as u8,
        ]
    }
}

/// A text pattern: `@` a letter from the pool, `#` a digit, `*` a letter or
/// digit, space a gap, and any other [`GLYPHS`] character literally.
#[derive(Clone, Debug, PartialEq)]
pub struct Format {
    pattern: Vec<char>,
    letters: Vec<char>,
}

impl Format {
    /// `letters` is the pool for `@` and `*`: distinct characters from A–Z.
    pub fn new(pattern: &str, letters: &str) -> Result<Self, RenderError> {
        let pattern: Vec<char> = pattern.chars().collect();
        if pattern.iter().all(|&c| c == ' ') {
            return Err(RenderError::InvalidFormat("pattern has no character"));
        }
        if !pattern
            .iter()
            .all(|&c| matches!(c, '@' | '#' | '*' | ' ') || strokes(c).is_some())
        {
            return Err(RenderError::InvalidFormat(
                "pattern holds a character with no meaning",
            ));
        }
        let letters: Vec<char> = letters.chars().collect();
        if letters.is_empty() {
            return Err(RenderError::InvalidFormat("letter pool is empty"));
        }
        if !letters.iter().all(char::is_ascii_uppercase) {
            return Err(RenderError::InvalidFormat("letter pool holds a non-letter"));
        }
        if letters
            .iter()
            .enumerate()
            .any(|(i, c)| letters[..i].contains(c))
        {
            return Err(RenderError::InvalidFormat("letter pool repeats a letter"));
        }
        Ok(Self { pattern, letters })
    }

    /// The text this pattern yields for `seed`.
    pub fn generate(&self, seed: u64) -> String {
        let mut rng = Rng::new(seed ^ 0x5eed_f0f0_7a3e_0001);
        let letters = self.letters.len() as u64;
        self.pattern
            .iter()
            .map(|&slot| match slot {
                '@' => self.letters[rng.below(letters) as usize],
                '#' => char::from(b'0' + rng.below(10) as u8),
                '*' => {
                    let pick = rng.below(letters + 10);
                    if pick < 10 {
                        char::from(b'0' + pick as u8)
                    } else {
                        self.letters[(pick - 10) as usize]
                    }
                }
                literal => literal,
            })
            .collect()
    }
}

fn luma(rgb: [u8; 3]) -> f64 {
    (0.2126 * f64::from(rgb[0]) + 0.7152 * f64::from(rgb[1]) + 0.0722 * f64::from(rgb[2])) / 255.0
}

/// Plate width in character heights for `glyphs` characters and `spaces` gaps.
fn plate_units(glyphs: usize, spaces: usize) -> f64 {
    let slots = glyphs + spaces;
    2.0 * MARGIN_X
        + glyphs as f64 * cell_width()
        + spaces as f64 * SPACE
        + (slots.max(1) - 1) as f64 * GAP
}

/// Largest pixels-per-character-height that keeps the plate and its
/// perspective jitter inside the frame.
fn max_scale(scene: &Scene, plate_width: f64) -> f64 {
    let plate_height = 1.0 + 2.0 * MARGIN_Y;
    let jitter = 2.0 * scene.max_perspective * plate_height;
    let by_width = scene.width as f64 / (plate_width + jitter);
    let by_height = scene.height as f64 / (plate_height + jitter);
    FILL * by_width.min(by_height)
}

/// 3 × 3 homography mapping `from` corners onto `to` corners, solved by
/// Gaussian elimination with partial pivoting (h33 = 1).
fn homography(from: &[[f64; 2]; 4], to: &[[f64; 2]; 4]) -> [f64; 9] {
    let mut rows = [[0.0_f64; 9]; 8];
    for i in 0..4 {
        let [x, y] = from[i];
        let [u, v] = to[i];
        rows[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
        rows[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
    }
    for column in 0..8 {
        let pivot = (column..8)
            .max_by(|&a, &b| rows[a][column].abs().total_cmp(&rows[b][column].abs()))
            .unwrap_or(column);
        rows.swap(column, pivot);
        let lead = rows[column][column];
        for value in rows[column].iter_mut() {
            *value /= lead;
        }
        let pivot_row = rows[column];
        for (index, row) in rows.iter_mut().enumerate() {
            let factor = row[column];
            if index != column && factor != 0.0 {
                for (target, source) in row[column..].iter_mut().zip(&pivot_row[column..]) {
                    *target -= factor * source;
                }
            }
        }
    }
    let mut h = [1.0; 9];
    for (i, row) in rows.iter().enumerate() {
        h[i] = row[8];
    }
    h
}

/// True when every turn of the clockwise (y down) corner sequence is a
/// strict right turn.
fn strictly_convex(quad: &[[f64; 2]; 4]) -> bool {
    (0..4).all(|i| {
        let [a, b, c] = [quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]];
        (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) > 0.0
    })
}

fn apply(h: &[f64; 9], x: f64, y: f64) -> [f64; 2] {
    let w = h[6] * x + h[7] * y + h[8];
    [
        (h[0] * x + h[1] * y + h[2]) / w,
        (h[3] * x + h[4] * y + h[5]) / w,
    ]
}

/// Box blur with radius `radius`, separable, edge-clamped, rounded.
fn blur(rgb: &mut [u8], width: usize, height: usize, radius: usize) {
    if radius == 0 {
        return;
    }
    let taps = (2 * radius + 1) as u32;
    let mut scratch = vec![0_u8; rgb.len()];
    for (stride, count, lines, line_stride) in
        [(3, width, height, width * 3), (width * 3, height, width, 3)]
    {
        for line in 0..lines {
            for position in 0..count {
                for channel in 0..3 {
                    let mut sum = 0_u32;
                    for offset in 0..taps as usize {
                        let p = (position + offset).saturating_sub(radius).min(count - 1);
                        sum += u32::from(rgb[line * line_stride + p * stride + channel]);
                    }
                    scratch[line * line_stride + position * stride + channel] =
                        ((sum + taps / 2) / taps) as u8;
                }
            }
        }
        rgb.copy_from_slice(&scratch);
    }
}

/// Renders `text` (characters from [`GLYPHS`] and spaces) on a plate placed
/// in a random scene. The same arguments always give the same sample.
pub fn render(scene: &Scene, text: &str, seed: u64) -> Result<Sample, RenderError> {
    scene.validate()?;
    if let Some(bad) = text.chars().find(|&c| c != ' ' && strokes(c).is_none()) {
        return Err(RenderError::UnknownGlyph(bad));
    }
    let glyphs = text.chars().filter(|&c| c != ' ').count();
    if glyphs == 0 {
        return Err(RenderError::EmptyText);
    }
    let spaces = text.chars().count() - glyphs;
    let plate_width = plate_units(glyphs, spaces);
    let scale_limit = max_scale(scene, plate_width);
    if scale_limit < MIN_CHAR_PX {
        let max = (1..=glyphs)
            .take_while(|&n| max_scale(scene, plate_units(n, 0)) >= MIN_CHAR_PX)
            .count();
        return Err(RenderError::TextTooLong { glyphs, max });
    }

    let mut rng = Rng::new(seed);
    let (width, height) = (scene.width, scene.height);

    // Colours.
    let (mut plate_rgb, mut text_rgb) = ([255, 255, 255], [0, 0, 0]);
    for _ in 0..64 {
        let (paper, ink) = (rng.colour(), rng.colour());
        if (luma(paper) - luma(ink)).abs() >= scene.min_contrast {
            (plate_rgb, text_rgb) = (paper, ink);
            break;
        }
    }

    // Background: a gradient with rectangles of clutter.
    let (top_colour, bottom_colour) = (rng.colour(), rng.colour());
    let mut background = vec![0_u8; width * height * 3];
    for y in 0..height {
        let t = y as f64 / (height - 1) as f64;
        for x in 0..width {
            for c in 0..3 {
                let value = f64::from(top_colour[c])
                    + (f64::from(bottom_colour[c]) - f64::from(top_colour[c])) * t;
                background[(y * width + x) * 3 + c] = (value + 0.5) as u8;
            }
        }
    }
    for _ in 0..3 + rng.below(6) {
        let colour = rng.colour();
        let x0 = rng.below(width as u64) as usize;
        let y0 = rng.below(height as u64) as usize;
        let x1 = (x0 + 1 + rng.below((width / 3) as u64) as usize).min(width);
        let y1 = (y0 + 1 + rng.below((height / 3) as u64) as usize).min(height);
        for y in y0..y1 {
            for x in x0..x1 {
                background[(y * width + x) * 3..][..3].copy_from_slice(&colour);
            }
        }
    }

    // Plate geometry in pixels.
    let scale = rng.range((0.7 * scale_limit).max(MIN_CHAR_PX), scale_limit);
    let plate_height = 1.0 + 2.0 * MARGIN_Y;
    let (pw, ph) = (plate_width * scale, plate_height * scale);
    let jitter = scene.max_perspective * ph;
    let x0 = rng.range(jitter, width as f64 - pw - jitter);
    let y0 = rng.range(jitter, height as f64 - ph - jitter);
    let base = [[x0, y0], [x0 + pw, y0], [x0 + pw, y0 + ph], [x0, y0 + ph]];
    // A homography maps the plate rectangle continuously only onto a convex
    // quad, so concave jitter is drawn again (the flat plate after 32 tries).
    let mut plate_quad = base;
    if jitter > 0.0 {
        for _ in 0..32 {
            let mut candidate = base;
            for corner in &mut candidate {
                corner[0] += rng.range(-jitter, jitter);
                corner[1] += rng.range(-jitter, jitter);
            }
            if strictly_convex(&candidate) {
                plate_quad = candidate;
                break;
            }
        }
    }
    let plate_rect = [
        [0.0, 0.0],
        [plate_width, 0.0],
        [plate_width, plate_height],
        [0.0, plate_height],
    ];
    let to_frame = homography(&plate_rect, &plate_quad);
    let to_plate = homography(&plate_quad, &plate_rect);

    // Character cells in plate units (character heights).
    let mut cells: Vec<(f64, char)> = Vec::with_capacity(glyphs);
    let mut cursor = MARGIN_X;
    for c in text.chars() {
        if c == ' ' {
            cursor += SPACE + GAP;
        } else {
            cells.push((cursor, c));
            cursor += cell_width() + GAP;
        }
    }
    let chars = cells
        .iter()
        .map(|&(left, glyph)| {
            let right = left + cell_width();
            let quad = [
                apply(&to_frame, left, MARGIN_Y),
                apply(&to_frame, right, MARGIN_Y),
                apply(&to_frame, right, MARGIN_Y + 1.0),
                apply(&to_frame, left, MARGIN_Y + 1.0),
            ];
            let xs = quad.map(|p| p[0]);
            let ys = quad.map(|p| p[1]);
            let bbox = [
                xs.iter().copied().fold(f64::INFINITY, f64::min),
                ys.iter().copied().fold(f64::INFINITY, f64::min),
                xs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                ys.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            ];
            CharLabel { glyph, quad, bbox }
        })
        .collect();

    // Shade each pixel from four subsamples.
    let ink_at = |u: f64, v: f64| -> bool {
        let y = v - MARGIN_Y;
        if !(0.0..=1.0).contains(&y) {
            return false;
        }
        cells.iter().any(|&(left, glyph)| {
            let x = u - left;
            (0.0..=cell_width()).contains(&x)
                && strokes(glyph).is_some_and(|lines| on_stroke(lines, x, y))
        })
    };
    let mut rgb = background.clone();
    let bounds = |axis: usize, size: usize| {
        let low = plate_quad
            .iter()
            .map(|p| p[axis])
            .fold(f64::INFINITY, f64::min);
        let high = plate_quad
            .iter()
            .map(|p| p[axis])
            .fold(f64::NEG_INFINITY, f64::max);
        (
            low.floor().max(0.0) as usize,
            (high.ceil() as usize).min(size),
        )
    };
    let (x_low, x_high) = bounds(0, width);
    let (y_low, y_high) = bounds(1, height);
    for y in y_low..y_high {
        for x in x_low..x_high {
            let at = (y * width + x) * 3;
            let mut sum = [0_u32; 3];
            for (dx, dy) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                let [u, v] = apply(&to_plate, x as f64 + dx, y as f64 + dy);
                let colour =
                    if (0.0..=plate_width).contains(&u) && (0.0..=plate_height).contains(&v) {
                        if ink_at(u, v) { text_rgb } else { plate_rgb }
                    } else {
                        [background[at], background[at + 1], background[at + 2]]
                    };
                for c in 0..3 {
                    sum[c] += u32::from(colour[c]);
                }
            }
            for c in 0..3 {
                rgb[at + c] = ((sum[c] + 2) / 4) as u8;
            }
        }
    }

    // Degradations.
    let radius = rng.below(u64::from(scene.max_blur) + 1) as usize;
    blur(&mut rgb, width, height, radius);
    if scene.noise_amplitude > 0 {
        let amplitude = i32::from(scene.noise_amplitude);
        let spread = (2 * amplitude + 1) as u64;
        for value in &mut rgb {
            let noise = (rng.below(spread) as i32 + rng.below(spread) as i32 - 2 * amplitude) / 2;
            *value = (i32::from(*value) + noise).clamp(0, 255) as u8;
        }
    }

    Ok(Sample {
        width,
        height,
        rgb,
        text: text.to_owned(),
        plate_quad,
        chars,
        plate_rgb,
        text_rgb,
    })
}
