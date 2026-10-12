// SPDX-License-Identifier: Apache-2.0
//! Checked views of top-down packed, sRGB-encoded host pixels.
//!
//! No allocation, pixel copy, decoding, resizing or color-profile conversion.
//! RGB/BGR are three bytes per pixel; RGBA is straight alpha, ignored on access.
use crate::{
    Error,
    error::INVALID,
    ffi::{descriptors, types::*},
};

/// Explicit positive limits. Zero is invalid, never an unlimited sentinel.
#[derive(Clone, Copy, Debug)]
pub struct ImageLimits {
    pub max_bytes: usize,
    pub max_pixels: usize,
}

const LIMIT: Error = Error {
    status: FY_LIMIT_EXCEEDED,
    message: "image resource limit exceeded",
};

/// Immutable borrowed storage and checked layout. No ownership is transferred.
///
/// Safe construction cannot outlive its input allocation:
///
/// ```compile_fail
/// use facts_yolo::image::{ImageLimits, ImageView};
/// let view;
/// {
///     let pixels = vec![1_u8, 2, 3];
///     view = ImageView::from_bytes(
///         &pixels, 1, 1, 3, 1, ImageLimits { max_bytes: 3, max_pixels: 1 }
///     ).unwrap();
/// }
/// let _ = view.rgb_at(0, 0);
/// ```
#[derive(Debug)]
pub struct ImageView<'a> {
    data: &'a [u8],
    width: u32,
    height: u32,
    row_stride: usize,
    channels: usize,
    pixel_format: u32,
}

impl<'a> ImageView<'a> {
    /// Borrow initialized bytes, including any inter-row padding.
    ///
    /// The whole supplied slice counts against max_bytes. Only the necessary
    /// prefix is retained; unused final-row padding is not required.
    pub fn from_bytes(
        data: &'a [u8],
        width: u32,
        height: u32,
        row_stride: usize,
        pixel_format: u32,
        limits: ImageLimits,
    ) -> Result<Self, Error> {
        let layout = Layout::checked(data.len(), width, height, row_stride, pixel_format, limits)?;
        Ok(Self {
            data: &data[..layout.required_bytes],
            width,
            height,
            row_stride,
            channels: layout.channels,
            pixel_format,
        })
    }

    /// Borrow foreign pixels after checking the initialized descriptor.
    ///
    /// # Safety
    /// `raw` itself is a complete, initialized Rust record, not a cast reference
    /// to a short foreign prefix. A C entry point must validate its raw prefix
    /// before forming that reference. For accepted metadata, `data` must cover
    /// the declared range in one live allocation; every byte of the retained
    /// prefix (including row padding) must be initialized and immutable for 'a.
    /// Neither the host nor another thread may free, detach or mutate that
    /// storage while the view or any slice borrowed from it exists. The borrow
    /// of `raw` alone does not prove the pixel allocation lives long enough.
    /// Null pointers and invalid numeric metadata are rejected without accessing
    /// pixels. These checks cannot detect dangling pointers or false lengths.
    pub unsafe fn from_ffi(raw: &'a FyImage, limits: ImageLimits) -> Result<Self, Error> {
        // SAFETY: the reference supplies the full initialized descriptor.
        unsafe { descriptors::validate(raw)? };
        let layout = Layout::checked(
            raw.byte_length,
            raw.width,
            raw.height,
            raw.row_stride,
            raw.pixel_format,
            limits,
        )?;
        if raw.data.is_null() || (raw.data as usize).checked_add(raw.byte_length).is_none() {
            return Err(INVALID);
        }
        // SAFETY: numeric checks establish positive, bounded slice length.
        // Actual allocation validity, initialization and the entire lifetime
        // are the caller's obligations described above. No wider slice is made.
        let data = unsafe { std::slice::from_raw_parts(raw.data, layout.required_bytes) };
        Ok(Self {
            data,
            width: raw.width,
            height: raw.height,
            row_stride: raw.row_stride,
            channels: layout.channels,
            pixel_format: raw.pixel_format,
        })
    }

    /// Read RGB channels; RGBA alpha is ignored, not composited or unpremultiplied.
    pub fn rgb_at(&self, x: u32, y: u32) -> Option<[u8; 3]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let offset = (y as usize)
            .checked_mul(self.row_stride)?
            .checked_add((x as usize).checked_mul(self.channels)?)?;
        let pixel = self.data.get(offset..offset.checked_add(3)?)?;
        if self.pixel_format == FY_BGR8 {
            Some([pixel[2], pixel[1], pixel[0]])
        } else {
            Some([pixel[0], pixel[1], pixel[2]])
        }
    }

    /// Necessary contiguous prefix, including inter-row padding but no trailing
    /// bytes beyond the final pixel. This is borrowed storage, not a copy.
    pub fn as_bytes(&self) -> &[u8] {
        self.data
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn row_stride(&self) -> usize {
        self.row_stride
    }
}

struct Layout {
    channels: usize,
    required_bytes: usize,
}

impl Layout {
    fn checked(
        byte_length: usize,
        width: u32,
        height: u32,
        row_stride: usize,
        pixel_format: u32,
        limits: ImageLimits,
    ) -> Result<Self, Error> {
        if limits.max_bytes == 0
            || limits.max_pixels == 0
            || width == 0
            || height == 0
            || row_stride == 0
        {
            return Err(INVALID);
        }
        let channels = match pixel_format {
            FY_RGB8 | FY_BGR8 => 3,
            FY_RGBA8 => 4,
            _ => return Err(INVALID),
        };
        let width = usize::try_from(width).map_err(|_| INVALID)?;
        let height = usize::try_from(height).map_err(|_| INVALID)?;
        let row_bytes = width.checked_mul(channels).ok_or(INVALID)?;
        if row_stride < row_bytes {
            return Err(INVALID);
        }
        let pixels = width.checked_mul(height).ok_or(INVALID)?;
        let required_bytes = (height - 1)
            .checked_mul(row_stride)
            .and_then(|prefix| prefix.checked_add(row_bytes))
            .ok_or(INVALID)?;
        if required_bytes > isize::MAX as usize || byte_length < required_bytes {
            return Err(INVALID);
        }
        if byte_length > limits.max_bytes || pixels > limits.max_pixels {
            return Err(LIMIT);
        }
        Ok(Self {
            channels,
            required_bytes,
        })
    }
}
