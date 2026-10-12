// SPDX-License-Identifier: Apache-2.0
//! Borrowed image-view contract. RED-stage constructors deliberately reject input.
use crate::{Error, error::INVALID, ffi::types::FyImage};
use std::marker::PhantomData;

#[derive(Clone, Copy, Debug)]
pub struct ImageLimits {
    pub max_bytes: usize,
    pub max_pixels: usize,
}

#[derive(Debug)]
pub struct ImageView<'a> {
    _storage: PhantomData<&'a [u8]>,
}

impl<'a> ImageView<'a> {
    pub fn from_bytes(
        _data: &'a [u8],
        _width: u32,
        _height: u32,
        _row_stride: usize,
        _pixel_format: u32,
        _limits: ImageLimits,
    ) -> Result<Self, Error> {
        Err(INVALID)
    }

    /// # Safety
    /// On accepted metadata, non-null storage must cover the declared range,
    /// remain immutable/live for 'a, and belong to one allocation. Null pointers
    /// and invalid numeric metadata are rejected before accessing pixel storage.
    pub unsafe fn from_ffi(_raw: &'a FyImage, _limits: ImageLimits) -> Result<Self, Error> {
        Err(INVALID)
    }

    pub fn rgb_at(&self, _x: u32, _y: u32) -> Option<[u8; 3]> {
        None
    }

    pub fn as_bytes(&self) -> &[u8] {
        &[]
    }

    pub fn width(&self) -> u32 {
        0
    }

    pub fn height(&self) -> u32 {
        0
    }

    pub fn row_stride(&self) -> usize {
        0
    }
}
