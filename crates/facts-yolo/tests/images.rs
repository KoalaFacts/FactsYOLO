// SPDX-License-Identifier: Apache-2.0
use facts_yolo::{
    Error,
    ffi::types::*,
    image::{ImageLimits, ImageView},
};

fn limits() -> ImageLimits {
    ImageLimits {
        max_bytes: 512,
        max_pixels: 64,
    }
}

fn raw(data: &[u8], width: u32, height: u32, stride: usize, format: u32) -> FyImage {
    FyImage {
        struct_size: size_of::<FyImage>() as u32,
        struct_version: 1,
        required_flags: 0,
        data: data.as_ptr(),
        byte_length: data.len(),
        width,
        height,
        row_stride: stride,
        pixel_format: format,
    }
}

fn status(result: Result<ImageView<'_>, Error>, expected: i32) {
    assert_eq!(result.unwrap_err().status, expected);
}

#[test]
fn padded_rows_need_fourteen_bytes_not_sixteen() {
    let bytes = [1, 2, 3, 4, 5, 6, 99, 99, 7, 8, 9, 10, 11, 12];
    let view = ImageView::from_bytes(&bytes, 2, 2, 8, FY_RGB8, limits()).unwrap();
    assert_eq!((view.width(), view.height(), view.row_stride()), (2, 2, 8));
    assert_eq!(view.as_bytes().len(), 14);
    assert_eq!(view.as_bytes().as_ptr(), bytes.as_ptr());
    assert_eq!(view.rgb_at(0, 0), Some([1, 2, 3]));
    assert_eq!(view.rgb_at(1, 0), Some([4, 5, 6]));
    assert_eq!(view.rgb_at(0, 1), Some([7, 8, 9]));
    assert_eq!(view.rgb_at(1, 1), Some([10, 11, 12]));
}

#[test]
fn truncated_last_row_is_rejected() {
    status(
        ImageView::from_bytes(&[0; 13], 2, 2, 8, FY_RGB8, limits()),
        FY_INVALID_ARGUMENT,
    );
    let backing = [0; 14];
    let mut image = raw(&backing, 2, 2, 8, FY_RGB8);
    image.byte_length = 13;
    status(
        unsafe { ImageView::from_ffi(&image, limits()) },
        FY_INVALID_ARGUMENT,
    );
}

#[test]
fn rgb_bgr_and_straight_rgba_share_channel_semantics() {
    let rgb = [1, 2, 3, 4, 5, 6];
    let bgr = [3, 2, 1, 6, 5, 4];
    let rgba = [1, 2, 3, 0, 4, 5, 6, 255];
    for (data, format, stride) in [
        (&rgb[..], FY_RGB8, 6),
        (&bgr[..], FY_BGR8, 6),
        (&rgba[..], FY_RGBA8, 8),
    ] {
        let safe = ImageView::from_bytes(data, 2, 1, stride, format, limits()).unwrap();
        let image = raw(data, 2, 1, stride, format);
        let ffi = unsafe { ImageView::from_ffi(&image, limits()) }.unwrap();
        for (x, expected) in [(0, [1, 2, 3]), (1, [4, 5, 6])] {
            assert_eq!(safe.rgb_at(x, 0), Some(expected));
            assert_eq!(ffi.rgb_at(x, 0), safe.rgb_at(x, 0));
        }
    }
}

#[test]
fn invalid_dimensions_strides_and_formats_are_rejected() {
    let bytes = [0; 32];
    for (width, height, stride, format) in [
        (0, 1, 3, FY_RGB8),
        (1, 0, 3, FY_RGB8),
        (1, 1, 0, FY_RGB8),
        (2, 1, 5, FY_RGB8),
        (1, 1, 3, 0),
        (1, 1, 3, u32::MAX),
    ] {
        status(
            ImageView::from_bytes(&bytes, width, height, stride, format, limits()),
            FY_INVALID_ARGUMENT,
        );
    }
}

#[test]
fn positive_limits_are_required_and_boundaries_are_inclusive() {
    let bytes = [0; 14];
    let exact = ImageLimits {
        max_bytes: 14,
        max_pixels: 4,
    };
    assert!(ImageView::from_bytes(&bytes, 2, 2, 8, FY_RGB8, exact).is_ok());
    for limit in [
        ImageLimits {
            max_bytes: 13,
            ..exact
        },
        ImageLimits {
            max_pixels: 3,
            ..exact
        },
    ] {
        status(
            ImageView::from_bytes(&bytes, 2, 2, 8, FY_RGB8, limit),
            FY_LIMIT_EXCEEDED,
        );
    }
    for limit in [
        ImageLimits {
            max_bytes: 0,
            ..exact
        },
        ImageLimits {
            max_pixels: 0,
            ..exact
        },
    ] {
        status(
            ImageView::from_bytes(&bytes, 2, 2, 8, FY_RGB8, limit),
            FY_INVALID_ARGUMENT,
        );
    }
}

#[test]
fn ffi_validates_header_and_null_before_pixel_access() {
    let bytes = [0; 14];
    let mut image = raw(&bytes, 2, 2, 8, FY_RGB8);
    image.struct_size = 8;
    status(
        unsafe { ImageView::from_ffi(&image, limits()) },
        FY_INVALID_ARGUMENT,
    );
    image.struct_size = size_of::<FyImage>() as u32;
    image.struct_version = 2;
    status(
        unsafe { ImageView::from_ffi(&image, limits()) },
        FY_UNSUPPORTED_ABI,
    );
    image.struct_version = 1;
    image.required_flags = 1;
    status(
        unsafe { ImageView::from_ffi(&image, limits()) },
        FY_UNSUPPORTED_FEATURE,
    );
    image.required_flags = 0;
    image.data = std::ptr::null();
    status(
        unsafe { ImageView::from_ffi(&image, limits()) },
        FY_INVALID_ARGUMENT,
    );
}

#[test]
fn declared_length_budget_is_not_only_the_accessed_prefix_budget() {
    let bytes = [7; 64];
    let image = raw(&bytes, 1, 1, 3, FY_RGB8);
    let small = ImageLimits {
        max_bytes: 16,
        max_pixels: 1,
    };
    status(
        unsafe { ImageView::from_ffi(&image, small) },
        FY_LIMIT_EXCEEDED,
    );
    let view = unsafe { ImageView::from_ffi(&image, limits()) }.unwrap();
    assert_eq!(view.as_bytes().len(), 3);
    assert_eq!(view.as_bytes().as_ptr(), bytes.as_ptr());
}

#[test]
fn arithmetic_overflow_is_rejected_before_storage_is_read() {
    let backing = [0; 8];
    let huge = ImageLimits {
        max_bytes: usize::MAX,
        max_pixels: usize::MAX,
    };
    let mut image = raw(&backing, 1, 2, usize::MAX, FY_RGB8);
    image.byte_length = usize::MAX;
    status(
        unsafe { ImageView::from_ffi(&image, huge) },
        FY_INVALID_ARGUMENT,
    );
    image.row_stride = isize::MAX as usize;
    status(
        unsafe { ImageView::from_ffi(&image, huge) },
        FY_INVALID_ARGUMENT,
    );
    image.width = u32::MAX;
    image.height = u32::MAX;
    image.pixel_format = FY_RGBA8;
    image.row_stride = usize::MAX;
    status(
        unsafe { ImageView::from_ffi(&image, huge) },
        FY_INVALID_ARGUMENT,
    );
}

#[test]
fn one_row_does_not_require_unused_final_padding() {
    let bytes = [10, 20, 30];
    let view = ImageView::from_bytes(&bytes, 1, 1, usize::MAX, FY_RGB8, limits()).unwrap();
    assert_eq!(view.as_bytes(), &bytes);
    assert_eq!(view.rgb_at(0, 0), Some(bytes));
}

#[test]
fn out_of_bounds_coordinates_never_read_storage() {
    let bytes = [0; 12];
    let view = ImageView::from_bytes(&bytes, 2, 2, 6, FY_RGB8, limits()).unwrap();
    for (x, y) in [
        (2, 0),
        (0, 2),
        (u32::MAX, 0),
        (0, u32::MAX),
        (u32::MAX, u32::MAX),
    ] {
        assert_eq!(view.rgb_at(x, y), None);
    }
}

#[test]
fn immutable_view_can_be_shared_between_threads() {
    let bytes = [1, 2, 3, 4, 5, 6];
    let view = ImageView::from_bytes(&bytes, 2, 1, 6, FY_RGB8, limits()).unwrap();
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                for _ in 0..100 {
                    assert_eq!(view.rgb_at(1, 0), Some([4, 5, 6]));
                }
            });
        }
    });
}

#[test]
fn ten_thousand_backed_descriptors_match_u128_reference() {
    let mut backing = [0_u8; 512];
    for (index, byte) in backing.iter_mut().enumerate() {
        *byte = index as u8;
    }
    let mut seed = 0xFAC7_2026_u64;
    for _ in 0..10_000 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut width = 1 + (seed as u32 & 7);
        let mut height = 1 + ((seed >> 8) as u32 & 7);
        let mut format = 1 + ((seed >> 16) as u32 % 3);
        let channels = if format == FY_RGBA8 { 4 } else { 3 };
        let mut stride = width as usize * channels + ((seed >> 24) as usize & 7);
        let mut length = (height as usize - 1) * stride + width as usize * channels;
        let mut budget = limits();
        match (seed >> 32) & 7 {
            1 => length -= 1,
            2 => width = 0,
            3 => height = 0,
            4 => stride = width as usize * channels - 1,
            5 => format = 99,
            6 => budget.max_pixels = 1,
            7 => budget.max_bytes = length - 1,
            _ => {}
        }
        let c = if format == FY_RGBA8 { 4_u128 } else { 3 };
        let row = u128::from(width) * c;
        let needed = u128::from(height.saturating_sub(1)) * stride as u128 + row;
        let expected_ok = width > 0
            && height > 0
            && (1..=3).contains(&format)
            && stride as u128 >= row
            && length as u128 >= needed
            && length <= budget.max_bytes
            && u128::from(width) * u128::from(height) <= budget.max_pixels as u128;
        let result =
            ImageView::from_bytes(&backing[..length], width, height, stride, format, budget);
        let descriptor = raw(&backing[..length], width, height, stride, format);
        // SAFETY: the entire advertised range is backed by initialized storage
        // kept immutable and alive until both views are dropped this iteration.
        let ffi_result = unsafe { ImageView::from_ffi(&descriptor, budget) };
        assert_eq!(result.is_ok(), expected_ok, "seed={seed:#x}");
        assert_eq!(ffi_result.is_ok(), expected_ok, "FFI seed={seed:#x}");
        if let (Err(safe_error), Err(ffi_error)) = (&result, &ffi_result) {
            assert_eq!(safe_error.status, ffi_error.status);
        }
        if let Ok(view) = result {
            let ffi_view = ffi_result.unwrap();
            assert_eq!(view.as_bytes().len() as u128, needed);
            assert_eq!(ffi_view.as_bytes(), view.as_bytes());
            for y in 0..height {
                for x in 0..width {
                    let start = y as usize * stride + x as usize * c as usize;
                    let p = &backing[start..start + 3];
                    let expected = if format == FY_BGR8 {
                        [p[2], p[1], p[0]]
                    } else {
                        [p[0], p[1], p[2]]
                    };
                    assert_eq!(view.rgb_at(x, y), Some(expected));
                    assert_eq!(ffi_view.rgb_at(x, y), Some(expected));
                }
            }
        }
    }
}
