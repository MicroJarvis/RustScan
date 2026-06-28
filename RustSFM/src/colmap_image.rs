use anyhow::{bail, Context, Result};
use std::ffi::{CStr, CString};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct ColmapGrayImage {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

#[cfg(colmap_freeimage)]
mod ffi {
    use std::os::raw::{c_char, c_int};

    #[repr(C)]
    pub struct RustSfmColmapGrayImage {
        pub data: *mut u8,
        pub width: c_int,
        pub height: c_int,
        pub error_message: *mut c_char,
    }

    extern "C" {
        pub fn rustsfm_colmap_load_grayscale_u8(
            path: *const c_char,
            out: *mut RustSfmColmapGrayImage,
        ) -> c_int;
        pub fn rustsfm_colmap_free_gray_image(image: *mut RustSfmColmapGrayImage);
    }
}

pub fn load_colmap_grayscale_u8(path: &Path) -> Result<ColmapGrayImage> {
    #[cfg(colmap_freeimage)]
    {
        return load_colmap_grayscale_u8_freeimage(path);
    }
    #[cfg(not(colmap_freeimage))]
    {
        load_colmap_grayscale_u8_image_crate(path)
    }
}

#[cfg(colmap_freeimage)]
fn load_colmap_grayscale_u8_freeimage(path: &Path) -> Result<ColmapGrayImage> {
    use ffi::{rustsfm_colmap_free_gray_image, rustsfm_colmap_load_grayscale_u8, RustSfmColmapGrayImage};

    let path = CString::new(path.to_string_lossy().as_ref())
        .context("image path contains interior NUL byte")?;
    let mut out = RustSfmColmapGrayImage {
        data: std::ptr::null_mut(),
        width: 0,
        height: 0,
        error_message: std::ptr::null_mut(),
    };
    let ok = unsafe { rustsfm_colmap_load_grayscale_u8(path.as_ptr(), &mut out) };
    if ok == 0 {
        let message = unsafe {
            if out.error_message.is_null() {
                "FreeImage grayscale load failed".to_string()
            } else {
                CStr::from_ptr(out.error_message)
                    .to_string_lossy()
                    .into_owned()
            }
        };
        unsafe {
            rustsfm_colmap_free_gray_image(&mut out);
        }
        bail!(message);
    }

    let width = out.width.max(0) as u32;
    let height = out.height.max(0) as u32;
    let expected = width as usize * height as usize;
    let data = unsafe {
        if out.data.is_null() || expected == 0 {
            Vec::new()
        } else {
            std::slice::from_raw_parts(out.data, expected).to_vec()
        }
    };
    unsafe {
        rustsfm_colmap_free_gray_image(&mut out);
    }
    Ok(ColmapGrayImage {
        data,
        width,
        height,
    })
}

#[cfg(not(colmap_freeimage))]
fn load_colmap_grayscale_u8_image_crate(path: &Path) -> Result<ColmapGrayImage> {
    use image::ImageReader;

    let decoded = ImageReader::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?
        .decode()
        .with_context(|| format!("failed to decode {}", path.display()))?
        .to_rgb8();
    let (width, height) = decoded.dimensions();
    let gray = crate::sift::rgb_to_colmap_gray_u8(decoded.as_raw(), width, height)?;
    Ok(ColmapGrayImage {
        data: gray,
        width,
        height,
    })
}
