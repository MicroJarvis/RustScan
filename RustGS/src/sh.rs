#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SplatColorRepresentation {
    #[default]
    Rgb,
    SphericalHarmonics {
        degree: usize,
    },
}

#[cfg_attr(not(test), allow(dead_code))]
impl SplatColorRepresentation {
    pub const fn sh_degree(self) -> usize {
        match self {
            Self::Rgb => 0,
            Self::SphericalHarmonics { degree } => degree,
        }
    }
}

pub const SH_C0: f32 = 0.282_094_8;

pub const fn sh_coeff_count_for_degree(degree: usize) -> usize {
    (degree + 1) * (degree + 1)
}

pub fn rgb_to_sh0_value(rgb: f32) -> f32 {
    (rgb - 0.5) / SH_C0
}

pub fn sh0_to_rgb_value(sh: f32) -> f32 {
    (sh * SH_C0) + 0.5
}
