mod plan;
mod types;

pub(crate) use plan::SiftPlan;
pub(crate) use types::{GpuKeypoint, SiftUniforms};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sift::SiftExtractionOptions;

    #[test]
    fn octave_plan_matches_sift_level_and_sigma_schedule() {
        let options = SiftExtractionOptions {
            first_octave: -1,
            num_octaves: 4,
            octave_resolution: 3,
            ..SiftExtractionOptions::default()
        };
        let plan = SiftPlan::new(640, 480, &options).unwrap();
        assert_eq!(plan.first_octave, -1);
        assert_eq!(plan.octaves[0].dimensions(), (1280, 960));
        assert_eq!(plan.octaves[0].octave, -1);
        assert_eq!(plan.octaves[0].pixel_count().unwrap(), 1280 * 960);
        assert_eq!(plan.octaves[0].gaussian_levels, 6);
        assert_eq!(plan.octaves[0].dog_levels, 5);
        assert!((plan.sigma_step - 2.0f32.powf(1.0 / 3.0)).abs() < 1.0e-6);
        assert_eq!(plan.octaves[1].dimensions(), (640, 480));
        assert!(plan.candidate_capacity >= options.max_num_features as u32);
    }

    #[test]
    fn octave_plan_stops_before_images_become_too_small() {
        let options = SiftExtractionOptions {
            first_octave: 0,
            num_octaves: 4,
            ..SiftExtractionOptions::default()
        };
        let plan = SiftPlan::new(33, 33, &options).unwrap();
        assert_eq!(plan.octaves.len(), 1);
    }

    #[test]
    fn gpu_sift_abi_records_have_wgsl_compatible_sizes() {
        assert_eq!(std::mem::size_of::<SiftUniforms>(), 32);
        assert_eq!(std::mem::size_of::<GpuKeypoint>(), 32);
    }
}
