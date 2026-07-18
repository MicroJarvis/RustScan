mod plan;
mod types;

pub(crate) use plan::SiftPlan;
pub(crate) use types::{GpuKeypoint, PyramidParams, SiftUniforms};

use crate::gpu::WgpuContext;
use anyhow::{bail, Context, Result};
use std::sync::Arc;
use wgpu::util::DeviceExt;

const PYRAMID_SHADER: &str = include_str!("../shaders/sift_pyramid.wgsl");
const WORKGROUP_SIZE: u32 = 16;

pub(crate) struct SiftPyramid {
    context: Arc<WgpuContext>,
    bind_group_layout: wgpu::BindGroupLayout,
    gaussian_pipeline: wgpu::ComputePipeline,
    dog_pipeline: wgpu::ComputePipeline,
    downsample_pipeline: wgpu::ComputePipeline,
    dummy_buffer: wgpu::Buffer,
}

impl SiftPyramid {
    pub(crate) fn new(context: Arc<WgpuContext>) -> Result<Self> {
        let device = context.device();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rustsfm SIFT pyramid bind group layout"),
            entries: &[
                storage_layout_entry(0, true),
                storage_layout_entry(1, true),
                storage_layout_entry(2, false),
                storage_layout_entry(3, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rustsfm SIFT pyramid shader"),
            source: wgpu::ShaderSource::Wgsl(PYRAMID_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rustsfm SIFT pyramid pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let gaussian_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rustsfm SIFT Gaussian pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("gaussian_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let dog_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rustsfm SIFT DoG pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("dog_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let downsample_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("rustsfm SIFT downsample pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("downsample_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let dummy_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm SIFT pyramid dummy"),
            contents: bytemuck::bytes_of(&0.0f32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        Ok(Self {
            context,
            bind_group_layout,
            gaussian_pipeline,
            dog_pipeline,
            downsample_pipeline,
            dummy_buffer,
        })
    }

    #[cfg(test)]
    pub(crate) fn gaussian_for_test(
        &self,
        input: &[f32],
        width: u32,
        height: u32,
        sigma: f32,
    ) -> Result<Vec<f32>> {
        validate_level(input, width, height)?;
        let weights = gaussian_weights(sigma)?;
        let radius = u32::try_from(weights.len() / 2).context("Gaussian radius exceeds u32")?;
        let device = self.context.device();
        let source = storage_buffer_from_slice(
            device,
            "rustsfm SIFT Gaussian source",
            input,
            wgpu::BufferUsages::STORAGE,
        );
        let temporary = storage_buffer(
            device,
            "rustsfm SIFT Gaussian temporary",
            input.len(),
            wgpu::BufferUsages::STORAGE,
        )?;
        let output = storage_buffer(
            device,
            "rustsfm SIFT Gaussian output",
            input.len(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )?;
        let weights_buffer = storage_buffer_from_slice(
            device,
            "rustsfm SIFT Gaussian weights",
            &weights,
            wgpu::BufferUsages::STORAGE,
        );
        let horizontal = uniform_buffer(
            device,
            "rustsfm SIFT Gaussian horizontal params",
            &PyramidParams {
                width,
                height,
                radius,
                direction: 0,
            },
        );
        let vertical = uniform_buffer(
            device,
            "rustsfm SIFT Gaussian vertical params",
            &PyramidParams {
                width,
                height,
                radius,
                direction: 1,
            },
        );

        let horizontal_group = self.bind_group(
            &source,
            &self.dummy_buffer,
            &temporary,
            &weights_buffer,
            &horizontal,
        );
        let vertical_group = self.bind_group(
            &temporary,
            &self.dummy_buffer,
            &output,
            &weights_buffer,
            &vertical,
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rustsfm SIFT Gaussian encoder"),
        });
        dispatch_2d(
            &mut encoder,
            &self.gaussian_pipeline,
            &horizontal_group,
            width,
            height,
            "rustsfm SIFT Gaussian horizontal pass",
        );
        dispatch_2d(
            &mut encoder,
            &self.gaussian_pipeline,
            &vertical_group,
            width,
            height,
            "rustsfm SIFT Gaussian vertical pass",
        );
        self.context.queue().submit(Some(encoder.finish()));
        self.context.read_buffer(&output, input.len())
    }

    #[cfg(test)]
    pub(crate) fn dog_for_test(
        &self,
        lower: &[f32],
        upper: &[f32],
        width: u32,
        height: u32,
    ) -> Result<Vec<f32>> {
        validate_level(lower, width, height)?;
        validate_level(upper, width, height)?;
        let device = self.context.device();
        let lower_buffer = storage_buffer_from_slice(
            device,
            "rustsfm SIFT lower Gaussian",
            lower,
            wgpu::BufferUsages::STORAGE,
        );
        let upper_buffer = storage_buffer_from_slice(
            device,
            "rustsfm SIFT upper Gaussian",
            upper,
            wgpu::BufferUsages::STORAGE,
        );
        let output = storage_buffer(
            device,
            "rustsfm SIFT DoG output",
            lower.len(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )?;
        let params = uniform_buffer(
            device,
            "rustsfm SIFT DoG params",
            &PyramidParams {
                width,
                height,
                radius: 0,
                direction: 0,
            },
        );
        let group = self.bind_group(
            &lower_buffer,
            &upper_buffer,
            &output,
            &self.dummy_buffer,
            &params,
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rustsfm SIFT DoG encoder"),
        });
        dispatch_2d(
            &mut encoder,
            &self.dog_pipeline,
            &group,
            width,
            height,
            "rustsfm SIFT DoG pass",
        );
        self.context.queue().submit(Some(encoder.finish()));
        self.context.read_buffer(&output, lower.len())
    }

    #[cfg(test)]
    pub(crate) fn downsample_for_test(
        &self,
        input: &[f32],
        width: u32,
        height: u32,
    ) -> Result<Vec<f32>> {
        validate_level(input, width, height)?;
        let output_width = width / 2;
        let output_height = height / 2;
        if output_width == 0 || output_height == 0 {
            bail!("GPU SIFT downsample output dimensions must be non-zero");
        }
        let output_count = usize::try_from(u64::from(output_width) * u64::from(output_height))
            .context("GPU SIFT downsample size does not fit usize")?;
        let device = self.context.device();
        let source = storage_buffer_from_slice(
            device,
            "rustsfm SIFT downsample source",
            input,
            wgpu::BufferUsages::STORAGE,
        );
        let output = storage_buffer(
            device,
            "rustsfm SIFT downsample output",
            output_count,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )?;
        let params = uniform_buffer(
            device,
            "rustsfm SIFT downsample params",
            &PyramidParams {
                width,
                height,
                radius: 0,
                direction: 0,
            },
        );
        let group = self.bind_group(
            &source,
            &self.dummy_buffer,
            &output,
            &self.dummy_buffer,
            &params,
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rustsfm SIFT downsample encoder"),
        });
        dispatch_2d(
            &mut encoder,
            &self.downsample_pipeline,
            &group,
            output_width,
            output_height,
            "rustsfm SIFT downsample pass",
        );
        self.context.queue().submit(Some(encoder.finish()));
        self.context.read_buffer(&output, output_count)
    }

    fn bind_group(
        &self,
        source_a: &wgpu::Buffer,
        source_b: &wgpu::Buffer,
        destination: &wgpu::Buffer,
        weights: &wgpu::Buffer,
        params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rustsfm SIFT pyramid bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    entire_buffer_entry(0, source_a),
                    entire_buffer_entry(1, source_b),
                    entire_buffer_entry(2, destination),
                    entire_buffer_entry(3, weights),
                    entire_buffer_entry(4, params),
                ],
            })
    }
}

fn storage_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn entire_buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn storage_buffer_from_slice<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &str,
    values: &[T],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(values),
        usage,
    })
}

fn uniform_buffer<T: bytemuck::Pod>(device: &wgpu::Device, label: &str, value: &T) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(value),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

fn storage_buffer(
    device: &wgpu::Device,
    label: &str,
    element_count: usize,
    usage: wgpu::BufferUsages,
) -> Result<wgpu::Buffer> {
    let size = element_count
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .context("GPU SIFT buffer size overflow")?;
    if size == 0 {
        bail!("GPU SIFT buffers must not be empty");
    }
    Ok(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    }))
}

fn dispatch_2d(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    width: u32,
    height: u32,
    label: &str,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(
        width.div_ceil(WORKGROUP_SIZE),
        height.div_ceil(WORKGROUP_SIZE),
        1,
    );
}

fn gaussian_weights(sigma: f32) -> Result<Vec<f32>> {
    if !sigma.is_finite() || sigma <= 0.0 {
        bail!("GPU SIFT Gaussian sigma must be finite and positive");
    }
    let radius = (3.0 * sigma).ceil().max(1.0) as i32;
    let mut weights = (-radius..=radius)
        .map(|offset| (-0.5 * (offset as f32 / sigma).powi(2)).exp())
        .collect::<Vec<_>>();
    let sum = weights.iter().sum::<f32>();
    for weight in &mut weights {
        *weight /= sum;
    }
    Ok(weights)
}

fn validate_level(values: &[f32], width: u32, height: u32) -> Result<()> {
    let expected = usize::try_from(u64::from(width) * u64::from(height))
        .context("GPU SIFT level dimensions do not fit usize")?;
    if values.len() != expected {
        bail!(
            "GPU SIFT level has {} elements, expected {} for {}x{}",
            values.len(),
            expected,
            width,
            height
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::WgpuContext;
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

    #[test]
    fn gpu_gaussian_preserves_a_constant_image() -> anyhow::Result<()> {
        let Some(context) = WgpuContext::try_new_optional()? else {
            return Ok(());
        };
        let pyramid = SiftPyramid::new(context)?;
        let input = vec![0.25f32; 17 * 13];
        let output = pyramid.gaussian_for_test(&input, 17, 13, 1.6)?;
        assert!(
            output.iter().all(|value| (value - 0.25).abs() < 2.0e-5),
            "constant Gaussian output range: {:?}",
            output
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), value| (
                    min.min(*value),
                    max.max(*value)
                ))
        );
        Ok(())
    }

    #[test]
    fn gpu_dog_is_zero_for_equal_levels() -> anyhow::Result<()> {
        let Some(context) = WgpuContext::try_new_optional()? else {
            return Ok(());
        };
        let pyramid = SiftPyramid::new(context)?;
        let level = vec![0.75f32; 11 * 9];
        let dog = pyramid.dog_for_test(&level, &level, 11, 9)?;
        assert!(dog.iter().all(|value| value.abs() < 1.0e-7));
        Ok(())
    }

    #[test]
    fn gpu_downsample_takes_even_source_pixels() -> anyhow::Result<()> {
        let Some(context) = WgpuContext::try_new_optional()? else {
            return Ok(());
        };
        let pyramid = SiftPyramid::new(context)?;
        let input = (0..16).map(|value| value as f32).collect::<Vec<_>>();
        let output = pyramid.downsample_for_test(&input, 4, 4)?;
        assert_eq!(output, vec![0.0, 2.0, 8.0, 10.0]);
        Ok(())
    }
}
