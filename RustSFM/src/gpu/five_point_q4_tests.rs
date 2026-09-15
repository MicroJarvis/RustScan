//! Actual device test of the generated polynomial on fixed real GPU B matrices.
use super::*;

#[test]
fn five_point_q4_arithmetic_probe() -> Result<()> {
    let gpu = WgpuFivePointF32::try_new()?;
    let module = shader_module(
        gpu.context.device(),
        r#"
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read_write> y: array<f32>;
@compute @workgroup_size(1)
fn probe() {
  let p=x[0]*x[1];
  let pe=fma(x[0],x[1],-p);
  let s=x[2]+x[3];
  let se=fma(-1.0,s,x[2])+x[3];
  y[0]=p; y[1]=pe; y[2]=s; y[3]=se; y[4]=(x[2]-s)+x[3];
}
"#,
    )?;
    let probe = kernel(gpu.context.device(), &module, "probe", 1, false)?;
    let input = [1.0f32 + f32::EPSILON, 1.0f32 - f32::EPSILON, 1e8, 1.0];
    let out = gpu.dispatch(&[(&probe, 1)], &[bytemuck::cast_slice(&input)], 1, 5)?;
    let expected = [1.0, -f32::EPSILON * f32::EPSILON, 1e8, 1.0];
    eprintln!("Q4 arithmetic probe actual={out:?} expected={expected:?}");
    assert_eq!(
        &out[..4],
        &expected,
        "GPU must retain both compensation residuals"
    );
    Ok(())
}

#[test]
fn five_point_q4_same_b_compensation_survives_gpu_compiler() -> Result<()> {
    // Required device: this numerical experiment must not silently skip.
    let context = WgpuContext::try_new()?;
    eprintln!("Q4 actual device: {:?}", context.capabilities());
    let gpu = WgpuFivePointF32::from_context(context)?;
    let fixtures: Vec<serde_json::Value> =
        serde_json::from_str(include_str!("../../examples/q4/polynomial-fixtures.json"))?;
    let bs: Vec<[f32; 39]> = fixtures
        .iter()
        .map(|f| std::array::from_fn(|i| f32::from_bits(f["B_bits"][i].as_u64().unwrap() as u32)))
        .collect();
    let bs: Vec<_> = [
        1.0,
        -1.0,
        2.0f32.powi(-20),
        -2.0f32.powi(-20),
        2.0f32.powi(20),
        -2.0f32.powi(20),
    ]
    .into_iter()
    .flat_map(|scale| bs.iter().map(move |b| b.map(|x| x * scale)))
    .collect();
    let loader = shader_module(
        gpu.context.device(),
        &format!(
            "{ALGEBRA}\n{}",
            r#"
@compute @workgroup_size(1)
fn q4_load(@builtin(workgroup_id) id: vec3<u32>) {
  for(var i=0u;i<39u;i++) { algebra_output[id.x*352u+300u+i]=bases[id.x*39u+i]; }
}
"#
        ),
    )?;
    let load = kernel(gpu.context.device(), &loader, "q4_load", 1, false)?;
    let run = || {
        gpu.dispatch(
            &[(&load, 1), (&gpu.polynomial, 11)],
            &[bytemuck::cast_slice(&bs)],
            bs.len(),
            352,
        )
    };
    let values = run()?;
    let again = run()?;
    assert_eq!(
        values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        again.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    let mut failures = Vec::new();
    for (i, b) in bs.iter().enumerate() {
        let reference = determinant_coeffs(&b.map(f64::from));
        let error = relative_error(&values[i * 352 + 339..i * 352 + 350], &reference);
        let fixture = &fixtures[i % fixtures.len()];
        let old = fixture["baseline_same_B_error"].as_f64().unwrap();
        if i < fixtures.len() || error >= old * 0.1 {
            eprintln!(
                "Q4 trial {} scale_case={} baseline={old:e} candidate={error:e}",
                fixture["trial"],
                i / fixtures.len()
            );
        }
        if error >= old * 0.1 {
            failures.push((i, error, old));
        }
    }
    eprintln!(
        "Q4 verified {} same-B/sign/power-of-two cases twice",
        bs.len()
    );
    assert!(
        failures.is_empty(),
        "compensation must improve every fixture by >=10x: {failures:?}"
    );
    Ok(())
}
