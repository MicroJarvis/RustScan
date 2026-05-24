use super::*;
use crate::sh::rgb_to_sh0_value as rgb_to_sh_dc;
use tempfile::tempdir;

#[cfg(feature = "gpu")]
#[test]
fn test_save_splats_ply_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("splats.ply");
    let metadata = SplatMetadata {
        iterations: 12,
        final_loss: 0.25,
        gaussian_count: 1,
        sh_degree: 0,
    };
    let splats = crate::core::HostSplats::from_components(
        vec![0.0, 0.0, 1.0],
        vec![0.1f32.ln(), 0.1f32.ln(), 0.1f32.ln()],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0],
        vec![rgb_to_sh_dc(0.2), rgb_to_sh_dc(0.3), rgb_to_sh_dc(0.4)],
        0,
    )
    .unwrap();

    save_splats_ply(&path, &splats, &metadata).unwrap();

    let (loaded, loaded_meta) = load_splats_ply(&path).unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded_meta.iterations, 12);
    assert_eq!(loaded_meta.sh_degree, 0);
    assert!((loaded.opacity_logit(0) - splats.opacity_logit(0)).abs() < 1e-6);
}

#[cfg(feature = "gpu")]
#[test]
fn test_save_splats_ply_roundtrip_with_sh_coeffs() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("splats_sh.ply");
    let metadata = SplatMetadata {
        iterations: 7,
        final_loss: 0.125,
        gaussian_count: 1,
        sh_degree: 3,
    };
    let mut sh_coeffs = vec![rgb_to_sh_dc(0.2), rgb_to_sh_dc(0.3), rgb_to_sh_dc(0.4)];
    sh_coeffs.extend((0..45).map(|idx| idx as f32 * 0.01));
    let splats = crate::core::HostSplats::from_components(
        vec![1.0, 2.0, 3.0],
        vec![0.2f32.ln(), 0.3f32.ln(), 0.4f32.ln()],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.25],
        sh_coeffs.clone(),
        3,
    )
    .unwrap();

    save_splats_ply(&path, &splats, &metadata).unwrap();

    let (loaded, loaded_meta) = load_splats_ply(&path).unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded_meta.sh_degree, 3);
    assert!((loaded.opacity_logit(0) - splats.opacity_logit(0)).abs() < 1e-6);
    for (actual, expected) in loaded.as_view().sh_coeffs.iter().zip(sh_coeffs.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[cfg(feature = "gpu")]
#[test]
fn test_save_splats_splat_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("splats.splat");
    let metadata = SplatMetadata {
        iterations: 12,
        final_loss: 0.25,
        gaussian_count: 2,
        sh_degree: 0,
    };
    let splats = crate::core::HostSplats::from_components(
        vec![0.0, 0.0, 1.0, 1.0, 2.0, 3.0],
        vec![
            0.1f32.ln(),
            0.2f32.ln(),
            0.3f32.ln(),
            0.4f32.ln(),
            0.5f32.ln(),
            0.6f32.ln(),
        ],
        vec![1.0, 0.0, 0.0, 0.0, 0.707, 0.0, 0.707, 0.0],
        vec![0.0, 0.42],
        vec![
            rgb_to_sh_dc(0.2),
            rgb_to_sh_dc(0.3),
            rgb_to_sh_dc(0.4),
            rgb_to_sh_dc(0.8),
            rgb_to_sh_dc(0.7),
            rgb_to_sh_dc(0.6),
        ],
        0,
    )
    .unwrap();

    save_splats(&path, &splats, &metadata).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(bytes.len(), 64);

    let (loaded, loaded_meta) = load_splats(&path).unwrap();
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded_meta.gaussian_count, 2);
    assert_eq!(loaded_meta.sh_degree, 0);
    assert_eq!(loaded.position(1), [1.0, 2.0, 3.0]);
    for (actual, expected) in loaded.scale(0).into_iter().zip([0.1, 0.2, 0.3]) {
        assert!((actual - expected).abs() < 1e-6);
    }
    assert!((loaded.opacity(0) - 0.5019608).abs() < 1e-6);
    assert!((loaded.rgb_color(1)[0] - 0.8).abs() <= 1.0 / 255.0 + 1e-6);
}

#[cfg(feature = "gpu")]
#[test]
fn test_save_splats_dispatches_ply_by_extension() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("splats.ply");
    let metadata = SplatMetadata {
        iterations: 1,
        final_loss: 0.0,
        gaussian_count: 1,
        sh_degree: 0,
    };
    let splats = crate::core::HostSplats::from_components(
        vec![0.0, 0.0, 1.0],
        vec![0.1f32.ln(), 0.1f32.ln(), 0.1f32.ln()],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0],
        vec![rgb_to_sh_dc(0.2), rgb_to_sh_dc(0.3), rgb_to_sh_dc(0.4)],
        0,
    )
    .unwrap();

    save_splats(&path, &splats, &metadata).unwrap();
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(contents.starts_with("ply\n"));
}

#[cfg(feature = "gpu")]
#[test]
fn test_load_external_raw_logit_opacity() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("external_raw_logit.ply");
    let contents = concat!(
        "ply\n",
        "format ascii 1.0\n",
        "comment exported elsewhere\n",
        "comment iterations 1\n",
        "comment final_loss 0.0\n",
        "comment sh_degree 0\n",
        "element vertex 1\n",
        "property float x\n",
        "property float y\n",
        "property float z\n",
        "property float nx\n",
        "property float ny\n",
        "property float nz\n",
        "property float f_dc_0\n",
        "property float f_dc_1\n",
        "property float f_dc_2\n",
        "property float f_rest_0\n",
        "property float f_rest_1\n",
        "property float f_rest_2\n",
        "property float f_rest_3\n",
        "property float f_rest_4\n",
        "property float f_rest_5\n",
        "property float f_rest_6\n",
        "property float f_rest_7\n",
        "property float f_rest_8\n",
        "property float f_rest_9\n",
        "property float f_rest_10\n",
        "property float f_rest_11\n",
        "property float f_rest_12\n",
        "property float f_rest_13\n",
        "property float f_rest_14\n",
        "property float f_rest_15\n",
        "property float f_rest_16\n",
        "property float f_rest_17\n",
        "property float f_rest_18\n",
        "property float f_rest_19\n",
        "property float f_rest_20\n",
        "property float f_rest_21\n",
        "property float f_rest_22\n",
        "property float f_rest_23\n",
        "property float f_rest_24\n",
        "property float f_rest_25\n",
        "property float f_rest_26\n",
        "property float f_rest_27\n",
        "property float f_rest_28\n",
        "property float f_rest_29\n",
        "property float f_rest_30\n",
        "property float f_rest_31\n",
        "property float f_rest_32\n",
        "property float f_rest_33\n",
        "property float f_rest_34\n",
        "property float f_rest_35\n",
        "property float f_rest_36\n",
        "property float f_rest_37\n",
        "property float f_rest_38\n",
        "property float f_rest_39\n",
        "property float f_rest_40\n",
        "property float f_rest_41\n",
        "property float f_rest_42\n",
        "property float f_rest_43\n",
        "property float f_rest_44\n",
        "property float opacity\n",
        "property float scale_0\n",
        "property float scale_1\n",
        "property float scale_2\n",
        "property float rot_0\n",
        "property float rot_1\n",
        "property float rot_2\n",
        "property float rot_3\n",
        "end_header\n",
        "0 0 1 0 0 0 0.1 0.2 0.3 ",
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ",
        "2.0 -2.3025851 -2.3025851 -2.3025851 1 0 0 0\n",
    );
    std::fs::write(&path, contents).unwrap();

    let (loaded, _) = load_splats_ply(&path).unwrap();
    assert!((loaded.opacity_logit(0) - 2.0).abs() < 1e-6);
}

#[cfg(feature = "gpu")]
#[test]
fn test_load_external_channel_major_sh_coeffs() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("external_channel_major_sh.ply");
    let contents = concat!(
        "ply\n",
        "format ascii 1.0\n",
        "comment Exported from Brush\n",
        "comment SH degree: 1\n",
        "element vertex 1\n",
        "property float x\n",
        "property float y\n",
        "property float z\n",
        "property float nx\n",
        "property float ny\n",
        "property float nz\n",
        "property float f_dc_0\n",
        "property float f_dc_1\n",
        "property float f_dc_2\n",
        "property float f_rest_0\n",
        "property float f_rest_1\n",
        "property float f_rest_2\n",
        "property float f_rest_3\n",
        "property float f_rest_4\n",
        "property float f_rest_5\n",
        "property float f_rest_6\n",
        "property float f_rest_7\n",
        "property float f_rest_8\n",
        "property float f_rest_9\n",
        "property float f_rest_10\n",
        "property float f_rest_11\n",
        "property float f_rest_12\n",
        "property float f_rest_13\n",
        "property float f_rest_14\n",
        "property float f_rest_15\n",
        "property float f_rest_16\n",
        "property float f_rest_17\n",
        "property float f_rest_18\n",
        "property float f_rest_19\n",
        "property float f_rest_20\n",
        "property float f_rest_21\n",
        "property float f_rest_22\n",
        "property float f_rest_23\n",
        "property float f_rest_24\n",
        "property float f_rest_25\n",
        "property float f_rest_26\n",
        "property float f_rest_27\n",
        "property float f_rest_28\n",
        "property float f_rest_29\n",
        "property float f_rest_30\n",
        "property float f_rest_31\n",
        "property float f_rest_32\n",
        "property float f_rest_33\n",
        "property float f_rest_34\n",
        "property float f_rest_35\n",
        "property float f_rest_36\n",
        "property float f_rest_37\n",
        "property float f_rest_38\n",
        "property float f_rest_39\n",
        "property float f_rest_40\n",
        "property float f_rest_41\n",
        "property float f_rest_42\n",
        "property float f_rest_43\n",
        "property float f_rest_44\n",
        "property float opacity\n",
        "property float scale_0\n",
        "property float scale_1\n",
        "property float scale_2\n",
        "property float rot_0\n",
        "property float rot_1\n",
        "property float rot_2\n",
        "property float rot_3\n",
        "end_header\n",
        "0 0 1 0 0 0 0.1 0.2 0.3 ",
        "1 2 3 4 5 6 7 8 9 ",
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ",
        "2.0 -2.3025851 -2.3025851 -2.3025851 1 0 0 0\n",
    );
    std::fs::write(&path, contents).unwrap();

    let (loaded, meta) = load_splats_ply(&path).unwrap();
    assert_eq!(meta.sh_degree, 1);
    assert_eq!(
        loaded.as_view().sh_coeffs,
        &[0.1, 0.2, 0.3, 1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]
    );
}

#[cfg(feature = "gpu")]
#[test]
fn test_load_binary_brush_export_ply() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("brush_binary_export.ply");

    let mut bytes = Vec::new();
    bytes.extend_from_slice(
        concat!(
            "ply\n",
            "format binary_little_endian 1.0\n",
            "comment Exported from Brush\n",
            "comment SH degree: 3\n",
            "comment SplatRenderMode: default\n",
            "element vertex 1\n",
            "property float x\n",
            "property float y\n",
            "property float z\n",
            "property float scale_0\n",
            "property float scale_1\n",
            "property float scale_2\n",
            "property float opacity\n",
            "property float rot_0\n",
            "property float rot_1\n",
            "property float rot_2\n",
            "property float rot_3\n",
            "property float f_dc_0\n",
            "property float f_dc_1\n",
            "property float f_dc_2\n",
            "property float f_rest_0\n",
            "property float f_rest_1\n",
            "property float f_rest_2\n",
            "property float f_rest_3\n",
            "property float f_rest_4\n",
            "property float f_rest_5\n",
            "property float f_rest_6\n",
            "property float f_rest_7\n",
            "property float f_rest_8\n",
            "property float f_rest_9\n",
            "property float f_rest_10\n",
            "property float f_rest_11\n",
            "property float f_rest_12\n",
            "property float f_rest_13\n",
            "property float f_rest_14\n",
            "property float f_rest_15\n",
            "property float f_rest_16\n",
            "property float f_rest_17\n",
            "property float f_rest_18\n",
            "property float f_rest_19\n",
            "property float f_rest_20\n",
            "property float f_rest_21\n",
            "property float f_rest_22\n",
            "property float f_rest_23\n",
            "property float f_rest_24\n",
            "property float f_rest_25\n",
            "property float f_rest_26\n",
            "property float f_rest_27\n",
            "property float f_rest_28\n",
            "property float f_rest_29\n",
            "property float f_rest_30\n",
            "property float f_rest_31\n",
            "property float f_rest_32\n",
            "property float f_rest_33\n",
            "property float f_rest_34\n",
            "property float f_rest_35\n",
            "property float f_rest_36\n",
            "property float f_rest_37\n",
            "property float f_rest_38\n",
            "property float f_rest_39\n",
            "property float f_rest_40\n",
            "property float f_rest_41\n",
            "property float f_rest_42\n",
            "property float f_rest_43\n",
            "property float f_rest_44\n",
            "end_header\n",
        )
        .as_bytes(),
    );

    let mut row = vec![
        1.0f32,
        2.0,
        3.0,
        -std::f32::consts::LN_10,
        -1.609438,
        -1.2039728,
        2.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.1,
        0.2,
        0.3,
    ];
    row.extend((1..=45).map(|value| value as f32));
    for value in row {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(&path, bytes).unwrap();

    let (loaded, meta) = load_splats_ply(&path).unwrap();
    assert_eq!(meta.sh_degree, 3);
    assert_eq!(loaded.position(0), [1.0, 2.0, 3.0]);
    assert!((loaded.opacity_logit(0) - 2.0).abs() < 1e-6);

    let mut expected = vec![0.1, 0.2, 0.3];
    for coeff_idx in 0..15 {
        expected.push((coeff_idx + 1) as f32);
        expected.push((coeff_idx + 16) as f32);
        expected.push((coeff_idx + 31) as f32);
    }
    assert_eq!(loaded.as_view().sh_coeffs, expected.as_slice());
}
