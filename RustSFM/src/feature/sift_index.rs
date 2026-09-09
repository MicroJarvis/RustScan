const DESCRIPTOR_DIM: usize = 128;

#[derive(Debug)]
pub struct SiftDescriptorIndex {
    descriptors: Vec<[u8; DESCRIPTOR_DIM]>,
}

#[derive(Debug, Clone, Copy)]
pub struct SiftNearestNeighbors {
    pub best_l2: f32,
    pub best_index: u32,
    pub second_best_l2: f32,
    pub second_best_index: u32,
}

impl SiftDescriptorIndex {
    pub fn build(descriptors: &[[u8; DESCRIPTOR_DIM]]) -> Self {
        Self {
            descriptors: descriptors.to_vec(),
        }
    }

    pub fn search_two_nearest(&self, query: &[u8; DESCRIPTOR_DIM]) -> Option<SiftNearestNeighbors> {
        if self.descriptors.is_empty() {
            return None;
        }

        let mut best_index = 0u32;
        let mut best_l2 = f32::INFINITY;
        let mut second_best_index = 0u32;
        let mut second_best_l2 = f32::INFINITY;

        for (idx, descriptor) in self.descriptors.iter().enumerate() {
            let l2_dist = colmap_uint8_l2_distance2(query, descriptor);
            if l2_dist < best_l2 {
                second_best_l2 = best_l2;
                second_best_index = best_index;
                best_l2 = l2_dist;
                best_index = idx as u32;
            } else if l2_dist < second_best_l2 {
                second_best_l2 = l2_dist;
                second_best_index = idx as u32;
            }
        }

        Some(SiftNearestNeighbors {
            best_l2,
            best_index,
            second_best_l2,
            second_best_index,
        })
    }
}

#[inline]
pub(crate) fn colmap_uint8_l2_distance2(
    left: &[u8; DESCRIPTOR_DIM],
    right: &[u8; DESCRIPTOR_DIM],
) -> f32 {
    // The maximum sum is 128 * 255² = 8,323,200 < 2²⁴: both the old
    // float reduction and this integer reduction are exact. Integer addition
    // permits SIMD reassociation without changing distances or tie-breaking.
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| {
            let delta = i32::from(*a) - i32::from(*b);
            (delta * delta) as u32
        })
        .sum::<u32>() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn float_reference(left: &[u8; 128], right: &[u8; 128]) -> f32 {
        left.iter()
            .zip(right)
            .map(|(a, b)| {
                let delta = i32::from(*a) - i32::from(*b);
                (delta * delta) as f32
            })
            .sum()
    }

    #[test]
    fn integer_distance_is_bit_exact_for_extremes_and_random_descriptors() {
        use rand::{RngCore, SeedableRng};
        assert_eq!(
            colmap_uint8_l2_distance2(&[0; 128], &[255; 128]),
            8_323_200.0
        );
        for a in [0, 1, 127, 128, 254, 255] {
            for b in 0..=255 {
                let left = [a; 128];
                let right = [b; 128];
                assert_eq!(
                    colmap_uint8_l2_distance2(&left, &right).to_bits(),
                    float_reference(&left, &right).to_bits()
                );
            }
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for _ in 0..4096 {
            let mut left = [0; 128];
            let mut right = [0; 128];
            rng.fill_bytes(&mut left);
            rng.fill_bytes(&mut right);
            assert_eq!(
                colmap_uint8_l2_distance2(&left, &right).to_bits(),
                float_reference(&left, &right).to_bits()
            );
        }
    }

    #[test]
    fn nearest_two_preserve_float_reference_and_ties() {
        use rand::{RngCore, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        for count in [0, 1, 2, 17, 128] {
            let mut train = vec![[0u8; 128]; count];
            for descriptor in &mut train {
                rng.fill_bytes(descriptor);
            }
            if count >= 2 {
                train[1] = train[0];
            }
            let index = SiftDescriptorIndex::build(&train);
            for query in train
                .iter()
                .copied()
                .chain([[0; 128], [255; 128], [128; 128]])
            {
                let got = index.search_two_nearest(&query);
                if count == 0 {
                    assert!(got.is_none());
                    continue;
                }
                let mut expected: Vec<_> = train
                    .iter()
                    .enumerate()
                    .map(|(i, row)| (float_reference(&query, row), i as u32))
                    .collect();
                expected.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                let first = expected[0];
                let second = expected.get(1).copied().unwrap_or((f32::INFINITY, 0));
                let got = got.unwrap();
                assert_eq!(
                    (got.best_index, got.best_l2.to_bits()),
                    (first.1, first.0.to_bits())
                );
                assert_eq!(
                    (got.second_best_index, got.second_best_l2.to_bits()),
                    (second.1, second.0.to_bits())
                );
            }
        }
        let index = SiftDescriptorIndex::build(&[[0; 128], [2; 128], [0; 128]]);
        let tied = index.search_two_nearest(&[1; 128]).unwrap();
        assert_eq!((tied.best_index, tied.second_best_index), (0, 1));
    }

    #[test]
    fn index_finds_exact_nearest_neighbor() {
        let mut a = [0u8; DESCRIPTOR_DIM];
        let mut near = [0u8; DESCRIPTOR_DIM];
        let mut far = [0u8; DESCRIPTOR_DIM];
        a.fill(255);
        near.fill(255);
        near[0] = 253;
        far.fill(0);

        let index = SiftDescriptorIndex::build(&[near, far]);
        let neighbors = index.search_two_nearest(&a).expect("neighbors");
        assert_eq!(neighbors.best_index, 0);
        assert!(neighbors.best_l2 < neighbors.second_best_l2);
    }
}
