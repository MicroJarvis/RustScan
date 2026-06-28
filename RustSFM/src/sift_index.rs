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

fn colmap_uint8_l2_distance2(left: &[u8; DESCRIPTOR_DIM], right: &[u8; DESCRIPTOR_DIM]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| {
            let delta = i32::from(*a) - i32::from(*b);
            (delta * delta) as f32
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

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
