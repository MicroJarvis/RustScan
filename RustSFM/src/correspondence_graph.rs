use nalgebra::Matrix3;
use rustslam::Match;
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub type ImageId = u32;
pub type Point2DIdx = u32;
pub type ImagePairId = u64;

pub const INVALID_IMAGE_ID: ImageId = u32::MAX;
pub const INVALID_POINT2D_IDX: Point2DIdx = u32::MAX;
pub const INVALID_IMAGE_PAIR_ID: ImagePairId = u64::MAX;
pub const MAX_NUM_IMAGES: ImagePairId = i32::MAX as ImagePairId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Correspondence {
    pub image_id: ImageId,
    pub point2d_idx: Point2DIdx,
}

impl Correspondence {
    pub fn new(image_id: ImageId, point2d_idx: Point2DIdx) -> Self {
        Self {
            image_id,
            point2d_idx,
        }
    }
}

impl Default for Correspondence {
    fn default() -> Self {
        Self {
            image_id: INVALID_IMAGE_ID,
            point2d_idx: INVALID_POINT2D_IDX,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeatureMatch {
    pub point2d_idx1: Point2DIdx,
    pub point2d_idx2: Point2DIdx,
}

impl FeatureMatch {
    pub fn new(point2d_idx1: Point2DIdx, point2d_idx2: Point2DIdx) -> Self {
        Self {
            point2d_idx1,
            point2d_idx2,
        }
    }
}

impl From<&Match> for FeatureMatch {
    fn from(value: &Match) -> Self {
        Self {
            point2d_idx1: value.query_idx,
            point2d_idx2: value.train_idx,
        }
    }
}

impl From<FeatureMatch> for Match {
    fn from(value: FeatureMatch) -> Self {
        Self {
            query_idx: value.point2d_idx1,
            train_idx: value.point2d_idx2,
            distance: 0.0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TwoViewGeometryRecord {
    pub config: i32,
    pub inlier_matches: Vec<FeatureMatch>,
    pub f_matrix: Option<[f64; 9]>,
    pub e_matrix: Option<[f64; 9]>,
    pub h_matrix: Option<[f64; 9]>,
    pub qvec: Option<[f64; 4]>,
    pub tvec: Option<[f64; 3]>,
}

impl TwoViewGeometryRecord {
    pub fn with_inlier_matches(inlier_matches: Vec<FeatureMatch>) -> Self {
        Self {
            config: 0,
            inlier_matches,
            ..Self::default()
        }
    }

    pub fn invert(&mut self) {
        self.f_matrix = self.f_matrix.map(transpose3);
        self.e_matrix = self.e_matrix.map(transpose3);
        self.h_matrix = self.h_matrix.and_then(invert_matrix3);
        if let (Some(qvec), Some(tvec)) = (self.qvec, self.tvec) {
            let rotation = glam::DQuat::from_xyzw(qvec[1], qvec[2], qvec[3], qvec[0]).normalize();
            let translation = glam::DVec3::from_array(tvec);
            let inverse_rotation = rotation.inverse();
            let inverse_translation = -(inverse_rotation * translation);
            self.qvec = Some([
                inverse_rotation.w,
                inverse_rotation.x,
                inverse_rotation.y,
                inverse_rotation.z,
            ]);
            self.tvec = Some(inverse_translation.to_array());
        }
        for match_ in &mut self.inlier_matches {
            std::mem::swap(&mut match_.point2d_idx1, &mut match_.point2d_idx2);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorrespondenceGraphError {
    ImageAlreadyExists(ImageId),
    ImageDoesNotExist(ImageId),
    ImagePairAlreadyExists(ImageId, ImageId),
    ImagePairDoesNotExist(ImageId, ImageId),
    ImageIdExceedsMax(ImageId),
    AlreadyFinalized,
}

pub type Result<T> = std::result::Result<T, CorrespondenceGraphError>;

#[derive(Debug, Default, Clone)]
pub struct CorrespondenceGraph {
    finalized: bool,
    images: HashMap<ImageId, ImageData>,
    image_pairs: HashMap<ImagePairId, ImagePairData>,
}

#[derive(Debug, Default, Clone)]
struct ImageData {
    num_observations: Point2DIdx,
    num_correspondences: Point2DIdx,
    corrs: Vec<Vec<Correspondence>>,
    flat_corrs: Vec<Correspondence>,
    flat_corr_begs: Vec<usize>,
}

#[derive(Debug, Default, Clone)]
struct ImagePairData {
    num_matches: Point2DIdx,
    two_view_geometry: TwoViewGeometryRecord,
}

pub fn should_swap_image_pair(image_id1: ImageId, image_id2: ImageId) -> bool {
    image_id1 > image_id2
}

pub fn image_pair_to_pair_id(image_id1: ImageId, image_id2: ImageId) -> Result<ImagePairId> {
    throw_if_gt_max_images(image_id1)?;
    throw_if_gt_max_images(image_id2)?;
    let image_id1 = image_id1 as ImagePairId;
    let image_id2 = image_id2 as ImagePairId;
    if image_id1 > image_id2 {
        Ok(MAX_NUM_IMAGES * image_id2 + image_id1)
    } else {
        Ok(MAX_NUM_IMAGES * image_id1 + image_id2)
    }
}

pub fn pair_id_to_image_pair(pair_id: ImagePairId) -> Result<(ImageId, ImageId)> {
    let image_id2 = (pair_id % MAX_NUM_IMAGES) as ImageId;
    let image_id1 = ((pair_id - image_id2 as ImagePairId) / MAX_NUM_IMAGES) as ImageId;
    throw_if_gt_max_images(image_id1)?;
    throw_if_gt_max_images(image_id2)?;
    Ok((image_id1, image_id2))
}

fn throw_if_gt_max_images(image_id: ImageId) -> Result<()> {
    if image_id as ImagePairId >= MAX_NUM_IMAGES {
        Err(CorrespondenceGraphError::ImageIdExceedsMax(image_id))
    } else {
        Ok(())
    }
}

impl CorrespondenceGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn finalize(&mut self) -> Result<()> {
        if self.finalized {
            return Err(CorrespondenceGraphError::AlreadyFinalized);
        }
        self.finalized = true;

        for image in self.images.values_mut() {
            let mut num_total_corrs = 0usize;
            let mut expected_num_observations = 0u32;
            for corrs in &image.corrs {
                num_total_corrs += corrs.len();
                if !corrs.is_empty() {
                    expected_num_observations += 1;
                }
            }
            debug_assert_eq!(image.num_observations, expected_num_observations);

            let num_points2d = image.corrs.len();
            image.flat_corrs.reserve(num_total_corrs);
            image.flat_corr_begs.resize(num_points2d + 1, 0);
            for point2d_idx in 0..num_points2d {
                image.flat_corr_begs[point2d_idx] = image.flat_corrs.len();
                image
                    .flat_corrs
                    .extend_from_slice(&image.corrs[point2d_idx]);
            }
            image.flat_corr_begs[num_points2d] = image.flat_corrs.len();
            image.corrs.clear();
            image.corrs.shrink_to_fit();
        }

        Ok(())
    }

    pub fn num_images(&self) -> usize {
        self.images.len()
    }

    pub fn num_image_pairs(&self) -> usize {
        self.image_pairs.len()
    }

    pub fn exists_image(&self, image_id: ImageId) -> bool {
        self.images.contains_key(&image_id)
    }

    pub fn image_pairs(&self) -> Vec<ImagePairId> {
        self.image_pairs.keys().copied().collect()
    }

    pub fn add_image(&mut self, image_id: ImageId, num_points2d: usize) -> Result<()> {
        if self.finalized {
            return Err(CorrespondenceGraphError::AlreadyFinalized);
        }
        if self.exists_image(image_id) {
            return Err(CorrespondenceGraphError::ImageAlreadyExists(image_id));
        }
        self.images.insert(
            image_id,
            ImageData {
                corrs: vec![Vec::new(); num_points2d],
                ..ImageData::default()
            },
        );
        Ok(())
    }

    pub fn add_two_view_geometry(
        &mut self,
        image_id1: ImageId,
        image_id2: ImageId,
        mut two_view_geometry: TwoViewGeometryRecord,
    ) -> Result<()> {
        if self.finalized {
            return Err(CorrespondenceGraphError::AlreadyFinalized);
        }
        if image_id1 == image_id2 {
            return Ok(());
        }
        if !self.exists_image(image_id1) {
            return Err(CorrespondenceGraphError::ImageDoesNotExist(image_id1));
        }
        if !self.exists_image(image_id2) {
            return Err(CorrespondenceGraphError::ImageDoesNotExist(image_id2));
        }

        let pair_id = image_pair_to_pair_id(image_id1, image_id2)?;
        if self.image_pairs.contains_key(&pair_id) {
            return Err(CorrespondenceGraphError::ImagePairAlreadyExists(
                image_id1, image_id2,
            ));
        }

        let mut image1 = self
            .images
            .remove(&image_id1)
            .expect("checked image exists");
        let mut image2 = self
            .images
            .remove(&image_id2)
            .expect("checked image exists");
        let match_count = saturating_point2d_len(two_view_geometry.inlier_matches.len());
        image1.num_correspondences = image1.num_correspondences.saturating_add(match_count);
        image2.num_correspondences = image2.num_correspondences.saturating_add(match_count);
        let mut num_matches = match_count;

        for match_ in &two_view_geometry.inlier_matches {
            let idx1 = match_.point2d_idx1 as usize;
            let idx2 = match_.point2d_idx2 as usize;
            let valid_idx1 = idx1 < image1.corrs.len();
            let valid_idx2 = idx2 < image2.corrs.len();

            if valid_idx1 && valid_idx2 {
                let duplicate = image1.corrs[idx1].iter().any(|corr| {
                    corr.image_id == image_id2 && corr.point2d_idx == match_.point2d_idx2
                });
                if duplicate {
                    image1.num_correspondences = image1.num_correspondences.saturating_sub(1);
                    image2.num_correspondences = image2.num_correspondences.saturating_sub(1);
                    num_matches = num_matches.saturating_sub(1);
                } else {
                    image1.corrs[idx1].push(Correspondence::new(image_id2, match_.point2d_idx2));
                    if image1.corrs[idx1].len() == 1 {
                        image1.num_observations = image1.num_observations.saturating_add(1);
                    }
                    image2.corrs[idx2].push(Correspondence::new(image_id1, match_.point2d_idx1));
                    if image2.corrs[idx2].len() == 1 {
                        image2.num_observations = image2.num_observations.saturating_add(1);
                    }
                }
            } else {
                image1.num_correspondences = image1.num_correspondences.saturating_sub(1);
                image2.num_correspondences = image2.num_correspondences.saturating_sub(1);
                num_matches = num_matches.saturating_sub(1);
            }
        }

        self.images.insert(image_id1, image1);
        self.images.insert(image_id2, image2);
        two_view_geometry.inlier_matches.clear();
        if should_swap_image_pair(image_id1, image_id2) {
            two_view_geometry.invert();
        }
        self.image_pairs.insert(
            pair_id,
            ImagePairData {
                num_matches,
                two_view_geometry,
            },
        );
        Ok(())
    }

    pub fn num_observations_for_image(&self, image_id: ImageId) -> Result<Point2DIdx> {
        Ok(self.image(image_id)?.num_observations)
    }

    pub fn num_correspondences_for_image(&self, image_id: ImageId) -> Result<Point2DIdx> {
        Ok(self.image(image_id)?.num_correspondences)
    }

    pub fn num_points2d_for_image(&self, image_id: ImageId) -> Result<usize> {
        let image = self.image(image_id)?;
        if self.finalized {
            Ok(image.flat_corr_begs.len().saturating_sub(1))
        } else {
            Ok(image.corrs.len())
        }
    }

    pub fn num_matches_between_images(
        &self,
        image_id1: ImageId,
        image_id2: ImageId,
    ) -> Result<Point2DIdx> {
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)?;
        Ok(self
            .image_pairs
            .get(&pair_id)
            .map(|pair| pair.num_matches)
            .unwrap_or(0))
    }

    pub fn num_matches_between_all_images(&self) -> HashMap<ImagePairId, Point2DIdx> {
        self.image_pairs
            .iter()
            .map(|(&pair_id, pair)| (pair_id, pair.num_matches))
            .collect()
    }

    pub fn find_correspondences(
        &self,
        image_id: ImageId,
        point2d_idx: Point2DIdx,
    ) -> Result<&[Correspondence]> {
        let image = self.image(image_id)?;
        let idx = point2d_idx as usize;
        if self.finalized {
            let begin = *image
                .flat_corr_begs
                .get(idx)
                .ok_or(CorrespondenceGraphError::ImageDoesNotExist(image_id))?;
            let end = *image
                .flat_corr_begs
                .get(idx + 1)
                .ok_or(CorrespondenceGraphError::ImageDoesNotExist(image_id))?;
            Ok(&image.flat_corrs[begin..end])
        } else {
            Ok(image
                .corrs
                .get(idx)
                .ok_or(CorrespondenceGraphError::ImageDoesNotExist(image_id))?
                .as_slice())
        }
    }

    pub fn extract_correspondences(
        &self,
        image_id: ImageId,
        point2d_idx: Point2DIdx,
    ) -> Result<Vec<Correspondence>> {
        Ok(self.find_correspondences(image_id, point2d_idx)?.to_vec())
    }

    pub fn extract_transitive_correspondences(
        &self,
        image_id: ImageId,
        point2d_idx: Point2DIdx,
        transitivity: usize,
    ) -> Result<Vec<Correspondence>> {
        if transitivity == 1 {
            return self.extract_correspondences(image_id, point2d_idx);
        }
        if !self.has_correspondences(image_id, point2d_idx)? {
            return Ok(Vec::new());
        }

        let mut corrs = vec![Correspondence::new(image_id, point2d_idx)];
        let mut image_corrs = BTreeMap::<ImageId, BTreeSet<Point2DIdx>>::new();
        image_corrs.entry(image_id).or_default().insert(point2d_idx);

        let mut corr_queue_beg = 0usize;
        let mut corr_queue_end = 1usize;
        for _ in 0..transitivity {
            let refs: Vec<_> = corrs[corr_queue_beg..corr_queue_end].to_vec();
            for ref_corr in refs {
                for corr in self.find_correspondences(ref_corr.image_id, ref_corr.point2d_idx)? {
                    let entry = image_corrs.entry(corr.image_id).or_default();
                    if entry.insert(corr.point2d_idx) {
                        corrs.push(*corr);
                    }
                }
            }
            corr_queue_beg = corr_queue_end;
            corr_queue_end = corrs.len();
            if corr_queue_beg == corr_queue_end {
                break;
            }
        }

        if corrs.len() > 1 {
            let last = *corrs.last().expect("non-empty");
            corrs[0] = last;
        }
        corrs.pop();
        Ok(corrs)
    }

    pub fn extract_matches_between_images(
        &self,
        image_id1: ImageId,
        image_id2: ImageId,
    ) -> Result<Vec<FeatureMatch>> {
        if self.num_matches_between_images(image_id1, image_id2)? == 0 {
            return Ok(Vec::new());
        }

        let image1 = self.image(image_id1)?;
        let num_points2d1 = if self.finalized {
            image1.flat_corr_begs.len().saturating_sub(1)
        } else {
            image1.corrs.len()
        };
        let mut matches = Vec::new();
        for point2d_idx1 in 0..num_points2d1 {
            for corr in self.find_correspondences(image_id1, point2d_idx1 as Point2DIdx)? {
                if corr.image_id == image_id2 {
                    matches.push(FeatureMatch::new(
                        point2d_idx1 as Point2DIdx,
                        corr.point2d_idx,
                    ));
                }
            }
        }
        Ok(matches)
    }

    pub fn extract_two_view_geometry(
        &self,
        image_id1: ImageId,
        image_id2: ImageId,
        extract_inlier_matches: bool,
    ) -> Result<TwoViewGeometryRecord> {
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)?;
        let pair = self.image_pairs.get(&pair_id).ok_or(
            CorrespondenceGraphError::ImagePairDoesNotExist(image_id1, image_id2),
        )?;
        let mut two_view_geometry = pair.two_view_geometry.clone();
        if should_swap_image_pair(image_id1, image_id2) {
            two_view_geometry.invert();
        }
        if extract_inlier_matches {
            two_view_geometry.inlier_matches =
                self.extract_matches_between_images(image_id1, image_id2)?;
        }
        Ok(two_view_geometry)
    }

    pub fn update_two_view_geometry(
        &mut self,
        image_id1: ImageId,
        image_id2: ImageId,
        mut two_view_geometry: TwoViewGeometryRecord,
    ) -> Result<()> {
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)?;
        let pair = self.image_pairs.get_mut(&pair_id).ok_or(
            CorrespondenceGraphError::ImagePairDoesNotExist(image_id1, image_id2),
        )?;
        two_view_geometry.inlier_matches.clear();
        if should_swap_image_pair(image_id1, image_id2) {
            two_view_geometry.invert();
        }
        pair.two_view_geometry = two_view_geometry;
        Ok(())
    }

    pub fn has_correspondences(&self, image_id: ImageId, point2d_idx: Point2DIdx) -> Result<bool> {
        Ok(!self.find_correspondences(image_id, point2d_idx)?.is_empty())
    }

    pub fn is_two_view_observation(
        &self,
        image_id: ImageId,
        point2d_idx: Point2DIdx,
    ) -> Result<bool> {
        let range = self.find_correspondences(image_id, point2d_idx)?;
        if range.len() != 1 {
            return Ok(false);
        }
        let other_range = self.find_correspondences(range[0].image_id, range[0].point2d_idx)?;
        Ok(other_range.len() == 1)
    }

    fn image(&self, image_id: ImageId) -> Result<&ImageData> {
        self.images
            .get(&image_id)
            .ok_or(CorrespondenceGraphError::ImageDoesNotExist(image_id))
    }
}

fn saturating_point2d_len(len: usize) -> Point2DIdx {
    len.min(Point2DIdx::MAX as usize) as Point2DIdx
}

fn transpose3(matrix: [f64; 9]) -> [f64; 9] {
    [
        matrix[0], matrix[3], matrix[6], matrix[1], matrix[4], matrix[7], matrix[2], matrix[5],
        matrix[8],
    ]
}

fn invert_matrix3(matrix: [f64; 9]) -> Option<[f64; 9]> {
    let inverse = Matrix3::from_row_slice(&matrix).try_inverse()?;
    Some([
        inverse[(0, 0)],
        inverse[(0, 1)],
        inverse[(0, 2)],
        inverse[(1, 0)],
        inverse[(1, 1)],
        inverse[(1, 2)],
        inverse[(2, 0)],
        inverse[(2, 1)],
        inverse[(2, 2)],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(point2d_idx1: Point2DIdx, point2d_idx2: Point2DIdx) -> FeatureMatch {
        FeatureMatch::new(point2d_idx1, point2d_idx2)
    }

    #[test]
    fn pair_id_matches_colmap_formula() {
        assert_eq!(image_pair_to_pair_id(0, 0).unwrap(), 0);
        assert_eq!(image_pair_to_pair_id(0, 1).unwrap(), 1);
        assert_eq!(image_pair_to_pair_id(0, 2).unwrap(), 2);
        assert_eq!(image_pair_to_pair_id(0, 3).unwrap(), 3);
        assert_eq!(image_pair_to_pair_id(1, 2).unwrap(), MAX_NUM_IMAGES + 2);
        assert_eq!(
            image_pair_to_pair_id(2, 1).unwrap(),
            image_pair_to_pair_id(1, 2).unwrap()
        );

        for i in 0..10 {
            for j in 0..10 {
                let pair_id = image_pair_to_pair_id(i, j).unwrap();
                assert_eq!(
                    pair_id_to_image_pair(pair_id).unwrap(),
                    (i.min(j), i.max(j))
                );
            }
        }
        assert_eq!(
            image_pair_to_pair_id(i32::MAX as u32, 1),
            Err(CorrespondenceGraphError::ImageIdExceedsMax(i32::MAX as u32))
        );
    }

    #[test]
    fn add_geometry_filters_duplicate_and_out_of_bounds_matches() {
        let mut graph = CorrespondenceGraph::new();
        graph.add_image(1, 3).unwrap();
        graph.add_image(2, 3).unwrap();

        graph
            .add_two_view_geometry(
                1,
                2,
                TwoViewGeometryRecord::with_inlier_matches(vec![
                    m(0, 0),
                    m(0, 0),
                    m(1, 2),
                    m(3, 0),
                    m(2, 3),
                ]),
            )
            .unwrap();

        assert_eq!(graph.num_image_pairs(), 1);
        assert_eq!(graph.num_observations_for_image(1).unwrap(), 2);
        assert_eq!(graph.num_observations_for_image(2).unwrap(), 2);
        assert_eq!(graph.num_correspondences_for_image(1).unwrap(), 2);
        assert_eq!(graph.num_correspondences_for_image(2).unwrap(), 2);
        assert_eq!(graph.num_matches_between_images(1, 2).unwrap(), 2);
        assert_eq!(
            graph.extract_matches_between_images(1, 2).unwrap(),
            vec![m(0, 0), m(1, 2)]
        );
        assert_eq!(
            graph.extract_matches_between_images(2, 1).unwrap(),
            vec![m(0, 0), m(2, 1)]
        );
    }

    #[test]
    fn finalize_preserves_queries_and_rejects_second_finalize() {
        let mut graph = CorrespondenceGraph::new();
        graph.add_image(1, 2).unwrap();
        graph.add_image(2, 2).unwrap();
        graph
            .add_two_view_geometry(
                1,
                2,
                TwoViewGeometryRecord::with_inlier_matches(vec![m(0, 1), m(1, 0)]),
            )
            .unwrap();

        assert_eq!(
            graph.find_correspondences(1, 0).unwrap(),
            &[Correspondence::new(2, 1)]
        );
        graph.finalize().unwrap();
        assert_eq!(
            graph.find_correspondences(1, 0).unwrap(),
            &[Correspondence::new(2, 1)]
        );
        assert_eq!(
            graph.extract_matches_between_images(1, 2).unwrap(),
            vec![m(0, 1), m(1, 0)]
        );
        assert_eq!(
            graph.finalize(),
            Err(CorrespondenceGraphError::AlreadyFinalized)
        );
    }

    #[test]
    fn transitive_correspondences_follow_colmap_queue_semantics() {
        let mut graph = CorrespondenceGraph::new();
        graph.add_image(1, 2).unwrap();
        graph.add_image(2, 2).unwrap();
        graph.add_image(3, 2).unwrap();
        graph
            .add_two_view_geometry(
                1,
                2,
                TwoViewGeometryRecord::with_inlier_matches(vec![m(0, 0)]),
            )
            .unwrap();
        graph
            .add_two_view_geometry(
                2,
                3,
                TwoViewGeometryRecord::with_inlier_matches(vec![m(0, 1)]),
            )
            .unwrap();

        assert_eq!(
            graph.extract_transitive_correspondences(1, 0, 1).unwrap(),
            vec![Correspondence::new(2, 0)]
        );
        let mut transitive = graph.extract_transitive_correspondences(1, 0, 2).unwrap();
        transitive.sort();
        assert_eq!(
            transitive,
            vec![Correspondence::new(2, 0), Correspondence::new(3, 1)]
        );
    }

    #[test]
    fn two_view_observation_requires_singleton_on_both_sides() {
        let mut graph = CorrespondenceGraph::new();
        graph.add_image(1, 2).unwrap();
        graph.add_image(2, 2).unwrap();
        graph.add_image(3, 2).unwrap();
        graph
            .add_two_view_geometry(
                1,
                2,
                TwoViewGeometryRecord::with_inlier_matches(vec![m(0, 0), m(1, 1)]),
            )
            .unwrap();
        graph
            .add_two_view_geometry(
                1,
                3,
                TwoViewGeometryRecord::with_inlier_matches(vec![m(0, 1)]),
            )
            .unwrap();

        assert!(!graph.is_two_view_observation(1, 0).unwrap());
        assert!(graph.is_two_view_observation(1, 1).unwrap());
        assert!(graph.is_two_view_observation(2, 1).unwrap());
    }

    #[test]
    fn geometry_storage_keeps_matches_separate_like_colmap() {
        let mut graph = CorrespondenceGraph::new();
        graph.add_image(1, 2).unwrap();
        graph.add_image(2, 2).unwrap();
        let f_matrix = Some([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let h_matrix = Some([1.0, 0.0, 2.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0]);
        graph
            .add_two_view_geometry(
                2,
                1,
                TwoViewGeometryRecord {
                    config: 7,
                    inlier_matches: vec![m(1, 0)],
                    f_matrix,
                    h_matrix,
                    qvec: Some([1.0, 0.0, 0.0, 0.0]),
                    tvec: Some([0.3, 0.4, 0.5]),
                    ..TwoViewGeometryRecord::default()
                },
            )
            .unwrap();

        let without_matches = graph.extract_two_view_geometry(2, 1, false).unwrap();
        assert_eq!(without_matches.config, 7);
        assert!(without_matches.inlier_matches.is_empty());
        assert_eq!(without_matches.f_matrix, f_matrix);
        assert_eq!(without_matches.h_matrix, h_matrix);
        assert_eq!(without_matches.tvec, Some([0.3, 0.4, 0.5]));

        let sorted_direction = graph.extract_two_view_geometry(1, 2, false).unwrap();
        assert_eq!(sorted_direction.f_matrix, f_matrix.map(transpose3));
        assert!(sorted_direction.h_matrix.is_some());
        assert_eq!(sorted_direction.tvec, Some([-0.3, -0.4, -0.5]));

        let with_matches = graph.extract_two_view_geometry(2, 1, true).unwrap();
        assert_eq!(with_matches.inlier_matches, vec![m(1, 0)]);

        graph
            .update_two_view_geometry(
                2,
                1,
                TwoViewGeometryRecord {
                    config: 9,
                    inlier_matches: vec![m(0, 0)],
                    e_matrix: Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
                    ..TwoViewGeometryRecord::default()
                },
            )
            .unwrap();
        let updated = graph.extract_two_view_geometry(2, 1, true).unwrap();
        assert_eq!(updated.config, 9);
        assert_eq!(updated.inlier_matches, vec![m(1, 0)]);
        assert_eq!(
            updated.e_matrix,
            Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        );
        assert_eq!(updated.f_matrix, None);
    }
}
