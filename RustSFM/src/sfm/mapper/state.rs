use super::{initial_image_pair_id, registration_unit_key, MapperConfig, RegistrationUnitKey};
use crate::correspondence_graph::ImagePairId;
use crate::types::Reconstruction;
use std::collections::{HashMap, HashSet};
use std::fmt;

#[derive(Debug, Clone, Default)]
pub(super) struct RegistrationStats {
    pub(super) init_num_reg_trials: Vec<usize>,
    pub(super) init_image_pairs: HashSet<ImagePairId>,
    pub(super) existing_registration_units: HashSet<RegistrationUnitKey>,
    pub(super) num_reg_frames_per_rig: HashMap<u32, usize>,
    pub(super) num_reg_images_per_camera: HashMap<u32, usize>,
    pub(super) num_registrations: Vec<usize>,
    pub(super) num_total_reg_images: usize,
    pub(super) num_shared_reg_images: usize,
}

impl RegistrationStats {
    pub(super) fn from_reconstruction(reconstruction: &Reconstruction) -> Self {
        let mut stats = Self {
            init_num_reg_trials: vec![0; reconstruction.poses.len()],
            num_registrations: vec![0; reconstruction.poses.len()],
            ..Self::default()
        };
        let mut registered_frames = HashSet::new();
        for image in 0..reconstruction.poses.len() {
            if reconstruction.poses[image].is_none() {
                continue;
            }
            if let Some(frame_idx) = reconstruction.frame_index_for_image(image) {
                if registered_frames.insert(frame_idx) {
                    stats.register_frame_event(reconstruction, frame_idx);
                }
            } else {
                stats.register_image_event(reconstruction, image);
            }
        }
        stats
    }

    pub(super) fn register_image_event(&mut self, reconstruction: &Reconstruction, image: usize) {
        if image >= reconstruction.poses.len() {
            return;
        }
        let camera_id = reconstruction.camera_id_for_image(image);
        *self.num_reg_images_per_camera.entry(camera_id).or_default() += 1;
        if image >= self.num_registrations.len() {
            self.num_registrations.resize(image + 1, 0);
        }
        self.num_registrations[image] += 1;
        if self.num_registrations[image] == 1 {
            self.num_total_reg_images += 1;
        } else {
            self.num_shared_reg_images += 1;
        }
    }

    #[allow(dead_code)]
    pub(super) fn deregister_image_event(&mut self, reconstruction: &Reconstruction, image: usize) {
        if image >= reconstruction.poses.len() {
            return;
        }
        let camera_id = reconstruction.camera_id_for_image(image);
        if let Some(count) = self.num_reg_images_per_camera.get_mut(&camera_id) {
            *count = count.saturating_sub(1);
        }
        if let Some(registrations) = self.num_registrations.get_mut(image) {
            if *registrations > 0 {
                *registrations -= 1;
                if *registrations == 0 {
                    self.num_total_reg_images = self.num_total_reg_images.saturating_sub(1);
                } else {
                    self.num_shared_reg_images = self.num_shared_reg_images.saturating_sub(1);
                }
            }
        }
    }

    pub(super) fn register_frame_for_image_event(
        &mut self,
        reconstruction: &Reconstruction,
        image: usize,
    ) {
        if let Some(frame_idx) = reconstruction.frame_index_for_image(image) {
            self.register_frame_event(reconstruction, frame_idx);
        } else {
            self.register_image_event(reconstruction, image);
        }
    }

    #[allow(dead_code)]
    pub(super) fn deregister_frame_for_image_event(
        &mut self,
        reconstruction: &Reconstruction,
        image: usize,
    ) {
        if let Some(frame_idx) = reconstruction.frame_index_for_image(image) {
            self.deregister_frame_event(reconstruction, frame_idx);
        } else {
            self.deregister_image_event(reconstruction, image);
        }
    }

    pub(super) fn register_frame_event(
        &mut self,
        reconstruction: &Reconstruction,
        frame_idx: usize,
    ) {
        let Some(frame) = reconstruction.frames.get(frame_idx) else {
            return;
        };
        *self.num_reg_frames_per_rig.entry(frame.rig_id).or_default() += 1;
        for image in reconstruction.image_indices_for_frame_index(frame_idx) {
            self.register_image_event(reconstruction, image);
        }
    }

    #[allow(dead_code)]
    pub(super) fn deregister_frame_event(
        &mut self,
        reconstruction: &Reconstruction,
        frame_idx: usize,
    ) {
        let Some(frame) = reconstruction.frames.get(frame_idx) else {
            return;
        };
        if let Some(count) = self.num_reg_frames_per_rig.get_mut(&frame.rig_id) {
            *count = count.saturating_sub(1);
        }
        for image in reconstruction.image_indices_for_frame_index(frame_idx) {
            self.deregister_image_event(reconstruction, image);
        }
    }

    pub(super) fn registered_images_with_camera_id(&self, camera_id: u32) -> usize {
        self.num_reg_images_per_camera
            .get(&camera_id)
            .copied()
            .unwrap_or(0)
    }

    pub(super) fn set_initial_pair_selection_state(&mut self, state: InitialPairSelectionState) {
        self.init_num_reg_trials = state.init_num_reg_trials;
        self.init_image_pairs = state.init_image_pairs;
    }

    pub(super) fn set_existing_registration_units_from_reconstruction(
        &mut self,
        reconstruction: &Reconstruction,
    ) {
        self.existing_registration_units.clear();
        for image in 0..reconstruction.poses.len() {
            if reconstruction.poses[image].is_none() {
                continue;
            }
            self.existing_registration_units
                .insert(registration_unit_key(reconstruction, image));
        }
    }

    pub(super) fn is_existing_registration_unit(
        &self,
        reconstruction: &Reconstruction,
        image: usize,
    ) -> bool {
        self.existing_registration_units
            .contains(&registration_unit_key(reconstruction, image))
    }

    pub(super) fn existing_registered_images(&self, reconstruction: &Reconstruction) -> Vec<usize> {
        let mut images = Vec::new();
        for image in 0..reconstruction.poses.len() {
            if reconstruction.poses[image].is_some()
                && self.is_existing_registration_unit(reconstruction, image)
            {
                images.push(image);
            }
        }
        images
    }
}

#[derive(Debug, Clone, Default)]
pub(super) struct InitialPairSelectionState {
    pub(super) init_num_reg_trials: Vec<usize>,
    pub(super) num_registrations: Vec<usize>,
    pub(super) init_image_pairs: HashSet<ImagePairId>,
}

impl InitialPairSelectionState {
    pub(super) fn from_reconstruction(reconstruction: &Reconstruction) -> Self {
        Self {
            init_num_reg_trials: vec![0; reconstruction.poses.len()],
            num_registrations: vec![0; reconstruction.poses.len()],
            init_image_pairs: HashSet::new(),
        }
    }

    pub(super) fn init_trials_for_image(&self, image: usize) -> usize {
        self.init_num_reg_trials.get(image).copied().unwrap_or(0)
    }

    pub(super) fn num_registrations_for_image(&self, image: usize) -> usize {
        self.num_registrations.get(image).copied().unwrap_or(0)
    }

    pub(super) fn first_image_available_for_initialization(
        &self,
        image: usize,
        config: &MapperConfig,
    ) -> bool {
        self.init_trials_for_image(image) < config.init_max_reg_trials
            && self.num_registrations_for_image(image) == 0
    }

    pub(super) fn image_not_registered_in_other_reconstruction(&self, image: usize) -> bool {
        self.num_registrations_for_image(image) == 0
    }

    pub(super) fn mark_initial_pair_tried(
        &mut self,
        reconstruction: &Reconstruction,
        left: usize,
        right: usize,
    ) -> bool {
        let Some(pair_id) = initial_image_pair_id(reconstruction, left, right) else {
            return false;
        };
        self.init_image_pairs.insert(pair_id)
    }

    pub(super) fn register_initial_pair(
        &mut self,
        reconstruction: &Reconstruction,
        left: usize,
        right: usize,
    ) {
        self.increment_init_trial(left);
        self.increment_init_trial(right);
        self.mark_initial_pair_tried(reconstruction, left, right);
    }

    pub(super) fn increment_init_trial(&mut self, image: usize) {
        if image >= self.init_num_reg_trials.len() {
            self.init_num_reg_trials.resize(image + 1, 0);
        }
        self.init_num_reg_trials[image] += 1;
    }
}

#[derive(Debug, Clone, Default)]
pub(super) struct IncrementalMapperSession {
    init_num_reg_trials: HashMap<u32, usize>,
    init_image_pairs: HashSet<ImagePairId>,
    num_registrations: HashMap<u32, usize>,
    current_num_shared_reg_images: usize,
    current_reconstruction_committed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitialPairFailure {
    NoInitialPair,
    BadInitialPair,
}

impl fmt::Display for InitialPairFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoInitialPair => f.write_str("no initial pair"),
            Self::BadInitialPair => f.write_str("bad initial pair"),
        }
    }
}

impl std::error::Error for InitialPairFailure {}

impl IncrementalMapperSession {
    pub(super) fn reset_initialization_stats(&mut self) {
        self.init_num_reg_trials.clear();
        self.init_image_pairs.clear();
    }

    pub(super) fn num_total_registered_images(&self) -> usize {
        self.num_registrations
            .values()
            .filter(|&&count| count > 0)
            .count()
    }

    pub(super) fn num_shared_registered_image_events(&self) -> usize {
        self.current_num_shared_reg_images
    }

    pub(super) fn begin_reconstruction(&mut self, reconstruction: &Reconstruction) {
        self.current_num_shared_reg_images =
            self.num_current_reconstruction_images_with_registration_count(reconstruction, 1);
        self.current_reconstruction_committed = false;
    }

    fn num_current_reconstruction_images_with_registration_count(
        &self,
        reconstruction: &Reconstruction,
        min_count: usize,
    ) -> usize {
        reconstruction
            .poses
            .iter()
            .enumerate()
            .filter(|(_, pose)| pose.is_some())
            .filter(|(image, _)| {
                self.num_registrations
                    .get(&reconstruction.image_id(*image))
                    .copied()
                    .unwrap_or(0)
                    >= min_count
            })
            .count()
    }

    pub(super) fn initial_pair_selection_state(
        &self,
        reconstruction: &Reconstruction,
    ) -> InitialPairSelectionState {
        InitialPairSelectionState {
            init_num_reg_trials: (0..reconstruction.poses.len())
                .map(|image| {
                    self.init_num_reg_trials
                        .get(&reconstruction.image_id(image))
                        .copied()
                        .unwrap_or(0)
                })
                .collect(),
            num_registrations: (0..reconstruction.poses.len())
                .map(|image| {
                    self.num_registrations
                        .get(&reconstruction.image_id(image))
                        .copied()
                        .unwrap_or(0)
                })
                .collect(),
            init_image_pairs: self.init_image_pairs.clone(),
        }
    }

    pub(super) fn commit_initial_pair_selection_state(
        &mut self,
        reconstruction: &Reconstruction,
        state: &InitialPairSelectionState,
    ) {
        for image in 0..reconstruction.poses.len() {
            let image_id = reconstruction.image_id(image);
            match state.init_num_reg_trials.get(image).copied().unwrap_or(0) {
                0 => {
                    self.init_num_reg_trials.remove(&image_id);
                }
                trials => {
                    self.init_num_reg_trials.insert(image_id, trials);
                }
            }
        }
        self.init_image_pairs = state.init_image_pairs.clone();
    }

    pub(super) fn end_reconstruction(&mut self, reconstruction: &Reconstruction, discard: bool) {
        if discard && !self.current_reconstruction_committed {
            self.current_num_shared_reg_images = 0;
            return;
        }
        let mut registered_frames = HashSet::new();
        for image in 0..reconstruction.poses.len() {
            if reconstruction.poses[image].is_none() {
                continue;
            }
            if let Some(frame_idx) = reconstruction.frame_index_for_image(image) {
                if registered_frames.insert(frame_idx) {
                    self.apply_frame_registration_event(reconstruction, frame_idx, discard);
                }
            } else {
                self.apply_image_registration_event(reconstruction, image, discard);
            }
        }
        self.current_num_shared_reg_images = if discard {
            0
        } else {
            self.num_current_reconstruction_images_with_registration_count(reconstruction, 2)
        };
        self.current_reconstruction_committed = !discard;
    }

    fn apply_frame_registration_event(
        &mut self,
        reconstruction: &Reconstruction,
        frame_idx: usize,
        discard: bool,
    ) {
        for image in reconstruction.image_indices_for_frame_index(frame_idx) {
            self.apply_image_registration_event(reconstruction, image, discard);
        }
    }

    fn apply_image_registration_event(
        &mut self,
        reconstruction: &Reconstruction,
        image: usize,
        discard: bool,
    ) {
        let image_id = reconstruction.image_id(image);
        if discard {
            match self.num_registrations.get_mut(&image_id) {
                Some(count) if *count > 1 => *count -= 1,
                Some(_) => {
                    self.num_registrations.remove(&image_id);
                }
                None => {}
            }
        } else {
            let count = self.num_registrations.entry(image_id).or_default();
            *count += 1;
        }
    }
}
