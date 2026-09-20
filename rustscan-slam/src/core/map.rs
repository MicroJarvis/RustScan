//! Map representation

use crate::core::{KeyFrame, MapPoint};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// The map containing all map points and keyframes
#[derive(Debug, Default)]
pub struct Map {
    /// All map points indexed by ID
    points: HashMap<u64, MapPoint>,
    /// All keyframes indexed by ID
    keyframes: HashMap<u64, KeyFrame>,
    /// Next available map point ID (atomic for thread safety)
    next_point_id: AtomicU64,
    /// Next available keyframe ID (atomic for thread safety)
    next_keyframe_id: AtomicU64,
}

impl Map {
    /// Create a new map
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a map point
    pub fn add_point(&mut self, mut point: MapPoint) -> u64 {
        let id = self.next_point_id.fetch_add(1, Ordering::Relaxed);
        point.id = id;
        self.points.insert(id, point);
        id
    }

    /// Insert a map point with a specific ID
    pub fn insert_point_with_id(&mut self, id: u64, mut point: MapPoint) {
        point.id = id;
        self.points.insert(id, point);
        self.next_point_id.store(
            self.next_point_id
                .load(Ordering::Relaxed)
                .max(id.saturating_add(1)),
            Ordering::Relaxed,
        );
    }

    /// Add a keyframe
    pub fn add_keyframe(&mut self, mut keyframe: KeyFrame) -> u64 {
        let id = self.next_keyframe_id.fetch_add(1, Ordering::Relaxed);
        keyframe.frame.id = id;
        self.keyframes.insert(id, keyframe);
        id
    }

    /// Insert a keyframe with a specific ID
    pub fn insert_keyframe_with_id(&mut self, id: u64, mut keyframe: KeyFrame) {
        keyframe.frame.id = id;
        self.keyframes.insert(id, keyframe);
        self.next_keyframe_id.store(
            self.next_keyframe_id
                .load(Ordering::Relaxed)
                .max(id.saturating_add(1)),
            Ordering::Relaxed,
        );
    }

    /// Set the next IDs for map points and keyframes
    pub fn set_next_ids(&mut self, next_point_id: u64, next_keyframe_id: u64) {
        self.next_point_id.store(next_point_id, Ordering::Relaxed);
        self.next_keyframe_id
            .store(next_keyframe_id, Ordering::Relaxed);
    }

    /// Get a map point by ID
    pub fn get_point(&self, id: u64) -> Option<&MapPoint> {
        self.points.get(&id)
    }

    /// Get a map point mutably
    pub fn get_point_mut(&mut self, id: u64) -> Option<&mut MapPoint> {
        self.points.get_mut(&id)
    }

    /// Get a keyframe by ID
    pub fn get_keyframe(&self, id: u64) -> Option<&KeyFrame> {
        self.keyframes.get(&id)
    }

    /// Get a keyframe mutably by ID
    pub fn get_keyframe_mut(&mut self, id: u64) -> Option<&mut KeyFrame> {
        self.keyframes.get_mut(&id)
    }

    /// Get all map points
    pub fn points(&self) -> impl Iterator<Item = &MapPoint> {
        self.points.values()
    }

    /// Get all keyframes
    pub fn keyframes(&self) -> impl Iterator<Item = &KeyFrame> {
        self.keyframes.values()
    }

    /// Number of map points
    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    /// Number of keyframes
    pub fn num_keyframes(&self) -> usize {
        self.keyframes.len()
    }

    /// Get map points that are not outliers
    pub fn valid_points(&self) -> impl Iterator<Item = &MapPoint> {
        self.points.values().filter(|p| !p.is_outlier)
    }

    /// Remove outliers
    pub fn remove_outliers(&mut self) {
        self.points.retain(|_, p| !p.is_outlier);
    }
}
