use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, OnceLock,
};

use crate::resource::Ledger;
use crate::task::{Slot, Work};
use crate::{Error, ResourceRequest, RuntimeConfig, TaskHandle, TaskId, TaskVariant};

static NEXT_GRAPH: AtomicU64 = AtomicU64::new(1);

/// A one-shot graph. Tasks may be added in any order; cycles and foreign handles
/// are rejected before the graph can execute. Variants are in preference order.
pub struct TaskGraph {
    pub(crate) id: u64,
    pub(crate) nodes: Vec<Node>,
}

pub(crate) struct Variant {
    pub name: String,
    pub resources: ResourceRequest,
    pub work: Option<Work>,
}

pub(crate) struct Node {
    pub name: String,
    pub deps: Vec<usize>,
    pub variants: Vec<Variant>,
    pub slot: Slot,
}

impl Default for TaskGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskGraph {
    pub fn new() -> Self {
        Self {
            id: NEXT_GRAPH.fetch_add(1, Ordering::Relaxed),
            nodes: Vec::new(),
        }
    }
    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn task<T: Send + Sync + 'static>(
        &mut self,
        name: impl Into<String>,
        variants: Vec<TaskVariant<T>>,
    ) -> Result<TaskHandle<T>, Error> {
        if variants.is_empty() {
            return Err(Error::Invalid(
                "a task needs at least one implementation".into(),
            ));
        }
        let mut names = std::collections::HashSet::new();
        for variant in &variants {
            variant.resources.validate()?;
            if variant.name.is_empty() || !names.insert(&variant.name) {
                return Err(Error::Invalid(
                    "variant names must be nonempty and unique within a task".into(),
                ));
            }
        }
        let slot = Arc::new(OnceLock::new());
        let handle = TaskHandle {
            id: TaskId {
                graph: self.id,
                index: self.nodes.len(),
            },
            slot: slot.clone(),
            marker: PhantomData,
        };
        self.nodes.push(Node {
            name: name.into(),
            deps: Vec::new(),
            slot,
            variants: variants
                .into_iter()
                .map(|v| Variant {
                    name: v.name,
                    resources: v.resources,
                    work: Some(v.work),
                })
                .collect(),
        });
        Ok(handle)
    }

    /// Add a dependency `prerequisite -> task`. Duplicate edges are idempotent.
    pub fn depends_on(&mut self, task: TaskId, prerequisite: TaskId) -> Result<(), Error> {
        for id in [task, prerequisite] {
            if id.graph != self.id || id.index >= self.nodes.len() {
                return Err(Error::ForeignTask);
            }
        }
        if task == prerequisite {
            return Err(Error::Cycle);
        }
        let deps = &mut self.nodes[task.index].deps;
        if !deps.contains(&prerequisite.index) {
            deps.push(prerequisite.index);
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), Error> {
        let mut counts: Vec<_> = self.nodes.iter().map(|n| n.deps.len()).collect();
        let mut consumers = vec![Vec::new(); self.len()];
        for (i, node) in self.nodes.iter().enumerate() {
            for &dep in &node.deps {
                consumers[dep].push(i);
            }
        }
        let mut ready: VecDeque<_> = counts
            .iter()
            .enumerate()
            .filter(|(_, n)| **n == 0)
            .map(|(i, _)| i)
            .collect();
        let mut visited = 0;
        while let Some(node) = ready.pop_front() {
            visited += 1;
            for &consumer in &consumers[node] {
                counts[consumer] -= 1;
                if counts[consumer] == 0 {
                    ready.push_back(consumer);
                }
            }
        }
        if visited == self.len() {
            Ok(())
        } else {
            Err(Error::Cycle)
        }
    }

    pub(crate) fn validate_resources(&self, config: &RuntimeConfig) -> Result<(), Error> {
        let ledger = Ledger::new(config.clone());
        let mut max_outputs = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let mut max_output = 0;
            let mut feasible = false;
            for variant in &node.variants {
                if let Some(grant) = ledger.fits(&variant.name, &variant.resources) {
                    feasible = true;
                    max_output = max_output.max(grant.output_memory_bytes);
                }
            }
            if !feasible {
                return Err(Error::Unschedulable(node.name.clone()));
            }
            max_outputs.push(max_output);
        }

        // A dependency can keep every ancestor artifact alive until the consumer
        // has acquired its working memory. Build a conservative topological
        // liveness bound so an impossible graph is rejected instead of waiting
        // forever after its producers have completed.
        let mut counts: Vec<_> = self.nodes.iter().map(|node| node.deps.len()).collect();
        let mut consumers = vec![Vec::new(); self.len()];
        for (index, node) in self.nodes.iter().enumerate() {
            for &dependency in &node.deps {
                consumers[dependency].push(index);
            }
        }
        let mut ready: VecDeque<_> = counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count == 0)
            .map(|(index, _)| index)
            .collect();
        let mut order = Vec::with_capacity(self.len());
        while let Some(index) = ready.pop_front() {
            order.push(index);
            for &consumer in &consumers[index] {
                counts[consumer] -= 1;
                if counts[consumer] == 0 {
                    ready.push_back(consumer);
                }
            }
        }
        if order.len() != self.len() {
            return Err(Error::Cycle);
        }

        let mut retained = vec![0u64; self.len()];
        for index in order {
            let retained_before = self.nodes[index]
                .deps
                .iter()
                .try_fold(0u64, |total, &dependency| {
                    total
                        .checked_add(retained[dependency])
                        .and_then(|total| total.checked_add(max_outputs[dependency]))
                })
                .ok_or_else(|| Error::Unschedulable(self.nodes[index].name.clone()))?;
            retained[index] = retained_before;

            let has_memory_feasible_variant = self.nodes[index].variants.iter().any(|variant| {
                ledger
                    .fits(&variant.name, &variant.resources)
                    .is_some_and(|grant| {
                        retained_before
                            .checked_add(grant.working_memory_bytes)
                            .and_then(|total| total.checked_add(grant.output_memory_bytes))
                            .is_some_and(|total| total <= config.budget.memory_bytes)
                    })
            });
            if !has_memory_feasible_variant {
                return Err(Error::Unschedulable(self.nodes[index].name.clone()));
            }
        }
        Ok(())
    }
}
