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
        for node in &self.nodes {
            if !node
                .variants
                .iter()
                .any(|v| ledger.fits(&v.name, &v.resources).is_some())
            {
                return Err(Error::Unschedulable(node.name.clone()));
            }
        }
        Ok(())
    }
}
