mod backend;
mod device_status;
mod loss;
mod optimizer;
mod runtime;
mod splats;
mod topology_accum;
mod trainer;

pub(crate) use backend::{GsBackendBase, GsDevice, GsDiffBackend};
#[allow(unused_imports)]
pub(crate) use device_status::{
    DeviceTrainingStatus, TrainingStatusSnapshot, STATUS_FORWARD_OVERFLOW, STATUS_NON_FINITE_LOSS,
};
pub(crate) use runtime::train_splats;
pub(crate) use splats::{device_splats_to_host, host_splats_to_device, DeviceSplats};
