use burn::tensor::{DType, TensorMetadata};
use burn_cubecl::cubecl::CubeCount;
use burn_cubecl::{kernel::into_contiguous, BoolElement, CubeBackend, FloatElement, IntElement};
use burn_wgpu::WgpuRuntime;

const WORKGROUP_SIZE: u32 = 256;

pub trait RadixSortBackend: burn::tensor::backend::Backend {
    fn radix_sort_by_key_u32_primitive(
        keys: Self::IntTensorPrimitive,
        values: Self::IntTensorPrimitive,
    ) -> Result<(Self::IntTensorPrimitive, Self::IntTensorPrimitive), String>;

    /// Sort `keys[0..count]` without reading `count` back to the host.
    ///
    /// `dispatch` is a 3-wide indirect workgroup count for `div_ceil(count, 256)`.
    /// Output slots at and after the device count stay at `value_fill`.
    fn radix_sort_counted_primitive(
        keys: Self::IntTensorPrimitive,
        values: Self::IntTensorPrimitive,
        logical_count: Self::IntTensorPrimitive,
        dispatch: Self::IntTensorPrimitive,
        value_fill: i32,
    ) -> Result<(Self::IntTensorPrimitive, Self::IntTensorPrimitive), String>;
}

impl<F, I, BT> RadixSortBackend for CubeBackend<WgpuRuntime, F, I, BT>
where
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn radix_sort_by_key_u32_primitive(
        keys: Self::IntTensorPrimitive,
        values: Self::IntTensorPrimitive,
    ) -> Result<(Self::IntTensorPrimitive, Self::IntTensorPrimitive), String> {
        let keys = into_contiguous(keys);
        let values = into_contiguous(values);

        if keys.dtype() != DType::U32 && keys.dtype() != DType::I32 {
            return Err(format!(
                "radix_sort_by_key_u32 expects 32-bit integer keys, got {:?}",
                keys.dtype()
            ));
        }
        if values.dtype() != DType::U32 && values.dtype() != DType::I32 {
            return Err(format!(
                "radix_sort_by_key_u32 expects 32-bit integer values, got {:?}",
                values.dtype()
            ));
        }
        if keys.shape()[0] != values.shape()[0] {
            return Err(format!(
                "radix_sort_by_key_u32 expects matching lengths, got {} keys and {} values",
                keys.shape()[0],
                values.shape()[0]
            ));
        }
        if keys.device != values.device {
            return Err("radix_sort_by_key_u32 expects keys and values on the same device".into());
        }

        let len = keys.shape()[0];
        if len <= 1 {
            return Ok((keys, values));
        }

        let device = keys.device.clone();
        let count = burn::tensor::Tensor::<Self, 1, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(vec![len as i32], [1]),
            &device,
        )
        .into_primitive();
        let groups = (len as u32).div_ceil(WORKGROUP_SIZE);
        super::device_radix::radix_sort_counted::<F, I, BT>(
            keys,
            values,
            count,
            CubeCount::Static(groups, 1, 1),
            0,
        )
    }

    fn radix_sort_counted_primitive(
        keys: Self::IntTensorPrimitive,
        values: Self::IntTensorPrimitive,
        logical_count: Self::IntTensorPrimitive,
        dispatch: Self::IntTensorPrimitive,
        value_fill: i32,
    ) -> Result<(Self::IntTensorPrimitive, Self::IntTensorPrimitive), String> {
        let dispatch = CubeCount::Dynamic(dispatch.handle.clone().binding());
        super::device_radix::radix_sort_counted::<F, I, BT>(
            keys,
            values,
            logical_count,
            dispatch,
            value_fill,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::RadixSortBackend;
    use crate::training::engine::GsBackendBase;
use burn::prelude::*;
use burn::tensor::{Int, TensorData};

    fn cpu_unsigned_sort(keys: &[i32], values: &[i32]) -> (Vec<i32>, Vec<i32>) {
        let mut pairs: Vec<(u32, i32, usize)> = keys
            .iter()
            .zip(values)
            .enumerate()
            .map(|(index, (key, value))| (*key as u32, *value, index))
            .collect();
        pairs.sort_by_key(|pair| (pair.0, pair.2));
        let keys = pairs.iter().map(|pair| pair.0 as i32).collect();
        let values = pairs.iter().map(|pair| pair.1).collect();
        (keys, values)
    }

    async fn sort_on_device(
        keys: Vec<i32>,
        values: Vec<i32>,
        count: usize,
        value_fill: i32,
    ) -> (Vec<i32>, Vec<i32>) {
        let device = <GsBackendBase as Backend>::Device::default();
        let len = keys.len();
        if len == 0 {
            return (Vec::new(), Vec::new());
        }
        let key_tensor = Tensor::<GsBackendBase, 1, Int>::from_data(
            TensorData::new(keys, [len]),
            &device,
        );
        let value_tensor = Tensor::<GsBackendBase, 1, Int>::from_data(
            TensorData::new(values, [len]),
            &device,
        );
        let count_tensor = Tensor::<GsBackendBase, 1, Int>::from_data(
            TensorData::new(vec![count as i32], [1]),
            &device,
        );
        let groups = (count as u32).div_ceil(256).max(1);
        let (sorted_keys, sorted_values) = if count == len {
            GsBackendBase::radix_sort_by_key_u32_primitive(
                key_tensor.into_primitive(),
                value_tensor.into_primitive(),
            )
            .expect("radix sort")
        } else {
            GsBackendBase::radix_sort_counted_primitive(
                key_tensor.into_primitive(),
                value_tensor.into_primitive(),
                count_tensor.into_primitive(),
                Tensor::<GsBackendBase, 1, Int>::from_data(
                    TensorData::new(vec![groups as i32, 1, 1], [3]),
                    &device,
                )
                .into_primitive(),
                value_fill,
            )
            .expect("counted radix sort")
        };
        let keys = Tensor::<GsBackendBase, 1, Int>::from_primitive(sorted_keys)
            .into_data_async()
            .await
            .expect("keys")
            .into_vec::<i32>()
            .expect("key data");
        let values = Tensor::<GsBackendBase, 1, Int>::from_primitive(sorted_values)
            .into_data_async()
            .await
            .expect("values")
            .into_vec::<i32>()
            .expect("value data");
        (keys, values)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn radix_sort_matches_stable_unsigned_reference() {
        let mut cases = vec![
            vec![],
            vec![42],
            vec![4, 1, 4, 2, 0],
            vec![9, 8, 7, 6, 5, 4, 3],
            vec![0, 0, 0, 1, 1],
            vec![-2, -1, 0, 1, 2, i32::MAX, i32::MIN],
        ];
        for len in [17usize, 255, 256, 257, 4093] {
            cases.push(
                (0..len)
                    .map(|index| ((index * 37) as i32).wrapping_mul(17))
                    .collect(),
            );
        }
        for keys in cases {
            let values: Vec<i32> = (0..keys.len() as i32).collect();
            let (expected_keys, expected_values) = cpu_unsigned_sort(&keys, &values);
            let (sorted_keys, sorted_values) =
                sort_on_device(keys, values, expected_keys.len(), 0).await;
            assert_eq!(sorted_keys, expected_keys);
            assert_eq!(sorted_values, expected_values);
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn counted_radix_leaves_the_unscanned_tail_filled() {
        let keys = vec![5, 1, 4, 9, 8, 7];
        let values = vec![10, 11, 12, 13, 14, 15];
        let (sorted_keys, sorted_values) = sort_on_device(keys.clone(), values.clone(), 3, 99).await;
        let (prefix_keys, prefix_values) = cpu_unsigned_sort(&keys[..3], &values[..3]);
        assert_eq!(&sorted_keys[..3], prefix_keys.as_slice());
        assert_eq!(&sorted_values[..3], prefix_values.as_slice());
        assert_eq!(&sorted_keys[3..], &[-1, -1, -1]);
        assert_eq!(&sorted_values[3..], &[99, 99, 99]);
    }
}
