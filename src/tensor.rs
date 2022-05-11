use std::{
    any::Any,
    sync::{Arc, Mutex, Weak},
};

use super::dtypes::Dtype;

pub struct TensorCollection {
    tensor_references: Mutex<Vec<Arc<Mutex<dyn Any + Sync + Send>>>>,
}

impl TensorCollection {
    fn new() -> Self {
        Self {
            tensor_references: Mutex::new(Vec::new()),
        }
    }

    pub fn clear_unused(&self) {
        let mut tensor_references = match self.tensor_references.lock() {
            Ok(tensor_references) => tensor_references,
            Err(_) => panic!("Failed to lock tensor references"),
        };

        let mut i = 0;
        while i < tensor_references.len() {
            if Arc::weak_count(&tensor_references[i]) == 0 {
                drop(tensor_references.swap_remove(i));
            } else {
                i += 1;
            }
        }
    }

    pub fn allocate_tensor<T: Dtype>(&self, size: usize) -> Weak<Mutex<Box<[T]>>> {
        match self.tensor_references.lock() {
            Ok(mut tensor_references) => {
                let data = {
                    let data = vec![T::default(); size];
                    data.into_boxed_slice()
                };
                let tensor_reference = Arc::new(Mutex::new(data));
                tensor_references.push(tensor_reference.clone());
                Arc::downgrade(&tensor_reference)
            }
            Err(_) => panic!("Failed to lock tensor references"),
        }
    }

    pub fn reference_count(&self) -> usize {
        match self.tensor_references.lock() {
            Ok(tensor_references) => tensor_references.len(),
            Err(_) => panic!("Failed to lock tensor references"),
        }
    }
}

lazy_static::lazy_static! {
    pub static ref TENSOR_COLLECTION: TensorCollection = TensorCollection::new();
}

struct TensorData<T: Dtype> {
    data: Weak<Mutex<Box<[T]>>>,
    layout: TensorLayout,
}

/// TODO: Properly implement layout
struct TensorLayout {
    shape: Vec<usize>,
}

impl<T: Dtype> TensorData<T> {
    pub fn new(layout: TensorLayout) -> Self {
        Self {
            data: TENSOR_COLLECTION.allocate_tensor(layout.size()),
            layout,
        }
    }
}

impl TensorLayout {
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_tensor_allocation() {
        let tensor = TensorData::<f32>::new(TensorLayout {
            shape: vec![1, 2, 3],
        });

        let valid_reference_count = TENSOR_COLLECTION.reference_count();
        assert_eq!(valid_reference_count, 1);

        assert_eq!(tensor.layout.size(), 6);
        let data = tensor.data.upgrade().unwrap().lock().unwrap().to_vec();
        assert_eq!(data, vec![0.0; 6]);

        TENSOR_COLLECTION.clear_unused();

        // The tensor should not have been deallocated yet
        let valid_reference_count = TENSOR_COLLECTION.reference_count();
        assert_eq!(valid_reference_count, 1);
        assert!(matches!(tensor.data.upgrade(), Some(_)));

        // Drop the tensor
        drop(tensor);

        TENSOR_COLLECTION.clear_unused();

        // The tensor should have been deallocated
        let valid_reference_count = TENSOR_COLLECTION.reference_count();
        assert_eq!(valid_reference_count, 0);
    }
}
