use std::{
    fmt::Debug,
    sync::{Arc, Mutex},
};

use crate::{ops::Operation, scalar, shape};

#[derive(Debug)]
pub enum DataRef {
    Tensor((Arc<Mutex<Box<[f32]>>>, TensorLayout)),
    Scalar(Arc<Mutex<f32>>),
}

impl DataRef {
    pub fn new_scalar(value: f32) -> Self {
        Self::Scalar(Arc::new(Mutex::new(value)))
    }

    pub fn new_tensor(value: Vec<f32>, layout: TensorLayout) -> Self {
        Self::Tensor((Arc::new(Mutex::new(value.into_boxed_slice())), layout))
    }

    pub fn get_scalar(&self) -> Option<f32> {
        match self {
            Self::Scalar(scalar) => Some(*scalar.lock().expect("Scalar data is poisoned.")),
            _ => None,
        }
    }

    pub fn get_tensor(&self) -> Option<(Vec<f32>, TensorLayout)> {
        match self {
            Self::Tensor((tensor, layout)) => {
                let tensor = tensor.lock().expect("Tensor data is poisoned.");
                Some((tensor.to_vec(), layout.clone()))
            }
            _ => None,
        }
    }

    pub fn refcount(&self) -> usize {
        match self {
            Self::Scalar(scalar) => Arc::strong_count(scalar),
            Self::Tensor((tensor, _)) => Arc::strong_count(tensor),
        }
    }
}

#[derive(Debug)]
pub struct TensorData {
    pub(crate) data: Arc<Mutex<Box<[f32]>>>,
    #[allow(dead_code)]
    pub(crate) grad: Option<Arc<Mutex<Box<[f32]>>>>,
    pub layout: TensorLayout,
    pub op: Option<Operation>,
}

/// TODO: Properly implement layout
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorLayout {
    pub shape: Vec<usize>,
}

impl TensorData {
    pub fn new(layout: TensorLayout) -> Self {
        let data = {
            let data = vec![f32::default(); layout.size()];
            data.into_boxed_slice()
        };
        Self {
            data: Arc::new(Mutex::new(data)),
            grad: None,
            layout,
            op: None,
        }
    }

    pub fn new_with_data(layout: TensorLayout, data: Vec<f32>) -> Self {
        Self {
            data: Arc::new(Mutex::new(data.into_boxed_slice())),
            grad: None,
            layout,
            op: None,
        }
    }

    pub(crate) fn new_with_op(layout: TensorLayout, data: Vec<f32>, op: Operation) -> Self {
        Self {
            data: Arc::new(Mutex::new(data.into_boxed_slice())),
            grad: None,
            layout,
            op: Some(op),
        }
    }

    pub fn data(&self) -> Vec<f32> {
        self.data.lock().expect("Scalar data is poisoned").to_vec()
    }

    pub fn data_ref(&self) -> DataRef {
        DataRef::Tensor((Arc::clone(&self.data), self.layout.clone()))
    }

    pub fn layout(&self) -> TensorLayout {
        self.layout.clone()
    }

    pub fn op(&self) -> Option<&Operation> {
        self.op.as_ref()
    }

    pub fn ndims(&self) -> usize {
        self.layout.ndims()
    }
}

impl TensorLayout {
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn transpose(&mut self) {
        self.shape.reverse();
    }

    pub fn ndims(&self) -> usize {
        self.shape.len()
    }
}

#[derive(Debug)]
pub struct ScalarData {
    pub(crate) data: Arc<Mutex<f32>>,
    #[allow(dead_code)]
    pub(crate) grad: Option<Arc<Mutex<f32>>>,
    pub(crate) op: Option<Operation>,
}

impl ScalarData {
    pub fn new(data: f32) -> Self {
        Self {
            data: Arc::new(Mutex::new(data)),
            grad: None,
            op: None,
        }
    }

    pub(crate) fn new_with_op(data: f32, op: Operation) -> Self {
        Self {
            data: Arc::new(Mutex::new(data)),
            grad: None,
            op: Some(op),
        }
    }

    pub fn data(&self) -> f32 {
        *self.data.lock().expect("Scalar data is poisoned")
    }

    pub fn data_ref(&self) -> DataRef {
        DataRef::Scalar(Arc::clone(&self.data))
    }

    pub fn layout(&self) -> TensorLayout {
        shape!(1)
    }

    pub fn op(&self) -> Option<&Operation> {
        self.op.as_ref()
    }
}

#[macro_export]
macro_rules! scalar_data {
    ($x:expr) => {
        $crate::values::ScalarData::new($x)
    };
}

#[derive(Debug)]
pub enum Value {
    Tensor(TensorData),
    Scalar(ScalarData),
}

impl Value {
    pub fn tensor_data(&self) -> Option<Vec<f32>> {
        match self {
            Value::Tensor(tensor) => Some(tensor.data()),
            _ => None,
        }
    }

    pub fn scalar_data(&self) -> Option<f32> {
        match self {
            Value::Scalar(scalar) => Some(scalar.data()),
            _ => None,
        }
    }

    pub fn layout(&self) -> TensorLayout {
        match self {
            Value::Tensor(tensor) => tensor.layout.clone(),
            Value::Scalar(scalar) => scalar.layout(),
        }
    }

    pub fn op(&self) -> Option<&Operation> {
        match self {
            Value::Tensor(tensor) => tensor.op.as_ref(),
            Value::Scalar(scalar) => scalar.op.as_ref(),
        }
    }

    pub fn refcount(&self) -> usize {
        match self {
            Value::Tensor(tensor) => Arc::strong_count(&tensor.data),
            Value::Scalar(scalar) => Arc::strong_count(&scalar.data),
        }
    }

    pub fn new_scalar(data: f32) -> Self {
        Self::Scalar(ScalarData::new(data))
    }

    pub fn new_tensor(layout: TensorLayout) -> Self {
        Self::Tensor(TensorData::new(layout))
    }

    pub fn new_tensor_with_data(layout: TensorLayout, data: Vec<f32>) -> Self {
        Self::Tensor(TensorData::new_with_data(layout, data))
    }

    pub fn new_tensor_with_op(layout: TensorLayout, data: Vec<f32>, op: Operation) -> Self {
        Self::Tensor(TensorData::new_with_op(layout, data, op))
    }

    pub fn new_scalar_with_op(data: f32, op: Operation) -> Self {
        Self::Scalar(ScalarData::new_with_op(data, op))
    }
}

impl TryFrom<Value> for TensorData {
    type Error = &'static str;
    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Tensor(tensor) => Ok(tensor),
            _ => Err("Expected tensor"),
        }
    }
}

impl TryFrom<Value> for ScalarData {
    type Error = &'static str;
    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Scalar(scalar) => Ok(scalar),
            _ => Err("Expected scalar"),
        }
    }
}

impl TryFrom<Value> for f32 {
    type Error = &'static str;
    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Scalar(scalar) => Ok(scalar.data()),
            _ => Err("Expected scalar"),
        }
    }
}

impl From<f32> for Value {
    fn from(data: f32) -> Self {
        scalar!(data)
    }
}

impl<const N: usize> From<[f32; N]> for Value {
    fn from(data: [f32; N]) -> Self {
        Value::Tensor(TensorData::new_with_data(shape!(N), data.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor;

    use super::*;
    #[test]
    fn test_tensor_allocation() {
        let tensor = tensor!(1., 2., 3., 4., 5., 6.);

        let data = tensor.tensor_data().expect("Expected tensor");
        assert_eq!(data, vec![1., 2., 3., 4., 5., 6.]);

        let layout = tensor.layout().clone();
        assert_eq!(layout, shape!(6));
        assert_eq!(layout.size(), 6);

        assert!(tensor.op().is_none());
    }

    #[test]
    fn test_scalar_allocation() {
        let scalar = scalar!(1.0);

        let data = scalar.scalar_data().expect("Expected scalar");
        assert_eq!(data, 1.0);

        assert!(scalar.op().is_none());
    }

    #[test]
    fn test_sizes() {
        let tensor = tensor!(1., 2., 3., 4., 5., 6.);
        let scalar = scalar!(1.0);

        assert_eq!(tensor.layout().size(), 6);
        assert_eq!(scalar.layout().size(), 1);
    }
}
