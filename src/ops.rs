use crate::{
    shape,
    values::{DataRef, Value},
};

#[derive(Debug)]
pub enum Operation {
    Add(DataRef, DataRef),
    Sub(DataRef, DataRef),
    Mul(DataRef, DataRef),
    Div(DataRef, DataRef),
    Pow(DataRef, DataRef),
    Neg(DataRef),
    Sin(DataRef),
    Cos(DataRef),
    Tan(DataRef),
    Asin(DataRef),
    Acos(DataRef),
    Atan(DataRef),
    Sinh(DataRef),
    Cosh(DataRef),
    Tanh(DataRef),
    Asinh(DataRef),
    Acosh(DataRef),
    Atanh(DataRef),
    Sqrt(DataRef),
    Exp(DataRef),
    Log(DataRef),
    Log1p(DataRef),
    Abs(DataRef),
    Transpose(DataRef),
    Matmul(DataRef, DataRef),
}

use auto_ops::*;

impl_op_ex!(+|a: &Value, b: &Value| -> Value {
    match (a, b) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            assert_eq!(a.layout, b.layout);
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            {
                // Drop locks in opposite order to avoid deadlocks (hence the scoping).
                let b_data = b.data.lock().expect("Tensor data is poisoned");
                Value::new_tensor_with_op(
                    a.layout(),
                    a_data.iter().zip(b_data.iter()).map(|(a, b)| a + b).collect(),
                    Operation::Add(a.data_ref(), b.data_ref()),
                )
            }
        },
        (Value::Scalar(a), Value::Scalar(b)) => {
            Value::new_scalar_with_op(a.data() + b.data(), Operation::Add(a.data_ref(), b.data_ref()))
        },
        (Value::Tensor(a), Value::Scalar(b)) => {
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            let b_data = b.data();
            Value::new_tensor_with_op(
                a.layout(),
                a_data.iter().map(|a| a + b_data).collect(),
                Operation::Add(a.data_ref(), b.data_ref()),
            )
        },
        (Value::Scalar(a), Value::Tensor(b)) => {
            let a_data = a.data();
            let b_data = b.data.lock().expect("Tensor data is poisoned");
            Value::new_tensor_with_op(
                b.layout(),
                b_data.iter().map(|b| a_data + b).collect(),
                Operation::Add(a.data_ref(), b.data_ref()),
            )
        },
    }
});

impl_op_ex!(-|a: &Value, b: &Value| -> Value {
    match (a, b) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            assert_eq!(a.layout, b.layout);
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            {
                // Drop locks in opposite order to avoid deadlocks (hence the scoping).
                let b_data = b.data.lock().expect("Tensor data is poisoned");
                Value::new_tensor_with_op(
                    a.layout(),
                    a_data
                        .iter()
                        .zip(b_data.iter())
                        .map(|(a, b)| a - b)
                        .collect(),
                    Operation::Sub(a.data_ref(), b.data_ref()),
                )
            }
        }
        (Value::Scalar(a), Value::Scalar(b)) => Value::new_scalar_with_op(
            a.data() - b.data(),
            Operation::Sub(a.data_ref(), b.data_ref()),
        ),
        (Value::Tensor(a), Value::Scalar(b)) => {
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            let b_data = b.data();
            Value::new_tensor_with_op(
                a.layout(),
                a_data.iter().map(|a| a - b_data).collect(),
                Operation::Sub(a.data_ref(), b.data_ref()),
            )
        }
        (Value::Scalar(a), Value::Tensor(b)) => {
            let a_data = a.data();
            let b_data = b.data.lock().expect("Tensor data is poisoned");
            Value::new_tensor_with_op(
                b.layout(),
                b_data.iter().map(|b| a_data - b).collect(),
                Operation::Sub(a.data_ref(), b.data_ref()),
            )
        }
    }
});

impl_op_ex!(*|a: &Value, b: &Value| -> Value {
    match (a, b) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            assert_eq!(a.layout, b.layout);
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            {
                // Drop locks in opposite order to avoid deadlocks (hence the scoping).
                let b_data = b.data.lock().expect("Tensor data is poisoned");
                Value::new_tensor_with_op(
                    a.layout(),
                    a_data
                        .iter()
                        .zip(b_data.iter())
                        .map(|(a, b)| a * b)
                        .collect(),
                    Operation::Mul(a.data_ref(), b.data_ref()),
                )
            }
        }
        (Value::Scalar(a), Value::Scalar(b)) => Value::new_scalar_with_op(
            a.data() * b.data(),
            Operation::Mul(a.data_ref(), b.data_ref()),
        ),
        (Value::Tensor(a), Value::Scalar(b)) => {
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            let b_data = b.data();
            Value::new_tensor_with_op(
                a.layout(),
                a_data.iter().map(|a| a * b_data).collect(),
                Operation::Mul(a.data_ref(), b.data_ref()),
            )
        }
        (Value::Scalar(a), Value::Tensor(b)) => {
            let a_data = a.data();
            let b_data = b.data.lock().expect("Tensor data is poisoned");
            Value::new_tensor_with_op(
                b.layout(),
                b_data.iter().map(|b| a_data * b).collect(),
                Operation::Mul(a.data_ref(), b.data_ref()),
            )
        }
    }
});

impl_op_ex!(/|a: &Value, b: &Value| -> Value {
    match (a, b) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            assert_eq!(a.layout, b.layout);
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            {
                // Drop locks in opposite order to avoid deadlocks (hence the scoping).
                let b_data = b.data.lock().expect("Tensor data is poisoned");
                Value::new_tensor_with_op(
                    a.layout(),
                    a_data
                        .iter()
                        .zip(b_data.iter())
                        .map(|(&a, &b)| {
                            if b == 0.0 {
                                std::f32::NAN
                            } else {
                                a / b
                            }
                        })
                        .collect(),
                    Operation::Div(a.data_ref(), b.data_ref()),
                )
            }
        }
        (Value::Scalar(a), Value::Scalar(b)) => Value::new_scalar_with_op(
            {
                let a_data = a.data();
                let b_data = b.data();
                if b_data == 0.0 {
                    std::f32::NAN
                } else {
                    a_data / b_data
                }
            },
            Operation::Div(a.data_ref(), b.data_ref()),
        ),
        (Value::Tensor(a), Value::Scalar(b)) => {
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            let b_data = b.data();
            Value::new_tensor_with_op(
                a.layout(),
                if b_data == 0.0 {
                    vec![std::f32::NAN; a_data.len()]
                } else {
                    a_data.iter().map(|a| a / b_data).collect()
                },
                Operation::Div(a.data_ref(), b.data_ref()),
            )
        }
        (Value::Scalar(a), Value::Tensor(b)) => {
            let a_data = a.data();
            let b_data = b.data.lock().expect("Tensor data is poisoned");
            Value::new_tensor_with_op(
                b.layout(),
                b_data.iter().map(|&b| {
                    if b == 0.0 {
                        std::f32::NAN
                    } else {
                        a_data / b
                    }
                }).collect(),
                Operation::Div(a.data_ref(), b.data_ref()),
            )
        }
    }
});

pub fn add(a: &Value, b: &Value) -> Value {
    a + b
}

pub fn sub(a: &Value, b: &Value) -> Value {
    a - b
}

pub fn mul(a: &Value, b: &Value) -> Value {
    a * b
}

pub fn div(a: &Value, b: &Value) -> Value {
    a / b
}

pub fn pow(a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            assert_eq!(a.layout, b.layout);
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            {
                // Drop locks in opposite order to avoid deadlocks (hence the scoping).
                let b_data = b.data.lock().expect("Tensor data is poisoned");
                Value::new_tensor_with_op(
                    a.layout(),
                    a_data
                        .iter()
                        .zip(b_data.iter())
                        .map(|(&a, &b)| {
                            if a == 0.0 && b == 0.0 || a.is_nan() || b.is_nan() {
                                std::f32::NAN
                            } else {
                                a.powf(b)
                            }
                        })
                        .collect(),
                    Operation::Pow(a.data_ref(), b.data_ref()),
                )
            }
        }
        (Value::Scalar(a), Value::Scalar(b)) => Value::new_scalar_with_op(
            {
                let a_data = a.data();
                let b_data = b.data();
                if a_data == 0.0 && b_data == 0.0 || a_data.is_nan() || b_data.is_nan() {
                    std::f32::NAN
                } else {
                    a_data.powf(b_data)
                }
            },
            Operation::Pow(a.data_ref(), b.data_ref()),
        ),
        (Value::Tensor(a), Value::Scalar(b)) => {
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            let b_data = b.data();
            Value::new_tensor_with_op(
                a.layout(),
                a_data
                    .iter()
                    .map(|&a| {
                        if a == 0.0 && b_data == 0.0 || a.is_nan() || b_data.is_nan() {
                            std::f32::NAN
                        } else {
                            a.powf(b_data)
                        }
                    })
                    .collect(),
                Operation::Pow(a.data_ref(), b.data_ref()),
            )
        }
        (Value::Scalar(a), Value::Tensor(b)) => {
            let a_data = a.data();
            let b_data = b.data.lock().expect("Tensor data is poisoned");
            Value::new_tensor_with_op(
                b.layout(),
                b_data
                    .iter()
                    .map(|&b| {
                        if a_data == 0.0 && b == 0.0 || a_data.is_nan() || b.is_nan() {
                            std::f32::NAN
                        } else {
                            a_data.powf(b)
                        }
                    })
                    .collect(),
                Operation::Pow(a.data_ref(), b.data_ref()),
            )
        }
    }
}

pub fn neg(a: &Value) -> Value {
    match a {
        Value::Tensor(a) => {
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            Value::new_tensor_with_op(
                a.layout(),
                a_data.iter().map(|&a| -a).collect(),
                Operation::Neg(a.data_ref()),
            )
        }
        Value::Scalar(a) => Value::new_scalar_with_op(-a.data(), Operation::Neg(a.data_ref())),
    }
}

pub fn sin(a: &Value) -> Value {
    match a {
        Value::Tensor(a) => {
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            Value::new_tensor_with_op(
                a.layout(),
                a_data.iter().map(|&a| a.sin()).collect(),
                Operation::Sin(a.data_ref()),
            )
        }
        Value::Scalar(a) => Value::new_scalar_with_op(a.data().sin(), Operation::Sin(a.data_ref())),
    }
}

pub fn cos(a: &Value) -> Value {
    match a {
        Value::Tensor(a) => {
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            Value::new_tensor_with_op(
                a.layout(),
                a_data.iter().map(|&a| a.cos()).collect(),
                Operation::Cos(a.data_ref()),
            )
        }
        Value::Scalar(a) => Value::new_scalar_with_op(a.data().cos(), Operation::Cos(a.data_ref())),
    }
}

pub fn tan(a: &Value) -> Value {
    match a {
        Value::Tensor(a) => {
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            Value::new_tensor_with_op(
                a.layout(),
                a_data.iter().map(|&a| a.tan()).collect(),
                Operation::Tan(a.data_ref()),
            )
        }
        Value::Scalar(a) => Value::new_scalar_with_op(a.data().tan(), Operation::Tan(a.data_ref())),
    }
}

pub fn asin(a: &Value) -> Value {
    match a {
        Value::Tensor(a) => {
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            Value::new_tensor_with_op(
                a.layout(),
                a_data.iter().map(|&a| a.asin()).collect(),
                Operation::Asin(a.data_ref()),
            )
        }
        Value::Scalar(a) => {
            Value::new_scalar_with_op(a.data().asin(), Operation::Asin(a.data_ref()))
        }
    }
}

pub fn acos(a: &Value) -> Value {
    match a {
        Value::Tensor(a) => {
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            Value::new_tensor_with_op(
                a.layout(),
                a_data.iter().map(|&a| a.acos()).collect(),
                Operation::Acos(a.data_ref()),
            )
        }
        Value::Scalar(a) => {
            Value::new_scalar_with_op(a.data().acos(), Operation::Acos(a.data_ref()))
        }
    }
}

pub fn atan(a: &Value) -> Value {
    match a {
        Value::Tensor(a) => {
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            Value::new_tensor_with_op(
                a.layout(),
                a_data.iter().map(|&a| a.atan()).collect(),
                Operation::Atan(a.data_ref()),
            )
        }
        Value::Scalar(a) => {
            Value::new_scalar_with_op(a.data().atan(), Operation::Atan(a.data_ref()))
        }
    }
}

pub fn sinh(a: &Value) -> Value {
    match a {
        Value::Tensor(a) => {
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            Value::new_tensor_with_op(
                a.layout(),
                a_data.iter().map(|&a| a.sinh()).collect(),
                Operation::Sinh(a.data_ref()),
            )
        }
        Value::Scalar(a) => {
            Value::new_scalar_with_op(a.data().sinh(), Operation::Sinh(a.data_ref()))
        }
    }
}

pub fn cosh(a: &Value) -> Value {
    match a {
        Value::Tensor(a) => {
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            Value::new_tensor_with_op(
                a.layout(),
                a_data.iter().map(|&a| a.cosh()).collect(),
                Operation::Cosh(a.data_ref()),
            )
        }
        Value::Scalar(a) => {
            Value::new_scalar_with_op(a.data().cosh(), Operation::Cosh(a.data_ref()))
        }
    }
}

pub fn tanh(a: &Value) -> Value {
    match a {
        Value::Tensor(a) => {
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            Value::new_tensor_with_op(
                a.layout(),
                a_data.iter().map(|&a| a.tanh()).collect(),
                Operation::Tanh(a.data_ref()),
            )
        }
        Value::Scalar(a) => {
            Value::new_scalar_with_op(a.data().tanh(), Operation::Tanh(a.data_ref()))
        }
    }
}

pub fn asinh(a: &Value) -> Value {
    match a {
        Value::Tensor(a) => {
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            Value::new_tensor_with_op(
                a.layout(),
                a_data.iter().map(|&a| a.asinh()).collect(),
                Operation::Asinh(a.data_ref()),
            )
        }
        Value::Scalar(a) => {
            Value::new_scalar_with_op(a.data().asinh(), Operation::Asinh(a.data_ref()))
        }
    }
}

pub fn acosh(a: &Value) -> Value {
    match a {
        Value::Tensor(a) => {
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            Value::new_tensor_with_op(
                a.layout(),
                a_data.iter().map(|&a| a.acosh()).collect(),
                Operation::Acosh(a.data_ref()),
            )
        }
        Value::Scalar(a) => {
            Value::new_scalar_with_op(a.data().acosh(), Operation::Acosh(a.data_ref()))
        }
    }
}

pub fn atanh(a: &Value) -> Value {
    match a {
        Value::Tensor(a) => {
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            Value::new_tensor_with_op(
                a.layout(),
                a_data.iter().map(|&a| a.atanh()).collect(),
                Operation::Atanh(a.data_ref()),
            )
        }
        Value::Scalar(a) => {
            Value::new_scalar_with_op(a.data().atanh(), Operation::Atanh(a.data_ref()))
        }
    }
}

pub fn sqrt(a: &Value) -> Value {
    match a {
        Value::Tensor(a) => {
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            Value::new_tensor_with_op(
                a.layout(),
                a_data.iter().map(|&a| a.sqrt()).collect(),
                Operation::Sqrt(a.data_ref()),
            )
        }
        Value::Scalar(a) => {
            Value::new_scalar_with_op(a.data().sqrt(), Operation::Sqrt(a.data_ref()))
        }
    }
}

pub fn exp(a: &Value) -> Value {
    match a {
        Value::Tensor(a) => {
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            Value::new_tensor_with_op(
                a.layout(),
                a_data.iter().map(|&a| a.exp()).collect(),
                Operation::Exp(a.data_ref()),
            )
        }
        Value::Scalar(a) => Value::new_scalar_with_op(a.data().exp(), Operation::Exp(a.data_ref())),
    }
}

pub fn log(a: &Value) -> Value {
    match a {
        Value::Tensor(a) => {
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            Value::new_tensor_with_op(
                a.layout(),
                a_data.iter().map(|&a| a.ln()).collect(),
                Operation::Log(a.data_ref()),
            )
        }
        Value::Scalar(a) => Value::new_scalar_with_op(a.data().ln(), Operation::Log(a.data_ref())),
    }
}

pub fn log1p(a: &Value) -> Value {
    match a {
        Value::Tensor(a) => {
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            Value::new_tensor_with_op(
                a.layout(),
                a_data.iter().map(|&a| a.ln_1p()).collect(),
                Operation::Log1p(a.data_ref()),
            )
        }
        Value::Scalar(a) => {
            Value::new_scalar_with_op(a.data().ln_1p(), Operation::Log1p(a.data_ref()))
        }
    }
}

pub fn abs(a: &Value) -> Value {
    match a {
        Value::Tensor(a) => {
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            Value::new_tensor_with_op(
                a.layout(),
                a_data.iter().map(|&a| a.abs()).collect(),
                Operation::Abs(a.data_ref()),
            )
        }
        Value::Scalar(a) => Value::new_scalar_with_op(a.data().abs(), Operation::Abs(a.data_ref())),
    }
}

pub fn transpose(a: &Value) -> Value {
    match a {
        Value::Tensor(a) => {
            let a_data = a.data.lock().expect("Tensor data is poisoned");
            if a.ndims() != 2 {
                todo!("Transpose for tensors with more than 2 dimensions")
            }

            let mut layout = a.layout();
            layout.transpose();

            let mut data = Vec::new();
            data.resize(a_data.len(), 0.0);
            for x in 0..layout.shape[1] {
                for y in 0..layout.shape[0] {
                    data[y * layout.shape[1] + x] = a_data[x * layout.shape[0] + y];
                }
            }

            Value::new_tensor_with_op(layout, data, Operation::Transpose(a.data_ref()))
        }
        Value::Scalar(a) => Value::new_scalar_with_op(a.data(), Operation::Transpose(a.data_ref())),
    }
}

pub fn matmul(a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            // TODO: Extrapolate to higher dimensional tensors
            assert!(a.layout().shape.len() == 2);
            assert!(b.layout().shape.len() == 2);
            let k_range = a.layout().shape[1];
            let a_rows = a.layout().shape[0];
            let b_cols = b.layout().shape[1];

            assert_eq!(
                a.layout().shape.last().expect("Tensor has no shape"),
                b.layout().shape.first().expect("Tensor has no shape")
            );

            let a_data = a.data.lock().expect("Tensor data is poisoned");
            {
                let b_data = b.data.lock().expect("Tensor data is poisoned");
                // TODO: Implement a non-native matrix multiplication algorithm
                let mut c_data = vec![0.0; a_rows * b_cols];
                for i in 0..a_rows {
                    for j in 0..b_cols {
                        c_data[i * b_cols + j] = (0..k_range)
                            .map(|k| {
                                let a_val = a_data[i * k_range + k];
                                let b_val = b_data[k * b_cols + j];
                                a_val * b_val
                            })
                            .sum();
                    }
                }
                Value::new_tensor_with_op(
                    shape!(a_rows, b_cols),
                    c_data,
                    Operation::Matmul(a.data_ref(), b.data_ref()),
                )
            }
        }
        _ => panic!("Matmul only works on tensors"),
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ops::{
            abs, acos, asin, atan, cos, exp, log, log1p, matmul, pow, sin, sqrt, tan, transpose,
            Operation,
        },
        scalar, shape, tensor,
    };

    fn are_close_float(test: f32, expected: f32, tol: f32) -> bool {
        if expected.is_infinite() {
            test.abs() > 1.0 / tol && test.signum() == expected.signum()
        } else if expected.is_nan() {
            test.is_nan()
        } else {
            (test - expected).abs() < tol
        }
    }

    fn are_close_vec(test: Vec<f32>, expected: Vec<f32>, tol: f32) -> bool {
        test.iter()
            .zip(expected.iter())
            .all(|(&a, &b)| are_close_float(a, b, tol))
    }

    #[test]
    fn test_dataref() {
        let a = tensor![1.0, 2.0, 3.0] + tensor![4.0, 5.0, 6.0];
        assert!(a.op().is_some());

        let a_op = a.op().unwrap();
        if let Operation::Add(a_data, b_data) = a_op {
            let (a_data, a_layout) = a_data.get_tensor().expect("Expected tensor");
            let (b_data, b_layout) = b_data.get_tensor().expect("Expected tensor");

            assert_eq!(a_layout, shape!(3));
            assert_eq!(b_layout, shape!(3));

            assert_eq!(a_data, vec![1.0, 2.0, 3.0]);
            assert_eq!(b_data, vec![4.0, 5.0, 6.0]);
        } else {
            panic!("Expected add operation");
        }

        let a = tensor![1.0, 2.0, 3.0] + scalar!(4.0);
        assert!(a.op().is_some());

        let a_op = a.op().unwrap();
        if let Operation::Add(a_data, b_data) = a_op {
            let (a_data, a_layout) = a_data.get_tensor().expect("Expected tensor");
            let b_data = b_data.get_scalar().expect("Expected scalar");

            assert_eq!(a_layout, shape!(3));

            assert_eq!(a_data, vec![1.0, 2.0, 3.0]);
            assert_eq!(b_data, 4.0);
        } else {
            panic!("Expected add operation");
        }

        let a = scalar!(1.0) + tensor![1.0, 2.0, 3.0];
        assert!(a.op().is_some());

        let a_op = a.op().unwrap();
        if let Operation::Add(a_data, b_data) = a_op {
            let a_data = a_data.get_scalar().expect("Expected scalar");
            let (b_data, b_layout) = b_data.get_tensor().expect("Expected tensor");

            assert_eq!(b_layout, shape!(3));

            assert_eq!(a_data, 1.0);
            assert_eq!(b_data, vec![1.0, 2.0, 3.0]);
        } else {
            panic!("Expected add operation");
        }

        let a = scalar!(1.0) + scalar!(2.0);
        assert!(a.op().is_some());

        let a_op = a.op().unwrap();
        if let Operation::Add(a_data, b_data) = a_op {
            let a_data = a_data.get_scalar().expect("Expected scalar");
            let b_data = b_data.get_scalar().expect("Expected scalar");

            assert_eq!(a_data, 1.0);
            assert_eq!(b_data, 2.0);
        } else {
            panic!("Expected add operation");
        }
    }

    #[test]
    fn test_value_refcount() {
        let a = tensor![1.0, 2.0, 3.0];
        let b = tensor![4.0, 5.0, 6.0];

        let c = &a + &b;

        assert_eq!(a.refcount(), 2);
        assert_eq!(b.refcount(), 2);
        drop(c);

        assert_eq!(a.refcount(), 1);
        assert_eq!(b.refcount(), 1);

        let c = &a + &b;
        drop(a);

        if let Operation::Add(a_data, b_data) = c.op().unwrap() {
            assert_eq!(a_data.refcount(), 1);
            assert_eq!(b_data.refcount(), 2);
            drop(b);
            assert_eq!(a_data.refcount(), 1);
        } else {
            panic!("Expected add operation");
        }
    }

    #[test]
    fn test_add() {
        let a = tensor![1.0, 2.0, 3.0] + tensor![4.0, 5.0, 6.0];
        assert_eq!(
            a.tensor_data().expect("Expected tensor"),
            vec![5.0, 7.0, 9.0]
        );

        let b = scalar!(1.0) + tensor![4.0, 5.0, 6.0];
        assert_eq!(
            b.tensor_data().expect("Expected tensor"),
            vec![5.0, 6.0, 7.0]
        );

        let c = tensor![1.0, 2.0, 3.0] + scalar!(1.0);
        assert_eq!(
            c.tensor_data().expect("Expected tensor"),
            vec![2.0, 3.0, 4.0]
        );

        let d = scalar!(2.0) + scalar!(1.0);
        assert_eq!(d.scalar_data().expect("Expected scalar"), 3.0);

        assert!(matches!(a.op(), Some(Operation::Add(_, _))));
        assert!(matches!(b.op(), Some(Operation::Add(_, _))));
        assert!(matches!(c.op(), Some(Operation::Add(_, _))));
        assert!(matches!(d.op(), Some(Operation::Add(_, _))));
    }

    #[test]
    fn test_sub() {
        let a = tensor![1.0, 2.0, 3.0] - tensor![4.0, 5.0, 6.0];
        assert_eq!(
            a.tensor_data().expect("Expected tensor"),
            vec![-3.0, -3.0, -3.0]
        );

        let b = scalar!(1.0) - tensor![4.0, 5.0, 6.0];
        assert_eq!(
            b.tensor_data().expect("Expected tensor"),
            vec![-3.0, -4.0, -5.0]
        );

        let c = tensor![1.0, 2.0, 3.0] - scalar!(1.0);
        assert_eq!(
            c.tensor_data().expect("Expected tensor"),
            vec![0.0, 1.0, 2.0]
        );

        let d = scalar!(2.0) - scalar!(1.0);
        assert_eq!(d.scalar_data().expect("Expected scalar"), 1.0);

        assert!(matches!(a.op(), Some(Operation::Sub(_, _))));
        assert!(matches!(b.op(), Some(Operation::Sub(_, _))));
        assert!(matches!(c.op(), Some(Operation::Sub(_, _))));
        assert!(matches!(d.op(), Some(Operation::Sub(_, _))));
    }

    #[test]
    fn test_mul() {
        let a = tensor![4.0, 2.0, 3.0] * tensor![4.0, 5.0, 6.0];
        assert_eq!(
            a.tensor_data().expect("Expected tensor"),
            vec![16.0, 10.0, 18.0]
        );

        let b = scalar!(2.0) * tensor![4.0, 5.0, 6.0];
        assert_eq!(
            b.tensor_data().expect("Expected tensor"),
            vec![8.0, 10.0, 12.0]
        );

        let c = tensor![1.0, 2.0, 3.0] * scalar!(2.0);
        assert_eq!(
            c.tensor_data().expect("Expected tensor"),
            vec![2.0, 4.0, 6.0]
        );

        let d = scalar!(2.0) * scalar!(4.0);
        assert_eq!(d.scalar_data().expect("Expected scalar"), 8.0);

        assert!(matches!(a.op(), Some(Operation::Mul(_, _))));
        assert!(matches!(b.op(), Some(Operation::Mul(_, _))));
        assert!(matches!(c.op(), Some(Operation::Mul(_, _))));
        assert!(matches!(d.op(), Some(Operation::Mul(_, _))));
    }

    #[test]
    fn test_div() {
        let a = tensor![1.0, 2.0, 3.0] / tensor![4.0, 16.0, 6.0];
        assert_eq!(
            a.tensor_data().expect("Expected tensor"),
            vec![0.25, 0.125, 0.5]
        );

        let b = scalar!(60.0) / tensor![4.0, 5.0, 6.0];
        assert_eq!(
            b.tensor_data().expect("Expected tensor"),
            vec![15.0, 12.0, 10.0]
        );

        let c = tensor![1.0, 2.0, 3.0] / scalar!(0.5);
        assert_eq!(
            c.tensor_data().expect("Expected tensor"),
            vec![2.0, 4.0, 6.0]
        );

        let d = scalar!(2.0) / scalar!(0.5);
        assert_eq!(d.scalar_data().expect("Expected scalar"), 4.0);

        assert!(matches!(a.op(), Some(Operation::Div(_, _))));
        assert!(matches!(b.op(), Some(Operation::Div(_, _))));
        assert!(matches!(c.op(), Some(Operation::Div(_, _))));
        assert!(matches!(d.op(), Some(Operation::Div(_, _))));

        // test division by 0 (should return all NaNs)
        let a = tensor![1.0, 2.0, 3.0] / tensor![0.0, 0.0, 0.0];
        assert!(a
            .tensor_data()
            .expect("Expected tensor")
            .iter()
            .all(|&x| x.is_nan()));

        let b = scalar!(60.0) / tensor![0.0, 0.0, 0.0];
        assert!(b
            .tensor_data()
            .expect("Expected tensor")
            .iter()
            .all(|&x| x.is_nan()));

        let c = tensor![1.0, 2.0, 3.0] / scalar!(0.0);
        assert!(c
            .tensor_data()
            .expect("Expected tensor")
            .iter()
            .all(|&x| x.is_nan()));

        let d = scalar!(2.0) / scalar!(0.0);
        assert!(d.scalar_data().expect("Expected scalar").is_nan());

        // Test partial division by 0 (should return some NaNs)
        let a = tensor![1.0, 2.0, 3.0] / tensor![0.0, 1.0, 0.0];
        let a_data = a.tensor_data().expect("Expected tensor");
        assert!(a_data[0].is_nan());
        assert_eq!(a_data[1], 2.0);
        assert!(a_data[2].is_nan());

        let b = scalar!(60.0) / tensor![0.0, 1.0, 0.0];
        let b_data = b.tensor_data().expect("Expected tensor");
        assert!(b_data[0].is_nan());
        assert_eq!(b_data[1], 60.0);
        assert!(b_data[2].is_nan());
    }

    #[test]
    fn test_pow() {
        let a = pow(&tensor![2.0, 3.0, 4.0], &tensor![4.0, 5.0, 6.0]);
        assert_eq!(
            a.tensor_data().expect("Expected tensor"),
            vec![16.0, 243.0, 4096.0]
        );

        let b = pow(&tensor![2.0, 3.0, 4.0], &2.0.into());
        assert_eq!(
            b.tensor_data().expect("Expected tensor"),
            vec![4.0, 9.0, 16.0]
        );

        let c = pow(&2.0.into(), &tensor![4.0, 5.0, 6.0]);
        assert_eq!(
            c.tensor_data().expect("Expected tensor"),
            vec![16.0, 32.0, 64.0]
        );

        let d = pow(&2.0.into(), &2.0.into());
        assert_eq!(d.scalar_data().expect("Expected scalar"), 4.0);

        assert!(matches!(a.op(), Some(Operation::Pow(_, _))));
        assert!(matches!(b.op(), Some(Operation::Pow(_, _))));
        assert!(matches!(c.op(), Some(Operation::Pow(_, _))));
        assert!(matches!(d.op(), Some(Operation::Pow(_, _))));

        // test pow(0, 0) (should return NaN)
        let a = pow(&tensor![0.0, 0.0, 0.0], &tensor![0.0, 0.0, 0.0]);
        assert!(a
            .tensor_data()
            .expect("Expected tensor")
            .iter()
            .all(|&x| x.is_nan()));

        let b = pow(&tensor![0.0, 0.0, 0.0], &0.0.into());
        assert!(b
            .tensor_data()
            .expect("Expected tensor")
            .iter()
            .all(|&x| x.is_nan()));

        let c = pow(&0.0.into(), &tensor![0.0, 0.0, 0.0]);
        assert!(c
            .tensor_data()
            .expect("Expected tensor")
            .iter()
            .all(|&x| x.is_nan()));

        let d = pow(&0.0.into(), &0.0.into());
        assert!(d.scalar_data().expect("Expected scalar").is_nan());

        // Test sqrt of negative number (should return NaN)
        let a = pow(
            &tensor![{ -2.0 }, { -3.0 }, { -4.0 }],
            &tensor![0.5, 0.5, 0.5],
        );
        assert!(a
            .tensor_data()
            .expect("Expected tensor")
            .iter()
            .all(|&x| x.is_nan()));

        // Test pow of NaN (should return NaN)
        let a = pow(&f32::NAN.into(), &1.0.into());
        assert!(a.scalar_data().expect("Expected scalar").is_nan());

        // Test pow with exponent NaN (should return NaN)
        let a = pow(&1.0.into(), &f32::NAN.into());
        println!("{:?}", a);
        assert!(a.scalar_data().expect("Expected scalar").is_nan());

        // Test pow with both NaN (should return NaN)
        let a = pow(&f32::NAN.into(), &f32::NAN.into());
        assert!(a.scalar_data().expect("Expected scalar").is_nan());

        // Test pow with base < 1 and exponent infinity (should return 0)
        let a = pow(&0.5.into(), &f32::INFINITY.into());
        assert_eq!(a.scalar_data().expect("Expected scalar"), 0.0);

        // Test pow with base < 1 (and > 0) and exponent -infinity (should return infinity)
        let a = pow(&0.5.into(), &f32::NEG_INFINITY.into());
        assert_eq!(a.scalar_data().expect("Expected scalar"), f32::INFINITY);

        // Test pow with base > 1 and exponent infinity (should return infinity)
        let a = pow(&2.0.into(), &f32::INFINITY.into());
        assert_eq!(a.scalar_data().expect("Expected scalar"), f32::INFINITY);

        // Test pow with base > 1 and exponent -infinity (should return 0)
        let a = pow(&2.0.into(), &f32::NEG_INFINITY.into());
        assert_eq!(a.scalar_data().expect("Expected scalar"), 0.0);
    }

    #[test]
    fn test_sin() {
        use std::f32::consts::{FRAC_PI_2, PI};

        let a = sin(&tensor![0.0, FRAC_PI_2, PI]);
        assert!(are_close_vec(
            a.tensor_data().expect("Expected tensor"),
            vec![0.0, 1.0, 0.0],
            1e-5
        ));

        let b = sin(&0.0.into());
        assert_eq!(b.scalar_data().expect("Expected scalar"), 0.0);

        assert!(matches!(a.op(), Some(Operation::Sin(_))));
        assert!(matches!(b.op(), Some(Operation::Sin(_))));

        // Test sin of NaN (should return NaN)
        let a = sin(&f32::NAN.into());
        assert!(a.scalar_data().expect("Expected scalar").is_nan());

        // Test sin of infinity (should return NaN)
        let a = sin(&f32::INFINITY.into());
        assert!(a.scalar_data().expect("Expected scalar").is_nan());

        // Test sin of -infinity (should return NaN)
        let a = sin(&f32::NEG_INFINITY.into());
        assert!(a.scalar_data().expect("Expected scalar").is_nan());
    }

    #[test]
    fn test_cos() {
        use std::f32::consts::{FRAC_PI_2, PI};

        let a = cos(&tensor![0.0, FRAC_PI_2, PI]);
        assert!(are_close_vec(
            a.tensor_data().expect("Expected tensor"),
            vec![1.0, 0.0, -1.0],
            1e-5
        ));

        let b = cos(&0.0.into());
        assert_eq!(b.scalar_data().expect("Expected scalar"), 1.0);

        assert!(matches!(a.op(), Some(Operation::Cos(_))));
        assert!(matches!(b.op(), Some(Operation::Cos(_))));

        // Test cos of NaN (should return NaN)
        let a = cos(&f32::NAN.into());
        assert!(a.scalar_data().expect("Expected scalar").is_nan());

        // Test cos of infinity (should return NaN)
        let a = cos(&f32::INFINITY.into());
        assert!(a.scalar_data().expect("Expected scalar").is_nan());

        // Test cos of -infinity (should return NaN)
        let a = cos(&f32::NEG_INFINITY.into());
        assert!(a.scalar_data().expect("Expected scalar").is_nan());
    }

    #[test]
    fn test_tan() {
        use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

        let a = tan(&tensor![0.0, { PI / 4.0 }, PI]);
        assert!(are_close_vec(
            a.tensor_data().expect("Expected tensor"),
            vec![0.0, 1.0, 0.0],
            1e-5
        ));

        let b = tan(&FRAC_PI_4.into());
        assert_eq!(b.scalar_data().expect("Expected scalar"), 1.0);

        assert!(matches!(a.op(), Some(Operation::Tan(_))));
        assert!(matches!(b.op(), Some(Operation::Tan(_))));

        // Test tan of PI/2 (should be some large number)
        let c = tan(&FRAC_PI_2.into());
        assert!(c.scalar_data().expect("Expected scalar").abs() > 1e6);

        // Test tan of NaN (should return NaN)
        let a = tan(&f32::NAN.into());
        assert!(a.scalar_data().expect("Expected scalar").is_nan());

        // Test tan of infinity (should return NaN)
        let a = tan(&f32::INFINITY.into());
        assert!(a.scalar_data().expect("Expected scalar").is_nan());

        // Test tan of -infinity (should return NaN)
        let a = tan(&f32::NEG_INFINITY.into());
        assert!(a.scalar_data().expect("Expected scalar").is_nan());
    }

    #[test]
    fn test_asin() {
        use std::f32::consts::PI;

        let a = asin(&tensor![0.0, 0.5, 1.0]);
        println!("{:?}", a.tensor_data().expect("Expected tensor"));
        assert!(are_close_vec(
            a.tensor_data().expect("Expected tensor"),
            vec![0.0, PI / 6.0, PI / 2.0],
            1e-5
        ));

        let b = asin(&0.5.into());
        assert_eq!(b.scalar_data().expect("Expected scalar"), PI / 6.0);

        assert!(matches!(a.op(), Some(Operation::Asin(_))));
        assert!(matches!(b.op(), Some(Operation::Asin(_))));

        // Test asin of value outside [-1, 1] (should return NaN)
        let c = asin(&tensor![{ -2.0 }, 3.0, { -4.0 }]);
        assert!(c
            .tensor_data()
            .expect("Expected tensor")
            .iter()
            .all(|&x| x.is_nan()));

        // Test asin of NaN (should return NaN)
        let a = asin(&f32::NAN.into());
        assert!(a.scalar_data().expect("Expected scalar").is_nan());

        // Test asin of infinity (should return NaN)
        let a = asin(&f32::INFINITY.into());
        assert!(a.scalar_data().expect("Expected scalar").is_nan());

        // Test asin of -infinity (should return NaN)
        let a = asin(&f32::NEG_INFINITY.into());
        assert!(a.scalar_data().expect("Expected scalar").is_nan());
    }

    #[test]
    fn test_acos() {
        use std::f32::consts::PI;

        let a = acos(&tensor![0.0, 0.5, 1.0]);
        assert!(are_close_vec(
            a.tensor_data().expect("Expected tensor"),
            vec![PI / 2.0, PI / 3.0, 0.0],
            1e-5
        ));

        let b = acos(&0.5.into());
        assert_eq!(b.scalar_data().expect("Expected scalar"), PI / 3.0);

        assert!(matches!(a.op(), Some(Operation::Acos(_))));
        assert!(matches!(b.op(), Some(Operation::Acos(_))));

        // Test acos of value outside [-1, 1] (should return NaN)
        let c = acos(&tensor![{ -2.0 }, 3.0, { -4.0 }]);
        assert!(c
            .tensor_data()
            .expect("Expected tensor")
            .iter()
            .all(|&x| x.is_nan()));

        // Test acos of NaN (should return NaN)
        let a = acos(&f32::NAN.into());
        assert!(a.scalar_data().expect("Expected scalar").is_nan());

        // Test acos of infinity (should return NaN)
        let a = acos(&f32::INFINITY.into());
        assert!(a.scalar_data().expect("Expected scalar").is_nan());

        // Test acos of -infinity (should return NaN)
        let a = acos(&f32::NEG_INFINITY.into());
        assert!(a.scalar_data().expect("Expected scalar").is_nan());
    }

    #[test]
    fn test_atan() {
        use std::f32::consts::PI;

        let a = atan(&tensor![0.0, 1.0, 1e+8]);
        assert!(are_close_vec(
            a.tensor_data().expect("Expected tensor"),
            vec![0.0, PI / 4.0, PI / 2.0],
            1e-5
        ));

        let b = atan(&1.0.into());
        assert_eq!(b.scalar_data().expect("Expected scalar"), PI / 4.0);

        assert!(matches!(a.op(), Some(Operation::Atan(_))));
        assert!(matches!(b.op(), Some(Operation::Atan(_))));

        // Test atan of NaN (should return NaN)
        let a = atan(&f32::NAN.into());
        assert!(a.scalar_data().expect("Expected scalar").is_nan());

        // Test atan of infinity (should return PI/2)
        let a = atan(&f32::INFINITY.into());
        assert!(are_close_float(
            a.scalar_data().expect("Expected scalar"),
            PI / 2.0,
            1e-5
        ));

        // Test atan of -infinity (should return -PI/2)
        let a = atan(&f32::NEG_INFINITY.into());
        assert!(are_close_float(
            a.scalar_data().expect("Expected scalar"),
            -PI / 2.0,
            1e-5
        ));
    }

    #[test]
    fn test_sqrt() {
        let a = sqrt(&tensor![0.0, 1.0, 4.0]);
        assert!(are_close_vec(
            a.tensor_data().expect("Expected tensor"),
            vec![0.0, 1.0, 2.0],
            1e-5
        ));

        let b = sqrt(&4.0.into());
        assert_eq!(b.scalar_data().expect("Expected scalar"), 2.0);

        assert!(matches!(a.op(), Some(Operation::Sqrt(_))));
        assert!(matches!(b.op(), Some(Operation::Sqrt(_))));

        // Test sqrt of negative number (should return NaN)
        let c = sqrt(&tensor![{ -2.0 }, { -3.0 }, { -4.0 }]);
        assert!(c
            .tensor_data()
            .expect("Expected tensor")
            .iter()
            .all(|&x| x.is_nan()));

        // Test sqrt of NaN (should return NaN)
        let a = sqrt(&f32::NAN.into());
        assert!(a.scalar_data().expect("Expected scalar").is_nan());

        // Test sqrt of infinity (should return infinity)
        let a = sqrt(&f32::INFINITY.into());
        assert!(a.scalar_data().expect("Expected scalar").is_infinite());

        // Test sqrt of -infinity (should return NaN)
        let a = sqrt(&f32::NEG_INFINITY.into());
        assert!(a.scalar_data().expect("Expected scalar").is_nan());
    }

    #[test]
    fn test_exp() {
        let a = exp(&tensor![0.0, 1.0, 2.0]);
        assert!(are_close_vec(
            a.tensor_data().expect("Expected tensor"),
            vec![1.0, 2.718281828459045, 7.38905609893065],
            1e-5
        ));

        let b = exp(&2.0.into());
        assert_eq!(b.scalar_data().expect("Expected scalar"), 7.38905609893065);

        assert!(matches!(a.op(), Some(Operation::Exp(_))));
        assert!(matches!(b.op(), Some(Operation::Exp(_))));

        // Test exp of NaN (should return NaN)
        let a = exp(&f32::NAN.into());
        assert!(a.scalar_data().expect("Expected scalar").is_nan());

        // Test exp of infinity (should return infinity)
        let c = exp(&f32::INFINITY.into());
        assert_eq!(c.scalar_data().expect("Expected scalar"), f32::INFINITY);

        // Test exp of -infinity (should return 0)
        let d = exp(&f32::NEG_INFINITY.into());
        assert_eq!(d.scalar_data().expect("Expected scalar"), 0.0);
    }

    #[test]
    pub fn test_log() {
        let a = log(&tensor![1.0, 2.718281828459045, 7.38905609893065]);
        assert!(are_close_vec(
            a.tensor_data().expect("Expected tensor"),
            vec![0.0, 1.0, 2.0],
            1e-5
        ));

        let b = log(&7.38905609893065.into());
        assert_eq!(b.scalar_data().expect("Expected scalar"), 2.0);

        assert!(matches!(a.op(), Some(Operation::Log(_))));
        assert!(matches!(b.op(), Some(Operation::Log(_))));

        // Test log of NaN (should return NaN)
        let a = log(&f32::NAN.into());
        assert!(a.scalar_data().expect("Expected scalar").is_nan());

        // Test log of infinity (should return infinity)
        let a = log(&f32::INFINITY.into());
        assert_eq!(a.scalar_data().expect("Expected scalar"), f32::INFINITY);

        // Test log of -infinity (should return NaN)
        let a = log(&f32::NEG_INFINITY.into());
        assert!(a.scalar_data().expect("Expected scalar").is_nan());

        // Test log of 0 (should return -infinity)
        let a = log(&0.0.into());
        assert_eq!(a.scalar_data().expect("Expected scalar"), f32::NEG_INFINITY);
    }

    #[test]
    pub fn test_log1p() {
        let a = log1p(&tensor![0.0, 1.0, 2.0]);
        assert!(are_close_vec(
            a.tensor_data().expect("Expected tensor"),
            vec![0.0, 0.6931471805599453, 1.0986122886681098],
            1e-5
        ));

        let b = log1p(&2.0.into());
        assert_eq!(
            b.scalar_data().expect("Expected scalar"),
            1.0986122886681098
        );

        assert!(matches!(a.op(), Some(Operation::Log1p(_))));
        assert!(matches!(b.op(), Some(Operation::Log1p(_))));

        // Test log1p of NaN (should return NaN)
        let a = log1p(&f32::NAN.into());
        assert!(a.scalar_data().expect("Expected scalar").is_nan());

        // Test log1p of infinity (should return infinity)
        let a = log1p(&f32::INFINITY.into());
        assert_eq!(a.scalar_data().expect("Expected scalar"), f32::INFINITY);

        // Test log1p of -infinity (should return NaN)
        let a = log1p(&f32::NEG_INFINITY.into());
        assert!(a.scalar_data().expect("Expected scalar").is_nan());

        // Test log1p of -1 (should return -infinity)
        let a = log1p(&{ -1.0 }.into());
        assert_eq!(a.scalar_data().expect("Expected scalar"), f32::NEG_INFINITY);
    }

    #[test]
    pub fn test_abs() {
        let a = abs(&tensor![{ -1.0 }, 2.0, { -3.0 }]);
        assert!(are_close_vec(
            a.tensor_data().expect("Expected tensor"),
            vec![1.0, 2.0, 3.0],
            1e-5
        ));

        let b = abs(&2.0.into());
        assert_eq!(b.scalar_data().expect("Expected scalar"), 2.0);

        assert!(matches!(a.op(), Some(Operation::Abs(_))));
        assert!(matches!(b.op(), Some(Operation::Abs(_))));

        // Test abs of NaN (should return NaN)
        let a = abs(&f32::NAN.into());
        assert!(a.scalar_data().expect("Expected scalar").is_nan());

        // Test abs of infinity (should return infinity)
        let a = abs(&f32::INFINITY.into());
        assert_eq!(a.scalar_data().expect("Expected scalar"), f32::INFINITY);

        // Test abs of -infinity (should return infinity)
        let a = abs(&f32::NEG_INFINITY.into());
        assert_eq!(a.scalar_data().expect("Expected scalar"), f32::INFINITY);
    }

    #[test]
    pub fn test_transpose() {
        let a = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let b = transpose(&a);
        let c = tensor![[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]];
        assert_eq!(
            b.tensor_data().expect("Expected tensor"),
            c.tensor_data().expect("Expected tensor")
        );
        assert_eq!(b.layout(), c.layout());

        let d = transpose(&scalar!(1.0));
        assert_eq!(d.scalar_data().expect("Expected scalar"), 1.0);
        assert_eq!(d.layout(), shape![1]);

        // Test transpose of non-rectangular tensor
        let a = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = transpose(&a);
        let c = tensor![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
        assert_eq!(
            b.tensor_data().expect("Expected tensor"),
            c.tensor_data().expect("Expected tensor")
        );
        assert_eq!(b.layout(), c.layout());

        assert!(matches!(b.op(), Some(Operation::Transpose(_))));
        assert!(matches!(d.op(), Some(Operation::Transpose(_))));

        // TODO: Implement tests for higher-dimensional tensors (currently not supported).
    }

    #[test]
    pub fn test_matmul() {
        // Only test 2D matmul for now.
        let a = matmul(
            &tensor![[1.0, 2.0], [3.0, 4.0]],
            &tensor![[5.0, 6.0], [7.0, 8.0]],
        );

        assert_eq!(
            a.tensor_data().expect("Expected tensor"),
            vec![19.0, 22.0, 43.0, 50.0]
        );
        assert_eq!(a.layout(), shape![2, 2]);

        assert!(matches!(a.op(), Some(Operation::Matmul(_, _))));

        // Test matmul of NaN (should return NaN)
        let a = matmul(
            &tensor![[{ f32::NAN }, 2.0], [3.0, 4.0]],
            &tensor![[5.0, 6.0], [7.0, 8.0]],
        );
        let a_data = a.tensor_data().expect("Expected tensor");
        assert!(a_data[0].is_nan());
        assert!(a_data[1].is_nan());
        assert_eq!(a_data[2], 43.0);
        assert_eq!(a_data[3], 50.0);

        // Test matmul of infinity (should return infinity)
        let a = matmul(
            &tensor![[{ f32::INFINITY }, 2.0], [3.0, 4.0]],
            &tensor![[5.0, 6.0], [7.0, 8.0]],
        );
        let a_data = a.tensor_data().expect("Expected tensor");
        assert_eq!(a_data[0], f32::INFINITY);
        assert_eq!(a_data[1], f32::INFINITY);
        assert_eq!(a_data[2], 43.0);
        assert_eq!(a_data[3], 50.0);

        // Test matmul of -infinity (should return -infinity)
        let a = matmul(
            &tensor![[{ f32::NEG_INFINITY }, 2.0], [3.0, 4.0]],
            &tensor![[5.0, 6.0], [7.0, 8.0]],
        );
        let a_data = a.tensor_data().expect("Expected tensor");
        assert_eq!(a_data[0], f32::NEG_INFINITY);
        assert_eq!(a_data[1], f32::NEG_INFINITY);
        assert_eq!(a_data[2], 43.0);
        assert_eq!(a_data[3], 50.0);

        // TODO: Implement tests for higher-dimensional matmul (currently not supported).
    }

    #[test]
    #[should_panic]
    pub fn test_matmul_scalar() {
        matmul(&1.0.into(), &2.0.into());
    }
}
