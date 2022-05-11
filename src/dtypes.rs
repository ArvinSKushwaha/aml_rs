//! TODO: Add documentation

/// A trait representing valid tensor values.
///
/// TODO: Add documentation
pub trait Dtype: Copy + Send + Sync + Default + 'static {
    fn bool(&self) -> bool;
    fn u8(&self) -> u8;
    fn u16(&self) -> u16;
    fn u32(&self) -> u32;
    fn u64(&self) -> u64;
    fn i8(&self) -> i8;
    fn i16(&self) -> i16;
    fn i32(&self) -> i32;
    fn i64(&self) -> i64;
    fn f32(&self) -> f32;
    fn f64(&self) -> f64;
}

// Implement Dtype for all primitive numeric types.
impl Dtype for bool {
    fn bool(&self) -> bool {
        *self
    }
    fn u8(&self) -> u8 {
        *self as u8
    }
    fn u16(&self) -> u16 {
        *self as u16
    }
    fn u32(&self) -> u32 {
        *self as u32
    }
    fn u64(&self) -> u64 {
        *self as u64
    }
    fn i8(&self) -> i8 {
        *self as i8
    }
    fn i16(&self) -> i16 {
        *self as i16
    }
    fn i32(&self) -> i32 {
        *self as i32
    }
    fn i64(&self) -> i64 {
        *self as i64
    }
    fn f32(&self) -> f32 {
        *self as u8 as f32
    }
    fn f64(&self) -> f64 {
        *self as u8 as f64
    }
}

impl Dtype for i8 {
    fn bool(&self) -> bool {
        *self != 0
    }
    fn u8(&self) -> u8 {
        *self as u8
    }
    fn u16(&self) -> u16 {
        *self as u16
    }
    fn u32(&self) -> u32 {
        *self as u32
    }
    fn u64(&self) -> u64 {
        *self as u64
    }
    fn i8(&self) -> i8 {
        *self
    }
    fn i16(&self) -> i16 {
        *self as i16
    }
    fn i32(&self) -> i32 {
        *self as i32
    }
    fn i64(&self) -> i64 {
        *self as i64
    }
    fn f32(&self) -> f32 {
        *self as f32
    }
    fn f64(&self) -> f64 {
        *self as f64
    }
}

impl Dtype for i16 {
    fn bool(&self) -> bool {
        *self != 0
    }
    fn u8(&self) -> u8 {
        *self as u8
    }
    fn u16(&self) -> u16 {
        *self as u16
    }
    fn u32(&self) -> u32 {
        *self as u32
    }
    fn u64(&self) -> u64 {
        *self as u64
    }
    fn i8(&self) -> i8 {
        *self as i8
    }
    fn i16(&self) -> i16 {
        *self
    }
    fn i32(&self) -> i32 {
        *self as i32
    }
    fn i64(&self) -> i64 {
        *self as i64
    }
    fn f32(&self) -> f32 {
        *self as f32
    }
    fn f64(&self) -> f64 {
        *self as f64
    }
}

impl Dtype for i32 {
    fn bool(&self) -> bool {
        *self != 0
    }
    fn u8(&self) -> u8 {
        *self as u8
    }
    fn u16(&self) -> u16 {
        *self as u16
    }
    fn u32(&self) -> u32 {
        *self as u32
    }
    fn u64(&self) -> u64 {
        *self as u64
    }
    fn i8(&self) -> i8 {
        *self as i8
    }
    fn i16(&self) -> i16 {
        *self as i16
    }
    fn i32(&self) -> i32 {
        *self
    }
    fn i64(&self) -> i64 {
        *self as i64
    }
    fn f32(&self) -> f32 {
        *self as f32
    }
    fn f64(&self) -> f64 {
        *self as f64
    }
}

impl Dtype for i64 {
    fn bool(&self) -> bool {
        *self != 0
    }
    fn u8(&self) -> u8 {
        *self as u8
    }
    fn u16(&self) -> u16 {
        *self as u16
    }
    fn u32(&self) -> u32 {
        *self as u32
    }
    fn u64(&self) -> u64 {
        *self as u64
    }
    fn i8(&self) -> i8 {
        *self as i8
    }
    fn i16(&self) -> i16 {
        *self as i16
    }
    fn i32(&self) -> i32 {
        *self as i32
    }
    fn i64(&self) -> i64 {
        *self
    }
    fn f32(&self) -> f32 {
        *self as f32
    }
    fn f64(&self) -> f64 {
        *self as f64
    }
}

impl Dtype for u8 {
    fn bool(&self) -> bool {
        *self != 0
    }
    fn u8(&self) -> u8 {
        *self
    }
    fn u16(&self) -> u16 {
        *self as u16
    }
    fn u32(&self) -> u32 {
        *self as u32
    }
    fn u64(&self) -> u64 {
        *self as u64
    }
    fn i8(&self) -> i8 {
        *self as i8
    }
    fn i16(&self) -> i16 {
        *self as i16
    }
    fn i32(&self) -> i32 {
        *self as i32
    }
    fn i64(&self) -> i64 {
        *self as i64
    }
    fn f32(&self) -> f32 {
        *self as f32
    }
    fn f64(&self) -> f64 {
        *self as f64
    }
}

impl Dtype for u16 {
    fn bool(&self) -> bool {
        *self != 0
    }
    fn u8(&self) -> u8 {
        *self as u8
    }
    fn u16(&self) -> u16 {
        *self
    }
    fn u32(&self) -> u32 {
        *self as u32
    }
    fn u64(&self) -> u64 {
        *self as u64
    }
    fn i8(&self) -> i8 {
        *self as i8
    }
    fn i16(&self) -> i16 {
        *self as i16
    }
    fn i32(&self) -> i32 {
        *self as i32
    }
    fn i64(&self) -> i64 {
        *self as i64
    }
    fn f32(&self) -> f32 {
        *self as f32
    }
    fn f64(&self) -> f64 {
        *self as f64
    }
}

impl Dtype for u32 {
    fn bool(&self) -> bool {
        *self != 0
    }
    fn u8(&self) -> u8 {
        *self as u8
    }
    fn u16(&self) -> u16 {
        *self as u16
    }
    fn u32(&self) -> u32 {
        *self
    }
    fn u64(&self) -> u64 {
        *self as u64
    }
    fn i8(&self) -> i8 {
        *self as i8
    }
    fn i16(&self) -> i16 {
        *self as i16
    }
    fn i32(&self) -> i32 {
        *self as i32
    }
    fn i64(&self) -> i64 {
        *self as i64
    }
    fn f32(&self) -> f32 {
        *self as f32
    }
    fn f64(&self) -> f64 {
        *self as f64
    }
}

impl Dtype for u64 {
    fn bool(&self) -> bool {
        *self != 0
    }
    fn u8(&self) -> u8 {
        *self as u8
    }
    fn u16(&self) -> u16 {
        *self as u16
    }
    fn u32(&self) -> u32 {
        *self as u32
    }
    fn u64(&self) -> u64 {
        *self
    }
    fn i8(&self) -> i8 {
        *self as i8
    }
    fn i16(&self) -> i16 {
        *self as i16
    }
    fn i32(&self) -> i32 {
        *self as i32
    }
    fn i64(&self) -> i64 {
        *self as i64
    }
    fn f32(&self) -> f32 {
        *self as f32
    }
    fn f64(&self) -> f64 {
        *self as f64
    }
}

impl Dtype for f32 {
    fn bool(&self) -> bool {
        *self != 0.
    }
    fn u8(&self) -> u8 {
        *self as u8
    }
    fn u16(&self) -> u16 {
        *self as u16
    }
    fn u32(&self) -> u32 {
        *self as u32
    }
    fn u64(&self) -> u64 {
        *self as u64
    }
    fn i8(&self) -> i8 {
        *self as i8
    }
    fn i16(&self) -> i16 {
        *self as i16
    }
    fn i32(&self) -> i32 {
        *self as i32
    }
    fn i64(&self) -> i64 {
        *self as i64
    }
    fn f32(&self) -> f32 {
        *self
    }
    fn f64(&self) -> f64 {
        *self as f64
    }
}

impl Dtype for f64 {
    fn bool(&self) -> bool {
        *self != 0.
    }
    fn u8(&self) -> u8 {
        *self as u8
    }
    fn u16(&self) -> u16 {
        *self as u16
    }
    fn u32(&self) -> u32 {
        *self as u32
    }
    fn u64(&self) -> u64 {
        *self as u64
    }
    fn i8(&self) -> i8 {
        *self as i8
    }
    fn i16(&self) -> i16 {
        *self as i16
    }
    fn i32(&self) -> i32 {
        *self as i32
    }
    fn i64(&self) -> i64 {
        *self as i64
    }
    fn f32(&self) -> f32 {
        *self as f32
    }
    fn f64(&self) -> f64 {
        *self
    }
}
