#[macro_export]
macro_rules! count_tts {
    () => { 0 };
    ($odd:tt $(,$a:tt,$b:tt)* $(,)?) => { ($crate::count_tts!($($a),*) << 1) | 1 };
    ($($a:tt,$even:tt),* $(,)?) => { $crate::count_tts!($($a),*) << 1 };
}

#[macro_export]
macro_rules! linearize {
    ([ $( $(,)* [ $($innards:tt)* ] ),* $(,)? ]) => (
        $crate::linearize! {[ $( $($innards)* ),*]}
    );

    ([ $( $(,)* $e:tt ),* $(,)? ]) => (
        [$( $e ),*]
    );
}

#[macro_export]
macro_rules! shape_of {
    // // Deal with form (array; current_shape)
    // ([$a:tt $($(,)+ $b:tt)* $(,)?]; [$($(,)+ $shape:tt)* $(,)?]) => {
    //     // Now, move into the first inner element and append the length of the
    //     // full current state to the shape
    //     $crate::shape_of!($a; [$($shape,)* $crate::count_tts!($a $(,$b)*)])
    // };
    //
    // // Deal with form ([]; current_shape)
    // ([]; [ $($(,)* $shape:tt)* $(,)?]) => {
    //     [$($shape,)* 0]
    // };
    //
    // // Deal with form (tt; current_shape)
    // ($(,)* $a:tt $(,)?; [$($(,)* $shape:tt)* $(,)?]) => {
    //     [$($shape),*]
    // };
    //
    // // Check if it's 2D or more
    // ([$([$($(,)* $a:tt)* $(,)?]),* $(,)?]) => {
    //     // If it is, then convert to the form (array; current_shape)
    //     $crate::shape_of!([$([$($a),*]),*]; [])
    // };
    // // If not 2D or more, we can simply return the length
    // ([$($(,)* $a:tt)* $(,)?]) => {
    //     [$crate::count_tts!($($a),*)]
    // };

    ([ $(,)* [ $($first:tt)* ], $( $(,)* [ $($innards:tt)* ] ),* $(,)? ]; [ $( $(,)* $e:tt ),* $(,)? ]) => (
        $crate::shape_of! {
            [ $($first)* ];
            [$( $e ,)* {$crate::count_tts! {
                [ $($first)* ], $( [ $($innards)* ] ),*
            }}]
        }
    );

    ([ $( $(,)* $e:tt ),* $(,)? ]; [ $( $(,)* $f:tt ),* $(,)? ]) => (
        [$( $f ,)* $crate::count_tts! {
            $( $e ),*
        }]
    );

    ([ $( $(,)* $e:tt ),* $(,)? ]) => (
        $crate::shape_of! {[$( $e ),*]; []}
    );
}

#[macro_export]
macro_rules! tensor_data {
    [$($x:tt),* $(,)?] => {{
        $crate::values::TensorData::new_with_data($crate::values::TensorLayout {
            shape: Vec::from($crate::shape_of!([$($x),*])),
        }, Vec::from($crate::linearize!([$($x),*])))
    }};
}

#[macro_export]
macro_rules! shape {
    [$($x:tt),* $(,)?] => {
        $crate::values::TensorLayout { shape: vec![$($x),*] }
    }
}

#[macro_export]
macro_rules! tensor {
    [$($x:tt),*] => {
        $crate::values::Value::Tensor($crate::tensor_data![$($x),*])
    };
}

#[macro_export]
macro_rules! scalar {
    ($x:tt) => {
        $crate::values::Value::Scalar($crate::scalar_data!($x))
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_count_tts() {
        assert_eq!(count_tts!(), 0);
        assert_eq!(count_tts!(1, 2, 3), 3);
        assert_eq!(count_tts!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 10);
    }

    #[test]
    fn test_linearize() {
        assert_eq!(linearize!([[1, 2, 3]]), [1, 2, 3]);
        assert_eq!(linearize!([[1, 2, 3], [4, 5, 6]]), [1, 2, 3, 4, 5, 6]);
        assert_eq!(
            linearize!([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        );
    }

    #[test]
    fn test_shape_of() {
        let a = Vec::from(shape_of!([[1, 2, 3], [4, 5, 6]]));
        assert_eq!(a, vec![2, 3]);
        let b = Vec::from(shape_of!([1, 2, 3]));
        assert_eq!(b, vec![3]);
        let c = Vec::from(shape_of!([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]]
        ]));
        assert_eq!(c, vec![2, 2, 3]);
    }
}
