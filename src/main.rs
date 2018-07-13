#[macro_use(array)]
extern crate ndarray;
use ndarray::Array2;

fn conjugate_gradients(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let a_dim = a.raw_dim();
    let shape = a_dim[1] as usize;
    let mut x = Array2::<f64>::zeros((shape, 1usize));
    let mut residuals = b - &a.dot(&x);
    let mut direction = residuals.clone();
    let mut error = residuals.t().dot(&residuals)[(0, 0)];

    //step along conjugate directions
    while error > 1e-8 {
        x = x + &direction * error / (direction.t().dot(a).dot(&direction));
        residuals = b - &a.dot(&x);
        let error1 = error;
        error = residuals.t().dot(&residuals)[(0, 0)];
        direction = &residuals + &(error / error1 * direction);
    }
    x
}

fn main() {
    let a = array![
        [0.01203809, 0.23617594, 0.62920203],
        [0.15002569, 0.72338376, 0.80522701],
        [0.47297399, 0.39777597, 0.52241356],
        [0.04486596, 0.4072091, 0.59443388],
        [0.07687325, 0.32479487, 0.56360606]
    ];

    let b = array![
        [0.40193925],
        [0.1467617],
        [0.66663369],
        [0.55384503],
        [0.55692321]
    ];
    // make system positive semidefinite
    let res = conjugate_gradients(&a.t().dot(&a), &a.t().dot(&b));
    assert_eq!(
        array![
            [1.00481519235758736],
            [-1.3641655835252542],
            [1.3960188889633491]
        ],
        res
    );
}
