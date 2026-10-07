//! Mathematical validation of scalar-polynomial Crouzeix fixtures (03-sandyrust-math-plan §2).
//!
//! Theorem (Crouzeix 2004, 2026): For any matrix A in M_n(C) and polynomial p(z):
//!   ||p(A)||_2 <= 2 * sup_{z in W(A)} |p(z)|,
//! where W(A) = { u^* A u : ||u||_2 = 1 } is the numerical range.
//!
//! Note: This module provides pure-Rust FP64 reference fixtures for candidate
//! recurrent state-space models. It is intentionally kept independent of the
//! baseline transformer forward path.

#[derive(Debug, Clone, Copy, PartialEq)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    fn one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }

    fn conj(self) -> Self {
        Self { re: self.re, im: -self.im }
    }

    fn norm_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    fn abs(self) -> f64 {
        self.norm_sq().sqrt()
    }

    fn add(self, other: Self) -> Self {
        Self { re: self.re + other.re, im: self.im + other.im }
    }

    fn sub(self, other: Self) -> Self {
        Self { re: self.re - other.re, im: self.im - other.im }
    }

    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    #[allow(dead_code)]
    fn scale(self, s: f64) -> Self {
        Self { re: self.re * s, im: self.im * s }
    }
}

/// 2x2 complex matrix
#[derive(Debug, Clone, Copy, PartialEq)]
struct Mat2 {
    m: [[Complex; 2]; 2],
}

impl Mat2 {
    fn zero() -> Self {
        Self { m: [[Complex::zero(); 2]; 2] }
    }

    fn identity() -> Self {
        Self {
            m: [
                [Complex::one(), Complex::zero()],
                [Complex::zero(), Complex::one()],
            ],
        }
    }

    fn add(self, other: Self) -> Self {
        let mut res = Self::zero();
        for r in 0..2 {
            for c in 0..2 {
                res.m[r][c] = self.m[r][c].add(other.m[r][c]);
            }
        }
        res
    }

    fn scale(self, s: Complex) -> Self {
        let mut res = Self::zero();
        for r in 0..2 {
            for c in 0..2 {
                res.m[r][c] = self.m[r][c].mul(s);
            }
        }
        res
    }

    fn mul(self, other: Self) -> Self {
        let mut res = Self::zero();
        for r in 0..2 {
            for c in 0..2 {
                let mut sum = Complex::zero();
                for k in 0..2 {
                    sum = sum.add(self.m[r][k].mul(other.m[k][c]));
                }
                res.m[r][c] = sum;
            }
        }
        res
    }

    fn conj_transpose(self) -> Self {
        Self {
            m: [
                [self.m[0][0].conj(), self.m[1][0].conj()],
                [self.m[0][1].conj(), self.m[1][1].conj()],
            ],
        }
    }

    /// Spectral operator norm ||M||_2 = sqrt(lambda_max(M^* M)).
    fn spectral_norm(self) -> f64 {
        let h = self.conj_transpose().mul(self);
        // H is 2x2 positive semi-definite Hermitian.
        // tr(H) = h00 + h11 (real), det(H) = h00*h11 - |h01|^2 (real)
        let tr = h.m[0][0].re + h.m[1][1].re;
        let det = h.m[0][0].re * h.m[1][1].re - h.m[0][1].norm_sq();
        let disc = (tr * tr - 4.0 * det).max(0.0).sqrt();
        let lambda_max = 0.5 * (tr + disc);
        lambda_max.max(0.0).sqrt()
    }
}

/// Evaluates scalar polynomial p(A) = c0 + c1*A + c2*A^2 + ... via direct power expansion.
fn eval_poly_direct(coeffs: &[Complex], a: Mat2) -> Mat2 {
    let mut acc = Mat2::zero();
    let mut power = Mat2::identity();
    for &c in coeffs {
        acc = acc.add(power.scale(c));
        power = power.mul(a);
    }
    acc
}

/// Evaluates scalar polynomial p(A) via Horner's method: p(A) = c0 + A*(c1 + A*(c2 + ...)).
fn eval_poly_horner(coeffs: &[Complex], a: Mat2) -> Mat2 {
    if coeffs.is_empty() {
        return Mat2::zero();
    }
    let mut acc = Mat2::identity().scale(*coeffs.last().unwrap());
    for &c in coeffs.iter().rev().skip(1) {
        acc = a.mul(acc).add(Mat2::identity().scale(c));
    }
    acc
}

#[test]
fn test_horner_matches_direct_power() {
    let a = Mat2 {
        m: [
            [Complex::new(0.5, 0.1), Complex::new(0.2, -0.3)],
            [Complex::new(-0.1, 0.4), Complex::new(0.3, 0.2)],
        ],
    };
    let coeffs = vec![
        Complex::new(1.0, 0.0),
        Complex::new(-0.5, 0.2),
        Complex::new(0.3, -0.1),
        Complex::new(0.1, 0.05),
    ];
    let direct = eval_poly_direct(&coeffs, a);
    let horner = eval_poly_horner(&coeffs, a);
    for r in 0..2 {
        for c in 0..2 {
            let diff = direct.m[r][c].sub(horner.m[r][c]).abs();
            assert!(diff < 1e-12, "Horner and direct power mismatch at ({r},{c})");
        }
    }
}

#[test]
fn test_fixture_a_jordan_block_sharp_constant_two() {
    // Fixture A: J = [[0, 1], [0, 0]], p(z) = z (coeffs = [0, 1])
    let j = Mat2 {
        m: [
            [Complex::zero(), Complex::one()],
            [Complex::zero(), Complex::zero()],
        ],
    };
    let coeffs = vec![Complex::zero(), Complex::one()];
    let p_j = eval_poly_direct(&coeffs, j);
    let norm_p_j = p_j.spectral_norm();

    // ||J||_2 = 1.0 exactly
    assert!((norm_p_j - 1.0).abs() < 1e-12);

    // Exact numerical range of J is the disk D(0, 1/2)
    // Thus sup_{z in W(J)} |p(z)| = 1/2 exactly.
    let sup_w = 0.5;

    // Ratio: ||p(J)||_2 / sup_{z in W(J)} |p(z)| = 1.0 / 0.5 = 2.0
    let ratio = norm_p_j / sup_w;
    assert!((ratio - 2.0).abs() < 1e-12, "Jordan block J must achieve exact factor 2");
    assert!(norm_p_j <= 2.0 * sup_w + 1e-12);
}

#[test]
fn test_fixture_b_scaled_jordan_and_powers() {
    // Fixture B: c * J, p(z) = z
    let c_val = Complex::new(1.5, -0.5);
    let j = Mat2 {
        m: [
            [Complex::zero(), c_val],
            [Complex::zero(), Complex::zero()],
        ],
    };
    let norm = j.spectral_norm();
    assert!((norm - c_val.abs()).abs() < 1e-12);

    // Radius of W(c*J) is |c| / 2
    let radius = c_val.abs() / 2.0;
    assert!((norm - 2.0 * radius).abs() < 1e-12);

    // For p(z) = z^k (k >= 2), matrix power J^k = 0
    let p_z2 = vec![Complex::zero(), Complex::zero(), Complex::one()];
    let j2 = eval_poly_direct(&p_z2, j);
    assert!(j2.spectral_norm() < 1e-12);
}

#[test]
fn test_fixture_c_scalar_identity() {
    // Fixture C: lambda * I, p(z)
    let lambda = Complex::new(0.7, -0.3);
    let a = Mat2::identity().scale(lambda);
    let coeffs = vec![Complex::new(1.0, 0.0), Complex::new(0.5, 0.0), Complex::new(-0.2, 0.1)];

    let p_a = eval_poly_direct(&coeffs, a);
    let norm_p_a = p_a.spectral_norm();

    // p(lambda)
    let mut p_lambda = Complex::zero();
    let mut pow = Complex::one();
    for &c in &coeffs {
        p_lambda = p_lambda.add(pow.mul(c));
        pow = pow.mul(lambda);
    }
    let sup_w = p_lambda.abs();

    assert!((norm_p_a - sup_w).abs() < 1e-12);
    assert!(norm_p_a <= 2.0 * sup_w + 1e-12);
}

#[test]
fn test_fixture_d_diagonal_matrix_and_kernel() {
    // Fixture D: diag(-1, 1), W = [-1, 1]
    let a = Mat2 {
        m: [
            [Complex::new(-1.0, 0.0), Complex::zero()],
            [Complex::zero(), Complex::new(1.0, 0.0)],
        ],
    };
    // p(z) = 1 - z^2
    let p_annihilate = vec![Complex::new(1.0, 0.0), Complex::zero(), Complex::new(-1.0, 0.0)];
    let res = eval_poly_direct(&p_annihilate, a);
    assert!(res.spectral_norm() < 1e-12, "diag(-1,1) must be annihilated by 1 - z^2");

    // Range supremum over [-1, 1] of |1 - x^2| is 1.0 (at x = 0)
    let sup_w = 1.0;
    assert!(res.spectral_norm() <= 2.0 * sup_w);
}

#[test]
fn test_numerical_range_sampling_warning() {
    // Demonstrates that sampled numerical range is a lower bound, not an upper bound.
    let _j = Mat2 {
        m: [
            [Complex::zero(), Complex::one()],
            [Complex::zero(), Complex::zero()],
        ],
    };
    // Sample with coarse grid of theta, phi
    let mut max_sampled = 0.0f64;
    for t_step in 0..10 {
        let theta = (t_step as f64) * std::f64::consts::FRAC_PI_2 / 10.0;
        for p_step in 0..10 {
            let phi = (p_step as f64) * std::f64::consts::PI * 2.0 / 10.0;
            let u0 = Complex::new(theta.cos(), 0.0);
            let u1 = Complex::new(theta.sin() * phi.cos(), theta.sin() * phi.sin());
            // u^* J u = conj(u0)*u1
            let z = u0.conj().mul(u1);
            max_sampled = max_sampled.max(z.abs());
        }
    }
    // Theoretical maximum radius is 0.5. Finite grid is strictly <= 0.5
    assert!(max_sampled <= 0.5 + 1e-12);
    // If a runner asserted ||J|| <= 2 * max_sampled with undersampled grid,
    // it could trigger false violations.
}
