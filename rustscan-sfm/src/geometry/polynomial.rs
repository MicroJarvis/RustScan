use crate::colmap_eigen;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };

    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }
}

impl std::ops::Add for Complex64 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl std::ops::Sub for Complex64 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl std::ops::Mul for Complex64 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl std::ops::Div for Complex64 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let denom = rhs.re * rhs.re + rhs.im * rhs.im;
        Self::new(
            (self.re * rhs.re + self.im * rhs.im) / denom,
            (self.im * rhs.re - self.re * rhs.im) / denom,
        )
    }
}

pub fn real_roots_durand_kerner(coeffs: &[f64], imag_eps: f64) -> Vec<f64> {
    durand_kerner_roots(coeffs)
        .into_iter()
        .filter(|root| root.im.abs() <= imag_eps)
        .map(|root| root.re)
        .filter(|value| value.is_finite())
        .collect()
}

pub fn real_roots_companion_matrix(coeffs: &[f64], imag_eps: f64) -> Vec<f64> {
    complex_roots_companion_matrix(coeffs)
        .into_iter()
        .filter(|root| root.im.abs() <= imag_eps)
        .map(|root| root.re)
        .filter(|value| value.is_finite())
        .collect()
}

pub fn complex_roots_companion_matrix(coeffs: &[f64]) -> Vec<Complex64> {
    let coeffs = trim_leading_zeros(coeffs);
    if coeffs.len() < 2 {
        return Vec::new();
    }
    let degree = coeffs.len() - 1;
    if degree == 1 {
        return vec![Complex64::new(-coeffs[1] / coeffs[0], 0.0)];
    }
    if degree == 2 {
        return quadratic_roots_complex(coeffs);
    }

    let coeffs = trim_trailing_zeros(coeffs);
    if coeffs.len() <= 1 {
        return vec![Complex64::ZERO];
    }
    let effective_degree = coeffs.len() - 1;
    let lead = coeffs[0];
    if lead.abs() < 1.0e-15 {
        return Vec::new();
    }

    if let Some(mut roots) = colmap_eigen::companion_roots(coeffs).map(|roots| {
        roots
            .into_iter()
            .map(|(re, im)| Complex64::new(re, im))
            .filter(|value| value.re.is_finite() && value.im.is_finite())
            .collect::<Vec<_>>()
    }) {
        if effective_degree < degree {
            roots.push(Complex64::ZERO);
        }
        return roots;
    }

    let mut companion = nalgebra::DMatrix::<f64>::zeros(effective_degree, effective_degree);
    for i in 1..effective_degree {
        companion[(i, i - 1)] = 1.0;
    }
    for j in 0..effective_degree {
        companion[(0, j)] = -coeffs[j + 1] / lead;
    }

    let mut roots = companion
        .complex_eigenvalues()
        .iter()
        .map(|root| Complex64::new(root.re, root.im))
        .filter(|value| value.re.is_finite() && value.im.is_finite())
        .collect::<Vec<_>>();
    if effective_degree < degree {
        roots.push(Complex64::ZERO);
    }
    roots
}

pub fn durand_kerner_roots(coeffs: &[f64]) -> Vec<Complex64> {
    let coeffs = trim_leading_zeros(coeffs);
    if coeffs.len() < 2 {
        return Vec::new();
    }
    let degree = coeffs.len() - 1;
    if degree == 1 {
        return vec![Complex64::new(-coeffs[1] / coeffs[0], 0.0)];
    }

    let lead = coeffs[0];
    if lead.abs() < 1.0e-15 {
        return Vec::new();
    }
    let coeffs = coeffs.iter().map(|c| c / lead).collect::<Vec<_>>();
    let radius = 1.0 + coeffs[1..].iter().map(|c| c.abs()).fold(0.0, f64::max);
    let mut roots = (0..degree)
        .map(|idx| {
            let theta = 2.0 * std::f64::consts::PI * idx as f64 / degree as f64;
            Complex64::new(radius * theta.cos(), radius * theta.sin())
        })
        .collect::<Vec<_>>();

    for _ in 0..128 {
        let mut max_step = 0.0f64;
        for i in 0..degree {
            let mut denom = Complex64::ONE;
            for j in 0..degree {
                if i != j {
                    denom = denom * (roots[i] - roots[j]);
                }
            }
            if denom.norm() < 1.0e-18 {
                continue;
            }
            let step = eval_poly_complex(&coeffs, roots[i]) / denom;
            roots[i] = roots[i] - step;
            max_step = max_step.max(step.norm());
        }
        if max_step < 1.0e-12 {
            break;
        }
    }
    roots
}

fn eval_poly_complex(coeffs: &[f64], x: Complex64) -> Complex64 {
    let mut value = Complex64::ZERO;
    for &coeff in coeffs {
        value = value * x + Complex64::new(coeff, 0.0);
    }
    value
}

fn trim_leading_zeros(coeffs: &[f64]) -> &[f64] {
    let first = coeffs
        .iter()
        .position(|c| c.abs() > 1.0e-15)
        .unwrap_or(coeffs.len());
    &coeffs[first..]
}

fn trim_trailing_zeros(coeffs: &[f64]) -> &[f64] {
    let last_nonzero = coeffs.iter().rposition(|c| c.abs() > 1.0e-15);
    match last_nonzero {
        Some(last) => &coeffs[..=last],
        None => &[],
    }
}

fn quadratic_roots_complex(coeffs: &[f64]) -> Vec<Complex64> {
    let a = coeffs[0];
    let b = coeffs[1];
    let c = coeffs[2];
    if a.abs() < 1.0e-15 {
        return vec![Complex64::new(-c / b, 0.0)];
    }
    let disc = b * b - 4.0 * a * c;
    if disc >= 0.0 {
        let sqrt_disc = disc.sqrt();
        vec![
            Complex64::new((-b - sqrt_disc) / (2.0 * a), 0.0),
            Complex64::new((-b + sqrt_disc) / (2.0 * a), 0.0),
        ]
    } else {
        let real = -b / (2.0 * a);
        let imag = (-disc).sqrt() / (2.0 * a);
        vec![Complex64::new(real, imag), Complex64::new(real, -imag)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_real_roots_of_quartic() {
        let mut roots = real_roots_durand_kerner(&[1.0, -10.0, 35.0, -50.0, 24.0], 1.0e-8);
        roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(roots.len(), 4);
        for (actual, expected) in roots.iter().zip([1.0, 2.0, 3.0, 4.0]) {
            assert!((actual - expected).abs() < 1.0e-7, "{actual} != {expected}");
        }
    }

    #[test]
    fn ignores_complex_roots() {
        let roots = real_roots_durand_kerner(&[1.0, 0.0, 1.0], 1.0e-8);
        assert!(roots.is_empty());
    }

    #[test]
    fn companion_finds_real_roots_of_quartic() {
        let mut roots = real_roots_companion_matrix(&[1.0, -10.0, 35.0, -50.0, 24.0], 1.0e-8);
        roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(roots.len(), 4);
        for (actual, expected) in roots.iter().zip([1.0, 2.0, 3.0, 4.0]) {
            assert!((actual - expected).abs() < 1.0e-7, "{actual} != {expected}");
        }
    }

    #[test]
    fn companion_returns_complex_roots_for_filtering() {
        let roots = complex_roots_companion_matrix(&[1.0, 0.0, 1.0]);

        assert_eq!(roots.len(), 2);
        assert!(roots.iter().all(|root| root.re.abs() < 1.0e-12));
        assert!(roots.iter().any(|root| (root.im - 1.0).abs() < 1.0e-12));
        assert!(roots.iter().any(|root| (root.im + 1.0).abs() < 1.0e-12));
    }
}
