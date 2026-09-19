// ============================================================================
// QuadricT - 二次误差度量
// 用于网格简化（Decimation）中的误差计算
// ============================================================================

use crate::{Point3, Vec3};

/// QuadricT - 存储 4x4 对称矩阵的上三角部分
///
/// 二次误差度量用于计算顶点合并的误差：
/// - 每个面贡献一个平面误差矩阵
/// - 合并顶点时，累加所有相关面的误差
/// - 找到最小化误差的新顶点位置
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadricT {
    // 4x4 对称矩阵的上三角部分（10 个独立值）
    a: f32,
    b: f32,
    c: f32,
    d: f32, // Row 1: a b c d
    e: f32,
    f: f32,
    g: f32, // Row 2: b e f g
    h: f32,
    i: f32, // Row 3: c f h i
    j: f32, // Row 4: d g i j
}

impl QuadricT {
    /// 从 4x4 对称矩阵的上三角创建
    #[inline]
    pub fn new(
        a: f32,
        b: f32,
        c: f32,
        d: f32,
        e: f32,
        f: f32,
        g: f32,
        h: f32,
        i: f32,
        j: f32,
    ) -> Self {
        Self {
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            h,
            i,
            j,
        }
    }

    /// 从平面方程 ax + by + cz + d = 0 创建
    ///
    /// 误差矩阵 Q = p * p^T，其中 p = (a, b, c, d)
    #[inline]
    pub fn from_plane(a: f32, b: f32, c: f32, d: f32) -> Self {
        Self::new(
            a * a,
            a * b,
            a * c,
            a * d,
            b * b,
            b * c,
            b * d,
            c * c,
            c * d,
            d * d,
        )
    }

    /// 从点和法线定义的面创建
    #[inline]
    pub fn from_face(normal: Vec3, point: Point3) -> Self {
        // 平面方程：ax + by + cz + d = 0
        // d = -dot(normal, point)
        let a = normal.x;
        let b = normal.y;
        let c = normal.z;
        let d = -normal.dot(&point.coords);
        Self::from_plane(a, b, c, d)
    }

    /// 从点创建（点自身的误差函数）
    #[inline]
    pub fn from_point(pt: Point3) -> Self {
        Self::new(
            1.0,
            0.0,
            0.0,
            -pt.x,
            1.0,
            0.0,
            -pt.y,
            1.0,
            -pt.z,
            pt.x * pt.x + pt.y * pt.y + pt.z * pt.z,
        )
    }

    /// 返回新的清零 Quadric
    #[inline]
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    /// 加法
    #[inline]
    pub fn add_values(&self, other: Self) -> Self {
        Self::new(
            self.a + other.a,
            self.b + other.b,
            self.c + other.c,
            self.d + other.d,
            self.e + other.e,
            self.f + other.f,
            self.g + other.g,
            self.h + other.h,
            self.i + other.i,
            self.j + other.j,
        )
    }

    /// 加法赋值
    #[inline]
    pub fn add_assign_values(&mut self, other: Self) {
        self.a += other.a;
        self.b += other.b;
        self.c += other.c;
        self.d += other.d;
        self.e += other.e;
        self.f += other.f;
        self.g += other.g;
        self.h += other.h;
        self.i += other.i;
        self.j += other.j;
    }

    /// 标量乘法
    #[inline]
    pub fn mul_scalar(&self, s: f32) -> Self {
        Self::new(
            self.a * s,
            self.b * s,
            self.c * s,
            self.d * s,
            self.e * s,
            self.f * s,
            self.g * s,
            self.h * s,
            self.i * s,
            self.j * s,
        )
    }

    /// 标量乘法赋值
    #[inline]
    pub fn mul_assign_scalar(&mut self, s: f32) {
        self.a *= s;
        self.b *= s;
        self.c *= s;
        self.d *= s;
        self.e *= s;
        self.f *= s;
        self.g *= s;
        self.h *= s;
        self.i *= s;
        self.j *= s;
    }

    /// 评估二次函数 Q(v) = v^T * Q * v
    /// v 是 3D 位置
    #[inline]
    pub fn value(&self, v: Point3) -> f32 {
        let x = v.x;
        let y = v.y;
        let z = v.z;

        self.a * x * x
            + 2.0 * self.b * x * y
            + 2.0 * self.c * x * z
            + 2.0 * self.d * x
            + self.e * y * y
            + 2.0 * self.f * y * z
            + 2.0 * self.g * y
            + self.h * z * z
            + 2.0 * self.i * z
            + self.j
    }

    /// 评估二次函数 Q(v) = v^T * Q * v（齐次坐标）
    #[inline]
    pub fn value4(&self, x: f32, y: f32, z: f32, w: f32) -> f32 {
        self.a * x * x
            + 2.0 * self.b * x * y
            + 2.0 * self.c * x * z
            + 2.0 * self.d * x * w
            + self.e * y * y
            + 2.0 * self.f * y * z
            + 2.0 * self.g * y * w
            + self.h * z * z
            + 2.0 * self.i * z * w
            + self.j * w * w
    }

    /// 找到最小化误差的 3D 点
    /// 通过求解 ∂Q/∂v = 0 得到
    ///
    /// 返回 (优化后的点, 最小误差)
    #[inline]
    pub fn optimize(&self) -> (Point3, f32) {
        // 构建 3x3 线性系统的系数矩阵
        let a11 = 2.0 * self.a;
        let a12 = 2.0 * self.b;
        let a13 = 2.0 * self.c;
        let a22 = 2.0 * self.e;
        let a23 = 2.0 * self.f;
        let a33 = 2.0 * self.h;

        // 右边向量
        let b1 = -2.0 * self.d;
        let b2 = -2.0 * self.g;
        let b3 = -2.0 * self.i;

        // 计算行列式
        let det = a11 * (a22 * a33 - a23 * a23) - a12 * (a12 * a33 - a23 * a13)
            + a13 * (a12 * a23 - a22 * a13);

        // 奇异矩阵检测
        if det.abs() < 1e-10 {
            // 返回原点作为默认值
            return (Point3::origin(), self.value(Point3::origin()));
        }

        // 使用 Cramer 法则求解
        let det1 = b1 * (a22 * a33 - a23 * a23) - a12 * (b2 * a33 - a23 * b3)
            + a13 * (b2 * a23 - a22 * b3);

        let det2 = a11 * (b2 * a33 - a23 * b3) - b1 * (a12 * a33 - a23 * a13)
            + a13 * (a12 * b3 - b2 * a13);

        let det3 = a11 * (a22 * b3 - a23 * b2) - a12 * (a12 * b3 - b1 * a23)
            + b1 * (a12 * a23 - a22 * a13);

        let x = det1 / det;
        let y = det2 / det;
        let z = det3 / det;

        let optimal = Point3::new(x, y, z);
        (optimal, self.value(optimal))
    }

    /// 获取矩阵元素
    #[inline]
    pub fn a(&self) -> f32 {
        self.a
    }
    #[inline]
    pub fn b(&self) -> f32 {
        self.b
    }
    #[inline]
    pub fn c(&self) -> f32 {
        self.c
    }
    #[inline]
    pub fn d(&self) -> f32 {
        self.d
    }
    #[inline]
    pub fn e(&self) -> f32 {
        self.e
    }
    #[inline]
    pub fn f(&self) -> f32 {
        self.f
    }
    #[inline]
    pub fn g(&self) -> f32 {
        self.g
    }
    #[inline]
    pub fn h(&self) -> f32 {
        self.h
    }
    #[inline]
    pub fn i(&self) -> f32 {
        self.i
    }
    #[inline]
    pub fn j(&self) -> f32 {
        self.j
    }

    /// 矩阵形式命名
    #[inline]
    pub fn xx(&self) -> f32 {
        self.a
    }
    #[inline]
    pub fn xy(&self) -> f32 {
        self.b
    }
    #[inline]
    pub fn xz(&self) -> f32 {
        self.c
    }
    #[inline]
    pub fn xw(&self) -> f32 {
        self.d
    }
    #[inline]
    pub fn yy(&self) -> f32 {
        self.e
    }
    #[inline]
    pub fn yz(&self) -> f32 {
        self.f
    }
    #[inline]
    pub fn yw(&self) -> f32 {
        self.g
    }
    #[inline]
    pub fn zz(&self) -> f32 {
        self.h
    }
    #[inline]
    pub fn zw(&self) -> f32 {
        self.i
    }
    #[inline]
    pub fn ww(&self) -> f32 {
        self.j
    }
}

impl std::ops::Add for QuadricT {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        self.add_values(other)
    }
}

impl std::ops::AddAssign for QuadricT {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.add_assign_values(other);
    }
}

impl std::ops::Mul<f32> for QuadricT {
    type Output = Self;
    #[inline]
    fn mul(self, s: f32) -> Self {
        self.mul_scalar(s)
    }
}

impl std::ops::MulAssign<f32> for QuadricT {
    #[inline]
    fn mul_assign(&mut self, s: f32) {
        self.mul_assign_scalar(s);
    }
}

/// Quadric 类型别名
pub type Quadricf = QuadricT;
pub type Quadricd = QuadricT; // 后续可改为 f64

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_plane() {
        // 平面 z = 0 (法线 (0,0,1)，d = 0)
        let q = QuadricT::from_plane(0.0, 0.0, 1.0, 0.0);

        // 验证矩阵元素
        assert!((q.a() - 0.0).abs() < 1e-6);
        assert!((q.b() - 0.0).abs() < 1e-6);
        assert!((q.c() - 0.0).abs() < 1e-6);
        assert!((q.d() - 0.0).abs() < 1e-6);
        assert!((q.e() - 0.0).abs() < 1e-6);
        assert!((q.f() - 0.0).abs() < 1e-6);
        assert!((q.g() - 0.0).abs() < 1e-6);
        assert!((q.h() - 1.0).abs() < 1e-6); // c*c = 1
        assert!((q.i() - 0.0).abs() < 1e-6);
        assert!((q.j() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_value() {
        // 平面 z = 0
        let q = QuadricT::from_plane(0.0, 0.0, 1.0, 0.0);

        // 点 (0,0,0) 的误差应该是 0
        assert!((q.value(Point3::origin()) - 0.0).abs() < 1e-6);

        // 点 (1,2,0) 的误差应该是 z^2 = 0
        assert!((q.value(Point3::new(1.0, 2.0, 0.0)) - 0.0).abs() < 1e-6);

        // 点 (1,2,3) 的误差应该是 z^2 = 9
        assert!((q.value(Point3::new(1.0, 2.0, 3.0)) - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_add() {
        let q1 = QuadricT::from_plane(1.0, 0.0, 0.0, 0.0);
        let q2 = QuadricT::from_plane(0.0, 1.0, 0.0, 0.0);

        let q = q1 + q2;

        // q1 有 a=1, q2 有 e=1
        assert!((q.a() - 1.0).abs() < 1e-6);
        assert!((q.e() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_optimize() {
        // 两个垂直平面的交线：x=0 和 y=0
        let q1 = QuadricT::from_plane(1.0, 0.0, 0.0, 0.0); // x = 0
        let q2 = QuadricT::from_plane(0.0, 1.0, 0.0, 0.0); // y = 0
        let q = q1 + q2;

        // 最优点应该是原点
        let (opt, error) = q.optimize();
        assert!((opt.x - 0.0).abs() < 1e-4);
        assert!((opt.y - 0.0).abs() < 1e-4);
        assert!((opt.z - 0.0).abs() < 1e-4);
        assert!((error - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_from_face() {
        // 三角形面：z = 0 平面上的点 (0,0,0), (1,0,0), (0,1,0)
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let point = Point3::new(0.0, 0.0, 0.0);

        let q = QuadricT::from_face(normal, point);

        // 应该是平面 z = 0
        assert!((q.a() - 0.0).abs() < 1e-6);
        assert!((q.h() - 1.0).abs() < 1e-6); // c*c = 1
    }

    #[test]
    fn test_from_point() {
        let q = QuadricT::from_point(Point3::new(1.0, 2.0, 3.0));

        // 点 (1,2,3) 的误差应该等于 ||v - p||^2
        assert!((q.value(Point3::new(1.0, 2.0, 3.0)) - 0.0).abs() < 1e-6);
        assert!((q.value(Point3::origin()) - 14.0).abs() < 1e-6); // 1+4+9=14
    }

    #[test]
    fn test_mul() {
        let q = QuadricT::from_plane(1.0, 0.0, 0.0, 0.0);
        let q2 = q * 2.0;

        assert!((q2.a() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_zero() {
        let q = QuadricT::zero();

        assert!((q.value(Point3::new(1.0, 2.0, 3.0)) - 0.0).abs() < 1e-6);
    }
}
