//! Micro-benchmark comparatif pour le Frustum Culling (A/B Test) :
//! Baseline (select/cmpge) vs Optimisé (Center + Extent SIMD).
//!
//! Exécution : `cargo run --profile fast -p horizon-client-render --example frustum_cull_bench`

use glam::{Mat4, Vec3, Vec4};
use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
struct TestFrustum {
    planes: [Vec4; 6],
}

impl TestFrustum {
    /// Baseline (historique) : test de coin via sign/select
    #[inline(never)]
    fn intersects_baseline(&self, min: Vec3, max: Vec3) -> bool {
        !self.planes.iter().any(|p| {
            let sign = p.truncate().cmpge(Vec3::ZERO);
            let v = Vec3::select(sign, max, min);
            p.truncate().dot(v) + p.w < 0.0
        })
    }

    /// Optimisé : Center + Extent SIMD
    #[inline(never)]
    fn intersects_optimized(&self, min: Vec3, max: Vec3) -> bool {
        let c = (max + min) * 0.5;
        let e = (max - min) * 0.5;
        !self.planes.iter().any(|p| {
            let n = p.truncate();
            let r = n.abs().dot(e);
            let d = n.dot(c) + p.w;
            d < -r
        })
    }

    /// Baseline : contains
    #[inline(never)]
    fn contains_baseline(&self, min: Vec3, max: Vec3) -> bool {
        self.planes.iter().all(|p| {
            let sign = p.truncate().cmpge(Vec3::ZERO);
            let v = Vec3::select(sign, min, max);
            p.truncate().dot(v) + p.w >= 0.0
        })
    }

    /// Optimisé : contains Center + Extent
    #[inline(never)]
    fn contains_optimized(&self, min: Vec3, max: Vec3) -> bool {
        let c = (max + min) * 0.5;
        let e = (max - min) * 0.5;
        self.planes.iter().all(|p| {
            let n = p.truncate();
            let r = n.abs().dot(e);
            let d = n.dot(c) + p.w;
            d >= r
        })
    }
}

/// Générateur LCG déterministe pour des tests reproductibles sans allocation
struct SimpleRng(u64);
impl SimpleRng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 32) as u32 as f32) / (u32::MAX as f32)
    }
    fn range(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_f32() * (max - min)
    }
}

fn main() {
    println!("=== Horizon Frustum Culling Benchmark (A/B Test) ===");

    // Frustum réaliste (FOV 70, Aspect 16:9, Near 0.05, Far 512)
    let fov_rad = 70.0_f32.to_radians();
    let aspect = 16.0 / 9.0;
    let proj = Mat4::perspective_rh(fov_rad, aspect, 0.05, 512.0);
    let view = Mat4::look_at_rh(Vec3::new(0.0, 10.0, 20.0), Vec3::ZERO, Vec3::Y);
    let vp = proj * view;

    // Extraction standard Gribb-Hartmann des 6 plans
    let row = |r: usize| Vec4::new(vp.row(0)[r], vp.row(1)[r], vp.row(2)[r], vp.row(3)[r]);
    let raw_planes = [
        row(3) + row(0), // Left
        row(3) - row(0), // Right
        row(3) + row(1), // Bottom
        row(3) - row(1), // Top
        row(3) + row(2), // Near
        row(3) - row(2), // Far
    ];
    let planes = raw_planes.map(|p| {
        let n = p.truncate();
        let len = n.length();
        if len > 0.0 { p / len } else { p }
    });

    let frustum = TestFrustum { planes };

    // Générer 200 000 boîtes AABB simulant des chunks (16x16x16) et entités (1x2x1)
    let count = 200_000;
    let mut rng = SimpleRng(123456789);
    let mut boxes = Vec::with_capacity(count);

    for _ in 0..count {
        let center = Vec3::new(
            rng.range(-150.0, 150.0),
            rng.range(-50.0, 100.0),
            rng.range(-300.0, 50.0),
        );
        let size = if rng.next_f32() < 0.5 {
            Vec3::splat(16.0) // chunk
        } else {
            Vec3::new(1.0, 2.0, 1.0) // entité / particule
        };
        let min = center - size * 0.5;
        let max = center + size * 0.5;
        boxes.push((min, max));
    }

    // Warm-up & validation d'équivalence stricte
    for &(min, max) in &boxes {
        assert_eq!(
            frustum.intersects_baseline(min, max),
            frustum.intersects_optimized(min, max),
            "intersects mismatch"
        );
        assert_eq!(
            frustum.contains_baseline(min, max),
            frustum.contains_optimized(min, max),
            "contains mismatch"
        );
    }
    println!("Validation de conformité : 100% équivalent sur {} AABBs.", count);

    // Itérations de mesure
    let iterations = 10;
    let total_calls = count as u64 * iterations as u64;

    println!("\n--- Benchmark 1 : `intersects` ({} millions de tests) ---", total_calls / 1_000_000);

    // Baseline
    let start = Instant::now();
    let mut acc_base = 0u64;
    for _ in 0..iterations {
        for &(min, max) in &boxes {
            if frustum.intersects_baseline(black_box(min), black_box(max)) {
                acc_base += 1;
            }
        }
    }
    let dur_base = start.elapsed();
    let ns_base = dur_base.as_nanos() as f64 / total_calls as f64;

    // Optimized
    let start = Instant::now();
    let mut acc_opt = 0u64;
    for _ in 0..iterations {
        for &(min, max) in &boxes {
            if frustum.intersects_optimized(black_box(min), black_box(max)) {
                acc_opt += 1;
            }
        }
    }
    let dur_opt = start.elapsed();
    let ns_opt = dur_opt.as_nanos() as f64 / total_calls as f64;

    assert_eq!(acc_base, acc_opt);
    let speedup_intersects = (ns_base - ns_opt) / ns_base * 100.0;
    println!("Baseline (select/cmpge)    : {:.3} ns/test ({:.2?} total)", ns_base, dur_base);
    println!("Optimized (Center+Extent)  : {:.3} ns/test ({:.2?} total)", ns_opt, dur_opt);
    println!("-> Speedup `intersects`   : {:+.1}% ({:.2}x plus rapide)", speedup_intersects, ns_base / ns_opt);

    println!("\n--- Benchmark 2 : `contains` ({} millions de tests) ---", total_calls / 1_000_000);

    // Baseline contains
    let start = Instant::now();
    let mut acc_base_c = 0u64;
    for _ in 0..iterations {
        for &(min, max) in &boxes {
            if frustum.contains_baseline(black_box(min), black_box(max)) {
                acc_base_c += 1;
            }
        }
    }
    let dur_base_c = start.elapsed();
    let ns_base_c = dur_base_c.as_nanos() as f64 / total_calls as f64;

    // Optimized contains
    let start = Instant::now();
    let mut acc_opt_c = 0u64;
    for _ in 0..iterations {
        for &(min, max) in &boxes {
            if frustum.contains_optimized(black_box(min), black_box(max)) {
                acc_opt_c += 1;
            }
        }
    }
    let dur_opt_c = start.elapsed();
    let ns_opt_c = dur_opt_c.as_nanos() as f64 / total_calls as f64;

    assert_eq!(acc_base_c, acc_opt_c);
    let speedup_contains = (ns_base_c - ns_opt_c) / ns_base_c * 100.0;
    println!("Baseline (select/cmpge)    : {:.3} ns/test ({:.2?} total)", ns_base_c, dur_base_c);
    println!("Optimized (Center+Extent)  : {:.3} ns/test ({:.2?} total)", ns_opt_c, dur_opt_c);
    println!("-> Speedup `contains`     : {:+.1}% ({:.2}x plus rapide)", speedup_contains, ns_base_c / ns_opt_c);
    println!("\nRésultats finaux : acc_intersects={}, acc_contains={}", black_box(acc_opt), black_box(acc_opt_c));
}
