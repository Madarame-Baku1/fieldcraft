//! The tapered-block benchmark of the projection pipeline (#409) and the determinism of its
//! parallel parts.

use std::time::Instant;

use super::smooth::Order;
use super::*;
use crate::surface::{BoundarySurface, SurfaceOptions};
use crate::tet::geometry::{AnalyticSurface, Tessellation};
use crate::vec3::{cross, norm, sub};

/// A frustum of a square pyramid, `2·half` wide at z = 0 and `2·top` wide at z = `h`, with its
/// six planes, as one block.
fn tapered(half: f64, top: f64, h: f64) -> (Topology, BoundarySurface, Tessellation) {
    let p = vec![
        [-half, -half, 0.0],
        [half, -half, 0.0],
        [half, half, 0.0],
        [-half, half, 0.0],
        [-top, -top, h],
        [top, -top, h],
        [top, top, h],
        [-top, top, h],
    ];
    let quads: [[u32; 4]; 6] = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]];
    let mut tris = Vec::new();
    let mut face = Vec::new();
    let mut analytic = Vec::new();
    for (f, q) in quads.iter().enumerate() {
        tris.push([q[0], q[1], q[2]]);
        tris.push([q[0], q[2], q[3]]);
        face.extend([f as u32; 2]);
        let n = cross(sub(p[q[1] as usize], p[q[0] as usize]), sub(p[q[2] as usize], p[q[0] as usize]));
        let normal = scale(n, 1.0 / norm(n));
        analytic.push(Some(AnalyticSurface::Plane { normal, point: p[q[0] as usize] }));
    }
    let s = BoundarySurface::new(&p, &tris, &face, &SurfaceOptions::default()).unwrap();
    let t = Tessellation::new(&s, analytic);
    (Topology { vertices: p, blocks: vec![[0, 1, 2, 3, 4, 5, 6, 7]] }, s, t)
}

fn options(divisions: usize, smoothing: Smoothing, iterations: usize) -> ProjectOptions {
    // The element size sets the smoother's stopping tolerance (1e-4 of it).
    ProjectOptions {
        size: 1.0 / divisions as f64,
        divisions: (0..3).map(|d| (0, d, divisions)).collect(),
        smoothing,
        iterations,
        ..ProjectOptions::default()
    }
}

/// The parallel parts give the same mesh on one thread and on many: the sequential sweeps bit for
/// bit as before (only the measurements run in parallel), the coloured sweeps whatever the thread
/// count.
#[test]
fn parallel_smoothing_does_not_depend_on_the_thread_count() {
    let (topo, s, t) = tapered(1.0, 0.5, 1.0);
    let opts = options(8, Smoothing::Elliptic, 20);
    for order in [Order::Sequential, Order::Coloured] {
        let one = fieldcraft_parallel::with_threads(1, || project_ordered(&topo, &s, &t, &opts, order)).unwrap();
        let four = fieldcraft_parallel::with_threads(4, || project_ordered(&topo, &s, &t, &opts, order)).unwrap();
        assert_eq!(one.nodes, four.nodes, "{order:?}");
        assert_eq!(one.report, four.report, "{order:?}");
        assert!(one.report.min_scaled_jacobian > 0.5, "{order:?}: {}", one.report.min_scaled_jacobian);
    }
}

/// Seconds of the fastest of `repeats` runs of `f`, and its last result.
fn timed<R>(repeats: usize, mut f: impl FnMut() -> R) -> (f64, f64, R) {
    let mut times = Vec::new();
    let mut last = None;
    for _ in 0..repeats.max(1) {
        let t0 = Instant::now();
        last = Some(f());
        times.push(t0.elapsed().as_secs_f64());
    }
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().copied().fold(f64::INFINITY, f64::min);
    (mean, min, last.unwrap())
}

fn env(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// The 1M-node tapered block (99 divisions a side: 1 000 000 nodes, 970 299 hexahedra) through
/// the projection pipeline with three Winslow sweeps, sequential and on every core:
/// `FIELDCRAFT_BENCH_DIVISIONS`, `FIELDCRAFT_BENCH_REPEATS`, `FIELDCRAFT_BENCH_SWEEPS` and
/// `FIELDCRAFT_THREADS` change it.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn tapered_block_benchmark() {
    let divisions = env("FIELDCRAFT_BENCH_DIVISIONS", 99);
    let repeats = env("FIELDCRAFT_BENCH_REPEATS", 3);
    let sweeps = env("FIELDCRAFT_BENCH_SWEEPS", 3);
    let threads = fieldcraft_parallel::default_threads();
    let (topo, s, t) = tapered(1.0, 0.5, 1.0);
    let full = options(divisions, Smoothing::Elliptic, sweeps);
    let none = options(divisions, Smoothing::None, sweeps);
    println!("tapered block: {} nodes, {} hexahedra, {sweeps} sweeps, {repeats} runs, {threads} threads", (divisions + 1).pow(3), divisions.pow(3));
    println!("| threads | order | pipeline (s) | without smoothing (s) | smoothing (s) | certified quality (s) | min SJ | mean SJ | refused |");
    for (n, order) in [(1, Order::Sequential), (1, Order::Coloured), (threads, Order::Sequential), (threads, Order::Coloured)] {
        fieldcraft_parallel::with_threads(n, || {
            let (all, all_min, m) = timed(repeats, || project_ordered(&topo, &s, &t, &full, order).unwrap());
            let (bare, bare_min, _) = timed(repeats, || project_ordered(&topo, &s, &t, &none, order).unwrap());
            let (certified, _, _) = timed(repeats, || {
                fieldcraft_parallel::map_grained(m.hexes.len(), smooth::HEX_GRAIN, |e| {
                    let c = m.hexes[e].map(|v| m.nodes[v as usize]);
                    element_quality(ElementKind::Hex8, &c).map(|q| q.size).unwrap_or(0.0)
                })
            });
            let r = &m.report;
            println!(
                "| {n} | {order:?} | {all:.2} (min {all_min:.2}) | {bare:.2} (min {bare_min:.2}) | {:.2} | {certified:.2} | {:.4} | {:.4} | {} |",
                all - bare,
                r.min_scaled_jacobian,
                r.mean_scaled_jacobian,
                r.smoothing.refused
            );
        });
    }
}
