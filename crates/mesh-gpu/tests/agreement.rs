//! The GPU sweeps against the CPU's coloured sweeps, their reference: the same projections with
//! `Backend::Coloured` and `Backend::Device`. Skipped without a GPU adapter unless
//! `FIELDCRAFT_REQUIRE_GPU=1`; `FIELDCRAFT_GPU` picks the adapter.

use std::f64::consts::{PI, TAU};
use std::time::Instant;

use fieldcraft_gpu::select::gpu_required;
use fieldcraft_mesh::project::{Backend, ProjectOptions, ProjectedMesh, Smoothing, Topology, project_mesh};
use fieldcraft_mesh::surface::{BoundarySurface, SurfaceOptions};
use fieldcraft_mesh::tet::geometry::{AnalyticSurface, Tessellation};

/// A UV sphere of radius `r` about the origin, one face, with its exact surface.
fn sphere(r: f64, n: usize) -> (BoundarySurface, Tessellation) {
    let rings = n / 2;
    let mut p = vec![[0.0, 0.0, -r], [0.0, 0.0, r]];
    for j in 1..rings {
        let th = PI * j as f64 / rings as f64 - PI / 2.0;
        for i in 0..n {
            let a = TAU * i as f64 / n as f64;
            p.push([r * th.cos() * a.cos(), r * th.cos() * a.sin(), r * th.sin()]);
        }
    }
    let id = |j: usize, i: usize| (2 + (j - 1) * n + i % n) as u32;
    let mut tris = Vec::new();
    for i in 0..n {
        tris.push([0, id(1, i + 1), id(1, i)]);
        tris.push([1, id(rings - 1, i), id(rings - 1, i + 1)]);
    }
    for j in 1..rings - 1 {
        for i in 0..n {
            tris.push([id(j, i), id(j, i + 1), id(j + 1, i + 1)]);
            tris.push([id(j, i), id(j + 1, i + 1), id(j + 1, i)]);
        }
    }
    let face = vec![0; tris.len()];
    let s = BoundarySurface::new(&p, &tris, &face, &SurfaceOptions::default()).unwrap();
    let t = Tessellation::new(&s, vec![Some(AnalyticSurface::Sphere { center: [0.0; 3], radius: r })]);
    (s, t)
}

/// A closed cylinder of radius `r` from z = 0 to `h`: faces 0 (bottom), 1 (top), 2 (side).
fn cylinder(r: f64, h: f64, n: usize) -> (BoundarySurface, Tessellation) {
    let mut p = vec![[0.0, 0.0, 0.0], [0.0, 0.0, h]];
    for z in [0.0, h] {
        for i in 0..n {
            let a = TAU * i as f64 / n as f64;
            p.push([r * a.cos(), r * a.sin(), z]);
        }
    }
    let (b, t) = (|i: usize| (2 + i % n) as u32, |i: usize| (2 + n + i % n) as u32);
    let mut tris = Vec::new();
    let mut face = Vec::new();
    for i in 0..n {
        tris.push([0, b(i + 1), b(i)]);
        face.push(0);
        tris.push([1, t(i), t(i + 1)]);
        face.push(1);
        tris.push([b(i), b(i + 1), t(i + 1)]);
        tris.push([b(i), t(i + 1), t(i)]);
        face.extend([2, 2]);
    }
    let s = BoundarySurface::new(&p, &tris, &face, &SurfaceOptions::default()).unwrap();
    let analytic = vec![
        Some(AnalyticSurface::Plane { normal: [0.0, 0.0, -1.0], point: [0.0; 3] }),
        Some(AnalyticSurface::Plane { normal: [0.0, 0.0, 1.0], point: [0.0, 0.0, h] }),
        Some(AnalyticSurface::Cylinder { origin: [0.0; 3], axis: [0.0, 0.0, 1.0], radius: r }),
    ];
    let t = Tessellation::new(&s, analytic);
    (s, t)
}

/// The frustum of `fieldcraft-mesh`'s benchmark: 2 wide at the bottom, 1 at the top, 1 high.
fn tapered() -> (Topology, BoundarySurface, Tessellation) {
    let (half, top, h): (f64, f64, f64) = (1.0, 0.5, 1.0);
    let p: Vec<[f64; 3]> = vec![
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
    let (mut tris, mut face, mut analytic) = (Vec::new(), Vec::new(), Vec::new());
    for (f, q) in quads.iter().enumerate() {
        tris.push([q[0], q[1], q[2]]);
        tris.push([q[0], q[2], q[3]]);
        face.extend([f as u32; 2]);
        let [a, b, c] = [q[0], q[1], q[2]].map(|v| p[v as usize]);
        let (u, w) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
        let n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        analytic.push(Some(AnalyticSurface::Plane { normal: n.map(|x| x / l), point: a }));
    }
    let s = BoundarySurface::new(&p, &tris, &face, &SurfaceOptions::default()).unwrap();
    let t = Tessellation::new(&s, analytic);
    (Topology { vertices: p, blocks: vec![[0, 1, 2, 3, 4, 5, 6, 7]] }, s, t)
}

/// The projection on the GPU, or `None` to skip (a failure when a GPU is required).
fn on_gpu(topo: &Topology, s: &BoundarySurface, t: &Tessellation, opts: &ProjectOptions) -> Option<ProjectedMesh> {
    fieldcraft_mesh_gpu::register();
    match project_mesh(topo, s, t, &ProjectOptions { backend: Backend::Device, ..opts.clone() }) {
        Ok(m) => Some(m),
        Err(e) if gpu_required() => panic!("a GPU is required: {e}"),
        // Only a missing device skips: a shader that does not compile is a failure anywhere.
        Err(e) if e.to_string().contains("no GPU device") => {
            println!("skipping: {e}");
            None
        }
        Err(e) => panic!("{e}"),
    }
}

/// The largest distance between the two meshes' nodes.
fn largest_gap(a: &ProjectedMesh, b: &ProjectedMesh) -> f64 {
    a.nodes.iter().zip(&b.nodes).map(|(p, q)| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()).fold(0.0, f64::max)
}

/// Both on the curved sphere and disc and on the tapered block: the same element count, the
/// GPU's mesh the same on a second run, and its quality next to the CPU's.
#[test]
fn gpu_sweeps_agree_with_the_coloured_cpu_sweeps() {
    let r = 10.0;
    let (ss, st) = sphere(r, 96);
    let ball = Topology::sphere([0.0; 3], r, 0.55).unwrap();
    let (cs, ct) = cylinder(r, 2.0, 128);
    let disc = Topology::butterfly([0.0; 3], [0.0, 0.0, 1.0], r, &[0.0, 2.0], 0.55).unwrap();
    let (block, bs, bt) = tapered();
    let block_opts = ProjectOptions { size: 1.0 / 16.0, divisions: (0..3).map(|d| (0, d, 16)).collect(), ..ProjectOptions::default() };
    let cases = [
        ("sphere", &ball, &ss, &st, ProjectOptions { size: 1.5, ..ProjectOptions::default() }),
        ("disc", &disc, &cs, &ct, ProjectOptions { size: 1.0, ..ProjectOptions::default() }),
        ("tapered block", &block, &bs, &bt, block_opts),
    ];
    for (name, topo, s, t, opts) in cases {
        let cpu = project_mesh(topo, s, t, &ProjectOptions { backend: Backend::Coloured, ..opts.clone() }).unwrap();
        let Some(gpu) = on_gpu(topo, s, t, &opts) else { return };
        let again = on_gpu(topo, s, t, &opts).unwrap();
        assert_eq!(gpu.nodes, again.nodes, "{name}: the GPU's mesh changed between runs");
        assert_eq!(cpu.hexes, gpu.hexes, "{name}");
        let (a, b) = (&cpu.report, &gpu.report);
        let gap = largest_gap(&cpu, &gpu) / opts.size;
        println!(
            "{name}: min SJ cpu {:.5} gpu {:.5}, mean SJ cpu {:.6} gpu {:.6}, largest node gap {gap:.2e} sizes, sweeps cpu {} gpu {}, refused cpu {} gpu {}",
            a.min_scaled_jacobian,
            b.min_scaled_jacobian,
            a.mean_scaled_jacobian,
            b.mean_scaled_jacobian,
            a.smoothing.iterations,
            b.smoothing.iterations,
            a.smoothing.refused,
            b.smoothing.refused
        );
    }
}

/// Throughput of three sweeps on the 1M-node tapered block (99 divisions a side; only its
/// boundary nodes move) and ten on a 7-block sphere of about a million nodes (its interior moves
/// too), on the CPU (sequential and coloured, every core) and on the GPU:
/// `FIELDCRAFT_BENCH_DIVISIONS`, `FIELDCRAFT_BENCH_SPHERE_SIZE`, `FIELDCRAFT_BENCH_REPEATS`.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn throughput() {
    let env = |name: &str, default: f64| std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default);
    let divisions = env("FIELDCRAFT_BENCH_DIVISIONS", 99.0) as usize;
    let repeats = env("FIELDCRAFT_BENCH_REPEATS", 3.0).max(1.0) as usize;
    let sphere_size = env("FIELDCRAFT_BENCH_SPHERE_SIZE", 0.16);
    let (block, bs, bt) = tapered();
    let (ss, st) = sphere(10.0, 192);
    let ball = Topology::sphere([0.0; 3], 10.0, 0.55).unwrap();
    let block_opts = ProjectOptions {
        size: 1.0 / divisions as f64,
        divisions: (0..3).map(|d| (0, d, divisions)).collect(),
        iterations: 3,
        ..ProjectOptions::default()
    };
    let sphere_opts = ProjectOptions { size: sphere_size, iterations: 10, ..ProjectOptions::default() };
    fieldcraft_mesh_gpu::register();
    println!("mean of {repeats} runs; smoothing = pipeline - the same pipeline without smoothing");
    println!("| case | backend | nodes | sweeps | pipeline (s) | smoothing (s) | min SJ | mean SJ | refused |");
    for (name, topo, s, t, opts) in [("tapered block", &block, &bs, &bt, block_opts), ("sphere", &ball, &ss, &st, sphere_opts)] {
        let timed = |o: &ProjectOptions| -> Option<(f64, ProjectedMesh)> {
            let t0 = Instant::now();
            let mut last = None;
            for _ in 0..repeats {
                match project_mesh(topo, s, t, o) {
                    Ok(m) => last = Some(m),
                    Err(e) if !gpu_required() && e.to_string().contains("no GPU device") => {
                        println!("skipping the GPU: {e}");
                        return None;
                    }
                    Err(e) => panic!("{name}: {e}"),
                }
            }
            last.map(|m| (t0.elapsed().as_secs_f64() / repeats as f64, m))
        };
        let Some((bare, _)) = timed(&ProjectOptions { smoothing: Smoothing::None, ..opts.clone() }) else { return };
        for backend in [Backend::Cpu, Backend::Coloured, Backend::Device] {
            let Some((all, m)) = timed(&ProjectOptions { backend, ..opts.clone() }) else { return };
            let r = &m.report;
            println!(
                "| {name} | {backend:?} | {} | {} | {all:.2} | {:.2} | {:.4} | {:.4} | {} |",
                m.nodes.len(),
                r.smoothing.iterations,
                all - bare,
                r.min_scaled_jacobian,
                r.mean_scaled_jacobian,
                r.smoothing.refused
            );
        }
    }
}
