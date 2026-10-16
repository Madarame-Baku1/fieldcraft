//! The GPU backend against the CPU scheme and the analytic cases of `fieldcraft-lbm`'s
//! verification suite. The stated tolerances of the backend, with populations stored as
//! deviations from their weights in single precision (measured on lavapipe at 2–5e-6, with
//! a margin): velocity fields within 2e-5 of the largest speed, densities within 1e-6, wall
//! forces within 5e-5 of their magnitude (a wall's shear within 5e-5 of itself), mass drift below
//! 1e-6. Skipped without a GPU adapter
//! unless `FIELDCRAFT_REQUIRE_GPU=1`, which makes a missing adapter a failure; `FIELDCRAFT_GPU` picks
//! the adapter (`hardware` refuses a software one).

#[path = "../../lbm/tests/common/mod.rs"]
mod common;

use common::*;
use fieldcraft_gpu::select::{Policy, gpu_required};
use fieldcraft_lbm::backend::Backend;
use fieldcraft_lbm::{Flow, Setup};
use fieldcraft_lbm_gpu::GpuFlow;
use fieldcraft_model::{BoxFace, FaceCondition, Procedure, Profile};

const PERIODIC: FaceCondition = FaceCondition::Periodic {};

/// The flow on the GPU `FIELDCRAFT_GPU` picks, or `None` to skip (a failure when a GPU is
/// required).
fn gpu(flow: &Flow) -> Option<GpuFlow> {
    let policy = Policy::from_env().expect("FIELDCRAFT_GPU names a policy");
    match GpuFlow::new(flow, policy) {
        Ok(g) => {
            println!("GPU: {}", g.adapter_name());
            Some(g)
        }
        Err(e) if gpu_required() => panic!("a GPU is required: {e}"),
        Err(e) => {
            println!("skipping: {e}");
            None
        }
    }
}

/// Both backends through every step; the CPU's fields and the GPU's.
fn both(setup: &Setup, mut cpu: Flow) -> Option<(Flow, GpuFlow)> {
    let mut gpu = gpu(&cpu)?;
    cpu.advance(setup.steps).unwrap();
    gpu.advance(setup.steps).unwrap();
    Some((cpu, gpu))
}

/// The largest difference of the velocity fields over the largest CPU speed, and of the density
/// fields.
fn field_difference(cpu: &mut Flow, gpu: &mut GpuFlow) -> (f64, f64) {
    let (a, b) = (cpu.fields().unwrap(), gpu.fields().unwrap());
    let scale = a.velocity.iter().map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()).fold(0.0, f64::max).max(1e-30);
    let du = a
        .velocity
        .iter()
        .zip(&b.velocity)
        .map(|(x, y)| ((x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2) + (x[2] - y[2]).powi(2)).sqrt())
        .fold(0.0, f64::max);
    let drho = a.density.iter().zip(&b.density).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    (du / scale, drho)
}

#[test]
fn plane_poiseuille_matches_the_analytic_profile_and_the_cpu() {
    let (h, nu, umax, n) = (1.0, 1.0 / 6.0, 0.1, 16usize);
    let dx = h / n as f64;
    let a = 8.0 * nu * umax / (h * h);
    let mut d = domain([0.0; 3], [dx, h, dx], dx);
    d.faces.x_min = PERIODIC;
    d.faces.x_max = PERIODIC;
    d.faces.z_min = PERIODIC;
    d.faces.z_max = PERIODIC;
    d.acceleration = [a, 0.0, 0.0];
    let m = model(1.0, nu, d, step(4.0 * h * h / nu, dx * dx, Some(3.0 / 16.0)));
    let (setup, flow) = prepare(&m, &[]);
    let Some((mut cpu, mut gpu)) = both(&setup, flow) else { return };
    let f = gpu.fields().unwrap();
    let mut err: f64 = 0.0;
    for j in 0..n {
        let y = (j as f64 + 0.5) * dx;
        let exact = a / (2.0 * nu) * y * (h - y);
        err = err.max((f.velocity[j][0] * setup.velocity_scale() - exact).abs() / umax);
    }
    let (du, drho) = field_difference(&mut cpu, &mut gpu);
    println!("Poiseuille on the GPU: max error {err:.2e} of the peak; CPU/GPU velocity {du:.2e}, density {drho:.2e}");
    assert!(err < 2e-5, "{err}");
    assert!(du < 2e-5 && drho < 1e-6, "{du} {drho}");
}

#[test]
fn couette_wall_forces_match_the_cpu() {
    let (h, nu, uw, n) = (1.0, 0.1, 0.05, 16usize);
    let dx = h / n as f64;
    let mut d = domain([0.0; 3], [dx, h, dx], dx);
    d.faces.x_min = PERIODIC;
    d.faces.x_max = PERIODIC;
    d.faces.z_min = PERIODIC;
    d.faces.z_max = PERIODIC;
    d.faces.y_max = FaceCondition::MovingWall { velocity: [uw, 0.0, 0.0] };
    let m = model(1.0, nu, d, step(3.0 * h * h / nu, 0.5 * dx * dx / (3.0 * nu), None));
    let (setup, flow) = prepare(&m, &[]);
    let Some((mut cpu, mut gpu)) = both(&setup, flow) else { return };
    let (fc, fg) = (Backend::forces(&mut cpu).unwrap(), gpu.forces().unwrap());
    let bottom = BoxFace::ALL.iter().position(|f| f.name() == "y_min").unwrap();
    let top = BoxFace::ALL.iter().position(|f| f.name() == "y_max").unwrap();
    let tau = nu * uw / h / setup.force_scale() * dx * dx;
    println!("Couette: wall force CPU {:?} / GPU {:?} (expected ±{tau:.3e} per step)", fc[bottom], fg[bottom]);
    // The tangential force is the shear: a cancellation of the links' exchanges to 1e-6 per link,
    // summed from the stored deviations alone (adding the weights back in single precision left
    // it 1e-4 of itself on an RTX 4090; 3e-6 on lavapipe this way). The normal force is the
    // pressure.
    for g in [bottom, top] {
        assert!((fc[g][0] - fg[g][0]).abs() <= 5e-5 * tau.abs(), "group {g} shear: {} vs {}", fc[g][0], fg[g][0]);
        assert!((fc[g][1] - fg[g][1]).abs() <= 1e-6 * fc[g][1].abs(), "group {g} normal: {} vs {}", fc[g][1], fg[g][1]);
    }
    let (du, drho) = field_difference(&mut cpu, &mut gpu);
    assert!(du < 2e-5 && drho < 1e-6, "{du} {drho}");
}

#[test]
fn taylor_green_decays_like_the_cpu_and_conserves_mass() {
    let (nu, u0, l, n) = (0.1, 0.1, 2.0 * std::f64::consts::PI, 32usize);
    let dx = l / n as f64;
    let mut d = domain([0.0; 3], [l, l, dx], dx);
    for f in BoxFace::ALL {
        *d.faces.get_mut(f) = PERIODIC;
    }
    let dt = 0.3 * dx * dx / (3.0 * nu);
    let half_life = std::f64::consts::LN_2 / (2.0 * nu);
    let m = model(1.0, nu, d, step(half_life, dt, None));
    let (setup, mut flow) = prepare(&m, &[]);
    let grid = *flow.grid();
    let to_lattice = 1.0 / setup.velocity_scale();
    flow.set_equilibrium(|c| {
        let [x, y, _] = grid.centre(c);
        let p = -u0 * u0 / 4.0 * ((2.0 * x).cos() + (2.0 * y).cos());
        (setup.lattice_density(p), [-x.cos() * y.sin() * u0 * to_lattice, x.sin() * y.cos() * u0 * to_lattice, 0.0])
    });
    let Some(mut gpu) = gpu(&flow) else { return };
    let mass0 = gpu.mass().unwrap();
    let mut cpu = flow;
    cpu.advance(setup.steps).unwrap();
    gpu.advance(setup.steps).unwrap();
    let drift = (gpu.mass().unwrap() - mass0).abs() / mass0;
    let (du, drho) = field_difference(&mut cpu, &mut gpu);
    let f = gpu.fields().unwrap();
    let (mut num, mut den) = (0.0, 0.0);
    for c in 0..grid.len() {
        let [x, y, _] = grid.centre(c);
        let mode = [-x.cos() * y.sin(), x.sin() * y.cos()];
        num += f.velocity[c][0] * mode[0] + f.velocity[c][1] * mode[1];
        den += mode[0] * mode[0] + mode[1] * mode[1];
    }
    let amplitude = num / den * setup.velocity_scale();
    let exact = u0 * (-2.0 * nu * setup.steps as f64 * setup.time_step).exp();
    let err = (amplitude / exact - 1.0).abs();
    println!("Taylor–Green on the GPU: amplitude error {err:.2e}; CPU/GPU velocity {du:.2e}, density {drho:.2e}; mass drift {drift:.2e}");
    // The amplitude error is the scheme's at 32 cells (the CPU test shows second order).
    assert!(err < 5e-3, "{err}");
    assert!(du < 2e-5 && drho < 1e-6, "{du} {drho}");
    assert!(drift < 1e-6, "{drift}");
}

#[test]
fn flow_past_a_sphere_matches_the_cpu_fields_and_forces() {
    let l = 24usize;
    let side = l as f64;
    let mut d = domain([0.0; 3], [side; 3], 1.0);
    for f in BoxFace::ALL {
        *d.faces.get_mut(f) = PERIODIC;
    }
    d.acceleration = [1e-5, 0.0, 0.0];
    d.obstacles = vec!["Ball".into()];
    let ball = obstacle("Ball", sphere([side / 2.0; 3], 6.0, 3), &["skin"]);
    let m = model(
        1.0,
        1.0 / 6.0,
        d,
        Procedure::Fluid { domain: "Flow".into(), end_time: None, steps: Some(300), time_step: Some(1.0), frames: 1, magic: None },
    );
    let (setup, flow) = prepare(&m, std::slice::from_ref(&ball));
    let Some((mut cpu, mut gpu)) = both(&setup, flow) else { return };
    let (du, drho) = field_difference(&mut cpu, &mut gpu);
    let (fc, fg) = (Backend::forces(&mut cpu).unwrap(), gpu.forces().unwrap());
    let skin = cpu.groups().iter().position(|g| g.name == "skin").unwrap();
    let scale = fc[skin].iter().map(|v| v.abs()).fold(0.0, f64::max);
    let dforce = (0..3).map(|a| (fc[skin][a] - fg[skin][a]).abs()).fold(0.0, f64::max) / scale;
    println!("Sphere: CPU/GPU velocity {du:.2e}, density {drho:.2e}; drag CPU {:?} GPU {:?} ({dforce:.2e})", fc[skin], fg[skin]);
    assert!(du < 2e-5 && drho < 1e-6, "{du} {drho}");
    assert!(dforce < 5e-5, "{dforce}");
}

#[test]
fn zou_he_channel_matches_the_cpu() {
    let (h, l, umax, nu) = (1.0, 4.0, 0.1, 0.05);
    let dx = h / 10.0;
    let mut d = domain([0.0; 3], [l, h, dx], dx);
    d.faces.x_min = FaceCondition::Velocity { velocity: [umax, 0.0, 0.0], profile: Profile::Parabolic };
    d.faces.x_max = FaceCondition::Pressure { pressure: 0.0 };
    d.faces.z_min = PERIODIC;
    d.faces.z_max = PERIODIC;
    let m = model(1.0, nu, d, step(2.0 * l / umax, 0.05 * dx / umax, None));
    let (setup, flow) = prepare(&m, &[]);
    let Some((mut cpu, mut gpu)) = both(&setup, flow) else { return };
    let (du, drho) = field_difference(&mut cpu, &mut gpu);
    println!("Zou–He channel: CPU/GPU velocity {du:.2e}, density {drho:.2e}");
    assert!(du < 2e-5 && drho < 1e-6, "{du} {drho}");
}
