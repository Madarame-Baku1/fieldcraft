//! Projection smoothing on the GPU: [`GpuSweep`] runs the free (interior) nodes of
//! `fieldcraft_mesh`'s coloured Winslow sweeps with the compute shader of `sweep.wgsl`, the CPU
//! sweep's update and guarded step written for the device; `fieldcraft_mesh::project` drives it
//! through its sweep device ([`fieldcraft_mesh::project::device`]) colour by colour and moves the
//! boundary nodes on the geometry itself.
//!
//! Single precision, positions relative to the mesh's centre: the CPU's coloured sweeps are the
//! reference. The device is deterministic on itself and gives the same verdicts as they do with a
//! smallest scaled Jacobian within 0.02 of theirs: near the guard's thresholds a step accepted in
//! double precision can be refused in single (or the reverse), and over many sweeps the grids part
//! (the meshing skill's corpus: up to ±0.015 on one part, on a GB10 and an M4 Max). On curved
//! analytic parts the nodes agree within a few millionths of an element (`tests/agreement.rs`). The mesh is certified on the CPU in double
//! precision afterwards, as always. The device is the one `fieldcraft_gpu`'s adapter policy picks
//! (`FIELDCRAFT_GPU`).
//!
//! Sources: the same as the CPU smoother (Winslow 1967; Thompson, Warsi and Mastin 1985;
//! Thomas and Middlecoff 1980; Knupp 2001 for the scaled Jacobian).
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]
#![forbid(unsafe_code)]

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
pub use native::{GpuSweep, register};

/// In the browser there is no device yet (WebGPU later): registering is a no-op and a projection
/// with the device backend reports that none is registered.
#[cfg(target_arch = "wasm32")]
pub fn register() {}
