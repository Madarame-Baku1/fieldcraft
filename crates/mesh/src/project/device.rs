//! Smoothing sweeps on a device (a GPU crate above this layer, #409).
//!
//! The coloured sweep ([`super::smooth`]) moves the nodes of one colour class at a time against a
//! grid frozen for that class, since no two nodes of a class share an element. A device takes the
//! free (interior) nodes of every class: Winslow's update from the node's structured stencil, or
//! the mean of its edge neighbours at an irregular node, then the guarded step (the full move,
//! else half, else a quarter, whichever keeps the quality around the node). The host moves the
//! class's boundary nodes, which need the geometry, against the same frozen grid, and tells the
//! device where they went. So a device sweep is the CPU's coloured sweep with the free nodes
//! computed elsewhere; [`reference_move`] is that computation in double precision, the CPU's own.
//!
//! An application registers the device ([`register_sweep_device`]); the CPU stays the
//! reference, and a device is deterministic on itself and agrees with it within the tolerance its
//! tests state.

use std::sync::OnceLock;

use super::smooth::{Stencil, guarded, quality_around, winslow};
use crate::Result;
use crate::vec3::{P, add, scale};

/// How a device moves one free node.
#[derive(Debug, Clone, PartialEq)]
pub enum Rule {
    /// Winslow's update from the node's structured stencil: its edge neighbours
    /// `nb[d] = [minus, plus]` along the three grid directions, the face diagonals
    /// `diag[pair][sa][sb]` for the direction pairs (0, 1), (0, 2), (1, 2), and the
    /// Thomas–Middlecoff control functions (zero without control).
    Winslow { nb: [[u32; 2]; 3], diag: [[[u32; 2]; 2]; 3], control: [f64; 3] },
    /// The mean of these edge neighbours (a node where three, five or more blocks meet).
    Mean(Vec<u32>),
}

/// A free node a device moves.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceNode {
    pub node: u32,
    pub rule: Rule,
}

/// What a device sweeps.
pub struct SweepMesh<'a> {
    /// The node positions when the sweeps start.
    pub nodes: &'a [P],
    pub hexes: &'a [[u32; 8]],
    /// The elements around every node.
    pub around: &'a [Vec<u32>],
    /// Per colour class, in sweep order, the free nodes the device moves (in node order).
    pub classes: Vec<Vec<DeviceNode>>,
    /// A move is kept when the smallest scaled Jacobian of the elements around the node stays at
    /// or above the lower of its value before and `floor`, and their largest edge ratio at or
    /// below the higher of its value before and [`super::smooth::EDGE_RATIO_CEILING`].
    pub floor: f64,
    /// A move shorter than this is not made.
    pub min_move: f64,
}

/// What a sweep does with a node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Move {
    Stay,
    To(P),
    /// Every step would have lowered the quality around the node.
    Refused,
}

/// A device that runs the free nodes' part of the coloured sweeps.
pub trait SweepDevice {
    /// For reports: `gpu: <adapter>`.
    fn name(&self) -> String;
    /// Move class `class`'s nodes on the grid as the device holds it, in place: one move per
    /// node of [`SweepMesh::classes`]`[class]`, in its order.
    fn sweep(&mut self, class: usize) -> Result<Vec<Move>>;
    /// Nodes the host moved since the last sweep, and where to.
    fn set(&mut self, moved: &[(u32, P)]) -> Result<()>;
}

/// Makes a device for one smoothing run.
pub type SweepFactory = dyn Fn(&SweepMesh) -> Result<Box<dyn SweepDevice>> + Send + Sync;

static FACTORY: OnceLock<Box<SweepFactory>> = OnceLock::new();

/// Register the process's sweep device (an application does this once at start; a second
/// registration is ignored).
pub fn register_sweep_device(factory: Box<SweepFactory>) {
    let _ = FACTORY.set(factory);
}

/// The registered sweep device's factory, if an application registered one.
pub fn sweep_device_factory() -> Option<&'static SweepFactory> {
    FACTORY.get().map(|b| b.as_ref())
}

/// The move of free node `n` on the grid `x`, in double precision: what a device computes, and
/// what the CPU sweep does with the same node.
pub fn reference_move(mesh: &SweepMesh, x: &[P], n: &DeviceNode) -> Move {
    let target = match &n.rule {
        Rule::Winslow { nb, diag, control } => winslow(x, &Stencil { nb: *nb, diag: *diag, control: *control }),
        Rule::Mean(edges) => {
            let (mut s, mut k) = ([0.0; 3], 0usize);
            for &w in edges {
                s = add(s, x.get(w as usize).copied().unwrap_or([0.0; 3]));
                k += 1;
            }
            (k > 0).then(|| scale(s, 1.0 / k as f64))
        }
    };
    let quality = |p: P| quality_around(x, mesh.hexes, mesh.around.get(n.node as usize).map_or(&[][..], Vec::as_slice), n.node, p);
    guarded(x, n.node, target, mesh.min_move, mesh.floor, quality, |p| p)
}
