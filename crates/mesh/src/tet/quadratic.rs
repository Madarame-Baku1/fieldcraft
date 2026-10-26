//! Quadratic tetrahedra (tet10): a node on every edge, at the midpoint inside the domain and on
//! the geometry on the boundary — projected onto the edge's feature curve, or onto the B-rep face
//! of a boundary edge inside one face — so curved boundaries are represented to second order.
//!
//! A curved edge can make an element invalid (a thin element against a strongly curved face).
//! Every element is checked with the certified Jacobian test that the validity gate uses
//! ([`crate::quality::element_validity`], which proves FieldCraft's criterion on the whole
//! element); an element that fails gets its curved edges backed off towards the straight midpoint
//! — half, then a quarter of the way to the geometry, then straight (for every element sharing
//! that edge) — and the elements around are checked again, so a node stays as near the geometry as
//! validity allows. Backing off only ever moves nodes towards the linear mesh, which is valid, and
//! each edge backs off at most [`BACK_OFF`]`.len()` times, so the loop ends.

use fieldcraft_model::ElementKind;

use crate::quality::{Validity, element_validity};
use crate::{Error, Result};

use super::TaggedFace;
use super::complex::{IdMap, NONE, face_toward, sorted2};
use super::geometry::{BoundaryGeometry, Tessellation};
use super::measure;

/// Tet10 mid-side nodes on edges 1-2, 2-3, 3-1, 1-4, 2-4, 3-4 (the model's node order).
use crate::quality::jacobian::TET_EDGES as MID_EDGES;

/// The fractions of the way from the straight midpoint to the geometry a mid-side node backs off
/// to, one after the other, while an element around it is invalid; the last is straight.
const BACK_OFF: [f64; 3] = [0.5, 0.25, 0.0];

/// A mid-side node on the boundary.
#[derive(Debug, Clone, Copy)]
struct Mid {
    id: u32,
    /// The edge's straight midpoint.
    straight: [f64; 3],
    /// Its point on the geometry.
    curved: [f64; 3],
    /// How many times it backed off ([`BACK_OFF`]).
    backed: usize,
}

/// The quadratic mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct Quadratic {
    pub points: Vec<[f64; 3]>,
    pub elements: Vec<Vec<u32>>,
    /// Curved edges backed off part of the way or left straight to keep their elements valid.
    pub straightened: usize,
}

/// Tet10 elements from tet4 ones (see the module docs).
pub fn tet10(
    points: &[[f64; 3]],
    tets: &[[u32; 4]],
    boundary: &[TaggedFace],
    curves: &IdMap<(u32, u32), u32>,
    curve_faces: &[[u32; 2]],
    geometry: &Tessellation,
) -> Result<Quadratic> {
    // Boundary edges and the B-rep face they lie on.
    let mut edge_face: IdMap<(u32, u32), u32> = IdMap::default();
    for b in boundary {
        let t = tets.get(b.tet as usize).ok_or_else(|| Error::Failed("tet10: boundary face of no tetrahedron".into()))?;
        let f = face_toward(*t, usize::from(b.opposite));
        for i in 0..3 {
            edge_face.entry(sorted2(f[i], f[(i + 1) % 3])).or_insert(b.face);
        }
    }
    let p = |v: u32| points.get(v as usize).copied().ok_or_else(|| Error::Failed("tet10: vertex id".into()));
    let mut out = points.to_vec();
    let mut node: IdMap<(u32, u32), Mid> = IdMap::default();
    let mut elements = Vec::with_capacity(tets.len());
    for t in tets {
        let mut e = t.to_vec();
        for &(i, j) in &MID_EDGES {
            let key = sorted2(t[i], t[j]);
            let id = match node.get(&key) {
                Some(m) => m.id,
                None => {
                    let m = measure::mid(p(t[i])?, p(t[j])?);
                    let curved = match curves.get(&key) {
                        Some(&c) => curve_faces.get(c as usize).and_then(|f| geometry.project_curve(*f, m)),
                        None => edge_face.get(&key).and_then(|&f| geometry.project_face(f, m)),
                    };
                    let id = u32::try_from(out.len()).map_err(|_| Error::Limit("too many nodes".into()))?;
                    let curved = curved.filter(|q| q.iter().all(|x| x.is_finite())).unwrap_or(m);
                    out.push(curved);
                    node.insert(key, Mid { id, straight: m, curved, backed: 0 });
                    id
                }
            };
            e.push(id);
        }
        elements.push(e);
    }
    let straightened = back_off(&mut out, &mut node, tets, &elements)?;
    Ok(Quadratic { points: out, elements, straightened })
}

/// Backs the curved edges of every invalid element off towards their straight midpoints
/// ([`BACK_OFF`]) until every element is valid, checking again the elements around each edge that
/// moved. Returns how many edges backed off.
fn back_off(out: &mut [[f64; 3]], node: &mut IdMap<(u32, u32), Mid>, tets: &[[u32; 4]], elements: &[Vec<u32>]) -> Result<usize> {
    let coords = |out: &[[f64; 3]], e: &[u32]| -> Result<Vec<[f64; 3]>> {
        e.iter().map(|&v| out.get(v as usize).copied().ok_or_else(|| Error::Failed("tet10: node id".into()))).collect()
    };
    let mut straightened = 0;
    let mut check: Vec<usize> = (0..elements.len()).collect();
    let mut edge_elements: IdMap<(u32, u32), Vec<usize>> = IdMap::default();
    for (n, t) in tets.iter().enumerate() {
        for &(i, j) in &MID_EDGES {
            edge_elements.entry(sorted2(t[i], t[j])).or_default().push(n);
        }
    }
    for _ in 0..=BACK_OFF.len() * node.len() {
        let mut next = Vec::new();
        for &n in &check {
            let Some(e) = elements.get(n) else { continue };
            if element_validity(ElementKind::Tet10, &coords(out, e)?)?.validity == Validity::Valid {
                continue;
            }
            let t = tets.get(n).copied().unwrap_or([NONE; 4]);
            for &(i, j) in &MID_EDGES {
                let key = sorted2(t[i], t[j]);
                let Some(mid) = node.get_mut(&key) else { continue };
                let (Some(&s), Some(slot)) = (BACK_OFF.get(mid.backed), out.get_mut(mid.id as usize)) else { continue };
                if *slot == mid.straight {
                    continue;
                }
                *slot = measure::lerp(mid.straight, mid.curved, s);
                straightened += usize::from(mid.backed == 0);
                mid.backed += 1;
                if let Some(around) = edge_elements.get(&key) {
                    next.extend_from_slice(around);
                }
            }
        }
        next.sort_unstable();
        next.dedup();
        if next.is_empty() {
            break;
        }
        check = next;
    }
    Ok(straightened)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The right tetrahedron with edge 1-2 on a face that curves into it: the edge's mid-side
    /// node sits at `(½, h, 0)` on the geometry. With λ2 = x, det J = 1 − 4hλ2, so the element is
    /// valid iff the node is less than ¼ off the straight midpoint. Returns where the node ends
    /// up after backing off, and how many edges backed off.
    fn backed_off(h: f64) -> ([f64; 3], usize) {
        let corners = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let tets = [[0, 1, 2, 3]];
        let mut out = corners.to_vec();
        let mut node: IdMap<(u32, u32), Mid> = IdMap::default();
        let mut element = vec![0, 1, 2, 3];
        for &(i, j) in &MID_EDGES {
            let id = out.len() as u32;
            let straight = measure::mid(corners[i], corners[j]);
            let curved = if (i, j) == (0, 1) { [0.5, h, 0.0] } else { straight };
            out.push(curved);
            node.insert(sorted2(i as u32, j as u32), Mid { id, straight, curved, backed: 0 });
            element.push(id);
        }
        let straightened = back_off(&mut out, &mut node, &tets, &[element]).unwrap();
        (out[4], straightened)
    }

    #[test]
    fn a_node_on_a_mildly_curved_face_stays_on_it() {
        assert_eq!(backed_off(0.2), ([0.5, 0.2, 0.0], 0));
    }

    #[test]
    fn a_node_backs_off_half_way() {
        // 0.4 is invalid, half of it (0.2) valid.
        assert_eq!(backed_off(0.4), ([0.5, 0.2, 0.0], 1));
    }

    #[test]
    fn a_node_backs_off_a_quarter_of_the_way() {
        // 0.6 and its half (0.3) are invalid, its quarter (0.15) valid.
        assert_eq!(backed_off(0.6), ([0.5, 0.15, 0.0], 1));
    }

    #[test]
    fn a_node_falls_back_to_straight_after_three_back_offs() {
        // 2, 1 and 0.5 are all invalid: the edge ends straight, counted once.
        assert_eq!(backed_off(2.0), ([0.5, 0.0, 0.0], 1));
    }
}
