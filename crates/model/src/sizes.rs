//! Size controls: local element sizes that every mesher honours.
//!
//! A control asks for elements of about `size` in a region: on B-rep faces or whole bodies
//! (`geometry`, by persistent name), inside an axis-aligned box or inside a sphere. Together with
//! the mesher's global size they make one size field: the size at a point is the smallest of the
//! global size and, over the controls, the control's size plus the growth rate times the
//! distance to its region, so the elements grow away from a refined region at most by the
//! growth rate per unit length. A control coarser than the global size leaves it alone, except a
//! control on a whole body, which sets that body's global size. Controls are definitions of the
//! model: journalled with the command that made them and saved in the `.fieldcraft` file.

use serde::{Deserialize, Serialize};

use crate::geometry::{Entity, GeometryRef};
use crate::{Error, Model, Result, check_coordinates, context, insert_new, positive, remove_existing, replace_existing};

/// Most faces or bodies one control names.
pub const MAX_SIZE_TARGETS: usize = 10_000;
/// The default growth rate: the largest change of element size per unit length.
pub const DEFAULT_GROWTH: f64 = 0.3;
/// The growth rates a model accepts.
pub const GROWTH_RANGE: std::ops::RangeInclusive<f64> = 0.05..=2.0;

/// Where a size control applies.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum SizeRegion {
    /// B-rep faces, or whole bodies (`{"body": "Body1"}`), by persistent name.
    Geometry(Vec<GeometryRef>),
    /// An axis-aligned box.
    Box { min: [f64; 3], max: [f64; 3] },
    /// A ball.
    Sphere { center: [f64; 3], radius: f64 },
}

/// A size control.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SizeControl {
    pub region: SizeRegion,
    /// Target element edge length.
    pub size: f64,
}

impl SizeControl {
    pub fn check(&self) -> Result<()> {
        positive("size", self.size)?;
        check_coordinates(&[self.size])?;
        match &self.region {
            SizeRegion::Geometry(refs) => {
                if refs.is_empty() || refs.len() > MAX_SIZE_TARGETS {
                    return Err(Error::Invalid(format!("a size control names 1 to {MAX_SIZE_TARGETS} faces or bodies")));
                }
                if let Some(r) = refs.iter().find(|r| !matches!(r.entity, Entity::Face(_) | Entity::Body)) {
                    return Err(Error::Invalid(format!("a size control is on faces or bodies, not the {r}")));
                }
            }
            SizeRegion::Box { min, max } => {
                check_coordinates(min)?;
                check_coordinates(max)?;
                if (0..3).any(|k| min[k] >= max[k]) {
                    return Err(Error::Invalid("a size box's `min` must be below its `max` on every axis".into()));
                }
            }
            SizeRegion::Sphere { center, radius } => {
                check_coordinates(center)?;
                positive("radius", *radius)?;
                check_coordinates(&[*radius])?;
            }
        }
        Ok(())
    }

    /// The bodies a control on whole bodies sizes.
    pub fn bodies(&self) -> impl Iterator<Item = &str> {
        let refs = match &self.region {
            SizeRegion::Geometry(refs) => refs.as_slice(),
            _ => &[],
        };
        refs.iter().filter(|r| r.entity == Entity::Body).map(|r| r.body.as_str())
    }
}

impl Model {
    /// The size controls, by name.
    pub fn size_controls(&self) -> &std::collections::BTreeMap<String, SizeControl> {
        &self.size_controls
    }

    /// Add a size control, or replace the one of that name.
    pub fn set_size_control(&mut self, name: &str, control: SizeControl) -> Result<()> {
        control.check().map_err(|e| context("size control", name, e))?;
        if self.size_controls.contains_key(name) {
            replace_existing(&mut self.size_controls, "size control", name, control)
        } else {
            insert_new(&mut self.size_controls, "size control", name, control)
        }
    }

    pub fn remove_size_control(&mut self, name: &str) -> Result<SizeControl> {
        remove_existing(&mut self.size_controls, "size control", name)
    }

    /// The growth rate of the size field: set by [`Model::set_size_growth`], else
    /// [`DEFAULT_GROWTH`].
    pub fn size_growth(&self) -> f64 {
        self.size_growth.unwrap_or(DEFAULT_GROWTH)
    }

    /// Whether the growth rate was set.
    pub fn size_growth_set(&self) -> bool {
        self.size_growth.is_some()
    }

    /// Set the growth rate (in [`GROWTH_RANGE`]), or go back to the default with `None`.
    pub fn set_size_growth(&mut self, rate: Option<f64>) -> Result<()> {
        if let Some(g) = rate
            && !GROWTH_RANGE.contains(&g)
        {
            return Err(Error::Invalid(format!("the growth rate must be from {} to {}, got {g}", GROWTH_RANGE.start(), GROWTH_RANGE.end())));
        }
        self.size_growth = rate;
        Ok(())
    }

    /// The size a control on the whole body `body` sets (the smallest, when several do).
    pub fn body_size(&self, body: &str) -> Option<f64> {
        self.size_controls.values().filter(|c| c.bodies().any(|b| b == body)).map(|c| c.size).reduce(f64::min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face(n: &str) -> GeometryRef {
        GeometryRef::new("Body1", Entity::Face(n.into())).unwrap()
    }

    #[test]
    fn controls_are_checked_replaced_and_round_trip() {
        let mut m = Model::default();
        m.set_size_control("Hole", SizeControl { region: SizeRegion::Geometry(vec![face("F3")]), size: 0.5 }).unwrap();
        m.set_size_control("Zone", SizeControl { region: SizeRegion::Box { min: [0.0; 3], max: [1.0, 2.0, 3.0] }, size: 1.0 }).unwrap();
        m.set_size_control("Zone", SizeControl { region: SizeRegion::Sphere { center: [0.0; 3], radius: 2.0 }, size: 0.8 }).unwrap();
        m.set_size_control("Whole", SizeControl { region: SizeRegion::Geometry(vec![GeometryRef::new("Body1", Entity::Body).unwrap()]), size: 3.0 })
            .unwrap();
        m.set_size_growth(Some(0.5)).unwrap();
        assert_eq!(m.size_controls().len(), 3);
        assert_eq!(m.body_size("Body1"), Some(3.0));
        assert_eq!(m.body_size("Body2"), None);
        let back: Model = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
        assert_eq!(back.size_controls(), m.size_controls());
        assert_eq!(back.size_growth(), 0.5);
        assert_eq!(Model::default().size_growth(), DEFAULT_GROWTH);
        m.remove_size_control("Hole").unwrap();
        assert!(m.remove_size_control("Hole").is_err());
    }

    #[test]
    fn hostile_controls_are_refused() {
        let mut m = Model::default();
        let ball = |radius: f64, size: f64| SizeControl { region: SizeRegion::Sphere { center: [0.0; 3], radius }, size };
        for c in [
            ball(1.0, 0.0),
            ball(1.0, -1.0),
            ball(1.0, f64::NAN),
            ball(1.0, 1e300),
            ball(0.0, 1.0),
            ball(f64::INFINITY, 1.0),
            SizeControl { region: SizeRegion::Box { min: [1.0, 0.0, 0.0], max: [0.0, 1.0, 1.0] }, size: 1.0 },
            SizeControl { region: SizeRegion::Box { min: [f64::NAN, 0.0, 0.0], max: [1.0; 3] }, size: 1.0 },
            SizeControl { region: SizeRegion::Geometry(vec![]), size: 1.0 },
            SizeControl { region: SizeRegion::Geometry(vec![GeometryRef::new("Body1", Entity::Edge("E".into())).unwrap()]), size: 1.0 },
        ] {
            assert!(m.set_size_control("Bad", c.clone()).is_err(), "{c:?}");
        }
        assert!(m.set_size_control("1bad", ball(1.0, 1.0)).is_err());
        for g in [0.0, 0.01, 3.0, f64::NAN] {
            assert!(m.set_size_growth(Some(g)).is_err(), "{g}");
        }
        assert!(m.size_controls().is_empty());
    }
}
