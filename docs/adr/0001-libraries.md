# ADR 0001: Libraries and licences

Status: accepted (2026-10-13)

## Context

ModelCraft is MIT OR Apache-2.0. Everything it links must allow that, and the meshing and solver
algorithms, where the obvious open-source implementations are GPL-family, must come from papers.

## Decision

**Geometry: SolidCraft's kernel**, as a git dependency pinned by rev
(`solidcraft-kernel`, `solidcraft-geom`; MIT OR Apache-2.0, built on the Apache-2.0 truck
crates). ModelCraft uses its B-rep bodies and its tessellation; it does not read CAD exchange
files itself. Only `modelcraft-geometry` may depend on it.

**Meshing: our own**, implemented from published papers (`docs/meshing/literature.md`).
Robust predicates follow Shewchuk's adaptive-precision arithmetic (the paper; his reference
`predicates.c` is public domain). A permissive crate may be adopted instead after review; record
it here.

**Linear algebra:** to be decided in the solver task, among permissively licensed pure-Rust
crates (for example `faer`, MIT). Record the choice here.

**Allowed:** MIT, BSD-2/3, Apache-2.0, ISC, Zlib, Unicode, MPL-2.0 (as an unmodified dependency),
CC0 and public domain.

**Rejected — never read, copy or link:**

| Project | Licence | Would have offered |
|---|---|---|
| TetGen | AGPL-3.0 | constrained Delaunay tetrahedralization |
| CGAL | GPL-3.0 / LGPL-3.0 | Delaunay, mesh generation |
| Gmsh | GPL-2.0+ | tet and hex meshing |
| Netgen | LGPL-2.1 | tet meshing |
| CalculiX | GPL-2.0 | implicit solver, Abaqus-like input |
| Code_Aster | GPL-3.0 | implicit solver |
| OpenRadioss | AGPL-3.0 | explicit solver |
| Elmer | GPL-2.0 / LGPL-2.1 | multiphysics solver |
| deal.II, FreeFEM, Salome | LGPL | FE libraries, pre-processing |
| OpenCASCADE | LGPL-2.1 | B-rep kernel |

**Commercial products** (Abaqus, Nastran, LS-DYNA, LS-PrePost, HyperMesh, ANSA, Femap, Patran,
ANSYS) are never run, scripted or used as a source of data. Deck formats come from the vendors'
public reference manuals, described in our own words with our own examples.

## Consequences

- The tet and hex meshers are significant original work; the literature survey decides the
  order and the methods.
- SolidCraft and ModelCraft move together: bumping the pinned rev bumps the repeated
  `[patch.crates-io]` entries in the same commit.
