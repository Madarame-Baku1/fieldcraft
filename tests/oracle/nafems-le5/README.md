# NAFEMS LE5 Z-section cantilever (hex20)

Our model of the published problem: a cantilever 10 m long (x) whose cross-section is a Z of
0.1 m thick walls, E = 210 GPa, ν = 0.3 (m, MN, MPa). On the mid-surface the web is 2 m wide, in
the plane z = 0 from y = −1 to y = 1, and the two flanges are 1 m high: the upper one at y = 1
rises to z = 1, the lower one at y = −1 drops to z = −1. Every displacement is zero at x = 0. A
torque of 1.2 MN·m acts at x = 10 as two shear forces of 0.6 MN, one along each flange's end
(+z on the upper flange, −z on the lower), 2 m apart. The target is the axial stress σ_xx at A,
x = 2.5 on the mid-surface of the lower flange at its free edge (2.5, −1, −1), where the restrained
warping of the section puts its largest axial stress.

## The mesh

The walls are solid hex20 blocks (`mesh.block`), the route LE1 and LE10 take: the issue allows
quad4 shells or hex20, and the shell model reaches a different limit (below). Five blocks,
2 elements through every wall: the two flanges (0.95 m outside the web, k = 4 elements), the two
corner blocks where a flange meets the web (2 × 2), and the web between them (8 elements); 32
elements along the length. The blocks share their faces, so `mesh.block` joins them. The shear
force on each flange is a total over the flange block's end face (z from ±0.05 to ±1), split by
area. A is a node: the mid-plane of the lower flange (y = −1) is a node plane of its 2 elements
through the thickness.

The recipe was generated with this script (`python3 le5.py 4 > recipe.json`):

```python
import json, sys

def le5_solid(k):
    """NAFEMS LE5 Z-section as hex20 solid blocks (m, MN, MPa): mid-surface web 2 wide in z = 0,
    flanges at y = ±1 reaching z = ±1, all walls 0.1 thick, length 10 along x; k sets the density
    (8k cells along the length, k per 0.95 m across the walls, 2 through the thickness)."""
    h = 0.05
    blocks = {  # name: (y0, y1, z0, z1, ny, nz)
        "lower_flange": (-1 - h, -1 + h, -1.0, -h, 2, k),
        "lower_corner": (-1 - h, -1 + h, -h, h, 2, 2),
        "web": (-1 + h, 1 - h, -h, h, 2 * k, 2),
        "upper_corner": (1 - h, 1 + h, -h, h, 2, 2),
        "upper_flange": (1 - h, 1 + h, h, 1.0, 2, k),
    }
    commands = []
    for name, (y0, y1, z0, z1, ny, nz) in blocks.items():
        base = [[0.0, y0, z0], [0.0, y1, z0], [0.0, y1, z1], [0.0, y0, z1]]
        corners = base + [[10.0, y, z] for _, y, z in base]
        commands.append({"command": "mesh.block", "params": {"body": name, "kind": "hex20", "corners": corners, "divisions": [ny, nz, 8 * k], "faces": {"1": "root", "2": "tip"}}})
    return commands + [
        {"command": "model.set.create", "params": {"name": "all", "geometry": [{"body": b} for b in blocks]}},
        {"command": "model.set.create", "params": {"name": "root", "geometry": [{"body": b, "face": "root"} for b in blocks]}},
        {"command": "model.set.create", "params": {"name": "lower_tip", "geometry": [{"body": "lower_flange", "face": "tip"}]}},
        {"command": "model.set.create", "params": {"name": "upper_tip", "geometry": [{"body": "upper_flange", "face": "tip"}]}},
        {"command": "model.material.create", "params": {"name": "Steel", "elastic": {"type": "isotropic", "young": 210000.0, "poisson": 0.3}}},
        {"command": "model.section.create", "params": {"name": "z_section", "set": "all", "material": "Steel", "kind": {"type": "solid"}}},
        {"command": "model.boundary_condition.create", "params": {"name": "root", "type": "fixed", "target": {"set": "root"}}},
        # the 1.2 MN·m torque: 0.6 MN along each flange's end face, 2 m apart
        {"command": "model.load.create", "params": {"name": "lower_shear", "type": "force", "target": {"set": "lower_tip"}, "vector": [0, 0, -0.6]}},
        {"command": "model.load.create", "params": {"name": "upper_shear", "type": "force", "target": {"set": "upper_tip"}, "vector": [0, 0, 0.6]}},
        {"command": "model.step.create", "params": {"name": "Torque", "procedure": {"type": "static"}, "boundary_conditions": ["root"], "loads": ["lower_shear", "upper_shear"]}},
        {"command": "solve.static", "params": {}, "id": "solve"},
    ]

k = int(sys.argv[1]) if len(sys.argv) > 1 else 4
print("[\n " + ",\n ".join(json.dumps(c) for c in le5_solid(k)) + "\n]")
```

## Refinement

Every row is the listing above with the row's k, solved and probed at A as the case does; the
case mesh is k = 4:

| k | elements | σ_xx at A | error |
|---|---|---|---|
| 1 | 128 | −108.93 | +0.86% |
| 2 | 384 | −109.27 | +1.17% |
| 4 (this case) | 1280 | −109.46 | +1.35% |
| 8 | 4608 | −109.60 | +1.48% |

The result converges to about −109.7, 1.6% beyond the target. The published value comes from a
thin-walled idealisation of the section; a solid model of the same section differs from it by
about that much (ESRD's guide reports −109.5 for its converged p-version solid model on two
meshes, p. 16). The next refinement of the case mesh moves the value by 0.13%, under a tenth of
the tolerance.

A model of quad4 MITC4 shells on the mid-surface (walls meeting at the junction lines, the edge
shears as consistent nodal forces) converges further away, to about −111.3 (320 elements along the
length and 16 per metre across the walls: −111.65 and −111.06 at A from the two elements that meet
there), 3.1% beyond the target (the Abaqus Benchmarks Guide, Table 4.2.5–1, reports −110.38 for its
S4 shell on the refined mesh); we have not separated how much of the difference is the shell
idealisation of the junctions. That is why this case is the solid model.

## Tolerance

2%: the target is published to three figures, and the solid model's converged value differs from
the thin-walled reference by about 1.5% (above). `fixture.json` holds the k = 4 value: the mesh is
built by `mesh.block` from the recipe's corners, so it is the same on every platform.

## Running it

```sh
cargo xtask oracle --case nafems-le5
cargo xtask oracle --bless nafems-le5   # only after the case passes
```
