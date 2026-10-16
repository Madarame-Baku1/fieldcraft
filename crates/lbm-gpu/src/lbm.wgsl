// The lattice Boltzmann step of fieldcraft_lbm::flow on the GPU: the same D3Q19 two-relaxation-
// time collision with Guo forcing, pull streaming with the per-link rules the CPU prepared
// (bounce-back with Ladd's moving-wall term, specular reflection, open placeholders), the Zou–He
// open boundaries, and the momentum exchange per force group. Populations are stored direction-
// major (f[q n + cell]) in two buffers, A–B: a step collides `f` in place, streams `f` into `g`
// and applies the open boundaries to `g`; the next step binds the buffers the other way round.
//
// Single precision, with every population stored as its deviation from the weight, h_q = f_q − w_q
// (Krüger et al. 2017, §13.3.4: the deviations are small, so f32 keeps their digits instead of
// the weight's): ρ = 1 + Σ h, momentum Σ h c (Σ w c = 0), and every rule below is written for h.
// The CPU scheme (double precision, fixed sums) is the reference; see the crate's tests for the
// tolerances.

const Q: u32 = 19u;
const WG: u32 = 256u;
const KIND_PULL: u32 = 0u;
const KIND_BOUNCE: u32 = 1u;
const KIND_REFLECT: u32 = 2u;
const KIND_OPEN: u32 = 3u;

struct Params {
    n: u32,
    nx: u32,
    ny: u32,
    nz: u32,
    groups_x: u32,
    links: u32,
    open: u32,
    groups: u32,
    omega_plus: f32,
    omega_minus: f32,
    s_plus: f32,
    s_minus: f32,
    // The body acceleration (xyz) and whether there is one (w).
    acc: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> fluid: array<u32>;
@group(0) @binding(2) var<storage, read_write> f: array<f32>;
@group(0) @binding(3) var<storage, read_write> g: array<f32>;
// Per link: (q | kind << 8 | q2 << 16, from, group, 0) and the moving-wall term.
@group(0) @binding(5) var<storage, read> link_u: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> link_f: array<f32>;
// Per open cell: (cell, axis, kind: 0 velocity / 1 density, 0) and (target xyz or density in x, inward sign in w).
@group(0) @binding(7) var<storage, read> open_u: array<vec4<u32>>;
@group(0) @binding(8) var<storage, read> open_f: array<vec4<f32>>;
// Per cell: velocity (with the half force) and density, from the moments kernel.
@group(0) @binding(9) var<storage, read_write> moments: array<vec4<f32>>;
// [0]: the largest squared lattice speed of the chunk (as f32 bits); [1]: a density was not positive.
@group(0) @binding(10) var<storage, read_write> stats: array<atomic<u32>>;
// Per link, the momentum it transferred to its group in the last step.
@group(0) @binding(11) var<storage, read_write> contrib: array<vec4<f32>>;
// Link indices ordered by group, and each group's range in that order (groups + 1 entries).
@group(0) @binding(12) var<storage, read> by_group: array<u32>;
@group(0) @binding(13) var<storage, read> group_start: array<u32>;
@group(0) @binding(14) var<storage, read_write> forces: array<vec4<f32>>;
// Per workgroup, the mass of its cells.
@group(0) @binding(15) var<storage, read_write> partials: array<f32>;

var<workgroup> red: array<vec4<f32>, 256>;
// The largest squared speed of a workgroup's cells (collide): one atomic per workgroup, not
// per cell, which serialise on a single address.
var<workgroup> speed: array<f32, 256>;

fn c_of(q: u32) -> vec3<i32> {
    switch q {
        case 0u: { return vec3<i32>(0, 0, 0); }
        case 1u: { return vec3<i32>(1, 0, 0); }
        case 2u: { return vec3<i32>(-1, 0, 0); }
        case 3u: { return vec3<i32>(0, 1, 0); }
        case 4u: { return vec3<i32>(0, -1, 0); }
        case 5u: { return vec3<i32>(0, 0, 1); }
        case 6u: { return vec3<i32>(0, 0, -1); }
        case 7u: { return vec3<i32>(1, 1, 0); }
        case 8u: { return vec3<i32>(-1, -1, 0); }
        case 9u: { return vec3<i32>(1, -1, 0); }
        case 10u: { return vec3<i32>(-1, 1, 0); }
        case 11u: { return vec3<i32>(1, 0, 1); }
        case 12u: { return vec3<i32>(-1, 0, -1); }
        case 13u: { return vec3<i32>(1, 0, -1); }
        case 14u: { return vec3<i32>(-1, 0, 1); }
        case 15u: { return vec3<i32>(0, 1, 1); }
        case 16u: { return vec3<i32>(0, -1, -1); }
        case 17u: { return vec3<i32>(0, 1, -1); }
        default: { return vec3<i32>(0, -1, 1); }
    }
}

fn opp(q: u32) -> u32 {
    if q == 0u { return 0u; }
    if (q & 1u) == 1u { return q + 1u; }
    return q - 1u;
}

fn w_of(q: u32) -> f32 {
    if q == 0u { return 1.0 / 3.0; }
    if q < 7u { return 1.0 / 18.0; }
    return 1.0 / 36.0;
}

fn cell_of(lid: u32, wg: vec3<u32>) -> u32 {
    return (wg.y * params.groups_x + wg.x) * WG + lid;
}

fn is_fluid(cell: u32) -> bool {
    return fluid[cell] != 0u;
}

// Two-relaxation-time collision with Guo forcing, in place in f (Ginzburg et al. 2008; Guo et
// al. 2002; Krüger et al. 2017 ch. 10), and the chunk's largest speed.
@compute @workgroup_size(256)
fn collide(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let cell = cell_of(lid, wg);
    speed[lid] = 0.0;
    if cell < params.n && is_fluid(cell) {
        collide_cell(cell, lid);
    }
    workgroupBarrier();
    for (var k = 128u; k > 0u; k = k >> 1u) {
        if lid < k { speed[lid] = max(speed[lid], speed[lid + k]); }
        workgroupBarrier();
    }
    if lid == 0u { atomicMax(&stats[0], bitcast<u32>(speed[0])); }
}

fn collide_cell(cell: u32, lid: u32) {
    let n = params.n;
    var fq: array<f32, 19>;
    var drho = 0.0;
    var j = vec3<f32>(0.0);
    for (var q = 0u; q < Q; q = q + 1u) {
        let v = f[q * n + cell];
        fq[q] = v;
        drho = drho + v;
        j = j + vec3<f32>(c_of(q)) * v;
    }
    let rho = 1.0 + drho;
    if !(rho > 0.0) || rho != rho || abs(rho) > 1e30 {
        atomicStore(&stats[1], 1u);
        return;
    }
    let a = params.acc.xyz;
    let force = rho * a;
    let u = (j + 0.5 * force) / rho;
    let uu = dot(u, u);
    speed[lid] = uu;
    let uf = dot(u, force);
    let forced = params.acc.w > 0.5;
    let wp = params.omega_plus;
    let wm = params.omega_minus;
    let sp = params.s_plus;
    let sm = params.s_minus;
    // Rest population: symmetric only. Equilibria are shifted by the weight like the populations.
    let eq0 = w_of(0u) * (drho - 1.5 * rho * uu);
    var out0 = fq[0] - wp * (fq[0] - eq0);
    if forced { out0 = out0 + sp * w_of(0u) * (-3.0 * uf); }
    f[cell] = out0;
    for (var q = 1u; q < Q; q = q + 2u) {
        let o = q + 1u;
        let cv = vec3<f32>(c_of(q));
        let cu = dot(cv, u);
        let w = w_of(q);
        let eq_plus = w * (drho + rho * (4.5 * cu * cu - 1.5 * uu));
        let eq_minus = w * rho * 3.0 * cu;
        let plus = 0.5 * (fq[q] + fq[o]);
        let minus = 0.5 * (fq[q] - fq[o]);
        let relax_plus = wp * (plus - eq_plus);
        let relax_minus = wm * (minus - eq_minus);
        var s_plus = 0.0;
        var s_minus = 0.0;
        if forced {
            let cf = dot(cv, force);
            s_plus = sp * w * (9.0 * cu * cf - 3.0 * uf);
            s_minus = sm * w * 3.0 * cf;
        }
        f[q * n + cell] = fq[q] - relax_plus - relax_minus + s_plus + s_minus;
        f[o * n + cell] = fq[o] - relax_plus + relax_minus + s_plus - s_minus;
    }
}

// Pull streaming of f into g, first pass: every population from the neighbour at −c_q
// (wrapped; a wall or an open face overrides it in the second pass).
@compute @workgroup_size(256)
fn stream(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let cell = cell_of(lid, wg);
    if cell >= params.n { return; }
    if !is_fluid(cell) { return; }
    let n = params.n;
    let nx = i32(params.nx);
    let ny = i32(params.ny);
    let nz = i32(params.nz);
    let i = i32(cell % params.nx);
    let jj = i32((cell / params.nx) % params.ny);
    let k = i32(cell / (params.nx * params.ny));
    for (var q = 0u; q < Q; q = q + 1u) {
        let c = c_of(q);
        let si = (i - c.x + nx) % nx;
        let sj = (jj - c.y + ny) % ny;
        let sk = (k - c.z + nz) % nz;
        let src = u32((sk * ny + sj) * nx + si);
        g[q * n + cell] = f[q * n + src];
    }
}

// Second pass, one thread per link: the populations with a rule (a periodic image, a wall, a
// slip face, an open face). Every (cell, direction) has at most one link, so no two threads
// write the same population.
@compute @workgroup_size(256)
fn stream_links(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let l = cell_of(lid, wg);
    if l >= params.links { return; }
    let n = params.n;
    let lu = link_u[l];
    let q = lu.x & 0xffu;
    let kind = (lu.x >> 8u) & 0xffu;
    let q2 = (lu.x >> 16u) & 0xffu;
    let cell = lu.w;
    var v = 0.0;
    if kind == KIND_PULL {
        v = f[q * n + lu.y];
    } else if kind == KIND_BOUNCE {
        v = f[opp(q) * n + cell] + link_f[l];
    } else if kind == KIND_REFLECT {
        // The mirrored direction has the same weight, so h reflects like f.
        v = f[q2 * n + lu.y];
    } else {
        v = f[opp(q) * n + cell];
    }
    g[q * n + cell] = v;
}

// Zou–He velocity and pressure boundaries on the streamed populations in g (Zou & He 1997;
// Hecht & Harting 2010 for the D3Q19 transverse correction).
@compute @workgroup_size(256)
fn open_boundaries(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let o = cell_of(lid, wg);
    if o >= params.open { return; }
    let n = params.n;
    let ou = open_u[o];
    let ot = open_f[o];
    let cell = ou.x;
    let axis = ou.y;
    let sign = ot.w;
    let a_half = 0.5 * params.acc.xyz;
    // The sums of f over the tangential (c_n = 0) and outgoing (c_n < 0) directions: the sums of
    // h plus the weights' (2/3 and 1/6 for D3Q19).
    var tangential = 2.0 / 3.0;
    var outgoing = 1.0 / 6.0;
    for (var q = 0u; q < Q; q = q + 1u) {
        let v = g[q * n + cell];
        let cn = f32(c_of(q)[axis]) * sign;
        if cn == 0.0 {
            tangential = tangential + v;
        } else if cn < 0.0 {
            outgoing = outgoing + v;
        }
    }
    var rho = 0.0;
    var bare = vec3<f32>(0.0);
    if ou.z == 0u {
        bare = ot.xyz - a_half;
        let un = sign * bare[axis];
        rho = (tangential + 2.0 * outgoing) / (1.0 - un);
    } else {
        rho = ot.x;
        let un = 1.0 - (tangential + 2.0 * outgoing) / rho;
        // Not `bare[axis] = …`: DX12's shader compiler (FXC) refuses a write through a
        // dynamic index into a local vector (X3500).
        bare = select(-a_half, vec3<f32>(sign * un), vec3<u32>(0u, 1u, 2u) == vec3<u32>(axis));
    }
    for (var q = 1u; q < Q; q = q + 1u) {
        let c = c_of(q);
        if f32(c[axis]) * sign > 0.0 {
            let cu = dot(vec3<f32>(c), bare);
            g[q * n + cell] = g[opp(q) * n + cell] + 6.0 * w_of(q) * rho * cu;
        }
    }
    for (var t = 0u; t < 3u; t = t + 1u) {
        if t == axis { continue; }
        var m = -rho * bare[t];
        for (var q = 0u; q < Q; q = q + 1u) {
            m = m + g[q * n + cell] * f32(c_of(q)[t]);
        }
        for (var q = 1u; q < Q; q = q + 1u) {
            let c = c_of(q);
            if f32(c[axis]) * sign > 0.0 && c[t] != 0 {
                g[q * n + cell] = g[q * n + cell] - 0.5 * m * f32(c[t]);
            }
        }
    }
}

// The momentum each link transferred to its group in the last step, from the post-collision
// populations in f (Ladd 1994; Mei et al. 2002): c_out f* − c_in f'.
@compute @workgroup_size(256)
fn link_forces(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let l = cell_of(lid, wg);
    if l >= params.links { return; }
    let n = params.n;
    let lu = link_u[l];
    let q = lu.x & 0xffu;
    let kind = (lu.x >> 8u) & 0xffu;
    let q2 = (lu.x >> 16u) & 0xffu;
    let cell = lu.w;
    // The deviations only: the weights' share of the exchange (2 w_q c_q per bounced link, which
    // cancels between a wall's links in exact arithmetic but is 10⁴ times the shear they carry)
    // is added on the host in double precision, so the sum keeps the deviations' digits.
    var v = vec3<f32>(0.0);
    if kind == KIND_BOUNCE {
        let out = f[opp(q) * n + cell];
        let back = out + link_f[l];
        v = -vec3<f32>(c_of(q)) * (out + back);
    } else if kind == KIND_REFLECT {
        let out = f[q2 * n + lu.y];
        v = vec3<f32>(c_of(q2)) * out - vec3<f32>(c_of(q)) * out;
    }
    contrib[l] = vec4<f32>(v, 0.0);
}

// One workgroup per group: the sum of its links' contributions.
@compute @workgroup_size(256)
fn group_forces(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let group = wg.y * params.groups_x + wg.x;
    if group >= params.groups { return; }
    let lo = group_start[group];
    let hi = group_start[group + 1u];
    var s = vec4<f32>(0.0);
    for (var i = lo + lid; i < hi; i = i + WG) {
        s = s + contrib[by_group[i]];
    }
    red[lid] = s;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k = k >> 1u) {
        if lid < k { red[lid] = red[lid] + red[lid + k]; }
        workgroupBarrier();
    }
    if lid == 0u { forces[group] = red[0]; }
}

// The density and velocity (with the half force) of every cell from f; zero in solid cells.
@compute @workgroup_size(256)
fn cell_moments(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let cell = cell_of(lid, wg);
    if cell >= params.n { return; }
    if !is_fluid(cell) {
        moments[cell] = vec4<f32>(0.0);
        return;
    }
    let n = params.n;
    var drho = 0.0;
    var j = vec3<f32>(0.0);
    for (var q = 0u; q < Q; q = q + 1u) {
        let v = f[q * n + cell];
        drho = drho + v;
        j = j + vec3<f32>(c_of(q)) * v;
    }
    let rho = 1.0 + drho;
    let u = (j + 0.5 * rho * params.acc.xyz) / rho;
    moments[cell] = vec4<f32>(u, rho);
}

// Per workgroup, the sum of its cells' densities (the mass, since every cell has unit volume).
@compute @workgroup_size(256)
fn mass(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let group = wg.y * params.groups_x + wg.x;
    let cell = group * WG + lid;
    var v = 0.0;
    if cell < params.n && is_fluid(cell) {
        v = moments[cell].w;
    }
    red[lid] = vec4<f32>(v, 0.0, 0.0, 0.0);
    workgroupBarrier();
    for (var k = 128u; k > 0u; k = k >> 1u) {
        if lid < k { red[lid] = red[lid] + red[lid + k]; }
        workgroupBarrier();
    }
    if lid == 0u { partials[group] = red[0].x; }
}
