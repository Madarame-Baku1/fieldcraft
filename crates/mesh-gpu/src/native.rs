//! The sweep device on wgpu (see the crate docs).

use fieldcraft_gpu::select::{self, Policy};
use fieldcraft_mesh::project::device::{Move, Rule, SweepDevice, SweepMesh, register_sweep_device};
use fieldcraft_mesh::{Error, Result};
use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("sweep.wgsl");

/// `u32`s per class node in the rule buffer (see `sweep.wgsl`).
const STRIDE: usize = 24;

/// Threads per workgroup (`@workgroup_size` in `sweep.wgsl`).
const GROUP: u32 = 64;

/// Most workgroups along one dispatch dimension.
const MAX_GROUPS: u32 = 65_535;

/// How long one wait for the device may take before smoothing gives up on it.
const DEVICE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

fn gpu_error(e: impl std::fmt::Display) -> Error {
    Error::Failed(format!("GPU smoothing: {e}"))
}

/// Register the GPU as the process's sweep device (an application does this once at start).
pub fn register() {
    register_sweep_device(Box::new(|mesh| GpuSweep::from_env(mesh).map(|g| Box::new(g) as Box<dyn SweepDevice>)));
}

/// The uniform of one dispatch (`Params` in `sweep.wgsl`).
fn params(start: u32, count: u32, floor: f32, min_move: f32, row: u32, around_base: u32) -> Vec<u8> {
    bytes_u32(&[start, count, floor.to_bits(), min_move.to_bits(), row, around_base, 0, 0])
}

fn bytes_u32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn index(n: usize, what: &str) -> Result<u32> {
    u32::try_from(n).map_err(|_| Error::Limit(format!("{n} {what} are more than the GPU smoother indexes")))
}

/// The free nodes of the coloured sweeps on a GPU.
pub struct GpuSweep {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: String,
    sweep: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    params: wgpu::Buffer,
    out: wgpu::Buffer,
    upd: wgpu::Buffer,
    upd_node: wgpu::Buffer,
    /// Where each class starts in the rule and output buffers, and its node count.
    classes: Vec<(u32, u32)>,
    /// The most host moves one [`SweepDevice::set`] takes.
    upd_room: usize,
    origin: [f64; 3],
    floor: f32,
    min_move: f32,
    around_base: u32,
}

impl GpuSweep {
    /// The sweeps of `mesh` on the device `FIELDCRAFT_GPU` picks.
    pub fn from_env(mesh: &SweepMesh) -> Result<GpuSweep> {
        GpuSweep::new(mesh, Policy::from_env().map_err(gpu_error)?)
    }

    /// The sweeps of `mesh` on the device `policy` picks.
    pub fn new(mesh: &SweepMesh, policy: Policy) -> Result<GpuSweep> {
        let Some((adapter, tier)) = select::adapter(policy).map_err(gpu_error)? else {
            let why = if policy == Policy::Cpu { format!("{}=cpu", select::ENV) } else { format!("no adapter matches {}={policy:?}", select::ENV) };
            return Err(Error::Invalid(format!("{why}: no GPU device; smooth with the `cpu` or `coloured` backend")));
        };
        let info = adapter.get_info();
        let have = adapter.limits();
        let mut limits = wgpu::Limits::default().using_resolution(have.clone());
        limits.max_buffer_size = have.max_buffer_size;
        limits.max_storage_buffer_binding_size = have.max_storage_buffer_binding_size;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("fieldcraft smoothing"),
            required_limits: limits,
            ..Default::default()
        }))
        .map_err(gpu_error)?;
        let adapter_name = format!("{}{}", info.name, if tier == select::Tier::Software { " (software)" } else { "" });

        // Positions relative to the centre of the nodes' box, where single precision is finest.
        let n = mesh.nodes.len();
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for p in mesh.nodes {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let origin = if n == 0 { [0.0; 3] } else { [0, 1, 2].map(|k| 0.5 * (lo[k] + hi[k])) };
        let mut pos = Vec::with_capacity(4 * n);
        for p in mesh.nodes {
            pos.extend([(p[0] - origin[0]) as f32, (p[1] - origin[1]) as f32, (p[2] - origin[2]) as f32, 0.0]);
        }
        let hexes: Vec<u32> = mesh.hexes.iter().flatten().copied().collect();
        let mut around = Vec::with_capacity(n + 1 + 8 * mesh.hexes.len());
        let mut offset = 0usize;
        around.push(0);
        for a in mesh.around {
            offset += a.len();
            around.push(index(offset, "element references")?);
        }
        let around_base = index(around.len(), "nodes")?;
        for a in mesh.around {
            around.extend_from_slice(a);
        }
        let mut rules = Vec::new();
        let mut means = Vec::new();
        let mut classes = Vec::with_capacity(mesh.classes.len());
        for class in &mesh.classes {
            classes.push((index(rules.len() / STRIDE, "class nodes")?, index(class.len(), "class nodes")?));
            for node in class {
                let mut r = [0u32; STRIDE];
                r[0] = node.node;
                match &node.rule {
                    Rule::Winslow { nb, diag, control } => {
                        for (slot, v) in r[2..8].iter_mut().zip(nb.iter().flatten()) {
                            *slot = *v;
                        }
                        for (slot, v) in r[8..20].iter_mut().zip(diag.iter().flatten().flatten()) {
                            *slot = *v;
                        }
                        for (slot, c) in r[20..23].iter_mut().zip(control) {
                            *slot = (*c as f32).to_bits();
                        }
                    }
                    Rule::Mean(edges) => {
                        r[1] = 1;
                        r[2] = index(means.len(), "neighbour references")?;
                        r[3] = index(edges.len(), "neighbours")?;
                        means.extend_from_slice(edges);
                    }
                }
                rules.extend_from_slice(&r);
            }
        }
        let class_nodes = rules.len() / STRIDE;
        // Every class's boundary nodes may move: room for the largest class's rest.
        let upd_room = mesh.around.len().max(1);

        let cap = device.limits().max_storage_buffer_binding_size.min(device.limits().max_buffer_size);
        for (what, bytes) in
            [("node positions", 16 * n), ("elements", 4 * hexes.len()), ("element lists", 4 * around.len()), ("stencils", 4 * rules.len())]
        {
            if bytes as u64 > cap {
                return Err(Error::Limit(format!("the {what} need {} MB; this GPU allows {} MB per buffer", bytes >> 20, cap >> 20)));
            }
        }

        let scopes = [wgpu::ErrorFilter::OutOfMemory, wgpu::ErrorFilter::Validation].map(|f| device.push_error_scope(f));
        let storage = |label: &str, bytes: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: if bytes.is_empty() { &[0u8; 16][..] } else { bytes },
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            })
        };
        let empty = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let pos_buf = storage("smoothing positions", &bytes_f32(&pos));
        let hex_buf = storage("smoothing elements", &bytes_u32(&hexes));
        let around_buf = storage("smoothing element lists", &bytes_u32(&around));
        let rule_buf = storage("smoothing stencils", &bytes_u32(&rules));
        let mean_buf = storage("smoothing neighbours", &bytes_u32(&means));
        let out = empty("smoothing moves", 16 * class_nodes as u64);
        let upd = empty("smoothing host moves", 16 * upd_room as u64);
        let upd_node = empty("smoothing host moved nodes", 4 * upd_room as u64);
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("smoothing params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fieldcraft smoothing"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let read = wgpu::BufferBindingType::Storage { read_only: true };
        let write = wgpu::BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fieldcraft smoothing"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, write),
                entry(2, read),
                entry(3, read),
                entry(4, read),
                entry(5, read),
                entry(6, write),
                entry(7, read),
                entry(8, read),
            ],
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fieldcraft smoothing"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: hex_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: around_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: rule_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: mean_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: upd.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: upd_node.as_entire_binding() },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fieldcraft smoothing"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |name: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(name),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let (sweep, scatter) = (pipeline("sweep"), pipeline("scatter"));
        for scope in scopes.into_iter().rev() {
            if let Some(e) = pollster::block_on(scope.pop()) {
                return Err(gpu_error(e));
            }
        }
        Ok(GpuSweep {
            device,
            queue,
            adapter: adapter_name,
            sweep,
            scatter,
            bind,
            params: params_buf,
            out,
            upd,
            upd_node,
            classes,
            upd_room,
            origin,
            floor: mesh.floor as f32,
            min_move: mesh.min_move as f32,
            around_base,
        })
    }

    /// Workgroups for `count` threads as `(x, y)`, and threads per row.
    fn grid(count: u32) -> (u32, u32, u32) {
        let groups = count.div_ceil(GROUP).max(1);
        let gx = groups.min(MAX_GROUPS);
        (gx, groups.div_ceil(gx), gx * GROUP)
    }

    fn dispatch(&self, pipeline: &wgpu::ComputePipeline, start: u32, count: u32, copy: Option<(&wgpu::Buffer, u64)>) {
        let (gx, gy, row) = GpuSweep::grid(count);
        self.queue.write_buffer(&self.params, 0, &params(start, count, self.floor, self.min_move, row, self.around_base));
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fieldcraft smoothing") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("fieldcraft smoothing"), timestamp_writes: None });
            pass.set_bind_group(0, &self.bind, &[]);
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        if let Some((staging, size)) = copy {
            enc.copy_buffer_to_buffer(&self.out, 16 * u64::from(start), staging, 0, size);
        }
        self.queue.submit([enc.finish()]);
    }
}

impl SweepDevice for GpuSweep {
    fn name(&self) -> String {
        format!("gpu: {}", self.adapter)
    }

    fn sweep(&mut self, class: usize) -> Result<Vec<Move>> {
        let &(start, count) = self.classes.get(class).ok_or_else(|| gpu_error(format!("no colour class {class}")))?;
        if count == 0 {
            return Ok(Vec::new());
        }
        let size = 16 * u64::from(count);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("smoothing readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.dispatch(&self.sweep, start, count, Some((&staging, size)));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // A bounded wait: a hung or lost device is an error, never a hang.
        self.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(DEVICE_TIMEOUT) })
            .map_err(|e| gpu_error(format!("the device did not answer within {} s: {e}", DEVICE_TIMEOUT.as_secs())))?;
        rx.try_recv().map_err(|_| gpu_error("the readback did not complete"))?.map_err(gpu_error)?;
        let bytes = slice.get_mapped_range().map_err(gpu_error)?.to_vec();
        staging.unmap();
        let o = self.origin;
        let moves = bytes
            .as_chunks::<16>()
            .0
            .iter()
            .map(|c| {
                let [x, y, z, w] = c.as_chunks::<4>().0 else { return Move::Stay };
                let f = |b: &[u8; 4]| f32::from_le_bytes(*b);
                match f(w) {
                    1.0 => Move::To([o[0] + f64::from(f(x)), o[1] + f64::from(f(y)), o[2] + f64::from(f(z))]),
                    2.0 => Move::Refused,
                    _ => Move::Stay,
                }
            })
            .collect();
        Ok(moves)
    }

    fn set(&mut self, moved: &[(u32, [f64; 3])]) -> Result<()> {
        for part in moved.chunks(self.upd_room) {
            let count = index(part.len(), "moved nodes")?;
            let o = self.origin;
            let at: Vec<f32> = part.iter().flat_map(|(_, p)| [(p[0] - o[0]) as f32, (p[1] - o[1]) as f32, (p[2] - o[2]) as f32, 0.0]).collect();
            let nodes: Vec<u32> = part.iter().map(|(v, _)| *v).collect();
            self.queue.write_buffer(&self.upd, 0, &bytes_f32(&at));
            self.queue.write_buffer(&self.upd_node, 0, &bytes_u32(&nodes));
            self.dispatch(&self.scatter, 0, count, None);
        }
        Ok(())
    }
}
