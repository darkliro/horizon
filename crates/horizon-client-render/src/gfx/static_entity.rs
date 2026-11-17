//! GPU static entity vertex layout and storage structures.

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct StaticEntityUniform {
    pub(super) camera_block: [i32; 4],
    pub(super) camera_fraction: [f32; 4],
}

pub(super) struct GpuStaticEntitySection {
    pub(super) buffer: wgpu::Buffer,
    pub(super) origin_buffer: wgpu::Buffer,
    pub(super) origin: [i32; 3],
    pub(super) capacity: u64,
    pub(super) revision: u64,
    pub(super) vertices: u32,
}
