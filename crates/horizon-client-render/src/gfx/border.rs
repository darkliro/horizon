//! World border rendering types and shaders.

use super::wgsl_prefix;

/// Returns the WGSL shader source for the world border pass.
pub fn border_shader() -> String {
    wgsl_prefix::HDR_LIGHT.to_owned() + include_str!("../border.wgsl")
}

/// One frame of the world border wall (`WorldBorderRenderer.render`).
#[derive(Clone, Debug)]
pub struct BorderDraw {
    /// The cached wall buffer; uploaded only when `rebuilt`.
    pub vertices: [crate::border_render::BorderVertex; 16],
    pub rebuilt: bool,
    pub transforms: crate::border_render::BorderTransforms,
    /// Walls to draw (indices of four-vertex quads), nearest first.
    pub sides: Vec<usize>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct BorderUniform {
    pub(super) color: [f32; 4],
    pub(super) offset: [f32; 4],
}
