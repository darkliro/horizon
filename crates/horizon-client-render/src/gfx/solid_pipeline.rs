/// A solid terrain pipeline without HDR output's lighting (terrain.wgsl's `HDR_LIGHTING`),
/// for the frames HDR is off, and with it where the display offers HDR output: the
/// lighting's code, even never run, slows the shader on Apple GPUs (~15% of the solid
/// pass at Retina sizes).
pub struct SolidPipeline {
    pub(super) plain: wgpu::RenderPipeline,
    pub(super) hdr: Option<wgpu::RenderPipeline>,
}

impl SolidPipeline {
    /// The pipeline for a frame with HDR output on (`hdr`) or off.
    pub fn get(&self, hdr: bool) -> &wgpu::RenderPipeline {
        match &self.hdr {
            Some(lit) if hdr => lit,
            _ => &self.plain,
        }
    }
}

/// The world pass's part that draws a shader pack's solid terrain: the world's colour and
/// depth loaded, and its `surface` target, cleared on the first of these passes.
pub(super) fn surface_pass<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    surface: &wgpu::TextureView,
    depth: &wgpu::TextureView,
    clear: bool,
) -> wgpu::RenderPass<'e> {
    let keep = |load| wgpu::Operations {
        load,
        store: wgpu::StoreOp::Store,
    };
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Horizon world, pack terrain"),
        color_attachments: &[
            Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: keep(wgpu::LoadOp::Load),
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: surface,
                resolve_target: None,
                depth_slice: None,
                ops: keep(if clear {
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                } else {
                    wgpu::LoadOp::Load
                }),
            }),
        ],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        ..Default::default()
    })
}
