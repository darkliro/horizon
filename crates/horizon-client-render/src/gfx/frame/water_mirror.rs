use super::super::*;

impl Renderer {
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn draw_water_mirror(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &Frame,
        globals: Globals,
        view_proj: glam::Mat4,
        scaled: Option<&(wgpu::TextureView, wgpu::TextureView)>,
        mirror_plane: Option<f32>,
        clear: wgpu::Color,
        sky_visible: bool,
        flat_frame: bool,
        solid_draws: u32,
    ) -> Option<wgpu::TextureView> {
        let mut reflection = None;
        // The water mirror, for one of Horizon's own packs that reads it: the world as a
        // camera mirrored about the water plane sees it, into a half-resolution target the
        // material stage samples as `reflection`.
        if let Some(plane) = mirror_plane {
            let world = self
                .world_post
                .size()
                .filter(|_| scaled.is_some())
                .unwrap_or([self.config.width, self.config.height]);
            let size = native_post::reflection_size(world);
            if self.mirror.as_ref().is_none_or(|m| m.size != size) {
                let target = |label: &'static str, format, usage| {
                    self.device
                        .create_texture(&wgpu::TextureDescriptor {
                            label: Some(label),
                            size: wgpu::Extent3d {
                                width: size[0],
                                height: size[1],
                                depth_or_array_layers: 1,
                            },
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format,
                            usage,
                            view_formats: &[],
                        })
                        .create_view(&Default::default())
                };
                self.mirror = Some(MirrorTarget {
                    size,
                    color: target(
                        "Water mirror",
                        self.output.frame.format,
                        wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                    ),
                    depth: target(
                        "Water mirror depth",
                        wgpu::TextureFormat::Depth32Float,
                        wgpu::TextureUsages::RENDER_ATTACHMENT,
                    ),
                });
            }
            if let Some(mirror) = &self.mirror {
                // Mirrored about y = `plane` (camera-relative).
                let plane = plane - frame.camera.y as f32;
                let inverse_view = view::mirror_inverse_view(globals.inverse_view, plane);
                // fog_ends.w: the plane, below which fs_solid_mirror cuts the world away
                // (only entity.wgsl reads the glint strength there, and no entity draws here).
                let mut fog_ends = globals.fog_ends;
                fog_ends[3] = plane;
                let mirror_globals = Globals {
                    view_proj: view::mirror_view_projection(view_proj, plane).to_cols_array_2d(),
                    inverse_view,
                    fog_ends,
                    hdr_light: [0.0, 0.0, globals.hdr_light[2], 0.0],
                    ..globals
                };
                self.uploads.buffer(
                    encoder,
                    &self.mirror_globals,
                    0,
                    bytemuck::bytes_of(&mirror_globals),
                );
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Water mirror"),
                    timestamp_writes: self
                        .own_timing
                        .as_mut()
                        .and_then(|t| t.writes("Water mirror")),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &mirror.color,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(clear),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &mirror.depth,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0.0),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });
                pass.set_bind_group(0, &self.mirror_group, &[]);
                if sky_visible && frame.skybox {
                    pass.set_pipeline(&self.sky);
                    pass.draw(0..3, 0..1);
                }
                // The solid terrain: occlusion culling is off with a mirror, so every draw
                // is phase 1's.
                if flat_frame && let Some(flat) = &self.flat {
                    pass.set_pipeline(&flat.solid_mirror);
                    let vertices = flat_vertices(&self.arena, self.translucent_arena.as_ref());
                    flat.draw(&mut pass, &flat.calls[..flat.solid_calls], vertices);
                } else {
                    pass.set_pipeline(&self.solid_mirror);
                    pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                    pass.set_vertex_buffer(1, self.instances.slice(..));
                    self.draw_solid_sections(&mut pass, &self.indirect, 0, 0, solid_draws);
                }
                drop(pass);
                reflection = Some(mirror.color.clone());
            }
        }
        reflection
    }
}
