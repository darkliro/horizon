use super::*;

pub(in crate::gfx) struct PostPassArgs<'a> {
    pub(in crate::gfx) frame: &'a Frame<'a>,
    pub(in crate::gfx) scaled: Option<(wgpu::TextureView, wgpu::TextureView)>,
    pub(in crate::gfx) view_target: wgpu::TextureView,
    pub(in crate::gfx) target_texture: wgpu::Texture,
    pub(in crate::gfx) surface_frame: Option<&'a wgpu::SurfaceTexture>,
    pub(in crate::gfx) fg_frames: Option<&'a (wgpu::Texture, wgpu::Texture)>,
    pub(in crate::gfx) pack_post: bool,
    pub(in crate::gfx) native_post: bool,
    pub(in crate::gfx) pack_draws_weather: bool,
    pub(in crate::gfx) pack_draws_terrain: bool,
    pub(in crate::gfx) pack_draws_shadow: bool,
    pub(in crate::gfx) pack_draws_outline: bool,
    pub(in crate::gfx) pack_draws_clouds: bool,
    pub(in crate::gfx) pack_draws_hand: bool,
    pub(in crate::gfx) flat_frame: bool,
    pub(in crate::gfx) outline_edges: usize,
    pub(in crate::gfx) cloud_quads: u32,
    pub(in crate::gfx) solid_draws: u32,
    pub(in crate::gfx) translucent_draws: u32,
    pub(in crate::gfx) shadow_first: u32,
    pub(in crate::gfx) shadow_solid: u32,
    pub(in crate::gfx) shadow_translucent: u32,
    pub(in crate::gfx) size: [u32; 2],
    pub(in crate::gfx) jittered: Mat4,
    pub(in crate::gfx) view_proj: Mat4,
    pub(in crate::gfx) hand_view_proj: Mat4,
    pub(in crate::gfx) hand_quads: u32,
    pub(in crate::gfx) made_hand_vertex: i32,
    pub(in crate::gfx) crosshair: u32,
    pub(in crate::gfx) debug_depth: u32,
    pub(in crate::gfx) debug_top: u32,
    pub(in crate::gfx) globals: Globals,
    pub(in crate::gfx) entity_quads: u32,
    pub(in crate::gfx) hand_in_world: bool,
    pub(in crate::gfx) world_quads: u32,
    pub(in crate::gfx) additive_quads: u32,
    pub(in crate::gfx) culling: bool,
    pub(in crate::gfx) gpu_frame: bool,
    pub(in crate::gfx) serial: u64,
    pub(in crate::gfx) cpu: [usize; 3],
}

pub(in crate::gfx) struct PostPassResult {
    pub(in crate::gfx) readback: Option<(ScreenshotRequest, wgpu::Buffer)>,
    pub(in crate::gfx) target_texture: wgpu::Texture,
    pub(in crate::gfx) fg_made: bool,
}

impl Renderer {
    pub(in crate::gfx) fn run_post_and_overlay(
        &mut self,
        mut encoder: wgpu::CommandEncoder,
        args: PostPassArgs<'_>,
    ) -> (wgpu::CommandEncoder, PostPassResult) {
        let PostPassArgs {
            frame,
            scaled,
            view_target,
            target_texture,
            surface_frame,
            fg_frames,
            pack_post,
            native_post,
            pack_draws_weather,
            pack_draws_terrain,
            pack_draws_shadow,
            pack_draws_outline,
            pack_draws_clouds,
            pack_draws_hand,
            flat_frame,
            outline_edges,
            cloud_quads,
            solid_draws,
            translucent_draws,
            shadow_first,
            shadow_solid,
            shadow_translucent,
            size,
            jittered,
            view_proj,
            hand_view_proj,
            hand_quads,
            made_hand_vertex,
            crosshair,
            debug_depth,
            debug_top,
            globals,
            entity_quads,
            hand_in_world,
            world_quads,
            additive_quads,
            culling,
            gpu_frame,
            serial,
            cpu,
        } = args;

        let mut fg_made = false;

        if let Some((color, depth)) = &scaled
            && pack_post
            && let Some((dimension, Some(mut post))) = self.pack_post.take()
        {
            let size = self.world_post.size().unwrap_or([1, 1]);
            let weather = match &self.hud.atlas_texture {
                Some(atlas) if pack_draws_weather => self
                    .pack_weather
                    .get_or_insert_with(|| pack_weather::PackWeather::new(&self.device))
                    .prepare(
                        &self.device,
                        &mut encoder,
                        &mut self.uploads,
                        atlas,
                        frame.weather,
                        frame.weather_time,
                        &self.indices,
                    ),
                _ => None,
            };
            post.set_weather(weather);
            let flat = self.flat.as_ref().filter(|_| flat_frame);
            self.pack_pages.clear();
            if flat.is_some() && (pack_draws_terrain || pack_draws_shadow) {
                self.pack_pages.extend(
                    self.arena
                        .page_buffers()
                        .map(|(p, b)| (p, b.clone(), self.arena.sources(p).cloned())),
                );
            }
            let pages = &self.pack_pages;
            let mut terrain =
                |rp: &mut wgpu::RenderPass<'_>,
                 list: pack_post::TerrainList,
                 page: &dyn Fn(&mut wgpu::RenderPass<'_>, u32),
                 geometry: Option<(u32, &wgpu::BindGroupLayout)>| {
                    use pack_post::TerrainList as L;
                    let shadow = matches!(list, L::ShadowSolid | L::ShadowTranslucent);
                    if !(if shadow {
                        pack_draws_shadow
                    } else {
                        pack_draws_terrain
                    }) {
                        return;
                    }
                    match flat {
                        Some(flat) => {
                            let [shadow, shadow_translucent] = flat.shadow_calls;
                            let calls = match list {
                                L::Solid => &flat.calls[..flat.solid_calls],
                                L::Translucent => flat.translucent_calls(),
                                L::ShadowSolid => &flat.calls[shadow..shadow_translucent],
                                L::ShadowTranslucent => flat.shadow_translucent_calls(),
                            };
                            match geometry {
                                Some((instances, layout)) => {
                                    let indices =
                                        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                            label: Some("Shader pack geometry indices"),
                                            layout,
                                            entries: &[wgpu::BindGroupEntry {
                                                binding: 0,
                                                resource: flat.indices.as_entire_binding(),
                                            }],
                                        });
                                    rp.set_bind_group(5, &indices, &[]);
                                    let vertices =
                                        flat_vertices(&self.arena, self.translucent_arena.as_ref());
                                    flat.draw_geometry(rp, calls, vertices, page, instances);
                                }
                                None => flat.draw_with(
                                    rp,
                                    calls,
                                    flat_vertices(&self.arena, self.translucent_arena.as_ref()),
                                    page,
                                ),
                            }
                        }
                        // A geometry stage reads flat draws' indices; without them it draws nothing.
                        None if geometry.is_some() => {}
                        None => {
                            rp.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                            rp.set_vertex_buffer(1, self.instances.slice(..));
                            let (first, count) = match list {
                                L::Solid => (0, solid_draws),
                                L::Translucent => (solid_draws, translucent_draws),
                                L::ShadowSolid => (shadow_first, shadow_solid),
                                L::ShadowTranslucent => {
                                    (shadow_first + shadow_solid, shadow_translucent)
                                }
                            };
                            self.draw_sections(rp, &self.indirect, 0, first, count);
                        }
                    }
                };
            // Both strokes of a high contrast outline, the black one first.
            post.set_lines(pack_draws_outline.then(|| pack_post::Lines {
                edges: self.outline.clone(),
                count: outline_edges as u32 * if frame.outline_high_contrast { 2 } else { 1 },
            }));
            post.set_clouds(pack_draws_clouds.then(|| pack_post::Clouds {
                vertices: self.cloud_buffer.clone(),
                group: self.cloud_group.clone(),
                indices: self.indices.clone(),
                quads: cloud_quads,
            }));
            post.run(
                &self.device,
                &self.queue,
                &mut encoder,
                color,
                depth,
                &view_target,
                frame,
                &frame.pack,
                size,
                [self.config.width, self.config.height],
                pages,
                &mut terrain,
            );
            // Keep the capacity, not the buffers: a page the arena frees must not stay alive
            // here until the next shader-pack frame (or forever once the pack is off).
            self.pack_pages.clear();
            self.pack_post = Some((dimension, Some(post)));
        } else if let Some((color, depth)) = &scaled
            && native_post
            && let Some(native) = &mut self.native
        {
            native.run_post(native_pack::WorldPost {
                device: &self.device,
                encoder: &mut encoder,
                world: (color, depth),
                screen: &view_target,
                size: self.world_post.size().unwrap_or([1, 1]),
                scene: self.ray_scene.as_ref().map(|s| (s, frame.camera)),
            });
        } else if let (Some((_, generated)), Some((_, depth))) = (fg_frames, &scaled) {
            let camera = frame_gen::Camera {
                jittered,
                view_proj,
                depth: depth.clone(),
                render_size: self.world_post.size().unwrap_or(size),
                jitter: self.world_post.jitter_pixels(),
                reproject: self.frame_gen.camera_motion(view_proj, frame.camera),
                near: NEAR_PLANE,
                far: frame.depth_far,
                fov_y: frame.fov_degrees.to_radians(),
            };
            let overlay = Overlay {
                hand_quads: hand_quads * u32::from(!pack_draws_hand),
                first_hand_vertex: made_hand_vertex,
                crosshair,
                first_crosshair: debug_depth + debug_top,
                world_depth: false,
                real: false,
                hand_drawn: !pack_draws_hand,
                hand_view_proj,
            };
            let onto = (&view_target, &target_texture, generated);
            fg_made = self.draw_generated(&mut encoder, frame, &camera, &overlay, onto);
        } else if scaled.is_some() {
            self.world_post.finish(
                &self.queue,
                &mut encoder,
                &view_target,
                self.own_timing.as_mut(),
            );
        }
        if !frame.title {
            self.draw_world_views(&mut encoder, frame, globals, flat_frame, entity_quads);
            let (depth, depth_size) = match &scaled {
                Some((_, depth)) => (depth, self.world_post.size().unwrap_or(size)),
                None => (&self.depth, size),
            };
            native_render::world_drawn(&mut native_render::WorldDrawn {
                device: &self.device,
                queue: &self.queue,
                encoder: &mut encoder,
                color: &target_texture,
                color_view: &view_target,
                color_format: self.output.frame.overlay_format,
                size,
                depth,
                depth_size,
                view_proj,
                jittered,
                camera: frame.camera,
                entities: frame.entities,
                spans: frame.entity_spans,
                motion: frame.entity_motion,
                atlas: self.hud.atlas_texture.as_ref(),
                quad_indices: &self.indices,
                block_sprites: native_render::BlockSpriteArrays {
                    views: &self.sprite_views,
                    split: self.sprite_split,
                    size: self.sprite_textures[0].width(),
                    nearest: &self.sprite_nearest,
                    filtered: &self.sprite_sampler,
                },
            });
        }
        // A 0.2 pack in SDR: its picture finishes on 8 bits before the hand, the camera overlays
        // and the HUD draw over it (`overlay_format`).
        let finished = (self.output.frame.overlay_format != self.output.frame.format)
            .then(|| {
                let image = surface_frame.as_ref().map(|f| &f.texture);
                image.or(self.offscreen.as_ref())
            })
            .flatten()
            .and_then(|image| {
                let screenshot = self.screenshot.is_some();
                self.readback
                    .finish(&self.device, &mut encoder, image, screenshot)
            });
        let (target_texture, view_target, overlay_format) = match finished {
            Some(texture) => {
                let view = texture.create_view(&wgpu::TextureViewDescriptor {
                    format: Some(self.output.frame.overlay_format),
                    ..Default::default()
                });
                (texture, view, self.output.frame.overlay_format)
            }
            None => (target_texture, view_target, self.output.frame.format),
        };
        let overlay = Overlay {
            hand_quads: hand_quads * u32::from(!pack_draws_hand && !hand_in_world),
            first_hand_vertex: match fg_made && frame_gen::generated_only() && self.hdr.is_some() {
                true => made_hand_vertex,
                false => ((world_quads + additive_quads) * 4) as i32,
            },
            crosshair,
            first_crosshair: debug_depth + debug_top,
            world_depth: scaled.is_none(),
            real: true,
            hand_drawn: !pack_draws_hand,
            hand_view_proj,
        };
        let readback =
            self.draw_overlay(&mut encoder, frame, &view_target, &target_texture, &overlay);
        if culling && let Some(cull) = &mut self.cull {
            cull.end(&self.device, &mut encoder);
        }
        // GPU culling's counts once its world passes are done (phase 2's too).
        if gpu_frame && let Some(gpu) = &mut self.gpu_culling {
            gpu.copy_stats(&mut encoder, serial, cpu);
        }
        if let Some(timer) = &mut self.gpu_timer {
            timer.end(&self.device, &mut encoder, (&view_target, overlay_format));
        }

        (
            encoder,
            PostPassResult {
                readback,
                target_texture,
                fg_made,
            },
        )
    }
}
