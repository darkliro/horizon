#[path = "frame/shadow.rs"]
mod shadow;
#[path = "frame/sky.rs"]
mod sky;
#[path = "frame/submit.rs"]
mod submit_frame;
use submit_frame::SubmitFrameArgs;
#[path = "frame/acquire.rs"]
mod acquire_frame;
use acquire_frame::AcquiredFrame;
#[path = "frame/post.rs"]
mod post;
#[path = "frame/water_mirror.rs"]
mod water_mirror;
use super::*;
use post::{PostPassArgs, PostPassResult};

impl Renderer {
    /// `checks`: the error scopes of a scaled screenshot's frame, checked before its PNG is written.
    /// `output`: what this frame is drawn for, the window's output or a screenshot's.
    pub(super) fn draw_frame(
        &mut self,
        frame: &Frame,
        checks: Option<GpuChecks>,
        output: output::EffectiveOutput,
    ) -> Result<bool> {
        debug_assert_eq!(output.encoding, self.hdr.as_ref().map(|hdr| hdr.output));
        // Its steps as spans (HORIZON_SPIKES): each lap names the step that just ended.
        let mut laps = crate::spans::laps();
        let landing = Instant::now();
        self.land_frame();
        // Waiting for the previous frame (offscreen: at most two on the GPU), not work.
        let landed = landing.elapsed();
        self.queue.take_waited();
        // World frames alone follow the shader pack: the game loads the picked one
        // before it draws one. Title frames draw no terrain and keep the layout
        // through a pack dropped (Shader Options) or not loaded yet (`Renderer::new`),
        // so the next world's sections, which the level loading screen uploads, stay.
        if !frame.title {
            self.follow_translucent_layout();
            self.follow_packed_layout();
        }
        let released = self.arena.trim();
        self.graveyard.add(released);
        let left = self.compact_arena();
        if let Some(arena) = &mut self.translucent_arena {
            self.graveyard.add(arena.trim());
        }
        self.graveyard.add(self.sorted_indices.trim());
        arena::free_later(self.graveyard.advance());
        self.shrink_translucent(left);
        if let Some(flat) = &mut self.flat {
            let translucent = self.translucent_arena.as_ref();
            flat.trim(|key| match key.checked_sub(flat::TRANSLUCENT_KEYS) {
                Some(page) => translucent.is_some_and(|a| a.has_page(page)),
                None => self.arena.has_page(key),
            });
        }
        if self.suspended {
            return Ok(false);
        }
        let started = Instant::now();
        let Some(AcquiredFrame {
            fg_frames,
            fg,
            surface_frame,
            reconfigure,
            target_texture,
        }) = self.acquire_frame_surface(frame, landed)?
        else {
            return Ok(false);
        };

        let size = [self.config.width, self.config.height];
        laps.lap("r_prepare");
        // One command buffer for the frame, after the section copies staged
        // since the last one: Metal commits every command buffer wgpu-core
        // makes (one per run of copies and two per pass) on its own.
        let section_copies = self.section_copies.take();
        let has_section_copies = section_copies.is_some();
        let mut encoder = section_copies.unwrap_or_else(|| {
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Horizon frame"),
                })
        });
        // Horizon's own passes are timed like a shader pack's, from the frame's first pass.
        if self.own_timing.is_some() != self.pass_timing {
            self.own_timing = self
                .pass_timing
                .then(|| pack_timing::PassTiming::new(&self.device, &self.queue))
                .flatten();
        }
        if let Some(t) = &mut self.own_timing {
            t.gap = self.pass_timing_gap;
            t.keep_frames(self.keep_timed_frames);
            t.set_detail(self.pass_detail);
            t.begin(&self.device, &self.queue);
        }
        if self.animated_textures {
            self.hud.animate(
                &mut self.uploads,
                &mut encoder,
                frame.ticks,
                &self.animations,
            );
        }
        self.update_lightmap(
            &mut encoder,
            &frame.sky,
            frame.darkness.1,
            (self.gamma - frame.darkness.0).max(0.0),
            frame.night_vision,
            frame.block_flicker,
            frame.boss_darken,
        );
        let views = self.views(frame);
        let pose = frame.bob * frame.spin;
        let view_proj = views.primary().view_proj(pose);
        laps.lap("r_acquire");
        // TAA runs where the world post pass does, not under a shader pack's own.
        let taa = !frame.title
            && self.pack_post.as_ref().is_none_or(|p| p.1.is_none())
            && self.executor.is_none();
        let hdr_lighting = self.shader_pack_kind().is_none();
        if let Some(hdr) = &mut self.hdr {
            hdr.lighting = hdr_lighting && hdr.look;
        }
        let hdr_gain = no_pack_hdr_gain(
            self.hdr.as_ref().is_some_and(|h| h.layered()),
            self.shader_pack_kind().is_some(),
            !frame.title,
        );
        let hdr_settings = self
            .hdr
            .as_ref()
            .map_or([0.0; 4], |h| h.draw_params(!frame.title));
        self.world_post.baked_light(hdr_gain, hdr_settings);
        let jitter = self.world_post.jitter(
            [self.config.width, self.config.height],
            taa,
            view_proj,
            frame.camera,
        );
        let jittered = jitter * view_proj;
        if frame.title {
            self.panorama.prepare(
                &self.device,
                &self.queue,
                self.config.width,
                self.config.height,
                frame.yaw,
                frame.pitch,
            );
        } else {
            self.panorama.release();
        }
        let view = views.primary().view;
        // The hand's bob and view (`hand_globals`).
        let hand_view = frame.bob * view;
        let sun_angle = frame.sky.sun_angle.to_radians();
        // Celestial bodies rotate around the X axis; angle 0 = noon.
        let sun = Vec3::new(0.0, sun_angle.cos(), -sun_angle.sin());
        let (fog_color, fog) = fog_uniform(frame);
        let globals = Globals {
            view_proj: jittered.to_cols_array_2d(),
            fog_color,
            fog,
            sky_color: [
                frame.sky.sky_color[0],
                frame.sky.sky_color[1],
                frame.sky.sky_color[2],
                1.0,
            ],
            misc: [
                1.0 - frame.sky.sky_light_factor,
                frame.game_time,
                views.primary().aspect,
                views.primary().tan_half_fov,
            ],
            // The sky turns with the view bob and the nausea spin, as the world,
            // the sun and the clouds do. `GameRenderer.renderLevel` puts both into
            // the projection the sky is drawn with; without them the sky stands
            // still while everything else sways.
            inverse_view: views
                .primary()
                .posed_view(pose)
                .inverse()
                .to_cols_array_2d(),
            sunrise: frame.sky.sunrise_color,
            sun: [sun.x, sun.y, sun.z, frame.sky.star_brightness],
            fog_ends: [
                frame.fog.map_or(frame.distance.min(512.0), |f| f.sky_end),
                frame.fog.map_or(2048.0, |f| f.cloud_end),
                if frame.dark_disc { 1.0 } else { 0.0 },
                frame.glint_alpha,
            ],
            end_portal_sprites: frame.end_portal_sprites,
            crumbling_sprite: frame.crumbling_sprite,
            time: [
                frame.shader_game_time,
                f32::from(frame.nether_lighting),
                frame.dynamic_lights.len().min(crate::dynamic_light::MAX) as f32,
                frame.glint_ticks,
            ],
            outline_params: outline_params(self.config.width, self.config.height),
            dynamic_lights: {
                let mut lights = [[0.0; 4]; crate::dynamic_light::MAX];
                for (to, from) in lights.iter_mut().zip(frame.dynamic_lights) {
                    *to = *from;
                }
                lights
            },
            colored_lights: {
                let mut lights = [[[0.0; 4]; 2]; crate::dynamic_light::MAX];
                for (to, from) in lights.iter_mut().zip(frame.colored_lights) {
                    *to = *from;
                }
                lights
            },
            colored_count: [
                frame.colored_lights.len().min(crate::dynamic_light::MAX) as u32,
                // PvP & HUD's hit colour: entity.wgsl's hurt texel, 0 for vanilla's.
                frame.pvp.hurt_texel,
                // PvP & HUD's glint colour: entity.wgsl's `tinted_glint`, 0 for vanilla's.
                frame.pvp.glint,
                0,
            ],
            emissive_layers: {
                let mut layers = [[0; 4]; crate::native_emissive::MAX];
                for (to, from) in layers.iter_mut().zip(frame.emissive_layers) {
                    *to = *from;
                }
                layers
            },
            emissive_count: [
                frame.emissive_layers.len().min(crate::native_emissive::MAX) as u32,
                0,
                0,
                0,
            ],
            sunrise_fan: sunrise_fan::normals(frame.sky.sunrise_color[3], sun),
            filtering: [
                self.texture_filtering as f32,
                self.sprite_textures[0].width() as f32,
                // The views' last level (`rebuild_terrain_group`), when a sprite is one texel
                // there: RGSS takes one tap from it on (terrain.wgsl's `rotated_grid`).
                if self.sprite_textures[0].width() >> self.mipmaps == 1 {
                    self.mipmaps as f32
                } else {
                    0.0
                },
                0.0,
            ],
            hdr_light: [
                0.0,
                0.0,
                // HDR output is on.
                if self.hdr.is_some() { 1.0 } else { 0.0 },
                f32::from(u8::from(hdr_gain)),
            ],
            hdr_settings,
            view_origin: [0.0; 4],
            world_view: [0.0; 4],
        };
        self.uploads
            .buffer(&mut encoder, &self.globals, 0, bytemuck::bytes_of(&globals));
        // The sunrise fan as sixteen triangles. Its clip positions use the sky's
        // own view (no TAA jitter; the bob and the spin, as `fs_sky`'s `view_ray`),
        // and its apex weight interpolates the barycentric the shader used to
        // compute for every sky pixel.
        if frame.sky.sunrise_color[3] > 0.001 && !frame.title && frame.skybox {
            let sky_view_proj = views.primary().view_proj(pose);
            let mut mesh = [[0.0f32; 5]; sunrise_fan::VERTICES];
            for (to, from) in mesh
                .iter_mut()
                .zip(sunrise_fan::mesh(frame.sky.sunrise_color[3], sun))
            {
                let clip = sky_view_proj * glam::Vec4::new(from[0], from[1], from[2], 1.0);
                *to = [clip.x, clip.y, clip.z, clip.w, from[3]];
            }
            self.uploads.buffer(
                &mut encoder,
                &self.sunrise_buffer,
                0,
                bytemuck::cast_slice(&mesh),
            );
        }
        laps.lap("r_targets");
        // GameRenderer.render3dHud: the hand's projection at hudFov, with the
        // bob in its pose rather than the projection (same product).
        let hand_projection = Mat4::from_cols_array(&crate::camera::hand_projection(
            frame.hand_fov_degrees,
            self.config.width as f32,
            self.config.height.max(1) as f32,
            frame.depth_far,
        ));
        let hand_view_proj = hand_projection * frame.bob * view;
        let hand_globals = Globals {
            view_proj: hand_view_proj.to_cols_array_2d(),
            ..globals
        };
        self.uploads.buffer(
            &mut encoder,
            &self.hand_globals,
            0,
            bytemuck::bytes_of(&hand_globals),
        );
        // GameRenderer.render3dHud: DebugCrosshairRenderer draws with the
        // hand's projection but no bob, and FogMode.NONE (no fog colour).
        let crosshair_globals = Globals {
            view_proj: (hand_projection * view).to_cols_array_2d(),
            fog_color: [0.0; 4],
            ..globals
        };
        self.uploads.buffer(
            &mut encoder,
            &self.crosshair_globals,
            0,
            bytemuck::bytes_of(&crosshair_globals),
        );
        // Frustum culling and camera-relative origins.
        let cull_proj = views.cull().view_proj(pose);
        let view = Frustum {
            planes: frustum(cull_proj),
        };
        let camera = frame.camera;
        let camera_section = camera_section(camera, self.layout.min_y());
        // The water mirror draws the same lists: they take in what its camera sees too
        // (above the screen's top, the far shore, the trees against the sky).
        let mirror_plane = frame
            .water_plane
            .filter(|_| !frame.title && self.wants_water_plane());
        let mirror = mirror_plane.map(|plane| Frustum {
            planes: frustum(view::mirror_view_projection(
                cull_proj,
                plane - frame.camera.y as f32,
            )),
        });
        // Solid quads facing away from the eye are left out: the bob moves
        // the eye off `camera`, so it is found from the projection. With the mirror, those
        // facing its eye are kept too: it sits straight below, at this camera-relative y.
        let eye = view::projection_center(cull_proj).filter(|_| self.sides_on);
        let mirror_low = mirror_plane
            .zip(eye)
            .map(|(plane, eye)| 2.0 * (plane - frame.camera.y as f32) - eye.y);
        let culling = !frame.title && frame.cave_culling;
        if culling {
            crate::spans::time("occlusion", || {
                self.start_occlusion(camera, frame.detail_distance)
            });
        }
        // A shader pack's shadow pass draws the sections in its camera's box, all of them (the
        // occlusion graph follows the eye, not the light), within shadowDistanceRenderMul; for
        // an Iris-format pack, only those whose shadow can fall into the camera's view.
        let state = external::FrameState {
            frame,
            output: &output,
        };
        if let Some(pipeline) = self.external.as_mut().filter(|_| !frame.title) {
            pipeline.begin_frame(&state);
        }
        let shadow_view = self
            .pack_post
            .as_ref()
            .and_then(|p| p.1.as_ref())
            .filter(|_| !frame.title)
            .and_then(|p| p.shadow_view(frame))
            .or_else(|| {
                // One of Horizon's own packs with `[shadow]` uses the same lists.
                (self.native.as_ref().filter(|_| !frame.title)?).shadow_request(frame)
            })
            .or_else(|| {
                self.external
                    .as_ref()
                    .filter(|_| !frame.title)?
                    .shadow_request(&state)
            });
        let pack_terrain = !frame.title
            && self
                .pack_post
                .as_ref()
                .and_then(|p| p.1.as_ref())
                .is_some_and(|p| p.has_terrain());
        // `LevelRenderer.extractSectionDrawGroups`' fade-in time, for the instances.
        let fade_ms = fade_millis(frame.chunk_fade);
        // GPU culling's fading sections, decided at `fades.now`: before the gate, the record
        // check and `prepare`, which sends their flags, and no later than the instances' `now`,
        // so a section the pass draws whole is whole there too. Title frames draw no terrain
        // (a loading screen's fade-in time is 0): they leave them as they are.
        let fades = FadeInputs {
            now: Instant::now(),
            fade_ms,
            uploaded: &self.section_uploaded,
            retiring: &self.lod_retiring,
        };
        if !frame.title
            && let Some(gpu) = &mut self.gpu_culling
        {
            let (sections, rows) = (&mut self.sections, self.layout.sections());
            mark_fades(sections, &mut gpu.records, &mut gpu.fades, rows, fades);
        }
        // The near solid sections culled and drawn on the GPU (`gpu_culling`): never where a
        // shader pack's terrain, a shadow pass or the water mirror draws the CPU's lists, nor
        // under a native pack's `[displacement]`, whose grid takes its quads from the walk's
        // `visible`. Decided before the walk, so that a pass whose buffers no longer fit can
        // still switch GPU culling off.
        let radius = (frame.detail_distance / 16.0) as i32;
        let displaced = self.executor.is_some_and(|c| c.displacement);
        let mut gpu_frame = self.gpu_culling.is_some()
            && !frame.title
            && !pack_terrain
            && shadow_view.is_none()
            && mirror_plane.is_none()
            && !displaced;
        #[cfg(debug_assertions)]
        if gpu_frame {
            if self.gpu_checks.is_multiple_of(64) {
                self.check_gpu_records(fades);
            }
            self.gpu_checks = self.gpu_checks.wrapping_add(1);
        }
        self.frame_serial += 1;
        let serial = self.frame_serial;
        // GPU frames' statistics come back a frame or a few later (`GpuCulling::collect`).
        if let Some(gpu) = &mut self.gpu_culling
            && let Some((frame, stats)) = gpu.collect(&self.device)
        {
            self.show_stats(frame, stats);
        }
        // The walk reads the graph GPU culling is given (checked after it).
        #[cfg(debug_assertions)]
        let graph_given = self.occlusion.generation();
        if gpu_frame && let Some(gpu) = &mut self.gpu_culling {
            let input = gpu_culling::FrameInputs {
                camera,
                min_y: self.sections.min_y(),
                sections: self.layout.sections() as i32,
                radius,
                frustum: &view,
                reach: self.occlusion.reach().filter(|_| culling),
                eye,
                view_proj: jittered,
                occlude: self.cull_on,
            };
            let prepared = crate::spans::time("gpu culling", || {
                gpu.prepare(
                    &self.device,
                    &self.queue,
                    &mut self.uploads,
                    &mut encoder,
                    &input,
                    self.own_timing.as_mut(),
                )
            });
            // A frame whose planes leave the test's integers (the camera millions of blocks
            // above or below the level), or whose eye is not a number, is the CPU's.
            gpu_frame = prepared == gpu_culling::Prepared::Gpu;
            if prepared == gpu_culling::Prepared::OutOfMemory {
                log::warn!(
                    "Horizon renderer: GPU culling off until the next start: its buffers do not fit in video memory"
                );
                self.offer_gpu_culling(crate::graphics::GpuCullingOffer::OutOfMemory);
                let _ = self.set_gpu_culling(false);
            }
        }
        self.gpu_frame = gpu_frame;
        let reachable = &self.occlusion;
        let cutout = &self.lod_cutout;
        // Sections fading out (`leave_section`) are past the render distance:
        // drawn there until they have, or they popped out at once.
        self.lod_leaving_scratch.clear();
        self.lod_leaving_scratch
            .extend(self.lod_leaving.iter().map(|&((x, _, z), _)| (x, z)));
        let leaving = &self.lod_leaving_scratch;
        let mut visible = std::mem::take(&mut self.visible);
        visible.clear();
        crate::spans::time("visibility", || {
            // GPU culling draws the solid quads of the near whole sections it has records of:
            // within the render distance the walk lists only those with translucent quads,
            // without a record or fading (`SectionGpu::listed`).
            self.sections.walk(
                camera,
                radius,
                &Frusta(view, mirror),
                |k| !culling || reachable.contains(k),
                |x, z| {
                    (!cutout.is_empty() && cutout.contains(&(x, z)))
                        || (!leaving.is_empty() && leaving.contains(&(x, z)))
                },
                gpu_frame,
                &mut visible,
            )
        });
        #[cfg(debug_assertions)]
        assert!(
            !gpu_frame || self.occlusion.generation() == graph_given,
            "the walk reads the graph GPU culling was given"
        );
        let now = std::time::Instant::now();
        let mut visible_lods = std::mem::take(&mut self.visible_lods);
        visible_lods.clear();
        if frame.title {
            visible.clear();
        } else {
            visible_lods.extend(
                self.lods
                    .iter()
                    .filter_map(|(&key, tile)| tile.visible(key, camera, &view, now)),
            );
            visible_lods.sort_unstable_by_key(|v| (v.distance, v.key));
        }
        let mut shadow_visible = std::mem::take(&mut self.shadow_visible);
        shadow_visible.clear();
        if let Some(v) = &shadow_view {
            let radius = (frame.detail_distance / 16.0) as i32;
            let light = ShadowCull {
                light: shadow_box(v),
                casters: v.view_casters.then(|| shadow_casters(cull_proj, v.light)),
                reach: v.distance,
            };
            let mut boxed = std::mem::take(&mut self.shadow_boxed_scratch);
            let mut seen = std::mem::take(&mut self.shadow_seen_scratch);
            crate::spans::time("shadow visibility", || {
                // Within the unculled distance every section, then those in the light's box.
                if let Some(d) = v.unculled {
                    let all = ShadowCull {
                        light: Frustum {
                            planes: [glam::Vec4::W; 6],
                        },
                        casters: None,
                        reach: v.distance,
                    };
                    let near = radius.min((d / 16.0).ceil() as i32);
                    self.sections.visible(
                        camera,
                        near,
                        &all,
                        |_| true,
                        |_, _| false,
                        &mut shadow_visible,
                    );
                }
                boxed.clear();
                self.sections
                    .visible(camera, radius, &light, |_| true, |_, _| false, &mut boxed);
                seen.clear();
                seen.extend(shadow_visible.iter().map(|s| s.key));
                shadow_visible.extend(boxed.drain(..).filter(|s| !seen.contains(&s.key)));
            });
            self.shadow_boxed_scratch = boxed;
            self.shadow_seen_scratch = seen;
            // Nearest the light first, so the depth test rejects what lies under it.
            let light = v.light;
            shadow_visible.sort_unstable_by(|a, b| {
                let height = |s: &resident::Visible| glam::Vec3::from(s.origin).dot(light);
                height(b).total_cmp(&height(a))
            });
            for v in &shadow_visible {
                self.sort_unseen(&mut encoder, v.slot, v.key, camera);
            }
        }
        crate::spans::time("world view visibility", || {
            self.walk_world_views(&mut encoder, frame, radius)
        });
        // A `shaders/` pack that draws the terrain, or its shadow, sees it as Iris does.
        let shaders_pack = !frame.title
            && self
                .pack_post
                .as_ref()
                .and_then(|p| p.1.as_ref())
                .is_some_and(|p| p.has_terrain() || p.has_shadow());
        if shaders_pack {
            let from = pack_terrain_camera(camera);
            let lists = self.world_views.sections.sections_mut();
            for v in visible
                .iter_mut()
                .chain(shadow_visible.iter_mut())
                .chain(lists)
            {
                v.origin = resident::origin(v.key, self.sections.min_y(), from)
                    .as_vec3()
                    .to_array();
            }
        }
        if self.animated_textures {
            self.active_animated_layers.fill(0);
            let lists = self.world_views.sections.sections();
            for v in visible.iter().chain(&shadow_visible).chain(lists) {
                let s = self.sections.get(v.slot);
                for &layer in &*s.animated_layers {
                    let layer = layer as usize;
                    self.active_animated_layers[layer / u64::BITS as usize] |=
                        1_u64 << (layer % u64::BITS as usize);
                }
            }
            for v in &visible_lods {
                if let Some(tile) = self.lods.get(&v.key) {
                    for &layer in &*tile.drawn().animated_layers {
                        let layer = layer as usize;
                        self.active_animated_layers[layer / u64::BITS as usize] |=
                            1_u64 << (layer % u64::BITS as usize);
                    }
                    for &layer in &*tile.mesh.animated_layers {
                        let layer = layer as usize;
                        self.active_animated_layers[layer / u64::BITS as usize] |=
                            1_u64 << (layer % u64::BITS as usize);
                    }
                }
            }
            // Visibility is collected before terrain draw commands are recorded,
            // so upload the current mask here and keep animation updates ahead
            // of every draw in this command buffer. A GPU frame's walk leaves out
            // the sections GPU culling draws: every layer animates.
            self.animate(&mut encoder, frame.ticks, !frame.title && !gpu_frame);
        }
        let listed = visible.len() + visible_lods.len();
        let viewed = listed + shadow_visible.len();
        let instances = viewed + self.world_views.sections.sections().count();
        if instances as u64 > self.instance_capacity {
            self.instance_capacity = (instances as u64).next_power_of_two();
            self.instances = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Section origins"),
                size: self.instance_capacity * 16,
                usage: wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
        }
        self.origins.clear();
        let (sections, retiring) = (&self.sections, &self.lod_retiring);
        laps.lap("r_cameras");
        // `LevelRenderer.extractSectionDrawGroups`: each section's `getVisibility`. Once
        // nothing is fading (the usual frame), every section is whole without a lookup.
        let fading_in = self.sections_fading_in(fade_ms, now);
        let fading_over_lods = self.sections_fading_over_lods(now);
        let fade_code = move |section: &SectionGpu, key: &SectionKey| {
            if fading_over_lods {
                section.fade(retiring.contains(key))
            } else {
                0
            }
        };
        let uploaded = &self.section_uploaded;
        let visibility = move |key: SectionKey| {
            if !fading_in {
                return 1.0;
            }
            section_visibility(uploaded.get(&key), fade_ms, now)
        };
        self.origins.extend(
            visible
                .iter()
                .map(|v| {
                    let code = fade_code(sections.get(v.slot), &v.key);
                    section_instance(v.origin, code, visibility(v.key))
                })
                .chain(visible_lods.iter().map(|v| lod::instance(v.origin, v.fade)))
                .chain(
                    shadow_visible
                        .iter()
                        .map(|v| section_instance(v.origin, 0, visibility(v.key))),
                )
                .chain(self.world_views.sections.sections().map(|v| {
                    let code = fade_code(sections.get(v.slot), &v.key);
                    section_instance(v.origin, code, visibility(v.key))
                })),
        );
        if !self.origins.is_empty() {
            self.uploads.buffer(
                &mut encoder,
                &self.instances,
                0,
                bytemuck::cast_slice(&self.origins),
            );
        }
        // `[displacement]`: the visible sections' solid quads, nearest first, for its grid to
        // take those it reaches.
        if let Some(n) = &mut self.native {
            let sections = visible.iter().filter(|_| !frame.title).filter_map(|v| {
                let s = self.sections.get(v.slot);
                (s.solid > 0).then(|| displace::Near {
                    buffer: self.arena.buffer(s.page),
                    base: s.solid_base(),
                    quads: s.solid,
                    origin: v.origin,
                })
            });
            n.prepare_displacement(native_material::DisplacementPrepare {
                device: &self.device,
                queue: &self.queue,
                encoder: &mut encoder,
                sections,
                view_proj,
            });
        }
        // Solid sections front to back, then translucent ones back to front,
        // as one list of indirect draws over the section arena.
        self.draws.clear();
        self.draw_pages.clear();
        // Occlusion culling leaves the sections its last result did not see
        // to phase 2, after the others, with instance counts of 0. Whether
        // the frame has draws enough goes by the last one's.
        // A shader pack drawing terrain draws all of it: occlusion culling reads the depth of
        // Horizon's own terrain draw.
        let mut cull = self
            .cull
            .as_mut()
            .filter(|c| {
                self.cull_on
                    && !frame.title
                    && !pack_terrain
                    && self.multi_draw
                    // Occlusion is found from the camera, not the mirror's eye.
                    && mirror.is_none()
                    // It tests the CPU's lists, which leave out GPU culling's sections.
                    && !gpu_frame
                    && visible.len() >= c.min_sections
            })
            .and_then(|cull| cull.begin(&self.device).then_some(cull));
        self.phase2_draws.clear();
        self.phase2_pages.clear();
        self.solid_runs.clear();
        self.phase2_runs.clear();
        // On Metal, phase 1 goes the opaque quads of every list, then the cutout
        // ones (`Part`), each solid pass drawing its own: `[..opaque_draws]`,
        // then the rest; phase 2 likewise. A fading list's quads all draw in the
        // cutout pass.
        let split_solid = self.info.backend == wgpu::Backend::Metal;
        let parts = |fading: bool| -> &'static [(u8, Quads)] {
            match (split_solid, fading) {
                (false, _) => &[(0, Quads::All)],
                (true, false) => &[(0, Quads::Opaque), (1, Quads::Cutout)],
                (true, true) => &[(1, Quads::All)],
            }
        };
        let (mut opaque_draws, mut phase2_opaque) = (0, 0);
        debug_assert!(
            !(split_solid && gpu_frame),
            "GPU culling is never offered on Metal"
        );
        // In a GPU frame the GPU draws the solid quads of the whole sections within the
        // render distance that have records (`SectionGpu::gpu_draws_solid`), and the walk
        // listed those among them with translucent quads. Here stay those fading in or out
        // (`gpu_fading`), those without a record and those past the distance (far terrain's
        // cutout, and columns leaving).
        let from = resident::camera_column(camera);
        // Listed sections whose solid quads the pass draws: it counts them among the
        // sections it kept.
        let mut gpu_solid = 0;
        for (i, v) in visible.iter().enumerate() {
            let s = self.sections.get(v.slot);
            if gpu_frame
                && s.gpu_draws_solid(crate::streaming::in_view_distance(
                    from, radius, v.key.0, v.key.2,
                ))
            {
                gpu_solid += 1;
                continue;
            }
            let first_phase = cull.as_ref().is_none_or(|c| c.first(v.slot));
            let (draws, pages) = if first_phase {
                (&mut self.draws, &mut self.draw_pages)
            } else {
                (&mut self.phase2_draws, &mut self.phase2_pages)
            };
            let instances = u32::from(first_phase);
            let mut indices = 0;
            let parts = parts(fade_code(s, &v.key) != 0);
            // Each part's (group, first draw, draws).
            let mut runs = [(0, 0, 0); 2];
            for (run, &(group, part)) in runs.iter_mut().zip(parts) {
                let start = draws.len();
                facing_runs_from(&s.sides, eye, mirror_low, v.origin, part, |first, quads| {
                    indices += quads * 6;
                    let base = s.solid_base() + first * 4;
                    draws.push([quads * 6, instances, 0, base, i as u32]);
                    pages.push((s.page, None));
                });
                *run = (group, start, draws.len() - start);
            }
            let runs = &runs[..parts.len()];
            if first_phase {
                for &(group, start, len) in runs {
                    if group == 0 {
                        opaque_draws += len;
                    }
                    self.solid_runs
                        .push_in(group, v.distance, s.page, start, len, [0; 3]);
                }
                if let Some(c) = &mut cull {
                    c.drawn(
                        i,
                        v.slot,
                        indices,
                        cull::section_box(self.sections.bounds(v.slot)),
                    );
                }
            } else if let Some(c) = &mut cull {
                // One test sees the section's draws in both passes.
                let total: usize = runs.iter().map(|r| r.2).sum();
                if let Some(test) = c.deferred(
                    i,
                    v.slot,
                    total,
                    indices,
                    cull::section_box(self.sections.bounds(v.slot)),
                ) {
                    for &(group, start, len) in runs {
                        if group == 0 {
                            phase2_opaque += len;
                        }
                        let tag = [test, u32::from(group), 0];
                        self.phase2_runs
                            .push_in(group, v.distance, s.page, start, len, tag);
                    }
                }
            }
        }
        // Distant terrain tiles are occlusion tested as sections are, with their meshes'
        // bounds, but for those fading or with a stand-in (`lod::Tile::cull_bounds`).
        for (i, v) in visible_lods.iter().enumerate() {
            let tile = &self.lods[&v.key];
            let s = tile.drawn();
            let index = visible.len() + i;
            let tested = cull.as_ref().and(tile.cull_bounds(v.fade));
            let slot = cull::TILE | tile.cull_slot;
            let first_phase = tested.is_none() || cull.as_ref().is_some_and(|c| c.first(slot));
            let (draws, pages) = if first_phase {
                (&mut self.draws, &mut self.draw_pages)
            } else {
                (&mut self.phase2_draws, &mut self.phase2_pages)
            };
            let instances = u32::from(first_phase);
            let mut indices = 0;
            let parts = parts(v.fade != 0);
            let mut runs = [(0, 0, 0); 2];
            for (run, &(group, part)) in runs.iter_mut().zip(parts) {
                let start = draws.len();
                facing_runs_from(&s.sides, eye, mirror_low, v.origin, part, |first, quads| {
                    indices += quads * 6;
                    let base = s.solid_base() + first * 4;
                    draws.push([quads * 6, instances, 0, base, index as u32]);
                    pages.push((s.page, None));
                });
                *run = (group, start, draws.len() - start);
            }
            let runs = &runs[..parts.len()];
            let (Some(bounds), Some(c)) = (tested, &mut cull) else {
                for &(group, start, len) in runs {
                    if group == 0 {
                        opaque_draws += len;
                    }
                    self.solid_runs
                        .push_in(group, v.distance, s.page, start, len, [0; 3]);
                }
                continue;
            };
            if first_phase {
                for &(group, start, len) in runs {
                    if group == 0 {
                        opaque_draws += len;
                    }
                    self.solid_runs
                        .push_in(group, v.distance, s.page, start, len, [0; 3]);
                }
                c.drawn(index, slot, indices, bounds);
            } else {
                let total: usize = runs.iter().map(|r| r.2).sum();
                if let Some(test) = c.deferred(index, slot, total, indices, bounds) {
                    for &(group, start, len) in runs {
                        if group == 0 {
                            phase2_opaque += len;
                        }
                        let tag = [test, u32::from(group), 0];
                        self.phase2_runs
                            .push_in(group, v.distance, s.page, start, len, tag);
                    }
                }
            }
        }
        // Phase 2's draws are tested where grouping left them.
        let runs = self
            .phase2_runs
            .regroup(&mut self.phase2_draws, &mut self.phase2_pages, 0);
        if let Some(c) = &mut cull {
            for r in runs {
                c.place(r.tag[0], r.tag[1], r.start, r.len);
            }
        }
        self.solid_runs
            .regroup(&mut self.draws, &mut self.draw_pages, 0);
        let phase1_draws = self.draws.len() as u32;
        // Draws `[..opaque_end]` hold opaque quads, `[cutout_start..phase1_draws]` cutout ones.
        let (opaque_end, cutout_start) = if split_solid {
            (opaque_draws, opaque_draws)
        } else {
            (phase1_draws as usize, 0)
        };
        self.draws.extend_from_slice(&self.phase2_draws);
        self.draw_pages.extend_from_slice(&self.phase2_pages);
        let culling = cull.is_some();
        let solid_draws = self.draws.len() as u32;
        // Phase 2's draws `[phase1_draws..phase2_cutout]` hold opaque quads, the rest
        // cutout ones (without the split, all of them go through one pipeline).
        let phase2_cutout = phase1_draws + phase2_opaque as u32;
        laps.lap("r_section_groups");
        // LevelRenderer: nearby sections each frame, plus a rotating eighth
        // (at least 15) of the visible list. Cached views avoid needless sorts.
        self.translucent_draws.clear();
        let count = visible.len() + visible_lods.len();
        let budget = (count / 8).max(15).min(count);
        self.translucency_cursor %= count.max(1);
        for (i, v) in visible.iter().enumerate().rev() {
            let s = self.sections.get_mut(v.slot);
            if let Some((index_page, index_start)) = s.sorted_indices {
                let nearby = (v.key.0 - camera_section.0).abs() <= 2
                    && (v.key.1 - camera_section.1).abs() <= 2
                    && (v.key.2 - camera_section.2).abs() <= 2;
                let scheduled = (i + count - self.translucency_cursor) % count < budget;
                let origin = [
                    v.key.0 * 16,
                    self.layout.min_y() + v.key.1 * 16,
                    v.key.2 * 16,
                ];
                if (!s.sort.is_sorted() || nearby || scheduled)
                    && let Some(indices) =
                        s.sort
                            .update(camera.to_array(), origin, &mut self.translucent_sorter)
                {
                    self.uploads.buffer(
                        &mut encoder,
                        self.sorted_indices.buffer(index_page),
                        index_start * 4,
                        bytemuck::cast_slice(indices),
                    );
                }
                let (page, first) = s.translucent_at;
                self.translucent_draws.push((
                    v.distance,
                    i,
                    [
                        s.translucent * 6,
                        1,
                        index_start as u32,
                        first as u32,
                        i as u32,
                    ],
                    (page, Some(index_page)),
                ));
            }
        }
        // Sections went back to front (`visible` is sorted by distance, then
        // key): tiles' water only needs sorting in when there is some.
        let section_translucent = self.translucent_draws.len();
        for (i, v) in visible_lods.iter().enumerate() {
            let instance = visible.len() + i;
            let tile = self.lods.get_mut(&v.key).expect("visible resident tile");
            let s = tile.drawn_mut();
            if let Some((index_page, index_start)) = s.sorted_indices {
                let scheduled = (instance + count - self.translucency_cursor) % count < budget;
                if (!s.sort.is_sorted() || scheduled)
                    && let Some(indices) = s.sort.update_lod(
                        camera.to_array(),
                        v.key.origin(),
                        &mut self.translucent_sorter,
                    )
                {
                    self.uploads.buffer(
                        &mut encoder,
                        self.sorted_indices.buffer(index_page),
                        index_start * 4,
                        bytemuck::cast_slice(indices),
                    );
                }
                let (page, first) = s.translucent_at;
                self.translucent_draws.push((
                    v.distance,
                    instance,
                    [
                        s.translucent * 6,
                        1,
                        index_start as u32,
                        first as u32,
                        instance as u32,
                    ],
                    (page, Some(index_page)),
                ));
            }
        }
        // Near water and distant water share one back-to-front draw order.
        if self.translucent_draws.len() > section_translucent {
            // Distant water goes after the near chunks', and replaced tiles' after all: each is a
            // hair behind the one before, so depth leaves it only where that one has none. Back to
            // front together, a far tile's water under a near chunk's (one fading in, or brought
            // into view by a longer render distance) blended twice and looked opaque. The cost: far
            // water just behind near water shows no surface of its own there.
            let tier = |instance: usize| match instance.checked_sub(visible.len()) {
                None => 0,
                Some(tile) if visible_lods[tile].old => 2,
                Some(_) => 1,
            };
            self.translucent_draws
                .sort_unstable_by_key(|&(distance, instance, ..)| {
                    (
                        tier(instance),
                        std::cmp::Reverse(distance),
                        std::cmp::Reverse(instance),
                    )
                });
        }
        for &(_, _, draw, pages) in &self.translucent_draws {
            self.draws.push(draw);
            self.draw_pages.push(pages);
        }
        self.translucency_cursor += budget;
        let translucent_draws = self.draws.len() as u32 - solid_draws;
        let section_draws = self.draws.len();
        // The shadow camera's sections after the camera's: quads that can face the light (all of
        // them for `every_face`), then translucent ones (in their last sorted order; depth
        // decides there).
        let shadow_first = self.draws.len() as u32;
        let mut shadow_translucent = std::mem::take(&mut self.shadow_translucent_scratch);
        shadow_translucent.clear();
        self.shadow_runs.clear();
        if let Some(view) = &shadow_view {
            let base = listed;
            // How far below the nearest section to the light (squared, as bits, as `PageRuns`
            // takes distances): each band's draws go by page, a call per page and band.
            let height = |v: &resident::Visible| glam::Vec3::from(v.origin).dot(view.light);
            let top = shadow_visible.first().map_or(0.0, height);
            for (k, v) in shadow_visible.iter().enumerate() {
                let s = self.sections.get(v.slot);
                let instance = (base + k) as u32;
                let depth = f64::from(top - height(v)).max(0.0);
                let distance = (depth * depth).to_bits();
                let before = self.draws.len();
                shadow_runs(&s.sides, view, v.origin, |first, quads| {
                    let start = s.solid_base() + first * 4;
                    self.draws.push([quads * 6, 1, 0, start, instance]);
                    self.draw_pages.push((s.page, None));
                });
                let len = self.draws.len() - before;
                self.shadow_runs.push(distance, s.page, before, len, [0; 3]);
                if let Some((index_page, index_start)) = s.sorted_indices {
                    let (page, start) = s.translucent_at;
                    shadow_translucent.push((
                        distance,
                        [
                            s.translucent * 6,
                            1,
                            index_start as u32,
                            start as u32,
                            instance,
                        ],
                        (page, Some(index_page)),
                    ));
                }
            }
        }
        let shadow_solid = self.draws.len() as u32 - shadow_first;
        self.shadow_runs
            .regroup(&mut self.draws, &mut self.draw_pages, shadow_first as usize);
        self.shadow_runs.clear();
        for (distance, draw, pages) in shadow_translucent.drain(..) {
            self.shadow_runs
                .push(distance, pages.0, self.draws.len(), 1, [0; 3]);
            self.draws.push(draw);
            self.draw_pages.push(pages);
        }
        self.shadow_translucent_scratch = shadow_translucent;
        self.shadow_runs.regroup(
            &mut self.draws,
            &mut self.draw_pages,
            (shadow_first + shadow_solid) as usize,
        );
        let shadow_translucent = self.draws.len() as u32 - shadow_first - shadow_solid;
        // Each world view's draws after them (`view_sections`).
        let (lists, draws) = (&mut self.world_views.sections, &mut self.draws);
        lists.push_draws(&self.sections, draws, &mut self.draw_pages, viewed as u32);
        laps.lap("r_nearby");
        // Metal replays a multi-draw as one draw call per entry.
        let mut section_draw_calls = if self.multi_draw && self.info.backend != wgpu::Backend::Metal
        {
            (0..section_draws)
                .filter(|&i| {
                    i == 0
                        || i == phase1_draws as usize
                        || i == solid_draws as usize
                        || self.draw_pages[i] != self.draw_pages[i - 1]
                })
                .count()
        } else {
            section_draws
        };
        // GPU culling's draw calls, one per page; its draws come back with its statistics
        // (`GpuCulling::copy_stats`).
        if gpu_frame && let Some(gpu) = &self.gpu_culling {
            section_draw_calls += gpu.draw_calls();
        }
        let debug_depth = frame.debug_lines.len() as u32;
        let debug_opaque = frame
            .debug_lines
            .iter()
            .filter(|l| debug_line_opaque(l))
            .count() as u32;
        let debug_top = frame.debug_lines_on_top.len() as u32;
        let crosshair = frame.debug_crosshair.len() as u32;
        if debug_depth + debug_top + crosshair > 0 {
            let needed = u64::from(debug_depth + debug_top + crosshair);
            if needed > self.debug_line_capacity {
                self.debug_line_capacity = needed.next_power_of_two();
                self.debug_line_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Debug lines"),
                    size: 32 * self.debug_line_capacity,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            let lines = &mut self.debug_lines_scratch;
            lines.clear();
            lines.extend(
                debug_lines_opaque_first(frame.debug_lines)
                    .chain(frame.debug_lines_on_top)
                    .chain(frame.debug_crosshair)
                    .copied(),
            );
            self.uploads.buffer(
                &mut encoder,
                &self.debug_line_buffer,
                0,
                bytemuck::cast_slice(lines),
            );
        }
        self.weather.prepare(
            &self.device,
            &mut encoder,
            &mut self.uploads,
            frame.weather,
            frame.weather_time,
        );
        // Native mods' world primitives, drawn as the world is: its jittered projection, the
        // line widths in the frame's pixels as the gizmo lines'.
        let world_primitives = !frame.title && {
            let (uploads, encoder) = (&mut self.uploads, &mut encoder);
            self.world_primitives.prepare(
                &self.device,
                &mut |buffer, bytes| uploads.buffer(encoder, buffer, 0, bytes),
                &frame.world_primitives,
                jittered,
                [self.config.width, self.config.height],
                &self.entity_atlas,
            )
        };
        let pack = self.pack_blocks(frame, fog, globals.fog_ends);
        self.write_core_text(&pack, views.primary(), pose, jitter, fog_color);
        let outline_edges = frame.outline.len().min(MAX_OUTLINE_EDGES as usize);
        if outline_edges > 0 {
            // The same per-line colour and pixel width layout as the gizmo lines. A high
            // contrast outline is two strokes: the black one first, the bright one after it.
            outline_vertices_into(
                &frame.outline[..outline_edges],
                camera,
                &mut self.outline_edges_scratch,
            );
            let width = outline_width(self.config.width);
            let edges = &self.outline_edges_scratch;
            let lines = &mut self.outline_lines_scratch;
            lines.clear();
            let stroke = |color: u32, width: f32, lines: &mut Vec<[f32; 8]>| {
                for e in edges {
                    let mut line = [0.0; 8];
                    line[..6].copy_from_slice(e);
                    line[6] = f32::from_bits(color);
                    line[7] = width;
                    lines.push(line);
                }
            };
            // A custom outline (PvP & HUD) takes the bright stroke's colour, scales both widths.
            let (custom, scale) = frame.pvp.outline.map_or((None, 1.0), |(c, s)| (Some(c), s));
            if frame.outline_high_contrast {
                stroke(0xff00_0000, 7.0 * scale, lines);
                stroke(custom.unwrap_or(0xff57_ffe1), width * scale, lines);
            } else {
                stroke(custom.unwrap_or(0x6600_0000), width * scale, lines);
            }
            self.uploads
                .buffer(&mut encoder, &self.outline, 0, bytemuck::cast_slice(lines));
            if let Some(core) = &self.core_lines {
                let data = core_lines::uniforms(
                    pack.globals,
                    views.primary(),
                    pose,
                    jitter,
                    globals.outline_params[3],
                    fog_color,
                    pack.fog,
                );
                self.uploads.buffer(&mut encoder, core.uniform(), 0, &data);
            }
        }
        // A pack's core/sky and core/stars: the disc with the sky colour, the stars with their
        // brightness, both turned with the camera only (SkyRenderer's model view).
        if let Some(core) = &self.core_sky
            && !frame.title
        {
            let (block, fog) = (pack.globals, pack.fog);
            let view = views.primary();
            // Far enough for the disc's rim, 512 out: the sky draws without a depth test.
            let fov = (2.0 * view.tan_half_fov.atan()).to_degrees();
            let projection = jitter * world_projection(fov, view.aspect, 2048.0) * pose;
            let [r, g, b, ..] = frame.sky.sky_color;
            let sky =
                core_sky::uniforms(block, projection, view.view, [r, g, b, 1.0], fog_color, fog);
            let stars = frame.stars.map(|(rotation, brightness)| {
                let b = brightness.clamp(0.0, 1.0);
                core_sky::uniforms(
                    block,
                    projection,
                    view.view * rotation,
                    [b; 4],
                    fog_color,
                    fog,
                )
            });
            core.prepare(&mut self.uploads, &mut encoder, &sky, stars.as_deref());
        }
        let skin = frame.skin.filter(|skin| match &mut self.entity_skin {
            Some(skinner) => skinner.register(&self.device, &mut self.uploads, &mut encoder, skin),
            None => false,
        });
        let pack_skin = !frame.title
            && skin.is_some()
            && self
                .pack_post
                .as_ref()
                .and_then(|(_, post)| post.as_ref())
                .is_some_and(|post| post.skins_entities());
        // While a shader pack draws entities, the runs of `entity_spans` its programs draw are
        // left out of Horizon's own entity draw.
        let mut entities = frame.entities;
        let mut pack_entities = false;
        if !frame.title
            && let Some((_, Some(post))) = &mut self.pack_post
            && post.has_entities()
        {
            pack_entities = true;
            post.prepare_entities(
                &self.device,
                &self.queue,
                frame.entities,
                frame.entity_spans,
                frame.crumbling_sprite,
                frame.end_portal_sprites[1],
                &mut self.pack_entity_rest,
                skin.filter(|_| pack_skin),
                (frame.entity_motion, frame.entity_motion_posed),
            );
            entities = &self.pack_entity_rest;
        }
        // GameRenderer.useImprovedTransparency. A shader pack keeps its own
        // path.
        let oit_on = !frame.title && self.improved_transparency_runs();
        self.cracks_transparent = oit_on && !hdr_gain;
        let [
            entity_quads,
            text_quads,
            crumbling_quads,
            translucent_crumbling_quads,
            water_mask_quads,
            swirl_quads,
            shadow_quads,
            translucent_particle_quads,
            translucent_item_quads,
            oit_entity_quads,
            fluid_quads,
            world_text_quads,
            world_text_offset_quads,
            native_solid_quads,
            native_cutout_quads,
            native_cutout_mipped_quads,
        ] = self.entity_batch.update(
            entities,
            frame.additive,
            frame.hand,
            frame.see_through,
            oit_on,
        );
        self.entity_batch.append_shadow_only(frame.shadow_only);
        // The posed models' vertices come first: the ordinary entities' draw takes them in.
        let posed = if pack_skin {
            0
        } else {
            skin.map_or(0, |s| s.vertices)
        };
        let entity_quads = entity_quads + posed / 4;
        let weather_quads = frame.weather.len() as u32 / 4;
        let world_quads = entity_quads
            + text_quads
            + crumbling_quads
            + translucent_crumbling_quads
            + water_mask_quads
            + swirl_quads
            + shadow_quads
            + translucent_particle_quads
            + translucent_item_quads
            + oit_entity_quads
            + fluid_quads
            + world_text_quads
            + world_text_offset_quads
            + native_solid_quads
            + native_cutout_quads
            + native_cutout_mipped_quads;
        laps.lap("r_draw_lists");
        // Where the tail of materials the classic and the improved passes
        // both draw ends: the shadows, particles and items count back from it.
        let base_quads = world_quads
            - oit_entity_quads
            - fluid_quads
            - world_text_quads
            - world_text_offset_quads
            - native_solid_quads
            - native_cutout_quads
            - native_cutout_mipped_quads;
        let native_clouds = self.executor.is_none_or(|c| !c.hides_clouds);
        let cloud_quads = if let Some(input) = frame.clouds.as_ref().filter(|_| native_clouds) {
            let (rebuilt, uniform) =
                crate::spans::time("clouds", || self.cloud_mesh.prepare(input));
            let vertices = self.cloud_mesh.vertices.len() as u64;
            if rebuilt && vertices > 0 {
                if vertices > self.cloud_capacity {
                    self.cloud_capacity = vertices.next_power_of_two();
                    self.cloud_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Cloud vertices"),
                        size: self.cloud_capacity * std::mem::size_of::<Vertex>() as u64,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                }
                self.uploads.buffer(
                    &mut encoder,
                    &self.cloud_buffer,
                    0,
                    bytemuck::cast_slice(&self.cloud_mesh.vertices),
                );
            }
            self.uploads.buffer(
                &mut encoder,
                &self.cloud_uniform,
                0,
                bytemuck::bytes_of(&uniform),
            );
            (vertices / 4) as u32
        } else {
            0
        };
        self.lightning.prepare(
            &self.device,
            &mut self.uploads,
            &mut encoder,
            &frame.lightning,
            camera,
        );
        // Frame generation's made frame draws its own hand, halfway, after the others.
        let made_hand_vertex = (posed as usize + self.entity_batch.combined.len()) as i32;
        if fg {
            self.frame_gen
                .halfway_hand(frame.hand, hand_view, &mut self.entity_batch.combined);
            self.frame_gen.track_entities(
                &self.device,
                &self.entity_atlas,
                frame.entities,
                frame.entity_motion,
                frame.camera,
            );
        }
        let all_entities = &self.entity_batch.combined;
        let hand_quads = (frame.hand.len() / 4) as u32;
        let additive_quads = (frame.additive.len() / 4) as u32 - self.entity_batch.moved_additive;
        let see_through_quads = (frame.see_through.len() / 4) as u32;
        let shadow_only_quads = (frame.shadow_only.len() / 4) as u32;
        // Glowing entities' silhouettes posed on the GPU go last, past every
        // material (the outline pass draws them from there).
        let outline_base = posed + all_entities.len() as u32;
        let outlined = skin.map_or(0, |s| s.outline_vertices);
        let entity_vertices = u64::from(outline_base) + u64::from(outlined);
        if entity_vertices > self.entity_capacity {
            self.entity_capacity = entity_vertices.next_power_of_two();
            self.entity_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Entity vertices"),
                size: self.entity_capacity * std::mem::size_of::<Vertex>() as u64,
                usage: self.entity_usage(),
                mapped_at_creation: false,
            });
        }
        if !all_entities.is_empty() {
            self.uploads.buffer(
                &mut encoder,
                &self.entity_buffer,
                u64::from(posed) * std::mem::size_of::<Vertex>() as u64,
                bytemuck::cast_slice(all_entities),
            );
        }
        if pack_entities {
            // The pack replaced `frame.entities`. A staged cut indexes the original stream,
            // and the game does not stage one while that pack is loaded.
            self.prepared_draws.clear();
        }
        let prepared_count = self.prepared_draws.len() as u64;
        let stride = std::mem::size_of::<PreparedModelInstance>() as u64;
        let slots = prepared_model::instance_slot_capacity(
            prepared_count,
            stride,
            self.device.limits().max_buffer_size,
        )?;
        if slots > self.prepared_instance_capacity {
            let checks = GpuChecks::push(&self.device);
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Prepared model instances"),
                size: slots * stride,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            if let Some(error) = checks.pop() {
                return Err(error.to_string().into());
            }
            self.prepared_instance_buffer = buffer;
            self.prepared_instance_capacity = slots;
        }
        if prepared_count > 0 {
            self.prepared_uploaded.clear();
            self.prepared_uploaded
                .extend(self.prepared_draws.iter().map(|draw| draw.instance));
            self.uploads.buffer(
                &mut encoder,
                &self.prepared_instance_buffer,
                0,
                bytemuck::cast_slice(&self.prepared_uploaded),
            );
        }
        self.prepared_ranges.clear();
        if prepared_count > 0 {
            entity_batch::ordinary_vertices_at_cuts(
                entities,
                self.prepared_draws.iter().map(|draw| draw.cpu_vertices),
                oit_on,
                &mut self.prepared_cuts,
            );
            prepared_model::fill_instance_ranges(
                entity_quads * 4,
                &self.prepared_draws,
                &self.prepared_cuts,
                posed,
                &mut self.prepared_ranges,
            );
        }
        if let (Some(skinner), Some(skin)) = (&mut self.entity_skin, skin) {
            if pack_skin {
                let target = self
                    .pack_post
                    .as_ref()
                    .and_then(|(_, post)| post.as_ref())
                    .and_then(|post| post.skin_target())
                    .expect("GPU pack entity target");
                skinner.dispatch_pack(
                    &self.device,
                    &mut self.uploads,
                    &mut encoder,
                    skin,
                    target,
                    &self.entity_buffer,
                    outline_base,
                );
            } else {
                skinner.dispatch(
                    &self.device,
                    &mut self.uploads,
                    &mut encoder,
                    skin,
                    &self.entity_buffer,
                    0,
                    outline_base,
                );
            }
        }
        let static_sections = frame.static_block_entities;
        let mut static_bytes_uploaded = 0;
        let catalog_revision = frame.static_block_entity_stats.snapshot_revision;
        if catalog_revision != self.static_entity_catalog_revision {
            self.static_entity_sections.retain(|section, _| {
                static_sections
                    .binary_search_by_key(section, |snapshot| snapshot.section)
                    .is_ok()
            });
            for snapshot in static_sections {
                let vertices = snapshot.vertices.len() as u64;
                let gpu = self
                    .static_entity_sections
                    .entry(snapshot.section)
                    .or_insert_with(|| {
                        let capacity = vertices.next_power_of_two().max(4);
                        let origin_buffer =
                            self.device
                                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some("Static block-entity section origin"),
                                    contents: bytemuck::bytes_of(&snapshot.origin),
                                    usage: wgpu::BufferUsages::VERTEX,
                                });
                        static_bytes_uploaded += std::mem::size_of_val(&snapshot.origin) as u64;
                        GpuStaticEntitySection {
                            buffer: self.device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("Static block-entity section vertices"),
                                size: capacity * std::mem::size_of::<Vertex>() as u64,
                                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            }),
                            origin_buffer,
                            origin: snapshot.origin,
                            capacity,
                            revision: 0,
                            vertices: 0,
                        }
                    });
                if gpu.origin != snapshot.origin {
                    gpu.origin_buffer =
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Static block-entity section origin"),
                                contents: bytemuck::bytes_of(&snapshot.origin),
                                usage: wgpu::BufferUsages::VERTEX,
                            });
                    gpu.origin = snapshot.origin;
                    static_bytes_uploaded += std::mem::size_of_val(&snapshot.origin) as u64;
                }
                if gpu.revision == snapshot.revision {
                    continue;
                }
                if vertices > gpu.capacity {
                    gpu.capacity = vertices.next_power_of_two();
                    gpu.buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Static block-entity section vertices"),
                        size: gpu.capacity * std::mem::size_of::<Vertex>() as u64,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                }
                self.uploads.buffer(
                    &mut encoder,
                    &gpu.buffer,
                    0,
                    bytemuck::cast_slice(&snapshot.vertices),
                );
                static_bytes_uploaded += vertices * std::mem::size_of::<Vertex>() as u64;
                gpu.revision = snapshot.revision;
                gpu.vertices = vertices as u32;
            }
            self.static_entity_catalog_revision = catalog_revision;
        }
        let static_entity_quads = frame
            .visible_static_block_entity_sections
            .iter()
            .filter_map(|section| self.static_entity_sections.get(section))
            .map(|section| section.vertices / 4)
            .sum::<u32>();
        let max_static_section_quads = frame
            .visible_static_block_entity_sections
            .iter()
            .filter_map(|section| self.static_entity_sections.get(section))
            .map(|section| section.vertices / 4)
            .max()
            .unwrap_or(0);
        if max_static_section_quads > self.static_entity_index_quads {
            self.static_entity_index_quads = max_static_section_quads.next_power_of_two();
            self.static_entity_indices =
                Self::quad_indices(&self.device, self.static_entity_index_quads);
        }
        if !frame.visible_static_block_entity_sections.is_empty()
            && self.static_entity_camera != Some(frame.camera)
        {
            let camera_floor = [
                frame.camera.x.floor() as i32,
                frame.camera.y.floor() as i32,
                frame.camera.z.floor() as i32,
            ];
            let transform = StaticEntityUniform {
                camera_block: [camera_floor[0], camera_floor[1], camera_floor[2], 0],
                camera_fraction: [
                    (frame.camera.x - f64::from(camera_floor[0])) as f32,
                    (frame.camera.y - f64::from(camera_floor[1])) as f32,
                    (frame.camera.z - f64::from(camera_floor[2])) as f32,
                    0.0,
                ],
            };
            self.uploads.buffer(
                &mut encoder,
                &self.static_entity_transform,
                0,
                bytemuck::bytes_of(&transform),
            );
            static_bytes_uploaded += std::mem::size_of::<StaticEntityUniform>() as u64;
            self.static_entity_camera = Some(frame.camera);
        }
        self.static_entity_frame_stats = frame.static_block_entity_stats;
        self.static_entity_frame_stats.static_bytes_uploaded = static_bytes_uploaded;
        self.entity_outline.prepare(
            &self.device,
            &mut self.uploads,
            &mut encoder,
            frame.glowing,
            (outline_base, outlined / 4),
            [self.config.width, self.config.height],
        );
        if let Some((rotation, brightness)) = frame.stars {
            self.stars.prepare(
                &mut self.uploads,
                &mut encoder,
                jittered * rotation,
                brightness,
                &globals,
            );
        }
        self.custom_sky.prepare(
            &mut self.uploads,
            &mut encoder,
            jittered,
            &self.custom_sky_draws,
            &globals,
        );
        let celestial_vertices = frame.celestial.len().min(2048 * 4);
        if frame.end_sky && !frame.title {
            self.end_sky.prepare(&self.device, frame.end_sky_sprite);
        }
        if celestial_vertices > 0 {
            self.uploads.buffer(
                &mut encoder,
                &self.celestial_buffer,
                0,
                bytemuck::cast_slice(&frame.celestial[..celestial_vertices]),
            );
        }
        let max_quads =
            (world_quads + additive_quads + hand_quads + see_through_quads + shadow_only_quads)
                .max(cloud_quads)
                .max(if frame.lightning.bolts.is_empty() {
                    0
                } else {
                    lightning::QUADS
                })
                .max(weather_quads)
                .max(frame.glowing.len() as u32 / 4)
                .max(outlined / 4);
        if max_quads > self.index_quads {
            self.index_quads = max_quads.next_power_of_two();
            self.indices = Self::quad_indices(&self.device, self.index_quads);
        }
        if let Some(b) = &frame.border {
            if b.rebuilt {
                self.uploads.buffer(
                    &mut encoder,
                    &self.border_vertices,
                    0,
                    bytemuck::cast_slice(&b.vertices),
                );
            }
            let t = &b.transforms;
            let uniform = BorderUniform {
                color: t.color,
                offset: [
                    t.model_offset[0],
                    t.model_offset[1],
                    t.model_offset[2],
                    t.texture_offset,
                ],
            };
            self.uploads.buffer(
                &mut encoder,
                &self.border_uniform,
                0,
                bytemuck::bytes_of(&uniform),
            );
        }
        laps.lap("r_materials");
        // Every upload before the frame's first pass: a pass ends wgpu-core's
        // command buffer, so copies between passes would need one more.
        if !frame.screen.is_empty() {
            let vertices = &mut self.screen_vertices_scratch;
            vertices.clear();
            vertices.extend(frame.screen.iter().flat_map(|q| q.vertices));
            if vertices.len() as u64 > self.screen_capacity {
                self.screen_capacity = (vertices.len() as u64).next_power_of_two();
                self.screen_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Screen effect vertices"),
                    size: self.screen_capacity
                        * std::mem::size_of::<crate::screen_effects::ScreenVertex>() as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            self.uploads.buffer(
                &mut encoder,
                &self.screen_buffer,
                0,
                bytemuck::cast_slice(vertices),
            );
        }
        self.hud.prepare(
            &self.device,
            &mut self.uploads,
            &mut encoder,
            frame.hud,
            [self.config.width, self.config.height],
        );
        if culling && let Some(cull) = &mut self.cull {
            cull.upload(
                &self.device,
                &mut self.uploads,
                &mut encoder,
                &self.draws[phase1_draws as usize..solid_draws as usize],
                &self.instances,
                &self.globals,
            );
        }
        let flat_frame = match &mut self.flat {
            Some(flat)
                if self.flat_on
                    // They copy the CPU's lists alone (GPU culling is never offered
                    // where vanilla frames draw flat).
                    && !gpu_frame
                    && !self.draws.is_empty()
                    && (listed >= flat.min_sections
                        || pack_terrain
                        || shadow_view.is_some()) =>
            {
                // The visibility buffer's phase 2 IDs need each call's page (`visbuffer`).
                flat.page_instances = self.visbuf.is_some()
                    && !pack_terrain
                    && shadow_view.is_none()
                    && self.executor.is_none_or(|c| !c.material);
                let sections = &self.sections;
                let lods = &self.lods;
                let (sorted, lists) = (&self.sorted_indices, &self.world_views.sections);
                let phase2 = self.cull.as_ref().filter(|_| culling).map(|c| &c.phase2);
                crate::spans::time("flatten", || {
                    flat.prepare(
                        &self.device,
                        &mut encoder,
                        &mut self.uploads,
                        &self.draws,
                        &self.draw_pages,
                        phase1_draws as usize,
                        solid_draws as usize,
                        lists.marks([shadow_first, shadow_first + shadow_solid]),
                        [opaque_end, cutout_start, phase2_cutout as usize],
                        phase2,
                        self.translucent_arena.is_some(),
                        visible
                            .iter()
                            .chain(&shadow_visible)
                            .chain(lists.sections())
                            .filter_map(|v| {
                                let section = sections.get(v.slot);
                                let code = fade_code(section, &v.key);
                                Some((
                                    section.flat_id?,
                                    section_instance(v.origin, code, visibility(v.key)),
                                ))
                            })
                            .chain(visible_lods.iter().filter_map(|v| {
                                let id = lods[&v.key].drawn().flat_id?;
                                Some((id, lod::instance(v.origin, v.fade)))
                            })),
                        |page| sorted.buffer(page).clone(),
                        self.own_timing.as_mut(),
                    )
                })
            }
            _ => false,
        };
        // Each page's group binds the page for the solid draws, flat or not.
        if let Some(flat) = &mut self.flat {
            let arena = &self.arena;
            flat.bind_pages(&self.device, |page| {
                arena.has_page(page).then(|| arena.buffer(page))
            });
        }
        if flat_frame {
            section_draw_calls = self.flat.as_ref().map_or(0, |f| f.shadow_calls[0]);
        } else if self.multi_draw && !self.draws.is_empty() {
            if self.draws.len() as u64 > self.indirect_capacity {
                self.indirect_capacity = (self.draws.len() as u64).next_power_of_two();
                self.indirect = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Section draws"),
                    size: self.indirect_capacity * 20,
                    usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            self.uploads.buffer(
                &mut encoder,
                &self.indirect,
                0,
                bytemuck::cast_slice(&self.draws),
            );
        }
        let view_target = target_texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.output.frame.format),
            ..Default::default()
        });
        if let Some(timer) = &mut self.gpu_timer {
            timer.begin(&mut encoder);
        }
        // The frame's statistics (`show_stats`): a CPU frame's at once; a GPU frame's when the
        // pass's counts come back, with the CPU's part of it: the sections the walk listed but
        // those whose solid quads the pass drew (it counts them), its draws and every call.
        let cpu = [listed - gpu_solid, section_draws, section_draw_calls];
        if gpu_frame && let Some(gpu) = &mut self.gpu_culling {
            gpu.dispatch(&mut encoder, self.own_timing.as_mut());
        } else {
            self.show_stats(serial, cpu);
        }
        // The world's targets: the screen, or the scaled target of `post`.
        let pack_post = !frame.title && self.pack_post.as_ref().is_some_and(|p| p.1.is_some());
        // One of Horizon's own packs reads the same world target.
        let native_post = !frame.title && self.executor.is_some();
        self.set_up_pack_timing();
        let pack_reads_world = pack_post || native_post;
        laps.lap("r_uploads");
        self.ray_prepare(&mut encoder, camera, frame);
        // Phase 2 of occlusion culling draws over the depth, and a shader pack or TAA reads it.
        // Post effects (a resource pack's chains, `use_depth_buffer`) read the world's depth
        // after the world pass, as `GameRenderer.render` hands them the main target's.
        let post_reads_depth = !frame.title && !frame.post_effects.effects.is_empty();
        let keep_depth = pack_reads_world
            || post_reads_depth
            || (taa && self.world_post.taa)
            || oit_on
            || native_render::needs().world_depth
            || fg;
        // The pack's program reads the vertices this frame's draws bind.
        let pack_draws_terrain = pack_terrain && flat_frame == self.pack_terrain_flat;
        // The shadow camera's lists, drawn by the pack's shadow programs.
        let pack_draws_shadow = shadow_view.is_some() && flat_frame == self.pack_terrain_flat;
        let pack_draws_sky = pack_post
            && self
                .pack_post
                .as_ref()
                .and_then(|p| p.1.as_ref())
                .is_some_and(|p| p.has_sky());
        let pack_draws_celestial = pack_post
            && self
                .pack_post
                .as_ref()
                .and_then(|p| p.1.as_ref())
                .is_some_and(|p| p.has_celestial());
        let pack_draws_water = pack_draws_terrain
            && self
                .pack_post
                .as_ref()
                .and_then(|p| p.1.as_ref())
                .is_some_and(|p| p.has_water());
        let pack_draws_clouds = pack_post
            && self
                .pack_post
                .as_ref()
                .and_then(|p| p.1.as_ref())
                .is_some_and(|p| p.has_clouds());
        let pack_draws_outline = pack_post
            && self
                .pack_post
                .as_ref()
                .and_then(|p| p.1.as_ref())
                .is_some_and(|p| p.has_lines());
        let pack_draws_hand = pack_post
            && self
                .pack_post
                .as_ref()
                .and_then(|p| p.1.as_ref())
                .is_some_and(|p| p.has_hand());
        let pack_draws_weather = pack_post
            && self
                .pack_post
                .as_ref()
                .and_then(|p| p.1.as_ref())
                .is_some_and(|p| p.has_weather());
        let scaled = self.world_post.prepare(
            &self.device,
            [self.config.width, self.config.height],
            pack_reads_world || fg,
        );
        if oit_on {
            let size = self
                .world_post
                .size()
                .filter(|_| scaled.is_some())
                .unwrap_or([self.config.width, self.config.height]);
            let (device, layout, sprites) =
                (&self.device, &self.terrain_layout, self.terrain_sprites);
            let clouds = [0, 1].map(|i| self.cloud_pipeline.get_bind_group_layout(i));
            let format = self.output.frame.format;
            let flat = self
                .flat
                .as_ref()
                .map(|f| f.translucent.get_bind_group_layout(1));
            let features = [
                self.entity_pipeline.get_bind_group_layout(0),
                self.weather.pipeline().get_bind_group_layout(1),
                self.border_pipeline.get_bind_group_layout(0),
            ];
            self.oit
                .get_or_insert_with(|| {
                    oit::Oit::new(
                        device,
                        layout,
                        flat.as_ref(),
                        [&features[0], &features[1], &features[2]],
                        [&clouds[0], &clouds[1]],
                        sprites,
                        format,
                    )
                })
                .prepare(
                    &self.device,
                    &self.queue,
                    size,
                    &views.primary().projection,
                    [water_mask_quads > 0, cloud_quads > 0],
                );
        }
        let pack_sky_clear = pack_draws_sky
            .then(|| self.pack_post.as_ref().and_then(|p| p.1.as_ref()))
            .flatten()
            .and_then(|p| p.sky_clear());
        let clear = if let Some([r, g, b, a]) = pack_sky_clear {
            wgpu::Color {
                r: f64::from(r),
                g: f64::from(g),
                b: f64::from(b),
                a: f64::from(a),
            }
        } else if frame.title || (frame.end_sky && !frame.underwater && sky_open(frame)) {
            wgpu::Color::BLACK
        } else {
            wgpu::Color {
                r: f64::from(fog_color[0]),
                g: f64::from(fog_color[1]),
                b: f64::from(fog_color[2]),
                a: 1.0,
            }
        };
        let clear = linear_blend::stored_clear(clear, self.output.frame.linear_light);
        let sky_visible = sky_open(frame);
        laps.lap("r_world_pass");
        // One of Horizon's own packs: this frame's inputs; in a timed frame its sky draws in a
        // pass of its own, so that it has a time of its own.
        let mut native_sky_pass = false;
        if !frame.title
            && let Some(native) = &mut self.native
        {
            native_sky_pass = native.start_frame(native_pack::FrameStart {
                device: &self.device,
                queue: &self.queue,
                frame,
                size: self
                    .world_post
                    .size()
                    .filter(|_| scaled.is_some())
                    .unwrap_or([self.config.width, self.config.height]),
                output: &output,
                lights: &self.light_blocks,
                light_revision: self.light_revision,
                sky_visible,
                scene: self.ray_scene.as_ref(),
            });
        }
        self.draw_shadow_passes(
            &mut encoder,
            frame,
            shadow_view.as_ref(),
            flat_frame,
            shadow_first,
            shadow_solid,
            shadow_translucent,
            entity_quads,
            shadow_only_quads,
            outline_base,
        );
        laps.lap("r_pack_inputs");
        let reflection = self.draw_water_mirror(
            &mut encoder,
            frame,
            globals,
            view_proj,
            scaled.as_ref(),
            mirror_plane,
            clear,
            sky_visible,
            flat_frame,
            solid_draws,
        );
        let (world_view, world_depth) = match &scaled {
            Some((color, depth)) => (color, depth),
            None => (&view_target, &self.depth),
        };
        if culling && let Some(cull) = &mut self.cull {
            cull.bind_depth(&self.device, world_depth);
        }
        let oit = self.oit.as_ref().filter(|_| oit_on);
        let cull = self.cull.as_ref().filter(|_| culling && solid_draws > 0);
        // The hand draws at the end of the world pass, after a depth reset, when nothing runs
        // between them (a scaled world's post pass, HDR's world copy, a native mod's
        // `world_drawn`): one render pass fewer, ~10 µs of wgpu's encoding a frame.
        let hand_in_world = (frame.hand.len() / 4) > 0
            && !pack_draws_hand
            && scaled.is_none()
            && self.hdr.is_none()
            // Frame generation draws the hand over the real and the made frame after the world.
            && fg_frames.is_none()
            && !native_render::world_drawn_hooked();
        if native_sky_pass && let Some(native) = &mut self.native {
            native.draw_sky_pass(native_sky::SkyPass {
                device: &self.device,
                encoder: &mut encoder,
                target: (world_view, self.output.frame.format),
                depth: world_depth,
                clear,
                globals: &self.group,
            });
        }
        // One of Horizon's own packs: the mirror its material reads, and its copies of the world
        // before translucent terrain and its surface target, the size of this frame's targets.
        if let Some(native) = &mut self.native {
            native.prepare_world_targets(native_pack::WorldTargets {
                device: &self.device,
                encoder: &mut encoder,
                reflection,
                title: frame.title,
                depth: world_depth.texture(),
                color: scaled.as_ref().map(|(c, _)| c.texture()),
            });
            if !frame.title {
                native.bind_scene(&self.device, self.ray_scene.as_ref(), frame.camera);
            }
        }
        laps.lap("r_mirror");
        // One of Horizon's own packs with a material stage, or an entity stage: terrain draws with
        // its layers, and the entities with its pipelines, which read the pack's frame at group 3.
        let native = self.native.as_ref().filter(|_| !frame.title);
        let material = native.and_then(native_pack::NativePack::terrain_layers);
        let entity_stage = native.and_then(|n| n.entities());
        // Without HDR output, the solid shaders without its lighting (`SolidPipeline`).
        let hdr = self.hdr.is_some();
        let solid_pipeline = material.map_or(self.solid.get(hdr), |m| m.solid);
        // Two solid passes only where a discard costs the GPU its hidden surface
        // removal: Apple's tiled GPUs, behind Metal. Elsewhere the second pass
        // would double the solid draw calls and vertex work for nothing.
        let split_solid = self.info.backend == wgpu::Backend::Metal;
        let translucent_pipeline = material.map_or(&self.translucent, |m| m.translucent);
        let pack_flat = material.and_then(|m| m.flat);
        // A material stage's translucent pass reads a copy of the depth the solid terrain
        // left: it is stored across the pass that ends before the copy.
        let keep_depth = keep_depth || material.is_some();
        // GPU culling's occlusion test splits the world pass as cull.rs's does.
        let gpu_occluding = gpu_frame && self.gpu_culling.as_ref().is_some_and(|g| g.occluding());
        let mut pack_frame = |pass: &mut wgpu::RenderPass<'_>| {
            if let Some(m) = material {
                m.bind(pass);
            }
        };
        // A pack's solid terrain that calls `set_surface` also writes the surface target, in
        // passes of its own between the sky and everything else.
        let surface = material
            .and_then(|m| m.surface)
            .filter(|_| !pack_draws_terrain);
        // Solid terrain through the visibility buffer (`visbuffer`): flat frames without a
        // pack drawing terrain, a material stage, a surface target or a shadow pass.
        let vis_frame = flat_frame
            && !frame.title
            && !pack_terrain
            && !pack_draws_terrain
            && material.is_none()
            && surface.is_none()
            && shadow_view.is_none()
            && self
                .arena
                .page_buffers()
                .all(|(page, _)| page < visbuffer::MAX_PAGES)
            && self.visbuf.is_some();
        if vis_frame && let (Some(vis), Some(flat)) = (&mut self.visbuf, &self.flat) {
            let depth = world_depth.texture();
            vis.fit(&self.device, [depth.width(), depth.height()], flat);
        }
        {
            let attachments = [Some(wgpu::RenderPassColorAttachment {
                view: world_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: if native_sky_pass {
                        wgpu::LoadOp::Load
                    } else {
                        wgpu::LoadOp::Clear(clear)
                    },
                    store: wgpu::StoreOp::Store,
                },
            })];
            let world_pixels = pixel_count(world_depth);
            let first_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Horizon world"),
                timestamp_writes: self.own_timing.as_mut().and_then(|t| t.writes("World")),
                color_attachments: &attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: world_depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        // The visibility buffer's passes, and those after an occlusion test,
                        // draw on after it.
                        store: if cull.is_some() || gpu_occluding || keep_depth || vis_frame {
                            wgpu::StoreOp::Store
                        } else {
                            wgpu::StoreOp::Discard
                        },
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            let mut pass = pack_timing::counted(
                self.own_timing.as_mut(),
                first_pass,
                world_pixels,
                "before translucent",
            );
            pass.set_bind_group(0, &self.group, &[]);
            pack_frame(&mut pass);
            pass.stage(self.own_timing.as_mut(), "sky");
            self.draw_sky_pass_content(
                &mut pass,
                frame,
                sky_visible,
                pack_draws_sky,
                native_sky_pass,
                celestial_vertices,
                pack_draws_celestial,
                hdr_gain,
                &mut pack_frame,
            );
            // The next stage of the mods' primitives to draw (`RenderLevelStage`): a stage this
            // route does not draw its part of the world at goes with the next one it reaches.
            let mut primitive_stage = 0;
            // RenderLevelStageEvent.AfterSky: behind every terrain, once this pass drew the sky.
            if world_primitives && !pack_draws_sky && !pack_draws_terrain {
                self.world_primitives
                    .draw_through(&mut pass, &mut primitive_stage, 0);
                pass.set_bind_group(0, &self.group, &[]);
                pack_frame(&mut pass);
                pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
            }
            let terrain_pass =
                |encoder: &mut wgpu::CommandEncoder, surface: &wgpu::TextureView, clear: bool| {
                    // Its lifetime forgotten: a closure cannot tie it to `encoder`'s.
                    let mut pass = surface_pass(encoder, world_view, surface, world_depth, clear)
                        .forget_lifetime();
                    pass.set_bind_group(0, &self.group, &[]);
                    pack_frame(&mut pass);
                    pack_timing::CountedPass::plain(pass)
                };
            // The occlusion test's pipeline draws into the world's colour alone.
            let plain_pass = |encoder: &mut wgpu::CommandEncoder| {
                let mut pass = after_culling_pass(encoder, world_view, world_depth, true, None)
                    .forget_lifetime();
                pass.set_bind_group(0, &self.group, &[]);
                pack_frame(&mut pass);
                pack_timing::CountedPass::plain(pass)
            };
            if let Some(surface) = &surface {
                drop(pass);
                pass = terrain_pass(&mut encoder, surface, true);
            }
            match self
                .flat
                .as_ref()
                .filter(|_| flat_frame && !pack_draws_terrain)
            {
                _ if pack_draws_terrain => {
                    pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                }
                // The visibility buffer: the sky stays in the world pass's colour; the solid
                // calls draw their IDs and depth (phase 1, the occlusion test, phase 2), then
                // the resolve shades them, and the world pass goes on over both.
                Some(flat) if vis_frame => {
                    let vis = self.visbuf.as_ref().expect("a visibility buffer frame");
                    let vertices = flat_vertices(&self.arena, self.translucent_arena.as_ref());
                    drop(pass);
                    let mut ids =
                        vis.id_pass(&mut encoder, world_depth, true, self.own_timing.as_mut());
                    ids.set_bind_group(0, &self.group, &[]);
                    vis.draw_ids(&mut ids, flat, vertices);
                    drop(ids);
                    if let Some(cull) = cull {
                        // The test's pipeline draws into the world's colour.
                        let mut test = plain_pass(&mut encoder);
                        cull.test(&mut test);
                        drop(test);
                        cull.test_pass(&mut encoder, self.own_timing.as_mut());
                        flat.flatten_phase2(&mut encoder, self.own_timing.as_mut());
                        let mut ids =
                            vis.id_pass(&mut encoder, world_depth, false, self.own_timing.as_mut());
                        ids.set_bind_group(0, &self.group, &[]);
                        vis.draw_ids_phase2(&mut ids, flat, vertices);
                    }
                    vis.resolve(
                        &self.device,
                        &mut encoder,
                        world_view,
                        &self.group,
                        flat,
                        &self.arena,
                        self.own_timing.as_mut(),
                    );
                    pass = world_pass_after_culling(
                        &mut encoder,
                        world_view,
                        world_depth,
                        keep_depth,
                        self.own_timing.as_mut(),
                        "World after visibility buffer",
                        world_pixels,
                    );
                    pass.set_bind_group(0, &self.group, &[]);
                    pack_frame(&mut pass);
                    pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                }
                Some(flat) => {
                    let vertices = flat_vertices(&self.arena, self.translucent_arena.as_ref());
                    // Without a pack, on Metal, the opaque pass, then the cutout one
                    // (terrain.wgsl's `opaque_quad`), each over phase 1's and phase 2's
                    // calls with quads for it (phase 2's counted from its first).
                    let [opaque, cutout, phase2_split] = flat.pass_calls;
                    let phase2_split = phase2_split - flat.phase1_calls;
                    let phase2_calls = flat.solid_calls - flat.phase1_calls;
                    type Calls = std::ops::Range<usize>;
                    let passes: &[(&wgpu::RenderPipeline, Calls, Calls)] = match pack_flat {
                        Some(p) => &[(&p.0, 0..flat.phase1_calls, 0..phase2_calls)],
                        None if split_solid => &[
                            (flat.solid_opaque.get(hdr), 0..opaque, 0..phase2_split),
                            (
                                flat.solid_cutout.get(hdr),
                                cutout..flat.phase1_calls,
                                phase2_split..phase2_calls,
                            ),
                        ],
                        None => &[(flat.solid.get(hdr), 0..flat.phase1_calls, 0..phase2_calls)],
                    };
                    let draw = |pass: &mut wgpu::RenderPass<'_>| {
                        for (pipeline, calls, _) in passes {
                            pass.set_pipeline(pipeline);
                            flat.draw(pass, &flat.calls[calls.clone()], vertices);
                        }
                    };
                    pass.stage(self.own_timing.as_mut(), "solid terrain");
                    match cull {
                        // As below: phase 1, the test, then phase 2's draws
                        // that the test saw, copied after it.
                        Some(cull) => {
                            draw(&mut pass);
                            if surface.is_some() {
                                drop(pass);
                                pass = plain_pass(&mut encoder);
                            }
                            pass.split(self.own_timing.as_mut(), "occlusion boxes");
                            pass.stage(self.own_timing.as_mut(), "occlusion boxes");
                            cull.test(&mut pass);
                            drop(pass);
                            cull.test_pass(&mut encoder, self.own_timing.as_mut());
                            flat.flatten_phase2(&mut encoder, self.own_timing.as_mut());
                            pass = match &surface {
                                Some(surface) => terrain_pass(&mut encoder, surface, false),
                                None => world_pass_after_culling(
                                    &mut encoder,
                                    world_view,
                                    world_depth,
                                    keep_depth,
                                    self.own_timing.as_mut(),
                                    "World after culling",
                                    world_pixels,
                                ),
                            };
                            pass.set_bind_group(0, &self.group, &[]);
                            pack_frame(&mut pass);
                            pass.stage(self.own_timing.as_mut(), "solid terrain phase 2");
                            for (pipeline, _, calls) in passes {
                                pass.set_pipeline(pipeline);
                                flat.draw_phase2(&mut pass, calls.clone(), vertices);
                            }
                        }
                        // Without culling, phase 1 is all the solid calls.
                        None => draw(&mut pass),
                    }
                    pass.set_bind_group(0, &self.group, &[]);
                    pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                }
                None => {
                    // Without a native material, on Metal, the opaque pass, then the
                    // cutout one (terrain.wgsl's `opaque_quad`), each over phase 1's
                    // and phase 2's lists with quads for it.
                    let (opaque, cutout) = (opaque_end as u32, cutout_start as u32);
                    type Draws = std::ops::Range<u32>;
                    let passes: &[(&wgpu::RenderPipeline, Draws, Draws)] = match material {
                        None if split_solid => &[
                            (
                                self.solid_opaque.get(hdr),
                                0..opaque,
                                phase1_draws..phase2_cutout,
                            ),
                            (
                                self.solid_cutout.get(hdr),
                                cutout..phase1_draws,
                                phase2_cutout..solid_draws,
                            ),
                        ],
                        _ => &[(solid_pipeline, 0..phase1_draws, phase1_draws..solid_draws)],
                    };
                    // The solid pipelines, a native material's too, read the page themselves.
                    let pulled = self.flat.is_some();
                    pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                    pass.set_vertex_buffer(1, self.instances.slice(..));
                    pass.stage(self.own_timing.as_mut(), "solid terrain");
                    match cull {
                        Some(cull) => {
                            // Phase 1 and the boxes against its depth (in
                            // its pass, or one of their own on Metal), then
                            // what they found in sight, in a pass that loads
                            // both targets (it reads the test's results).
                            for (pipeline, draws, _) in passes {
                                pass.set_pipeline(pipeline);
                                self.draw_sections_from(
                                    &mut pass,
                                    &self.indirect,
                                    0,
                                    draws.start,
                                    draws.len() as u32,
                                    pulled,
                                );
                            }
                            if surface.is_some() {
                                drop(pass);
                                pass = plain_pass(&mut encoder);
                            }
                            pass.split(self.own_timing.as_mut(), "occlusion boxes");
                            pass.stage(self.own_timing.as_mut(), "occlusion boxes");
                            cull.test(&mut pass);
                            drop(pass);
                            cull.test_pass(&mut encoder, self.own_timing.as_mut());
                            pass = match &surface {
                                Some(surface) => terrain_pass(&mut encoder, surface, false),
                                None => world_pass_after_culling(
                                    &mut encoder,
                                    world_view,
                                    world_depth,
                                    keep_depth,
                                    self.own_timing.as_mut(),
                                    "World after culling",
                                    world_pixels,
                                ),
                            };
                            pass.set_bind_group(0, &self.group, &[]);
                            pack_frame(&mut pass);
                            pass.set_index_buffer(
                                self.indices.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            pass.set_vertex_buffer(1, self.instances.slice(..));
                            pass.stage(self.own_timing.as_mut(), "solid terrain phase 2");
                            for (pipeline, _, draws) in passes {
                                pass.set_pipeline(pipeline);
                                self.draw_sections_from(
                                    &mut pass,
                                    &cull.phase2,
                                    phase1_draws,
                                    draws.start,
                                    draws.len() as u32,
                                    pulled,
                                );
                            }
                        }
                        // Without culling, phase 1 is all the solid draws; GPU culling's
                        // own test (`GpuCulling::occluding`) leaves the CPU's to phase 2.
                        None => {
                            let gpu = self.gpu_culling.as_ref().filter(|_| gpu_frame);
                            let occluding = gpu.is_some_and(|g| g.occluding());
                            let page_group = |page: u32| {
                                pulled
                                    .then(|| self.flat.as_ref()?.page_group(page))
                                    .flatten()
                            };
                            for (pipeline, draws, _) in passes {
                                pass.set_pipeline(pipeline);
                                // GPU culling's draws first, all within the render distance.
                                if let Some(gpu) = gpu {
                                    gpu.draw(&mut pass, &self.arena, page_group, 0);
                                    pass.set_vertex_buffer(1, self.instances.slice(..));
                                }
                                if !occluding {
                                    self.draw_sections_from(
                                        &mut pass,
                                        &self.indirect,
                                        0,
                                        draws.start,
                                        draws.len() as u32,
                                        pulled,
                                    );
                                }
                            }
                            // As cull.rs's arm: the boxes against phase 1's depth, then
                            // phase 2 and the CPU's draws in a pass that loads both targets.
                            if let Some(gpu) = gpu.filter(|_| occluding) {
                                if surface.is_some() {
                                    drop(pass);
                                    pass = plain_pass(&mut encoder);
                                }
                                pass.split(self.own_timing.as_mut(), "occlusion boxes");
                                pass.stage(self.own_timing.as_mut(), "occlusion boxes");
                                gpu.test(&mut pass);
                                drop(pass);
                                gpu.compact(&mut encoder, self.own_timing.as_mut());
                                pass = match &surface {
                                    Some(surface) => terrain_pass(&mut encoder, surface, false),
                                    None => world_pass_after_culling(
                                        &mut encoder,
                                        world_view,
                                        world_depth,
                                        keep_depth,
                                        self.own_timing.as_mut(),
                                        "World after culling",
                                        world_pixels,
                                    ),
                                };
                                pass.set_bind_group(0, &self.group, &[]);
                                pack_frame(&mut pass);
                                pass.set_index_buffer(
                                    self.indices.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                pass.stage(self.own_timing.as_mut(), "solid terrain phase 2");
                                for (pipeline, draws, _) in passes {
                                    pass.set_pipeline(pipeline);
                                    gpu.draw(&mut pass, &self.arena, page_group, 1);
                                    pass.set_vertex_buffer(1, self.instances.slice(..));
                                    self.draw_sections_from(
                                        &mut pass,
                                        &self.indirect,
                                        0,
                                        draws.start,
                                        draws.len() as u32,
                                        pulled,
                                    );
                                }
                            }
                        }
                    }
                }
            }
            // `[displacement]`'s grid over the flat quads it took over, writing its depth.
            if let Some(n) = self
                .native
                .as_ref()
                .filter(|_| !frame.title && !pack_draws_terrain)
                && let Some(grid) = material.and_then(|m| n.displaced_grid(m))
            {
                pass.stage(self.own_timing.as_mut(), "displacement");
                pass.set_bind_group(0, &self.group, &[]);
                grid.draw(&mut pass);
            }
            if surface.is_some() {
                drop(pass);
                pass = plain_pass(&mut encoder);
            }
            // RenderLevelStageEvent.AfterOpaqueBlocks: after solid and cutout terrain, before
            // entities.
            if world_primitives && !pack_draws_terrain {
                self.world_primitives
                    .draw_through(&mut pass, &mut primitive_stage, 1);
                pass.set_bind_group(0, &self.group, &[]);
                pack_frame(&mut pass);
                pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
            }
            if static_entity_quads > 0 {
                pass.stage(self.own_timing.as_mut(), "static block entities");
                native_entity::set_pipeline(
                    &mut pass,
                    entity_stage,
                    &self.static_entity_pipeline,
                    true,
                );
                pass.set_bind_group(0, &self.entity_group, &[]);
                pass.set_bind_group(1, &self.static_entity_group, &[]);
                pass.set_index_buffer(
                    self.static_entity_indices.slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                for section in frame.visible_static_block_entity_sections {
                    let Some(gpu) = self.static_entity_sections.get(section) else {
                        continue;
                    };
                    let quads = gpu.vertices / 4;
                    if quads == 0 {
                        continue;
                    }
                    pass.set_vertex_buffer(0, gpu.buffer.slice(..));
                    pass.set_vertex_buffer(1, gpu.origin_buffer.slice(..));
                    pass.draw_indexed(0..quads * 6, 0, 0..1);
                }
                pass.set_bind_group(0, &self.group, &[]);
            }
            pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
            if entity_quads > 0 || !self.prepared_draws.is_empty() {
                pass.stage(self.own_timing.as_mut(), "entities");
                native_entity::set_pipeline(&mut pass, entity_stage, &self.entity_pipeline, false);
                pass.set_bind_group(0, &self.entity_group, &[]);
                pass.set_vertex_buffer(0, self.entity_buffer.slice(..));
                if self.prepared_draws.is_empty() {
                    pass.draw_indexed(0..entity_quads * 6, 0, 0..1);
                } else {
                    prepared_model::draw_ordered(
                        &mut pass,
                        &prepared_model::EntityRangePass {
                            pipeline: &self.entity_pipeline,
                            vertices: &self.entity_buffer,
                            indices: &self.indices,
                        },
                        self.prepared_models.as_ref(),
                        &self.prepared_instance_buffer,
                        &self.prepared_draws,
                        &self.prepared_ranges,
                    );
                }
                pass.set_bind_group(0, &self.group, &[]);
                pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
            }
            if entity_stage.is_some() {
                // The entity stage binds the frame's own group; a material stage that traces
                // draws translucent terrain with the scene's.
                pack_frame(&mut pass);
            }
            if text_quads > 0 && !oit_on {
                pass.stage(self.own_timing.as_mut(), "entity text");
                pass.set_pipeline(&self.text_polygon_offset_pipeline);
                self.bind_world_text(&mut pass, true, false);
                pass.set_vertex_buffer(0, self.entity_buffer.slice(..));
                pass.draw_indexed(0..text_quads * 6, (entity_quads * 4) as i32, 0..1);
                pass.set_bind_group(0, &self.group, &[]);
            }
            // FeatureRenderDispatcher.PreparedFrame.executeTranslucent: the shadows
            // come before the translucent items, which then cover them (#7026).
            if shadow_quads > 0 && !oit_on {
                pass.stage(self.own_timing.as_mut(), "shadows");
                pass.set_pipeline(&self.shadow_blob_pipeline);
                pass.set_bind_group(0, &self.entity_group, &[]);
                pass.set_vertex_buffer(0, self.entity_buffer.slice(..));
                pass.draw_indexed(
                    0..shadow_quads * 6,
                    ((base_quads
                        - translucent_item_quads
                        - translucent_particle_quads
                        - shadow_quads)
                        * 4) as i32,
                    0..1,
                );
                pass.set_bind_group(0, &self.group, &[]);
            }
            if translucent_item_quads > 0 && !oit_on {
                pass.stage(self.own_timing.as_mut(), "translucent items");
                pass.set_pipeline(&self.translucent_item_pipeline);
                pass.set_bind_group(0, &self.entity_group, &[]);
                pass.set_vertex_buffer(0, self.entity_buffer.slice(..));
                pass.draw_indexed(
                    0..translucent_item_quads * 6,
                    ((base_quads - translucent_item_quads) * 4) as i32,
                    0..1,
                );
                pass.set_bind_group(0, &self.group, &[]);
            }
            // Native terrain models use the block sprite arrays with their own
            // alpha/mipmap policies and ordinary opaque depth. Storage follows
            // the existing materials without moving their indices or frame tails.
            let mut first =
                world_quads - native_solid_quads - native_cutout_quads - native_cutout_mipped_quads;
            for (i, count) in [
                native_solid_quads,
                native_cutout_quads,
                native_cutout_mipped_quads,
            ]
            .into_iter()
            .enumerate()
            {
                if count > 0 {
                    pass.stage(self.own_timing.as_mut(), "terrain models");
                    pass.set_pipeline(&self.terrain_model_pipelines[i]);
                    pass.set_bind_group(
                        0,
                        if i == 1 {
                            &self.terrain_model_group
                        } else {
                            &self.fluid_group
                        },
                        &[],
                    );
                    pass.set_vertex_buffer(0, self.entity_buffer.slice(..));
                    pass.draw_indexed(0..count * 6, (first * 4) as i32, 0..1);
                }
                first += count;
            }
            pass.set_bind_group(0, &self.group, &[]);
            // Still-fluid block entities draw as their own ordinary sorted
            // translucent material, before terrain transparency. This material
            // always writes depth and never joins the improved-transparency queues.
            if fluid_quads > 0 {
                pass.stage(self.own_timing.as_mut(), "fluid block entities");
                pass.set_pipeline(&self.fluid_pipeline);
                pass.set_bind_group(0, &self.fluid_group, &[]);
                pass.set_vertex_buffer(0, self.entity_buffer.slice(..));
                pass.draw_indexed(
                    0..fluid_quads * 6,
                    ((base_quads + oit_entity_quads) * 4) as i32,
                    0..1,
                );
                pass.set_bind_group(0, &self.group, &[]);
            }
            // A solid block's crack is classic whatever the setting (`DESTROY_TYPES`, solid phase).
            if crumbling_quads > 0 {
                pass.stage(self.own_timing.as_mut(), "block breaking");
                pass.set_pipeline(&self.crumbling_pipeline);
                pass.set_bind_group(0, &self.entity_group, &[]);
                pass.set_vertex_buffer(0, self.entity_buffer.slice(..));
                pass.draw_indexed(
                    0..crumbling_quads * 6,
                    ((entity_quads + text_quads) * 4) as i32,
                    0..1,
                );
                pass.set_bind_group(0, &self.group, &[]);
            }
            // BoatRenderer's water patch (RenderTypes.waterMask): depth only,
            // before translucent terrain, so no water shows inside the boat.
            if water_mask_quads > 0 && !oit_on {
                pass.stage(self.own_timing.as_mut(), "boat water mask");
                pass.set_pipeline(&self.water_mask_pipeline);
                pass.set_bind_group(0, &self.entity_group, &[]);
                pass.set_vertex_buffer(0, self.entity_buffer.slice(..));
                pass.draw_indexed(
                    0..water_mask_quads * 6,
                    ((entity_quads + text_quads + crumbling_quads + translucent_crumbling_quads)
                        * 4) as i32,
                    0..1,
                );
                pass.set_bind_group(0, &self.group, &[]);
            }
            if !frame.lightning.bolts.is_empty() && !oit_on {
                pass.stage(self.own_timing.as_mut(), "lightning");
                pass.set_pipeline(&self.lightning_pipeline);
                pass.set_bind_group(0, &self.entity_group, &[]);
                self.lightning.draw(&mut pass, &frame.lightning);
                pass.set_bind_group(0, &self.group, &[]);
            }
            if swirl_quads > 0 && !oit_on {
                pass.stage(self.own_timing.as_mut(), "energy swirl");
                pass.set_pipeline(&self.energy_swirl_pipeline);
                pass.set_bind_group(0, &self.entity_group, &[]);
                pass.set_vertex_buffer(0, self.entity_buffer.slice(..));
                pass.draw_indexed(
                    0..swirl_quads * 6,
                    ((entity_quads
                        + text_quads
                        + crumbling_quads
                        + translucent_crumbling_quads
                        + water_mask_quads)
                        * 4) as i32,
                    0..1,
                );
                pass.set_bind_group(0, &self.group, &[]);
            }
            if additive_quads > 0 && !oit_on {
                pass.stage(self.own_timing.as_mut(), "additive entities");
                pass.set_bind_group(0, &self.entity_group, &[]);
                pass.set_vertex_buffer(0, self.entity_buffer.slice(..));
                let mut first = 0;
                for run in frame.additive.as_chunks::<4>().0.chunk_by(|a, b| {
                    (a[0].layer & 0xffff == DRAGON_RAYS_OVERLAY)
                        == (b[0].layer & 0xffff == DRAGON_RAYS_OVERLAY)
                }) {
                    let rays = run[0][0].layer & 0xffff == DRAGON_RAYS_OVERLAY;
                    pass.set_pipeline(if rays {
                        &self.dragon_rays_pipeline
                    } else {
                        &self.additive_pipeline
                    });
                    let end = first + run.len() as u32;
                    pass.draw_indexed(first * 6..end * 6, (world_quads * 4) as i32, 0..1);
                    first = end;
                }
                pass.set_bind_group(0, &self.group, &[]);
            }
            if outline_edges > 0 && !frame.outline_after_terrain && !pack_draws_outline {
                pass.stage(self.own_timing.as_mut(), "block outline");
                self.draw_outline(&mut pass, outline_edges as u32, frame.outline_high_contrast);
            }
            // DrawableGizmoPrimitives' opaque group writes depth, the translucent one does not.
            if debug_opaque > 0 {
                pass.stage(self.own_timing.as_mut(), "debug lines");
                pass.set_pipeline(&self.debug_lines_opaque);
                pass.set_vertex_buffer(0, self.debug_line_buffer.slice(..));
                pass.draw(0..4, 0..debug_opaque);
            }
            if debug_depth > debug_opaque {
                pass.stage(self.own_timing.as_mut(), "debug lines");
                pass.set_pipeline(&self.debug_lines);
                pass.set_vertex_buffer(0, self.debug_line_buffer.slice(..));
                pass.draw(0..4, debug_opaque..debug_depth);
            }
            if debug_top > 0 {
                pass.stage(self.own_timing.as_mut(), "debug lines");
                pass.set_pipeline(&self.debug_lines_on_top);
                pass.set_vertex_buffer(0, self.debug_line_buffer.slice(..));
                pass.draw(0..4, debug_depth..debug_depth + debug_top);
            }
            // One of Horizon's own packs reads the world before translucent terrain
            // (`opaque_depth`, `opaque_color`): the pass ends, the targets are copied, and a
            // pass that loads both carries on.
            // A material pack's translucent terrain drawn per section (no flat draws) gets a
            // pass of its own too, copies or not. Drawn on in the solid world's pass, a pack
            // with `[displacement]` under HDR output lost the Vulkan device (an NVIDIA GPU
            // fault, cause not found); a pass of its own avoids it.
            let reads_opaque = material.is_some_and(|m| m.reads_opaque);
            let per_section = material.is_some()
                && !pack_draws_water
                && oit.is_none()
                && !(flat_frame && self.flat.is_some());
            if (reads_opaque || per_section)
                && let Some(native) = &self.native
            {
                drop(pass);
                if reads_opaque {
                    native.copy_opaque(native_material::OpaqueCopy {
                        device: &self.device,
                        encoder: &mut encoder,
                        depth: world_depth.texture(),
                        color: scaled.as_ref().map(|(c, _)| c),
                    });
                }
                pass = world_pass_after_culling(
                    &mut encoder,
                    world_view,
                    world_depth,
                    keep_depth,
                    self.own_timing.as_mut(),
                    "World after copy",
                    world_pixels,
                );
                pass.set_bind_group(0, &self.group, &[]);
                pack_frame(&mut pass);
            }
            match self.flat.as_ref().filter(|_| flat_frame) {
                // The pack draws translucent terrain after its deferred programs.
                _ if pack_draws_water => {}
                // LevelRenderer.executeOit in place of the classic blend; what
                // follows draws over its composite in a pass of its own.
                flat if let Some(oit) = oit => {
                    drop(pass);
                    // LevelRenderer.executeOitWaterMask and CloudRenderer.renderOit.
                    let mask = |pass: &mut wgpu::RenderPass<'_>| {
                        pass.set_pipeline(&self.water_mask_pipeline);
                        pass.set_bind_group(0, &self.entity_group, &[]);
                        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                        pass.set_vertex_buffer(0, self.entity_buffer.slice(..));
                        pass.draw_indexed(
                            0..water_mask_quads * 6,
                            ((entity_quads
                                + text_quads
                                + crumbling_quads
                                + translucent_crumbling_quads)
                                * 4) as i32,
                            0..1,
                        );
                    };
                    let clouds =
                        |pass: &mut wgpu::RenderPass<'_>, pipeline: &wgpu::RenderPipeline| {
                            pass.set_pipeline(pipeline);
                            pass.set_bind_group(0, &self.entity_group, &[]);
                            pass.set_bind_group(1, &self.cloud_group, &[]);
                            pass.set_index_buffer(
                                self.indices.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            pass.set_vertex_buffer(0, self.cloud_buffer.slice(..));
                            pass.draw_indexed(0..cloud_quads * 6, 0, 0..1);
                        };
                    // "Features, World Border, Weather": what follows the opaque
                    // world in the classic pass, one stage at a time.
                    let features = |pass: &mut wgpu::RenderPass<'_>, stage: oit::Stage| {
                        let f = oit.features();
                        let s = stage as usize;
                        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                        pass.set_bind_group(0, &self.entity_group, &[]);
                        pass.set_vertex_buffer(0, self.entity_buffer.slice(..));
                        // Quads of a material, `first` quads into the buffer.
                        let quads = |pass: &mut wgpu::RenderPass<'_>,
                                     pipeline: &wgpu::RenderPipeline,
                                     count: u32,
                                     first: u32| {
                            if count > 0 {
                                pass.set_pipeline(pipeline);
                                pass.draw_indexed(0..count * 6, (first * 4) as i32, 0..1);
                            }
                        };
                        quads(pass, &f.text[s], text_quads, entity_quads);
                        if !hdr_gain {
                            quads(
                                pass,
                                &f.crumbling[s],
                                translucent_crumbling_quads,
                                entity_quads + text_quads + crumbling_quads,
                            );
                        }
                        quads(
                            pass,
                            &f.swirl[s],
                            swirl_quads,
                            entity_quads
                                + text_quads
                                + crumbling_quads
                                + translucent_crumbling_quads
                                + water_mask_quads,
                        );
                        quads(
                            pass,
                            &f.shadow[s],
                            shadow_quads,
                            base_quads
                                - translucent_item_quads
                                - translucent_particle_quads
                                - shadow_quads,
                        );
                        quads(
                            pass,
                            &f.entity[s],
                            translucent_particle_quads,
                            base_quads - translucent_item_quads - translucent_particle_quads,
                        );
                        quads(
                            pass,
                            &f.entity[s],
                            translucent_item_quads,
                            base_quads - translucent_item_quads,
                        );
                        // Translucent, emissive, eyes and breeze wind models:
                        // the same fragment, their light and shading are in
                        // the vertices.
                        quads(pass, &f.entity[s], oit_entity_quads, base_quads);
                        if !frame.lightning.bolts.is_empty() {
                            pass.set_pipeline(&f.lightning[s]);
                            self.lightning.draw(pass, &frame.lightning);
                            pass.set_vertex_buffer(0, self.entity_buffer.slice(..));
                        }
                        quads(pass, &f.additive[s], additive_quads, world_quads);
                        self.weather
                            .draw_with(pass, &f.weather[s], &self.entity_group);
                        if let Some(b) = frame.border.as_ref().filter(|_| !frame.title) {
                            pass.set_pipeline(&f.border[s]);
                            pass.set_bind_group(0, &self.border_group, &[]);
                            pass.set_vertex_buffer(0, self.border_vertices.slice(..));
                            for &side in &b.sides {
                                let first = side as u32 * 6;
                                pass.draw_indexed(first..first + 6, 0, 0..1);
                            }
                        }
                    };
                    let extras = oit::Extras {
                        water_mask: (water_mask_quads > 0).then_some(&mask as _),
                        clouds: (cloud_quads > 0).then_some(&clouds as _),
                        features: Some(&features),
                    };
                    // The draw closure below borrows all of `self`, so the timer is taken out.
                    let mut own_timing = self.own_timing.take();
                    oit.run(
                        &mut encoder,
                        world_view,
                        world_depth,
                        &|pass, stage| {
                            pass.set_bind_group(0, &self.group, &[]);
                            // Flat section draws take the flat pipelines.
                            if let Some((flat, pipeline)) = flat.zip(oit.flat(stage)) {
                                pass.set_pipeline(pipeline);
                                flat.draw(
                                    pass,
                                    flat.translucent_calls(),
                                    flat_vertices(&self.arena, self.translucent_arena.as_ref()),
                                );
                                return;
                            }
                            pass.set_pipeline(oit.terrain(stage));
                            pass.set_index_buffer(
                                self.indices.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            pass.set_vertex_buffer(1, self.instances.slice(..));
                            self.draw_sections(
                                pass,
                                &self.indirect,
                                0,
                                solid_draws,
                                translucent_draws,
                            );
                        },
                        &extras,
                        own_timing.as_mut(),
                    );
                    self.own_timing = own_timing;
                    pass = world_pass_after_culling(
                        &mut encoder,
                        world_view,
                        world_depth,
                        true,
                        self.own_timing.as_mut(),
                        "World after OIT",
                        world_pixels,
                    );
                    pass.set_bind_group(0, &self.group, &[]);
                    pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                }
                Some(flat) => {
                    pass.split(self.own_timing.as_mut(), "translucent terrain");
                    pass.stage(self.own_timing.as_mut(), "translucent terrain");
                    pass.set_pipeline(pack_flat.map_or(&flat.translucent, |p| &p.1));
                    flat.draw(
                        &mut pass,
                        flat.translucent_calls(),
                        flat_vertices(&self.arena, self.translucent_arena.as_ref()),
                    );
                    pass.set_bind_group(0, &self.group, &[]);
                    pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                    pass.split(self.own_timing.as_mut(), "after translucent");
                }
                None => {
                    pass.split(self.own_timing.as_mut(), "translucent terrain");
                    pass.stage(self.own_timing.as_mut(), "translucent terrain");
                    pass.set_pipeline(translucent_pipeline);
                    pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                    pass.set_vertex_buffer(1, self.instances.slice(..));
                    self.draw_sections(
                        &mut pass,
                        &self.indirect,
                        0,
                        solid_draws,
                        translucent_draws,
                    );
                    pass.split(self.own_timing.as_mut(), "after translucent");
                }
            }
            // RenderLevelStageEvent.AfterTranslucentBlocks: after translucent terrain, here or
            // OIT's.
            if world_primitives && !pack_draws_water && !pack_draws_terrain {
                self.world_primitives
                    .draw_through(&mut pass, &mut primitive_stage, 2);
                pass.set_bind_group(0, &self.group, &[]);
                pack_frame(&mut pass);
                pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
            }
            // LevelRenderer.executeTranslucentAfterTerrain: vanilla's
            // translucent particle layer draws after translucent terrain —
            // the quads write depth, so before it they would hide the glass
            // or water behind them.
            if translucent_particle_quads > 0 && !oit_on {
                pass.stage(self.own_timing.as_mut(), "translucent particles");
                pass.set_pipeline(&self.entity_pipeline);
                pass.set_bind_group(0, &self.entity_group, &[]);
                pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.set_vertex_buffer(0, self.entity_buffer.slice(..));
                pass.draw_indexed(
                    0..translucent_particle_quads * 6,
                    ((base_quads - translucent_item_quads - translucent_particle_quads) * 4) as i32,
                    0..1,
                );
                pass.set_bind_group(0, &self.group, &[]);
            }
            // Native world text is an ordinary depth-tested overlay, after
            // particles (also after the OIT composite) and before the hand.
            // Its own geometry is independent of block-entity lifetime.
            if world_text_quads + world_text_offset_quads > 0 {
                pass.stage(self.own_timing.as_mut(), "world text");
                pass.set_bind_group(0, &self.entity_group, &[]);
                pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.set_vertex_buffer(0, self.entity_buffer.slice(..));
                let first = base_quads + oit_entity_quads + fluid_quads;
                for (pipeline, count, offset) in [
                    (&self.world_text_pipeline, world_text_quads, first),
                    (
                        &self.text_polygon_offset_pipeline,
                        world_text_offset_quads,
                        first + world_text_quads,
                    ),
                ] {
                    if count > 0 {
                        pass.set_pipeline(pipeline);
                        pass.draw_indexed(0..count * 6, (offset * 4) as i32, 0..1);
                    }
                }
                pass.set_bind_group(0, &self.group, &[]);
            }
            // LevelRenderer.executeClassicTransparency: translucent targets'
            // outlines follow terrain, so its blending cannot wash them out.
            if outline_edges > 0 && frame.outline_after_terrain && !pack_draws_outline {
                pass.stage(self.own_timing.as_mut(), "block outline");
                self.draw_outline(&mut pass, outline_edges as u32, frame.outline_high_contrast);
            }
            // Cracks multiply the existing light; scalar OIT coverage cannot carry that factor.
            if translucent_crumbling_quads > 0 && (!oit_on || hdr_gain) {
                pass.stage(self.own_timing.as_mut(), "block breaking");
                pass.set_pipeline(&self.crumbling_pipeline);
                pass.set_bind_group(0, &self.entity_group, &[]);
                pass.set_vertex_buffer(0, self.entity_buffer.slice(..));
                pass.draw_indexed(
                    0..translucent_crumbling_quads * 6,
                    ((entity_quads + text_quads + crumbling_quads) * 4) as i32,
                    0..1,
                );
            }
            if cloud_quads > 0 && !pack_draws_clouds && !oit_on {
                pass.stage(self.own_timing.as_mut(), "clouds");
                pass.set_pipeline(&self.cloud_pipeline);
                pass.set_bind_group(0, &self.entity_group, &[]);
                pass.set_bind_group(1, &self.cloud_group, &[]);
                pass.set_vertex_buffer(0, self.cloud_buffer.slice(..));
                pass.draw_indexed(0..cloud_quads * 6, 0, 0..1);
            }
            // LevelRenderer.executeClassicTransparency: weather after translucent
            // terrain and clouds, so the water behind the rain is drawn first.
            pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
            if !pack_draws_weather && !oit_on {
                pass.stage(self.own_timing.as_mut(), "weather");
                let native = self.native.as_ref().filter(|_| !frame.title);
                if !native
                    .is_some_and(|n| n.draw_weather(&mut pass, &self.weather, &self.entity_group))
                {
                    self.weather.draw(&mut pass, &self.entity_group);
                }
                pass.set_bind_group(0, &self.group, &[]);
            }
            // RenderLevelStageEvent.AfterWeather: after the rain and snow this pass drew.
            if world_primitives && !pack_draws_weather && !oit_on {
                self.world_primitives
                    .draw_through(&mut pass, &mut primitive_stage, 3);
                pass.set_bind_group(0, &self.group, &[]);
                pack_frame(&mut pass);
                pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
            }
            // LevelRenderer: the world border after translucent terrain and weather.
            if let Some(b) = frame.border.as_ref().filter(|_| !frame.title && !oit_on) {
                pass.stage(self.own_timing.as_mut(), "world border");
                pass.set_pipeline(&self.border_pipeline);
                pass.set_bind_group(0, &self.border_group, &[]);
                pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.set_vertex_buffer(0, self.border_vertices.slice(..));
                for &side in &b.sides {
                    let first = side as u32 * 6;
                    pass.draw_indexed(first..first + 6, 0, 0..1);
                }
                pass.set_bind_group(0, &self.group, &[]);
            }
            // LevelRenderer.executeSeeThrough, after the rest of the world.
            if see_through_quads > 0 {
                pass.stage(self.own_timing.as_mut(), "see-through");
                pass.set_pipeline(&self.see_through_pipeline);
                self.bind_world_text(&mut pass, true, true);
                pass.set_vertex_buffer(0, self.entity_buffer.slice(..));
                pass.draw_indexed(
                    0..see_through_quads * 6,
                    ((world_quads + additive_quads + hand_quads) * 4) as i32,
                    0..1,
                );
            }
            // Native mods' primitives: after the entities and the rest of the world, into its
            // targets (its resolution, reversed depth and TAA), before the hand
            // (RenderLevelStageEvent.AfterLevel), with any stage not drawn yet.
            if world_primitives {
                pass.stage(self.own_timing.as_mut(), "mod primitives");
                self.world_primitives.draw_through(
                    &mut pass,
                    &mut primitive_stage,
                    world_primitives::STAGES - 1,
                );
            }
            if hand_in_world {
                pass.stage(self.own_timing.as_mut(), "hand");
                pass.set_pipeline(&self.depth_reset);
                pass.set_bind_group(0, &self.group, &[]);
                pass.draw(0..3, 0..1);
                self.draw_hand(&mut pass, hand_quads, world_quads + additive_quads);
            }
        }
        let (
            encoder,
            PostPassResult {
                readback,
                target_texture,
                fg_made,
            },
        ) = self.run_post_and_overlay(
            encoder,
            PostPassArgs {
                frame,
                scaled,
                view_target,
                target_texture,
                surface_frame: surface_frame.as_ref(),
                fg_frames: fg_frames.as_ref(),
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
            },
        );
        laps.lap("r_passes");
        self.submit_and_present_frame(
            encoder,
            SubmitFrameArgs {
                frame,
                checks,
                surface_frame,
                reconfigure,
                readback,
                target_texture: &target_texture,
                fg_frames,
                fg_made,
                has_section_copies,
                started,
                visible,
                visible_lods,
                shadow_visible,
            },
        )
    }
}
