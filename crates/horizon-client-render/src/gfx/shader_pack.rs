use super::*;

impl Renderer {
    /// Runs `post` over the world from now on (`None`: no pack), for the dimension `dimension`;
    /// `block_ids` gives each source state its model and fluid material ids (two values
    /// per state); emission and ownership come from the mesher's quad records.
    /// Returns the programs that run and the notes of loading them, or why the pack can't run:
    /// the dimension then runs without a pack.
    pub fn set_shader_pack(
        &mut self,
        dimension: i32,
        pack: Option<horizon_shaders::Post>,
        block_ids: &[i32],
    ) -> std::result::Result<Option<(Vec<String>, Vec<String>)>, pack_post::Refused> {
        if pack.is_none() {
            self.pack_atlas = None;
            self.pack_material_atlases = None;
        }
        let instance = wgpu::vertex_attr_array![5=>Float32x4];
        let vertex = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<TerrainVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &TERRAIN_ATTRIBUTES,
        };
        let instances = wgpu::VertexBufferLayout {
            array_stride: 16,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &instance,
        };
        let buffers = [Some(vertex), Some(instances)];
        let flat = self.pack_terrain_flat;
        let sprite_rects = self
            .pack_atlas
            .as_ref()
            .map_or_else(Vec::new, |a| a.rects_table());
        let post = pack.map(|post| {
            let world = pack_post::World {
                atlas: self.pack_atlas.as_ref().map(|a| (a.view.clone(), a.size())),
                sprite_rects: &sprite_rects,
                material_maps: self
                    .pack_material_atlases
                    .as_ref()
                    .map(|a| [a[0].view.clone(), a[1].view.clone()]),
                lightmap: self.lightmap_view.clone(),
                gui_size: self
                    .hud
                    .atlas_texture
                    .as_ref()
                    .map_or([1, 1], |t| [t.width(), t.height()]),
                terrain: self.pack_atlas.as_ref().map(|_| pack_post::Geometry {
                    // Flat terrain programs pull their vertices (`pack_atlas::PULL_FLAT`).
                    buffers: if flat { &[] } else { &buffers },
                    table: self.flat.as_ref().filter(|_| flat).map(|f| &f.draw_layout),
                    depth_write: true,
                    atlas: None,
                    material_maps: true,
                    ids: None,
                    cull: true,
                }),
                block_ids,
                celestial: self
                    .hud
                    .atlas_texture
                    .as_ref()
                    .map(|t| t.create_view(&Default::default())),
                clouds: Some(&self.cloud_bindings),
                weather: true,
                hand: true,
            };
            pack_post::PackPost::new(
                &self.device,
                &self.queue,
                &self.adapter,
                post,
                self.output.frame.format,
                world,
                &std::sync::atomic::AtomicBool::new(false),
            )
        });
        self.pack_loading = None;
        self.install_shader_pack(dimension, post.transpose())
    }
    /// Runs the pack `set_shader_pack` or `load_shader_pack` built (`None`: no pack), or none when
    /// it can't run, for the dimension `dimension`; the programs that run and their notes.
    fn install_shader_pack(
        &mut self,
        dimension: i32,
        post: std::result::Result<Option<pack_post::PackPost>, pack_post::Refused>,
    ) -> std::result::Result<Option<(Vec<String>, Vec<String>)>, pack_post::Refused> {
        let mut post = match post {
            Ok(post) => post,
            Err(e) => {
                self.pack_post = Some((dimension, None));
                self.pack_atlas = None;
                self.pack_material_atlases = None;
                return Err(e);
            }
        };
        if let Some(post) = post.as_mut() {
            post.set_paged_maps(
                (self.shared_material_maps.each_ref())
                    .map(|t| t.as_ref().map(|t| t.create_view(&Default::default()))),
            );
        }
        let report = post.as_ref().map(|p| {
            let programs = p.programs().into_iter().map(str::to_owned).collect();
            (programs, p.notes.clone())
        });
        self.pack_post = Some((dimension, post));
        if self.pack_post.as_ref().is_some_and(|p| p.1.is_some()) {
            self.replace_native(None);
        }
        Ok(report)
    }
    /// Sets the external shader pipeline world frames call (`external`), or none.
    pub fn set_external_pipeline(&mut self, pipeline: Option<Box<dyn external::ExternalPipeline>>) {
        self.external = pipeline;
    }
    /// Makes the running shader pack, of either kind, forget what it kept from earlier frames:
    /// the next frame draws as a freshly loaded pack's first. Nothing to do without a pack.
    pub fn reset_shader_pack_history(&mut self) {
        // A frame still on the submit thread uses the targets and readbacks replaced
        // here.
        self.land_frame();
        if let Some(pipeline) = &mut self.external {
            pipeline.reset_history();
        }
        if let Some((_, Some(post))) = &mut self.pack_post {
            post.reset_history(&self.device);
        }
        if let Some(native) = &mut self.native {
            native.frame.reset_history();
            native.post.reset_history();
        }
    }
    /// The running shader pack's kind: `iris` (a `shaders/` pack) or `native` (one of
    /// Horizon's own).
    pub fn shader_pack_kind(&self) -> Option<&'static str> {
        match (&self.pack_post, &self.native) {
            (Some((_, Some(_))), _) => Some("iris"),
            (_, Some(_)) => Some("native"),
            _ => None,
        }
    }
    /// The running Horizon pack's name, from its manifest.
    pub fn native_pack_name(&self) -> Option<&str> {
        self.native.as_ref().map(|pack| pack.name.as_str())
    }
    /// `screenshot-sequence`: waits (30 seconds at most) for the frame just drawn to finish on
    /// the GPU, and takes the centre depth a `shaders/` pack copied in it, which its next frame
    /// reads: that frame's, or an error, never an older one. The last frame's PNG has already
    /// waited for it, without a limit, as `screenshot` does.
    pub fn finish_sequence_frame(&mut self) -> Result<()> {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(30)),
            })
            .map_err(|e| format!("the frame didn't finish on the GPU: {e}"))?;
        if self.device_lost() {
            return Err("the graphics device was lost".into());
        }
        if let Some((_, Some(post))) = &mut self.pack_post {
            post.take_center_depth(&self.device)?;
        }
        Ok(())
    }
    /// As `set_shader_pack` with a pack, but its pipelines build on a thread of their own while
    /// the game keeps drawing without one: the driver compiles a big pack's programs for seconds.
    /// `poll_shader_pack` runs it once they are built. `label` and `started` come back with it.
    pub fn load_shader_pack(
        &mut self,
        dimension: i32,
        pack: horizon_shaders::Post,
        block_ids: Vec<i32>,
        label: String,
        started: std::time::Instant,
    ) {
        let device = self.device.clone();
        let queue = self.queue.clone();
        let adapter = self.adapter.clone();
        let format = self.output.frame.format;
        let atlas = self.pack_atlas.as_ref().map(|a| (a.view.clone(), a.size()));
        let sprite_rects = self
            .pack_atlas
            .as_ref()
            .map_or_else(Vec::new, |a| a.rects_table());
        let material_maps = self
            .pack_material_atlases
            .as_ref()
            .map(|a| [a[0].view.clone(), a[1].view.clone()]);
        let lightmap = self.lightmap_view.clone();
        let gui_size = self
            .hud
            .atlas_texture
            .as_ref()
            .map_or([1, 1], |t| [t.width(), t.height()]);
        let celestial = self
            .hud
            .atlas_texture
            .as_ref()
            .map(|t| t.create_view(&Default::default()));
        let flat = self.pack_terrain_flat;
        let table = (self.flat.as_ref())
            .filter(|_| flat)
            .map(|f| f.draw_layout.clone());
        let terrain = self.pack_atlas.is_some();
        let clouds = self.cloud_bindings.clone();
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let cancelled = cancel.clone();
        let build = move || {
            let instance = wgpu::vertex_attr_array![5=>Float32x4];
            let vertex = wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<TerrainVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &TERRAIN_ATTRIBUTES,
            };
            let instances = wgpu::VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &instance,
            };
            let buffers = [Some(vertex), Some(instances)];
            let world = pack_post::World {
                atlas,
                sprite_rects: &sprite_rects,
                material_maps,
                lightmap,
                gui_size,
                terrain: terrain.then(|| pack_post::Geometry {
                    // Flat terrain programs pull their vertices (`pack_atlas::PULL_FLAT`).
                    buffers: if flat { &[] } else { &buffers },
                    table: table.as_ref(),
                    depth_write: true,
                    atlas: None,
                    material_maps: true,
                    ids: None,
                    cull: true,
                }),
                block_ids: &block_ids,
                celestial,
                clouds: Some(&clouds),
                weather: true,
                hand: true,
            };
            pack_post::PackPost::new(&device, &queue, &adapter, pack, format, world, &cancelled)
        };
        // A pack is not the game's code: a panic building it turns the pack off (#3525).
        let thread = std::thread::Builder::new()
            .name("Shader pack".into())
            .spawn(move || crate::crash::contain(build));
        match thread {
            Ok(thread) => {
                // The last pack doesn't draw this dimension meanwhile.
                self.pack_post = None;
                self.pack_loading = Some(PackLoading {
                    dimension,
                    thread,
                    label,
                    started,
                    _cancel: CancelOnDrop(cancel),
                });
            }
            Err(e) => {
                log::error!("shader pack {label}: no thread to build it on: {e}");
                // Off for this dimension, as a pack that fails, not tried again every frame.
                self.pack_loading = None;
                self.pack_post = Some((dimension, None));
            }
        }
    }
    /// The pack `load_shader_pack` builds, once its thread is done: it runs from now on, as
    /// `set_shader_pack` would run it. Its dimension, label, start and what `set_shader_pack`
    /// returns.
    #[allow(clippy::type_complexity)]
    pub fn poll_shader_pack(
        &mut self,
    ) -> Option<(
        i32,
        String,
        std::time::Instant,
        std::result::Result<Option<(Vec<String>, Vec<String>)>, pack_post::Refused>,
    )> {
        if !self.pack_loading.as_ref()?.thread.is_finished() {
            return None;
        }
        let loading = self.pack_loading.take()?;
        let built = match loading.thread.join() {
            Ok(Ok(built)) => built,
            Ok(Err(panic)) => Err(pack_post::Refused {
                reason: format!("building it panicked: {panic}"),
                notes: Vec::new(),
            }),
            Err(_) => Err(pack_post::Refused {
                reason: "its thread panicked".into(),
                notes: Vec::new(),
            }),
        };
        let report = self.install_shader_pack(loading.dimension, built.map(Some));
        Some((loading.dimension, loading.label, loading.started, report))
    }
    /// Whether a `shaders/` pack is still building (`load_shader_pack`).
    pub fn shader_pack_loading(&self) -> bool {
        self.pack_loading.is_some()
    }
    /// Drops the running shader pack, of either kind.
    pub fn drop_shader_packs(&mut self) {
        self.pack_loading = None;
        self.pack_post = None;
        self.replace_native(None);
        self.pack_atlas = None;
        self.separate_ao = false;
    }
    /// Sets whether shader pack terrain inputs pass vertex alpha to `hz_Color.a` (`separateAo=true`).
    pub fn set_separate_ao(&mut self, separate_ao: bool) {
        if self.separate_ao != separate_ao {
            self.separate_ao = separate_ao;
            if let Some(atlas) = &mut self.pack_atlas {
                atlas.separate_ao = separate_ao;
            }
        }
    }
    /// Lava's sprite layers (still and flowing), whose faces a pack's solid terrain draws.
    pub fn set_pack_lava_layers(&mut self, layers: Vec<u16>) {
        self.pack_lava_layers = layers;
    }
    /// Runs the sky, material and post stages of one of Horizon's own shader packs from now
    /// on (`None`: no pack), with its options' values; replaces a `shaders/` pack. Returns the
    /// stages' labels; on an error the pack that ran keeps running.
    pub fn set_native_shader_pack(
        &mut self,
        pack: Option<(
            horizon_shader_api::Report,
            Vec<(String, f64)>,
            Vec<(u32, u32)>,
            Vec<(u32, u32)>,
            block_light::LightTable,
        )>,
    ) -> std::result::Result<Vec<String>, String> {
        let linear = pack.is_some();
        if (linear || self.output.frame.format == LINEAR_FRAME)
            && linear != (self.output.frame.format == LINEAR_FRAME)
            && !std::mem::replace(&mut self.frame_kept_logged, true)
        {
            log::info!(
                "Horizon renderer: the {:?} frame stays until the renderer is made again; a shader pack picked or dropped since runs on it",
                self.output.frame.format
            );
        }
        let Some((report, constants, sprite_tags, sprite_materials, lights)) = pack else {
            self.replace_native(None);
            return Ok(Vec::new());
        };
        let terrain = native_material::TerrainShape {
            globals: &self.terrain_layout,
            flat: self.flat.as_ref().map(|f| &f.draw_layout),
            sprites: self.terrain_sprites,
            sprite_views: &self.sprite_views,
            format: self.output.frame.format,
            linear: self.output.frame.linear_light,
            sprite_tags: &sprite_tags,
            sprite_materials: &sprite_materials,
            entities: &self.entity_bindings,
            retained_entities: &self.static_entity_pipeline.get_bind_group_layout(1),
            weather_groups: [&self.weather.entity_layout, &self.weather.time_layout],
            pbr: self.pbr_maps.as_ref().and_then(|arrays| arrays.first()),
        };
        let (pack, labels) = native_pack::NativePack::new(
            &self.device,
            &self.queue,
            &terrain,
            report,
            &constants,
            lights,
        )?;
        if let Some(refused) = self.replace_native(Some(pack)) {
            return Err(refused);
        }
        self.pack_post = None;
        self.pack_atlas = None;
        self.pack_material_atlases = None;
        Ok(labels)
    }
    /// What a shader pack's gbuffers programs read from Horizon's geometry: the sky's, and
    /// terrain's in the pack's block atlas (made here when missing; none when it does not fit).
    pub fn pack_inputs(&mut self) -> horizon_shaders::GameInputs {
        let terrain = self.pack_terrain_inputs();
        // A geometry stage reads flat draws' vertices itself, through their index buffer.
        let terrain_geometry = terrain
            .clone()
            .filter(|_| self.pack_terrain_flat)
            .map(|mut inputs| {
                inputs.declarations.push_str(
                    "layout(set = 5, binding = 0) readonly buffer HzIndices { uint hz_indices[]; };\n",
                );
                horizon_shaders::GeometryInputs {
                    inputs,
                    // Flat terrain inputs pull their own vertex (`PULL_FLAT`, `gl_VertexIndex` → `hz_vertex_index`).
                    pull: String::new(),
                    index: "hz_indices[@]".to_owned(),
                }
            });
        horizon_shaders::GameInputs {
            terrain_geometry,
            entity_geometry: Some(pack_entities::geometry_inputs()),
            sky: Some(pack_sky::inputs()),
            celestial: Some(pack_sky::celestial_inputs()),
            entities: Some(pack_entities::inputs()),
            clouds: Some(pack_entities::cloud_inputs()),
            weather: Some(pack_weather::inputs()),
            lines: Some(pack_lines::inputs()),
            hand: Some(pack_hand::inputs()),
            ..pack_atlas::game_terrain(terrain, &self.pack_lava_layers)
        }
    }

    pub(super) fn pack_terrain_inputs(&mut self) -> Option<horizon_shaders::GbuffersInputs> {
        // Flat draws whenever the device has them (`render` keeps them on with a pack).
        let flat = self.flat.is_some() && self.flat_on;
        // Flat draws read each sprite's place from a table: the atlas is stitched as the game's.
        // Other draws need the grid.
        if !flat && self.pack_atlas.as_ref().is_some_and(|a| a.is_stitched()) {
            self.pack_atlas = None;
        }
        if self.pack_atlas.is_none() {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            let layers = self
                .sprite_textures
                .iter()
                .map(|t| t.depth_or_array_layers())
                .sum();
            let sprite = self.sprite_textures[0].width();
            // Vanilla's padding counts the anisotropy bit only with anisotropic filtering.
            let anisotropy_bit = if self.texture_filtering == 2 {
                u32::from(self.max_anisotropy).trailing_zeros()
            } else {
                0
            };
            let stitched = flat
                .then(|| {
                    pack_atlas::PackAtlas::stitched(
                        &self.device,
                        &mut encoder,
                        &self.sprite_textures,
                        self.sprite_split,
                        sprite,
                        &self.sprite_native,
                        self.mipmap_option,
                        anisotropy_bit,
                    )
                })
                .flatten();
            if flat && stitched.is_none() {
                log::warn!(
                    "shader pack block atlas: the sprites don't stitch on this device, a grid instead"
                );
            }
            let mut atlas = stitched.or_else(|| {
                pack_atlas::PackAtlas::new(
                    &self.device,
                    &mut encoder,
                    &self.sprite_textures,
                    self.sprite_split,
                    layers,
                    sprite,
                )
            });
            if let Some(atlas) = &mut atlas {
                atlas.separate_ao = self.separate_ao;
            }
            self.pack_material_atlases = atlas.as_ref().and_then(|colour| {
                let maps = self.pbr_maps.as_ref()?;
                let make = |kind: usize, encoder: &mut wgpu::CommandEncoder| {
                    let arrays: Vec<_> = maps.iter().map(|pair| pair[kind].clone()).collect();
                    colour.material_for(&self.device, encoder, &arrays, layers)
                };
                Some([make(0, &mut encoder)?, make(1, &mut encoder)?])
            });
            self.pack_atlas = atlas;
            self.queue.submit([encoder.finish()]);
        }
        self.pack_terrain_flat = flat;
        Some(
            self.pack_atlas
                .as_ref()?
                .terrain_inputs(self.pack_terrain_flat),
        )
    }
    /// The macros this renderer gives shader packs: the GL extensions this device has
    /// (subgroups, when it has them), and the vendor, renderer and mipmap levels
    /// (`Macros::define_renderer`).
    pub fn define_pack_macros(&self, macros: &mut horizon_shaders::Macros) {
        macros.define_renderer(self.mipmaps);
        if self.device.features().contains(wgpu::Features::SUBGROUP) {
            for ext in [
                "basic",
                "vote",
                "arithmetic",
                "ballot",
                "shuffle",
                "shuffle_relative",
            ] {
                macros.define(format!("MC_GL_KHR_shader_subgroup_{ext}"), "");
            }
        }
    }

    /// The dimension the shader pack was last set for.
    pub fn shader_pack_dimension(&self) -> Option<i32> {
        let loading = self.pack_loading.as_ref().map(|l| l.dimension);
        loading.or_else(|| self.pack_post.as_ref().map(|p| p.0))
    }
}
