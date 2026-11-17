use super::*;

impl Renderer {
    /// Whether sections are drawn as one index list per frame (`flat`).
    pub fn flat_draws(&self) -> bool {
        self.flat_on
    }
    /// Switches between flat and per-section draws, for A/B checks.
    pub fn set_flat_draws(&mut self, on: bool) -> Result<()> {
        if on && self.flat.is_none() {
            return Err("flat section draws are off on this device".into());
        }
        self.flat_on = on;
        Ok(())
    }
    /// Whether one of Horizon's own packs reads the water mirror, which needs the frame's
    /// `water_plane`.
    pub fn wants_water_plane(&self) -> bool {
        self.executor.is_some_and(|c| c.reflection)
    }
    /// Whether solid quads facing away from the eye are left out.
    pub fn section_sides(&self) -> bool {
        self.sides_on
    }
    /// Switches leaving out solid quads facing away, for A/B checks.
    pub fn set_section_sides(&mut self, on: bool) {
        self.sides_on = on;
    }
    /// Whether solid sections are occlusion culled (`cull`).
    pub fn section_cull(&self) -> bool {
        self.cull_on
    }
    /// Switches occlusion culling, for A/B checks.
    pub fn set_section_cull(&mut self, on: bool) -> Result<()> {
        if on && self.cull.is_none() {
            return Err("occlusion culling is off on this device".into());
        }
        self.cull_on = on;
        Ok(())
    }
    /// Of the boxes occlusion culling's last test read back drew, how many
    /// it did not see and how many it drew, and the share of solid indices
    /// it hid.
    pub fn culled_sections(&self) -> Option<([usize; 2], f32)> {
        if self.gpu_frame {
            // GPU culling's own test, on Vulkan.
            let gpu = self.gpu_culling.as_ref().filter(|_| self.cull_on);
            return gpu.and_then(|g| g.hidden());
        }
        self.cull
            .as_ref()
            .filter(|_| self.cull_on)
            .map(|c| c.hidden)
    }
    /// Whether the near solid sections are culled and drawn on the GPU where
    /// frames allow it (`gpu_culling`).
    pub fn gpu_culling(&self) -> bool {
        self.gpu_culling.is_some()
    }
    /// Whether the last frame GPU culling drew ran its occlusion test.
    pub fn gpu_occluding(&self) -> bool {
        self.gpu_frame && self.gpu_culling.as_ref().is_some_and(|g| g.occluding())
    }
    /// Whether the last frame drew them so.
    pub fn gpu_frame(&self) -> bool {
        self.gpu_frame
    }
    /// Of the sections GPU culling keeps records of, those it leaves to the CPU
    /// while they fade (`SectionGpu::gpu_fading`); `None` while it is off.
    pub fn gpu_fading(&self) -> Option<u32> {
        self.gpu_culling.as_ref().map(|g| g.records.fading())
    }
    /// Switches GPU culling: the GPU Culling setting sets it (`apply_ui_settings`, after
    /// every settings change and every new renderer), and so does the control word. It
    /// turns on only where it is offered (`gpu_culling_offer`); unchanged, it does nothing.
    /// Its pass, its records of the resident sections and their flags while they
    /// fade are made when it is switched on (the next world frame decides the
    /// flags, `mark_fades`) and dropped when it is switched off, with the walk
    /// marks the flags gave: nothing is kept for it while it is off but the
    /// sections' other walk marks (`SectionGpu::listed`) and the cave-culling
    /// graph's bits (`occlusion::Reach`), which are the same either way.
    pub fn set_gpu_culling(&mut self, on: bool) -> Result<()> {
        if !on {
            if self.gpu_culling.take().is_some() {
                forget_records(&mut self.sections, self.layout.sections());
            }
            self.gpu_frame = false;
            return Ok(());
        }
        if self.gpu_culling.is_some() {
            return Ok(());
        }
        if let Some(why) = self.gpu_culling_offer.reason() {
            return Err(why.into());
        }
        let test = self
            .cull
            .as_ref()
            .filter(|c| c.early())
            .map(|_| self.output.frame.format);
        let Some(mut gpu) = gpu_culling::GpuCulling::new(&self.device, test) else {
            // Not offered from now on: its switch greys out.
            self.offer_gpu_culling(crate::graphics::GpuCullingOffer::Unsupported);
            return Err("GPU culling could not be set up on this device".into());
        };
        let sections = self.layout.sections();
        for (key, s, bounds) in self.sections.iter_mut() {
            if s.gpu_recorded(key, sections) {
                s.gpu_id = Some(gpu.records.insert(s.record(key, bounds)));
            }
        }
        self.gpu_culling = Some(gpu);
        #[cfg(debug_assertions)]
        {
            self.gpu_checks = 0;
        }
        Ok(())
    }
    /// GPU culling offered or not from now on, here and in the settings' switch.
    pub(super) fn offer_gpu_culling(&mut self, offer: crate::graphics::GpuCullingOffer) {
        self.gpu_culling_offer = offer;
        crate::graphics::offer_gpu_culling(offer);
    }
    /// Whether GPU culling is offered here, and why not (the status's `gpu_culling_offer`).
    pub fn gpu_culling_offer(&self) -> crate::graphics::GpuCullingOffer {
        self.gpu_culling_offer
    }
    /// Debug builds: GPU culling's records are the sections' as they are now,
    /// compaction's moves included: each section it keeps a record of
    /// (`SectionGpu::gpu_recorded`) has its own, no record is left without its
    /// section, and each page they count is the arena's. A record left behind
    /// would draw another section's quads, or from a freed page. And each
    /// section's walk mark is `SectionGpu::listed`: one unmarked that should be
    /// would leave its translucent quads, or all of it, undrawn. And the bits
    /// the pass copies of the cave-culling graph are its cells'
    /// (`occlusion::Reach`): one off would hide a section the walk shows. And,
    /// at `f.now`, a recorded section draws faded only while it is flagged
    /// (`SectionGpu::gpu_fading`, its record's `FADING`), and an end is waited
    /// for while it is unless it leaves or is held: one the pass drew whole
    /// would pop in, one flagged for good would cost the CPU its draws.
    #[cfg(debug_assertions)]
    pub(super) fn check_gpu_records(&self, f: FadeInputs<'_>) {
        let Some(gpu) = &self.gpu_culling else {
            return;
        };
        assert!(
            self.occlusion.reach_is_exact(),
            "the reach bits are the graph's cells"
        );
        let (records, sections) = (&gpu.records, self.layout.sections());
        let pending: crate::fast_hash::FxHashSet<SectionKey> = gpu.fades.pending().collect();
        let mut pages = vec![0; records.page_sections().len()];
        let (mut live, mut flagged) = (0, 0);
        for (key, s, bounds) in self.sections.iter() {
            let recorded = s.gpu_recorded(key, sections);
            assert_eq!(s.gpu_id.is_some(), recorded, "{key:?}: a record");
            let marked = self.sections.is_marked(key);
            assert_eq!(marked, s.listed(key, sections), "{key:?}: its walk mark");
            assert!(
                !s.gpu_fading || s.gpu_id.is_some(),
                "{key:?}: flagged without a record"
            );
            assert!(
                s.gpu_id.is_none() || s.gpu_fading || !s.drawn_faded(key, f),
                "{key:?}: GPU culling draws it whole while it fades"
            );
            assert!(
                !s.gpu_fading || s.fade_end(key, f).is_none() || pending.contains(&key),
                "{key:?}: flagged, with no end waited for"
            );
            if let Some(id) = s.gpu_id {
                let record = s.record(key, bounds);
                assert_eq!(records.at(id), Some(&record), "{key:?}: its record {id}");
                pages[s.page as usize] += 1;
                live += 1;
                flagged += u32::from(s.gpu_fading);
            }
        }
        assert_eq!(records.live(), live, "records of sections gone");
        assert_eq!(records.fading(), flagged, "flagged records");
        assert_eq!(records.page_sections(), pages, "records per page");
        for (page, &n) in pages.iter().enumerate() {
            assert!(
                n == 0 || self.arena.has_page(page as u32),
                "records on freed page {page}"
            );
        }
    }
    pub fn section_count(&self) -> usize {
        self.sections.len()
    }
    /// Whether the running native pack gives any block a `[[lights]]` colour.
    pub fn has_light_colors(&self) -> bool {
        self.native
            .as_ref()
            .is_some_and(|native| !native.light_colors.is_empty())
    }
    /// The running native pack's `[[lights]]` colour of the block `name`.
    pub fn light_color(&self, name: &str) -> Option<[f32; 3]> {
        let native = self.native.as_ref()?;
        horizon_shader_api::manifest::light_color(&native.light_colors, name)
    }
    /// The running native pack's `[[particle_lights]]`.
    pub fn particle_lights(&self) -> Option<&[horizon_shader_api::manifest::ParticleLight]> {
        self.native.as_ref().map(|n| n.particle_lights.as_slice())
    }
    /// Whether one of Horizon's own packs with `[shadow]` draws, whose shadow pass takes the
    /// frame's `shadow_only` quads.
    pub fn native_shadow(&self) -> bool {
        self.executor.is_some_and(|c| c.shadow)
    }
    /// What a shader pack's shadow pass draws of the entities this frame: (entities or block
    /// entities other than the player, so those Horizon's culling hides from the camera are
    /// kept for it; the local player, so its model is built in first person too). (false,
    /// false) without a pack or its shadow pass.
    pub fn pack_shadow_casters(&self) -> (bool, bool) {
        let Some((_, Some(post))) = &self.pack_post else {
            return (false, false);
        };
        let c = post.shadow_casters();
        ((c.entities && !c.only_player) || c.blocks, c.entities)
    }

    /// A standard pack replaces mobs' vanilla blob shadows only when its compiled
    /// shadow path actually draws entities other than the player.
    pub fn pack_entity_shadows(&self) -> bool {
        self.pack_post
            .as_ref()
            .and_then(|(_, post)| post.as_ref())
            .is_some_and(|post| {
                let casters = post.shadow_casters();
                casters.entities && !casters.only_player
            })
    }

    pub fn has_section(&self, key: SectionKey) -> bool {
        self.sections.contains(key)
    }
    /// Bring the cave culling graph up to date for `camera` without waiting:
    /// a search this starts runs on its worker while the frame is prepared,
    /// and `render` takes it if it has finished.
    pub fn start_occlusion(&mut self, camera: DVec3, distance: f32) {
        self.occlusion.update(camera, (distance / 16.0) as i32 + 2);
    }
    /// Whether the camera's current section graph knows this section is
    /// reachable. `None` means the graph cannot safely answer for this camera
    /// or section, so callers should keep the geometry.
    pub fn section_occlusion_visible(
        &self,
        key: SectionKey,
        camera_section: SectionKey,
    ) -> Option<bool> {
        self.occlusion.visible_for_camera(key, camera_section)
    }
}
