use super::*;

impl Renderer {
    pub fn upload(&mut self, key: SectionKey, mesh: &SectionMesh) {
        if let Some(copies) = &mut self.capture_copies {
            copies.section(key, mesh);
        }
        let facts = MeshFacts::of(mesh);
        let fading = self.before_mesh(key, &facts);
        let section = self.upload_mesh(mesh);
        self.after_mesh(key, &facts, section, fading);
    }
    /// Section `key`'s new mesh drawn from now on, its quads uploaded (`section`: none when it
    /// has none, or the arena had no room).
    fn after_mesh(
        &mut self,
        key: SectionKey,
        mesh: &MeshFacts,
        section: Option<SectionGpu>,
        fading: Option<Instant>,
    ) {
        self.ray_mark(key);
        let section = section.map(|section| SectionGpu { fading, ..section });
        self.set_detail(key, section.is_some() || mesh.empty);
        if let Some(mut section) = section {
            let sections = self.layout.sections();
            if let Some(gpu) = &mut self.gpu_culling
                && section.gpu_recorded(key, sections)
            {
                // A fade starts with its section: flagged from the first frame that
                // draws it, once a frame has decided the flags (`mark_fades`).
                if let Some(fade_ms) = gpu.fades.fade_ms {
                    let f = FadeInputs {
                        now: Instant::now(),
                        fade_ms,
                        uploaded: &self.section_uploaded,
                        retiring: &self.lod_retiring,
                    };
                    section.gpu_fading = decide_fade(&section, key, &mut gpu.fades, f);
                }
                section.gpu_id = Some(gpu.records.insert(section.record(key, mesh.bounds)));
            }
            let marked = section.listed(key, sections);
            self.sections
                .insert_marked(key, section, mesh.bounds, marked);
        }
    }
    pub(super) fn upload_mesh(&mut self, mesh: &SectionMesh) -> Option<SectionGpu> {
        let solid = (mesh.solid.len() / 4) as u32;
        let translucent = (mesh.translucent.len() / 4) as u32;
        if solid + translucent == 0 {
            return None;
        }
        let max = solid.max(translucent);
        if max > self.index_quads {
            self.index_quads = max.next_power_of_two();
            self.indices = Self::quad_indices(&self.device, self.index_quads);
        }
        let encoder = self.section_copies.get_or_insert_with(|| {
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Section uploads"),
                })
        });
        // The solid quads, and the translucent ones after them unless they have
        // an arena of their own. With video and system memory both full, the
        // section is not drawn. The solid ones in the compact layout where the
        // mesher made it and the passes read it (`packed_quads`).
        let packed = self.packed_quads && !mesh.packed.is_empty() && solid > 0;
        let solid_slots = if packed {
            mesh.packed.len() as u64
        } else {
            mesh.solid.len() as u64
        };
        let split = self.translucent_arena.is_some();
        let main = if split {
            solid_slots
        } else {
            solid_slots + mesh.translucent.len() as u64
        };
        let (page, start) = if main > 0 {
            self.arena.alloc(&self.device, encoder, main)?
        } else {
            (0, 0)
        };
        let release_main = |arena: &mut arena::Arena| {
            if main > 0 {
                arena.release(page, start, main);
            }
        };
        let translucent_vertices = mesh.translucent.len() as u64;
        let translucent_at = match &mut self.translucent_arena {
            Some(arena) if translucent > 0 => {
                match arena.alloc(&self.device, encoder, translucent_vertices) {
                    Some(at) => at,
                    None => {
                        release_main(&mut self.arena);
                        return None;
                    }
                }
            }
            _ => (page, start + solid_slots),
        };
        let sorted_indices = if translucent > 0 {
            let indices =
                self.sorted_indices
                    .alloc(&self.device, encoder, u64::from(translucent) * 6);
            if indices.is_none() {
                release_main(&mut self.arena);
                if let Some(arena) = &mut self.translucent_arena {
                    arena.release(translucent_at.0, translucent_at.1, translucent_vertices);
                }
                return None;
            }
            indices
        } else {
            None
        };
        // Flat draws name the blocks of its ranges by its id: its range in `arena` and,
        // split, its translucent quads' in `translucent_arena` (`flat::TRANSLUCENT_KEYS`).
        let translucent_key = (split && translucent > 0)
            .then(|| (translucent_at.0 + flat::TRANSLUCENT_KEYS, translucent_at.1));
        let flat_id = self.flat.as_mut().and_then(|f| {
            let ranges = [(main > 0).then_some((page, start, main))]
                .into_iter()
                .chain([translucent_key.map(|(key, first)| (key, first, translucent_vertices))]);
            let mut id = None;
            for (key, first, vertices) in ranges.flatten() {
                let uploads = &mut self.section_uploads;
                match id {
                    None => id = Some(f.add(&self.device, encoder, uploads, key, first, vertices)),
                    Some(id) => f.place(&self.device, encoder, uploads, id, key, first, vertices),
                }
            }
            id
        });
        let stride = std::mem::size_of::<TerrainVertex>() as u64;
        if solid_slots > 0 {
            self.arena.write_sources(
                &self.device,
                encoder,
                &mut self.section_uploads,
                page,
                start,
                solid_slots,
                if packed { &[] } else { &mesh.solid_sources },
            );
        }
        if translucent_vertices > 0 {
            self.translucent_arena
                .as_mut()
                .unwrap_or(&mut self.arena)
                .write_sources(
                    &self.device,
                    encoder,
                    &mut self.section_uploads,
                    translucent_at.0,
                    translucent_at.1,
                    translucent_vertices,
                    &mesh.translucent_sources,
                );
        }
        let translucent_arena = self.translucent_arena.as_ref().unwrap_or(&self.arena);
        let solid_bytes: &[u8] = if packed {
            bytemuck::cast_slice(&mesh.packed)
        } else {
            bytemuck::cast_slice(&mesh.solid)
        };
        for (bytes, arena, (part_page, first)) in [
            (solid_bytes, &self.arena, (page, start)),
            (
                bytemuck::cast_slice(&mesh.translucent),
                translucent_arena,
                translucent_at,
            ),
        ] {
            if !bytes.is_empty() {
                self.uploaded_bytes += bytes.len() as u64;
                self.section_copy_bytes += bytes.len() as u64;
                self.section_uploads.buffer(
                    encoder,
                    arena.buffer(part_page),
                    first * stride,
                    bytes,
                );
            }
        }
        if self.section_copy_bytes >= SECTION_COPY_LIMIT
            && let Some(copies) = self.take_section_copies()
        {
            self.queue.submit([copies]);
        }
        let animated_layers =
            collect_animated_layers(mesh, |layer| self.animations.is_animated(layer));
        Some(SectionGpu {
            page,
            start,
            translucent_at,
            solid,
            solid_slots,
            packed,
            packed_far: packed && mesh.packed_far,
            sides: if mesh.sides.quads() == solid {
                mesh.sides
            } else {
                Sides::unsorted(solid)
            },
            translucent,
            sort: translucency::SortState::new(
                mesh.translucent
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|q| translucency::quad_center(q[0].pos, q[2].pos))
                    .collect(),
            ),
            sorted_indices,
            flat_id,
            gpu_id: None,
            gpu_fading: false,
            fading: None,
            leaving: None,
            animated_layers,
        })
    }
    /// The section copies recorded since the last submission, closed.
    pub(super) fn take_section_copies(&mut self) -> Option<wgpu::CommandBuffer> {
        let encoder = self.section_copies.take()?;
        self.section_copy_bytes = 0;
        self.section_uploads.finish(&encoder);
        Some(encoder.finish())
    }
    /// Draws `count` section draws from `first` with the arenas and section
    /// origins bound: one indirect multi-draw, or a loop without it.
    /// Consecutive draws in one arena page share a vertex buffer binding,
    /// keeping the list's order (translucent sections go back to front).
    /// `indirect` holds the list's draws from `base`. Leaves the quad
    /// indices bound, as the draws after it expect.
    pub(super) fn draw_sections(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        indirect: &wgpu::Buffer,
        base: u32,
        first: u32,
        count: u32,
    ) {
        self.draw_sections_from(pass, indirect, base, first, count, false);
    }
    /// `draw_sections` through the solid pipelines that read the arena page themselves
    /// (`vs_section`): each page's group is bound as their group 1.
    pub(super) fn draw_solid_sections(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        indirect: &wgpu::Buffer,
        base: u32,
        first: u32,
        count: u32,
    ) {
        self.draw_sections_from(pass, indirect, base, first, count, self.flat.is_some());
    }
    pub(super) fn draw_sections_from(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        indirect: &wgpu::Buffer,
        base: u32,
        first: u32,
        count: u32,
        pulled: bool,
    ) {
        let lists = SectionLists {
            draws: &self.draws,
            pages: &self.draw_pages,
            arena: &self.arena,
            translucent_arena: self.translucent_arena.as_ref(),
            sorted_indices: &self.sorted_indices,
            indices: &self.indices,
            flat: self.flat.as_ref(),
            multi_draw: self.multi_draw,
        };
        lists.draw(pass, indirect, base, first, count, pulled);
    }
    pub(super) fn release_section(&mut self, key: SectionKey) {
        if let Some(s) = self.sections.remove(key) {
            self.release_mesh(s);
        }
    }
    pub(super) fn release_mesh(&mut self, s: SectionGpu) {
        if let (Some(flat), Some(id)) = (&mut self.flat, s.flat_id) {
            flat.remove(id);
        }
        if let (Some(gpu), Some(id)) = (&mut self.gpu_culling, s.gpu_id) {
            gpu.records.remove(id);
        }
        if let Some((page, start)) = s.sorted_indices {
            self.sorted_indices
                .release(page, start, u64::from(s.translucent) * 6);
        }
        let (main, translucent) = s.vertex_ranges(self.translucent_arena.is_some());
        if let Some((page, start, vertices)) = main {
            self.arena.release(page, start, vertices);
        }
        if let (Some(arena), Some((page, start, vertices))) =
            (&mut self.translucent_arena, translucent)
        {
            arena.release(page, start, vertices);
        }
    }
    /// Video memory the arena keeps because fragmentation holds its pages
    /// partly used (a render distance that went down, far tiles that were
    /// replaced): moves the meshes off the sparsest page, a few MiB a frame,
    /// and `Arena::trim` returns the page once it is empty. Frames with the
    /// arena in system memory (video memory full) are slow, so this is what
    /// keeps the terrain in video memory. The section arena's: the large pages
    /// of the translucent arena and the sorted indices are `shrink_translucent`'s.
    /// Returns what is left of the frame's `COMPACT_BUDGET`.
    pub(super) fn compact_arena(&mut self) -> u64 {
        let mut budget = COMPACT_BUDGET;
        if self.arena.closing().is_none()
            && let Some(page) = self.arena.sparsest()
        {
            self.arena.close(page);
        }
        let Some(page) = self.arena.closing() else {
            return budget;
        };
        let encoder = self.section_copies.get_or_insert_with(|| {
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Section uploads"),
                })
        });
        let meshes = self
            .sections
            .values_mut()
            .chain(self.lods.values_mut().flat_map(|tile| tile.meshes_mut()));
        let flat = self
            .flat
            .as_mut()
            .map(|f| (f, &self.device, &mut self.section_uploads));
        // GPU culling's records move with the sections.
        let gpu = self.gpu_culling.as_mut().map(|g| &mut g.records);
        let split = self.translucent_arena.is_some();
        let (arena, part) = (&mut self.arena, Part::Section { split });
        if !move_off_page(arena, encoder, flat, gpu, meshes, page, part, &mut budget) {
            // The other pages have no room after all.
            arena.reopen();
        }
        // With every mesh moved off, the page stays closed so this frame's
        // uploads can't refill it: `trim` returns it and clears `closing`.
        budget
    }
    /// Shrinks the large pages of the translucent arena and the sorted indices
    /// once terrain has settled (`set_terrain_loading`): one far larger than what
    /// it holds moves into a new page of their size (`begin_shrink`), through
    /// `move_off_page` with what `compact_arena` left of the frame's budget
    /// (`left`), and `trim` returns it once empty. One page at a time, vertices
    /// first; only starts wait, a move under way finishes even if terrain comes
    /// again (stopped, the page would stay closed while the new one fills).
    pub(super) fn shrink_translucent(&mut self, left: u64) {
        let Some(vertices) = self.translucent_arena.as_mut() else {
            return; // split off: nothing to shrink
        };
        let now = Instant::now();
        let mut arenas = [
            (vertices, Part::Translucent),
            (&mut self.sorted_indices, Part::Indices),
        ];
        if arenas.iter().all(|(arena, _)| arena.closing().is_none()) {
            if !self.shrink_gate.may_start(now, left) {
                return;
            }
            // The next start in a second, or `SHRINK_RETRY` after a try found no room.
            let mut wait = SHRINK_SETTLE;
            for (arena, part) in &mut arenas {
                let meshes = self
                    .sections
                    .values_mut()
                    .chain(self.lods.values_mut().flat_map(|tile| tile.meshes_mut()));
                match begin_shrink(arena, &self.device, meshes, *part) {
                    Some(true) => break,
                    Some(false) => wait = SHRINK_RETRY, // the other arena may still go
                    None => {}
                }
            }
            self.shrink_gate.wait(now, wait);
        }
        let Some((arena, part, page)) = arenas.into_iter().find_map(|(arena, part)| {
            let page = arena.closing()?;
            Some((arena, part, page))
        }) else {
            return;
        };
        let encoder = self.section_copies.get_or_insert_with(|| {
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Section uploads"),
                })
        });
        let meshes = self
            .sections
            .values_mut()
            .chain(self.lods.values_mut().flat_map(|tile| tile.meshes_mut()));
        // As many quads a frame either way: a quad's indices are a quarter of its bytes.
        let cap = match part {
            Part::Indices => COMPACT_BUDGET / 4,
            _ => COMPACT_BUDGET,
        };
        let mut budget = left.min(cap);
        // Flat draws name translucent quads' blocks by their section's id; indices have none.
        let flat = match part {
            Part::Translucent => self.flat.as_mut(),
            _ => None,
        }
        .map(|f| (f, &self.device, &mut self.section_uploads));
        // GPU culling's records follow the section arena alone.
        if !move_off_page(arena, encoder, flat, None, meshes, page, part, &mut budget) {
            // New ranges took the room: both pages stay, drawn, until a later try.
            arena.reopen();
            self.shrink_gate.wait(now, SHRINK_RETRY);
        }
    }
    /// Block-entity blocks of every meshed section (see `SectionMesh::block_entities`).
    pub fn block_entities(&self) -> impl Iterator<Item = &([i32; 3], u32)> {
        self.block_entities.values().flatten()
    }
    /// `block_entities` of the sections far enough into their fade-in to draw them
    /// (`LevelExtractor.extractVisibleBlockEntities`: `getVisibility(now, fade) >= 0.3`).
    pub fn visible_block_entities(
        &self,
        fade_seconds: f32,
    ) -> impl Iterator<Item = &([i32; 3], u32)> {
        let fade = fade_millis(fade_seconds);
        let now = std::time::Instant::now();
        let all = !self.sections_fading_in(fade, now);
        self.block_entities
            .iter()
            .filter(move |(key, _)| {
                all || chunk_visibility(self.section_elapsed_ms(**key, now), fade)
                    >= CHUNK_VISIBILITY_THRESHOLD
            })
            .flat_map(|(_, blocks)| blocks)
    }
    /// Whether a section may still be fading in over `fade_ms` (vanilla's chunk fade): one was
    /// uploaded more recently than that. Otherwise every section is whole.
    pub(super) fn sections_fading_in(&self, fade_ms: u64, now: std::time::Instant) -> bool {
        fade_ms > 0
            && self
                .newest_upload
                .is_some_and(|t| (now.saturating_duration_since(t).as_millis() as u64) < fade_ms)
    }
    /// Whether a section may be fading in or out over far terrain (`SectionGpu::fade`):
    /// otherwise every section's code is 0.
    pub(super) fn sections_fading_over_lods(&self, now: std::time::Instant) -> bool {
        !self.lod_retiring.is_empty()
            || !self.lod_leaving.is_empty()
            || self
                .newest_lod_fade
                .is_some_and(|t| now.saturating_duration_since(t) < lod::FADE)
    }
    /// Block entities in sections which can intersect the camera frustum.
    /// The section bounds include the same one-block margin as the per-block
    /// check in the game. Sections wholly inside the frustum skip that finer
    /// check; sections outside it are kept only when a renderer needs to draw
    /// geometry extending beyond its block (beacon beams, gateways, outlines).
    pub fn visible_block_entities_in_frustum<'a>(
        &'a self,
        fade_seconds: f32,
        camera: DVec3,
        frustum: Option<JoinedFrusta>,
        keep_section: impl Fn(u32) -> bool + 'a,
        keep_long_range: impl Fn(u32) -> bool + 'a,
    ) -> impl Iterator<Item = (&'a ([i32; 3], u32), bool)> + 'a {
        let fade = fade_millis(fade_seconds);
        let now = Instant::now();
        // No section uploaded within the fade: all are whole, none is looked up.
        let all = !self.sections_fading_in(fade, now);
        let min_y = self.layout.min_y();
        self.block_entities
            .iter()
            .filter(move |(key, _)| {
                all || chunk_visibility(self.section_elapsed_ms(**key, now), fade)
                    >= CHUNK_VISIBILITY_THRESHOLD
            })
            .filter_map(move |(key, blocks)| {
                let origin = DVec3::new(
                    f64::from(key.0 * 16 - 1),
                    f64::from(min_y + key.1 * 16 - 1),
                    f64::from(key.2 * 16 - 1),
                ) - camera;
                let min = origin.as_vec3();
                // The last block's expanded bound ends at section_origin + 17.
                let max = min + Vec3::splat(18.0);
                let distance_squared = Vec3::ZERO.max(min).min(max).length_squared();
                let in_range = distance_squared <= 64.0 * 64.0
                    || (distance_squared <= 256.0 * 256.0
                        && blocks.iter().any(|(_, state)| keep_long_range(*state)));
                if !in_range {
                    return None;
                }
                let (intersects, fully_inside) = frustum.map_or((true, true), |view| {
                    let fully_inside = view.contains(min, max);
                    (fully_inside || view.intersects(min, max), fully_inside)
                });
                (intersects || blocks.iter().any(|(_, state)| keep_section(*state)))
                    .then_some((blocks, fully_inside))
            })
            .flat_map(|(blocks, fully_inside)| {
                blocks.iter().map(move |block| (block, fully_inside))
            })
    }
    /// The block entities of the sections a shader pack's shadow pass drew in the last frame
    /// (`shadow_visible`), within the 64 blocks `visible_block_entities_in_frustum` keeps.
    /// Our probe packs' Iris 1.11.7 shadow maps hold a block entity wherever the shadow pass
    /// draws its section's terrain, whether or not the camera's view takes it in (#8464).
    pub fn shadow_pass_block_entities(
        &self,
        camera: DVec3,
    ) -> impl Iterator<Item = &([i32; 3], u32)> + '_ {
        block_entities_within_reach(
            self.shadow_visible.iter().map(|v| v.key),
            &self.block_entities,
            camera,
            self.layout.min_y(),
        )
    }
    /// Whether a shader pack's shadow pass drew the section holding block `pos` in the last
    /// frame (`shadow_pass_block_entities`).
    pub fn shadow_pass_drew(&self, pos: [i32; 3]) -> bool {
        let key = (
            pos[0] >> 4,
            (pos[1] - self.layout.min_y()) >> 4,
            pos[2] >> 4,
        );
        self.shadow_visible.iter().any(|v| v.key == key)
    }
    /// Milliseconds since `key` was uploaded; a section not in `section_uploaded` is old.
    fn section_elapsed_ms(&self, key: SectionKey, now: std::time::Instant) -> u64 {
        self.section_uploaded.get(&key).map_or(u64::MAX, |t| {
            now.saturating_duration_since(*t).as_millis() as u64
        })
    }
    /// A value that changes whenever `block_entities` gives something else.
    pub fn block_entity_revision(&self) -> u64 {
        self.block_entity_revision
    }
    /// Both mesh/light and block-entity membership revisions for a section
    /// key (`x`, section-y relative to the dimension's minimum, `z`).
    pub fn block_entity_section_revision(
        &self,
        section: [i32; 3],
    ) -> Option<StaticSectionRevision> {
        let key = (section[0], section[1], section[2]);
        Some(StaticSectionRevision {
            mesh: *self.block_entity_section_revisions.get(&key)?,
            block_entities: *self.block_entity_membership_revisions.get(&key)?,
        })
    }
    /// Block-entity cache work and upload volume from the last rendered frame.
    pub fn static_block_entity_frame_stats(&self) -> StaticCacheFrameStats {
        self.static_entity_frame_stats
    }
    /// Drops every section, for a level whose chunks hold `layout`.
    pub fn clear_sections(&mut self, layout: crate::chunk::ColumnLayout) {
        if let Some(copies) = &mut self.capture_copies {
            *copies = capture::Copies::default();
        }
        self.block_entities.clear();
        self.block_entity_section_revisions.clear();
        self.block_entity_membership_revisions.clear();
        self.section_uploaded.clear();
        self.newest_upload = None;
        self.newest_lod_fade = None;
        self.block_entity_revision = next_block_entity_revision();
        self.sections.clear();
        self.ray_clear();
        self.lods.clear();
        self.lod_cutout.clear();
        self.lod_retiring.clear();
        self.lod_leaving.clear();
        self.lod_pending = None;
        self.detail_columns.clear();
        self.detail_revision += 1;
        // A dimension change drops every page at once: freed later, off this thread.
        self.graveyard.add(self.arena.clear());
        if let Some(gpu) = &mut self.gpu_culling {
            gpu.records.clear();
            // The chunk fade-in time they were decided with stays.
            gpu.fades.clear();
        }
        if let Some(arena) = &mut self.translucent_arena {
            self.graveyard.add(arena.clear());
        }
        if let Some(flat) = &mut self.flat {
            flat.clear();
        }
        self.graveyard.add(self.sorted_indices.clear());
        self.translucency_cursor = 0;
        if layout == self.layout {
            self.occlusion.clear();
        } else {
            self.layout = layout;
            self.occlusion = occlusion::Occlusion::new(layout.min_y(), layout.sections() as i32);
            self.sections.set_min_y(layout.min_y());
        }
    }
    /// Lays the sections out as `translucent_split` says for the shader pack that
    /// runs, before a world frame draws them. A change drops every section and far
    /// tile, as a new renderer has none, and changes the sorted indices' arena with
    /// them (large pages beside the translucent arena, else pages flat draws bind):
    /// the game meshes them again (`take_dropped_sections`).
    pub(super) fn follow_translucent_layout(&mut self) {
        let pack = self.pack_post.as_ref().and_then(|p| p.1.as_ref());
        let split = translucent_split(
            self.translucent_batching,
            self.flat.is_some() && self.flat_on,
            pack.is_some_and(pack_post::PackPost::has_terrain),
            // One of Horizon's own packs with `[shadow]` draws from the same lists (`shadow_view`).
            pack.is_some_and(pack_post::PackPost::has_shadow) || self.native_shadow(),
        );
        if split == self.translucent_arena.is_some() {
            return;
        }
        self.sections_dropped |=
            self.sections.len() > 0 || !self.lods.is_empty() || self.lod_pending.is_some();
        self.clear_sections(self.layout);
        (self.sorted_indices, self.translucent_arena) = arena::Arena::translucent_arenas(
            &self.device,
            std::mem::size_of::<TerrainVertex>() as u64,
            self.flat.is_some(),
            split,
        );
        if let (Some(flat), Some(arena)) = (&mut self.flat, &self.translucent_arena) {
            flat.set_translucent_page_vertices(arena.page_vertices());
        }
    }
    /// Uploads solid quads in the compact layout (`packed_quads`) unless a pack reads the
    /// arena's vertices as `TerrainVertex`es: a `shaders/` pack's terrain and shadow
    /// programs outside flat draws (flat ones unpack the records, `pack_atlas::PULL_FLAT`).
    /// Leaving it drops every section and far tile for the game to mesh again.
    pub(super) fn follow_packed_layout(&mut self) {
        let pack = self.pack_post.as_ref().and_then(|p| p.1.as_ref()).is_some();
        let want = self.flat.is_some() && (!pack || self.pack_terrain_flat) && self.packed_on;
        if want == self.packed_quads {
            return;
        }
        log::info!(
            "Horizon: compact solid terrain {}",
            if want { "on" } else { "off" }
        );
        self.packed_quads = want;
        packed::pack_quads(want);
        let laid_out = self.sections.values().any(|s| s.packed)
            || self.lods.values().any(|t| t.holds_packed());
        if !want && laid_out {
            self.sections_dropped = true;
            self.clear_sections(self.layout);
        }
    }
    /// Allows the compact layout of solid quads or not, for A/B checks: every section and far
    /// tile is meshed again in the layout that follows. Replies whether sections now use it.
    pub fn set_packed_layout(&mut self, on: bool) -> bool {
        if on != self.packed_on {
            self.packed_on = on;
            self.follow_packed_layout();
            self.sections_dropped = true;
            self.clear_sections(self.layout);
        }
        self.packed_quads
    }
    /// Whether every section and far tile was dropped since the last call, to be
    /// laid out again (`follow_translucent_layout`): the game meshes them again,
    /// as for a new renderer.
    pub fn take_dropped_sections(&mut self) -> bool {
        std::mem::take(&mut self.sections_dropped)
    }
    /// Whether terrain is on its way (the view not all in, events, chunks or
    /// uploads still waiting, far tiles sampled): a large translucent page shrinks
    /// once that, new translucent sections (`upload`) and far tiles to upload
    /// (`upload_lods`) have been over for `SHRINK_SETTLE`, not at the low point
    /// between a view's columns leaving and the next one's arriving, after which
    /// it would grow back by whole copies.
    pub fn set_terrain_loading(&mut self, loading: bool) {
        if loading {
            self.shrink_gate.wait(Instant::now(), SHRINK_SETTLE);
        }
    }
    /// A column whose every section has its mesh (`lod_detail_columns`).
    fn complete_column(&self) -> SectionMask {
        SectionMask::below(self.layout.sections())
    }
    pub fn lod_detail_columns(&self) -> impl Iterator<Item = (i32, i32)> + '_ {
        let complete = self.complete_column();
        self.detail_columns
            .iter()
            .filter_map(move |(&column, &mask)| (mask == complete).then_some(column))
    }
    /// Changes whenever `lod_detail_columns` does.
    pub fn lod_detail_revision(&self) -> u64 {
        self.detail_revision
    }
    /// Whether the section at `key` has its mesh, for `lod_detail_columns`.
    fn set_detail(&mut self, key: SectionKey, meshed: bool) {
        let column = (key.0, key.2);
        let mask = self
            .detail_columns
            .get(&column)
            .copied()
            .unwrap_or_default();
        let mut next = mask;
        if meshed {
            next.insert(key.1 as usize);
        } else {
            next.remove(key.1 as usize);
        }
        let complete = self.complete_column();
        if (mask == complete) != (next == complete) {
            self.detail_revision += 1;
        }
        if next.is_empty() {
            self.detail_columns.remove(&column);
        } else {
            self.detail_columns.insert(column, next);
        }
    }
    /// Every section of these columns goes: one lookup per column where the
    /// tables are by column, and the meshes' ranges freed in one batch. A
    /// teleport removes a few thousand columns in one frame.
    /// A section whose column far terrain now draws fades out over it, chunk by
    /// chunk as it came, and is released once it has (`release_left`): taken
    /// away at once, the column's detail dropped to the far terrain's in one
    /// frame.
    pub(super) fn leave_section(&mut self, key: SectionKey) {
        let now = std::time::Instant::now();
        if let Some(section) = self.sections.find_mut(key)
            && section.leaving.is_none()
        {
            section.leaving = Some(now);
            self.lod_leaving.push((key, now));
            if let Some(gpu) = &mut self.gpu_culling {
                hold_fading(&mut self.sections, &mut gpu.records, key);
            }
        }
    }
    /// Releases the sections that have faded out (`leave_section`).
    pub(super) fn release_left(&mut self, now: std::time::Instant) {
        let mut done = Vec::new();
        self.lod_leaving.retain(|&(key, since)| {
            let left = now.saturating_duration_since(since) >= lod::FADE;
            if left {
                done.push((key, since));
            }
            !left
        });
        for (key, since) in done {
            // Meshed again since (the player came back): it stays.
            if self
                .sections
                .find(key)
                .is_some_and(|s| s.leaving == Some(since))
            {
                self.release_section(key);
            }
        }
    }
    pub fn remove_columns(&mut self, columns: impl IntoIterator<Item = (i32, i32)>) {
        let columns: Vec<(i32, i32)> = columns.into_iter().collect();
        if let Some(copies) = &mut self.capture_copies {
            copies.remove_columns(&columns);
        }
        // Most of the view at once (a teleport at render distance 32 drops some 3700
        // columns): a pass over each store instead of a lookup per section key, which
        // took 10 ms of one frame. Far terrain fades the sections out instead.
        let keys = columns.len() * self.layout.sections();
        if self.lods.is_empty()
            && self.lod_cutout.is_empty()
            && keys >= self.sections.len().max(BULK_REMOVAL_KEYS)
        {
            self.remove_columns_at_once(&columns);
            return;
        }
        let mut released = Vec::new();
        let mut block_entities = false;
        for (x, z) in columns {
            if self.detail_columns.remove(&(x, z)) == Some(self.complete_column()) {
                self.detail_revision += 1;
            }
            self.occlusion.remove_column(x, z);
            let retiring = !self.lod_cutout.is_empty() && self.lod_cutout.contains(&(x, z));
            for sy in 0..self.layout.sections() as i32 {
                let key = (x, sy, z);
                block_entities |= self.block_entities.remove(&key).is_some();
                self.block_entity_section_revisions.remove(&key);
                self.block_entity_membership_revisions.remove(&key);
                if self.light_blocks.remove(&key).is_some() {
                    self.light_revision += 1;
                }
                self.section_uploaded.remove(&key);
                if retiring {
                    // Distant tiles still leave its column out: drawn until they don't.
                    self.lod_retiring.insert(key);
                    if let Some(gpu) = &mut self.gpu_culling {
                        hold_fading(&mut self.sections, &mut gpu.records, key);
                    }
                } else if !self.lods.is_empty() && self.sections.contains(key) {
                    // Far terrain draws the column: it fades out over it.
                    self.leave_section(key);
                } else if let Some(section) = self.sections.remove(key) {
                    released.push(section);
                }
            }
        }
        if block_entities {
            self.block_entity_revision = next_block_entity_revision();
        }
        self.release_meshes(released);
    }
    /// `remove_columns` for a large share of the resident sections: each store filtered
    /// once by column.
    fn remove_columns_at_once(&mut self, columns: &[(i32, i32)]) {
        let gone: crate::fast_hash::FxHashSet<(i32, i32)> = columns.iter().copied().collect();
        let complete = self.complete_column();
        for &(x, z) in columns {
            if self.detail_columns.remove(&(x, z)) == Some(complete) {
                self.detail_revision += 1;
            }
        }
        self.occlusion.remove_columns(&gone);
        let kept = |key: &SectionKey| !gone.contains(&(key.0, key.2));
        let before = self.block_entities.len();
        self.block_entities.retain(|key, _| kept(key));
        if self.block_entities.len() != before {
            self.block_entity_revision = next_block_entity_revision();
        }
        let before = self.light_blocks.len();
        self.light_blocks.retain(|key, _| kept(key));
        self.light_revision += (before - self.light_blocks.len()) as u64;
        self.section_uploaded.retain(|key, _| kept(key));
        let released = self.sections.remove_columns(&gone);
        self.release_meshes(released);
    }
    /// Gives the removed sections' quads back to their arenas; the sections themselves
    /// (their sort centres) drop on a thread of their own when there are many.
    fn release_meshes(&mut self, released: Vec<SectionGpu>) {
        if let Some(flat) = &mut self.flat {
            for id in released.iter().filter_map(|s| s.flat_id) {
                flat.remove(id);
            }
        }
        if let Some(gpu) = &mut self.gpu_culling {
            gpu.records
                .remove_all(released.iter().filter_map(|s| s.gpu_id));
        }
        self.sorted_indices
            .release_all(released.iter().filter_map(|s| {
                let (page, start) = s.sorted_indices?;
                Some((page, start, u64::from(s.translucent) * 6))
            }));
        let split = self.translucent_arena.is_some();
        self.arena
            .release_all(released.iter().filter_map(|s| s.vertex_ranges(split).0));
        if let Some(arena) = &mut self.translucent_arena {
            arena.release_all(released.iter().filter_map(|s| s.vertex_ranges(true).1));
        }
        if released.len() >= BULK_REMOVAL_KEYS {
            arena::free_later(released);
        }
    }
}
