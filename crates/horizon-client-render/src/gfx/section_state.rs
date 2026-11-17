use super::*;

/// What a section's mesh brings with it besides its quads (`Renderer::upload`): placed with
/// the mesh.
pub(super) struct MeshFacts<'a> {
    pub(super) visibility: u64,
    pub(super) block_entities: &'a [([i32; 3], u32)],
    pub(super) lights: &'a [([i32; 3], u32)],
    pub(super) bounds: Option<[glam::Vec3; 2]>,
    /// It has translucent quads.
    pub(super) translucent: bool,
    /// It has no quads at all: that is its mesh.
    pub(super) empty: bool,
}
impl<'a> MeshFacts<'a> {
    pub(super) fn of(mesh: &'a SectionMesh) -> Self {
        Self {
            visibility: mesh.visibility,
            block_entities: &mesh.block_entities,
            lights: &mesh.lights,
            bounds: mesh.bounds,
            translucent: !mesh.translucent.is_empty(),
            empty: mesh.solid.is_empty() && mesh.translucent.is_empty(),
        }
    }
}
/// The water mirror's targets: the world as a camera mirrored about the water plane sees
/// it, at half the world's resolution (the material stage's `reflection`).
pub(super) struct MirrorTarget {
    pub(super) size: [u32; 2],
    pub(super) color: wgpu::TextureView,
    pub(super) depth: wgpu::TextureView,
}
pub(super) struct SectionGpu {
    /// Arena page and first vertex of its solid quads, and after them its
    /// translucent quads unless those have an arena of their own; (0, 0),
    /// unused, when that leaves it no quads in `arena`.
    pub(super) page: u32,
    pub(super) start: u64,
    /// The page and first vertex of its translucent quads: in
    /// `translucent_arena` when there is one, else after the solid quads.
    /// Unused without translucent quads.
    pub(super) translucent_at: (u32, u64),
    pub(super) solid: u32,
    /// Slots its solid quads take in `arena`: four vertices each, or in the compact layout
    /// (`packed`, `mesher::packed`) a record each and the vertices of those without one.
    pub(super) solid_slots: u64,
    pub(super) packed: bool,
    /// Its records are a far tile's (`lod::packed`), not a section's: the two
    /// layouts are told apart by the base vertex its draws use.
    pub(super) packed_far: bool,
    /// Where the solid quads' buckets by facing side end (`mesher::Sides`).
    pub(super) sides: Sides,
    pub(super) translucent: u32,
    pub(super) sort: translucency::SortState,
    pub(super) sorted_indices: Option<(u32, u64)>,
    /// Its id for flat draws (`flat::Flat::add`), when they are on.
    pub(super) flat_id: Option<u32>,
    /// Its record's id in GPU culling's table (`gpu_culling::Records`), while
    /// that is on and it has solid quads.
    pub(super) gpu_id: Option<u32>,
    /// Its record is `FADING` (GPU culling on): the CPU draws its solid quads
    /// while its instance carries a fade (`drawn_faded`), and GPU culling's
    /// pass leaves it out.
    pub(super) gpu_fading: bool,
    /// When it came in over the far terrain it replaces (drawn whole at once,
    /// `fade`): a section new where far tiles are drawn, which skips vanilla's
    /// fade-in. Meshed again, it stays as it was.
    pub(super) fading: Option<std::time::Instant>,
    /// When it began to fade out, its column gone over far terrain that
    /// stays behind it (`Renderer::leave_section`); released once it has.
    pub(super) leaving: Option<std::time::Instant>,
    /// Animated sprite layers present in this section.
    pub(super) animated_layers: Box<[u16]>,
}
impl SectionGpu {
    /// Its fade code this frame: 0, drawn whole, or `lod::HELD` (whole, in
    /// front of far terrain) while it leaves (`leaving`) or its column has left
    /// the render distance and far terrain is coming in under it (`retiring`).
    /// A near section never dithers over far terrain: the holes of its pattern
    /// showed its own inside (cave walls, the stone under the grass), nearer
    /// than the tile behind, an x-ray of each column coming in or going out.
    /// It comes in whole over the tile it replaces, which stays behind it until
    /// `fading` is `lod::FADE` old, and leaves whole after as long.
    pub(super) fn fade(&self, retiring: bool) -> u32 {
        if self.leaving.is_some() || retiring {
            lod::HELD
        } else {
            0
        }
    }
    /// Its vertices in `arena`, and in `translucent_arena` when `split` (there
    /// is one): (page, first vertex, vertices) of each.
    pub(super) fn vertex_ranges(
        &self,
        split: bool,
    ) -> (Option<(u32, u64, u64)>, Option<(u32, u64, u64)>) {
        if !split {
            let vertices = self.solid_slots + u64::from(self.translucent) * 4;
            return (Some((self.page, self.start, vertices)), None);
        }
        let (page, start) = self.translucent_at;
        (
            (self.solid > 0).then_some((self.page, self.start, self.solid_slots)),
            (self.translucent > 0).then(|| (page, start, u64::from(self.translucent) * 4)),
        )
    }
    /// Whether GPU culling keeps a record of it at `key` while it is on (`gpu_id`):
    /// it has solid quads, in one of the level's `sections`, which the pass's
    /// tables span.
    pub(super) fn gpu_recorded(&self, key: SectionKey, sections: usize) -> bool {
        self.solid > 0 && (0..sections as i32).contains(&key.1)
    }
    /// The base vertex of its solid draws: quad `q`'s corners are `base + 4 * q` on (the
    /// quad indices). In a compact layout they name its record's slot past
    /// `PACKED_BASE`, or past `FAR_BASE` for a far tile's records, which is how
    /// terrain.wgsl's `section_corner` tells the two apart.
    pub(super) fn solid_base(&self) -> u32 {
        if !self.packed {
            return self.start as u32;
        }
        let base = if self.packed_far {
            crate::lod::packed::FAR_BASE
        } else {
            packed::PACKED_BASE
        };
        base + 4 * self.start as u32
    }
    /// GPU culling's record of it at `key`, its quads within `bounds`
    /// (`SectionMesh::bounds`) and in their buckets by facing side (`sides`),
    /// `FADING` while it fades (`gpu_fading`).
    pub(super) fn record(&self, key: SectionKey, bounds: Option<[Vec3; 2]>) -> gpu_culling::Record {
        let boxed = gpu_culling::rounded_box(bounds);
        gpu_culling::Record::new(
            key,
            self.page,
            u64::from(self.solid_base()),
            self.solid,
            boxed,
        )
        .facing(&self.sides)
        .fading(self.gpu_fading)
    }
    /// Whether GPU frames' walk lists it within the render distance
    /// (`Resident::walk`): its mark. Its translucent quads are the CPU's, or
    /// GPU culling can have no record of it, which holds whether GPU culling is
    /// on or not; or, while it is, its solid quads are the CPU's while it fades
    /// (`gpu_fading`).
    pub(super) fn listed(&self, key: SectionKey, sections: usize) -> bool {
        self.translucent > 0 || !self.gpu_recorded(key, sections) || self.gpu_fading
    }
    /// Whether GPU culling draws its solid quads in a GPU frame, `near`: its
    /// column within the render distance.
    pub(super) fn gpu_draws_solid(&self, near: bool) -> bool {
        near && self.gpu_id.is_some() && !self.gpu_fading
    }
    /// Whether frames draw it faded at `f.now`: its instance, as the instance
    /// loop makes it (`section_instance` with its `fade` and its vanilla
    /// visibility), is not a whole section's. As time goes on a fade only ends:
    /// once false, this stays false until the section is meshed again, leaves,
    /// is held (`lod_retiring`), or the chunk fade-in time changes.
    pub(super) fn drawn_faded(&self, key: SectionKey, f: FadeInputs<'_>) -> bool {
        let fade = self.fade(f.retiring.contains(&key));
        let visibility = section_visibility(f.uploaded.get(&key), f.fade_ms, f.now);
        section_instance([0.0; 3], fade, visibility)[3].to_bits()
            != section_instance([0.0; 3], 0, 1.0)[3].to_bits()
    }
    /// When its fade is over at the latest (`drawn_faded`): the end of
    /// vanilla's fade-in (its first upload and `f.fade_ms`; one coming in over
    /// far terrain is whole at once, `fade`). None while it leaves or is held,
    /// which ends only when it goes or is meshed again, and without a fade-in.
    pub(super) fn fade_end(&self, key: SectionKey, f: FadeInputs<'_>) -> Option<Instant> {
        if self.leaving.is_some() || f.retiring.contains(&key) {
            return None;
        }
        let vanilla = f.uploaded.get(&key).filter(|_| f.fade_ms > 0);
        vanilla.map(|&t| t + std::time::Duration::from_millis(f.fade_ms))
    }
}
/// Which of a mesh's ranges `move_off_page` moves: what it holds in the section
/// arena (`split`: its solid quads alone, `SectionGpu::vertex_ranges`), its
/// translucent quads in `translucent_arena`, or its sorted indices.
#[derive(Clone, Copy)]
pub(super) enum Part {
    Section { split: bool },
    Translucent,
    Indices,
}
impl Part {
    /// (page, first vertex or index, count), as the release paths free it.
    pub(super) fn of(self, s: &SectionGpu) -> Option<(u32, u64, u64)> {
        match self {
            Self::Section { split } => s.vertex_ranges(split).0,
            Self::Translucent => s.vertex_ranges(true).1,
            Self::Indices => s
                .sorted_indices
                .map(|(page, at)| (page, at, u64::from(s.translucent) * 6)),
        }
    }
    /// Points the mesh where its range moved.
    pub(super) fn set(self, s: &mut SectionGpu, to: (u32, u64)) {
        match self {
            Self::Section { split } => {
                (s.page, s.start) = to;
                if !split {
                    // Its translucent quads came along, after the solid ones.
                    s.translucent_at = (to.0, to.1 + s.solid_slots);
                }
            }
            Self::Translucent => s.translucent_at = to,
            Self::Indices => s.sorted_indices = Some(to),
        }
    }
}
/// Staging buffer size for section meshes: a frame of uploads while
/// exploring fits in a few.
pub(super) const SECTION_UPLOAD_CHUNK: u64 = 4 << 20;
/// Section copies are submitted early past this many staged bytes, which
/// bounds the staging memory frames that are not drawn can hold.
pub(super) const SECTION_COPY_LIMIT: u64 = 64 << 20;
/// What compaction copies in a frame, `compact_arena` first and
/// `shrink_translucent` with what it left, and a range more where a call's
/// first is larger (`move_off_page`): a page empties over a few frames, with
/// no hitch.
pub(super) const COMPACT_BUDGET: u64 = 32 << 20;
/// A large translucent page starts to shrink (`Renderer::shrink_translucent`)
/// this long after terrain last arrived, and `SHRINK_RETRY` after a try found
/// no room.
/// `Renderer::remove_columns` filters each store once rather than looking up every
/// section key from this many keys, or from as many as there are resident sections.
pub(super) const BULK_REMOVAL_KEYS: usize = 256;
pub(super) const SHRINK_SETTLE: std::time::Duration = std::time::Duration::from_secs(1);
pub(super) const SHRINK_RETRY: std::time::Duration = std::time::Duration::from_secs(5);
/// Not before then may a large translucent page start to shrink.
#[derive(Default)]
pub(super) struct ShrinkGate(Option<Instant>);
impl ShrinkGate {
    /// Not before `wait` from `now`, nor before an earlier wait is over.
    pub(super) fn wait(&mut self, now: Instant, wait: std::time::Duration) {
        self.0 = self.0.max(Some(now + wait));
    }
    /// Whether one may start now: the waits are over, and `compact_arena` copied
    /// nothing this frame (it `left` the whole budget).
    pub(super) fn may_start(&self, now: Instant, left: u64) -> bool {
        left == COMPACT_BUDGET && self.0.is_none_or(|at| now >= at)
    }
}
