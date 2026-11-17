//! GPU representation and fading state of world sections.

use super::{
    FadeInputs, SectionKey, gpu_culling, lod, packed, resident, section_fade, translucency,
};

use crate::mesher::{SectionMesh, Sides};
use glam::Vec3;
use section_fade::{section_instance, section_visibility};
use std::time::Instant;

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

/// Extracts the animated sprite layers present in a section mesh. Meshing
/// records the distinct layers, so uploading does not rescan every vertex.
pub(crate) fn collect_animated_layers(
    mesh: &SectionMesh,
    is_animated: impl Fn(u16) -> bool,
) -> Box<[u16]> {
    mesh.sprite_layers
        .iter()
        .copied()
        .filter(|&layer| is_animated(layer))
        .collect()
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
            Self::Section { .. } => {
                s.page = to.0;
                s.start = to.1;
            }
            Self::Translucent => s.translucent_at = to,
            Self::Indices => s.sorted_indices = Some(to),
        }
    }
}

/// Starts moving a page of a large arena far larger than its ranges in use
/// (`Arena::oversized`) into a new page of the size they get (`Arena::open_page`),
/// for `Renderer::shrink_translucent`: the page closes and `move_off_page` empties
/// it. None: no page to shrink. False, and nothing changes: no video memory for
/// the new page, or a range there without its mesh in `meshes`, which would keep
/// the page from ever emptying.
pub(super) fn begin_shrink<'a>(
    arena: &mut super::arena::Arena,
    device: &wgpu::Device,
    meshes: impl Iterator<Item = &'a mut SectionGpu>,
    part: Part,
) -> Option<bool> {
    let (page, capacity) = arena.oversized()?;
    let held: u64 = meshes
        .filter_map(|s| part.of(s).filter(|r| r.0 == page))
        .map(|(.., n)| super::arena::Arena::span(n))
        .sum();
    debug_assert_eq!(held, arena.used(page), "a range in use without its mesh");
    if held != arena.used(page) || !arena.open_page(device, capacity) {
        return Some(false);
    }
    arena.close(page);
    Some(true)
}

/// GPU culling's flag (`SectionGpu::gpu_fading`) for its recorded section `s`
/// at `key`, decided at `f.now`: whether it draws faded (`drawn_faded`), and if
/// time ends that, its end waited for in `fades` (`fade_end`).
pub(super) fn decide_fade(
    s: &SectionGpu,
    key: SectionKey,
    fades: &mut gpu_culling::Fades,
    f: FadeInputs<'_>,
) -> bool {
    let fading = s.drawn_faded(key, f);
    if let Some(end) = s.fade_end(key, f).filter(|_| fading) {
        fades.push(end, key);
    }
    fading
}

/// GPU culling: the section at `key` began to fade out over far terrain
/// (`Renderer::leave_section`), or is held while far terrain comes in
/// (`Renderer::lod_retiring`). It draws faded until it goes or is meshed again,
/// so its record is flagged and the CPU draws it, with no end to wait for.
pub(super) fn hold_fading(
    sections: &mut resident::Resident<SectionGpu>,
    records: &mut gpu_culling::Records,
    key: SectionKey,
) {
    let Some(s) = sections.find_mut(key).filter(|s| !s.gpu_fading) else {
        return;
    };
    let Some(id) = s.gpu_id else {
        return;
    };
    s.gpu_fading = true;
    records.set_fading(id, true);
    sections.set_marked(key, true);
}

/// GPU culling switched off (`Renderer::set_gpu_culling`): no section keeps a
/// record or a flag, and those it flagged get back the walk marks they have
/// while it is off (`SectionGpu::listed`). `rows`: the level's sections.
pub(super) fn forget_records(sections: &mut resident::Resident<SectionGpu>, rows: usize) {
    let mut marks = Vec::new();
    for (key, s, _) in sections.iter_mut() {
        s.gpu_id = None;
        if std::mem::take(&mut s.gpu_fading) {
            marks.push((key, s.listed(key, rows)));
        }
    }
    for (key, marked) in marks {
        sections.set_marked(key, marked);
    }
}

/// GPU culling's fading sections brought up to `f.now`, before a frame's gate
/// and `prepare`: every recorded section anew when the flags were decided with
/// another chunk fade-in time (GPU culling just switched on, or Video Settings
/// changed it), else those whose fade was due to end, handed back to the GPU if
/// whole by then. A section's record (`FADING`) and its walk mark follow its
/// `gpu_fading`. `rows`: the level's sections.
pub(super) fn mark_fades(
    sections: &mut resident::Resident<SectionGpu>,
    records: &mut gpu_culling::Records,
    fades: &mut gpu_culling::Fades,
    rows: usize,
    f: FadeInputs<'_>,
) {
    let mut marks = Vec::new();
    if fades.fade_ms != Some(f.fade_ms) {
        fades.clear();
        fades.fade_ms = Some(f.fade_ms);
        for (key, s, _) in sections.iter_mut() {
            let Some(id) = s.gpu_id else {
                continue;
            };
            let fading = decide_fade(s, key, fades, f);
            if fading != s.gpu_fading {
                s.gpu_fading = fading;
                records.set_fading(id, fading);
                marks.push((key, s.listed(key, rows)));
            }
        }
    } else {
        while let Some(key) = fades.pop_due(f.now) {
            // Gone since, or no longer flagged.
            let Some(s) = sections.find_mut(key).filter(|s| s.gpu_fading) else {
                continue;
            };
            // Still fading: meshed again since, with its own end waited for
            // (`decide_fade`), or leaving or held, with none.
            if s.drawn_faded(key, f) {
                continue;
            }
            s.gpu_fading = false;
            records.set_fading(s.gpu_id.expect("flagged sections have records"), false);
            marks.push((key, s.listed(key, rows)));
        }
    }
    for (key, marked) in marks {
        sections.set_marked(key, marked);
    }
}
