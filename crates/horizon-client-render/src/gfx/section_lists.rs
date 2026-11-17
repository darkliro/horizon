//! World section draw lists and arena bindings.

use super::{arena, flat};

/// Whether Translucent Batching is offered (`graphics::offer_translucent_batching`): where
/// indirect multi-draws (`multi_draw`) take its batches, but not OpenGL's, which loop over
/// their entries on the CPU (and without them every section is a call of its own anyway).
/// Frames drawn flat (Metal, Intel Gen11/Gen12's Vulkan driver) take it too: their
/// translucent calls then come one per page of its arena rather than about one per section.
pub(super) fn translucent_batchable(info: &wgpu::AdapterInfo, multi_draw: bool) -> bool {
    multi_draw && info.backend != wgpu::Backend::Gl
}

/// The vertices flat draws bind for a call's page key: a page of the section arena, or of
/// the translucent arena up to its last range in use (`flat::TRANSLUCENT_KEYS`).
pub(super) fn flat_vertices<'a>(
    arena: &'a arena::Arena,
    translucent: Option<&'a arena::Arena>,
) -> impl Fn(u32) -> wgpu::BufferSlice<'a> + Copy {
    move |key| match (translucent, key.checked_sub(flat::TRANSLUCENT_KEYS)) {
        (Some(translucent), Some(page)) => translucent.binding(page),
        _ => arena.buffer(key).slice(..),
    }
}

/// The world's section lists and the arena pages they draw from, borrowed apart from the rest
/// of the renderer: a shader pack's shadow passes draw the light's lists through them.
#[derive(Clone, Copy)]
pub(super) struct SectionLists<'a> {
    pub(super) draws: &'a [[u32; 5]],
    pub(super) pages: &'a [(u32, Option<u32>)],
    pub(super) arena: &'a arena::Arena,
    pub(super) translucent_arena: Option<&'a arena::Arena>,
    pub(super) sorted_indices: &'a arena::Arena,
    /// The quad indices.
    pub indices: &'a wgpu::Buffer,
    /// The flat draws' lists, where they are made.
    pub flat: Option<&'a flat::Flat>,
    pub(super) multi_draw: bool,
}

impl<'a> SectionLists<'a> {
    /// `Renderer::draw_sections_from`.
    pub(super) fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        indirect: &wgpu::Buffer,
        base: u32,
        first: u32,
        count: u32,
        pulled: bool,
    ) {
        let (draws, pages) = (self.draws, self.pages);
        let (mut i, end) = (first as usize, (first + count) as usize);
        while i < end {
            let (page, index_page) = pages[i];
            let run = pages[i..end]
                .iter()
                .take_while(|&&p| p == (page, index_page))
                .count();
            // Translucent draws, and only they, have sorted indices; split, their large
            // pages are bound up to their last range in use (`Arena::binding`).
            let (vertices, indices) = match (self.translucent_arena, index_page) {
                (Some(arena), Some(p)) => (arena.binding(page), self.sorted_indices.binding(p)),
                (_, p) => (
                    self.arena.buffer(page).slice(..),
                    p.map_or(self.indices, |p| self.sorted_indices.buffer(p))
                        .slice(..),
                ),
            };
            pass.set_vertex_buffer(0, vertices);
            if pulled
                && index_page.is_none()
                && let Some(group) = self.flat.and_then(|f| f.page_group(page))
            {
                pass.set_bind_group(1, group, &[]);
            }
            pass.set_index_buffer(indices, wgpu::IndexFormat::Uint32);
            if self.multi_draw {
                let offset = (i - base as usize) as u64 * 20;
                pass.multi_draw_indexed_indirect(indirect, offset, run as u32);
            } else {
                for d in &draws[i..i + run] {
                    pass.draw_indexed(d[2]..d[2] + d[0], d[3] as i32, d[4]..d[4] + 1);
                }
            }
            i += run;
        }
        if pages[first as usize..end]
            .iter()
            .any(|(_, sorted)| sorted.is_some())
        {
            pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
        }
    }

    /// `flat_vertices` of these arenas.
    pub(super) fn flat_vertices(&self) -> impl Fn(u32) -> wgpu::BufferSlice<'a> + Copy {
        flat_vertices(self.arena, self.translucent_arena)
    }
}
