use super::super::*;

impl Renderer {
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn draw_shadow_passes(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &Frame,
        shadow_view: Option<&external::ShadowView>,
        flat_frame: bool,
        shadow_first: u32,
        shadow_solid: u32,
        shadow_translucent: u32,
        entity_quads: u32,
        shadow_only_quads: u32,
        outline_base: u32,
    ) {
        // One of Horizon's own packs: its shadow passes, from the shadow camera's lists.
        if !frame.title
            && let Some(native) = &mut self.native
        {
            let lists = shadow_view.is_some().then(|| native_shadow::ShadowLists {
                group: &self.group,
                sections: SectionLists {
                    draws: &self.draws,
                    pages: &self.draw_pages,
                    arena: &self.arena,
                    translucent_arena: self.translucent_arena.as_ref(),
                    sorted_indices: &self.sorted_indices,
                    indices: &self.indices,
                    flat: self.flat.as_ref(),
                    multi_draw: self.multi_draw,
                },
                flat_frame,
                instances: &self.instances,
                indirect: &self.indirect,
                draws: [shadow_first, shadow_solid, shadow_translucent],
                entity_group: &self.entity_group,
                entity_vertices: &self.entity_buffer,
                entity_quads,
                shadow_only_quads,
                outline_base,
            });
            native.draw_shadow(native_shadow::ShadowDraws {
                encoder,
                timing: self.own_timing.as_mut(),
                lists,
            });
            let scene = self.ray_scene.as_ref().and_then(|s| s.traced(frame.camera));
            native.before_world(&self.device, encoder, scene.as_ref());
        }
    }
}
