use super::super::*;

impl Renderer {
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn draw_sky_pass_content<F>(
        &self,
        pass: &mut pack_timing::CountedPass<'_>,
        frame: &Frame,
        sky_visible: bool,
        pack_draws_sky: bool,
        native_sky_pass: bool,
        celestial_vertices: usize,
        pack_draws_celestial: bool,
        hdr_gain: bool,
        mut pack_frame: F,
    ) where
        F: FnMut(&mut wgpu::RenderPass<'_>),
    {
        if frame.title || (sky_visible && (frame.skybox || frame.end_sky)) {
            let native_sky = !frame.title
                && frame.skybox
                && !pack_draws_sky
                && !native_sky_pass
                && self.native.as_ref().is_some_and(|n| n.draw_sky(pass));
            if native_sky {
                // The sky binds the frame's own group; a material stage that traces draws
                // with the scene's.
                pack_frame(pass);
            }
            if !frame.title && frame.skybox && !pack_draws_sky && !native_sky_pass && !native_sky {
                match self.core_sky.as_ref().filter(|c| c.draws_disc()) {
                    Some(core) => {
                        core.draw_disc(pass);
                        pass.set_bind_group(0, &self.group, &[]);
                    }
                    None => {
                        pass.set_pipeline(&self.sky_base);
                        pass.draw(0..3, 0..1);
                    }
                }
                if frame.sky.sunrise_color[3] > 0.001 {
                    pass.set_pipeline(&self.sunrise_pipeline);
                    pass.set_vertex_buffer(0, self.sunrise_buffer.slice(..));
                    pass.draw(0..sunrise_fan::VERTICES as u32, 0..1);
                }
                // OptiFine custom sky layers, over the disc and sunrise and under the
                // sun, moon and stars.
                if self.custom_sky.is_drawing() {
                    self.custom_sky.draw(pass);
                    pass.set_bind_group(0, &self.group, &[]);
                }
            }
            if frame.title {
                self.panorama.draw(pass);
                pass.set_bind_group(0, &self.group, &[]);
            } else if celestial_vertices > 0 || frame.end_sky {
                pass.set_pipeline(&self.celestial.added);
                pass.set_bind_group(0, &self.entity_group, &[]);
                pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
                if frame.end_sky {
                    self.end_sky.draw(pass);
                    // The End's custom sky layers go over its sky box.
                    if self.custom_sky.is_drawing() && !pack_draws_sky && self.executor.is_none() {
                        self.custom_sky.draw(pass);
                        pass.set_pipeline(&self.celestial.added);
                        pass.set_bind_group(0, &self.entity_group, &[]);
                    }
                }
                if celestial_vertices > 0 && !pack_draws_celestial {
                    // The Overworld's bodies are the sun's quad and the moon's
                    // (`Game::celestial_geometry`): a pack sky drawing its own skips one.
                    let own = |hides: fn(&external::Capabilities) -> bool| {
                        !frame.end_sky && self.executor.as_ref().is_some_and(hides)
                    };
                    let own_sun = own(|c| c.hides_sun);
                    let own_moon = own(|c| c.hides_moon);
                    // A resource pack's core/sky isn't vanilla's sky times the sky's
                    // lift, which the sun's disc pass assumes: today's add there.
                    let lifted = hdr_gain
                        && self.hdr.as_ref().is_some_and(|h| h.lighting)
                        && !self.core_sky.as_ref().is_some_and(|c| c.draws_disc());
                    let quads = &frame.celestial[..celestial_vertices];
                    let bodies = celestial::bodies(quads, own_sun, own_moon);
                    pass.set_vertex_buffer(0, self.celestial_buffer.slice(..));
                    self.celestial.draw_bodies(pass, lifted, bodies);
                }
                pass.set_bind_group(0, &self.group, &[]);
            }
            let own_stars = self.executor.is_some_and(|c| c.hides_stars);
            if !frame.title && frame.stars.is_some() && !pack_draws_sky && !own_stars {
                match self.core_sky.as_ref().filter(|c| c.draws_stars()) {
                    Some(core) => core.draw_stars(pass),
                    None => self.stars.draw(pass),
                }
                pass.set_bind_group(0, &self.group, &[]);
            }
            if !frame.title && frame.dark_disc && !pack_draws_sky {
                pass.set_pipeline(&self.dark_disc);
                pass.draw(0..3, 0..1);
            }
        }
    }
}
