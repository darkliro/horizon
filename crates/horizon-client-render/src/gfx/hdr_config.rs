use super::*;

impl Renderer {
    /// HDR calibration's test picture (`crate::hdr_calibration`) while it runs, else None.
    pub fn set_hdr_calibration(&mut self, calibration: Option<HdrCalibration>) {
        let Some(hdr) = &mut self.hdr else {
            return;
        };
        if hdr.calibration == calibration {
            return;
        }
        let declared = hdr.declared_peak();
        hdr.calibration = calibration;
        if hdr.declared_peak() != declared {
            self.send_hdr_metadata();
        }
    }
    pub(super) fn send_hdr_metadata(&self) {
        if let (Some(surface), Some(hdr)) = (&self.surface, &self.hdr) {
            horizon_hdr::set_metadata(&self.device, surface, hdr.declared_peak(), hdr.black);
        }
    }
    /// This renderer's output as applied, with what its destination offers and reports.
    pub fn output(&self) -> output::EffectiveOutput {
        self.output_with_rays()
    }
    /// What this renderer's display last reported: a renderer made again for the window
    /// starts from it (`Renderer::new`'s `display`).
    pub fn display_reading(&self) -> output::DisplayReading {
        self.reading
    }
    /// Whether HDR output is on: the setting, where the display offers it.
    pub fn hdr_on(&self) -> bool {
        self.output.hdr_on()
    }
    /// The F3 screen's HDR line while the setting is on: the swapchain HDR presents to with
    /// its paper white and peak, or that the surface offers none.
    pub fn hdr_line(&self) -> String {
        match &self.hdr {
            Some(hdr) => format!(
                "HDR: {}, {}/{} nits",
                hdr.output.name(),
                hdr.paper_white.round(),
                hdr.peak.round()
            ),
            None => "HDR: not offered by this display".into(),
        }
    }
    /// Whether the Look setting is Horizon's (`Hdr::look`).
    pub fn set_hdr_lighting(&mut self, on: bool) {
        self.hdr_lighting_setting = on;
        if let Some(hdr) = &mut self.hdr {
            hdr.look = on;
        }
    }
    /// Measures which pixels of each block sprite may shine (`hdr_stats`), at the current
    /// masks. Only with HDR output on: nothing else reads them.
    pub fn measure_emission(&mut self) {
        if let Some(hdr) = &self.hdr {
            let views = sprite_views(&self.sprite_textures, &self.sprite_unused, None);
            self.emission_stats
                .measure(&self.device, &self.queue, &views, hdr.masks);
        }
    }
}
