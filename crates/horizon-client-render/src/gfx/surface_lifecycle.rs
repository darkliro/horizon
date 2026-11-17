use super::*;

impl Renderer {
    pub fn release_surface(&mut self) {
        self.frame_gen.stop();
        self.queue.wait();
        *self
            .spare_surface
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            self.surface.take().and_then(|s| Arc::try_unwrap(s).ok());
    }
    /// Reattach a window after surface loss or a failed renderer replacement.
    pub fn recreate_surface(&mut self) -> Result<()> {
        self.queue.wait();
        if self.offscreen.is_some() {
            return Ok(());
        }
        // The surface set aside for a replacement that failed, else a new one
        // (Wayland allows only one tearing-control object per window surface,
        // so the old one goes first).
        let spare = self
            .spare_surface
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        self.frame_gen.stop_to_configure();
        self.surface = None;
        let surface = match spare {
            Some(surface) => surface,
            None => self.instance.create_surface(self.window.clone())?,
        };
        let configured = Self::configure(
            &self.device,
            &surface,
            &self.surface_config(),
            self.hdr.as_ref(),
            &self.device_lost,
        );
        if let Err(e) = &configured {
            log::warn!(
                "Horizon renderer: the new window surface did not configure ({e}); retrying"
            );
        }
        self.surface_unconfigured = configured.is_err();
        self.surface = Some(Arc::new(surface));
        self.surface_recreated();
        Ok(())
    }
    /// The world's depth, which occlusion culling's test and post effects may read.
    pub(super) fn depth(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> wgpu::TextureView {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Horizon depth"),
                size: wgpu::Extent3d {
                    width: config.width,
                    height: config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                // TEXTURE_BINDING: a post effect may read the depth buffer.
                // COPY_SRC: improved transparency and a native shader pack's
                // `opaque_depth` copy it.
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
            .create_view(&Default::default())
    }
    /// VSync on presents with Fifo, capped to the display's refresh rate in a
    /// window and in fullscreen alike; Horizon's present-mode setting applies
    /// only with VSync off (`window::chosen_present_mode`).
    pub fn set_present_mode(&mut self, vsync: bool, mode: crate::graphics::PresentMode) {
        use crate::graphics::PresentMode as P;
        use crate::window::{PresentMode, chosen_present_mode};
        let supported: Vec<PresentMode> = self
            .present_modes
            .iter()
            .filter_map(|m| match m {
                wgpu::PresentMode::Fifo => Some(PresentMode::Fifo),
                wgpu::PresentMode::Mailbox => Some(PresentMode::Mailbox),
                wgpu::PresentMode::Immediate => Some(PresentMode::Immediate),
                _ => None,
            })
            .collect();
        let setting = match mode {
            P::Auto => None,
            P::Fifo => Some(PresentMode::Fifo),
            P::Mailbox => Some(PresentMode::Mailbox),
            P::Immediate => Some(PresentMode::Immediate),
        };
        let mode = match chosen_present_mode(&supported, vsync, setting) {
            Some(PresentMode::Mailbox) => wgpu::PresentMode::Mailbox,
            Some(PresentMode::Immediate) => wgpu::PresentMode::Immediate,
            _ => wgpu::PresentMode::Fifo,
        };
        if self.config.present_mode != mode && self.surface.is_some() {
            self.config.present_mode = mode;
            self.configure_surface();
        }
    }
    /// Follows the window to `width` x `height`. A size the swapchain can't take
    /// ([`drawable_size`]) keeps the last one and draws nothing until the next.
    pub fn resize(&mut self, width: u32, height: u32) {
        let max = self.device.limits().max_texture_dimension_2d;
        self.suspended = !drawable_size(width, height, max);
        if self.suspended {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        if self.surface.is_some() {
            self.configure_surface();
        } else {
            self.offscreen = Some(Self::offscreen_target(&self.device, &self.config));
        }
        self.depth = Self::depth(&self.device, &self.config);
        if let Some(pipeline) = &mut self.external {
            pipeline.resize([width, height]);
        }
    }

    /// Horizon's resolution scale (world pixels per window pixel) and FXAA.
    /// Configures `surface` and, while HDR output is on, sends its HDR metadata
    /// again: a new swapchain starts without it. Errs when `surface` was left
    /// unconfigured (a validation error, or a lost device, which wgpu reports
    /// to no error scope).
    pub(super) fn configure(
        device: &wgpu::Device,
        surface: &wgpu::Surface<'_>,
        config: &wgpu::SurfaceConfiguration,
        hdr: Option<&hdr::Hdr>,
        device_lost: &std::sync::atomic::AtomicBool,
    ) -> std::result::Result<(), String> {
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        surface.configure(device, config);
        let memory = pollster::block_on(memory.pop());
        if let Some(e) = memory.or(pollster::block_on(validation.pop())) {
            return Err(e.to_string());
        }
        if device_lost.load(std::sync::atomic::Ordering::Relaxed) {
            return Err("the graphics device was lost".into());
        }
        if let Some(hdr) = hdr {
            horizon_hdr::set_metadata(device, surface, hdr.declared_peak(), hdr.black);
        }
        Ok(())
    }
    /// Configures the window surface again. One that fails is still the one
    /// wgpu configured before, which acquisition reports as a validation error.
    pub(super) fn configure_surface(&mut self) {
        self.frame_gen.stop_to_configure();
        // The submit thread may still present to the old swapchain.
        self.queue.wait();
        if let Some(surface) = &self.surface
            && let Err(e) = Self::configure(
                &self.device,
                surface,
                &self.surface_config(),
                self.hdr.as_ref(),
                &self.device_lost,
            )
        {
            log::warn!("Horizon renderer: the window surface did not configure ({e})");
            self.frame_gen.refused = true;
        }
    }
}
