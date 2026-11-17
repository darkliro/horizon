use super::super::*;
use std::time::Duration;

pub(in super::super) struct AcquiredFrame {
    pub fg_frames: Option<(wgpu::Texture, wgpu::Texture)>,
    pub fg: bool,
    pub surface_frame: Option<wgpu::SurfaceTexture>,
    pub reconfigure: bool,
    pub target_texture: wgpu::Texture,
}

impl Renderer {
    pub(in super::super) fn acquire_frame_surface(
        &mut self,
        frame: &Frame<'_>,
        landed: Duration,
    ) -> Result<Option<AcquiredFrame>> {
        if self.first_frame == FirstFrame::Acquire {
            // Once: a minimized window acquires nothing for a while.
            startup_step(format_args!("acquiring the first frame"));
            self.first_frame = FirstFrame::Submit;
        }
        if self.surface_unconfigured
            && let Some(surface) = &self.surface
        {
            // Acquiring from it would panic: configure it again, or skip the
            // frame. A lost device never configures it.
            if self.device_lost.load(std::sync::atomic::Ordering::Relaxed) {
                return Err("the graphics device was lost".into());
            }
            self.frame_gen.stop_to_configure();
            if Self::configure(
                &self.device,
                surface,
                &self.surface_config(),
                self.hdr.as_ref(),
                &self.device_lost,
            )
            .is_err()
            {
                return Ok(None);
            }
            self.surface_unconfigured = false;
        }
        // Frame generation: its present thread acquires and presents, this frame draws
        // into frames of its own. Not over menus that blur the frame, a shader pack's own post
        // passes, HDR output without the HUD's layer (or the float frame HDR was started
        // with) or the title screen.
        let fg = self.frame_gen.on
            && !(self.frame_gen.paused || self.frame_gen.refused)
            && (self.hdr.as_ref()).map_or(
                self.output.frame.format == self.config.format.remove_srgb_suffix(),
                hdr::Hdr::layered,
            )
            && !frame.title
            && frame.menu_blur.is_none()
            && self.pack_post.as_ref().is_none_or(|p| p.1.is_none())
            && self.executor.is_none()
            && (self.surface.is_none()
                || self.config.usage.contains(wgpu::TextureUsages::COPY_DST));
        if !fg {
            self.frame_gen.stop();
        }
        // Waits for the submit thread since the last frame's present (this frame's uploads).
        let queued = self.queue.take_waited();
        let acquire = Instant::now();
        let fg_frames = fg
            .then(|| {
                self.frame_gen.frames(
                    &self.device,
                    &self.queue,
                    self.surface.as_ref(),
                    self.offscreen.as_ref(),
                    &self.window,
                    &self.surface_config(),
                )
            })
            .flatten();
        // The present thread failed: this frame acquires as without frame generation.
        let fg = fg_frames.is_some();
        let acquired = if fg {
            None
        } else {
            self.surface.as_ref().map(|s| s.get_current_texture())
        };
        self.last_wait = landed + queued + acquire.elapsed();
        // Inside the acquire's time already.
        self.queue.take_waited();
        let (surface_frame, reconfigure) = match acquired {
            None => (None, false),
            Some(wgpu::CurrentSurfaceTexture::Success(f)) => (Some(f), false),
            Some(wgpu::CurrentSurfaceTexture::Suboptimal(f)) => (Some(f), true),
            Some(wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded) => {
                return Ok(None);
            }
            Some(wgpu::CurrentSurfaceTexture::Outdated) => {
                self.configure_surface();
                return Ok(None);
            }
            Some(wgpu::CurrentSurfaceTexture::Lost) => {
                self.recreate_surface()?;
                return Ok(None);
            }
            Some(wgpu::CurrentSurfaceTexture::Validation) => {
                // Acquire caught a validation error: the swapchain cannot be
                // used as configured (wgpu reports a configuration raced with
                // a submission this way). Give it a fresh configuration like
                // an Outdated one, and once that has not helped, reattach the
                // window like a Lost one — a fullscreen loading screen then
                // skips frames instead of failing every one of them.
                if self.acquire_validation_failures == 0 {
                    self.configure_surface();
                } else {
                    self.recreate_surface()?;
                }
                self.acquire_validation_failures =
                    self.acquire_validation_failures.saturating_add(1);
                return Ok(None);
            }
        };
        if surface_frame.is_some() {
            self.acquire_validation_failures = 0;
            self.frame_gen.paused = false;
        }
        let size = [self.config.width, self.config.height];
        let target_texture = match (&surface_frame, &mut self.hdr) {
            (_, Some(hdr)) => hdr.frame(&self.device, size),
            _ if let Some((real, _)) = &fg_frames => real.clone(),
            // A screenshot of a swapchain without COPY_SRC (OpenGL), and a float frame
            // with HDR off, draw into a texture of their own, then onto the swapchain or
            // the offscreen target.
            (f, None) => {
                let image = match f {
                    Some(f) => f.texture.clone(),
                    None => self.offscreen.clone().ok_or("no offscreen target")?,
                };
                self.readback
                    .target(
                        &self.device,
                        &image,
                        self.output.frame.format,
                        self.screenshot.is_some(),
                    )
                    .unwrap_or(image)
            }
        };
        Ok(Some(AcquiredFrame {
            fg_frames,
            fg,
            surface_frame,
            reconfigure,
            target_texture,
        }))
    }
}
