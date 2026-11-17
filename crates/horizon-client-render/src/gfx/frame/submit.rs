use super::super::*;

pub(in super::super) struct SubmitFrameArgs<'a, 'b> {
    pub frame: &'a Frame<'b>,
    pub checks: Option<GpuChecks>,
    pub surface_frame: Option<wgpu::SurfaceTexture>,
    pub reconfigure: bool,
    pub readback: Option<(ScreenshotRequest, wgpu::Buffer)>,
    pub target_texture: &'a wgpu::Texture,
    pub fg_frames: Option<(wgpu::Texture, wgpu::Texture)>,
    pub fg_made: bool,
    pub has_section_copies: bool,
    pub started: Instant,
    pub visible: Vec<resident::Visible>,
    pub visible_lods: Vec<lod::Visible>,
    pub shadow_visible: Vec<resident::Visible>,
}

impl Renderer {
    pub(in super::super) fn submit_and_present_frame(
        &mut self,
        mut encoder: wgpu::CommandEncoder,
        args: SubmitFrameArgs<'_, '_>,
    ) -> Result<bool> {
        let SubmitFrameArgs {
            frame,
            checks,
            surface_frame,
            reconfigure,
            readback,
            target_texture,
            fg_frames,
            fg_made,
            has_section_copies,
            started,
            visible,
            visible_lods,
            shadow_visible,
        } = args;

        // The clipped picture for offscreen SDR presentation. Layered screenshots reconstruct
        // their SDR picture in take_readback without changing this or the HDR display targets.
        let sdr_texture = self
            .hdr
            .as_ref()
            .and_then(|h| h.hud())
            .cloned()
            .unwrap_or_else(|| target_texture.clone());
        // Under `HORIZON_FRAMEGEN_DEBUG=generated` a screenshot is the made frame.
        let shown_texture = match &fg_frames {
            Some((_, generated)) if fg_made && frame_gen::generated_only() => generated,
            _ => &sdr_texture,
        };
        let readback = readback.or_else(|| {
            self.screenshot
                .take()
                .and_then(|request| self.take_readback(request, &mut encoder, shown_texture))
        });
        self.uploads.finish(&encoder);
        if has_section_copies {
            self.section_copy_bytes = 0;
            self.section_uploads.finish(&encoder);
        }
        if let Some(image) = surface_frame
            .as_ref()
            .map(|f| &f.texture)
            .or(self.offscreen.as_ref())
        {
            self.readback.present(&self.device, &mut encoder, image);
        }
        // With frame generation on a window, its real frame.
        let shown = surface_frame.as_ref().map(|f| &f.texture);
        let shown = shown.or(fg_frames
            .as_ref()
            .filter(|_| self.surface.is_some())
            .map(|f| &f.0));
        match (&self.hdr, shown) {
            (Some(hdr), Some(texture)) => {
                hdr.present(&self.queue, &mut encoder, texture, self.own_timing.as_mut())
            }
            // Offscreen nothing shows HDR: the target gets the frame as SDR shows it, for
            // splitscreen's compositor.
            (Some(_), None) => {
                if let Some(image) = &self.offscreen {
                    self.readback
                        .draw_onto(&self.device, &mut encoder, &sdr_texture, image);
                }
            }
            (None, _) => {}
        }
        // With HDR on, a full-frame screenshot gets an HDR10 copy beside it.
        let hdr_readback = readback
            .as_ref()
            .filter(|(request, _)| request.size.is_none())
            .zip(self.hdr.as_ref())
            .and_then(|(_, hdr)| {
                let texture = hdr.capture(&self.device, &self.queue, &mut encoder)?;
                let size = [self.config.width, self.config.height];
                Some(copy_frame(&self.device, &mut encoder, &texture, size))
            });
        let first_frame = self.first_frame == FirstFrame::Submit;
        // Finished, submitted and presented on the submit thread when nothing after the
        // submission needs this one: screenshots' readbacks, the error scopes (one per
        // thread), the first frame's startup log, VSync's frame callback, a reconfigure.
        // Frame generation presents its own frames (`frame_gen.present` below).
        // A `screenshot-sequence` frame is submitted here, so it can wait for it to
        // finish.
        let (encoder, surface_frame, after) = if readback.is_none()
            && frame.pack.frame_timing.is_none()
            && fg_frames.is_none()
            && hdr_readback.is_none()
            && checks.is_none()
            && !first_frame
            && !reconfigure
            && self.hdr_marker.is_none()
            && (surface_frame.is_none() || self.config.present_mode != wgpu::PresentMode::Fifo)
        {
            let offscreen = surface_frame.is_none();
            // Occlusion culling's results, GPU culling's statistics and a shader pack's centre
            // depth are read back as soon as the frame is submitted, as here.
            let culling = self.after_submit();
            let center = match &mut self.pack_post {
                Some((_, Some(post))) => post.submitted(),
                _ => None,
            };
            let after = (culling.is_some() || center.is_some()).then(|| {
                Box::new(move || {
                    if let Some(read) = culling {
                        read();
                    }
                    if let Some(read) = center {
                        read();
                    }
                }) as submit::AfterSubmit
            });
            let job = submit::Job {
                encoder,
                surface: surface_frame,
                after,
            };
            match self.queue.submit_later(job) {
                Ok(()) => {
                    self.deferred = Some(offscreen);
                    // The frame's waits for the previous one on the submit thread.
                    self.last_wait += self.queue.take_waited();
                    self.last_encode = started.elapsed().saturating_sub(self.last_wait);
                    self.visible = visible;
                    self.visible_lods = visible_lods;
                    self.shadow_visible = shadow_visible;
                    return Ok(true);
                }
                Err(back) => (back.encoder, back.surface, back.after),
            }
        } else {
            (encoder, surface_frame, None)
        };
        let commands = crate::spans::time("encode", || encoder.finish());
        if first_frame {
            startup_step(format_args!("submitting the first frame"));
            self.first_frame = FirstFrame::Drawn;
        }
        let submission = crate::spans::time("submit", || self.queue.submit([commands]));
        if let Some(read) = after.or_else(|| self.after_submit()) {
            read();
        }
        if let Some(timer) = &mut self.gpu_timer {
            timer.submitted(&self.queue);
            timer.collect(&self.device, &mut self.gpu_times);
        }
        self.pack_timing_submitted();
        if let Some(timer) = &mut self.own_timing {
            timer.submitted(&self.queue);
        }
        if let Some((_, Some(post))) = &mut self.pack_post
            && let Some(read) = post.submitted()
        {
            read();
        }
        let failed = checks.and_then(GpuChecks::pop);
        if let Some((request, buffer)) = readback {
            // Out of memory on the way: the PNG would be black; taken again at half the size.
            if let Some(e) = failed {
                log::warn!(
                    "Horizon renderer: a {}x{} screenshot failed ({e}); trying half the size",
                    self.config.width,
                    self.config.height
                );
                self.screenshot_halvings += 1;
                self.screenshot = Some(request);
            } else {
                self.screenshot_halvings = 0;
                let reader = screenshot_readback::Reader {
                    device: self.device.clone(),
                    config: self.config.clone(),
                };
                if request.background {
                    if let Err(error) =
                        self.screenshot_readbacks
                            .start(reader, request, buffer, hdr_readback)
                    {
                        self.screenshot_result = Some(Err(error.to_string()));
                    }
                } else {
                    self.screenshot_result = Some(
                        reader
                            .read_screenshot(&request, &buffer, hdr_readback.as_ref())
                            .map_err(|e| e.to_string()),
                    );
                }
            }
        }
        if self.screenshot_result.is_none() {
            self.screenshot_result = self.screenshot_readbacks.poll();
        }
        // The frame's own waits for the submit thread (its queue writes and passes).
        self.last_wait += self.queue.take_waited();
        let present = Instant::now();
        if let Some(surface_frame) = surface_frame {
            // On Wayland this ties RedrawRequested to the compositor's frame
            // callbacks, one per refresh. Only VSync (Fifo) waits for the
            // display: with it off, vanilla's window draws uncapped.
            if self.config.present_mode == wgpu::PresentMode::Fifo {
                self.window.pre_present_notify();
            }
            if first_frame {
                startup_step(format_args!("presenting the first frame"));
            }
            self.queue.present(surface_frame);
            if self.hdr.is_some()
                && let Some(marker) = self.hdr_marker.take()
            {
                let _ = std::fs::remove_file(marker);
                log::info!("Horizon renderer: HDR output presented its first frame");
            }
            if reconfigure {
                self.configure_surface();
            }
        } else if let Some((real, generated)) = fg_frames {
            self.present_generated(real, generated, fg_made);
        }
        if self.surface.is_none() {
            // No swapchain paces offscreen frames: keep two on the GPU at most.
            self.in_flight.push_back(submission);
            if !self.paced_elsewhere {
                self.wait_in_flight();
            }
        }
        self.last_wait += present.elapsed();
        // Inside the present's time already.
        self.queue.take_waited();
        self.last_encode = started.elapsed().saturating_sub(self.last_wait);
        let _ = self.started;
        self.visible = visible;
        self.visible_lods = visible_lods;
        self.shadow_visible = shadow_visible;
        Ok(true)
    }
}
