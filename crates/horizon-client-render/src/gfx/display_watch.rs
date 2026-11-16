//! What the system says about the window's display, kept up to date: on Wayland, the
//! compositor's preferred image description of the surface; elsewhere what wgpu reads
//! from the OS (on Windows, DXGI's luminances and the SDR white level).

use super::{DisplayReading, hdr};
use winit::window::Window;

/// What the system says about the window's display, kept up to date: on Wayland, the
/// compositor's preferred image description of the surface; elsewhere what wgpu reads
/// from the OS (on Windows, DXGI's luminances and the SDR white level).
#[derive(Default)]
pub(super) struct DisplayWatch {
    #[cfg(target_os = "linux")]
    pub(super) wayland: Option<horizon_hdr::wayland::Watch>,
    /// Without Wayland's event, when to read wgpu's values again: the OS sends no event for
    /// a moved window or a changed SDR brightness slider.
    pub(super) polled: Option<std::time::Instant>,
}

impl DisplayReading {
    /// What wgpu reads from the OS for `surface`'s display; without one, `hdr::offscreen_display`.
    pub(super) fn read(surface: Option<&wgpu::Surface>, adapter: &wgpu::Adapter) -> Self {
        surface.map_or_else(hdr::offscreen_display, |s| {
            Self::os(&s.display_hdr_info(adapter))
        })
    }
    pub(super) fn os(info: &wgpu::DisplayHdrInfo) -> Self {
        #[cfg(target_os = "macos")]
        let hdr = macos_hdr(info);
        #[cfg(not(target_os = "macos"))]
        let hdr = info.coarse.and_then(|coarse| coarse.high_dynamic_range);
        Self {
            luminance: os_reported(info),
            hdr,
        }
    }
}
/// Potential EDR is fixed for a screen; current EDR follows brightness and must not
/// repeatedly switch HDR output as the player adjusts the desktop's brightness.
#[cfg(any(target_os = "macos", test))]
pub(super) fn macos_hdr(info: &wgpu::DisplayHdrInfo) -> Option<bool> {
    info.headroom?
        .potential
        .filter(|n| n.is_finite() && *n > 0.0)
        .map(|n| n > 1.0)
}
/// How often [`DisplayWatch`] reads the OS's values again where no event says they changed.
pub(super) const DISPLAY_POLL: std::time::Duration = std::time::Duration::from_secs(2);
impl DisplayWatch {
    /// Starts watching, with what the system reports now.
    pub(super) fn new(
        window: &Window,
        surface: Option<&wgpu::Surface>,
        adapter: &wgpu::Adapter,
    ) -> (Self, DisplayReading) {
        #[cfg(target_os = "linux")]
        if surface.is_some()
            && let Some((wayland, now)) = horizon_hdr::wayland::Watch::new(window)
        {
            let watch = Self {
                wayland: Some(wayland),
                polled: None,
            };
            return (watch, wayland_reading(now));
        }
        let _ = window;
        let now = DisplayReading::read(surface, adapter);
        log_os_reported(now.luminance);
        let watch = Self {
            #[cfg(target_os = "linux")]
            wayland: None,
            polled: Some(std::time::Instant::now() + DISPLAY_POLL),
        };
        (watch, now)
    }
    /// What the system reports at each poll that is due at `time` (`read`; on Wayland, at each
    /// event), whether or not it is `last` again: the surface's capabilities are read with it.
    pub(super) fn poll(
        &mut self,
        last: DisplayReading,
        time: std::time::Instant,
        read: impl FnOnce() -> DisplayReading,
    ) -> Option<DisplayReading> {
        #[cfg(target_os = "linux")]
        if let Some(wayland) = &mut self.wayland {
            return wayland.changed().map(wayland_reading);
        }
        let next = self.polled.as_mut()?;
        if time < *next {
            return None;
        }
        *next = time + DISPLAY_POLL;
        let now = read();
        if now != last {
            log_os_reported(now.luminance);
        }
        Some(now)
    }
}

/// The log of what the system reports about the display: one line per change. The
/// renderer is made anew a few times while the game starts and each asks again, and a
/// slider dragged in the desktop's settings changes the values many times a second, so
/// a change within [`DISPLAY_LOG_QUIET`] of the last line waits until the values have
/// held that long, and only the one they settled on is logged.
struct DisplayLog {
    last: String,
    at: Option<std::time::Instant>,
    pending: Option<(String, std::time::Instant)>,
}
static DISPLAY_LOG: std::sync::Mutex<DisplayLog> = std::sync::Mutex::new(DisplayLog {
    last: String::new(),
    at: None,
    pending: None,
});
const DISPLAY_LOG_QUIET: std::time::Duration = std::time::Duration::from_secs(1);
fn log_display(line: String) {
    let now = std::time::Instant::now();
    let mut log = DISPLAY_LOG
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if line == log.last {
        log.pending = None;
    } else if log.at.is_none_or(|at| now - at >= DISPLAY_LOG_QUIET) {
        log::info!("{line}");
        (log.last, log.at, log.pending) = (line, Some(now), None);
    } else {
        log.pending = Some((line, now));
    }
}
/// Logs a change [`log_display`] held back, once the values have held for
/// [`DISPLAY_LOG_QUIET`].
pub(super) fn flush_display_log() {
    let now = std::time::Instant::now();
    let mut log = DISPLAY_LOG
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some((_, since)) = &log.pending
        && now - *since >= DISPLAY_LOG_QUIET
        && let Some((line, _)) = log.pending.take()
    {
        log::info!("{line}");
        (log.last, log.at) = (line, Some(now));
    }
}
fn log_os_reported(reported: Option<crate::graphics::Reported>) {
    log_display(match reported {
        Some(r) => format!(
            "Horizon renderer: the system reports SDR white at {} nits and a peak of {} nits",
            r.paper_white, r.peak
        ),
        None => "Horizon renderer: the system doesn't report the display's luminance".into(),
    });
}
/// What the OS reports in absolute nits, HDR output's scale (Windows: DXGI's SDR white level,
/// the brightness slider's, as Paper White, and its `MaxLuminance`, the display's or a calibration
/// profile's, as Peak Brightness), kept within the settings' ranges. The player's values win.
pub(super) fn os_reported(info: &wgpu::DisplayHdrInfo) -> Option<crate::graphics::Reported> {
    let luminance = info.luminance?;
    let usable = |n: f32| n.is_finite() && n > 0.0;
    let peak = luminance.max_nits.filter(|&n| usable(n))?;
    let white = luminance.sdr_white_nits.filter(|&n| usable(n))?;
    let paper_white = (white.round() as u32).clamp(80, 480);
    Some(crate::graphics::Reported {
        paper_white,
        // Windows' SDR brightness slider can set white above the reported peak (480 against
        // a calibrated 380): the highlights then stay at white instead of dropping below it.
        peak: (peak.round() as u32).clamp(200, 10_000).max(paper_white),
        peak_guessed: true,
    })
}

/// What a compositor's luminances mean in HDR output's scale.
///
/// KWin maps content's reference white (203 nits in PQ, BT.2408) onto its own, so the
/// desktop's white is 203 there and the peak is the target's maximum over that ratio.
/// Checked on KWin. Hyprland (0.57) reports the standard 203, and shows HDR windows below
/// the SDR white of its desktop (which it doesn't report), so Auto can't match its desktop
/// there. Other compositors are unchecked. The peak is kept within the settings' lowest and
/// PQ's top, so the metadata never ends below the black level.
#[cfg(target_os = "linux")]
fn wayland_reading(now: Option<horizon_hdr::wayland::Luminances>) -> DisplayReading {
    now.map_or_else(DisplayReading::default, |l| DisplayReading {
        luminance: Some(reported(l)),
        // The preferred target's headroom is the compositor's live report, not the
        // swapchain's supported formats. Invalid or absent readings stay unknown.
        hdr: (l.reference.is_finite()
            && l.reference > 0.0
            && l.target_max.is_finite()
            && l.target_max > 0.0)
            .then_some(l.target_max > l.reference),
    })
}
#[cfg(target_os = "linux")]
fn reported(l: horizon_hdr::wayland::Luminances) -> crate::graphics::Reported {
    let peak = (l.target_max * 203.0 / l.reference.max(1.0)).round() as u32;
    log_display(format!(
        "Horizon renderer: the compositor shows this window with reference white {} nits, \
         {}..{} nits: peak {peak} nits in HDR output's scale",
        l.reference, l.target_min, l.target_max,
    ));
    crate::graphics::Reported {
        paper_white: 203,
        peak: peak.clamp(200, 10_000),
        peak_guessed: false,
    }
}
