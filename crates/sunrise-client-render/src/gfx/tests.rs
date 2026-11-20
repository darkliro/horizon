//! Unit and integration test suites for gfx rendering, culling, and pipelines.

#[test]
fn taa_culling_covers_jittered_view_edges_but_still_rejects_behind_the_camera() {
    use super::*;

    let mut outside_unjittered = 0;
    for size in [[1, 1], [320, 180], [1280, 720], [3840, 2160]] {
        for (yaw, pitch) in [(0.0, 0.0), (73.0, -35.0), (-142.0, 79.0)] {
            let pose = Mat4::from_translation(Vec3::new(0.03, -0.02, 0.01))
                * Mat4::from_scale(Vec3::new(1.1, 0.9, 1.0))
                * crate::camera::view_matrix(yaw, pitch);
            let matrix = world_projection(70.0, 16.0 / 9.0, 512.0) * pose;
            let guarded = taa_frustum(matrix, size);
            let ordinary = Frustum::from_planes(frustum(matrix));
            for x in [-1.0, 1.0] {
                for y in [-1.0, 1.0] {
                    let jitter = Mat4::from_translation(Vec3::new(
                        x / size[0] as f32,
                        y / size[1] as f32,
                        0.0,
                    ));
                    let inverse = (jitter * matrix).inverse();
                    for clip_x in [-1.0, 0.0, 1.0] {
                        for clip_y in [-1.0, 0.0, 1.0] {
                            for depth in [0.001, 0.5, 0.999] {
                                // Independent oracle: a point on an actual jittered
                                // viewport edge, transformed back into world space.
                                let point = inverse * Vec4::new(clip_x, clip_y, depth, 1.0);
                                let point = point.truncate() / point.w;
                                assert!(guarded.intersects_sphere(point, 1.0e-4));
                                let slack = Vec3::splat(1.0e-4);
                                assert!(guarded.intersects(point - slack, point + slack));
                                if !ordinary.intersects_sphere(point, 0.0) {
                                    outside_unjittered += 1;
                                }
                            }
                        }
                    }
                }
            }
            let behind = pose.inverse().transform_point3(Vec3::new(0.0, 0.0, 10.0));
            assert!(!guarded.intersects_sphere(behind, 0.3));
        }
    }
    assert!(
        outside_unjittered > 0,
        "the test must need the jitter guard"
    );
}

#[test]
fn depth_culling_keeps_jittered_edges_and_depth_crossing_models() {
    use super::*;
    for (yaw, pitch) in [(0.0, 0.0), (73.0, -35.0), (-142.0, 79.0)] {
        let pose = Mat4::from_translation(Vec3::new(0.03, -0.02, 0.01))
            * crate::camera::view_matrix(yaw, pitch);
        let matrix = world_projection(70.0, 16.0 / 9.0, 512.0) * pose;
        let view = Frustum::from_planes(frustum(matrix));
        for size in [[1, 1], [320, 180], [1280, 720], [3840, 2160]] {
            for (x, y) in [(-1.0, 1.0), (1.0, -1.0)] {
                let jitter =
                    Mat4::from_translation(Vec3::new(x / size[0] as f32, y / size[1] as f32, 0.0));
                let inverse = (jitter * matrix).inverse();
                for (x, y) in [(-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0)] {
                    for z in [0.001, 0.5, 0.999] {
                        let p = inverse * Vec4::new(x, y, z, 1.0);
                        let p = p.truncate() / p.w;
                        assert!(
                            view.intersects_depth(p - Vec3::splat(1.0e-4), p + Vec3::splat(1.0e-4))
                        );
                    }
                }
            }
        }
        let point = |p| pose.inverse().transform_point3(p);
        for z in [10.0, -600.0] {
            let p = point(Vec3::new(0.0, 0.0, z));
            assert!(!view.intersects_depth(p - Vec3::splat(0.1), p + Vec3::splat(0.1)));
        }
        // An oversized model straddling the eye must stay, even with its origin behind.
        let p = point(Vec3::ZERO);
        assert!(view.intersects_depth(p - Vec3::splat(10.0), p + Vec3::splat(10.0)));
        assert!(view.intersects_depth(Vec3::splat(f32::NAN), Vec3::ONE));
    }
}
#[test]
fn opaque_gizmo_lines_upload_first_for_the_depth_writing_pipeline() {
    let line = |argb: u32| [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, f32::from_bits(argb), 2.5];
    // A hitbox's white, the chunk border's outer red columns (alpha 0x7F),
    // and the chunk border's yellow.
    let lines = [line(0xFFFF_FFFF), line(0x7FFF_0000), line(0xFFFF_FF00)];
    assert!(super::debug_line_opaque(&lines[0]));
    assert!(!super::debug_line_opaque(&lines[1]));
    assert!(!super::debug_line_opaque(&line(0xFE00_0000)));
    let order: Vec<u32> = super::debug_lines_opaque_first(&lines)
        .map(|l| l[6].to_bits())
        .collect();
    assert_eq!(order, [0xFFFF_FFFF, 0xFFFF_FF00, 0x7FFF_0000]);
}
use crate::gfx::static_entity_shader;

fn adapter(device_type: wgpu::DeviceType) -> wgpu::AdapterInfo {
    wgpu::AdapterInfo {
        name: "test adapter".into(),
        vendor: 0,
        device: 0,
        device_type,
        device_pci_bus_id: String::new(),
        driver: String::new(),
        driver_info: String::new(),
        backend: wgpu::Backend::Vulkan,
        subgroup_min_size: 0,
        subgroup_max_size: 0,
        transient_saves_memory: None,
        limit_bucket: None,
    }
}

#[test]
fn software_occlusion_culling_is_opt_in() {
    let cpu = adapter(wgpu::DeviceType::Cpu);
    let discrete = adapter(wgpu::DeviceType::DiscreteGpu);
    assert!(!super::occlusion_culling_requested(&cpu, None));
    assert!(super::occlusion_culling_requested(&discrete, None));
    assert!(super::occlusion_culling_requested(&cpu, Some("all")));
    assert!(!super::occlusion_culling_requested(&discrete, Some("0")));
}

/// Translucent Batching is offered with indirect multi-draws that aren't OpenGL's, flat
/// frames (Metal, Intel Gen11/Gen12 on Vulkan) included.
#[test]
fn translucent_batching_is_offered_with_indirect_multi_draws_except_on_opengl() {
    let vulkan = adapter(wgpu::DeviceType::DiscreteGpu);
    let mut metal = adapter(wgpu::DeviceType::IntegratedGpu);
    metal.backend = wgpu::Backend::Metal;
    let mut gl = adapter(wgpu::DeviceType::DiscreteGpu);
    gl.backend = wgpu::Backend::Gl;
    let batchable = super::translucent_batchable;
    assert!(batchable(&vulkan, true));
    assert!(batchable(&metal, true));
    assert!(!batchable(&vulkan, false));
    assert!(!batchable(&gl, true));
}

/// GPU culling is offered where the device draws counts the GPU wrote and runs compute
/// shaders with storage buffers enough, where Translucent Batching is (indirect
/// multi-draws that aren't OpenGL's) and vanilla frames are never drawn flat, and never
/// on Intel Gen11/Gen12's Vulkan driver, whose draws read instance-rate origins from the
/// wrong first instance, even with flat draws off there, nor on DX12 with flat page
/// groups, whose count draws leave the base vertex out of the pulled pipelines'
/// `vertex_index`.
#[test]
fn gpu_culling_is_offered_where_counts_are_drawn_and_vanilla_never_draws_flat() {
    let vulkan = adapter(wgpu::DeviceType::DiscreteGpu);
    let mut gl = adapter(wgpu::DeviceType::DiscreteGpu);
    gl.backend = wgpu::Backend::Gl;
    let mut dx12 = adapter(wgpu::DeviceType::DiscreteGpu);
    dx12.backend = wgpu::Backend::Dx12;
    let mut intel = adapter(wgpu::DeviceType::IntegratedGpu);
    (intel.vendor, intel.device) = (0x8086, 0x9a49); // Tiger Lake
    let counts = wgpu::Features::MULTI_DRAW_INDIRECT_COUNT;
    let compute = wgpu::DownlevelFlags::COMPUTE_SHADERS;
    let storage = |buffers| wgpu::Limits {
        max_storage_buffers_per_shader_stage: buffers,
        ..Default::default()
    };
    let offered = |info, features, downlevel, buffers, multi_draw, flat| {
        super::gpu_culling::supported(
            info,
            features,
            downlevel,
            &storage(buffers),
            multi_draw,
            flat,
        )
    };
    // No flat draws, or flat draws for a shader pack's terrain alone.
    assert!(offered(&vulkan, counts, compute, 8, true, None));
    assert!(offered(&vulkan, counts, compute, 8, true, Some(usize::MAX)));
    // Vanilla frames drawn flat past MIN_SECTIONS (Metal), or all of them.
    let min = super::flat::MIN_SECTIONS;
    assert!(!offered(&vulkan, counts, compute, 8, true, Some(min)));
    assert!(!offered(&vulkan, counts, compute, 8, true, Some(0)));
    let none = wgpu::Features::empty();
    assert!(!offered(&vulkan, none, compute, 8, true, None));
    let no_compute = wgpu::DownlevelFlags::empty();
    assert!(!offered(&vulkan, counts, no_compute, 8, true, None));
    assert!(!offered(&vulkan, counts, compute, 7, true, None));
    assert!(!offered(&vulkan, counts, compute, 8, false, None));
    assert!(!offered(&gl, counts, compute, 8, true, None));
    assert!(!offered(&intel, counts, compute, 8, true, None));
    assert!(offered(&dx12, counts, compute, 8, true, None));
    assert!(!offered(&dx12, counts, compute, 8, true, Some(usize::MAX)));
    // The settings offer it there, but not on a software adapter (WARP, lavapipe),
    // whose pass would run on the CPU, and they say why not on DX12 with flat page groups.
    use crate::graphics::GpuCullingOffer as O;
    let offer = super::gpu_culling::offer;
    assert_eq!(offer(&vulkan, true, true), O::Offered);
    assert_eq!(
        offer(&adapter(wgpu::DeviceType::IntegratedGpu), true, false),
        O::Offered
    );
    assert_eq!(
        offer(&adapter(wgpu::DeviceType::Cpu), true, false),
        O::Software
    );
    assert_eq!(offer(&vulkan, false, false), O::Unsupported);
    assert_eq!(
        offer(&adapter(wgpu::DeviceType::Cpu), false, false),
        O::Unsupported
    );
    assert_eq!(offer(&dx12, true, false), O::Offered);
    assert_eq!(offer(&dx12, false, true), O::Dx12);
}

/// The pack-flat rule: with Translucent Batching translucent quads have an arena of
/// their own, unless a shader pack's terrain programs or its shadow pass alone can
/// draw flat (a pack picked and not loaded yet counts as drawing terrain).
#[test]
fn translucent_quads_stay_in_the_section_arena_while_a_pack_can_draw_flat() {
    // (batching, flat draws, the pack's terrain, its shadow pass)
    let split = super::translucent_split;
    assert!(split(true, true, false, false));
    assert!(!split(true, true, true, false));
    assert!(!split(true, true, false, true));
    assert!(!split(true, true, true, true));
    // Without flat draws, or with them off, the pack draws per section, from either.
    assert!(split(true, false, true, true));
    assert!(!split(false, true, false, false));
    assert!(!split(false, false, false, false));
}

/// No large translucent page starts to shrink until `SHRINK_SETTLE` after terrain
/// last arrived, nor in a frame `compact_arena` copied in; terrain arriving after
/// a try that found no room does not cut its `SHRINK_RETRY` short. Only starts
/// wait: a move under way finishes, or its page would stay closed while the new
/// one fills.
#[test]
fn shrinking_starts_once_terrain_has_settled() {
    use super::{COMPACT_BUDGET, SHRINK_RETRY, SHRINK_SETTLE, ShrinkGate};
    use std::time::{Duration, Instant};
    let (now, ms) = (Instant::now(), Duration::from_millis(1));
    let mut gate = ShrinkGate::default();
    assert!(gate.may_start(now, COMPACT_BUDGET));
    assert!(!gate.may_start(now, COMPACT_BUDGET - 1));
    gate.wait(now, SHRINK_SETTLE);
    assert!(!gate.may_start(now + SHRINK_SETTLE - ms, COMPACT_BUDGET));
    assert!(gate.may_start(now + SHRINK_SETTLE, COMPACT_BUDGET));
    gate.wait(now, SHRINK_RETRY);
    gate.wait(now + ms, SHRINK_SETTLE);
    assert!(!gate.may_start(now + SHRINK_RETRY - ms, COMPACT_BUDGET));
    assert!(gate.may_start(now + SHRINK_RETRY, COMPACT_BUDGET));
}

#[test]
fn the_mirror_sees_each_point_where_its_reflection_is() {
    use glam::{Mat4, Vec3, Vec4};
    let view_proj = super::world_projection(70.0, 1.5, 512.0)
        * Mat4::look_to_rh(Vec3::ZERO, Vec3::new(0.3, -0.4, -1.0), Vec3::Y);
    let plane = -3.25;
    let mirror = super::view::mirror_view_projection(view_proj, plane);
    for p in [Vec3::new(1.0, 2.0, -9.0), Vec3::new(-4.0, -1.5, -20.0)] {
        let reflected = Vec3::new(p.x, 2.0 * plane - p.y, p.z);
        let a = mirror * Vec4::from((p, 1.0));
        let b = view_proj * Vec4::from((reflected, 1.0));
        assert!((a - b).abs().max_element() < 1e-4, "{a} {b}");
    }
}
#[test]
fn the_mirrors_sky_rays_are_the_cameras_reflected() {
    use glam::{Mat4, Vec3, Vec4};
    // A camera looking down: its mirror looks up, by as much.
    let forward = Vec3::new(0.3, -0.64, -1.0).normalize();
    let view = Mat4::look_to_rh(Vec3::ZERO, forward, Vec3::Y);
    let mirrored = super::view::mirror_inverse_view(view.inverse().to_cols_array_2d(), -3.25);
    let ray = Mat4::from_cols_array_2d(&mirrored) * Vec4::new(0.0, 0.0, -1.0, 0.0);
    let reflected = Vec3::new(forward.x, -forward.y, forward.z);
    assert!((ray.truncate() - reflected).length() < 1e-5, "{ray}");
    // The view's origin, the camera, goes to the mirror's eye.
    let eye = Mat4::from_cols_array_2d(&mirrored) * Vec4::W;
    assert!(
        (eye.truncate() - Vec3::new(0.0, -6.5, 0.0)).length() < 1e-5,
        "{eye}"
    );
}
#[test]
fn scaled_screenshots_follow_resolution_scale_and_quality_within_limits() {
    use super::screenshot::scaled_screenshot_factor as factor;
    let big = u64::MAX / 2;
    // The window's size at 100% and 1x; Resolution Scale below 100% stays the window's.
    assert_eq!(factor([1920, 1080], 1.0, 1, 16384, big), 1.0);
    assert_eq!(factor([1920, 1080], 0.5, 1, 16384, big), 1.0);
    // The game's resolution, times the quality.
    assert_eq!(factor([1920, 1080], 2.0, 1, 16384, big), 2.0);
    assert_eq!(factor([1920, 1080], 1.5, 4, 16384, big), 6.0);
    assert_eq!(factor([1920, 1080], 0.5, 8, 16384, big), 8.0);
    // Cut to the largest texture side: 3840 * 4.266 = 16384.
    let f = factor([3840, 2160], 1.0, 8, 16384, big);
    assert!(
        (3840.0 * f) as u32 <= 16384 && (3840.0 * f) as u32 > 16300,
        "{f}"
    );
    // Cut to the largest buffer: 256 MiB holds about 3.8x of 1080p's readback.
    let max = 256 << 20;
    let f = factor([1920, 1080], 1.0, 8, 16384, max);
    let [w, h] = [1920, 1080].map(|s| (s as f32 * f) as u32);
    assert!(
        u64::from(super::screenshot::padded_row(w)) * u64::from(h) <= max,
        "{f}"
    );
    assert!(f > 3.7, "{f}");
}
#[test]
fn drawable_size_skips_minimized_and_oversized_windows() {
    use super::drawable_size;
    assert!(drawable_size(1280, 720, 8192));
    assert!(drawable_size(1, 1, 8192));
    assert!(drawable_size(8192, 8192, 8192));
    // Minimized.
    assert!(!drawable_size(0, 720, 8192));
    assert!(!drawable_size(1280, 0, 8192));
    assert!(!drawable_size(0, 0, 8192));
    // One past the device's largest texture.
    assert!(!drawable_size(8193, 720, 8192));
    assert!(!drawable_size(1280, 8193, 8192));
    // The crash: a window `SetWindowPos` made 65535 pixels tall, on an RTX 5080 (32768).
    assert!(!drawable_size(984, 65496, 32768));
    assert!(!drawable_size(984, u32::MAX, 32768));
}

#[test]
fn os_luminances_become_auto_values() {
    use super::os_reported;
    use crate::graphics::Reported;
    let info = |max, white| wgpu::DisplayHdrInfo {
        luminance: Some(wgpu::DisplayLuminance {
            max_nits: max,
            sdr_white_nits: white,
            ..Default::default()
        }),
        ..Default::default()
    };
    // Windows' SDR content brightness slider runs 80..480 nits.
    assert_eq!(
        os_reported(&info(Some(998.4), Some(240.0))),
        Some(Reported {
            paper_white: 240,
            peak: 998,
            peak_guessed: true,
        })
    );
    assert_eq!(
        os_reported(&info(Some(120.0), Some(1000.0))),
        Some(Reported {
            paper_white: 480,
            peak: 480,
            peak_guessed: true,
        })
    );
    // White above the calibrated peak (a Samsung QD-OLED's 380, the slider at 456).
    assert_eq!(
        os_reported(&info(Some(380.0), Some(456.0))),
        Some(Reported {
            paper_white: 456,
            peak: 456,
            peak_guessed: true,
        })
    );
    assert_eq!(os_reported(&info(Some(1000.0), None)), None);
    assert_eq!(os_reported(&info(Some(f32::NAN), Some(80.0))), None);
    assert_eq!(os_reported(&wgpu::DisplayHdrInfo::default()), None);
}
#[test]
fn system_hdr_changes_are_kept_without_luminance_changes() {
    let reading = |hdr| {
        super::DisplayReading::os(&wgpu::DisplayHdrInfo {
            coarse: Some(wgpu::DisplayCoarseRange {
                high_dynamic_range: hdr,
                ..Default::default()
            }),
            headroom: Some(wgpu::DisplayHeadroom {
                potential: hdr.map(|on| if on { 2.0 } else { 1.0 }),
                ..Default::default()
            }),
            ..Default::default()
        })
    };
    let off = reading(Some(false));
    let on = reading(Some(true));
    let unknown = reading(None);
    assert_eq!(on.hdr, Some(true));
    assert_eq!(off.hdr, Some(false));
    assert_eq!(unknown.hdr, None);
    assert_eq!(on.luminance, None); // macOS reports EDR headroom, not absolute nits.
    assert_ne!(on, off);
    assert_ne!(off, unknown);
}
#[test]
fn macos_hdr_uses_potential_instead_of_brightness_dependent_current() {
    let info = |current, potential| wgpu::DisplayHdrInfo {
        headroom: Some(wgpu::DisplayHeadroom {
            current: Some(current),
            potential,
            ..Default::default()
        }),
        coarse: Some(wgpu::DisplayCoarseRange {
            high_dynamic_range: Some(current > 1.0),
            ..Default::default()
        }),
        ..Default::default()
    };
    for current in [1.0, 2.0, 4.0] {
        assert_eq!(super::macos_hdr(&info(current, Some(2.0))), Some(true));
        assert_eq!(super::macos_hdr(&info(current, Some(1.0))), Some(false));
    }
    for potential in [
        None,
        Some(0.0),
        Some(-1.0),
        Some(f32::NAN),
        Some(f32::INFINITY),
    ] {
        assert_eq!(super::macos_hdr(&info(4.0, potential)), None);
    }
}
#[test]
fn surface_is_8_bit_and_not_srgb() {
    use super::surface_format;
    use wgpu::TextureFormat::{Bgra8Unorm, Bgra8UnormSrgb, Rgba8Unorm, Rgba16Float};
    // RADV's Wayland surfaces list the linear fp16 format first; choosing it makes the
    // compositor read gamma-encoded values as linear light.
    assert_eq!(
        surface_format(&[Rgba16Float, Bgra8UnormSrgb, Bgra8Unorm], Bgra8UnormSrgb),
        Bgra8Unorm
    );
    assert_eq!(
        surface_format(&[Bgra8UnormSrgb, Rgba8Unorm], Bgra8UnormSrgb),
        Rgba8Unorm
    );
    assert_eq!(surface_format(&[Rgba16Float], Bgra8UnormSrgb), Rgba16Float);
}
/// A Wayland surface on a compositor with colour management, as wgpu lists it (Mesa 26.2
/// ANV on an Intel HD 620 under weston 16): wide formats first. The pre-#1075 choice,
/// the first non-sRGB format, took `Rgba16Float` and the window looked washed out, as if
/// under a white overlay (#1007); XWayland lists no such format, which is why it looked fine.
#[test]
fn a_colour_managed_wayland_surface_still_gets_8_bit() {
    use super::surface_format;
    use wgpu::TextureFormat::*;
    let wayland = [
        Rgba16Float,
        Rgba16Unorm,
        Rgb10a2Unorm,
        Bgra8UnormSrgb,
        Bgra8Unorm,
        Rgba8UnormSrgb,
        Rgba8Unorm,
    ];
    assert_eq!(surface_format(&wayland, Bgra8UnormSrgb), Bgra8Unorm);
    // Without Bgra8: Rgba8, still not a 16- or 10-bit format.
    let without_bgra = [
        Rgba16Float,
        Rgba16Unorm,
        Rgb10a2Unorm,
        Rgba8UnormSrgb,
        Rgba8Unorm,
    ];
    assert_eq!(surface_format(&without_bgra, Rgba8UnormSrgb), Rgba8Unorm);
}
/// The 16-bit frame only with HDR on, Smooth Gradients on, an HDR surface and a device
/// that draws it; in every other case the swapchain's 8-bit format, as before it existed.
#[test]
fn the_frame_is_16_bit_only_with_hdr_and_a_device_that_draws_it() {
    use super::frame_format;
    use wgpu::TextureFormat::{Bgra8Unorm, Bgra8UnormSrgb, Rgba8Unorm, Rgba16Unorm};
    // No float frame asked for (or a device without it) leaves it as before.
    for linear in [
        (false, false, false),
        (false, false, true),
        (true, false, false),
        (false, true, false),
        (true, true, false),
    ] {
        assert_eq!(
            frame_format(true, true, true, Bgra8Unorm, linear),
            Rgba16Unorm
        );
        for (wide_frame, hdr_offered, draws) in [
            (false, true, true),
            (true, false, true),
            (true, true, false),
            (false, false, false),
        ] {
            assert_eq!(
                frame_format(wide_frame, hdr_offered, draws, Bgra8UnormSrgb, linear),
                Bgra8Unorm
            );
            assert_eq!(
                frame_format(wide_frame, hdr_offered, draws, Rgba8Unorm, linear),
                Rgba8Unorm
            );
        }
    }
}
/// A shader API 0.2 pack picked under `SUNRISE_HDR_FRAME=float` gets the float frame
/// wherever the device draws it, HDR on or off.
#[test]
fn a_0_2_pack_gets_the_float_frame_where_the_device_draws_it() {
    use super::{LINEAR_FRAME, frame_format};
    use wgpu::TextureFormat::Bgra8UnormSrgb;
    for (wide_frame, hdr_offered, draws) in [(true, true, true), (false, false, false)] {
        assert_eq!(
            frame_format(
                wide_frame,
                hdr_offered,
                draws,
                Bgra8UnormSrgb,
                (true, false, true)
            ),
            LINEAR_FRAME
        );
    }
}
/// HDR on without a pack under `SUNRISE_HDR_FRAME=float` gets the float frame where the
/// surface offers HDR output, Smooth Gradients or not; where it doesn't, the frame is as
/// without the switch.
#[test]
fn hdr_without_a_pack_gets_the_float_frame_where_hdr_output_is_offered() {
    use super::{LINEAR_FRAME, frame_format};
    use wgpu::TextureFormat::{Bgra8Unorm, Bgra8UnormSrgb};
    for (wide_frame, draws) in [(true, true), (false, false)] {
        assert_eq!(
            frame_format(wide_frame, true, draws, Bgra8UnormSrgb, (false, true, true)),
            LINEAR_FRAME
        );
        assert_eq!(
            frame_format(
                wide_frame,
                false,
                draws,
                Bgra8UnormSrgb,
                (false, true, true)
            ),
            Bgra8Unorm
        );
    }
}
/// What `draws_wide_frame` needs of a device: both features, and the usages and flags
/// every pass takes of the frame. Missing any one of them keeps the 8-bit frame.
#[test]
fn the_16_bit_frame_needs_every_usage_and_flag_the_passes_take() {
    use super::draws_wide_frame;
    use wgpu::{Features as F, TextureFormatFeatureFlags as Fl, TextureUsages as U};
    let features = F::TEXTURE_FORMAT_16BIT_NORM | F::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
    let usages = U::RENDER_ATTACHMENT | U::TEXTURE_BINDING | U::COPY_SRC | U::COPY_DST;
    let flags = Fl::BLENDABLE | Fl::FILTERABLE;
    let format = |allowed_usages, flags| wgpu::TextureFormatFeatures {
        allowed_usages,
        flags,
    };
    // A desktop GPU: all of it, and more.
    assert!(draws_wide_frame(
        features | F::FLOAT32_FILTERABLE,
        format(usages | U::STORAGE_BINDING, flags | Fl::MULTISAMPLE_X4),
    ));
    assert!(!draws_wide_frame(
        F::TEXTURE_FORMAT_16BIT_NORM,
        format(usages, flags)
    ));
    assert!(!draws_wide_frame(
        F::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
        format(usages, flags)
    ));
    // A downlevel device reports no usages at all for a format it lacks.
    assert!(!draws_wide_frame(
        F::empty(),
        format(U::empty(), Fl::empty())
    ));
    for usage in [
        U::RENDER_ATTACHMENT,
        U::TEXTURE_BINDING,
        U::COPY_SRC,
        U::COPY_DST,
    ] {
        assert!(!draws_wide_frame(features, format(usages - usage, flags)));
    }
    for flag in [Fl::BLENDABLE, Fl::FILTERABLE] {
        assert!(!draws_wide_frame(features, format(usages, flags - flag)));
    }
}
#[test]
fn special_material_batches_preserve_quad_order_and_offsets() {
    let quad = |layer, id| {
        [super::Vertex {
            pos: [0.0; 3],
            uv: [0.0; 2],
            layer,
            color: 0xffff_ffff,
            light: id,
            normal: [0; 4],
            glint_uv: [0.0, 0.0],
        }; 4]
    };
    let entities: Vec<_> = [
        (0, 0),
        (14, 9),
        (4, 1),
        (11, 2),
        (15, 10),
        (7, 3),
        (35, 11),
        (4, 4),
        (12, 7),
    ]
    .into_iter()
    .flat_map(|(layer, id)| quad(layer, id))
    .collect();
    let mut batch = super::EntityBatch::default();
    let counts = batch.update(&entities, &quad(3, 5), &quad(0, 6), &quad(0, 8), false);
    let vertices = &batch.combined;
    assert_eq!(counts, [4, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(
        vertices
            .as_chunks::<4>()
            .0
            .iter()
            .map(|q| q[0].light)
            .collect::<Vec<_>>(),
        [0, 1, 3, 4, 11, 2, 7, 9, 10, 5, 6, 8]
    );
}
#[test]
fn software_adapters_are_last_resort() {
    use super::is_software;
    assert!(is_software(wgpu::DeviceType::Cpu));
    assert!(!is_software(wgpu::DeviceType::DiscreteGpu));
    assert!(!is_software(wgpu::DeviceType::IntegratedGpu));
    assert!(!is_software(wgpu::DeviceType::VirtualGpu));
}
#[test]
fn auto_uses_directx_12_first_on_windows_arm64() {
    use super::{GraphicsApi, backend_order};
    use wgpu::Backends;
    let order = |api, os, arch| backend_order(api, os, arch, false);
    assert_eq!(
        order(GraphicsApi::Auto, "windows", "x86_64"),
        [Backends::VULKAN, Backends::DX12, Backends::GL]
    );
    assert_eq!(
        order(GraphicsApi::Auto, "windows", "aarch64"),
        [Backends::DX12, Backends::VULKAN, Backends::GL]
    );
    assert_eq!(
        order(GraphicsApi::Vulkan, "windows", "aarch64"),
        [Backends::VULKAN]
    );
    assert_eq!(
        order(GraphicsApi::Dx12, "windows", "aarch64"),
        [Backends::DX12]
    );
    assert_eq!(
        order(GraphicsApi::OpenGl, "windows", "aarch64"),
        [Backends::GL]
    );
    assert_eq!(
        order(GraphicsApi::Auto, "linux", "x86_64"),
        [Backends::VULKAN, Backends::GL]
    );
    assert_eq!(
        order(GraphicsApi::Dx12, "linux", "x86_64"),
        [Backends::VULKAN]
    );
    assert_eq!(
        order(GraphicsApi::OpenGl, "linux", "x86_64"),
        [Backends::GL]
    );
    // WGPU_BACKEND picks the backends itself; macOS has Metal only.
    for api in GraphicsApi::ALL {
        assert_eq!(
            backend_order(api, "windows", "aarch64", true),
            [Backends::default()]
        );
        assert_eq!(order(api, "macos", "aarch64"), [Backends::default()]);
    }
}
#[test]
fn a_chosen_api_without_a_gpu_tries_the_others_before_software() {
    use super::{GraphicsApi, device_search};
    use wgpu::Backends;
    let (v, d, g) = (Backends::VULKAN, Backends::DX12, Backends::GL);
    let search = |api, os, arch| device_search(api, os, arch, false);
    // #5185: DirectX 12 chosen on an Intel HD 2500 (WARP only): OpenGL's GPU before WARP.
    assert_eq!(
        search(GraphicsApi::Dx12, "windows", "x86_64"),
        [(vec![d], false), (vec![g], false), (vec![d], true)]
    );
    assert_eq!(
        search(GraphicsApi::Vulkan, "windows", "x86_64"),
        [(vec![v], false), (vec![d, g], false), (vec![v], true)]
    );
    assert_eq!(
        search(GraphicsApi::Auto, "windows", "x86_64"),
        [(vec![v, d, g], false), (vec![v, d, g], true)]
    );
    assert_eq!(
        search(GraphicsApi::Auto, "windows", "aarch64"),
        [(vec![d, v, g], false), (vec![d, v, g], true)]
    );
    assert_eq!(
        search(GraphicsApi::OpenGl, "linux", "x86_64"),
        [(vec![g], false), (vec![v], false), (vec![g], true)]
    );
    // WGPU_BACKEND and macOS: one set of backends, nothing else to try.
    let all = Backends::default();
    for (os, env) in [("windows", true), ("macos", false)] {
        assert_eq!(
            device_search(GraphicsApi::Vulkan, os, "x86_64", env),
            [(vec![all], false), (vec![all], true)]
        );
    }
}
/// The second try at a backend shows the adapters wgpu hides for reporting
/// no Vulkan conformance version (Mesa's hasvk, which drives Ivy Bridge and
/// Haswell, reports 0.0.0.0 and works).
#[test]
fn the_second_try_at_a_backend_shows_nonconformant_adapters() {
    let shown = |allow| {
        super::instance_descriptor(wgpu::Backends::VULKAN, None, allow, wgpu::Dx12Compiler::Fxc)
            .flags
            .contains(wgpu::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER)
    };
    assert!(shown(true));
    // The first try keeps wgpu's own filtering, unless `with_env` turns the
    // flag on from the environment (WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER).
    if std::env::var_os("WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER").is_none() {
        assert!(!shown(false));
    }
}
/// Every WGSL shader file translates to HLSL as wgpu's DirectX 12 backend
/// writes it for FXC (shader model 5.1), entry point by entry point, and
/// every shader the OpenGL backend runs also translates to GLSL.
#[test]
#[ignore = "daily"]
fn shaders_translate_to_hlsl_for_directx_12_and_glsl_for_opengl() {
    fn wgsl_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                wgsl_files(&path, out);
            } else if path.extension().is_some_and(|e| e == "wgsl") {
                out.push(path);
            }
        }
    }
    let mut files = Vec::new();
    wgsl_files(
        std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src")),
        &mut files,
    );
    assert!(files.len() >= 10, "{files:?}");
    // Improved transparency's files are appended to terrain.wgsl and entity.wgsl.
    let mut sources: Vec<(std::path::PathBuf, String)> = Vec::new();
    for path in files {
        let text = std::fs::read_to_string(&path).unwrap();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name == "oit.wgsl" || name == "gpu_occlusion.wgsl" || name == "hdr_light.wgsl" {
            // The shared maths: it needs the file it is appended to. GPU culling's
            // occlusion test is appended to gpu_culling.wgsl, on Vulkan alone.
            continue;
        } else if name == "rt_unpack.wgsl" {
            // Appended to terrain.wgsl (`ray_scene`'s `Unpacker`), on ray tracing devices.
            continue;
        } else if name == "terrain.wgsl"
            || name == "entity.wgsl"
            || name == "stars.wgsl"
            || name == "custom_sky.wgsl"
            || name == "border.wgsl"
            || name == "world_primitives.wgsl"
        {
            let base = include_str!("../hdr_light.wgsl").to_owned() + &text;
            sources.push((path, crate::overlay::wgsl_constants() + &base));
        } else if name == "hdr.wgsl" {
            sources.push((path, super::hdr::shader()));
        } else if name == "oit_world.wgsl" {
            let base = super::terrain_shader() + include_str!("../oit.wgsl");
            sources.push((path, base + &text));
        } else if name == "oit_features.wgsl" {
            let base = super::entity_shader() + include_str!("../oit.wgsl");
            sources.push((path, base + &text));
        } else if name == "static_entity.wgsl" {
            sources.push((path, static_entity_shader()));
        } else if name == "prepared_model.wgsl" {
            sources.push((
                path,
                super::entity_shader() + include_str!("../prepared_model.wgsl"),
            ));
        } else if name == "visbuffer.wgsl" {
            // Appended to terrain.wgsl (gfx/visbuffer.rs).
            sources.push((path, super::terrain_shader() + &text));
        } else if name == "oit_border.wgsl" {
            let base = super::border_shader() + include_str!("../oit.wgsl");
            sources.push((path, base + &text));
        } else if name == "oit_clouds.wgsl" {
            let base = super::entity_shader() + include_str!("../oit.wgsl");
            sources.push((path, base + &text));
        } else {
            sources.push((path, text));
        }
    }
    // The default override specializes written light away. Translate its active form too.
    let written: Vec<_> = sources
        .iter()
        .filter(|(path, _)| {
            matches!(
                path.file_name().and_then(|n| n.to_str()),
                Some("terrain.wgsl" | "entity.wgsl" | "stars.wgsl" | "custom_sky.wgsl")
            )
        })
        .map(|(path, source)| {
            (
                path.clone(),
                source.replace(
                    "override HDR_WRITE_LIGHT: bool = false;",
                    "override HDR_WRITE_LIGHT: bool = true;",
                ),
            )
        })
        .collect();
    sources.extend(written);
    let mut glsl_checked = 0;
    // Shaders the OpenGL backend runs must translate: skipping the ones
    // that fail hides exactly the bug where a shader GL needs cannot
    // compile (#9448).
    let mut glsl_failed = Vec::new();
    let mut translated = std::collections::HashSet::new();
    for (path, text) in sources {
        // entity.wgsl comes after the overlay's constants (entity_shader).
        let module = naga::front::wgsl::parse_str(&text)
            .or_else(|_| naga::front::wgsl::parse_str(&(crate::overlay::wgsl_constants() + &text)))
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}: {e:?}", path.display()));
        // A select between integer vectors on a vector of bools becomes GLSL's
        // mix(uvec3, uvec3, bvec3) (GLSL 4.5 or EXT_shader_integer_mix). Old AMD OpenGL
        // drivers (Radeon HD 7400M, driver 15.301) pick the float overload and refuse to
        // compile the terrain shader (#9892). Scalar selects become `?:` and work everywhere.
        let functions = module
            .functions
            .iter()
            .map(|(handle, f)| (f, &info[handle]))
            .chain(
                module
                    .entry_points
                    .iter()
                    .enumerate()
                    .map(|(i, e)| (&e.function, info.get_entry_point(i))),
            );
        for (function, function_info) in functions {
            for (handle, expression) in function.expressions.iter() {
                let naga::Expression::Select {
                    condition, accept, ..
                } = *expression
                else {
                    continue;
                };
                let vector_of = |h: naga::Handle<naga::Expression>| match *function_info[h]
                    .ty
                    .inner_with(&module.types)
                {
                    naga::TypeInner::Vector { scalar, .. } => Some(scalar.kind),
                    _ => None,
                };
                assert!(
                    vector_of(condition).is_none()
                        || !matches!(
                            vector_of(accept),
                            Some(naga::ScalarKind::Uint | naga::ScalarKind::Sint)
                        ),
                    "{} {:?}: select between integer vectors ({handle:?}): use scalar selects",
                    path.display(),
                    function.name
                );
            }
        }
        for entry in &module.entry_points {
            let (module, info) = naga::back::pipeline_constants::process_overrides(
                &module,
                &info,
                Some((entry.stage, &entry.name)),
                &Default::default(),
            )
            .unwrap();
            let mut hlsl = String::new();
            naga::back::hlsl::Writer::new(
                &mut hlsl,
                &naga::back::hlsl::Options::default(),
                &naga::back::hlsl::PipelineOptions {
                    entry_point: Some((entry.stage, entry.name.clone())),
                },
            )
            .write(&module, &info, None)
            .unwrap_or_else(|e| panic!("{} {}: {e}", path.display(), entry.name));
            assert!(!hlsl.is_empty());
            // OpenGL below 4.3 lacks ARB_arrays_of_arrays (#6267): no nested array may reach it.
            // Compute and storage-buffer shaders (culling, frame generation) never run on GL.
            if entry.stage == naga::ShaderStage::Compute
                || text.contains("var<storage")
                || text.contains("var<workgroup>")
            {
                continue;
            }
            for version in [330, 400] {
                let mut glsl = String::new();
                let glsl_result = naga::back::glsl::Writer::new(
                    &mut glsl,
                    &module,
                    &info,
                    &naga::back::glsl::Options {
                        version: naga::back::glsl::Version::Desktop(version),
                        ..Default::default()
                    },
                    &naga::back::glsl::PipelineOptions {
                        shader_stage: entry.stage,
                        entry_point: entry.name.clone(),
                        multiview: None,
                    },
                    Default::default(),
                )
                .and_then(|mut w| w.write());
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if let Err(e) = glsl_result {
                    glsl_failed.push(format!(
                        "{} {} (GLSL {version}): {e}",
                        path.display(),
                        entry.name
                    ));
                    continue;
                }
                translated.insert((name.to_owned(), entry.name.clone()));
                glsl_checked += 1;
                assert!(
                    !glsl.contains("ARB_arrays_of_arrays"),
                    "{} {} needs ARB_arrays_of_arrays on GLSL {version}",
                    path.display(),
                    entry.name
                );
            }
        }
    }
    assert!(
        glsl_failed.is_empty(),
        "shaders the OpenGL backend runs must translate:\n{}",
        glsl_failed.join("\n")
    );
    // These read the world's depth through `textureLoad`; before #9448 their
    // depth binding could not translate, so TAA and the blit never came up. The HUD's
    // CIT glint sampled one image through two samplers until #2386.
    for (file, entry) in [
        ("world_post.wgsl", "fs_taa"),
        ("pack_post.wgsl", "fs_blit"),
        ("hud.wgsl", "fs_cit_glint"),
    ] {
        assert!(
            translated.contains(&(file.to_owned(), entry.to_owned())),
            "{file} {entry} must translate to GLSL"
        );
    }
    assert!(glsl_checked >= 10, "{glsl_checked} GLSL translations");
}
#[test]
fn gl_sprite_arrays_are_neither_2d_textures_nor_cube_maps() {
    for (count, gl) in [(1, 2), (2, 2), (5, 5), (6, 7), (12, 13), (2048, 2048)] {
        assert_eq!(super::array_layers(count, true), gl);
        assert_eq!(super::array_layers(count, false), count);
    }
}
#[test]
fn plain_solid_shaders_translate_for_every_backend() {
    // `SolidPipeline::plain` builds terrain.wgsl's solid entries with `HDR_LIGHTING`
    // off on every backend: they must still translate (the other test translates them
    // with the constants' defaults).
    let module = naga::front::wgsl::parse_str(&super::terrain_shader()).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    let stage = naga::ShaderStage::Fragment;
    let constants = [("HDR_LIGHTING".to_owned(), 0.0)].into_iter().collect();
    for entry in ["fs_solid", "fs_opaque"] {
        let (module, info) = naga::back::pipeline_constants::process_overrides(
            &module,
            &info,
            Some((stage, entry)),
            &constants,
        )
        .unwrap();
        let msl = naga::back::msl::write_string(
            &module,
            &info,
            &naga::back::msl::Options {
                fake_missing_bindings: true,
                ..Default::default()
            },
            &naga::back::msl::PipelineOptions {
                entry_point: Some((stage, entry.to_owned())),
                ..Default::default()
            },
        );
        // Metal compiles the lighting out from this constant.
        assert!(
            msl.unwrap()
                .0
                .contains("constant bool HDR_LIGHTING = false;"),
            "{entry}"
        );
        let mut hlsl = String::new();
        naga::back::hlsl::Writer::new(
            &mut hlsl,
            &Default::default(),
            &naga::back::hlsl::PipelineOptions {
                entry_point: Some((stage, entry.to_owned())),
            },
        )
        .write(&module, &info, None)
        .unwrap_or_else(|e| panic!("{entry}: {e}"));
        let mut glsl = String::new();
        naga::back::glsl::Writer::new(
            &mut glsl,
            &module,
            &info,
            &naga::back::glsl::Options {
                version: naga::back::glsl::Version::Desktop(330),
                ..Default::default()
            },
            &naga::back::glsl::PipelineOptions {
                shader_stage: stage,
                entry_point: entry.to_owned(),
                multiview: None,
            },
            Default::default(),
        )
        .and_then(|mut w| w.write())
        .unwrap_or_else(|e| panic!("{entry}: {e}"));
    }
}
#[test]
fn entity_shader_validates_with_its_overlay_constants() {
    let module = naga::front::wgsl::parse_str(&super::entity_shader()).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}
#[test]
fn chunk_fade_is_vanillas_get_visibility() {
    use super::{chunk_visibility, fade_millis};
    assert_eq!(fade_millis(0.75), 750);
    assert_eq!(fade_millis(0.0), 0);
    assert_eq!(chunk_visibility(0, 750), 0.0_f32);
    assert_eq!(chunk_visibility(375, 750), 0.5);
    assert_eq!(chunk_visibility(750, 750), 1.0);
    assert_eq!(chunk_visibility(0, 0), 1.0, "Off: whole at once");
    assert_eq!(chunk_visibility(u64::MAX, 750), 1.0);
}
/// Iris 1.11.7's terrain `gl_Vertex`, read back by a probe pack of ours, is each block's
/// position minus the camera with its fraction rounded to 1/65536 block: these are the
/// fractions (in 65536ths) it implied, for eyes at y 69.62 and 82.92, at x 100.3 and
/// 40000.3, at z -13.7 and 20000.7, and off the grid by a fraction of a 65536th.
#[test]
fn a_shader_pack_s_terrain_is_drawn_from_the_camera_in_65536ths() {
    let eye = |x: f64, feet: f64, z: f64| glam::DVec3::new(x, feet + f64::from(1.62f32), z);
    let cases = [
        (eye(13.5, 68.0, -13.5), [32768.0, 40632.0, 32768.0]),
        (eye(13.5, 81.3, -13.5), [32768.0, 60293.0, 32768.0]),
        (eye(100.3, 90.0, -13.7), [19661.0, 40632.0, 19661.0]),
        (eye(40000.3, 120.0, 20000.7), [19661.0, 40632.0, 45875.0]),
        (
            eye(
                13.0 + 0.7 / 8192.0,
                81.0 + 0.3 / 8192.0,
                -13.0 - 0.6 / 8192.0,
            ),
            [6.0, 40635.0, 65531.0],
        ),
    ];
    for (camera, sixty_fifths) in cases {
        let from = super::pack_terrain_camera(camera);
        let fraction = (from - camera.floor()) * 65536.0;
        assert_eq!(fraction.to_array(), sixty_fifths, "{camera}");
    }
}
#[test]
fn section_instance_carries_the_fade_in_the_step() {
    use super::section_instance;
    let step = 1.0 / crate::mesher::UV_STEPS;
    // Fully visible (the option Off) is exactly the record from before the fade.
    assert_eq!(
        section_instance([1.0, 2.0, 3.0], 0, 1.0),
        [1.0, 2.0, 3.0, step]
    );
    for visibility in [0.0_f32, 0.25, 0.5, 0.999] {
        let w = section_instance([0.0; 3], 0, visibility)[3];
        let bits = w.to_bits();
        assert_eq!(f32::from_bits(bits & 0xff80_0000), step);
        let read = 1.0 - (bits & 0x7f_ffff) as f32 / 8_388_608.0;
        assert!((read - visibility).abs() < 1e-6, "{visibility}: {read}");
    }
    // A section fading in over far terrain keeps its dither code unchanged.
    assert_eq!(section_instance([0.0; 3], 7, 0.5)[3], -(7.0 + step));
}
/// A section as `Renderer::upload_mesh` makes one, with `solid` and
/// `translucent` quads, fading in over far terrain since `fading`.
fn section(solid: u32, translucent: u32, fading: Option<std::time::Instant>) -> super::SectionGpu {
    super::SectionGpu {
        page: 0,
        start: 0,
        translucent_at: (0, u64::from(solid) * 4),
        solid,
        solid_slots: u64::from(solid) * 4,
        packed_far: false,
        packed: false,
        sides: crate::mesher::Sides::unsorted(solid),
        translucent,
        sort: super::translucency::SortState::new(Vec::new()),
        sorted_indices: None,
        flat_id: None,
        gpu_id: None,
        gpu_fading: false,
        fading,
        leaving: None,
        animated_layers: Box::new([]),
    }
}
/// A section's fade (`SectionGpu::drawn_faded`) only ends as time goes on,
/// never while it leaves or is held, and by its `fade_end`, at most 10 ms
/// after it has (a fade-in over far terrain is whole 9.4 ms before
/// `lod::FADE`): fading in over far terrain or not, first uploaded or not,
/// both, at every chunk fade-in time, in 1 ms steps over 3 s and at its end.
#[test]
fn a_fade_ends_by_its_deadline_and_never_comes_back() {
    use super::FadeInputs;
    use crate::fast_hash::{FxHashMap, FxHashSet};
    use std::time::{Duration, Instant};
    let ms = Duration::from_millis;
    // The steps start at `zero`; what began `t` ms before it began off the steps.
    let zero = Instant::now() + Duration::from_secs(3);
    let ago = |t: u64| zero - ms(t) + Duration::from_micros(370);
    let key = (3, 5, -2);
    // Neither leaving nor held, leaving, held, both.
    let states = [(false, false), (true, false), (false, true), (true, true)];
    for fading in [None, Some(0), Some(300), Some(590), Some(700)] {
        for up in [None, Some(0), Some(1), Some(740), Some(2_500)] {
            let uploaded: FxHashMap<_, _> = up.map(|t| (key, ago(t))).into_iter().collect();
            for (leaving, held) in states {
                let mut s = section(1, 0, fading.map(ago));
                s.leaving = leaving.then(|| ago(250));
                let retiring: FxHashSet<_> = held.then_some(key).into_iter().collect();
                for fade_ms in [0, 1, 50, 300, 750, 2_000] {
                    let case = (fading, up, leaving, held, fade_ms);
                    let at = |now: Instant| FadeInputs {
                        now,
                        fade_ms,
                        uploaded: &uploaded,
                        retiring: &retiring,
                    };
                    let faded = |now: Instant| s.drawn_faded(key, at(now));
                    let end = s.fade_end(key, at(zero));
                    if leaving || held {
                        assert!(end.is_none(), "{case:?}: an end");
                    }
                    let mut was = true;
                    for t in 0..=3_000 {
                        let now = zero + ms(t);
                        let is = faded(now);
                        assert!(was || !is, "{case:?}: faded again at {t} ms");
                        was = is;
                        if leaving || held {
                            assert!(is, "{case:?}: whole at {t} ms");
                        } else if end.is_none_or(|end| now >= end) {
                            assert!(!is, "{case:?}: faded at {t} ms, past its end");
                        }
                    }
                    if let Some(end) = end {
                        assert!(!faded(end), "{case:?}: faded at its end");
                        assert!(faded(end - ms(10)), "{case:?}: whole 10 ms before it");
                    }
                }
            }
        }
    }
}
/// GPU culling's fading sections over 5 s of 16 ms frames, with no device
/// (`mark_fades`): some 2,000 sections come, fading in as vanilla does or
/// over far terrain, are meshed again, leave over far terrain, are held while
/// it comes in and let go, and go, while the chunk fade-in time goes 750 ms,
/// 2 s, Off and 750 ms. After every frame the GPU keeps only whole sections,
/// the walk lists every other one, each record's flag is its section's, and
/// a flagged section that neither leaves nor is held still faded a frame
/// before. Each kind of event flags sections, and thousands go back. Once
/// GPU culling is switched off (`forget_records`), every mark is the one a
/// section has while it is off.
#[test]
#[ignore = "daily"]
fn fading_sections_go_to_the_cpu_and_come_back() {
    use super::{
        FadeInputs, SectionGpu, SectionKey, decide_fade, forget_records, gpu_culling, hold_fading,
        mark_fades,
    };
    use crate::fast_hash::{FxHashMap, FxHashSet};
    use std::time::{Duration, Instant};
    /// The level's sections: those at y -1 and 14 have no record.
    const ROWS: usize = 14;
    const FRAME: Duration = Duration::from_millis(16);
    /// The renderer's sections and what their fades depend on, and GPU
    /// culling's records and ends.
    #[derive(Default)]
    struct Sim {
        sections: super::resident::Resident<SectionGpu>,
        records: gpu_culling::Records,
        fades: gpu_culling::Fades,
        uploaded: FxHashMap<SectionKey, Instant>,
        retiring: FxHashSet<SectionKey>,
    }
    impl Sim {
        /// `Renderer::upload` at `at`: a new section, fading in over far
        /// terrain if `far`, or one meshed again.
        fn upload(&mut self, key: SectionKey, quads: (u32, u32), far: bool, at: Instant) {
            self.retiring.remove(&key);
            let fading = match self.sections.find(key) {
                Some(old) => old.fading,
                None => far.then_some(at),
            };
            if fading.is_none() {
                self.uploaded.entry(key).or_insert(at);
            }
            self.release(key);
            let mut s = section(quads.0, quads.1, fading);
            if s.gpu_recorded(key, ROWS) {
                if let Some(fade_ms) = self.fades.fade_ms {
                    let f = FadeInputs {
                        now: at,
                        fade_ms,
                        uploaded: &self.uploaded,
                        retiring: &self.retiring,
                    };
                    s.gpu_fading = decide_fade(&s, key, &mut self.fades, f);
                }
                s.gpu_id = Some(self.records.insert(s.record(key, None)));
            }
            let marked = s.listed(key, ROWS);
            self.sections.insert_marked(key, s, None, marked);
        }
        /// `Renderer::release_section`.
        fn release(&mut self, key: SectionKey) {
            if let Some(id) = self.sections.remove(key).and_then(|s| s.gpu_id) {
                self.records.remove(id);
            }
        }
        /// `Renderer::leave_section`: its column gone (`remove_columns`), or
        /// no longer left out by far terrain while held (`set_lod_cutout`).
        fn leave(&mut self, key: SectionKey, at: Instant) {
            self.uploaded.remove(&key);
            if let Some(s) = self.sections.find_mut(key)
                && s.leaving.is_none()
            {
                s.leaving = Some(at);
                hold_fading(&mut self.sections, &mut self.records, key);
            }
        }
        /// `Renderer::remove_columns`' held section: far terrain comes in under it.
        fn hold(&mut self, key: SectionKey) {
            self.uploaded.remove(&key);
            self.retiring.insert(key);
            hold_fading(&mut self.sections, &mut self.records, key);
        }
        fn flagged(&self, key: SectionKey) -> bool {
            self.sections.find(key).is_some_and(|s| s.gpu_fading)
        }
    }
    let mut seed = 0x2545_f491_u32;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    // 16 x 16 columns of sections at y -1 to 14.
    let key = |r: u32| {
        let [x, y, z] = [0, 4, 8].map(|shift| ((r >> shift) % 16) as i32);
        (x, y - 1, z)
    };
    // Solid quads alone, with translucent ones, or translucent ones alone.
    let kinds = [(64, 0), (64, 0), (64, 0), (64, 16), (64, 16), (0, 16)];
    let quads = |r: u32| kinds[r as usize % 6];
    let start = Instant::now();
    let mut sim = Sim::default();
    for r in 0..4_096 {
        if next().is_multiple_of(2) {
            sim.upload(key(r), quads(next()), next().is_multiple_of(2), start);
        }
    }
    // Flags set by an upload, `leave_section`, a held section and a fade-in
    // time (GPU culling switched on in the first frame), and dropped.
    let (mut set_by, mut handed_back) = ([0; 4], 0);
    let mut last = start;
    for frame in 0..=312 {
        let now = start + FRAME * frame;
        for event in 0..12 {
            let at = last + (now - last) * (event + 1) / 13;
            let r = next();
            let k = key(r);
            let was = sim.flagged(k);
            let kind = match (r >> 20) % 16 {
                0..=7 => {
                    sim.upload(k, quads(r >> 12), (r >> 24).is_multiple_of(2), at);
                    0
                }
                8 | 9 => {
                    sim.leave(k, at);
                    1
                }
                10 | 11 => {
                    sim.hold(k);
                    2
                }
                12 => {
                    // Far terrain no longer leaves a held section's column out.
                    let held = sim.retiring.iter().next().copied();
                    if let Some(held) = held {
                        sim.retiring.remove(&held);
                        sim.leave(held, at);
                    }
                    continue;
                }
                _ => {
                    sim.uploaded.remove(&k);
                    sim.release(k);
                    continue;
                }
            };
            if !was && sim.flagged(k) {
                set_by[kind] += 1;
            }
        }
        // Those faded out go (`release_left`).
        let gone = |t: Instant| now.saturating_duration_since(t) >= super::lod::FADE;
        let left: Vec<SectionKey> = sim
            .sections
            .iter()
            .filter(|(_, s, _)| s.leaving.is_some_and(gone))
            .map(|(k, ..)| k)
            .collect();
        for k in left {
            sim.release(k);
        }
        let flagged_before: FxHashSet<SectionKey> = sim
            .sections
            .iter()
            .filter(|(_, s, _)| s.gpu_fading)
            .map(|(k, ..)| k)
            .collect();
        let fade_ms = [750, 2_000, 0, 750][frame as usize * 4 / 313];
        let f = FadeInputs {
            now,
            fade_ms,
            uploaded: &sim.uploaded,
            retiring: &sim.retiring,
        };
        mark_fades(&mut sim.sections, &mut sim.records, &mut sim.fades, ROWS, f);
        let a_frame_before = FadeInputs {
            now: now - FRAME,
            ..f
        };
        let mut flagged = 0;
        for (k, s, _) in sim.sections.iter() {
            let marked = sim.sections.is_marked(k);
            assert_eq!(marked, s.listed(k, ROWS), "frame {frame}, {k:?}: its mark");
            // The GPU draws its solid quads whole, or the walk lists it.
            assert!(
                s.solid == 0 || s.gpu_draws_solid(true) || marked,
                "frame {frame}, {k:?}: drawn by neither"
            );
            assert!(
                !s.gpu_draws_solid(true) || !s.drawn_faded(k, f),
                "frame {frame}, {k:?}: drawn whole by the GPU while it fades"
            );
            if let Some(id) = s.gpu_id {
                let record = s.record(k, None);
                assert_eq!(sim.records.at(id), Some(&record), "frame {frame}, {k:?}");
            }
            if s.gpu_fading {
                flagged += 1;
                set_by[3] += usize::from(!flagged_before.contains(&k));
                if s.leaving.is_none() && !sim.retiring.contains(&k) {
                    assert!(
                        s.drawn_faded(k, a_frame_before),
                        "frame {frame}, {k:?}: flagged a frame after its fade"
                    );
                }
            } else {
                handed_back += usize::from(flagged_before.contains(&k));
            }
        }
        assert_eq!(sim.records.fading(), flagged, "frame {frame}");
        last = now;
    }
    assert!(set_by.iter().all(|&n| n > 20), "flags set {set_by:?}");
    // Over a thousand: the half of the uploads over far terrain come in whole.
    assert!(handed_back > 1_000, "{handed_back} handed back");
    // Switched off with sections still flagged: none keeps a record or a
    // flag, and each is marked as while GPU culling is off.
    assert!(sim.records.fading() > 0, "none flagged at the end");
    forget_records(&mut sim.sections, ROWS);
    for (k, s, _) in sim.sections.iter() {
        assert!(s.gpu_id.is_none() && !s.gpu_fading, "{k:?}: kept once off");
        let off = s.translucent > 0 || !s.gpu_recorded(k, ROWS);
        assert_eq!(sim.sections.is_marked(k), off, "{k:?}: its mark once off");
    }
}
#[test]
fn terrain_attributes_follow_the_terrain_vertex() {
    use super::TerrainVertex;
    use std::mem::offset_of;
    let offsets = super::TERRAIN_ATTRIBUTES.map(|a| (a.shader_location, a.offset as usize));
    assert_eq!(
        offsets,
        [
            (0, offset_of!(TerrainVertex, pos)),
            (1, offset_of!(TerrainVertex, uv)),
            (2, offset_of!(TerrainVertex, layer)),
            (4, offset_of!(TerrainVertex, light)),
            (3, offset_of!(TerrainVertex, color)),
        ]
    );
    assert_eq!(std::mem::size_of::<TerrainVertex>(), 24);
}
#[test]
fn cracks_multiply_instead_of_covering_the_block() {
    let blend = super::CRUMBLING_BLEND;
    assert_eq!(blend.color.src_factor, wgpu::BlendFactor::Dst);
    assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::Src);
    assert_eq!(blend.alpha.src_factor, wgpu::BlendFactor::One);
    assert_eq!(blend.alpha.dst_factor, wgpu::BlendFactor::Zero);
}
#[test]
fn hdr_screenshots_keep_every_10_bit_step() {
    // Rgb10a2Unorm: red in the low 10 bits, then green, then blue.
    let px = |r: u32, g: u32, b: u32| (r | g << 10 | b << 20 | 3 << 30).to_le_bytes();
    let out = super::screenshot::hdr10_to_rgb16(&[px(0, 1023, 512)]);
    assert_eq!(out, [0, 0, 0xff, 0xff, 0x80, 0x20]);
    assert_eq!(
        super::screenshot::hdr_screenshot_path(std::path::Path::new(
            "screenshots/2026-10-28_12.00.00.png"
        )),
        std::path::Path::new("screenshots/2026-10-28_12.00.00-hdr.png")
    );
}
#[test]
fn texture_filtering_samplers() {
    use super::{sampler_anisotropy, sprite_sampler_descriptor};
    // LevelRenderer.chunkLayerSampler: linear both ways, clamped, in every mode.
    for aniso in [1, 8] {
        let d = sprite_sampler_descriptor(aniso);
        assert_eq!(d.mag_filter, wgpu::FilterMode::Linear);
        assert_eq!(d.min_filter, wgpu::FilterMode::Linear);
        assert_eq!(d.mipmap_filter, wgpu::MipmapFilterMode::Linear);
        assert_eq!(d.address_mode_v, wgpu::AddressMode::ClampToEdge);
        assert_eq!(d.anisotropy_clamp, aniso);
    }
    assert_eq!(
        [1, 2, 3].map(sampler_anisotropy),
        [2, 4, 8],
        "Options.maxAnisotropyValue is 1 << bit"
    );
}
#[test]
fn distant_tiles_repeat_their_sprite_under_the_clamped_sampler() {
    // terrain.wgsl's tile_uv, evaluated by naga as a constant expression.
    let source = super::terrain_shader();
    let body = source.split("fn tile_uv(").nth(1).unwrap();
    let body = body.split("return ").nth(1).unwrap();
    let expr = &body[..body.find(';').unwrap()];
    let tile_uv = |u: f32, v: f32| -> [f32; 2] {
        let uv = format!("vec2<f32>({u:?}, {v:?})");
        let wgsl = format!("const UV: vec2<f32> = {};", expr.replace("uv", &uv));
        let module = naga::front::wgsl::parse_str(&wgsl).unwrap();
        let (_, constant) = module.constants.iter().next().unwrap();
        let naga::Expression::Compose { components, .. } =
            &module.global_expressions[constant.init]
        else {
            panic!("{wgsl}");
        };
        components
            .iter()
            .map(|&c| match module.global_expressions[c] {
                naga::Expression::Literal(naga::Literal::F32(x)) => x,
                ref e => panic!("{e:?}"),
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    };
    // A section's sprite UVs, edges included, stay where the clamped sampler reads them.
    for (u, v) in [(0.0, 1.0), (0.25, 0.75), (1.0, 0.0)] {
        assert_eq!(tile_uv(u, v), [u, v]);
    }
    // A merged face's UV counts blocks: each takes the sprite from its own corner.
    assert_eq!(tile_uv(1.25, 2.0), [0.25, 1.0]);
    assert_eq!(tile_uv(1023.5, 3.75), [0.5, 0.75]);
}
#[test]
fn shader_globals_match_the_rust_layout() {
    // terrain.wgsl and entity.wgsl bind the same buffer: same fields, same order.
    let fields = |source: &str| -> Vec<String> {
        let body = source.split("struct Globals {").nth(1).unwrap();
        body[..body.find('}').unwrap()]
            .lines()
            .filter_map(|l| {
                l.trim()
                    .split_once(':')
                    .map(|(name, _)| name.trim_start_matches("pub(super) ").to_string())
            })
            .filter(|name| !name.starts_with("//"))
            .collect()
    };
    let terrain = fields(include_str!("../terrain.wgsl"));
    assert_eq!(terrain, fields(include_str!("../entity.wgsl")));
    let rust = fields(include_str!("frame.rs"));
    assert_eq!(terrain, rust);
}
#[test]
fn selection_outline_uses_vanilla_pixel_width() {
    for width in [640, 1280, 1920] {
        assert_eq!(super::outline_width(width), 2.5);
    }
    assert_eq!(super::outline_width(2560), 2560.0 / 1920.0 * 2.5);
    assert_eq!(super::outline_width(3840), 5.0);
}
#[test]
fn the_projection_center_follows_the_view_bob() {
    use glam::{Mat4, Vec3};
    let projection = super::world_projection(70.0, 16.0 / 9.0, 512.0);
    let view = crate::camera::view_matrix(30.0, 20.0);
    let eye = super::view::projection_center(projection * view).unwrap();
    assert!(eye.length() < 1e-5, "{eye}");
    // A pose translated by the bob: the eye is where the pose's inverse
    // takes the view's origin.
    let bob = Mat4::from_rotation_z(0.02) * Mat4::from_translation(Vec3::new(0.03, -0.08, 0.0));
    let expected = (bob * view).inverse().transform_point3(Vec3::ZERO);
    let eye = super::view::projection_center(projection * bob * view).unwrap();
    assert!(eye.distance(expected) < 1e-5, "{eye} {expected}");
    assert!(expected.length() > 0.05);
    assert!(
        super::view::projection_center(Mat4::orthographic_rh(-1.0, 1.0, -1.0, 1.0, 0.0, 1.0))
            .is_none()
    );
}
#[test]
fn indirect_draws_are_validated_only_in_debug_builds_and_on_directx_12() {
    let validated = |backends| {
        super::instance_descriptor(backends, None, false, wgpu::Dx12Compiler::Fxc)
            .flags
            .contains(wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL)
    };
    if std::env::var_os("WGPU_VALIDATION_INDIRECT_CALL").is_none()
        && std::env::var_os("WGPU_BACKEND").is_none()
    {
        assert_eq!(validated(wgpu::Backends::default()), cfg!(debug_assertions));
        assert_eq!(validated(wgpu::Backends::VULKAN), cfg!(debug_assertions));
        assert_eq!(
            validated(wgpu::Backends::DX12),
            cfg!(debug_assertions) || cfg!(windows)
        );
    }
}
#[test]
fn far_faces_a_hair_apart_keep_their_depth_order() {
    let m = super::world_projection(70.0, 16.0 / 9.0, 768.0);
    let depth = |distance: f32| {
        let clip = m * glam::Vec4::new(0.0, 0.0, -distance, 1.0);
        clip.z / clip.w
    };
    // fire_side at 150 blocks, 0.01 pixel in front of the block it burns on.
    let (block, fire) = (depth(150.0), depth(150.0 - 0.01 / 16.0));
    assert!(fire > block, "{fire} {block}");
    assert!((0.0..=1.0).contains(&block));
    assert!((depth(0.05) - 1.0).abs() < 1e-6);
}
#[test]
fn selection_outline_has_vanilla_perspective_layering() {
    assert_eq!(
        super::outline_params(1280, 720),
        [1280.0, 720.0, 2.5, 0.99975586]
    );
    assert_eq!(
        super::outline_params(3840, 0),
        [3840.0, 1.0, 5.0, 0.99975586]
    );
}
#[test]
fn selection_outline_preserves_shape_edges_without_inflation() {
    let edges = [[2.0, 97.5, 3.0, 3.0, 97.5, 3.0]];
    assert_eq!(
        super::outline_vertices(&edges, glam::DVec3::new(0.5, 98.62, 0.5)),
        vec![[
            1.5,
            (97.5 - 98.62) as f32,
            2.5,
            2.5,
            (97.5 - 98.62) as f32,
            2.5
        ]]
    );
    assert!(super::outline_vertices(&[], glam::DVec3::ZERO).is_empty());
}
