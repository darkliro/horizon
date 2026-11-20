//! GPU renderer: terrain, sky, block outline and HUD. Camera-relative section origins keep world coordinates precise.
use crate::{
    Result,
    block_entity_render::{StaticCacheFrameStats, StaticSectionRevision, StaticSectionSnapshot},
    environment::Sky,
    fast_hash::FxHashMap,
    graphics::GraphicsApi,
    hud::{HudBatch, HudRenderer},
    mesher::{Part as Quads, SectionMesh, Sides, TerrainVertex, Vertex, packed},
    section_mask::SectionMask,
    sprites::SpriteAnimation,
};
#[cfg(any(test, feature = "test-support"))]
use glam::Vec4;
use glam::{DVec3, Mat4, Vec3};
use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Instant};
use wgpu::util::DeviceExt;
use winit::window::Window;
// RenderPipelines.CRUMBLING: dst * src + src * dst, alpha replaced.
const CRUMBLING_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Dst,
        dst_factor: wgpu::BlendFactor::Src,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent::REPLACE,
};
/// `TerrainVertex` for `vs_terrain`: position, UV, layer, light, colour.
const TERRAIN_ATTRIBUTES: [wgpu::VertexAttribute; 5] =
    wgpu::vertex_attr_array![0=>Float32x3,1=>Uint16x2,2=>Uint16,4=>Uint8x2,3=>Unorm8x4];

fn no_pack_hdr_gain(layered: bool, pack: bool, world: bool) -> bool {
    layered && !pack && world
}

fn light_marks(format: wgpu::TextureFormat, hdr: bool) -> bool {
    hdr && format != LINEAR_FRAME
}

fn terrain_shader() -> String {
    include_str!("hdr_light.wgsl").to_owned() + include_str!("terrain.wgsl")
}

pub(super) use border::border_shader;

pub mod border;
#[cfg(test)]
mod crumbling_tests;
mod gpu_checks;
mod section_gpu;
mod static_entity;

pub(crate) use section_gpu::collect_animated_layers;
use section_gpu::{
    Part, SectionGpu, begin_shrink, decide_fade, forget_records, hold_fading, mark_fades,
};
use static_entity::{GpuStaticEntitySection, StaticEntityUniform};
pub mod outline;
mod overlay;
pub use border::BorderDraw;
use border::BorderUniform;
use gpu_checks::GpuChecks;
use outline::{debug_line_opaque, debug_lines_opaque_first, outline_vertices_into};
pub use outline::{outline_params, outline_vertices, outline_width};
use overlay::Overlay;
mod entity_batch;
mod entity_outline;
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub mod entity_pixel_test;
mod fluid;
mod prepared_model;
#[cfg(feature = "test-support")]
pub use prepared_model::fail_next_prepared_upload;
pub use prepared_model::{
    PreparedInstanceBuilder, PreparedModelDraw, PreparedModelInstance, PreparedModelVertex,
    PreparedModels, camera_matrix,
};
#[cfg(feature = "test-support")]
pub use prepared_model::{
    cap_prepared_instance_slots, capped_prepared_room, set_test_prepared_generation,
    test_prepared_generation,
};
mod block_entity_positions;
mod culling_control;
mod draw_frame;
mod entity_draw;
mod hdr_config;
#[cfg(test)]
mod inter_stage_tests;
mod section_fade;
mod section_upload;
mod shader_pack;
mod surface_lifecycle;
use entity_draw::*;
#[cfg(test)]
mod sprite_arrays_tests;
mod terrain_model;
#[cfg(any(test, feature = "test-support"))]
pub(crate) mod test_gpu;
mod weather;
mod world_text;
use cull::occlusion_culling_requested;
use entity_batch::EntityBatch;
pub use entity_batch::{
    ADDITIVE_PARTICLE_OVERLAY, DISSOLVE_OVERLAY, DRAGON_RAYS_OVERLAY, ENERGY_SWIRL_OVERLAY,
    FLUID_OVERLAY, OIT_ENTITY, SHADOW_OVERLAY, TERRAIN_MODEL_CUTOUT_MIPPED_OVERLAY,
    TERRAIN_MODEL_CUTOUT_OVERLAY, TERRAIN_MODEL_SOLID_OVERLAY, TEXT_POLYGON_OFFSET_OVERLAY,
    TRANSLUCENT_GLINT_ITEM_OVERLAY, TRANSLUCENT_ITEM_OVERLAY, TRANSLUCENT_PARTICLE_OVERLAY,
    WATER_MASK_OVERLAY, WORLD_TEXT_OVERLAY, WORLD_TEXT_POLYGON_OFFSET_OVERLAY, is_native_terrain,
    mark_oit,
};
use external::{shadow_box, shadow_runs};
pub use frustum::{Cull, Frusta, JoinedFrusta, ShadowCull};
use frustum::{block_entities_within_reach, frustum, shadow_casters, taa_frustum};
use section_fade::{
    CHUNK_VISIBILITY_THRESHOLD, FadeInputs, chunk_visibility, fade_millis, section_instance,
    section_visibility,
};
pub use view::{View, ViewSet};

/// Whether Translucent Batching is offered (`graphics::offer_translucent_batching`): where
/// indirect multi-draws (`multi_draw`) take its batches, but not OpenGL's, which loop over
/// their entries on the CPU (and without them every section is a call of its own anyway).
/// Frames drawn flat (Metal, Intel Gen11/Gen12's Vulkan driver) take it too: their
/// translucent calls then come one per page of its arena rather than about one per section.
fn translucent_batchable(info: &wgpu::AdapterInfo, multi_draw: bool) -> bool {
    multi_draw && info.backend != wgpu::Backend::Gl
}

/// The vertices flat draws bind for a call's page key: a page of the section arena, or of
/// the translucent arena up to its last range in use (`flat::TRANSLUCENT_KEYS`).
fn flat_vertices<'a>(
    arena: &'a arena::Arena,
    translucent: Option<&'a arena::Arena>,
) -> impl Fn(u32) -> wgpu::BufferSlice<'a> + Copy {
    move |key| match (translucent, key.checked_sub(flat::TRANSLUCENT_KEYS)) {
        (Some(translucent), Some(page)) => translucent.binding(page),
        _ => arena.buffer(key).slice(..),
    }
}

/// Bytes of `texture` with every mip level of every layer.
pub(crate) fn texture_bytes(texture: &wgpu::Texture) -> u64 {
    let size = texture.size();
    let block = u64::from(texture.format().block_copy_size(None).unwrap_or(4));
    (0..texture.mip_level_count())
        .map(|level| {
            let side = |v: u32| u64::from((v >> level).max(1));
            side(size.width) * side(size.height) * block
        })
        .sum::<u64>()
        * u64::from(size.depth_or_array_layers)
}

/// Whether translucent quads have an arena of their own: with Translucent Batching
/// (`batching`), unless a shader pack's frames can draw terrain flat (as for `flat_frame`),
/// which reads them in the section arena through its tables: where flat draws exist and
/// are on (`flat_draws`), for a `shaders/` pack's terrain programs (`terrain`) or a shadow
/// pass (`shadow`: a `shaders/` pack's, or the `[shadow]` of one of Sunrise's own).
fn translucent_split(batching: bool, flat_draws: bool, terrain: bool, shadow: bool) -> bool {
    batching && !(flat_draws && (terrain || shadow))
}

/// The compact layout of solid quads, unless `SUNRISE_PACKED_QUADS=0` (A/B checks).
fn packed_wanted() -> bool {
    std::env::var("SUNRISE_PACKED_QUADS").as_deref() != Ok("0")
}

pub(crate) mod animate;
mod arena;
pub mod capture;
mod celestial;
mod census;
pub mod compositor;
pub(crate) mod core_lines;
pub(crate) mod core_sky;
pub(crate) mod core_text;
mod cull;
pub(crate) mod display_watch;
mod dx12_compiler;
mod end_sky;
pub mod entity_skin;
pub mod external;
mod flat;
mod frame_gen;
mod frustum;
mod gpu_culling;
mod hdr;
mod hdr_icc;
mod hdr_start;
mod hdr_stats;
pub(crate) mod hud_coverage;
mod in_flight;
use display_watch::DisplayWatch;
#[cfg(test)]
use display_watch::{macos_hdr, os_reported};
pub use hdr::{Calibration as HdrCalibration, Output as HdrOutput};
pub use output::{DisplayReading, EffectiveOutput, FrameConfiguration, FrameRequest};

pub mod block_light;
mod custom_sky;
mod displace;
mod frame_policy;
pub mod lightning;
mod linear_blend;
#[cfg(test)]
use frame_policy::frame_format;
use frame_policy::{frame_configuration, overlay_format, sunrise_tone_curve};
mod lod;
pub mod mod_pipelines;
mod native_material;
mod native_pack;
mod native_post;
use crate::render_slots as native_render;
mod native_shadow;
mod native_sky;
mod occlusion;
mod oit;
mod oit_features;
pub(crate) mod output;
mod output_apply;
mod pack_atlas;
mod pack_center_depth;
pub mod pack_draw;
mod pack_entities;
mod pack_hand;
mod pack_images;
mod pack_lines;
mod pack_post;
mod pack_shadow;
mod pack_sky;
mod pack_textures;
pub mod pack_timing;
pub mod pack_uniforms;
mod pack_weather;
mod pass_timing_api;
use pass_timing_api::{after_culling_pass, pixel_count, world_pass_after_culling};
mod page_runs;
pub mod panorama;
mod readback;
mod resident;
mod reuse;
mod screenshot;
pub use screenshot::{PendingScreenshot, ScreenshotSave};
mod screenshot_readback;
mod stars;
mod surface_setup;
use surface_setup::{AdapterChoice, AdapterFailure, open_adapter, pick_adapter, set_up_surface};
mod off_page;
use off_page::move_off_page;
pub mod submit;
mod sunrise_fan;
mod timing;
pub mod translucency;
pub(crate) mod upload;
mod view;
mod view_sections;
mod visbuffer;
mod wgsl_prefix;
mod world_handover;
mod world_post;
pub mod world_primitives;
mod world_views;

/// A section's instance: its camera-relative origin and, in `w`, the step of
/// its fixed-point UVs (`terrain.wgsl`'s `terrain_uv`).
/// The world pass again after occlusion culling's test: it loads both
/// targets (phase 2 draws over phase 1, reading the test's results).
/// `LevelRenderer.addSkyPass`: fluids and camera effects block the sky pass.
fn sky_open(frame: &Frame) -> bool {
    !frame.sky_blocked
}

/// The Overworld-style sky (disc, sunset, stars, void) draws this frame.
fn sky_visible(frame: &Frame) -> bool {
    !frame.title && frame.skybox && sky_open(frame)
}

/// The End during the dragon fight, whose world fog leaves out vanilla's sky: a shader pack's
/// `gbuffers_skybasic` still draws a fan around the horizon (`pack_sky`), as our probe packs read
/// it in Iris 1.11.7.
fn end_fog_sky(frame: &Frame) -> bool {
    pack_sky::end_fog_sky(frame.title, frame.end_sky, frame.boss_world_fog)
}

/// The point a `shaders/` pack's terrain is drawn relative to: the camera with the fraction of
/// each coordinate rounded to the nearest 1/65536 block. Read back with a probe pack of ours,
/// Iris 1.11.7's terrain `gl_Vertex` is the block's position minus this, for cameras from 13
/// to 40000 blocks out (#8464); `cameraPosition` and the matrices keep the exact camera. Off by
/// up to 2^-17 blocks, a fragment on a texel boundary of the shadow map reads the other texel.
fn pack_terrain_camera(camera: DVec3) -> DVec3 {
    let whole = camera.floor();
    whole + ((camera - whole) * 65536.0).round() / 65536.0
}
/// A new value for `Renderer::block_entity_revision`.
fn next_block_entity_revision() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

pub type SectionKey = (i32, i32, i32);
const MAX_OUTLINE_EDGES: u64 = 256;
/// Staging buffer size for section meshes: a frame of uploads while
/// exploring fits in a few.
const SECTION_UPLOAD_CHUNK: u64 = 4 << 20;
/// Section copies are submitted early past this many staged bytes, which
/// bounds the staging memory frames that are not drawn can hold.
const SECTION_COPY_LIMIT: u64 = 64 << 20;
/// What compaction copies in a frame, `compact_arena` first and
/// `shrink_translucent` with what it left, and a range more where a call's
/// first is larger (`move_off_page`): a page empties over a few frames, with
/// no hitch.
const COMPACT_BUDGET: u64 = 32 << 20;
/// A large translucent page starts to shrink (`Renderer::shrink_translucent`)
/// this long after terrain last arrived, and `SHRINK_RETRY` after a try found
/// no room.
/// `Renderer::remove_columns` filters each store once rather than looking up every
/// section key from this many keys, or from as many as there are resident sections.
const BULK_REMOVAL_KEYS: usize = 256;
const SHRINK_SETTLE: std::time::Duration = std::time::Duration::from_secs(1);
const SHRINK_RETRY: std::time::Duration = std::time::Duration::from_secs(5);
pub struct Renderer {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    /// The window's swapchain; `None` when rendering offscreen (`--offscreen`). Shared
    /// with frame generation's present thread while it runs.
    surface: Option<Arc<wgpu::Surface<'static>>>,
    /// The surface `release_surface` set aside for the renderer that replaces
    /// this one on the same device: reused, not created again, as a new
    /// surface for the window can come back lost (Wayland on Mesa).
    spare_surface: std::sync::Mutex<Option<wgpu::Surface<'static>>>,
    /// The offscreen colour target, sized like `config`.
    offscreen: Option<wgpu::Texture>,
    /// Offscreen submissions still on the GPU: at most two frames run ahead.
    in_flight: std::collections::VecDeque<wgpu::SubmissionIndex>,
    /// Which steps of the first frame on this device still go to the log
    /// (see `startup_step`).
    first_frame: FirstFrame,
    gpu_timer: Option<timing::GpuTimer>,
    /// GPU frame times (ms) read back since the caller last drained them.
    pub gpu_times: Vec<f32>,
    /// The last lightmap written, RGBA rows of sky level by block level
    /// (F3+4 shows it).
    pub lightmap_pixels: [u8; 16 * 16 * 4],
    /// Time the last `render` spent blocked on the GPU or the display
    /// (acquiring, presenting, offscreen pacing, the submit thread holding the queue),
    /// excluded from CPU time.
    pub last_wait: std::time::Duration,
    /// CPU time of the last `render`, waits excluded.
    pub last_encode: std::time::Duration,
    /// Watched only while HDR is On or Auto, including Auto in SDR;
    /// before `window`, so it's dropped first.
    display: Option<DisplayWatch>,
    /// What this renderer's display last reported (`DisplayWatch`), its own: another
    /// renderer's watch never changes it.
    reading: output::DisplayReading,
    /// The watch was just made: the settings' Auto values take what it reported.
    display_fresh: bool,
    window: Arc<Window>,
    pub device: wgpu::Device,
    /// The shader modules, layouts and pipelines kept for the next renderer on
    /// this device ([`reuse`]).
    reuse: Arc<reuse::Reuse>,
    /// Frames are finished and submitted on its submit thread ([`submit`]).
    pub queue: submit::Queue,
    /// The last frame went to the submit thread; whether it had no window surface.
    /// The next frame takes it in ([`Self::land_frame`]).
    deferred: Option<bool>,
    /// Set by the device's lost callback. wgpu reports nothing else when an
    /// operation fails on a lost device (no error scope sees it).
    device_lost: Arc<std::sync::atomic::AtomicBool>,
    pub config: wgpu::SurfaceConfiguration,
    solid: SolidPipeline,
    /// `solid` split in two passes for the world pass (terrain.wgsl's `opaque_quad`).
    solid_opaque: SolidPipeline,
    solid_cutout: SolidPipeline,
    translucent: wgpu::RenderPipeline,
    translucent_mark: wgpu::RenderPipeline,
    /// Flat section draws (`flat`), where multi-draw is emulated.
    flat: Option<flat::Flat>,
    /// Sections upload their solid quads in the compact layout (`mesher::packed`): the solid
    /// passes read the arena themselves (`flat`) and no shader pack reads it as vertices.
    packed_quads: bool,
    /// The terrain's acceleration structures, with ray tracing on (`ray_scene`).
    ray_scene: Option<surface_setup::ray_query::ray_scene::RayScene>,
    /// The compact layout is allowed (`SUNRISE_PACKED_QUADS`, `set_packed_layout`).
    packed_on: bool,
    /// The visibility buffer for flat frames' solid terrain (`visbuffer`,
    /// `SUNRISE_VISBUFFER=1`).
    visbuf: Option<visbuffer::Visbuffer>,
    /// `flat` draws this frame's sections (off: the per-section draws).
    flat_on: bool,
    /// Occlusion culling of solid section draws, where the GPU's instance
    /// counts reach a multi-draw's entries (Vulkan, Metal).
    cull: Option<cull::Cull>,
    cull_on: bool,
    /// Solid quads facing away from the eye are left out (`mesher::Sides`).
    sides_on: bool,
    /// The near solid sections culled and drawn on the GPU (`gpu_culling`): its pass,
    /// records and fades while it is on, where it is offered (`gpu_culling_offer`, mirrored
    /// for the settings in `graphics::gpu_culling_offer`), and whether the last frame drew
    /// them so (`gpu_frame`).
    gpu_culling: Option<gpu_culling::GpuCulling>,
    gpu_culling_offer: crate::graphics::GpuCullingOffer,
    gpu_frame: bool,
    /// GPU frames since GPU culling was switched on: the first and every 64th after it
    /// check its records (`check_gpu_records`).
    #[cfg(debug_assertions)]
    gpu_checks: u32,
    sky: wgpu::RenderPipeline,
    /// The main sky without the sunrise fan, which `sunrise_pipeline` draws as
    /// geometry over it. The mirror keeps `sky`: its per-pixel fan is cheap at
    /// the reflection's size and needs no second projection.
    sky_base: wgpu::RenderPipeline,
    sunrise_pipeline: wgpu::RenderPipeline,
    sunrise_buffer: wgpu::Buffer,
    dark_disc: wgpu::RenderPipeline,
    /// Clears the world pass's depth to the far plane so the first-person hand draws in the
    /// same pass (`hand_in_world`), as the hand's own pass would clear it.
    depth_reset: wgpu::RenderPipeline,
    lines: wgpu::RenderPipeline,
    lines_secondary: wgpu::RenderPipeline,
    /// A resource pack's `core/rendertype_lines`, which draws the block outline in their place.
    core_lines: Option<core_lines::CoreLines>,
    core_text: Option<core_text::CoreText>,
    /// The packs' `core/sky` and `core/stars`, drawn in place of the native sky and stars.
    core_sky: Option<core_sky::CoreSky>,
    /// Gizmo lines: opaque ones (`RenderTypes.lines`, which writes depth),
    /// translucent ones (`RenderTypes.linesTranslucentNoDepthWrite`) and the
    /// always-on-top ones.
    debug_lines_opaque: wgpu::RenderPipeline,
    debug_lines: wgpu::RenderPipeline,
    debug_lines_on_top: wgpu::RenderPipeline,
    debug_crosshair: Option<wgpu::RenderPipeline>,
    /// The debug crosshair and the first-person hand in `overlay_format`, where it isn't the
    /// frame's.
    overlay_crosshair: Option<wgpu::RenderPipeline>,
    overlay_hand: Option<wgpu::RenderPipeline>,
    debug_line_buffer: wgpu::Buffer,
    debug_line_capacity: u64,
    /// The terrain resources with the debug crosshair's projection
    /// (`crosshair_globals`).
    crosshair_group: wgpu::BindGroup,
    crosshair_globals: wgpu::Buffer,
    /// The water mirror: its solid terrain pipeline, its camera's globals and their group,
    /// and its targets once drawn.
    solid_mirror: wgpu::RenderPipeline,
    mirror_globals: wgpu::Buffer,
    mirror_group: wgpu::BindGroup,
    mirror: Option<MirrorTarget>,
    /// Native mods' world views' targets and pipelines, and the largest side the device gives one.
    world_views: world_views::Kept,
    world_view_side: u32,
    entity_pipeline: wgpu::RenderPipeline,
    static_entity_pipeline: wgpu::RenderPipeline,
    static_entity_group: wgpu::BindGroup,
    static_entity_transform: wgpu::Buffer,
    static_entity_camera: Option<DVec3>,
    static_entity_catalog_revision: u64,
    static_entity_frame_stats: StaticCacheFrameStats,
    hdr_mark_pipeline: wgpu::RenderPipeline,
    /// `hdr_mark_pipeline` at `entity_pipeline`'s depth bias, for what that drew.
    hdr_entity_mark_pipeline: wgpu::RenderPipeline,
    hdr_hand_unmark_pipeline: wgpu::RenderPipeline,
    hdr_mark_lightning_pipeline: wgpu::RenderPipeline,
    weather: weather::Weather,
    crumbling_pipeline: wgpu::RenderPipeline,
    shadow_blob_pipeline: wgpu::RenderPipeline,
    translucent_item_pipeline: wgpu::RenderPipeline,
    /// `RenderPipelines.WATER_MASK`: depth writes only, no colour.
    water_mask_pipeline: wgpu::RenderPipeline,
    cloud_pipeline: wgpu::RenderPipeline,
    hdr_cloud_mark_pipeline: wgpu::RenderPipeline,
    cloud_mesh: crate::clouds::Mesh,
    cloud_buffer: wgpu::Buffer,
    cloud_capacity: u64,
    cloud_uniform: wgpu::Buffer,
    cloud_group: wgpu::BindGroup,
    additive_pipeline: wgpu::RenderPipeline,
    dragon_rays_pipeline: wgpu::RenderPipeline,
    /// `RenderPipelines.ENERGY_SWIRL`: additive, unlit, the sprite repeated within itself.
    energy_swirl_pipeline: wgpu::RenderPipeline,
    lightning_pipeline: wgpu::RenderPipeline,
    lightning: lightning::Renderer,
    see_through_pipeline: wgpu::RenderPipeline,
    /// Native mods' world primitives (`Frame::world_primitives`).
    world_primitives: world_primitives::WorldPrimitives,
    text_polygon_offset_pipeline: wgpu::RenderPipeline,
    world_text_pipeline: wgpu::RenderPipeline,
    fluid_pipeline: wgpu::RenderPipeline,
    terrain_model_pipelines: [wgpu::RenderPipeline; 3],
    celestial: celestial::Celestial,
    celestial_buffer: wgpu::Buffer,
    end_sky: end_sky::EndSky,
    stars: stars::Stars,
    /// Custom sky layers (`crate::custom_sky`), and the frame's from the game.
    custom_sky: custom_sky::CustomSky,
    custom_sky_draws: Vec<crate::custom_sky::Draw>,
    /// The glowing entities' outline (`LevelRenderer.entityOutlineTarget`).
    entity_outline: entity_outline::EntityOutline,
    panorama: panorama::Panorama,
    /// `RenderPipelines.WORLD_BORDER` and its buffers.
    border_pipeline: wgpu::RenderPipeline,
    border_group: wgpu::BindGroup,
    border_uniform: wgpu::Buffer,
    border_vertices: wgpu::Buffer,
    /// Screen effects: alpha-blended, then vignette (`dst * (1 - src)`)
    /// pipelines, each for every sampling (index `blend * 4 + sampling`).
    screen_pipelines: [wgpu::RenderPipeline; 16],
    screen_buffer: wgpu::Buffer,
    screen_capacity: u64,
    entity_group: wgpu::BindGroup,
    /// The layout of `entity_group`, for the native shadow pass.
    entity_bindings: wgpu::BindGroupLayout,
    /// Generation-owned prepared model geometry. World block entities instance it
    /// when the draw is not kept, spinning, or claimed by a shader pack.
    prepared_models: Option<PreparedModels>,
    prepared_draws: Vec<PreparedModelDraw>,
    /// Ordinary-stream cuts for `prepared_draws`, filled once per frame.
    prepared_cuts: Vec<u32>,
    prepared_ranges: Vec<prepared_model::OrderedSpan>,
    prepared_uploaded: Vec<PreparedModelInstance>,
    prepared_instance_buffer: wgpu::Buffer,
    prepared_instance_capacity: u64,
    /// The entity resources with the hand's projection (`hand_globals`).
    hand_group: wgpu::BindGroup,
    hand_globals: wgpu::Buffer,
    entity_buffer: wgpu::Buffer,
    entity_capacity: u64,
    static_entity_sections: HashMap<[i32; 3], GpuStaticEntitySection>,
    static_entity_indices: wgpu::Buffer,
    static_entity_index_quads: u32,
    entity_batch: EntityBatch,
    /// Ordinary entity models posed on the GPU; `None` without compute shaders.
    entity_skin: Option<entity_skin::EntitySkin>,
    group: wgpu::BindGroup,
    fluid_group: wgpu::BindGroup,
    fluid_sampler: wgpu::Sampler,
    terrain_model_group: wgpu::BindGroup,
    terrain_model_sampler: wgpu::Sampler,
    globals: wgpu::Buffer,
    /// Resource-pack normal/specular maps, split like the colour sprite arrays.
    pbr_maps: Option<Vec<[wgpu::Texture; 2]>>,
    lightmap: wgpu::Texture,
    /// The block sprites, `sprite_split` layers per array (the first is
    /// `sprites` in terrain.wgsl, the others `sprites_1`...).
    sprite_textures: Vec<wgpu::Texture>,
    /// Block sprite layers per array on this device ([`layer_split`]).
    sprite_split: u32,
    /// A 1-layer array bound where a pack needs fewer arrays.
    sprite_unused: wgpu::TextureView,
    /// The sprite arrays as the terrain samples them (`SPRITE_ARRAYS` views, `sprite_unused`
    /// past the ones in use, the Mipmap Levels option's levels), for native mods
    /// (`WorldDrawn::block_sprites`).
    sprite_views: Vec<wgpu::TextureView>,
    /// Nearest, repeating, nearest mip level: for drawing a sprite's texels sharp elsewhere.
    sprite_nearest: wgpu::Sampler,
    mip_levels: u32,
    /// The mipmap levels sampled (the Mipmap Levels option), at most `mip_levels - 1`.
    mipmaps: u32,
    terrain_layout: wgpu::BindGroupLayout,
    /// terrain.wgsl's `SPRITE_ARRAYS` and `SPRITE_SPLIT`, for a shader pack's terrain.
    terrain_sprites: [f64; 2],
    sprite_sampler: wgpu::Sampler,
    /// `Options.textureFiltering`: 0 none, 1 RGSS, 2 anisotropic.
    texture_filtering: u32,
    /// The anisotropic sampler's clamp (`Options.maxAnisotropyValue`); 1 unless anisotropic.
    max_anisotropy: u16,
    lightmap_view: wgpu::TextureView,
    lightmap_sampler: wgpu::Sampler,
    animations: animate::SpriteAnimator,
    material_animations: Vec<SpriteAnimation>,
    active_animated_layers: Vec<u64>,
    uploads: upload::Uploads,
    /// Section meshes are staged through their own belt into
    /// `section_copies`, which the next frame records its commands after:
    /// `Queue::write_buffer` would make a staging buffer per write.
    section_uploads: upload::Uploads,
    section_copies: Option<wgpu::CommandEncoder>,
    /// Bytes staged in `section_copies`.
    section_copy_bytes: u64,
    /// When a large translucent page may start to shrink (`shrink_translucent`).
    shrink_gate: ShrinkGate,
    /// Whether `lightmap` holds `lightmap_pixels` yet.
    lightmap_uploaded: bool,
    sprite_minimum_size: u32,
    /// Each sprite layer's own side, before the arrays scaled it up (`mipmap::Settings::native_size`).
    sprite_native: Vec<u32>,
    /// The Mipmap Levels option (`set_mipmap_levels`), which sets a shader pack atlas's padding.
    mipmap_option: u32,
    indices: wgpu::Buffer,
    index_quads: u32,
    instances: wgpu::Buffer,
    instance_capacity: u64,
    outline: wgpu::Buffer,
    depth: wgpu::TextureView,
    sections: resident::Resident<SectionGpu>,
    lods: FxHashMap<crate::lod::TileKey, lod::Tile>,
    /// The same tiles by cell (`lod::TileGrid`), to test them a handful of
    /// boxes at a time (`draw_frame`).
    lod_cells: lod::TileGrid,
    /// Far tiles' occlusion culling slots (`lod::Tile::cull_slot`): those
    /// free, and the next never used.
    lod_cull_free: Vec<u32>,
    lod_cull_next: u32,
    /// The chunk columns the resident tiles leave out (`Lod::take_cutout`):
    /// their sections are drawn even past the render distance, so terrain
    /// leaving it never opens a hole before the tiles fill it in.
    lod_cutout: crate::fast_hash::FxHashSet<(i32, i32)>,
    /// Removed sections of `lod_cutout` columns, released once tiles cover them.
    lod_retiring: crate::fast_hash::FxHashSet<SectionKey>,
    /// Removed sections fading out over far terrain (`leave_section`).
    lod_leaving: Vec<(SectionKey, std::time::Instant)>,
    /// Finished tile meshes `upload_lods` has yet to upload.
    lod_pending: Option<lod::Pending>,
    detail_columns: FxHashMap<(i32, i32), SectionMask>,
    /// Bumped whenever a column becomes or stops being complete.
    detail_revision: u64,
    arena: arena::Arena,
    /// Translucent quads' own arena, with large pages, while the layout is split
    /// (`translucent_split`): the back-to-front list then stays in one page (and
    /// one sorted-index page) and draws in a few multi-draws rather than one per
    /// few sections.
    translucent_arena: Option<arena::Arena>,
    /// Translucent Batching (`Graphics::translucent_batching`, read when a renderer
    /// is made), where it is offered (`graphics::offer_translucent_batching`).
    translucent_batching: bool,
    /// Every section and far tile was dropped to change the layout since the game
    /// last asked (`take_dropped_sections`).
    sections_dropped: bool,
    sorted_indices: arena::Arena,
    /// Buffers of released arena pages, freed a few frames later on a thread of their own.
    graveyard: arena::Graveyard<wgpu::Buffer>,
    translucent_sorter: translucency::Sorter,
    translucency_cursor: usize,
    /// `DrawIndexedIndirectArgs` for this frame's solid then translucent draws.
    indirect: wgpu::Buffer,
    indirect_capacity: u64,
    /// Indirect draws may set `first_instance` (the section origin).
    pub multi_draw: bool,
    /// Per-frame scratch, kept to avoid allocating every frame.
    visible: Vec<resident::Visible>,
    visible_lods: Vec<lod::Visible>,
    /// The sections a shader pack's shadow camera sees (`external::ShadowView`).
    shadow_visible: Vec<resident::Visible>,
    /// The external shader pipeline world frames call (`external`); `None` draws vanilla's way.
    external: Option<Box<dyn external::ExternalPipeline>>,
    translucent_draws: Vec<(u64, usize, [u32; 5], (u32, Option<u32>))>,
    /// Per visible section: the occlusion test that also sees its translucent quads
    /// (`NO_TEST` without one): the section is left to phase 2.
    translucent_tests: Vec<u32>,
    /// This frame's translucent draws that a test sees: (index among them, test).
    translucent_marks: Vec<(u32, u32)>,
    /// Phase 2's draws and the translucent ones after them, with the instance counts the
    /// tests give (`Cull::upload`).
    phase2_upload: Vec<[u32; 5]>,
    origins: Vec<[f32; 4]>,
    /// `DrawIndexedIndirectArgs` words: index count, instance count, first
    /// index, base vertex, first instance.
    draws: Vec<[u32; 5]>,
    /// Vertex arena page and optional sorted index arena page of each draw:
    /// with sorted indices (a translucent draw), a page of `translucent_arena`
    /// when there is one.
    draw_pages: Vec<(u32, Option<u32>)>,
    /// Solid draws left to occlusion culling's phase 2, while the list is
    /// built, and their pages.
    phase2_draws: Vec<[u32; 5]>,
    phase2_pages: Vec<(u32, Option<u32>)>,
    /// Each section's solid draws, to group them by arena page.
    solid_runs: page_runs::PageRuns,
    phase2_runs: page_runs::PageRuns,
    /// The shadow camera's draws by page, in bands of distance from the light.
    shadow_runs: page_runs::PageRuns,
    /// Block-entity blocks per meshed section (world position, state id).
    block_entities: FxHashMap<SectionKey, Vec<([i32; 3], u32)>>,
    /// Changes on every mesh upload for that section; block-entity static
    /// geometry also depends on the section's packed light.
    block_entity_section_revisions: FxHashMap<SectionKey, u64>,
    /// Changes only when the block-entity positions/states in a section
    /// change. Empty-section tombstones remain until that section unloads so
    /// the static cache can remove deleted models without a global flush.
    block_entity_membership_revisions: FxHashMap<SectionKey, u64>,
    /// When each section was first uploaded (`RenderSection.uploadedTime`): kept across meshing
    /// it again, gone with its column. Sections that fade in over far terrain are not in it.
    section_uploaded: FxHashMap<SectionKey, std::time::Instant>,
    /// When `section_uploaded` last gained a section: once the fade time has passed since,
    /// every section is whole and frames skip the per-section lookups.
    newest_upload: Option<std::time::Instant>,
    /// When a section last began to fade in over far terrain (`SectionGpu::fading`), for the
    /// same shortcut.
    newest_lod_fade: Option<std::time::Instant>,
    /// Changes whenever `block_entities` does (unique across renderers).
    block_entity_revision: u64,
    /// The light sources of each section and their revision, for `block_light`.
    light_blocks: FxHashMap<SectionKey, Vec<([i32; 3], u32)>>,
    light_revision: u64,
    /// VisGraph face connectivity per meshed section (cave culling).
    occlusion: occlusion::Occlusion,
    /// The column `occlusion`, `sections` and `detail_columns` count in: the
    /// level's (`clear_sections`).
    layout: crate::chunk::ColumnLayout,
    /// Resolution scale, FXAA and TAA (Sunrise settings).
    world_post: world_post::Post,
    /// Frame generation (Sunrise setting).
    frame_gen: frame_gen::FrameGen,
    /// The entities' texture atlas (`entity_group`'s), for frame generation's motion.
    entity_atlas: wgpu::TextureView,
    /// HDR output (Sunrise setting): `Some` while the swapchain is HDR.
    hdr: Option<hdr::Hdr>,
    /// Which pixels of each block sprite may shine in HDR output (terrain.wgsl's
    /// `emission_thresholds`), measured when HDR output turns on.
    emission_stats: hdr_stats::EmissionStats,
    /// What this renderer's destination offers and its device draws.
    output_caps: output::OutputCapabilities,
    /// An output that failed to apply, not tried again until something it needs changes.
    output_rejected: Option<output_apply::Rejected>,
    /// Since when Auto keeps HDR on through the display's loss of its report.
    auto_off_since: Option<std::time::Instant>,
    /// The output applied (`update_output`), with the frame every pipeline that draws it
    /// targets: the swapchain's 8-bit format, or `Rgba16Unorm` when the renderer was made with
    /// HDR output on and Smooth Gradients and the device draws and blends into it, or
    /// `LINEAR_FRAME` (`frame_format`). A 16-bit frame with HDR off since reaches the
    /// swapchain through `readback`.
    output: output::EffectiveOutput,
    /// Whether a pack picked or dropped since the renderer was made has been told it runs on
    /// the frame that exists (`set_native_shader_pack`).
    frame_kept_logged: bool,
    /// The marker file of an HDR output that has not presented a frame yet
    /// (`graphics::HDR_MARKER`).
    hdr_marker: Option<std::path::PathBuf>,
    /// The window's graphics driver (`hdr_start`); `None` offscreen.
    hdr_driver: Option<String>,
    /// The HDR Lighting setting, kept for the HDR output made when HDR turns on.
    hdr_lighting_setting: bool,
    /// The shader pack's post-processing, and the dimension (legacy id) it was loaded for.
    pack_post: Option<(i32, Option<pack_post::PackPost>)>,
    /// A `shaders/` pack whose pipelines build on a thread of their own (`load_shader_pack`).
    pack_loading: Option<PackLoading>,
    /// The post passes of one of Sunrise's own shader packs; never beside `pack_post`'s.
    native: Option<native_pack::NativePack>,
    /// Time the shader pack's passes on the GPU (F3, or a capture made with `--gpu`).
    pass_timing: bool,
    /// Times the parts of passes too (`set_pass_detail`).
    pass_detail: bool,
    /// Least time between timed frames (`set_pass_timing_gap`).
    pass_timing_gap: std::time::Duration,
    /// Whether the timers keep every timed frame for `take_timed_frames`.
    keep_timed_frames: bool,
    /// Times Sunrise's own passes on the GPU, as `pass_timing` does (`own_pass_times`).
    own_timing: Option<pack_timing::PassTiming>,
    /// The entity vertices Sunrise draws itself while a shader pack draws the others.
    pack_entity_rest: Vec<Vertex>,
    /// Set 1 of the cloud pipeline, for a shader pack's `gbuffers_clouds`.
    cloud_bindings: wgpu::BindGroupLayout,
    /// Rain and snow handed to a shader pack's `gbuffers_weather`, once a pack draws them.
    pack_weather: Option<pack_weather::PackWeather>,
    /// The block sprites as one atlas, while a shader pack is loaded.
    pack_atlas: Option<pack_atlas::PackAtlas>,
    pack_material_atlases: Option<[pack_atlas::PackAtlas; 2]>,
    /// The shared atlas's LabPBR normal and specular paged maps (`set_shared_material_maps`),
    /// which shader packs read for entities, items and the hand.
    shared_material_maps: [Option<wgpu::Texture>; 2],
    /// The pack's terrain program reads flat draws' vertices (`pack_terrain_inputs`).
    pack_terrain_flat: bool,
    /// OptiFine/Iris separate ambient occlusion: pass vertex alpha to hz_Color.a.
    pub separate_ao: bool,
    /// Shader-pack terrain pages used by post processing, retained between frames.
    pack_pages: Vec<pack_post::TerrainPage>,
    /// Camera-relative selection outline vertices, retained between frames.
    outline_edges_scratch: Vec<[f32; 6]>,
    /// Selection outline line vertices, retained between frames.
    outline_lines_scratch: Vec<[f32; 8]>,
    /// Debug line vertices, retained between frames.
    debug_lines_scratch: Vec<[f32; 8]>,
    /// Screen effect vertices, retained between frames.
    screen_vertices_scratch: Vec<crate::screen_effects::ScreenVertex>,
    /// LOD leaving chunk columns, retained between frames.
    lod_leaving_scratch: crate::fast_hash::FxHashSet<(i32, i32)>,
    /// Shadow visible boxed sections, retained between frames.
    shadow_boxed_scratch: Vec<resident::Visible>,
    /// Shadow visible seen section keys, retained between frames.
    shadow_seen_scratch: crate::fast_hash::FxHashSet<SectionKey>,
    /// Shadow translucent section draws, retained between frames.
    shadow_translucent_scratch: Vec<(u64, [u32; 5], (u32, Option<u32>))>,
    /// Lava's sprite layers, which a pack's solid terrain draws (`pack_atlas::game_terrain`).
    pack_lava_layers: Vec<u16>,
    pub hud: HudRenderer,
    /// Vanilla's menu background blur, made on first use.
    menu_blur: Option<crate::menu_blur::MenuBlur>,
    /// `post_effect` chains, made on first use.
    post_effects: Option<crate::post_chain::PostEffects>,
    /// Where a screenshot's frame is drawn when the swapchain cannot be read back.
    readback: readback::Readback,
    pub info: wgpu::AdapterInfo,
    pub suspended: bool,
    /// Consecutive frames whose surface acquisition failed validation. The
    /// first gets the swapchain a fresh configuration; later ones reattach
    /// the window, as a lost surface does.
    acquire_validation_failures: u32,
    /// A surface from `recreate_surface` whose configuration failed. wgpu
    /// panics when such a surface is acquired from (it never had one that
    /// worked), so frames configure it again first and skip until it takes.
    surface_unconfigured: bool,
    /// The last frame's statistics, all of one frame (`show_stats`): its sections drawn
    /// (F3's `C:`, the world icon's check), section draws (face ranges and translucent
    /// sections) and the draw calls the backend issued for them. A CPU frame's are its own;
    /// in GPU frames (`gpu_culling`) they are the newest GPU frame's whose counts came back,
    /// a frame or a few earlier (`stats_lag`).
    pub drawn_sections: usize,
    pub section_draws: usize,
    pub section_draw_calls: usize,
    /// `draw_frame`'s frames past its early returns, and the one whose statistics are shown.
    frame_serial: u64,
    stats_frame: u64,
    pub uploaded_bytes: u64,
    started: Instant,
    /// The next frame to write to a PNG path (control socket, F2, world icon).
    pub screenshot: Option<ScreenshotRequest>,
    /// Readback outcome: CPU pixels for the game to approve before writing, or an error.
    pub screenshot_result: Option<std::result::Result<PendingScreenshot, String>>,
    screenshot_readbacks: screenshot_readback::Readbacks,
    /// Screenshot Quality (Sunrise setting): a `scaled` screenshot's frame is this many
    /// times the game's resolution (1, 2, 4 or 8).
    pub screenshot_quality: u32,
    /// How many times the pending scaled screenshot ran out of video memory: each
    /// halves its size, down to the window's.
    screenshot_halvings: u32,
    /// Brightness option (lightmap gamma), 0..1.
    pub gamma: f32,
    /// Optional full-light lightmap (Sunrise Extras).
    pub fullbright: bool,
    /// The Improved transparency option (`GameRenderer.useImprovedTransparency`).
    pub improved_transparency: bool,
    /// Its passes and targets, made when first drawn with.
    oit: Option<oit::Oit>,
    /// Advance animated block textures (Sunrise setting; vanilla always does).
    pub animated_textures: bool,
    present_modes: Vec<wgpu::PresentMode>,
    /// CPU copies of the section and far tile meshes uploaded while `render-capture` is on
    /// (`keep_capture_copies`, or `SUNRISE_RENDER_CAPTURE` at creation); `None` otherwise.
    capture_copies: Option<capture::Copies>,
    /// A scene to write from the next frame (`sunrise-ctl render-capture DIR`).
    pub capture: Option<capture::Request>,
    /// Outcome of the last capture: its summary, or the error.
    pub capture_result: Option<std::result::Result<serde_json::Value, String>>,
}
/// Every layer's mip levels, layer major (`first` is the first layer's
/// index into `settings`), made by `MipmapGenerator`.
/// Block sprite layers per texture array at most: the
/// `max_texture_array_layers` of most GPUs. Mods add sprites past it into more
/// arrays (terrain.wgsl's `sprite`); a GPU with fewer layers per array gets a
/// smaller split ([`layer_split`]).
pub(crate) const SPRITE_SPLIT: u32 = 2048;
/// A smaller split from `SUNRISE_SPRITE_SPLIT` (dev only: vanilla's sprites
/// drawn from several arrays must look the same).
fn requested_split() -> Option<u32> {
    std::env::var("SUNRISE_SPRITE_SPLIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| (1..=SPRITE_SPLIT).contains(&n))
}
fn adapter_requested_split(info: &wgpu::AdapterInfo, asked: Option<u32>, n: u32) -> Option<u32> {
    // Metal can fail on big mipmapped arrays (#8471): 512 layers each, more only for n > 8 x 512.
    asked.or((info.backend == wgpu::Backend::Metal).then(|| n.div_ceil(SPRITE_ARRAYS).max(512)))
}
/// The split the CPU mip chains are cut by before the adapter is known.
fn sprite_split() -> u32 {
    requested_split().unwrap_or(SPRITE_SPLIT)
}
/// The block sprite layers per texture array on an adapter that holds
/// `max_layers` layers per array: [`SPRITE_SPLIT`], `requested`
/// (`SUNRISE_SPRITE_SPLIT`) or `max_layers`, whichever is smallest, with a full
/// array still fitting once [`array_layers`] pads it for OpenGL. Errors when
/// `layers` sprites need more than the [`SPRITE_ARRAYS`] arrays terrain.wgsl binds.
fn layer_split(
    max_layers: u32,
    gl: bool,
    requested: Option<u32>,
    layers: u32,
) -> std::result::Result<u32, String> {
    let mut split = requested.unwrap_or(SPRITE_SPLIT).min(max_layers).max(1);
    if split > 1 && array_layers(split, gl) > max_layers {
        split -= 1;
    }
    if layers > split * SPRITE_ARRAYS {
        return Err(format!(
            "GPU supports {max_layers} texture layers per array; {layers} block textures need \
             more than the {SPRITE_ARRAYS} arrays of {split} Sunrise draws from"
        ));
    }
    Ok(split)
}
/// Sprite arrays terrain.wgsl binds (`sprites`, `sprites_1`...).
const SPRITE_ARRAYS: u32 = 8;
/// The layers to give a square texture of `count` layers viewed as a 2D
/// array. wgpu's OpenGL backend picks a texture's target from its layer
/// count when it is created: one layer makes a 2D texture and a multiple of
/// six a cube map, neither of which can be viewed as an array.
pub(crate) fn array_layers(count: u32, gl: bool) -> u32 {
    match count {
        _ if !gl => count,
        0 | 1 => 2,
        n if n % 6 == 0 => n + 1,
        n => n,
    }
}
/// The views of terrain.wgsl's sprite arrays, the unused ones a 1-layer
/// stand-in.
fn sprite_views(
    textures: &[wgpu::Texture],
    unused: &wgpu::TextureView,
    mips: Option<u32>,
) -> Vec<wgpu::TextureView> {
    (0..SPRITE_ARRAYS as usize)
        .map(|i| match textures.get(i) {
            Some(t) => t.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                mip_level_count: mips,
                ..Default::default()
            }),
            None => unused.clone(),
        })
        .collect()
}
/// The block sprites sampled sharp, for native mods (`WorldDrawn::block_sprites`): nearest,
/// repeating, the nearest mip level.
const SPRITE_NEAREST: wgpu::SamplerDescriptor<'static> = wgpu::SamplerDescriptor {
    label: Some("Block sprites (nearest)"),
    address_mode_u: wgpu::AddressMode::Repeat,
    address_mode_v: wgpu::AddressMode::Repeat,
    address_mode_w: wgpu::AddressMode::ClampToEdge,
    mag_filter: wgpu::FilterMode::Nearest,
    min_filter: wgpu::FilterMode::Nearest,
    mipmap_filter: wgpu::MipmapFilterMode::Nearest,
    lod_min_clamp: 0.0,
    lod_max_clamp: 32.0,
    compare: None,
    anisotropy_clamp: 1,
    border_color: None,
};
/// The terrain sprites' sampler: LevelRenderer.chunkLayerSampler's linear one (clamped to the
/// edge) in every Texture Filtering mode, with `anisotropy` samples when it is above 1 (wgpu
/// then needs a linear mipmap filter too).
fn sprite_sampler_descriptor(anisotropy: u16) -> wgpu::SamplerDescriptor<'static> {
    wgpu::SamplerDescriptor {
        label: Some("Block sprites (filtered)"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        anisotropy_clamp: anisotropy,
        ..Default::default()
    }
}
/// `Options.maxAnisotropyValue` for a `maxAnisotropyBit`: `1 << bit`, at most wgpu's 16.
fn sampler_anisotropy(bit: i32) -> u16 {
    1u16 << bit.clamp(0, 4)
}
/// terrain.wgsl's group 0.
#[allow(clippy::too_many_arguments)]
fn terrain_group(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    globals: &wgpu::Buffer,
    sprites: &[wgpu::TextureView],
    sprite_sampler: &wgpu::Sampler,
    lightmap: &wgpu::TextureView,
    lightmap_sampler: &wgpu::Sampler,
    emission_thresholds: &wgpu::TextureView,
) -> wgpu::BindGroup {
    let view = |binding, view| wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    };
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: globals.as_entire_binding(),
            },
            view(1, &sprites[0]),
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sprite_sampler),
            },
            view(3, lightmap),
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(lightmap_sampler),
            },
            view(5, &sprites[1]),
            view(6, &sprites[2]),
            view(7, &sprites[3]),
            view(8, &sprites[4]),
            view(9, &sprites[5]),
            view(10, &sprites[6]),
            view(11, &sprites[7]),
            view(12, emission_thresholds),
        ],
    })
}
mod prepared_textures;
pub use prepared_textures::*;
mod solid_pipeline;
pub use solid_pipeline::*;
mod frame;
pub use frame::*;
mod device;
pub use device::*;
/// Logs a step of the renderer's start, for a native crash in the graphics
/// driver: it leaves no panic or crash report, so the log's last line is the
/// only sign of where the game stopped. The log file is written unbuffered;
/// the flush is for loggers that buffer.
fn startup_step(step: std::fmt::Arguments) {
    log::info!("Sunrise renderer: {step}");
    log::logger().flush();
}
/// How far the first frame on a new device got, for `startup_step`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FirstFrame {
    Acquire,
    Submit,
    Drawn,
}
fn on_off(on: bool) -> &'static str {
    if on { "on" } else { "off" }
}
impl Renderer {
    pub async fn new(
        window: Arc<Window>,
        offscreen: bool,
        sprite_pixels: &[u8],
        sprite_size: u32,
        sprite_minimum_size: u32,
        layers: u32,
        sprite_mips: &[crate::mipmap::Settings],
        animations: &[SpriteAnimation],
        hud_atlas: (&[u8], u32, u32),
        glint: (&[u8], u32, u32),
        glint_armor: (&[u8], u32, u32),
        forcefield: (&[u8], u32),
        panorama_faces: Arc<panorama::Faces>,
        prepared: Option<PreparedTextures>,
        previous: Option<&Self>,
        api: GraphicsApi,
        translucent_batching: bool,
        (shader_pack, pack_starts_wide): (bool, bool),
        // The settings and the pack's `request` pick the frame (`output::resolve_frame`),
        // with what this window's display reported to the renderer before (`display`).
        settings: &crate::graphics::Graphics,
        request: output::FrameRequest,
        display: output::DisplayReading,
    ) -> Result<Self> {
        let Opened {
            instance,
            surface,
            adapter,
            device,
            queue,
            device_lost,
            configured,
        } = match previous {
            // Reload resources on the same device; replacing live Vulkan
            // devices can invalidate the driver's swapchain state on the same
            // window.
            Some(renderer) => {
                describe_adapter(&renderer.adapter);
                check_layers(&renderer.adapter, layers)?;
                Opened {
                    surface: if offscreen {
                        None
                    } else {
                        let spare = renderer
                            .spare_surface
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .take();
                        Some(match spare {
                            Some(surface) => surface,
                            None => renderer.instance.create_surface(window.clone())?,
                        })
                    },
                    instance: renderer.instance.clone(),
                    adapter: renderer.adapter.clone(),
                    device: renderer.device.clone(),
                    queue: renderer.queue.clone(),
                    device_lost: renderer.device_lost.clone(),
                    configured: None,
                }
            }
            None => {
                let _step = crate::spans::startup::step("renderer.open_device");
                open_device(&window, offscreen, api, layers).await?
            }
        };
        // Computed once: the sprite arrays, the shaders' constants and the
        // shader pack atlas all cut the sprites by it.
        let split = check_layers(&adapter, layers)?;
        let info = adapter.get_info();
        surface_setup::log_sprite_split(&info, split, layers);
        crate::graphics::suggest_reduced_far_detail(crate::graphics::lod_capped(&info));
        let size = window.inner_size();
        let max_side = device.limits().max_texture_dimension_2d;
        let configured = match (&surface, configured) {
            (Some(_), Some(configured)) => Some(configured),
            (Some(surface), None) => Some(set_up_surface(
                surface,
                &adapter,
                &device,
                &device_lost,
                surface.get_capabilities(&adapter),
                size,
                previous.map(|renderer| renderer.config.present_mode),
                previous.is_none(),
            )?),
            (None, _) => None,
        };
        let (config, present_modes, hdr_output) = if let Some((config, caps)) = configured {
            let hdr_output = hdr::output(&caps);
            if previous.is_none() {
                log::info!(
                    "Sunrise renderer: HDR output {}",
                    hdr_output.map_or("not offered by this display", hdr::Output::name)
                );
            }
            (config, caps.present_modes, hdr_output)
        } else {
            let format = wgpu::TextureFormat::Bgra8Unorm;
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                format,
                width: size.width.clamp(1, max_side),
                height: size.height.clamp(1, max_side),
                present_mode: wgpu::PresentMode::Immediate,
                desired_maximum_frame_latency: 2,
                alpha_mode: wgpu::CompositeAlphaMode::Opaque,
                view_formats: vec![format],
                color_space: wgpu::SurfaceColorSpace::default(),
            };
            let hdr_output = hdr::offscreen_output();
            if let Some(output) = hdr_output
                && previous.is_none()
            {
                log::info!(
                    "Sunrise renderer: HDR output {} offscreen (SUNRISE_HDR_OUTPUT)",
                    output.name()
                );
            }
            (config, Vec::new(), hdr_output)
        };
        let offscreen = surface
            .is_none()
            .then(|| Self::offscreen_target(&device, &config));
        let wide_ok = draws_wide_frame(
            device.features(),
            adapter.get_texture_format_features(WIDE_FRAME),
        );
        let float_ok = draws_frame(adapter.get_texture_format_features(LINEAR_FRAME));
        if request.linear_pack && !float_ok {
            log::warn!(
                "Sunrise renderer: the device can't draw a float frame; the shader API 0.2 pack runs on the 8-bit one"
            );
        }
        let coverage_ok = hud_coverage::supported(
            device.limits().max_color_attachments,
            adapter.get_downlevel_capabilities().flags,
            adapter.get_texture_format_features(hud_coverage::FORMAT),
        );
        let hdr_driver = surface.is_some().then(|| output::window_driver(&info));
        let output_caps = output::OutputCapabilities {
            hdr: hdr_output,
            wide_frame: wide_ok,
            float_frame: float_ok,
            hud_layer: coverage_ok,
            held: output::held(settings, hdr_driver.as_deref()),
        };
        let frame = output::resolve_frame(settings, request, output_caps, display, config.format);
        let (frame_format, hud_coverage, overlay_format) =
            (frame.format, frame.hud_layer, frame.overlay_format);
        let linear_light = frame.linear_light;
        if previous.is_none() {
            startup_step(format_args!("creating the pipelines"));
        }
        // A renderer on the same device takes the previous one's shader
        // modules, layouts and pipelines where they come out the same.
        let reuse = previous.map_or_else(Default::default, |r| r.reuse.clone());
        let generation = reuse.begin();
        let kept = reuse.on(&device);
        let pipelines_step = crate::spans::startup::step("renderer.pipelines_textures");
        let terrain_step = crate::spans::startup::step("renderer.terrain_pipelines");
        let gpu_timer = timing::GpuTimer::new(&device, &queue);
        let shader = kept.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sunrise terrain"),
            source: wgpu::ShaderSource::Wgsl(terrain_shader().into()),
        });
        let entry = |binding, visibility, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty,
            count: None,
        };
        let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let sprite_array = wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2Array,
            multisampled: false,
        };
        let sprite_arrays = layers.div_ceil(split).max(1);
        if sprite_arrays > 1 {
            log::info!("Sunrise: {layers} block textures in {sprite_arrays} arrays");
        }
        let sprite_constants = [
            ("SPRITE_ARRAYS", f64::from(sprite_arrays)),
            ("SPRITE_SPLIT", f64::from(split)),
        ];
        let bindings = kept.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sunrise terrain bindings"),
            entries: &[
                entry(
                    0,
                    vf,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                entry(
                    1,
                    vf,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                ),
                entry(
                    1 + 1,
                    vf,
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                ),
                entry(
                    3,
                    vf,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                entry(
                    4,
                    vf,
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                ),
                entry(5, vf, sprite_array),
                entry(6, vf, sprite_array),
                entry(7, vf, sprite_array),
                entry(8, vf, sprite_array),
                entry(9, vf, sprite_array),
                entry(10, vf, sprite_array),
                entry(11, vf, sprite_array),
                entry(
                    12,
                    vf,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
            ],
        });
        let layout = kept.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sunrise terrain layout"),
            bind_group_layouts: &[Some(&bindings)],
            ..Default::default()
        });
        let fluid_pipeline = fluid::pipeline(
            &device,
            Some(&layout),
            &shader,
            config.format.remove_srgb_suffix(),
            &sprite_constants,
        );
        let terrain_model_pipelines = terrain_model::pipelines(
            &device,
            Some(&layout),
            &shader,
            config.format.remove_srgb_suffix(),
            &sprite_constants,
        );
        let instance_attributes = wgpu::vertex_attr_array![5=>Float32x4];
        let terrain_buffers = [
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<TerrainVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &TERRAIN_ATTRIBUTES,
            }),
            Some(wgpu::VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &instance_attributes,
            }),
        ];
        let alpha_blend = wgpu::BlendState::ALPHA_BLENDING;
        let debug_line_attributes =
            wgpu::vertex_attr_array![0=>Float32x3,1=>Float32x3,2=>Unorm8x4,3=>Float32];
        let debug_line_buffers = [Some(wgpu::VertexBufferLayout {
            array_stride: 32,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &debug_line_attributes,
        })];
        // The solid terrain's fragment constants without HDR output's lighting, for the
        // frames HDR is off (`SolidPipeline`).
        let plain_constants = [
            sprite_constants[0],
            sprite_constants[1],
            ("HDR_LIGHTING", 0.0),
        ];
        // `hud`: the debug crosshair, drawn with the HUD in the overlays' format, with its
        // coverage or without.
        let pipeline_with = |constants: &[(&str, f64)],
                             layout: &wgpu::PipelineLayout,
                             label: &str,
                             vertex: &str,
                             fragment: &str,
                             buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
                             topology: wgpu::PrimitiveTopology,
                             cull: Option<wgpu::Face>,
                             depth_write: bool,
                             depth_test: bool,
                             blend: Option<wgpu::BlendState>,
                             hud: Option<bool>| {
            let mut constants = constants.to_vec();
            // The debug crosshair draws with the HUD, in gamma.
            constants.extend(linear_blend::frame_constants(
                frame_format,
                linear_light && hud.is_none(),
            ));
            let desc = wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vertex),
                    compilation_options: Default::default(),
                    buffers,
                },
                primitive: wgpu::PrimitiveState {
                    topology,
                    cull_mode: cull,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(depth_write),
                    depth_compare: Some(if depth_test {
                        wgpu::CompareFunction::GreaterEqual
                    } else {
                        wgpu::CompareFunction::Always
                    }),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fragment),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        ..Default::default()
                    },
                    targets: &[Some(wgpu::ColorTargetState {
                        format: if hud.is_some() {
                            overlay_format
                        } else {
                            frame_format
                        },
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            };
            if hud == Some(true) {
                hud_coverage::pipeline(
                    &device,
                    &desc,
                    wgpu::ShaderSource::Wgsl(terrain_shader().into()),
                )
                .expect("the debug crosshair's coverage")
            } else {
                kept.create_render_pipeline(&desc)
            }
        };
        let pipeline_in = |layout: &wgpu::PipelineLayout,
                           label: &str,
                           vertex: &str,
                           fragment: &str,
                           buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
                           topology: wgpu::PrimitiveTopology,
                           cull: Option<wgpu::Face>,
                           depth_write: bool,
                           depth_test: bool,
                           blend: Option<wgpu::BlendState>| {
            pipeline_with(
                &sprite_constants,
                layout,
                label,
                vertex,
                fragment,
                buffers,
                topology,
                cull,
                depth_write,
                depth_test,
                blend,
                None,
            )
        };
        // A solid terrain pipeline (`cull` faces culled, depth written, no blend) without HDR
        // output's lighting, and with it where the display offers HDR output.
        let solid_in = |layout: &wgpu::PipelineLayout,
                        label: &str,
                        vertex: &str,
                        fragment: &str,
                        buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
                        cull: wgpu::Face| {
            let build = |constants: &[(&str, f64)]| {
                pipeline_with(
                    constants,
                    layout,
                    label,
                    vertex,
                    fragment,
                    buffers,
                    wgpu::PrimitiveTopology::TriangleList,
                    Some(cull),
                    true,
                    true,
                    None,
                    None,
                )
            };
            SolidPipeline {
                plain: build(&plain_constants),
                hdr: hdr_output.is_some().then(|| build(&sprite_constants)),
            }
        };
        let pipeline = |label: &str,
                        vertex: &str,
                        fragment: &str,
                        buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
                        topology: wgpu::PrimitiveTopology,
                        cull: Option<wgpu::Face>,
                        depth_write: bool,
                        depth_test: bool,
                        blend: Option<wgpu::BlendState>| {
            pipeline_in(
                &layout,
                label,
                vertex,
                fragment,
                buffers,
                topology,
                cull,
                depth_write,
                depth_test,
                blend,
            )
        };
        let tri = wgpu::PrimitiveTopology::TriangleList;
        let solid = solid_in(
            &layout,
            "Terrain solid",
            "vs_terrain",
            "fs_solid",
            &terrain_buffers,
            wgpu::Face::Back,
        );
        // `solid` in two passes (terrain.wgsl's `opaque_quad`, `mesher::Part`), for the
        // world pass without a shader pack: no cutout first, through a shader that never
        // discards, then the rest.
        let solid_opaque = solid_in(
            &layout,
            "Terrain solid opaque",
            "vs_terrain_opaque",
            "fs_opaque",
            &terrain_buffers,
            wgpu::Face::Back,
        );
        let solid_cutout = solid_in(
            &layout,
            "Terrain solid cutout",
            "vs_terrain_cutout",
            "fs_solid",
            &terrain_buffers,
            wgpu::Face::Back,
        );
        // The water mirror's camera sees the world from the other side of the plane, and a
        // mirrored world view's picture is flipped (`world_views::Flipped`): each turns the
        // winding round, so their terrain culls the front.
        let front = |label, fragment, blend| {
            pipeline(
                label,
                "vs_terrain",
                fragment,
                &terrain_buffers,
                tri,
                Some(wgpu::Face::Front),
                true,
                true,
                blend,
            )
        };
        let solid_mirror = front("Terrain solid (mirrored)", "fs_solid_mirror", None);
        let flipped = solid_in(
            &layout,
            "Terrain solid (flipped)",
            "vs_terrain",
            "fs_solid",
            &terrain_buffers,
            wgpu::Face::Front,
        );
        let alpha = Some(alpha_blend);
        let translucent_flipped = front("Terrain translucent (flipped)", "fs_translucent", alpha);
        let translucent = pipeline(
            "Terrain translucent",
            "vs_terrain",
            "fs_translucent",
            &terrain_buffers,
            tri,
            Some(wgpu::Face::Back),
            true,
            true,
            Some(alpha_blend),
        );
        let translucent_mark = pipeline(
            "Terrain translucent HDR mark",
            "vs_terrain",
            "fs_translucent_mark",
            &terrain_buffers,
            tri,
            Some(wgpu::Face::Back),
            false,
            true,
            Some(flat::ALPHA_ONLY),
        );
        let sky = pipeline(
            "Sky",
            "vs_sky",
            "fs_sky",
            &[],
            tri,
            None,
            false,
            false,
            None,
        );
        let sky_base = pipeline(
            "Sky base",
            "vs_sky",
            "fs_sky_base",
            &[],
            tri,
            None,
            false,
            false,
            None,
        );
        let sunrise_attributes = wgpu::vertex_attr_array![0=>Float32x4, 1=>Float32];
        let sunrise_buffers = [Some(wgpu::VertexBufferLayout {
            array_stride: 20,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &sunrise_attributes,
        })];
        let sunrise_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            // The sky's HDR alpha is the destination's; the fan only tints it.
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let sunrise_pipeline = pipeline(
            "Sunrise fan",
            "vs_sunrise",
            "fs_sunrise",
            &sunrise_buffers,
            tri,
            None,
            false,
            false,
            Some(sunrise_blend),
        );
        let dark_disc = pipeline(
            "Dark disc",
            "vs_sky",
            "fs_dark_disc",
            &[],
            tri,
            None,
            false,
            false,
            Some(alpha_blend),
        );
        let depth_reset = kept.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Hand depth reset"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_sky"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_depth_reset"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &sprite_constants,
                    ..Default::default()
                },
                targets: &[Some(wgpu::ColorTargetState {
                    format: frame_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::empty(),
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let lines = pipeline(
            "Block outline",
            "vs_line",
            "fs_line",
            &debug_line_buffers,
            wgpu::PrimitiveTopology::TriangleStrip,
            None,
            true,
            true,
            Some(alpha_blend),
        );
        // RenderTypes.secondaryBlockOutline: the black stroke under a high contrast outline.
        let lines_secondary = pipeline(
            "Block outline secondary",
            "vs_line",
            "fs_line",
            &debug_line_buffers,
            wgpu::PrimitiveTopology::TriangleStrip,
            None,
            false,
            true,
            Some(alpha_blend),
        );
        // RenderPipelines.LINES: DepthStencilState.DEFAULT, so an opaque hitbox or
        // chunk border keeps the water and clouds drawn after it behind it.
        let debug_lines_opaque = pipeline(
            "Debug lines opaque",
            "vs_debug_line",
            "fs_debug_line",
            &debug_line_buffers,
            wgpu::PrimitiveTopology::TriangleStrip,
            None,
            true,
            true,
            Some(alpha_blend),
        );
        let debug_lines = pipeline(
            "Debug lines",
            "vs_debug_line",
            "fs_debug_line",
            &debug_line_buffers,
            wgpu::PrimitiveTopology::TriangleStrip,
            None,
            false,
            true,
            Some(alpha_blend),
        );
        let debug_lines_on_top = pipeline(
            "Debug lines on top",
            "vs_debug_line",
            "fs_debug_line",
            &debug_line_buffers,
            wgpu::PrimitiveTopology::TriangleStrip,
            None,
            false,
            false,
            Some(alpha_blend),
        );
        // The debug crosshair with the HUD: with its coverage on the layered frame, and in the
        // overlays' format where that isn't the frame's.
        let crosshair = |coverage| {
            pipeline_with(
                &sprite_constants,
                &layout,
                "3d crosshair",
                "vs_debug_line",
                "fs_debug_line",
                &debug_line_buffers,
                wgpu::PrimitiveTopology::TriangleStrip,
                None,
                false,
                false,
                Some(alpha_blend),
                Some(coverage),
            )
        };
        let debug_crosshair = hud_coverage.then(|| crosshair(true));
        let overlay_crosshair =
            (overlay_format != frame_format || linear_light).then(|| crosshair(false));
        drop(terrain_step);
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sunrise globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let hand_globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("First-person hand globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let crosshair_globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug crosshair globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mirror_globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Water mirror globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let prepared = match prepared {
            Some(prepared) => prepared.resplit(layers, split),
            None => PreparedTextures::with_split(
                sprite_pixels,
                sprite_size,
                layers,
                sprite_mips,
                forcefield,
                split,
            ),
        };
        let mut mip_levels = 1;
        let gl = adapter.get_info().backend == wgpu::Backend::Gl;
        // Keep every colour layer's companion, split by the same device limit. Native
        // material stages read the first array; terrain packs stitch all of them.
        let pbr_maps = prepared
            .pbr
            .map(|(normals, specular, levels, map_layers, size)| {
                let layer_bytes = normals.len() / map_layers as usize;
                (0..map_layers.div_ceil(split))
                    .map(|array| {
                        let first = array * split;
                        let used = (map_layers - first).min(split);
                        let count = array_layers(used, gl);
                        [("LabPBR normals", &normals), ("LabPBR specular", &specular)].map(
                            |(label, data)| {
                                let start = first as usize * layer_bytes;
                                let mut data =
                                    data[start..start + used as usize * layer_bytes].to_vec();
                                data.resize(layer_bytes * count as usize, 0);
                                device.create_texture_with_data(
                                    &queue,
                                    &wgpu::TextureDescriptor {
                                        label: Some(label),
                                        size: wgpu::Extent3d {
                                            width: size,
                                            height: size,
                                            depth_or_array_layers: count,
                                        },
                                        mip_level_count: levels,
                                        sample_count: 1,
                                        dimension: wgpu::TextureDimension::D2,
                                        format: wgpu::TextureFormat::Rgba8Unorm,
                                        usage: wgpu::TextureUsages::TEXTURE_BINDING
                                            | wgpu::TextureUsages::COPY_SRC
                                            | wgpu::TextureUsages::COPY_DST,
                                        view_formats: &[],
                                    },
                                    wgpu::util::TextureDataOrder::LayerMajor,
                                    &data,
                                )
                            },
                        )
                    })
                    .collect()
            });
        let sprite_textures: Vec<wgpu::Texture> = prepared
            .sprites
            .into_iter()
            .enumerate()
            .map(|(array, (mut mips, levels))| {
                let first = array as u32 * split;
                let used = layers.saturating_sub(first).clamp(1, split);
                let count = array_layers(used, gl);
                // The extra layers are empty.
                mips.resize(mips.len() / used as usize * count as usize, 0);
                mip_levels = levels;
                device.create_texture_with_data(
                    &queue,
                    &wgpu::TextureDescriptor {
                        label: Some("Block sprites"),
                        size: wgpu::Extent3d {
                            width: sprite_size,
                            height: sprite_size,
                            depth_or_array_layers: count,
                        },
                        mip_level_count: levels,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::COPY_DST
                            // Shader packs' atlas copies them (pack_atlas.rs).
                            | wgpu::TextureUsages::COPY_SRC
                            | wgpu::TextureUsages::STORAGE_BINDING,
                        view_formats: &[],
                    },
                    wgpu::util::TextureDataOrder::LayerMajor,
                    &mips,
                )
            })
            .collect();
        let material_animations = animations
            .iter()
            .filter(|a| a.has_materials())
            .cloned()
            .collect();
        let animations = animate::SpriteAnimator::new(
            &device,
            animations,
            sprite_size,
            sprite_mips,
            &sprite_textures,
            split,
            animate::SpriteAnimator::compute(&adapter, &device),
        );
        let active_layer_words = animations.active_layer_words();
        let sprite_unused = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Block sprites (unused)"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: array_layers(1, gl),
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });
        let sprite_views = sprite_views(&sprite_textures, &sprite_unused, None);
        let sprite_sampler = device.create_sampler(&sprite_sampler_descriptor(1));
        let sprite_nearest = device.create_sampler(&SPRITE_NEAREST);
        let lightmap = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Lightmap"),
            size: wgpu::Extent3d {
                width: 16,
                height: 16,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let lightmap_view = lightmap.create_view(&Default::default());
        let lightmap_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Lightmap"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let emission_stats =
            hdr_stats::EmissionStats::new(&device, split, sprite_textures.len() as u32);
        let group = terrain_group(
            &device,
            "Sunrise terrain resources",
            &bindings,
            &globals,
            &sprite_views,
            &sprite_sampler,
            &lightmap_view,
            &lightmap_sampler,
            &emission_stats.view,
        );
        let fluid_sampler = device.create_sampler(&fluid::SAMPLER);
        let fluid_group = terrain_group(
            &device,
            "Still fluid resources",
            &bindings,
            &globals,
            &sprite_views,
            &fluid_sampler,
            &lightmap_view,
            &lightmap_sampler,
            &emission_stats.view,
        );
        let terrain_model_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Native cutout model (unmipped)"),
            lod_max_clamp: 0.0,
            ..fluid::SAMPLER
        });
        let terrain_model_group = terrain_group(
            &device,
            "Native cutout model resources",
            &bindings,
            &globals,
            &sprite_views,
            &terrain_model_sampler,
            &lightmap_view,
            &lightmap_sampler,
            &emission_stats.view,
        );
        let crosshair_group = terrain_group(
            &device,
            "Debug crosshair resources",
            &bindings,
            &crosshair_globals,
            &sprite_views,
            &sprite_sampler,
            &lightmap_view,
            &lightmap_sampler,
            &emission_stats.view,
        );
        let mirror_group = terrain_group(
            &device,
            "Water mirror resources",
            &bindings,
            &mirror_globals,
            &sprite_views,
            &sprite_sampler,
            &lightmap_view,
            &lightmap_sampler,
            &emission_stats.view,
        );
        let index_quads = 1 << 16;
        let indices = Self::quad_indices(&device, index_quads);
        let instance_capacity = 4096;
        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Section origins"),
            size: instance_capacity * 16,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let prepared_instance_capacity = 64;
        let prepared_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Prepared model instances"),
            size: prepared_instance_capacity * std::mem::size_of::<PreparedModelInstance>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let outline = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Outline"),
            size: 32 * 2 * MAX_OUTLINE_EDGES,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let debug_line_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug lines"),
            size: 32 * 4096,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (atlas_pixels, atlas_width, atlas_height) = hud_atlas;
        let limit = device.limits().max_texture_dimension_2d;
        if atlas_width.max(atlas_height) > limit {
            return Err(format!(
                "the shared atlas is {atlas_width}x{atlas_height}; this GPU allows {limit}x{limit}"
            )
            .into());
        }
        let atlas = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("Shared atlas"),
                size: wgpu::Extent3d {
                    width: atlas_width,
                    height: atlas_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                // COPY_SRC: a shader pack's weather copies the rain and snow sprites out.
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            atlas_pixels,
        );
        let atlas_view = atlas.create_view(&Default::default());
        // vanilla's `GlintSampler`: enchanted_glint_item as its own texture,
        // repeated per pixel (`clamp: false`), linear without mipmaps
        // (`blur: true`).
        let (glint_pixels, glint_width, glint_height) = glint;
        let glint_texture = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("Enchanted glint"),
                size: wgpu::Extent3d {
                    width: glint_width,
                    height: glint_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            glint_pixels,
        );
        let glint_view = glint_texture.create_view(&Default::default());
        let glint_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Glint"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        // ARMOR_ENTITY_GLINT_TEXTURING's sampler is vanilla's GlintSampler: the
        // same repeat and linear filtering as the item glint.
        let (glint_armor_pixels, glint_armor_width, glint_armor_height) = glint_armor;
        let glint_armor_texture = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("Enchanted armor glint"),
                size: wgpu::Extent3d {
                    width: glint_armor_width,
                    height: glint_armor_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            glint_armor_pixels,
        );
        let glint_armor_view = glint_armor_texture.create_view(&Default::default());
        let mut hud = HudRenderer::new(
            &device,
            overlay_format,
            &atlas_view,
            &glint_view,
            &glint_armor_view,
            &glint_sampler,
            hud_coverage,
        )?;
        hud.atlas_texture = Some(atlas.clone());
        let entity_step = crate::spans::startup::step("renderer.entity_pipelines");
        let entity_shader = kept.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sunrise entities"),
            source: wgpu::ShaderSource::Wgsl(entity_shader().into()),
        });
        let static_entity_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sunrise static block entities"),
            source: wgpu::ShaderSource::Wgsl(static_entity_shader().into()),
        });
        let entity_bindings = kept.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Entity bindings"),
            entries: &entity_binding_entries(),
        });
        let static_entity_bindings =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Static block-entity transform"),
                entries: &[entry(
                    0,
                    wgpu::ShaderStages::VERTEX,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                )],
            });
        let weather = weather::Weather::new(
            &device,
            (frame_format, linear_light),
            &entity_bindings,
            &entity_shader,
        );
        let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Atlas"),
            ..Default::default()
        });
        let entity_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Entity resources"),
            layout: &entity_bindings,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: globals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&atlas_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&lightmap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&lightmap_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&glint_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&glint_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&glint_armor_view),
                },
            ],
        });
        let static_entity_transform =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Static block-entity camera position"),
                contents: bytemuck::bytes_of(&StaticEntityUniform {
                    camera_block: [0; 4],
                    camera_fraction: [0.0; 4],
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let static_entity_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Static block-entity transform"),
            layout: &static_entity_bindings,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: static_entity_transform.as_entire_binding(),
            }],
        });
        let hand_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("First-person hand resources"),
            layout: &entity_bindings,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: hand_globals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&atlas_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&lightmap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&lightmap_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&glint_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&glint_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&glint_armor_view),
                },
            ],
        });
        let entity_layout = kept.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Entity layout"),
            bind_group_layouts: &[Some(&entity_bindings)],
            ..Default::default()
        });
        let static_entity_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Static block-entity layout"),
            bind_group_layouts: &[Some(&entity_bindings), Some(&static_entity_bindings)],
            ..Default::default()
        });
        let cloud_bindings = kept.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cloud bindings"),
            entries: &[entry(
                0,
                wgpu::ShaderStages::VERTEX,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            )],
        });
        let cloud_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cloud transforms and colors"),
            size: std::mem::size_of::<crate::clouds::Uniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cloud_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cloud resources"),
            layout: &cloud_bindings,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: cloud_uniform.as_entire_binding(),
            }],
        });
        let cloud_layout = kept.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cloud layout"),
            bind_group_layouts: &[Some(&entity_bindings), Some(&cloud_bindings)],
            ..Default::default()
        });
        let cloud_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cloud vertices"),
            size: 4 * std::mem::size_of::<Vertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let entity_vertex = [Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ENTITY_ATTRIBUTES,
        })];
        let static_entity_origin_attributes = wgpu::vertex_attr_array![7=>Sint32x3];
        let static_entity_vertex = [
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &ENTITY_ATTRIBUTES,
            }),
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[i32; 3]>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &static_entity_origin_attributes,
            }),
        ];
        let entity_pipeline_with = entity_pipeline_factory(
            &kept,
            &entity_layout,
            &cloud_layout,
            &entity_shader,
            &entity_vertex,
            frame_format,
            linear_light,
        );
        let entity_pipeline = |label, cull_mode, depth_write, compare, bias, blend, clouds| {
            entity_pipeline_with(EntityPipelineOptions {
                label,
                cull_mode,
                depth_write,
                compare,
                bias,
                blend,
                clouds,
                vertex: "vs_entity",
                fragment: "fs_entity",
                write_mask: wgpu::ColorWrites::ALL,
            })
        };
        // RenderPipelines.WATER_MASK: the boat's water patch writes depth only.
        let water_mask_pipeline = entity_pipeline_with(EntityPipelineOptions {
            label: "Water mask",
            cull_mode: None,
            depth_write: true,
            compare: wgpu::CompareFunction::GreaterEqual,
            bias: Default::default(),
            blend: alpha_blend,
            clouds: false,
            vertex: "vs_entity",
            fragment: "fs_water_mask",
            write_mask: wgpu::ColorWrites::empty(),
        });
        let crumbling_pipeline = entity_pipeline(
            "Breaking cracks",
            Some(wgpu::Face::Back),
            false,
            wgpu::CompareFunction::GreaterEqual,
            wgpu::DepthBiasState {
                constant: 10,
                slope_scale: 1.0,
                clamp: 0.0,
            },
            CRUMBLING_BLEND,
            false,
        );
        // RenderPipelines.ENTITY_SHADOW: translucent, back-face culled, depth
        // test without writes. The quads lie exactly on the block tops like
        // vanilla's, facing +Y so looking up from below culls them, but
        // terrain and shadows are separate draw paths, so their fp32 depth
        // interpolations disagree by a few dozen ulps of the fragment depth
        // and the shadow's rim z-fights with the block top (speckle that
        // re-rolls as the camera drifts a subpixel). A constant bias of 128
        // ulps covers that spread at any resolution while staying far below
        // the 0.001 a model's feet sit higher, so shadows never bleed onto
        // the feet.
        let shadow_blob_pipeline = entity_pipeline_with(EntityPipelineOptions {
            label: "Entity shadows",
            cull_mode: Some(wgpu::Face::Back),
            depth_write: false,
            compare: wgpu::CompareFunction::GreaterEqual,
            bias: wgpu::DepthBiasState {
                constant: 128,
                slope_scale: 0.015,
                clamp: 0.0,
            },
            blend: alpha_blend,
            clouds: false,
            vertex: "vs_entity",
            fragment: "fs_entity_shadow",
            write_mask: wgpu::ColorWrites::ALL,
        });
        // RenderPipelines.ITEM_TRANSLUCENT: blended, depth tested and, like
        // every vanilla pipeline that does not turn it off, depth writing.
        // `LevelRenderer.executeClassicTransparency` draws the translucent
        // features before the translucent terrain, so without the writes
        // ice or water behind a dropped slime block or glass item would blend
        // over it and the item would all but vanish (#7031, #7033).
        let translucent_item_pipeline = entity_pipeline(
            "Translucent items",
            None,
            true,
            wgpu::CompareFunction::GreaterEqual,
            Default::default(),
            alpha_blend,
            false,
        );
        let cloud_pipeline = entity_pipeline(
            "Clouds",
            Some(wgpu::Face::Back),
            true,
            wgpu::CompareFunction::GreaterEqual,
            Default::default(),
            alpha_blend,
            true,
        );
        let hdr_cloud_mark_pipeline = entity_pipeline_with(EntityPipelineOptions {
            label: "HDR cloud mark",
            cull_mode: Some(wgpu::Face::Back),
            depth_write: false,
            compare: wgpu::CompareFunction::GreaterEqual,
            bias: Default::default(),
            blend: wgpu::BlendState::REPLACE,
            clouds: true,
            vertex: "vs_entity",
            fragment: "fs_hdr_cloud_mark",
            write_mask: wgpu::ColorWrites::ALPHA,
        });
        // RenderPipelines.TEXT_SEE_THROUGH: translucent, no depth test or
        // writes, culled like every world text pipeline (TEXT_SNIPPET keeps
        // RenderPipeline.Builder's default): a name tag or text display hides
        // its back.
        let see_through_pipeline = entity_pipeline_with(EntityPipelineOptions {
            label: "See-through text",
            cull_mode: Some(wgpu::Face::Back),
            depth_write: false,
            compare: wgpu::CompareFunction::Always,
            bias: Default::default(),
            blend: alpha_blend,
            clouds: false,
            vertex: "vs_entity",
            fragment: "fs_entity_unmarked",
            write_mask: wgpu::ColorWrites::ALL,
        });
        // RenderPipelines.TEXT_POLYGON_OFFSET: a text display's text and
        // background win against the surface they lie on (bias 1, 10), from
        // the front only.
        let text_polygon_offset_pipeline = entity_pipeline_with(EntityPipelineOptions {
            label: "Polygon offset text",
            cull_mode: Some(wgpu::Face::Back),
            depth_write: true,
            compare: wgpu::CompareFunction::GreaterEqual,
            bias: wgpu::DepthBiasState {
                constant: 10,
                slope_scale: 1.0,
                clamp: 0.0,
            },
            blend: alpha_blend,
            clouds: false,
            vertex: "vs_entity",
            fragment: "fs_entity_unmarked",
            write_mask: wgpu::ColorWrites::ALL,
        });
        let world_text_pipeline = world_text::pipeline(
            &device,
            Some(&entity_layout),
            &entity_shader,
            (frame_format, linear_light),
        );
        // RenderPipelines.ENTITY_SOLID: a surface of an entity that is coplanar with
        // terrain wins the tie in vanilla (the boat's floor over the trapdoor on the
        // ground it is stopped by, #5950). Terrain and entities are separate draw
        // paths here, so their fp32 depths of the same plane differ by a few dozen
        // ulps and the tie came out either way, as a jagged strip of the trapdoor
        // through the hull's floor. The same constant as the shadows' covers that
        // spread; a model's feet sit 0.001 above the ground, far beyond it. HDR output's
        // marks on what it draws take it too (`hdr_entity_mark_pipeline`).
        let entity_bias = wgpu::DepthBiasState {
            constant: 128,
            slope_scale: 0.0,
            clamp: 0.0,
        };
        let entity_pipeline = entity_pipeline(
            "Entities",
            None,
            true,
            wgpu::CompareFunction::GreaterEqual,
            entity_bias,
            alpha_blend,
            false,
        );
        // The hand as `entity_pipeline` draws it, in the overlays' format.
        let overlay_hand = (overlay_format != frame_format || linear_light).then(|| {
            entity_pipeline_factory(
                &kept,
                &entity_layout,
                &cloud_layout,
                &entity_shader,
                &entity_vertex,
                overlay_format,
                false,
            )(EntityPipelineOptions {
                label: "First-person hand",
                cull_mode: None,
                depth_write: true,
                compare: wgpu::CompareFunction::GreaterEqual,
                bias: entity_bias,
                blend: alpha_blend,
                clouds: false,
                vertex: "vs_entity",
                fragment: "fs_entity",
                write_mask: wgpu::ColorWrites::ALL,
            })
        });
        let static_entity_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Static block entities"),
                layout: Some(&static_entity_layout),
                vertex: wgpu::VertexState {
                    module: &static_entity_shader,
                    entry_point: Some("vs_static_entity"),
                    compilation_options: Default::default(),
                    buffers: &static_entity_vertex,
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &static_entity_shader,
                    entry_point: Some("fs_entity_unmarked"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &linear_blend::frame_constants(frame_format, linear_light),
                        ..Default::default()
                    },
                    targets: &[Some(wgpu::ColorTargetState {
                        format: frame_format,
                        // fs_entity's discard handles binary cutout alpha. Any
                        // sprite with fractional alpha stays on the blended
                        // dynamic entity path.
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        // RenderPipelines.ENERGY_SWIRL: BlendFunction.ADDITIVE (one, one), no culling; no depth
        // writes either: vanilla sorts the quads far to near, so every face passes the test.
        let one_one = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let energy_swirl_pipeline = entity_pipeline_with(EntityPipelineOptions {
            label: "Energy swirls",
            cull_mode: None,
            depth_write: false,
            compare: wgpu::CompareFunction::GreaterEqual,
            bias: Default::default(),
            blend: wgpu::BlendState {
                color: one_one,
                alpha: one_one,
            },
            clouds: false,
            vertex: "vs_entity",
            fragment: "fs_energy_swirl",
            write_mask: wgpu::ColorWrites::ALL,
        });
        // HDR output's mark on light-giving particles (entity.wgsl fs_hdr_mark).
        let hdr_mark_pipeline = entity_pipeline_with(EntityPipelineOptions {
            label: "HDR particle mark",
            cull_mode: None,
            depth_write: false,
            compare: wgpu::CompareFunction::GreaterEqual,
            bias: Default::default(),
            blend: wgpu::BlendState::REPLACE,
            clouds: false,
            vertex: "vs_entity",
            fragment: "fs_hdr_mark",
            write_mask: wgpu::ColorWrites::ALPHA,
        });
        let hdr_entity_mark_pipeline = entity_pipeline_with(EntityPipelineOptions {
            label: "HDR entity mark",
            cull_mode: None,
            depth_write: false,
            compare: wgpu::CompareFunction::GreaterEqual,
            bias: entity_bias,
            blend: wgpu::BlendState::REPLACE,
            clouds: false,
            vertex: "vs_entity",
            fragment: "fs_hdr_mark",
            write_mask: wgpu::ColorWrites::ALPHA,
        });
        let hdr_hand_unmark_pipeline = entity_pipeline_with(EntityPipelineOptions {
            label: "HDR hand unmark",
            cull_mode: None,
            depth_write: false,
            compare: wgpu::CompareFunction::GreaterEqual,
            bias: entity_bias,
            blend: wgpu::BlendState::REPLACE,
            clouds: false,
            vertex: "vs_entity",
            fragment: "fs_hdr_hand_unmark",
            write_mask: wgpu::ColorWrites::ALPHA,
        });
        // Attribute 7: 6 is the entity vertex buffer's glint UV.
        let lightning_attributes = wgpu::vertex_attr_array![7=>Float32x3];
        let lightning_vertex = [
            entity_vertex[0].clone(),
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 3]>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &lightning_attributes,
            }),
        ];
        // BlendFunction.LIGHTNING: src alpha, one; no depth writes.
        let additive_descriptor = wgpu::RenderPipelineDescriptor {
            label: Some("Additive entities"),
            layout: Some(&entity_layout),
            vertex: wgpu::VertexState {
                module: &entity_shader,
                entry_point: Some("vs_entity"),
                compilation_options: Default::default(),
                buffers: &entity_vertex,
            },
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &entity_shader,
                entry_point: Some("fs_celestial"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &linear_blend::frame_constants(frame_format, linear_light),
                    ..Default::default()
                },
                targets: &[Some(wgpu::ColorTargetState {
                    format: frame_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        };
        let additive_pipeline = kept.create_render_pipeline(&additive_descriptor);
        // `RenderPipelines.DRAGON_RAYS` uses DepthStencilState.DEFAULT:
        // unlike eyes, a ray hides the faces and rays behind it.
        let dragon_rays_pipeline = kept.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Dragon death rays"),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..additive_descriptor.primitive
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                depth_write_enabled: Some(true),
                ..additive_descriptor.depth_stencil.clone().unwrap()
            }),
            ..additive_descriptor.clone()
        });
        // HDR output's mark on lightning bolts: fs_hdr_mark over the bolts' own vertex layout,
        // as hdr_mark_pipeline marks entities and particles (only alpha written).
        let mark_targets = [Some(wgpu::ColorTargetState {
            format: frame_format,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALPHA,
        })];
        let hdr_mark_lightning_pipeline =
            kept.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("HDR lightning mark"),
                vertex: wgpu::VertexState {
                    entry_point: Some("vs_lightning"),
                    buffers: &lightning_vertex,
                    ..additive_descriptor.vertex.clone()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &entity_shader,
                    entry_point: Some("fs_hdr_mark"),
                    compilation_options: Default::default(),
                    targets: &mark_targets,
                }),
                ..additive_descriptor.clone()
            });
        let lightning_pipeline = kept.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Lightning"),
            vertex: wgpu::VertexState {
                entry_point: Some("vs_lightning"),
                buffers: &lightning_vertex,
                ..additive_descriptor.vertex.clone()
            },
            ..additive_descriptor
        });
        let celestial = celestial::Celestial::new(
            &kept,
            &entity_layout,
            &entity_shader,
            &entity_vertex,
            (frame_format, linear_light),
        );
        drop(entity_step);
        // WorldBorderRenderer: misc/forcefield.png as a MipmappedTexture
        // (repeat, linear minification with mipmaps, nearest magnification).
        let forcefield_size = forcefield.1;
        let (forcefield_mips, forcefield_levels) = prepared.forcefield;
        let forcefield_texture = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("Forcefield"),
                size: wgpu::Extent3d {
                    width: forcefield_size,
                    height: forcefield_size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: forcefield_levels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &forcefield_mips,
        );
        let forcefield_view = forcefield_texture.create_view(&Default::default());
        let forcefield_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Forcefield"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let border_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("World border"),
            size: std::mem::size_of::<BorderUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let border_vertices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("World border vertices"),
            size: 16 * std::mem::size_of::<crate::border_render::BorderVertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let border_bindings = kept.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("World border bindings"),
            entries: &[
                entry(
                    0,
                    vf,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                entry(
                    1,
                    vf,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                entry(
                    2,
                    vf,
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                ),
                entry(
                    3,
                    vf,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        });
        let border_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("World border resources"),
            layout: &border_bindings,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: globals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&forcefield_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&forcefield_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: border_uniform.as_entire_binding(),
                },
            ],
        });
        let border_layout = kept.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("World border layout"),
            bind_group_layouts: &[Some(&border_bindings)],
            ..Default::default()
        });
        let border_shader = kept.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sunrise world border"),
            source: wgpu::ShaderSource::Wgsl(border_shader().into()),
        });
        let border_attributes = wgpu::vertex_attr_array![0=>Float32x3,1=>Float32x2];
        // RenderPipelines.WORLD_BORDER: BlendFunction.OVERLAY, no culling,
        // depth writes with a polygon offset of 3 towards the camera.
        let border_pipeline = kept.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("World border"),
            layout: Some(&border_layout),
            vertex: wgpu::VertexState {
                module: &border_shader,
                entry_point: Some("vs_border"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<crate::border_render::BorderVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &border_attributes,
                })],
            },
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                stencil: Default::default(),
                bias: wgpu::DepthBiasState {
                    constant: 3,
                    slope_scale: 3.0,
                    clamp: 0.0,
                },
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &border_shader,
                entry_point: Some("fs_border"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &linear_blend::frame_constants(frame_format, linear_light),
                    ..Default::default()
                },
                targets: &[Some(wgpu::ColorTargetState {
                    format: frame_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::Zero,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let screen_shader = kept.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sunrise screen effects"),
            source: wgpu::ShaderSource::Wgsl(include_str!("screen.wgsl").into()),
        });
        let screen_attributes =
            wgpu::vertex_attr_array![0=>Float32x4,1=>Float32x2,2=>Float32x4,3=>Float32x4];
        let screen_vertex = [Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<crate::screen_effects::ScreenVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &screen_attributes,
        })];
        let vignette_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        // One per blend and `.png.mcmeta` sampling (screen_effects::SAMPLE_*):
        // the shader is specialised, so a pixel pays for none of the others.
        let screen_pipelines: [wgpu::RenderPipeline; 16] = std::array::from_fn(|i| {
            let blur = i as u32 & crate::screen_effects::SAMPLE_BLUR != 0;
            let (entry, blend) = match (i & 4 == 0, blur) {
                (true, false) => ("fs_screen", alpha_blend),
                (true, true) => ("fs_screen_blur", alpha_blend),
                (false, false) => ("fs_vignette", vignette_blend),
                (false, true) => ("fs_vignette_blur", vignette_blend),
            };
            let clamp = i as u32 & crate::screen_effects::SAMPLE_CLAMP != 0;
            let constants = [("CLAMP", f64::from(u8::from(clamp)))];
            let compilation_options = wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            };
            kept.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Screen effects"),
                layout: Some(&entity_layout),
                vertex: wgpu::VertexState {
                    module: &screen_shader,
                    entry_point: Some("vs_screen"),
                    compilation_options: compilation_options.clone(),
                    buffers: &screen_vertex,
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &screen_shader,
                    entry_point: Some(entry),
                    compilation_options,
                    targets: &[Some(wgpu::ColorTargetState {
                        format: overlay_format,
                        blend: Some(blend),
                        // 8 and up: colour only, for a tint of the world in HDR output.
                        write_mask: if i & 8 == 0 {
                            wgpu::ColorWrites::ALL
                        } else {
                            wgpu::ColorWrites::COLOR
                        },
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        });
        let screen_capacity = 64;
        let screen_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Screen effect vertices"),
            size: screen_capacity
                * std::mem::size_of::<crate::screen_effects::ScreenVertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let celestial_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Celestial vertices"),
            size: 2048 * 4 * std::mem::size_of::<Vertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sunrise_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sunrise fan vertices"),
            size: (sunrise_fan::VERTICES * 5 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let entity_capacity = 1 << 14;
        let entity_skin = entity_skin::EntitySkin::new(
            &device,
            animate::SpriteAnimator::compute(&adapter, &device),
        );
        let entity_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Entity vertices"),
            size: entity_capacity * std::mem::size_of::<Vertex>() as u64,
            usage: entity_skin::EntitySkin::usage(entity_skin.is_some()),
            mapped_at_creation: false,
        });
        let static_entity_index_quads = 1 << 14;
        let static_entity_indices = Self::quad_indices(&device, static_entity_index_quads);
        let indirect_execution = adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::INDIRECT_EXECUTION);
        let flat_wanted = flat::wanted(&adapter, &device);
        // Metal draws solid terrain in two passes (`split_solid`): meshes put all
        // their opaque quads first, so each pass draws its part in a run or two.
        crate::mesher::group_parts(adapter.get_info().backend == wgpu::Backend::Metal);
        // Readable as storage where flat draws run: a shader pack's terrain program reads them.
        let arena = arena::Arena::new(
            &device,
            std::mem::size_of::<TerrainVertex>() as u64,
            1 << 20,
            flat_wanted,
        );
        // Sections as one index list per frame where multi-draw is emulated.
        let mut flat = flat_wanted.then(|| {
            let buffers = [Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<TerrainVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &TERRAIN_ATTRIBUTES,
            })];
            let info = adapter.get_info();
            // Metal's per-section path drops cutout leaves (the canopy above trunks vanishes),
            // and cave culling can drop the listed count below MIN_SECTIONS on a normal view:
            // so Metal draws vanilla terrain flat at any count.
            let min_sections = if flat::is_broken_intel_vulkan(&info)
                || flat::is_adreno_dx12(&info)
                || info.backend == wgpu::Backend::Metal
                || std::env::var("SUNRISE_FLAT_DRAWS").as_deref() == Ok("1")
            {
                0
            } else if flat::for_vanilla(&info) {
                flat::MIN_SECTIONS
            } else {
                // Only a shader pack's terrain and shadow pass draw flat here.
                usize::MAX
            };
            flat::Flat::new(
                &device,
                &bindings,
                arena.page_vertices(),
                min_sections,
                indirect_execution,
                |layout, label, vertex, fragment, blend, cull, pulled| {
                    let buffers: &[Option<wgpu::VertexBufferLayout<'_>>] =
                        if pulled { &[] } else { &buffers };
                    pipeline_in(
                        layout, label, vertex, fragment, buffers, tri, cull, true, true, blend,
                    )
                },
                |layout, label, vertex, fragment, cull| {
                    solid_in(layout, label, vertex, fragment, &[], cull)
                },
            )
        });
        // With flat draws' page groups the per-section solid draws read their vertices from
        // the page too (`Flat::page_group`), in either layout of the solid quads
        // (`mesher::packed`): only the origins stay a vertex buffer, at slot 1.
        let (solid, solid_opaque, solid_cutout, solid_mirror, flipped) = match &flat {
            Some(f) => {
                let pulled_layout = kept.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Sunrise terrain layout (pulled)"),
                    bind_group_layouts: &[Some(&bindings), Some(&f.draw_layout)],
                    ..Default::default()
                });
                let origins = [None, terrain_buffers[1].clone()];
                let solid = |label: &str, vertex: &str, fragment: &str, cull| {
                    solid_in(&pulled_layout, label, vertex, fragment, &origins, cull)
                };
                let (back, front) = (wgpu::Face::Back, wgpu::Face::Front);
                (
                    solid("Terrain solid", "vs_section", "fs_solid", back),
                    solid(
                        "Terrain solid opaque",
                        "vs_section_opaque",
                        "fs_opaque",
                        back,
                    ),
                    solid(
                        "Terrain solid cutout",
                        "vs_section_cutout",
                        "fs_solid",
                        back,
                    ),
                    pipeline_in(
                        &pulled_layout,
                        "Terrain solid (mirrored)",
                        "vs_section",
                        "fs_solid_mirror",
                        &origins,
                        tri,
                        Some(front),
                        true,
                        true,
                        None,
                    ),
                    solid("Terrain solid (flipped)", "vs_section", "fs_solid", front),
                )
            }
            None => (solid, solid_opaque, solid_cutout, solid_mirror, flipped),
        };
        // The solid quads in the compact layout where every pass reads it; the first world
        // frames settle it for the picked pack (`follow_packed_layout`).
        let packed_quads = flat.is_some() && !pack_starts_wide && packed_wanted();
        packed::pack_quads(packed_quads);
        let visbuf = flat
            .as_ref()
            .filter(|_| visbuffer::wanted(&device))
            .map(|f| {
                log::info!(
                    "Sunrise: solid terrain through a visibility buffer (SUNRISE_VISBUFFER)"
                );
                visbuffer::Visbuffer::new(
                    &device,
                    &bindings,
                    f,
                    (frame_format, linear_light),
                    &sprite_constants,
                )
            });
        let multi_draw = device
            .features()
            .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE)
            && indirect_execution;
        let batchable = translucent_batchable(&adapter.get_info(), multi_draw);
        crate::graphics::offer_translucent_batching(batchable);
        let translucent_batching = batchable && translucent_batching;
        let gpu_culling_offer = gpu_culling::offer(
            &adapter.get_info(),
            gpu_culling::supported(
                &adapter.get_info(),
                device.features(),
                adapter.get_downlevel_capabilities().flags,
                &device.limits(),
                multi_draw,
                flat.as_ref().map(|f| f.min_sections),
            ),
            flat.is_some(),
        );
        // Not again after the device was lost in one of its frames, until Sunrise restarts.
        let gpu_culling_offer = if crate::graphics::gpu_culling_lost() {
            crate::graphics::GpuCullingOffer::DeviceLost
        } else {
            gpu_culling_offer
        };
        crate::graphics::offer_gpu_culling(gpu_culling_offer);
        // A new renderer runs no shader pack: the game loads the picked one on its
        // first world frame (`shader_pack`: the picked pack draws flat, a `shaders/`
        // pack or one of Sunrise's own with `[shadow]`), and the frames before it, the
        // level loading screen's among them, keep the layout the sections upload in
        // (`draw_frame`). It starts as that pack's (`pack_starts_wide`: a `shaders/` pack
        // taken to draw terrain), so the pack loading keeps it.
        let (sorted_indices, translucent_arena) = arena::Arena::translucent_arenas(
            &device,
            std::mem::size_of::<TerrainVertex>() as u64,
            flat.is_some(),
            translucent_split(translucent_batching, flat.is_some(), shader_pack, false),
        );
        if let (Some(flat), Some(arena)) = (&mut flat, &translucent_arena) {
            flat.set_translucent_page_vertices(arena.page_vertices());
        }
        // Without indirect draws (OpenGL before 4.0 without indirect execution, or
        // before 4.2 + GL_ARB_shader_draw_parameters without indirect first instance)
        // even creating an indirect buffer fails; one draw per section reads
        // `draws` on the CPU instead.
        let indirect = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Section draws"),
            size: 1024 * 20,
            usage: if multi_draw {
                wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST
            } else {
                wgpu::BufferUsages::COPY_DST
            },
            mapped_at_creation: false,
        });
        // SUNRISE_GPU_CULL=0 turns occlusion culling off, `all` culls every
        // frame however few its draws and little it hides, `shader` tests in
        // the shader as on Metal (A/B checks).
        let cull_env = std::env::var("SUNRISE_GPU_CULL");
        let adapter_info = adapter.get_info();
        let cull = cull::Cull::supported(&adapter, &device)
            .filter(|_| occlusion_culling_requested(&adapter_info, cull_env.as_deref().ok()))
            .map(|depth_test| match cull_env.as_deref() {
                Ok("shader") => cull::DepthTest::InShader,
                _ => depth_test,
            })
            .map(|depth_test| {
                let mut cull = cull::Cull::new(&device, frame_format, depth_test);
                cull.always = cull_env.as_deref() == Ok("all");
                if !cull.always && adapter_info.backend == wgpu::Backend::Metal {
                    cull.min_sections = cull::TILED_MIN_SECTIONS;
                }
                cull
            });
        let depth = Self::depth(&device, &config);
        let world_post = world_post::Post::new(&device, frame_format);
        let uploads = upload::Uploads::new(&device);
        let section_uploads = upload::Uploads::with_chunk(&device, SECTION_UPLOAD_CHUNK);
        let stars = stars::Stars::new(&device, (frame_format, linear_light));
        let custom_sky = custom_sky::CustomSky::new(&device, (frame_format, linear_light));
        let entity_outline = entity_outline::EntityOutline::new(
            &device,
            (&entity_shader, &entity_layout, &entity_vertex),
            overlay_format,
        );
        let panorama = panorama::Panorama::new(&device, frame_format, panorama_faces);
        drop(pipelines_step);
        reuse.finish(generation);
        if previous.is_some() {
            log::info!(
                "Sunrise renderer: rebuilt on the same device ({} shader modules, layouts and pipelines kept)",
                reuse.len()
            );
        }
        if previous.is_none() {
            startup_step(format_args!(
                "pipelines ready (multi-draw {}, occlusion culling {}, GPU culling {}, GPU timer {})",
                on_off(multi_draw),
                on_off(cull.is_some()),
                if gpu_culling_offer == crate::graphics::GpuCullingOffer::Offered {
                    "available"
                } else {
                    "unavailable"
                },
                on_off(gpu_timer.is_some()),
            ));
        }
        let mut world_primitives =
            world_primitives::WorldPrimitives::new(&device, (frame_format, linear_light));
        world_primitives.reserve(
            &device,
            &mut |buffer, bytes| queue.write_buffer(buffer, 0, bytes),
            world_primitives::DEFAULT_CAPACITY,
            world_primitives::DEFAULT_CAPACITY,
        );
        let views = world_views::Kept::new(flipped, translucent_flipped, entity_pipeline_with);
        Ok(Self {
            display: None,
            reading: display,
            display_fresh: false,
            instance,
            adapter,
            surface: surface.map(Arc::new),
            spare_surface: std::sync::Mutex::new(None),
            offscreen,
            in_flight: std::collections::VecDeque::new(),
            first_frame: if previous.is_none() {
                FirstFrame::Acquire
            } else {
                FirstFrame::Drawn
            },
            gpu_timer,
            gpu_times: Vec::new(),
            lightmap_pixels: [0; 16 * 16 * 4],
            last_wait: std::time::Duration::ZERO,
            last_encode: std::time::Duration::ZERO,
            window,
            ray_scene: surface_setup::ray_query::ray_scene::RayScene::new(&device),
            device,
            reuse,
            queue: submit::Queue::new(queue, submit::wanted()),
            deferred: None,
            device_lost,
            config,
            solid,
            solid_opaque,
            solid_cutout,
            translucent,
            translucent_mark,
            flat_on: flat.is_some(),
            packed_quads,
            packed_on: packed_wanted(),
            cull_on: cull.is_some(),
            sides_on: true,
            gpu_culling: None,
            gpu_culling_offer,
            gpu_frame: false,
            #[cfg(debug_assertions)]
            gpu_checks: 0,
            cull,
            flat,
            visbuf,
            sky,
            sky_base,
            sunrise_pipeline,
            sunrise_buffer,
            dark_disc,
            depth_reset,
            lines,
            lines_secondary,
            core_lines: None,
            core_text: None,
            core_sky: None,
            debug_lines_opaque,
            debug_lines,
            debug_lines_on_top,
            debug_crosshair,
            overlay_crosshair,
            overlay_hand,
            debug_line_buffer,
            debug_line_capacity: 4096,
            crosshair_group,
            crosshair_globals,
            solid_mirror,
            mirror_globals,
            mirror_group,
            mirror: None,
            world_views: views,
            world_view_side: max_side,
            entity_pipeline,
            static_entity_pipeline,
            hdr_mark_pipeline,
            hdr_entity_mark_pipeline,
            hdr_hand_unmark_pipeline,
            hdr_mark_lightning_pipeline,
            weather,
            crumbling_pipeline,
            shadow_blob_pipeline,
            translucent_item_pipeline,
            water_mask_pipeline,
            cloud_pipeline,
            hdr_cloud_mark_pipeline,
            cloud_mesh: Default::default(),
            cloud_buffer,
            cloud_capacity: 4,
            cloud_uniform,
            cloud_group,
            additive_pipeline,
            dragon_rays_pipeline,
            energy_swirl_pipeline,
            lightning_pipeline,
            lightning: lightning::Renderer::new(),
            see_through_pipeline,
            world_primitives,
            text_polygon_offset_pipeline,
            world_text_pipeline,
            fluid_pipeline,
            terrain_model_pipelines,
            celestial,
            celestial_buffer,
            end_sky: end_sky::EndSky::default(),
            stars,
            custom_sky,
            custom_sky_draws: Vec::new(),
            entity_outline,
            panorama,
            border_pipeline,
            border_group,
            border_uniform,
            border_vertices,
            screen_pipelines,
            screen_buffer,
            screen_capacity,
            entity_group,
            entity_bindings,
            prepared_models: None,
            prepared_draws: Vec::new(),
            prepared_cuts: Vec::new(),
            prepared_ranges: Vec::new(),
            prepared_uploaded: Vec::new(),
            prepared_instance_buffer,
            prepared_instance_capacity,
            static_entity_group,
            static_entity_transform,
            static_entity_camera: None,
            static_entity_catalog_revision: 0,
            static_entity_frame_stats: Default::default(),
            hand_group,
            hand_globals,
            entity_buffer,
            entity_capacity,
            static_entity_sections: HashMap::default(),
            static_entity_indices,
            static_entity_index_quads,
            entity_batch: EntityBatch::default(),
            entity_skin,
            group,
            fluid_group,
            fluid_sampler,
            terrain_model_group,
            terrain_model_sampler,
            globals,
            pbr_maps,
            lightmap,
            sprite_textures,
            sprite_unused,
            sprite_views,
            mip_levels,
            mipmaps: mip_levels - 1,
            terrain_layout: bindings,
            terrain_sprites: sprite_constants.map(|(_, v)| v),
            sprite_split: split,
            sprite_sampler,
            sprite_nearest,
            texture_filtering: 0,
            max_anisotropy: 1,
            lightmap_view,
            lightmap_sampler,
            animations,
            material_animations,
            active_animated_layers: vec![0; active_layer_words],
            uploads,
            section_uploads,
            section_copies: None,
            section_copy_bytes: 0,
            shrink_gate: ShrinkGate::default(),
            lightmap_uploaded: false,
            sprite_minimum_size,
            sprite_native: sprite_mips.iter().map(|s| s.native_size).collect(),
            mipmap_option: 4,
            indices,
            index_quads,
            instances,
            instance_capacity,
            outline,
            depth,
            sections: resident::Resident::default(),
            lods: FxHashMap::default(),
            lod_cells: lod::TileGrid::default(),
            lod_cull_free: Vec::new(),
            lod_cull_next: 0,
            lod_cutout: Default::default(),
            lod_retiring: Default::default(),
            lod_leaving: Vec::new(),
            lod_pending: None,
            detail_columns: Default::default(),
            detail_revision: 0,
            arena,
            translucent_arena,
            translucent_batching,
            sections_dropped: false,
            sorted_indices,
            graveyard: Default::default(),
            translucent_sorter: translucency::Sorter::default(),
            translucency_cursor: 0,
            indirect,
            indirect_capacity: 1024,
            multi_draw,
            visible: Vec::new(),
            visible_lods: Vec::new(),
            shadow_visible: Vec::new(),
            translucent_draws: Vec::new(),
            translucent_tests: Vec::new(),
            translucent_marks: Vec::new(),
            phase2_upload: Vec::new(),
            origins: Vec::new(),
            draws: Vec::new(),
            draw_pages: Vec::new(),
            phase2_draws: Vec::new(),
            phase2_pages: Vec::new(),
            solid_runs: page_runs::PageRuns::default(),
            phase2_runs: page_runs::PageRuns::default(),
            shadow_runs: page_runs::PageRuns::default(),
            external: None,
            block_entities: Default::default(),
            block_entity_section_revisions: Default::default(),
            block_entity_membership_revisions: Default::default(),
            section_uploaded: Default::default(),
            newest_upload: None,
            newest_lod_fade: None,
            block_entity_revision: next_block_entity_revision(),
            light_blocks: Default::default(),
            light_revision: 0,
            occlusion: occlusion::Occlusion::new(crate::chunk::MIN_Y, crate::chunk::HEIGHT / 16),
            layout: crate::chunk::ColumnLayout::VANILLA,
            world_post,
            frame_gen: frame_gen::FrameGen::new(),
            entity_atlas: atlas_view.clone(),
            hdr: None,
            hdr_lighting_setting: true,
            emission_stats,
            output_caps,
            output_rejected: None,
            auto_off_since: None,
            // Nothing is presented as HDR yet: `update_output` applies the settings.
            output: output::resolve_output(settings, output_caps, display, frame, false),
            frame_kept_logged: false,
            hdr_marker: None,
            hdr_driver,
            pack_post: None,
            pack_loading: None,
            native: None,
            pass_timing: false,
            pass_detail: false,
            pass_timing_gap: std::time::Duration::ZERO,
            keep_timed_frames: false,
            own_timing: None,
            pack_entity_rest: Vec::new(),
            cloud_bindings,
            pack_weather: None,
            pack_atlas: None,
            pack_material_atlases: None,
            shared_material_maps: [None, None],
            pack_terrain_flat: false,
            separate_ao: false,
            pack_pages: Vec::new(),
            outline_edges_scratch: Vec::new(),
            outline_lines_scratch: Vec::new(),
            debug_lines_scratch: Vec::new(),
            screen_vertices_scratch: Vec::new(),
            lod_leaving_scratch: crate::fast_hash::FxHashSet::default(),
            shadow_boxed_scratch: Vec::new(),
            shadow_seen_scratch: crate::fast_hash::FxHashSet::default(),
            shadow_translucent_scratch: Vec::new(),
            pack_lava_layers: Vec::new(),
            hud,
            menu_blur: None,
            post_effects: None,
            readback: readback::Readback::default(),
            info,
            suspended: !drawable_size(size.width, size.height, max_side),
            acquire_validation_failures: 0,
            surface_unconfigured: false,
            drawn_sections: 0,
            section_draws: 0,
            section_draw_calls: 0,
            frame_serial: 0,
            stats_frame: 0,
            uploaded_bytes: 0,
            started: Instant::now(),
            screenshot: None,
            screenshot_result: None,
            screenshot_readbacks: screenshot_readback::Readbacks::default(),
            screenshot_quality: 1,
            screenshot_halvings: 0,
            gamma: 0.5,
            improved_transparency: false,
            oit: None,
            fullbright: false,
            animated_textures: true,
            present_modes,
            capture_copies: std::env::var_os("SUNRISE_RENDER_CAPTURE")
                .is_some()
                .then(capture::Copies::default),
            capture: None,
            capture_result: None,
        })
    }
    /// The first-person hand's quads, which follow `first` quads of the entity buffer.
    /// Unmarks the clouds just drawn (`fs_hdr_cloud_mark`).
    fn mark_clouds(&self, pass: &mut wgpu::RenderPass<'_>, quads: u32) {
        if !light_marks(self.output.frame.format, self.hdr.is_some())
            || self.hdr.as_ref().is_none_or(|h| !h.lighting)
        {
            return;
        }
        pass.set_pipeline(&self.hdr_cloud_mark_pipeline);
        pass.set_bind_group(0, &self.entity_group, &[]);
        pass.set_bind_group(1, &self.cloud_group, &[]);
        pass.set_vertex_buffer(0, self.cloud_buffer.slice(..));
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..quads * 6, 0, 0..1);
        pass.set_bind_group(0, &self.group, &[]);
    }
    /// Unmarks translucent terrain, over the draws that just blended it (`fs_translucent_mark`).
    fn mark_translucent_terrain(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        flat: Option<&flat::Flat>,
        (indirect, base): (&wgpu::Buffer, u32),
        solid_draws: u32,
        translucent_draws: u32,
    ) {
        if !light_marks(self.output.frame.format, self.hdr.is_some())
            || self.hdr.as_ref().is_none_or(|h| !h.lighting)
        {
            return;
        }
        match flat {
            Some(flat) => {
                pass.set_pipeline(&flat.translucent_mark);
                flat.draw(
                    pass,
                    flat.translucent_calls(),
                    flat_vertices(&self.arena, self.translucent_arena.as_ref()),
                );
                pass.set_bind_group(0, &self.group, &[]);
            }
            None => {
                pass.set_pipeline(&self.translucent_mark);
                pass.set_vertex_buffer(1, self.instances.slice(..));
                self.draw_sections(pass, indirect, base, solid_draws, translucent_draws);
            }
        }
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
    }
    /// The shared atlas's LabPBR maps (`HudAtlas::material_maps`), uploaded for the shader packs
    /// loaded from now on; a map too large for this GPU is left out.
    pub fn set_shared_material_maps(&mut self, maps: &[Option<crate::paged_maps::PagedMap>; 2]) {
        let limit = self.device.limits().max_texture_dimension_2d;
        let sizes: Vec<String> = (maps.iter().zip(["normals", "specular"]))
            .filter_map(|(m, name)| {
                m.as_ref()
                    .map(|m| format!("{name} {}x{}", m.width, m.height))
            })
            .collect();
        if !sizes.is_empty() {
            let bytes: usize = maps.iter().flatten().map(|m| m.pixels.len()).sum();
            log::info!(
                "shared atlas LabPBR maps: {} ({:.1} MiB)",
                sizes.join(", "),
                bytes as f64 / f64::from(1 << 20)
            );
        }
        self.shared_material_maps = maps.each_ref().map(|map| {
            let map = map.as_ref().filter(|m| m.width.max(m.height) <= limit)?;
            Some(self.device.create_texture_with_data(
                &self.queue,
                &wgpu::TextureDescriptor {
                    label: Some("Shared atlas material map"),
                    size: wgpu::Extent3d {
                        width: map.width,
                        height: map.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &map.pixels,
            ))
        });
    }
    /// The pack's custom sky layer textures (`crate::custom_sky::load`).
    pub fn set_custom_sky(&mut self, pack: &dyn crate::custom_sky::SkyLayers) {
        self.custom_sky.set_pack(&self.device, &self.queue, pack);
    }
    /// The custom sky layers the next frame draws (`SkyLayers::draws`): the game refills it each
    /// frame, empty when the world's sky has none.
    pub fn custom_sky_draws(&mut self) -> &mut Vec<crate::custom_sky::Draw> {
        &mut self.custom_sky_draws
    }
    /// The resource packs' `core/rendertype_lines` (`HudAtlas::lines_shader`) for the block
    /// outline, or the built-in one.
    pub fn set_lines_shader(&mut self, program: Option<&crate::core_shader::CoreProgram>) {
        let format = self.output.frame.format;
        // A pack's GLSL writes vanilla's gamma: a frame of linear light keeps the built-in one.
        let program = program.filter(|_| !self.output.frame.linear_light);
        self.core_lines =
            program.and_then(|program| core_lines::CoreLines::new(&self.device, format, program));
    }
    /// The resource packs' `core/sky` and `core/stars` (`HudAtlas::sky_shader`,
    /// `stars_shader`), or the built-in sky and stars.
    pub fn set_sky_shaders(
        &mut self,
        disc: Option<&crate::core_shader::CoreProgram>,
        stars: Option<&crate::core_shader::CoreProgram>,
    ) {
        // As `set_lines_shader`: not on a frame of linear light.
        let keep = !self.output.frame.linear_light;
        let (disc, stars) = (disc.filter(|_| keep), stars.filter(|_| keep));
        self.core_sky = core_sky::CoreSky::new(&self.device, self.output.frame.format, disc, stars);
    }
    fn quad_indices(device: &wgpu::Device, quads: u32) -> wgpu::Buffer {
        surface_setup::ray_query::quad_indices(device, quads)
    }
    fn offscreen_target(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Offscreen frame"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: config.usage,
            view_formats: &config.view_formats,
        })
    }
    /// The share of real frames frame generation shows a made frame before (1 while it
    /// keeps up), `None` while it isn't running.
    /// The posed models' store size (`SkinFrame::base`), with entity skinning.
    pub fn skinned_models(&self) -> Option<u32> {
        self.entity_skin
            .as_ref()
            .map(entity_skin::EntitySkin::stored)
    }

    /// Whether ordinary entity models may be posed on the GPU (`Frame::skin`, `entity_skin::poses`).
    pub fn skins_entities(&self) -> bool {
        self.entity_skin.is_some()
            && entity_skin::poses(
                self.frame_gen.on,
                (self.pack_post.as_ref())
                    .map(|(_, p)| p.as_ref().is_some_and(|p| p.skins_entities())),
                self.native.is_some(),
                || native_render::needs().entity_spans,
            )
    }

    /// Whether the layers tagged `OIT_ENTITY` (the eyes render type's, horse
    /// markings, a slime's outer cube) may be posed on the GPU too
    /// (`mobs::skin::SkinModels::tagged`): only with improved transparency off,
    /// where they are ordinary entity quads. On, `EntityBatch` sorts them into
    /// its stages by the CPU vertices' `OIT_ENTITY` tag.
    pub fn skins_tagged(&self) -> bool {
        self.skins_entities() && !self.improved_transparency && self.pack_post.is_none()
    }

    pub fn frame_generation_share(&self) -> Option<f32> {
        self.frame_gen.made_share
    }
    /// Frame generation on or off (FSR 3.1's frame interpolation).
    pub fn set_frame_generation(&mut self, on: bool) {
        if !on {
            self.frame_gen.stop();
        }
        self.frame_gen.on = on;
    }
    /// `upscale`: output pixels per rendered pixel, for TAA to rebuild (1: native).
    /// `sharpness`: how much TAA's upscaled picture is sharpened, 0..=1.
    pub fn set_world_post(
        &mut self,
        scale: f32,
        antialias: bool,
        taa: bool,
        upscale: f32,
        sharpness: f32,
    ) {
        self.world_post.set_sharpness(&self.device, sharpness);
        let taa = taa && self.world_post.supports_taa();
        self.world_post.scale = scale.clamp(0.25, 2.0);
        self.world_post.antialias = antialias && !taa;
        self.world_post.taa = taa;
        self.world_post.upscale = if taa { upscale.clamp(1.0, 4.0) } else { 1.0 };
    }
    /// The Texture Filtering and Max Anisotropy options (`filtering`: 0 none, 1 RGSS, 2
    /// anisotropic; `bit`: `maxAnisotropyBit`). All take LevelRenderer's chunk layer sampler
    /// (linear, clamped, `bit` anisotropy when anisotropic) and terrain.wgsl's sampleNearest,
    /// or sampleRGSS for RGSS.
    pub fn set_texture_filtering(&mut self, filtering: i32, bit: i32) {
        let filtering = filtering.clamp(0, 2) as u32;
        let anisotropy = if filtering == 2 {
            sampler_anisotropy(bit)
        } else {
            1
        };
        if filtering == self.texture_filtering && anisotropy == self.max_anisotropy {
            return;
        }
        self.texture_filtering = filtering;
        self.max_anisotropy = anisotropy;
        self.sprite_sampler = self
            .device
            .create_sampler(&sprite_sampler_descriptor(anisotropy));
        self.rebuild_terrain_group();
    }
    fn rebuild_terrain_group(&mut self) {
        self.sprite_views = sprite_views(
            &self.sprite_textures,
            &self.sprite_unused,
            Some(self.mipmaps + 1),
        );
        self.group = terrain_group(
            &self.device,
            "Sunrise terrain resources",
            &self.terrain_layout,
            &self.globals,
            &self.sprite_views,
            &self.sprite_sampler,
            &self.lightmap_view,
            &self.lightmap_sampler,
            &self.emission_stats.view,
        );
        self.fluid_group = terrain_group(
            &self.device,
            "Still fluid resources",
            &self.terrain_layout,
            &self.globals,
            &self.sprite_views,
            &self.fluid_sampler,
            &self.lightmap_view,
            &self.lightmap_sampler,
            &self.emission_stats.view,
        );
        self.terrain_model_group = terrain_group(
            &self.device,
            "Native cutout model resources",
            &self.terrain_layout,
            &self.globals,
            &self.sprite_views,
            &self.terrain_model_sampler,
            &self.lightmap_view,
            &self.lightmap_sampler,
            &self.emission_stats.view,
        );
    }
    /// Block sprite layers per array on this device: a sprite's array is its layer over it, its
    /// layer in that array the rest (`block_sprites::Lookup`).
    pub fn sprite_split(&self) -> u32 {
        self.sprite_split
    }
    /// The Mipmap Levels option: block textures sample at most `levels`
    /// reduced copies (0: none).
    pub fn set_mipmap_levels(&mut self, levels: u32) {
        self.mipmap_option = levels;
        let levels = crate::sprites::sprite_mipmap_limit(
            levels,
            self.sprite_textures[0].width(),
            self.sprite_minimum_size,
            self.mip_levels,
        );
        if levels == self.mipmaps {
            return;
        }
        self.mipmaps = levels;
        self.rebuild_terrain_group();
    }
    fn update_lightmap(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        sky: &Sky,
        darkness: f32,
        gamma: f32,
        night_vision: f32,
        flicker: f32,
        boss_darken: f32,
    ) {
        // lightmap.fsh: per (block, sky) level.
        let brightness = |level: f32| level / (4.0 - 3.0 * level);
        let mut pixels = [0u8; 16 * 16 * 4];
        for s in 0..16 {
            for b in 0..16 {
                let block_level = b as f32 / 15.0;
                let sky_level = s as f32 / 15.0;
                let block_brightness = brightness(block_level) * (1.4 + flicker);
                let sky_brightness = brightness(sky_level) * sky.sky_light_factor;
                // NightVisionColor defaults to 0x999999.
                let mut color =
                    Vec3::from(sky.ambient_light_color).max(Vec3::splat(0.6 * night_vision));
                color += Vec3::from(sky.sky_light_color) * sky_brightness;
                let parabolic = (2.0 * block_level - 1.0).powi(2);
                let block_color = Vec3::from(sky.block_light_tint).lerp(Vec3::ONE, 0.9 * parabolic);
                color += block_color * block_brightness;
                // lightmap.fsh: a boss bar mixes the light toward 70/60/60%.
                color = color.lerp(color * Vec3::new(0.7, 0.6, 0.6), boss_darken);
                color -= Vec3::splat(darkness);
                color = color.clamp(Vec3::ZERO, Vec3::ONE);
                let max = color.max_element().max(1e-6);
                let inverted = 1.0 - max;
                let scaled = 1.0 - inverted.powi(4);
                let not_gamma = color * (scaled / max);
                color = color.lerp(not_gamma, gamma);
                let i = (s * 16 + b) * 4;
                pixels[i] = (color.x * 255.0) as u8;
                pixels[i + 1] = (color.y * 255.0) as u8;
                pixels[i + 2] = (color.z * 255.0) as u8;
                pixels[i + 3] = 255;
            }
        }
        if self.fullbright {
            pixels.fill(255);
        }
        // The lightmap changes with the time of day, flicker and effects,
        // not every frame: upload only what differs from the texture.
        if pixels == self.lightmap_pixels && self.lightmap_uploaded {
            return;
        }
        self.uploads.texture(
            encoder,
            wgpu::TexelCopyTextureInfo {
                texture: &self.lightmap,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            16,
            16,
            &pixels,
        );
        self.lightmap_pixels = pixels;
        self.lightmap_uploaded = true;
    }
    fn animate(&mut self, encoder: &mut wgpu::CommandEncoder, tick: u64, cull_inactive: bool) {
        let active = cull_inactive.then_some(&self.active_animated_layers[..]);
        self.animations.animate_modes(
            encoder,
            &mut self.uploads,
            tick,
            active,
            self.own_timing.as_mut(),
        );
        if let Some(arrays) = &self.pbr_maps {
            for animation in &self.material_animations {
                if !self.animations.changed().contains(&animation.layer) {
                    continue;
                }
                let Some(pair) = arrays.get((animation.layer / self.sprite_split) as usize) else {
                    continue;
                };
                for kind in 0..2 {
                    let Some(pixels) = animation.material_pixels(kind, tick) else {
                        continue;
                    };
                    let texture = &pair[kind];
                    let from = (pixels.len() / 4).isqrt() as u32;
                    for level in 0..texture.mip_level_count() {
                        let size = texture.width() >> level;
                        let pixels = crate::sprites::resample(pixels, from, size);
                        self.uploads.texture(
                            encoder,
                            wgpu::TexelCopyTextureInfo {
                                texture,
                                mip_level: level,
                                origin: wgpu::Origin3d {
                                    x: 0,
                                    y: 0,
                                    z: animation.layer % self.sprite_split,
                                },
                                aspect: wgpu::TextureAspect::All,
                            },
                            size,
                            size,
                            &pixels,
                        );
                    }
                }
            }
            if let Some(atlases) = &self.pack_material_atlases {
                for kind in 0..2 {
                    let arrays: Vec<_> = arrays.iter().map(|pair| pair[kind].clone()).collect();
                    atlases[kind].copy(
                        &self.device,
                        encoder,
                        &arrays,
                        self.animations.changed().iter().copied(),
                    );
                }
            }
        }
        if let Some(atlas) = &self.pack_atlas {
            atlas.copy(
                &self.device,
                encoder,
                &self.sprite_textures,
                self.animations.changed().iter().copied(),
            );
        }
    }
    #[sunrise_mod_api_macros::replaceable]
    pub fn render(&mut self, frame: &Frame) -> Result<bool> {
        // A lost device cannot accept uploads either. Leave recovery to the
        // owner instead of submitting atlas patches to the invalid device.
        if self.device_lost() {
            return Err("the graphics device was lost".into());
        }
        if let Some(request) = self.capture.take() {
            let result = self.write_capture(&request, frame);
            self.capture_result = Some(result.map_err(|e| e.to_string()));
        }
        let scaled = self.screenshot_size();
        let rendered = match scaled {
            Some(size) => self.draw_scaled_frame(frame, size),
            None => self.draw_frame(frame, None, self.output),
        };
        self.finish_prepared_frame();
        // A frame that was not drawn still submits the section copies and
        // atlas patches (glyphs, skins, maps, mods' dynamic textures), which the caller
        // already took.
        if self.device_lost() {
            return Err("the graphics device was lost".into());
        }
        let patched = !frame.hud.patches.is_empty() || !frame.hud.image_patches.is_empty();
        if !matches!(rendered, Ok(true)) && patched {
            let section_copies = self.section_copies.take();
            let has_section_copies = section_copies.is_some();
            let mut encoder = section_copies.unwrap_or_else(|| {
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Sunrise atlas patches"),
                    })
            });
            self.hud
                .upload_patches(&mut self.uploads, &mut encoder, frame.hud);
            self.uploads.finish(&encoder);
            if has_section_copies {
                self.section_copy_bytes = 0;
                self.section_uploads.finish(&encoder);
            }
            self.queue.submit([encoder.finish()]);
        } else if let Some(copies) = self.take_section_copies() {
            self.queue.submit([copies]);
        }
        // A scaled screenshot's frame never reached the window.
        rendered.map(|drawn| drawn && scaled.is_none())
    }
    /// Whether the device's loss callback has fired. A surface reconfigure
    /// alone cannot restore resources belonging to a lost device.
    pub fn device_lost(&self) -> bool {
        self.device_lost.load(std::sync::atomic::Ordering::Relaxed)
    }
}
/// Encodes the copy of `texture`'s `size` pixels into a mappable buffer.
pub(crate) fn copy_frame(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    [width, height]: [u32; 2],
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sunrise screenshot"),
        size: u64::from(screenshot::padded_row(width)) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(screenshot::padded_row(width)),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    buffer
}
/// Centre-crop the frame to a square and average it down to `size`.
pub(crate) fn square_downscale(pixels: &[u8], width: u32, height: u32, size: [u32; 2]) -> Vec<u8> {
    let (side, cx, cy) = if width > height {
        (height, (width - height) / 2, 0)
    } else {
        (width, 0, (height - width) / 2)
    };
    let [tw, th] = size;
    let mut out = vec![0u8; (tw * th * 4) as usize];
    for ty in 0..th {
        let y0 = cy + ty * side / th;
        let y1 = cy + (ty + 1) * side / th;
        for tx in 0..tw {
            let x0 = cx + tx * side / tw;
            let x1 = cx + (tx + 1) * side / tw;
            let (mut r, mut g, mut b) = (0u32, 0u32, 0u32);
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = ((y * width + x) * 4) as usize;
                    r += u32::from(pixels[p]);
                    g += u32::from(pixels[p + 1]);
                    b += u32::from(pixels[p + 2]);
                }
            }
            let count = (x1 - x0) * (y1 - y0);
            let o = ((ty * tw + tx) * 4) as usize;
            out[o..o + 4].copy_from_slice(&[
                (r / count) as u8,
                (g / count) as u8,
                (b / count) as u8,
                255,
            ]);
        }
    }
    out
}
mod view_frustum;
pub use view_frustum::Frustum;

#[cfg(feature = "test-support")]
pub use view::test_frustum;
/// Vanilla's `Projection`: reversed depth (near maps to 1, far to 0, tests
/// are GREATER_THAN_OR_EQUAL), which keeps far faces a hair apart, like
/// fire_side's 0.01 pixel in front of its block, from z-fighting.
/// The section the camera is in.
fn camera_section(camera: DVec3, min_y: i32) -> SectionKey {
    (
        (camera.x.floor() as i32) >> 4,
        (camera.y.floor() as i32 - min_y) >> 4,
        (camera.z.floor() as i32) >> 4,
    )
}

/// The near plane of the world's projection, in blocks.
const NEAR_PLANE: f32 = 0.05;

pub(crate) fn world_projection(fov_degrees: f32, aspect: f32, far: f32) -> Mat4 {
    Mat4::perspective_rh(fov_degrees.to_radians(), aspect, far, NEAR_PLANE)
}

/// Calls `draw(first, quads)` for each run of a mesh's solid quads that can
/// face `eye` (`view::projection_center`), the mesh at the camera-relative
/// `origin`; all of them when there is no eye.
fn facing_runs(sides: &Sides, eye: Option<Vec3>, origin: [f32; 3], draw: impl FnMut(u32, u32)) {
    facing_runs_from(sides, eye, None, origin, Quads::All, draw)
}
/// `facing_runs` for the eye and, with `low`, every eye straight below it down to that
/// camera-relative height (the water mirror's eye).
fn facing_runs_from(
    sides: &Sides,
    eye: Option<Vec3>,
    low: Option<f32>,
    origin: [f32; 3],
    part: Quads,
    draw: impl FnMut(u32, u32),
) {
    match eye {
        Some(eye) => {
            let eye = eye - Vec3::from(origin);
            let low = low.map_or(eye.y, |y| (y - origin[1]).min(eye.y));
            sides.visible_from(eye.to_array(), low, part, draw)
        }
        None => sides.every(part, draw),
    }
}

#[cfg(test)]
mod frame_policy_tests;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod sprite_upload_tests;

#[cfg(test)]
mod shadow_tests;

/// A `shaders/` pack building on its own thread (`Renderer::load_shader_pack`): the dimension it
/// is for, the thread (a panic in it contained), and what the game named it with and when.
/// Dropping it (another pack, none, or another dimension) cancels the build: the thread stops at
/// the next program instead of compiling the rest for nothing.
struct PackLoading {
    dimension: i32,
    /// Held to be dropped with it: that cancels the build.
    _cancel: CancelOnDrop,
    #[allow(clippy::type_complexity)]
    thread: std::thread::JoinHandle<
        std::result::Result<std::result::Result<pack_post::PackPost, pack_post::Refused>, String>,
    >,
    label: String,
    started: std::time::Instant,
}

/// Sets its flag when dropped: `PackLoading`'s cancel, which `PackPost::new` checks per program.
struct CancelOnDrop(std::sync::Arc<std::sync::atomic::AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}
