use super::*;

pub struct Frame<'a> {
    pub camera: DVec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_degrees: f32,
    /// The camera entity's eye and view rotation (yaw, pitch): the camera's own except in a
    /// third-person view.
    pub eye: DVec3,
    pub eye_rotation: [f32; 2],
    /// The camera is the entity's eye: no third-person view.
    pub first_person: bool,
    /// The `GameRenderer.bobHurt`/`bobView` pose multiplied into the
    /// projection (`crate::camera::bob_matrix`).
    pub bob: Mat4,
    /// The spinning effect multiplied into the level's projection only
    /// (`crate::camera::spinning_effect_matrix`), not the hand's.
    pub spin: Mat4,
    pub sky: Sky,
    /// Block edits and chunk membership that can change the shader pack's shadow map.
    pub shadow_scene_revision: (u64, u64),
    /// Opt-in reuse of the shader pack's last shadow map when the key still matches
    /// (`horizon-graphics.txt` `shadowCache`; off by default).
    pub shadow_cache: bool,
    /// Render distance in blocks (fog end).
    pub distance: f32,
    /// Full-detail terrain distance in blocks; distant LODs bypass the cave graph.
    pub detail_distance: f32,
    /// `Camera.depthFar`: the far plane (`crate::camera::depth_far`).
    pub depth_far: f32,
    pub underwater: bool,
    /// The y of the horizontal water plane around the camera, in world blocks, while one
    /// of Horizon's own packs reads the water mirror (`Game::water_plane`,
    /// `Renderer::wants_water_plane`).
    pub water_plane: Option<f32>,
    /// Skip sections the camera cannot see through open faces (vanilla's
    /// occlusion graph; off in spectator-like views inside blocks).
    pub cave_culling: bool,
    /// Title screen: only the dedicated panorama cube map and the HUD.
    pub title: bool,
    /// Lava, powder snow, blindness, darkness or a boss bar's world fog leaves out the sky pass
    /// (`fog::sky_pass_skipped`).
    pub sky_blocked: bool,
    /// Dimensions without a skybox (the Nether) clear to the fog colour.
    pub skybox: bool,
    /// The End: clear to black and draw the resident sky box behind the world.
    pub end_sky: bool,
    /// End sky atlas bounds (u0, v0, u1, v1); absent when the sprite is missing.
    pub end_sky_sprite: Option<[f32; 4]>,
    /// A boss bar makes world fog (`BossHealthOverlay.shouldCreateWorldFog`: the Ender Dragon's).
    pub boss_world_fog: bool,
    /// Environmental fog start/end (Nether 10..96), if any.
    pub environment_fog: Option<(f32, f32)>,
    pub water_fog: [f32; 3],
    /// `FogRenderer.setupFog` result; replaces the underwater/environment fields when set.
    pub fog: Option<crate::fog::FogData>,
    /// Camera eye below the horizon height and not under water (`shouldRenderDarkDisc`).
    pub dark_disc: bool,
    /// The hit block outline as line segments (from, to) in world space.
    pub outline: Vec<[f64; 6]>,
    /// `BlockOutlineRenderState.isTranslucent`: draw after translucent terrain.
    pub outline_after_terrain: bool,
    /// `BlockOutlineRenderState.highContrast`: a thick black outline under a bright one.
    pub outline_high_contrast: bool,
    /// The PvP & HUD page's outline and hit colour (`pvp::Frame`; default: vanilla's).
    pub pvp: crate::pvp::Frame,
    /// `lines` with a colour and pixel width (`gizmos::camera_relative`):
    /// the fishing line and debug gizmos, depth tested, then gizmos drawn
    /// over everything.
    pub debug_lines: &'a [[f32; 8]],
    pub debug_lines_on_top: &'a [[f32; 8]],
    /// `DebugCrosshairRenderer`'s axes, camera relative like `debug_lines`:
    /// drawn after the hand at its projection, without the bob or fog.
    pub debug_crosshair: &'a [[f32; 8]],
    pub hud: &'a HudBatch,
    /// `GlintAlpha` (accessibility Glint Strength); the entity shader uses it
    /// for held/world items, independent of what the HUD batch last drew.
    pub glint_alpha: f32,
    /// The glint scroll clock (`Glint::scroll_ticks`): entity.wgsl's
    /// `ARMOR_ENTITY_GLINT_TEXTURING` phase.
    pub glint_ticks: f32,
    /// CardinalLighting.Type.NETHER, otherwise DEFAULT.
    pub nether_lighting: bool,
    /// Camera-relative entity geometry textured from the shared atlas.
    pub entities: &'a [Vertex],
    /// Per-section cacheable block-entity models. Positions are section-local.
    pub static_block_entities: &'a [StaticSectionSnapshot],
    /// Sections currently intersecting the coarse camera frustum.
    pub visible_static_block_entity_sections: &'a [[i32; 3]],
    /// Cache work for this frame, reported in bench summaries.
    pub static_block_entity_stats: StaticCacheFrameStats,
    /// Which entity `entities`' vertices belong to: (first vertex, entity id), each run up
    /// to the next; id 0 for none. Frame generation follows these entities' own motion.
    pub entity_motion: &'a [(u32, u64)],
    /// For each mark of `entity_motion`, the GPU-posed vertex count at it (`skin`'s
    /// `vertices`): where that entity's posed vertices start. A shader pack's `at_velocity`
    /// matches both parts of an entity with last frame's.
    pub entity_motion_posed: &'a [u32],
    /// The glowing entities' silhouettes in their team colours (`OutlineBufferSource`),
    /// outlined through walls after the hand.
    pub glowing: &'a [Vertex],
    /// Runs of `entities` by entity, while a shader pack draws them (`pack_draw::EntitySpan`).
    pub entity_spans: &'a [pack_draw::EntitySpan],
    /// Ordinary entity models posed on the GPU (`entity_skin`), drawn before `entities`'
    /// ordinary ones; only when `Renderer::skins_entities` said so.
    pub skin: Option<&'a entity_skin::SkinFrame>,
    pub weather: &'a [crate::weather::Vertex],
    pub weather_time: crate::weather::Time,
    /// Horizon Extras' dynamic lights, nearest first (`dynamic_light::select`); empty when off.
    pub dynamic_lights: &'a [[f32; 4]],
    pub colored_lights: &'a [crate::dynamic_light::Colored],
    pub emissive_layers: &'a [[u32; 4]],
    /// Cloud mesh inputs; geometry and GPU storage are retained by the renderer.
    pub clouds: Option<crate::clouds::Input<'a>>,
    /// Retained lightning bolts, drawn before the other additive entities.
    pub lightning: lightning::Input<'a>,
    /// Additive translucent geometry after lightning (dragon rays).
    pub additive: &'a [Vertex],
    /// Name tags over everything, without depth (`TEXT_SEE_THROUGH`).
    pub see_through: &'a [Vertex],
    /// Native mods' lines, boxes, beams and text (`native_world_primitives`), drawn at the end
    /// of the world pass, before the first-person hand.
    pub world_primitives: world_primitives::Batch<'a>,
    /// Opaque entity quads only a native pack's shadow pass draws (the local player in first
    /// person).
    pub shadow_only: &'a [Vertex],
    /// Camera-relative sun and moon, or the animated End flash (additive,
    /// behind everything and after the resident End sky box).
    pub celestial: &'a [Vertex],
    /// The stars, drawn from their static buffer after the sun and moon:
    /// the sky's rotation (camera-relative world from star space) and the
    /// star brightness.
    pub stars: Option<(Mat4, f32)>,
    /// First-person hand and held item, drawn after clearing depth.
    pub hand: &'a [Vertex],
    /// Where the arm and each held item start in `hand` (first vertex), with the
    /// `currentRenderedItemId` a shader pack's `gbuffers_hand` reads for them.
    pub hand_items: &'a [(u32, i32)],
    /// `CameraRenderState.hudFov`: the hand's own field of view
    /// (`GameRenderer.render3dHud`).
    pub hand_fov_degrees: f32,
    /// Full-screen camera effects drawn over the world and hand, under the HUD.
    pub screen: &'a [crate::screen_effects::ScreenQuad],
    /// What a shader pack's uniforms need from the game.
    pub pack: pack_uniforms::PackInputs,
    /// The world border wall, when near enough to be drawn.
    pub border: Option<BorderDraw>,
    pub game_time: f32,
    /// `GlobalSettingsUniform` GameTime: (game time % 24000 + partial tick) / 24000.
    pub shader_game_time: f32,
    /// Atlas rects (u0, v0, width, height) of end_sky.png and end_portal.png
    /// (`rendertype_end_portal`'s Sampler0 and Sampler1).
    pub end_portal_sprites: [[f32; 4]; 2],
    /// A destroy-stage sprite: projected UVs repeat inside a rect of this size, from the
    /// origin each crack vertex carries (`breaking::stage_light`).
    pub crumbling_sprite: [f32; 4],
    pub ticks: u64,
    /// The darkness effect on the lightmap: the brightness it takes off the
    /// gamma option and the pulsing scale subtracted from the light.
    pub darkness: (f32, f32),
    /// Night vision (or conduit water vision) intensity on the lightmap.
    pub night_vision: f32,
    /// Torch flicker added to the block light factor of 1.4.
    pub block_flicker: f32,
    /// `lightmap.fsh` BossOverlayWorldDarkeningFactor (a wither's bar).
    pub boss_darken: f32,
    /// `Screen.extractBlurredBackground`: blur the frame by this radius
    /// (`menuBackgroundBlurriness`) before the screen is drawn over it.
    pub menu_blur: Option<f32>,
    /// `Options.chunkSectionFadeInTime`, in seconds: how long a section takes to fade in once
    /// uploaded (0 draws it whole at once).
    pub chunk_fade: f32,
    /// `GameRenderer.requestedPostEffects`: the chains that run over the world.
    pub post_effects: crate::post_chain::Request<'a>,
}
/// One frame of the world border wall (`WorldBorderRenderer.render`).
#[derive(Clone, Debug)]
pub struct BorderDraw {
    /// The cached wall buffer; uploaded only when `rebuilt`.
    pub vertices: [crate::border_render::BorderVertex; 16],
    pub rebuilt: bool,
    pub transforms: crate::border_render::BorderTransforms,
    /// Walls to draw (indices of four-vertex quads), nearest first.
    pub sides: Vec<usize>,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct BorderUniform {
    pub(super) color: [f32; 4],
    pub(super) offset: [f32; 4],
}
/// A frame to save as a PNG: F2's screenshot, an agent's `game_screenshot`,
/// or the icon the Select World list shows.
pub struct ScreenshotRequest {
    pub path: PathBuf,
    /// `GameRenderer.takeAutoScreenshot`: centre-crop the frame to a square
    /// and average it down to this size. `None` saves the frame as drawn.
    pub size: Option<[u32; 2]>,
    /// F2's: drawn at the game's resolution (Resolution Scale above 100%) times
    /// `Renderer::screenshot_quality`, not the window's.
    pub scaled: bool,
    /// Read back on a worker (F2 and world icons); CPU pixels arrive in
    /// `Renderer::screenshot_result` for the owning thread's hook before encoding.
    pub background: bool,
}
