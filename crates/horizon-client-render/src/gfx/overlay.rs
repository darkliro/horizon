//! Overlay counts and offsets for Renderer::draw_overlay.

use glam::Mat4;

/// Counts and offsets `Renderer::draw_overlay` draws from this frame's buffers.
pub(super) struct Overlay {
    /// Zero when a shader pack draws the hand.
    pub(super) hand_quads: u32,
    pub(super) first_hand_vertex: i32,
    pub(super) crosshair: u32,
    pub(super) first_crosshair: u32,
    /// The world drew into `depth` (post effects may read it), not a scaled target's.
    pub(super) world_depth: bool,
    /// The real frame, not frame generation's made one: the world icon grabs it.
    pub(super) real: bool,
    /// The game drew the first-person hand, in its own pass or at the end of the world's
    /// (`hand_in_world`), not a shader pack.
    pub(super) hand_drawn: bool,
    /// Camera-relative world to clip space for the hand: its own field of view and the bob.
    pub(super) hand_view_proj: Mat4,
}
