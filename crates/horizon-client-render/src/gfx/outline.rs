//! Hit block selection outline and debug gizmo lines helpers.

/// Window.getAppropriateLineWidth, in framebuffer pixels.
pub fn outline_width(width: u32) -> f32 {
    2.5f32.max(width as f32 / 1920.0 * 2.5)
}

/// ScreenSize, stroke width and ProjectionType.PERSPECTIVE layering.
pub fn outline_params(width: u32, height: u32) -> [f32; 4] {
    [
        width as f32,
        height.max(1) as f32,
        outline_width(width),
        1.0 - 1.0 / 4096.0,
    ]
}

/// Camera-relative segment endpoints; no shape expansion in vanilla 26.3.
pub fn outline_vertices(edges: &[[f64; 6]], camera: glam::DVec3) -> Vec<[f32; 6]> {
    let mut out = Vec::with_capacity(edges.len());
    outline_vertices_into(edges, camera, &mut out);
    out
}

pub(super) fn outline_vertices_into(
    edges: &[[f64; 6]],
    camera: glam::DVec3,
    out: &mut Vec<[f32; 6]>,
) {
    let camera = camera.to_array();
    out.clear();
    out.extend(
        edges
            .iter()
            .map(|e| std::array::from_fn(|i| (e[i] - camera[i % 3]) as f32)),
    );
}

/// `DrawableGizmoPrimitives.getGroup`: a line whose ARGB alpha is 255 joins the
/// opaque group, drawn with `RenderTypes.lines` (depth written); the rest
/// with `RenderTypes.linesTranslucentNoDepthWrite`.
pub(super) fn debug_line_opaque(line: &[f32; 8]) -> bool {
    line[6].to_bits() >> 24 == 0xFF
}

/// The depth-tested gizmo lines in their upload order: the opaque group first,
/// then the translucent one, each in submission order.
pub(super) fn debug_lines_opaque_first(lines: &[[f32; 8]]) -> impl Iterator<Item = &[f32; 8]> {
    let opaque = lines.iter().filter(|l| debug_line_opaque(l));
    opaque.chain(lines.iter().filter(|l| !debug_line_opaque(l)))
}
