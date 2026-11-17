use super::*;
use crate::fast_hash::FxHashMap;
use crate::mesher::TerrainVertex;

use glam::{DVec3, Mat4, Vec3, Vec4};

mod shadow_face_tests {
    use super::*;

    /// The faces of one block at `at` (section-relative), each with its own layer:
    /// 0 +X, 1 -X, 2 +Y (up), 3 -Y (down), 4 +Z, 5 -Z; counter-clockwise from outside.
    fn block(at: [f32; 3]) -> Vec<TerrainVertex> {
        let [x, y, z] = at;
        let (a, b, c) = (x + 1.0, y + 1.0, z + 1.0);
        [
            [[a, y, z], [a, b, z], [a, b, c], [a, y, c]],
            [[x, y, z], [x, y, c], [x, b, c], [x, b, z]],
            [[x, b, z], [x, b, c], [a, b, c], [a, b, z]],
            [[x, y, z], [a, y, z], [a, y, c], [x, y, c]],
            [[x, y, c], [a, y, c], [a, b, c], [x, b, c]],
            [[x, y, z], [x, b, z], [a, b, z], [a, y, z]],
        ]
        .into_iter()
        .enumerate()
        .flat_map(|(layer, q)| q.map(|p| TerrainVertex::new(p, [0.0; 2], layer as u16, 0, 0)))
        .collect()
    }

    /// The layers (faces) of the quads the shadow pass draws of `solid`.
    fn drawn(view: &external::ShadowView, origin: [f32; 3]) -> Vec<u16> {
        let mut solid = block([7.0, 7.0, 7.0]);
        let sides = crate::mesher::sort_tinted(&mut solid, &mut Vec::new());
        let mut layers = Vec::new();
        shadow_runs(&sides, view, origin, |first, quads| {
            for q in first..first + quads {
                layers.push(solid[q as usize * 4].layer);
            }
        });
        layers.sort_unstable();
        layers
    }

    fn view(every_face: bool) -> external::ShadowView {
        external::ShadowView {
            cull: Mat4::IDENTITY,
            light: Vec3::Y,
            distance: None,
            unculled: None,
            every_face,
            view_casters: true,
            depth_planes: false,
        }
    }

    /// Iris 1.11.7's shadow map holds faces turned away from the light (#9033): the block's
    /// bottom face, under a sun straight above, is drawn too.
    #[test]
    fn the_shadow_pass_draws_faces_turned_away_from_the_light() {
        // A section 32 blocks under the camera.
        let origin = [-8.0, -40.0, -8.0];
        assert_eq!(drawn(&view(true), origin), [0, 1, 2, 3, 4, 5]);
        // Horizon's own packs keep drawing only what can face the light: no bottom face.
        assert!(!drawn(&view(false), origin).contains(&3));
    }
}

#[cfg(test)]
mod shadow_caster_tests {
    use super::*;

    #[test]
    fn the_shadow_pass_takes_the_block_entities_of_its_sections_within_64_blocks() {
        // Sections 64 to 79 high (the world from -64): a chest in each of three. The shadow
        // pass drew `near` and `far`; `left_out` lies as near but the pass didn't draw it, and
        // `far`'s nearest block is 78.5 blocks away.
        let (near, left_out, far) = ((0, 8, 0), (1, 8, 0), (5, 8, 0));
        let mut block_entities = FxHashMap::default();
        block_entities.insert(near, vec![([3, 70, 3], 7)]);
        block_entities.insert(left_out, vec![([20, 70, 3], 7)]);
        block_entities.insert(far, vec![([85, 70, 3], 7)]);
        let camera = DVec3::new(0.5, 70.0, 0.5);
        let drawn = [near, far].into_iter();
        let taken: Vec<_> = block_entities_within_reach(drawn, &block_entities, camera, -64)
            .copied()
            .collect();
        assert_eq!(taken, vec![([3, 70, 3], 7)]);
    }

    /// What the shadow pass keeps of a camera-relative section at `min`, the camera at the
    /// origin looking along -X (yaw 90) with the scenes' 70 degree, 16:9 view, and the light in
    /// direction `light`; the light's own box takes in everything.
    fn kept(light: Vec3, min: [f32; 3]) -> bool {
        kept_looking(Vec3::NEG_X, light, min)
    }

    /// `kept` with the camera looking along `forward`.
    fn kept_looking(forward: Vec3, light: Vec3, min: [f32; 3]) -> bool {
        let camera = world_projection(70.0, 16.0 / 9.0, 192.0)
            * Mat4::look_to_rh(Vec3::ZERO, forward, Vec3::Y);
        let cull = ShadowCull {
            light: Frustum {
                planes: [Vec4::W; 6],
            },
            casters: Some(shadow_casters(camera, light)),
            reach: None,
        };
        let min = Vec3::from(min);
        cull.intersects(min, min + 16.0)
    }

    /// Our probes' Iris 1.11.7 shadow maps fit keeping only the terrain whose shadow can fall
    /// into the camera's view (#8464).
    #[test]
    fn the_shadow_pass_keeps_what_can_shade_the_view() {
        // A noon sun: a treetop high above the ground ahead shades it, though the camera doesn't
        // see it; behind the camera, or far off to the side, nothing does.
        assert!(kept(Vec3::Y, [-48.0, 64.0, -8.0]));
        assert!(!kept(Vec3::Y, [32.0, 0.0, -8.0]));
        assert!(!kept(Vec3::Y, [-32.0, 0.0, 96.0]));
        // Nor does what lies under the ground the camera looks along.
        assert!(!kept(Vec3::Y, [-48.0, -96.0, -8.0]));
        // A low sun ahead (west, -X) throws shadows toward the camera: what is behind it can't
        // reach the view.
        let west = Vec3::new(-1.0, 0.1, 0.0).normalize();
        assert!(kept(west, [-48.0, 0.0, -8.0]));
        assert!(!kept(west, [32.0, 0.0, -8.0]));
        // The same sun behind the camera throws the terrain behind it forward, into the view.
        let east = Vec3::new(1.0, 0.1, 0.0).normalize();
        assert!(kept(east, [32.0, 0.0, -8.0]));
        assert!(kept(east, [96.0, 0.0, -8.0]));
        // Looking 20 degrees down under a noon sun, the view's sides lean out below the camera:
        // a treetop above the ground ahead still shades it, one off to the side doesn't, though
        // it lies inside each of the view's planes that faces the light.
        let down = Vec3::new(-20f32.to_radians().cos(), -20f32.to_radians().sin(), 0.0);
        assert!(kept_looking(down, Vec3::Y, [-48.0, 16.0, -8.0]));
        assert!(!kept_looking(down, Vec3::Y, [-32.0, 16.0, -96.0]));
    }

    /// Our probes' Iris 1.11.7 maps keep the sections whose box, grown by a margin of 1.1224 to
    /// 1.1252 blocks, meets the shape (#8464): a camera at (13.5, 69.62, -13.5) under a noon sun with the scenes' flying
    /// field of view (77°).
    #[test]
    fn the_shadow_pass_keeps_a_margin_around_the_view() {
        let kept = |yaw: f32, column: [f32; 2]| {
            let camera =
                world_projection(77.0, 16.0 / 9.0, 1024.0) * crate::camera::view_matrix(yaw, 0.0);
            let cull = ShadowCull {
                light: Frustum {
                    planes: [Vec4::W; 6],
                },
                casters: Some(shadow_casters(camera, Vec3::Y)),
                reach: None,
            };
            let min = Vec3::new(
                column[0] * 16.0 - 13.5,
                64.0 - 69.62,
                column[1] * 16.0 + 13.5,
            );
            cull.intersects(min, min + 16.0)
        };
        // Looking west, the column at chunk (-2, -6) needs a margin of 0.9 blocks: Iris keeps it.
        assert!(kept(90.0, [-2.0, -6.0]));
        // Looking south (yaw 7.5), the one at (3, 0) needs a margin of 1.3 blocks: Iris drops it.
        assert!(!kept(7.5, [3.0, 0.0]));
    }

    /// With `shadowDistanceRenderMul`, our probe's Iris 1.11.7 maps keep the sections within
    /// `shadowDistance × mul` of the camera along each axis, out to the cube's corners, and with
    /// some slack (the camera at (13.5, 69.62, -13.5)):
    /// - at 100 blocks, the column at chunk (7, 5), 98.5 and 93.5 blocks out (136 diagonally), and
    ///   the one at (-6, -7) are kept; the one at (8, -4), 114.5 out on x, is not;
    /// - at 49.5 blocks, the column at (-2, -5), 50.5 out on z, is kept; at 49 it is not;
    /// - at 50 blocks from a camera at y 151.62, the ground's sections 70 blocks below are not.
    #[test]
    fn the_shadow_distance_render_mul_keeps_a_cube() {
        let kept = |camera: DVec3, reach: f32, keys: &[(i32, i32)]| {
            let y = (64 - crate::chunk::MIN_Y) / 16;
            let mut sections = resident::Resident::<()>::default();
            for &(x, z) in keys {
                sections.insert_with_bounds((x, y, z), (), None);
            }
            let cull = ShadowCull {
                light: Frustum {
                    planes: [Vec4::W; 6],
                },
                casters: None,
                reach: Some(reach),
            };
            let mut out = Vec::new();
            sections.visible(camera, 12, &cull, |_| true, |_, _| false, &mut out);
            let mut kept: Vec<_> = out.iter().map(|v| (v.key.0, v.key.2)).collect();
            kept.sort_unstable();
            kept
        };
        let ground = DVec3::new(13.5, 69.62, -13.5);
        let columns = [(7, 5), (8, -4), (-6, -7)];
        assert_eq!(kept(ground, 100.0, &columns), [(-6, -7), (7, 5)]);
        assert_eq!(kept(ground, 49.5, &[(-2, -5)]), [(-2, -5)]);
        assert_eq!(kept(ground, 49.0, &[(-2, -5)]), []);
        assert_eq!(kept(DVec3::new(13.5, 151.62, -13.5), 50.0, &[(0, 0)]), []);
    }

    /// With the camera looking straight up under a noon sun, our probe's Iris 1.11.7 map holds
    /// 7 columns there, which testing whole sections reproduces (a guess): the sections beside
    /// the camera whose 16 blocks reach above it are kept, even where their terrain lies below
    /// the camera.
    #[test]
    fn the_shadow_pass_tests_whole_sections() {
        let camera = DVec3::new(8.0, 6.0, 8.0);
        let up = world_projection(70.0, 16.0 / 9.0, 192.0)
            * Mat4::look_to_rh(Vec3::ZERO, Vec3::Y, Vec3::Z);
        let view = Frustum {
            planes: frustum(up),
        };
        // The section east of the camera's, y 0..16, with ground only in its lowest 2 blocks.
        let mut sections = resident::Resident::<()>::default();
        let key = (1, -crate::chunk::MIN_Y / 16, 0);
        sections.insert_with_bounds(key, (), Some([Vec3::ZERO, Vec3::new(16.0, 2.0, 16.0)]));
        let drawn = |cull: &dyn Fn(&mut Vec<resident::Visible>)| {
            let mut out = Vec::new();
            cull(&mut out);
            out.iter().map(|v| v.key).collect::<Vec<_>>()
        };
        let all = Frustum {
            planes: [Vec4::W; 6],
        };
        let shadow = ShadowCull {
            light: all,
            casters: Some(shadow_casters(up, Vec3::Y)),
            reach: None,
        };
        assert_eq!(
            drawn(&|out| sections.visible(camera, 4, &shadow, |_| true, |_, _| false, out)),
            [key]
        );
        // The camera's own view goes by the terrain's box, which lies below it.
        assert!(
            drawn(&|out| sections.visible(camera, 4, &view, |_| true, |_, _| false, out))
                .is_empty()
        );
    }

    /// Our probe pack's Iris 1.11.7 shadow maps, under a noon sun with `shadowDistance` 128, hold
    /// the terrain at every depth along the light whatever planes the pack gives (#10403): from
    /// a cave at y -7, the surface's sections ~70 blocks toward the light with `shadowNearPlane`
    /// 0.05 and `shadowFarPlane` 256; from y 600, the ground ~535 blocks away with no planes set;
    /// and both with the planes at 200 and 210. Only the box's sides cull.
    #[test]
    fn the_shadow_pass_draws_terrain_outside_the_pack_s_planes() {
        let kept = |near: Option<f32>, far: Option<f32>, y: f64, min: [f32; 3], depth: bool| {
            let shadow = horizon_shaders::Shadow {
                distance: 128.0,
                near,
                far,
                ..horizon_shaders::Shadow::default()
            };
            let camera = DVec3::new(13.5, y + 1.62, -13.5);
            let view = external::ShadowView {
                cull: pack_uniforms::shadow_culling_at(Vec3::Y, camera, &shadow),
                light: Vec3::Y,
                distance: None,
                unculled: None,
                every_face: true,
                view_casters: false,
                depth_planes: depth,
            };
            let cull = ShadowCull {
                light: shadow_box(&view),
                casters: None,
                reach: None,
            };
            let min = Vec3::from(min);
            cull.intersects(min, min + 16.0)
        };
        // The section holding the surface (y 64..80) above the cave, and the ground under y 600.
        let above = [-8.0, 64.0 - (-7.0 + 1.62), -8.0];
        let below = [-8.0, 64.0 - (600.0 + 1.62), -8.0];
        assert!(kept(Some(0.05), Some(256.0), -7.0, above, false));
        assert!(kept(None, None, 600.0, below, false));
        assert!(kept(Some(200.0), Some(210.0), -7.0, above, false));
        assert!(kept(Some(200.0), Some(210.0), 600.0, below, false));
        // A section 200 blocks off to the side is still outside the box.
        let side = [200.0, 0.0, -8.0];
        assert!(!kept(Some(0.05), Some(256.0), -7.0, side, false));
        // Horizon's own packs keep culling by their planes.
        assert!(!kept(Some(0.05), Some(256.0), -7.0, above, true));
    }

    /// The margin doesn't follow the field of view: from the same camera under a noon sun, our
    /// probe's Iris 1.11.7 maps keep a section that needs a growth of 1.1224 blocks at a field of
    /// view of 70 (77° flying) and drop ones that need 1.1252 to 1.1422 at 50 (55° flying) and
    /// 70 (#8464).
    #[test]
    fn the_shadow_pass_keeps_the_same_margin_at_every_field_of_view() {
        let kept = |fov: f32, yaw: f32, pitch: f32, section: [f32; 3]| {
            let camera = world_projection(fov * 1.1, 16.0 / 9.0, 1024.0)
                * crate::camera::view_matrix(yaw, pitch);
            let cull = ShadowCull {
                light: Frustum {
                    planes: [Vec4::W; 6],
                },
                casters: Some(shadow_casters(camera, Vec3::Y)),
                reach: None,
            };
            let min = Vec3::new(
                section[0] * 16.0 - 13.5,
                section[1] * 16.0 - 69.62,
                section[2] * 16.0 + 13.5,
            );
            cull.intersects(min, min + 16.0)
        };
        // fov 70, yaw 159, pitch 20: (4, 4, -4) needs 1.1224 blocks; Iris keeps it.
        assert!(kept(70.0, 159.0, 20.0, [4.0, 4.0, -4.0]));
        // fov 70, yaw 247.5: (1, 4, 0) needs 1.1253; Iris drops it.
        assert!(!kept(70.0, 247.5, 0.0, [1.0, 4.0, 0.0]));
        // fov 50, yaw 339, pitch -20: (3, 4, -1) needs 1.1252; looking south, (-2, 3, 0) needs
        // 1.1422. Iris drops both.
        assert!(!kept(50.0, 339.0, -20.0, [3.0, 4.0, -1.0]));
        assert!(!kept(50.0, 0.0, 0.0, [-2.0, 3.0, 0.0]));
    }
}
