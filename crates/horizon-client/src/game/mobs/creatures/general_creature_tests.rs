use super::mob_effect_instance;
use super::thrown_item_axes;
use super::thrown_item_quad;
use super::{ProjectileKind, is_item_billboard};
use horizon_client::camera::CameraType;

#[test]
#[ignore = "daily"]
fn conversion_transfers_tags_and_team() {
    use super::super::commands::Target;
    use horizon_client::{
        mobs::{Mob, conversion::Params},
        save::uuid_text,
    };
    use horizon_core::geometry::Vec3;

    super::super::native_entity_refs::tests::with_game(|game| {
        let kind = game
            .res
            .mob_types()
            .get("minecraft:villager")
            .unwrap()
            .clone();
        let from = Mob::new(100, kind, Vec3::ZERO, 0.0);
        game.mobs.push(from.clone());
        game.next_entity = 100;
        game.set_entity_tags(Target::Mob(100), &["tracked".into()]);
        let old = uuid_text(game.uuid_for(100));
        game.load_scoreboard();
        game.score.board.add_team("test");
        game.score.board.add_to_team(&old, "test");
        let id = game
            .convert_to(
                &from,
                "minecraft:zombie_villager",
                Params {
                    keep_equipment: false,
                    preserve_can_pick_up_loot: true,
                },
                false,
                |_, _| {},
            )
            .unwrap();
        assert_eq!(game.tags_of(Target::Mob(id)), ["tracked"]);
        let new = uuid_text(game.uuid_for(id));
        assert_eq!(game.score.board.players_team(&new).unwrap().name, "test");
        assert!(game.score.board.players_team(&old).is_none());
    });
}

#[test]
#[ignore = "daily"]
fn wolf_owner_rejoining_receives_the_kill_statistic() {
    use horizon_client::advancements::{Advancement, Criterion, Rewards, Tree};
    use horizon_client::mobs::{Damage, Mob};
    use horizon_core::{geometry::Vec3, position::BlockPos};
    use std::sync::Arc;

    for dedicated in [false, true] {
        super::super::native_guest_turn_tests::with_guests(|game| {
            if dedicated {
                game.options.dedicated = Some(Vec::new());
            }
            let mut tree = Tree::default();
            tree.list.push(Advancement {
                id: "test:catalyst".into(),
                parent: None,
                display: None,
                criteria: vec![Criterion {
                    name: "kill".into(),
                    trigger: "minecraft:kill_mob_near_sculk_catalyst".into(),
                    conditions: serde_json::json!({}).into(),
                }],
                requirements: vec![vec!["kill".into()]],
                rewards: Rewards::default(),
            });
            tree.by_id.insert("test:catalyst".into(), 0);
            tree.by_trigger.insert(
                "minecraft:kill_mob_near_sculk_catalyst".into(),
                vec![(0, 0)],
            );
            game.adv.tree = Arc::new(tree);
            game.adv.loaded = true;
            game.world
                .chunks
                .insert((0, 0), Arc::new(horizon_client::chunk::Chunk::empty(0, 0)));
            let catalyst = game
                .pack
                .assets
                .registry
                .default_state("minecraft:sculk_catalyst")
                .unwrap();
            game.with_level(|b, level, ticks| {
                b.set_block(level, ticks, BlockPos::new(1, 0, 0), catalyst, 2 | 16);
            });
            game.sculk_prepare();
            game.sculk_sync_index();
            assert!(game.world.block_entities.contains_key(&(1, 0, 0)));
            let wolf_type = game.res.mob_types().get("minecraft:wolf").unwrap().clone();
            let pig_type = game.res.mob_types().get("minecraft:pig").unwrap().clone();
            let mut wolf = Mob::new(100, wolf_type, Vec3::ZERO, 0.0);
            wolf.extra.tame = true;
            wolf.extra.owner = Some(horizon_client::lan::playerdata::uuid_ints(12));
            game.mobs.push(wolf);
            game.mobs.push(Mob::new(101, pig_type, Vec3::ZERO, 0.0));
            {
                let value = false;
                game.lan
                    .as_mut()
                    .unwrap()
                    .hub
                    .guests
                    .update(0, |guest| guest.placed = value);
            }
            game.resolve_player_responsible(101, Damage::mob(100, [0.0; 3]));
            let credit = *game.last_hurt_by_player.get(&101).unwrap();
            assert!(game.resolve_player_credit(credit).is_none());
            {
                let value = true;
                game.lan
                    .as_mut()
                    .unwrap()
                    .hub
                    .guests
                    .update(0, |guest| guest.placed = value);
            }
            assert_eq!(game.resolve_player_credit(credit), Some(Some(12)));
            let victim = game.mobs[1].clone();
            game.mob_died(&victim);
            let stats = &game.lan.as_ref().unwrap().play.get(&12).unwrap().stats;
            assert_eq!(stats.get("custom", "minecraft:mob_kills"), 1);
            game.lan.as_mut().unwrap().hub.guests.update(0, |guest| {
                assert!(guest.advancements.has("test:catalyst", "kill"));
                guest.advancements = Default::default();
            });
            let mut second = victim.clone();
            second.id = 102;
            game.last_hurt_by_player
                .insert(102, super::super::combat::PlayerCredit(12));
            game.as_guest(12, |game| game.mob_died(&second)).unwrap();
            let host = game.lan.as_ref().unwrap();
            assert_eq!(host.play[&12].stats.get("custom", "minecraft:mob_kills"), 2);
            assert!(host.hub.guests[0].advancements.has("test:catalyst", "kill"));
            {
                let value = "minecraft:the_nether".into();
                game.lan
                    .as_mut()
                    .unwrap()
                    .hub
                    .guests
                    .update(0, |guest| guest.dimension = value);
            }
            let mut third = victim.clone();
            third.id = 103;
            game.resolve_player_responsible(101, Damage::mob(100, [0.0; 3]));
            let credit = game.last_hurt_by_player.remove(&101).unwrap();
            game.last_hurt_by_player.insert(103, credit);
            game.mob_died(&third);
            let host = game.lan.as_ref().unwrap();
            assert_eq!(host.play[&12].stats.get("custom", "minecraft:mob_kills"), 3);
            game.mobs[0].extra.owner = None;
            game.resolve_player_responsible(101, Damage::mob(100, [0.0; 3]));
            assert!(!game.last_hurt_by_player.contains_key(&101));
        });
    }
}

/// The component-wise lowest and highest corner of the quad.
fn bounds(corners: &[glam::Vec3; 4]) -> (glam::Vec3, glam::Vec3) {
    (
        corners.iter().copied().reduce(glam::Vec3::min).unwrap(),
        corners.iter().copied().reduce(glam::Vec3::max).unwrap(),
    )
}

#[test]
#[ignore = "daily"]
fn ender_pearl_clears_impulse_before_next_fall() {
    use super::{Game, Hit, Projectile, Vec3, Who};
    use crate::game::{Audio, Options, Progress, Settings, load_resources};
    use horizon_client::motion::{MotionWorld, PlayerMotion};

    let root = std::path::PathBuf::from(std::env::var_os("HORIZON_TEST_RESOURCES").unwrap());
    let settings = Settings::default();
    let loaded = load_resources(
        horizon_assets::resource_source::ResourceRoot::directory(root.clone()),
        Vec::new(),
        None,
        &root,
        &settings,
        std::time::Instant::now(),
        false,
        crate::game::execution_role::ExecutionRole::IntegratedServer,
        &Progress::new(),
        &mut |_, _| unreachable!(),
    )
    .unwrap();
    let mut options = Options::parse(std::iter::empty()).unwrap().unwrap();
    options.resources = root.clone();
    options.execution_role = crate::game::execution_role::ExecutionRole::IntegratedServer;
    let mut game = Game::new(
        &loaded.resources,
        loaded.pack,
        loaded.gui.unwrap(),
        options,
        settings,
        Audio::silent(&horizon_assets::resource_source::ResourceSource::directory(
            root.clone(),
        )),
        None,
    )
    .unwrap();
    let mut player = PlayerMotion::new(game.motion, Vec3::new(0.0, 64.0, 0.0)).unwrap();
    player
        .impulse
        .set_ignore_fall_damage(true, Some(player.position));
    player.fall_distance = 4.0;
    game.player = Some(player);
    let destination = Vec3::new(0.0, 80.0, 0.0);
    let mut pearl = Projectile::new(ProjectileKind::EnderPearl, destination, Some(Who::Player));
    pearl.previous = destination;
    game.projectile_hit(
        &pearl,
        Hit::Block {
            pos: [0, 80, 0],
            at: destination,
            face: [0; 3],
        },
    )
    .unwrap();
    let player = game.player.as_mut().unwrap();
    assert_eq!(player.position, destination);
    assert_eq!(player.fall_distance, 0.0);
    let damaged = MotionWorld {
        entities: &[],
        player,
        scene: &mut game.world,
        assets: &game.pack.assets,
        collision_binding: game.collision,
        resources: game.motion,
        health: &game.health_resources,
        tag_items: Some(game.res.items()),
        loop_context: &game.loop_context,
        damage_hooks: &horizon_client::health::DamageHooks::NONE,
        block_registries: Some(&game.ui.registries),
    }
    .cause_mount_fall(10.0, 1.0)
    .unwrap();
    assert!(
        damaged,
        "a ten-block fall after the teleport must cause damage"
    );
}

#[test]
fn nautilus_breath_lands_in_creative_as_ambient() {
    // AbstractNautilus.applyEffects: MobEffectInstance(BREATH_OF_THE_NAUTILUS,
    // 60, 0, ambient true, visible true, showIcon true), no game-mode gate.
    let e = mob_effect_instance("minecraft:breath_of_the_nautilus", 60, 0, true, true)
        .expect("applies to a creative player");
    assert_eq!(
        (e.id, e.duration, e.amplifier),
        ("breath_of_the_nautilus", 60, 0)
    );
    assert!(e.ambient && e.visible);
    // Survival gets the same ambient instance.
    let e =
        mob_effect_instance("minecraft:breath_of_the_nautilus", 60, 0, true, false).unwrap();
    assert!(e.ambient);
}

#[test]
fn other_mob_effects_stay_blocked_in_creative() {
    assert!(mob_effect_instance("minecraft:poison", 140, 0, false, true).is_none());
    let e = mob_effect_instance("minecraft:poison", 140, 0, false, false).unwrap();
    assert!(!e.ambient);
}

#[test]
fn thrown_item_billboard_lifts_by_the_generated_ground_transform() {
    // The kinds vanilla draws with `ThrownItemRenderer` as a billboard: the
    // only lifted ones. A fireball hands `getItem` to the item renderer
    // instead, which carries the ground transform itself.
    assert!(is_item_billboard(&ProjectileKind::Snowball));
    assert!(is_item_billboard(&ProjectileKind::Egg));
    assert!(is_item_billboard(&ProjectileKind::EnderPearl));
    assert!(!is_item_billboard(&ProjectileKind::SmallFireball));
    let large = ProjectileKind::LargeFireball { power: 1 };
    assert!(!is_item_billboard(&large));
    // A projectile at y = 100 with right = +X, up = +Y: the sprite sits
    // 0.125 blocks above the entity (2 of `item/generated`'s 16 units) and
    // stays 0.5 blocks wide, so the quad's y runs 99.875 .. 100.375.
    let at = glam::Vec3::Y * 100.0;
    let (right, up) = (glam::Vec3::X, glam::Vec3::Y);
    let lifted = is_item_billboard(&ProjectileKind::Snowball);
    let (lo, hi) = bounds(&thrown_item_quad(at, right, up, 1.0, lifted));
    assert_eq!((lo.y, hi.y), (99.875, 100.375));
    assert_eq!((lo.x, hi.x), (-0.25, 0.25));
    // Llama spit shares the quad here but is a model in vanilla
    // (`LlamaSpitRenderer`): no lift at all.
    let lifted = is_item_billboard(&ProjectileKind::LlamaSpit);
    assert!(!lifted);
    let (lo, hi) = bounds(&thrown_item_quad(at, right, up, 1.0, lifted));
    assert_eq!((lo.y, hi.y), (99.75, 100.25));
    // The rise follows the billboard's own up axis, not world up.
    let (right, up) = (glam::Vec3::Z, glam::Vec3::X);
    let lifted = is_item_billboard(&ProjectileKind::Egg);
    let (lo, hi) = bounds(&thrown_item_quad(at, right, up, 1.0, lifted));
    assert_eq!((lo.x, hi.x), (-0.125, 0.375));
    assert_eq!((lo.y, hi.y), (100.0, 100.0));
}

#[test]
fn thrown_item_billboard_does_not_flip_at_straight_up_or_down() {
    for yaw in [0.0f32, 37.5, 90.0, 180.0, -135.0] {
        for (edge, near) in [(-90.0f32, -89.9f32), (90.0, 89.9)] {
            let (right_edge, up_edge) = thrown_item_axes(CameraType::FirstPerson, yaw, edge);
            let (right_near, up_near) = thrown_item_axes(CameraType::FirstPerson, yaw, near);
            assert!(
                right_edge.dot(right_near) > 0.99 && up_edge.dot(up_near) > 0.99,
                "yaw {yaw}, pitch {edge}: {right_edge}/{up_edge} vs {right_near}/{up_near} at {near}"
            );
        }
    }
}

#[test]
fn thrown_item_billboard_faces_the_front_camera() {
    // Camera.setup turns the front camera round (yaw + 180, -pitch): its
    // right is the player's left.
    let (right, up) = thrown_item_axes(CameraType::FirstPerson, 0.0, 0.0);
    assert!(right.abs_diff_eq(glam::Vec3::NEG_X, 1e-5), "{right}");
    assert!(up.abs_diff_eq(glam::Vec3::Y, 1e-5), "{up}");
    let (right, up) = thrown_item_axes(CameraType::ThirdPersonFront, 0.0, 0.0);
    assert!(right.abs_diff_eq(glam::Vec3::X, 1e-5), "{right}");
    assert!(up.abs_diff_eq(glam::Vec3::Y, 1e-5), "{up}");
    let (right, _) = thrown_item_axes(CameraType::ThirdPersonBack, 0.0, 0.0);
    assert!(right.abs_diff_eq(glam::Vec3::NEG_X, 1e-5), "{right}");
}
