use super::*;
use crate::game::native_entity_refs::tests::with_game;
use horizon_client::{entity_nbt, save};
use horizon_core::nbt::Tag;
use horizon_core::position::BlockPos;

fn parse(text: &str) -> Tag {
    let (tag, used) = horizon_core::nbt::parse_snbt(text).unwrap();
    assert_eq!(used, text.len());
    tag
}

#[test]
#[ignore = "daily"]
fn loaded_mob_arrow_predicate_survives_model_and_controls_actual_block_hit() {
    with_game(|game| {
        game.build_registries();
        game.set_allow_commands(true);
        game.block_ticks
            .add_empty_chunk(BlockPos::new(0, 0, 0).chunk())
            .unwrap();
        // Supply the synthetic chunk before forceload's synchronous load barrier.
        game.world.chunks.insert(
            (0, 0),
            std::sync::Arc::new(horizon_client::chunk::Chunk::empty(0, 0)),
        );
        assert!(game.run_command("forceload add 0 0"));
        assert!(game.run_command("fill 0 199 0 15 199 15 stone"));
        assert!(game.world.loaded(3, 1));
        assert!(
            game.ui
                .rules
                .bool_at(horizon_client::gamerules::vanilla::PROJECTILES_CAN_BREAK_BLOCKS)
        );
        assert!(
            game.ui
                .rules
                .bool_at(horizon_client::gamerules::vanilla::BLOCK_DROPS)
        );
        assert!(game.run_command(
            "summon pig 14.5 200 14.5 {UUID:[I;197,0,0,1],NoAI:1b,NoGravity:1b,Health:10f}"
        ));
        let owner = game
            .mobs
            .iter()
            .find(|m| game.persist.uuid_of(m.id) == Some([197, 0, 0, 1]))
            .unwrap()
            .id;
        assert_eq!(
            game.mobs.iter().find(|m| m.id == owner).unwrap().health,
            10.0
        );
        let clock = game.world.game_time;
        let expected = parse("{blocks:\"minecraft:chorus_flower\"}");
        let stale = parse("{can_break:{blocks:\"minecraft:stone\"},marker:37}");
        let mut missing = Vec::new();
        for (n, predicate, griefing, breaks) in [
            (1, Some("chorus_flower"), true, true),
            (2, Some("stone"), true, false),
            (3, None, true, false),
            (4, Some("chorus_flower"), false, false),
            (5, Some("stone"), false, false),
            (6, None, false, false),
        ] {
            let x = if n <= 3 { 3 } else { 11 };
            let z = 1 + 4 * ((n - 1) % 3);
            if game
                .ui
                .rules
                .bool_at(horizon_client::gamerules::vanilla::MOB_GRIEFING)
                != griefing
            {
                assert!(game.run_command(&format!("gamerule mob_griefing {griefing}")));
            }
            assert_eq!(
                game.ui
                    .rules
                    .bool_at(horizon_client::gamerules::vanilla::MOB_GRIEFING),
                griefing
            );
            assert!(game.run_command(&format!("setblock {x} 199 {z} end_stone")));
            assert!(game.run_command(&format!("setblock {x} 200 {z} chorus_flower[age=0]")));
            let field = predicate.map_or(String::new(), |p| {
                format!(",can_break:{{blocks:\"minecraft:{p}\"}}")
            });
            assert!(game.run_command(&format!("summon arrow {} 200.5 {} {{UUID:[I;197,0,{n},2],NoGravity:1b,HasBeenShot:1b,Motion:[2d,0d,0d],Owner:[I;197,0,0,1],item:{{id:\"minecraft:arrow\",count:1}}{field}}}", f64::from(x)-1.5, f64::from(z)+0.5)));
            let p = game
                .creatures
                .projectiles
                .iter()
                .find(|p| game.persist.uuid_of(p.id) == Some([197, 0, n, 2]))
                .unwrap()
                .clone();
            assert_eq!(p.owner_uuid, Some([197, 0, 0, 1]));
            assert_eq!(
                p.owner,
                Some(Who::Mob(owner)),
                "actual resolved loaded Mob, not raw Owner alone"
            );
            assert!(matches!(p.kind, ProjectileKind::Arrow { .. }));
            assert!(p.no_gravity);
            assert_eq!(p.velocity, Vec3::new(2.0, 0.0, 0.0));
            assert!(!p.arrow.as_ref().unwrap().in_ground);
            game.persist.raw.remove(&p.id);
            for (route, raw) in [("no-raw", None), ("stale-raw", Some(&stale))] {
                let saved =
                    entity_nbt::arrow_to_nbt(&p, [197, 0, n, 2], p.owner_uuid, None, raw)
                        .unwrap();
                let want = predicate.map(|id| parse(&format!("{{blocks:\"minecraft:{id}\"}}")));
                if saved.get("can_break") != want.as_ref() {
                    missing.push(format!(
                        "{n} cloned loaded model {route}: {:?} != {want:?}",
                        saved.get("can_break")
                    ));
                }
                if raw.is_some() {
                    assert_eq!(save::number(saved.get("marker")), Some(37));
                }
            }
            if n == 1 {
                assert_eq!(game.persist.uuid_of(p.id), Some([197, 0, n, 2]));
                // The expected value is the actual admitted input, never an injected model field.
                assert_eq!(expected, parse("{blocks:\"minecraft:chorus_flower\"}"));
            }
            let drops_before = game.dropped.len();
            game.tick_projectiles().unwrap();
            let after = game
                .creatures
                .projectiles
                .iter()
                .find(|a| a.id == p.id)
                .unwrap();
            assert!(
                after.arrow.as_ref().unwrap().in_ground,
                "actual air-to-ground block collision"
            );
            let block = &*game.res.facts().get(game.world.block(x, 200, z)).name;
            let want = if breaks {
                "minecraft:air"
            } else {
                "minecraft:chorus_flower"
            };
            if block != want {
                missing.push(format!("{n} actual hit block {block} != {want}"));
            }
            let dropped: Vec<_> = game
                .dropped
                .iter()
                .skip(drops_before)
                .map(|e| (&*e.stack.item, e.stack.count, e.position, e.velocity))
                .collect();
            println!("actual collision {n}: block={block}, unfiltered new drops={dropped:?}");
            if !breaks {
                assert!(dropped.is_empty(), "denied hits cannot create drops");
            }
        }
        assert!(game.run_command("setblock 13 200 12 dirt"));
        let before = game.dropped.len();
        game.destroy_block_dropping([13, 200, 12]).unwrap();
        assert_eq!(
            &*game.res.facts().get(game.world.block(13, 200, 12)).name,
            "minecraft:air"
        );
        let direct: Vec<_> = game.dropped.iter().skip(before).collect();
        assert_eq!(
            direct.len(),
            1,
            "independent actual destruction/drop callback"
        );
        assert_eq!(&*direct[0].stack.item, "minecraft:dirt");
        assert_eq!(direct[0].stack.count, 1);
        assert_eq!(
            game.world.game_time, clock,
            "one projectile step per scene; no fabricated world ticks"
        );
        assert!(
            missing.is_empty(),
            "actual loaded Mob arrow model/permission mismatches: {missing:?}"
        );
    });
}
