#[cfg(test)]
mod teleport_player_tests {
    use crate::game::native_entity_refs::tests::with_game;
    use horizon_client::motion::PlayerMotion;
    use horizon_core::geometry::Vec3;

    #[test]
    #[ignore = "daily"]
    fn command_moves_velocity_per_axis_and_keeps_fall_distance() {
        with_game(|game| {
            game.player = Some(PlayerMotion::new(game.motion, Vec3::ZERO).unwrap());
            crate::game::native_entity_refs::tests::bind_local_player(game, 1, 99);
            let player = game.player.as_mut().unwrap();
            player.position = Vec3::new(0.0, 64.0, 0.0);
            player.velocity = Vec3::new(1.0, -2.0, 3.0);
            player.fall_distance = 7.0;
            game.ui.registries.permission = 4;
            // An absolute teleport zeroes every axis and stands the player
            // on the ground, keeping the fall distance (performTeleport).
            assert!(game.run_command("tp 10 70 10"));
            let player = game.player.as_ref().unwrap();
            assert_eq!(player.position, Vec3::new(10.5, 70.0, 10.5));
            assert_eq!(player.velocity, Vec3::ZERO);
            assert!(player.ground.on_ground);
            assert_eq!(player.fall_distance, 7.0);
            assert!(player.teleported);
            assert_eq!(player.teleported_delta, [false; 3]);
            // A fall-flying player keeps the velocity of a relative axis
            // (calculateAbsolute's DELTA flags) and is not stood on the
            // ground; an absolute axis still loses its velocity.
            let player = game.player.as_mut().unwrap();
            player.velocity = Vec3::new(1.0, -3.0, 2.0);
            player.fall_flying = true;
            player.ground.on_ground = false;
            assert!(game.run_command("tp ~ 70 ~"));
            let player = game.player.as_ref().unwrap();
            assert_eq!(player.position, Vec3::new(10.5, 70.0, 10.5));
            assert_eq!(player.velocity, Vec3::new(1.0, 0.0, 2.0));
            assert_eq!(player.teleported_delta, [true, false, true]);
            assert!(!player.ground.on_ground);
            let player = game.player.as_mut().unwrap();
            player.velocity = Vec3::new(1.0, -3.0, 2.0);
            assert!(game.run_command("tp 12 ~ 12"));
            let player = game.player.as_ref().unwrap();
            assert_eq!(player.position, Vec3::new(12.5, 70.0, 12.5));
            assert_eq!(player.velocity, Vec3::new(0.0, -3.0, 0.0));
            // Without fall flying a relative Y axis loses its velocity too
            // (performTeleport zeroes it and sets onGround).
            game.player.as_mut().unwrap().fall_flying = false;
            assert!(game.run_command("tp ~ ~ ~"));
            let player = game.player.as_ref().unwrap();
            assert_eq!(player.velocity.y, 0.0);
            assert!(player.ground.on_ground);
            assert_eq!(player.fall_distance, 7.0);
        });
    }
}

#[cfg(test)]
mod fake_travel_tests {
    use super::super::super::Dim;
    use super::super::Target;

    #[test]
    #[ignore = "daily"]
    fn a_fake_player_is_not_teleported_to_another_dimension() {
        super::super::super::native_entity_refs::tests::with_game(|game| {
            super::super::super::bossbars::audience_tests::add_real_and_fake(game);
            let host = game.lan.as_ref().unwrap();
            let conn = |uuid| {
                host.hub
                    .guests
                    .iter()
                    .find(|g| g.uuid == uuid)
                    .unwrap()
                    .conn
            };
            let (fake, real) = (conn(2), conn(1));
            let to_nether = |game: &mut super::super::super::Game<'_>, conn| {
                game.as_guest(conn, |g| {
                    g.teleport_across(
                        Target::Player,
                        Dim::Nether,
                        [0.0, 64.0, 0.0],
                        None,
                        false,
                        super::super::super::vehicle_teleport::TeleportMotion::Stop,
                    )
                })
            };
            // Taken (true), so no same-level teleport stands in for it, and nothing queued.
            assert!(matches!(to_nether(game, fake), Some(Ok(true))));
            assert!(game.level_teleports.is_empty());
            // A real player's goes on the queue as before.
            assert!(matches!(to_nether(game, real), Some(Ok(true))));
            assert_eq!(game.level_teleports.len(), 1);
        });
    }

    /// A mod's `teleport_to_level`: a player to another level goes on the queue; to their own
    /// level, nothing is queued; an entity no level holds is refused.
    #[test]
    #[ignore = "daily"]
    fn a_mod_sends_a_player_to_another_level_and_not_a_missing_entity() {
        super::super::super::native_entity_refs::tests::with_game(|game| {
            super::super::super::bossbars::audience_tests::add_real_and_fake(game);
            let conn = game
                .lan
                .as_ref()
                .unwrap()
                .hub
                .guests
                .iter()
                .find(|g| g.uuid == 1)
                .unwrap()
                .conn;
            let sent = |game: &mut super::super::super::Game<'_>, level| {
                game.as_guest(conn, |g| {
                    g.teleport_to_level(Target::Player, level, [0.0, 64.0, 0.0], None)
                })
            };
            assert_eq!(sent(game, Dim::Nether), Some(true));
            assert_eq!(game.level_teleports.len(), 1);
            // Their own level: the same-level teleport, nothing queued.
            let home = game.target_level(Target::Guest(conn));
            assert_eq!(sent(game, home), Some(true));
            assert_eq!(game.level_teleports.len(), 1);
            // No level holds this item.
            assert!(!game.teleport_to_level(Target::Item(9_999), Dim::Nether, [0.0; 3], None));
        });
    }
}

#[cfg(test)]
mod teleport_rider_tests {
    #[test]
    #[ignore = "daily"]
    fn a_vehicle_carries_its_riders_and_a_rider_leaves_its_vehicle() {
        use super::super::Target;
        use horizon_core::geometry::Vec3;
        crate::game::native_entity_refs::tests::with_game(|game| {
            let horse = game.res.mob_types().get("minecraft:horse").unwrap().clone();
            let villager = game
                .res
                .mob_types()
                .get("minecraft:villager")
                .unwrap()
                .clone();
            // A villager riding a horse, 30 degrees off it.
            game.mobs.push(horizon_client::mobs::Mob::new(
                42,
                horse,
                Vec3::new(3.0, 64.0, 5.0),
                0.0,
            ));
            let mut rider =
                horizon_client::mobs::Mob::new(43, villager, Vec3::new(3.0, 64.5, 5.0), 30.0);
            rider.extra.mount = Some(42);
            game.mobs.push(rider);
            // Entity.teleportPassengers: the rider keeps its offset from the
            // vehicle and turns by what the vehicle turned.
            game.teleport_to_in_level(
                Target::Mob(42),
                [10.0, 70.0, -4.0],
                Some([90.0, 0.0]),
                [false; 3],
                true,
            );
            let vehicle = game.mobs.iter().find(|m| m.id == 42).unwrap();
            assert_eq!(vehicle.position, Vec3::new(10.0, 70.0, -4.0));
            let rider = game.mobs.iter().find(|m| m.id == 43).unwrap();
            assert_eq!(rider.position, Vec3::new(10.0, 70.5, -4.0));
            assert_eq!(rider.yaw, 120.0, "the rider keeps its angle to the vehicle");
            assert_eq!(
                rider.extra.mount,
                Some(42),
                "a teleported vehicle keeps its riders"
            );
            // A relative yaw (`~90`) is the same turn: the transition carries the
            // 90-degree delta and the rider adds it to its own angle.
            game.teleport_to_in_level(
                Target::Mob(42),
                [10.0, 70.0, -4.0],
                Some([180.0, 0.0]),
                [false; 3],
                true,
            );
            let rider = game.mobs.iter().find(|m| m.id == 43).unwrap();
            assert_eq!(rider.yaw, 210.0, "the rider turns by the vehicle's turn");
            // `facing` keeps the transition's rotation: the vehicle looks, the rider does not turn.
            game.teleport_to_in_level(
                Target::Mob(42),
                [10.0, 70.0, -4.0],
                Some([0.0, 0.0]),
                [false; 3],
                false,
            );
            let rider = game.mobs.iter().find(|m| m.id == 43).unwrap();
            assert_eq!(
                rider.yaw, 210.0,
                "a vehicle turned by `facing` leaves its rider"
            );
            // Entity.teleport's stopRiding(): a target that is itself a passenger
            // leaves its vehicle, or the next sync seats it where it was.
            game.teleport_to_in_level(Target::Mob(43), [0.0, 64.0, 0.0], None, [false; 3], false);
            let rider = game.mobs.iter().find(|m| m.id == 43).unwrap();
            assert_eq!(rider.extra.mount, None, "the teleported rider dismounted");
            assert_eq!(rider.position, Vec3::new(0.0, 64.0, 0.0));
            // #9335 (2): a rider keeps its angle to the vehicle through an
            // absolute turn, and a relative turn adds the delta: a villager 30
            // degrees off a horse turning 0 -> 40 ends at 70.
            let horse2 = game.res.mob_types().get("minecraft:horse").unwrap().clone();
            let villager2 = game
                .res
                .mob_types()
                .get("minecraft:villager")
                .unwrap()
                .clone();
            game.mobs.push(horizon_client::mobs::Mob::new(
                50,
                horse2,
                Vec3::new(10.0, 64.0, 10.0),
                0.0,
            ));
            let mut rider2 =
                horizon_client::mobs::Mob::new(51, villager2, Vec3::new(10.0, 64.5, 10.0), 30.0);
            rider2.extra.mount = Some(50);
            game.mobs.push(rider2);
            // Absolute turn: vehicle 0 -> 40, rider keeps its 30-degree offset.
            game.teleport_to_in_level(
                Target::Mob(50),
                [10.0, 64.0, 10.0],
                Some([40.0, 0.0]),
                [false; 3],
                true,
            );
            let rider2 = game.mobs.iter().find(|m| m.id == 51).unwrap();
            assert_eq!(
                rider2.yaw, 70.0,
                "absolute turn keeps the rider's angle to the vehicle"
            );
            // Relative turn (~10 on the vehicle): the transition carries the
            // 10-degree delta and the rider adds it to its own angle (70 -> 80).
            game.teleport_to_in_level(
                Target::Mob(50),
                [10.0, 64.0, 10.0],
                Some([50.0, 0.0]),
                [false; 3],
                true,
            );
            let rider2 = game.mobs.iter().find(|m| m.id == 51).unwrap();
            assert_eq!(
                rider2.yaw, 80.0,
                "relative turn shifts the rider by the delta"
            );
            // `facing` (turn_riders = false) leaves the rider's angle alone.
            game.teleport_to_in_level(
                Target::Mob(50),
                [10.0, 64.0, 10.0],
                Some([0.0, 0.0]),
                [false; 3],
                false,
            );
            let rider2 = game.mobs.iter().find(|m| m.id == 51).unwrap();
            assert_eq!(
                rider2.yaw, 80.0,
                "a vehicle turned by `facing` leaves its rider"
            );
        });
    }
}
