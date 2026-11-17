use super::*;

#[test]
#[ignore = "daily"]
fn projectile_handle_pierce_level_changes_real_hits() {
    super::super::native_entity_refs::tests::with_game(|game| {
        game.options.dedicated = Some(Vec::new());
        let zombie = game
            .res
            .mob_types()
            .get("minecraft:zombie")
            .unwrap()
            .clone();
        for (id, x) in [(11, 1.0), (12, 2.0), (13, 3.0)] {
            game.mobs
                .push(Mob::new(id, zombie.clone(), Vec3::new(x, 64.0, 0.0), 0.0));
        }
        let mut shot = Projectile::new(
            ProjectileKind::Arrow {
                base_damage: 2.0,
                critical: false,
                effect: None,
            },
            Vec3::new(0.0, 64.5, 0.0),
            None,
        );
        shot.velocity = Vec3::new(4.0, 0.0, 0.0);
        game.add_projectile(shot);
        let id = game.creatures.projectiles[0].id;
        let handle = game.projectile(game.entity(id).unwrap()).unwrap();
        assert_eq!(handle.pierce_level(game), Ok(0));
        assert_eq!(handle.set_pierce_level(game, 1), Ok(()));
        let health: Vec<_> = game.mobs.iter().map(|m| m.health).collect();
        game.tick_projectiles().unwrap();
        assert!(game.mobs[0].health < health[0]);
        game.tick_projectiles().unwrap();
        assert!(game.mobs[1].health < health[1]);
        assert_eq!(game.mobs[2].health, health[2]);
    });
}

#[test]
#[ignore = "daily"]
fn breeze_deflection_does_not_draw_from_level_random() {
    super::super::native_entity_refs::tests::with_game(|game| {
        let breeze = game
            .res
            .mob_types()
            .get("minecraft:breeze")
            .unwrap()
            .clone();
        game.mobs
            .push(Mob::new(1, breeze, Vec3::new(0.0, 64.0, 0.0), 0.0));
        let mut projectile =
            Projectile::new(ProjectileKind::Snowball, Vec3::new(-2.0, 64.5, 0.0), None);
        projectile.velocity = Vec3::new(3.0, 0.0, 0.0);
        game.creatures.projectiles.push(projectile);
        let before = game.world.random.state();
        game.tick_projectiles().unwrap();
        assert!(game.creatures.projectiles[0].velocity.x < 0.0);
        assert_eq!(game.world.random.state(), before);
    });
}
