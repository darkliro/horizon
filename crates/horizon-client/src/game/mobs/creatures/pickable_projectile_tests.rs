use super::{pickable_projectile_hit, pickable_projectile_hitboxes};
use horizon_client::mobs::Who;
use horizon_client::projectiles::{Hit, Projectile, ProjectileKind, ProjectileWorld};
use horizon_core::geometry::{Aabb, Vec3};

/// Open air with nothing in it but `.0`.
struct Open(Vec<(Who, Aabb)>);
impl ProjectileWorld for Open {
    fn clip(&self, _: Vec3, _: Vec3) -> Option<(Vec3, [i32; 3], [i32; 3])> {
        None
    }
    fn in_water(&self, _: Aabb) -> bool {
        false
    }
    fn targets(&self) -> &[(Who, Aabb)] {
        &self.0
    }
}

/// Which of `list` another projectile (the one at 0) can strike.
fn struck(list: &[Projectile]) -> Vec<usize> {
    let alive = vec![true; list.len()];
    pickable_projectile_hitboxes(list, &alive, 0)
        .iter()
        .map(|(who, _)| match who {
            Who::Mob(id) => pickable_projectile_hit(*id).expect("a pickable projectile id"),
            Who::Player => panic!("not a projectile"),
        })
        .collect()
}

/// `Projectile.isPickable` is `#minecraft:redirectable_projectile`: a ghast's
/// fireball is in it, as both wind charges are, so another projectile's hit
/// test can strike it. A blaze's small fireball is not, and a projectile
/// that is stuck is no target.
#[test]
fn a_ghast_fireball_is_pickable_like_a_wind_charge() {
    let at = Vec3::new(0.0, 100.0, 0.0);
    let fired = |kind| Projectile::fireball(kind, at, None, [1.0, 0.0, 0.0]);
    let mut stuck = fired(ProjectileKind::LargeFireball { power: 1 });
    stuck.stuck = true;
    let list = vec![
        Projectile::new(ProjectileKind::EnderPearl, at, Some(Who::Player)),
        fired(ProjectileKind::LargeFireball { power: 1 }),
        fired(ProjectileKind::WindCharge),
        fired(ProjectileKind::BreezeWindCharge),
        fired(ProjectileKind::SmallFireball),
        stuck,
    ];
    assert_eq!(struck(&list), vec![1, 2, 3]);
}

/// A pearl thrown through a ghast's fireball strikes it (`Projectile.onHit`
/// then turns the fireball), where it used to fly on through.
#[test]
fn a_pearl_thrown_through_a_ghast_fireball_strikes_it() {
    let mut pearl = Projectile::new(
        ProjectileKind::EnderPearl,
        Vec3::new(0.0, 100.0, 0.0),
        Some(Who::Player),
    );
    pearl.velocity = Vec3::new(1.5, 0.0, 0.0);
    let fireball = Projectile::fireball(
        ProjectileKind::LargeFireball { power: 1 },
        Vec3::new(1.0, 99.5, 0.0),
        None,
        [0.0, 0.0, 1.0],
    );
    let list = vec![pearl, fireball];
    let alive = vec![true; list.len()];
    let world = Open(pickable_projectile_hitboxes(&list, &alive, 0));
    let mut pearl = list[0].clone();
    let (keep, hit) = pearl.tick(&world);
    assert!(!keep, "the pearl is spent on the hit");
    match hit {
        Hit::Entity {
            who: Who::Mob(id), ..
        } => assert_eq!(
            pickable_projectile_hit(id),
            Some(1),
            "it struck the fireball"
        ),
        other => panic!("the pearl flew through the fireball: {other:?}"),
    }
}
