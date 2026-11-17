use super::*;
use horizon_client::health::{DamageHooks, DamageLevel};

/// What the hook of these tests saw: the player, its source (as damage
/// type, causing, direct) and the push.
type Pushes = std::sync::Mutex<Vec<(i32, (String, i32, i32), Knock)>>;
static PUSHES: Pushes = std::sync::Mutex::new(Vec::new());

/// Sets the native knockback hook of these tests for players 9201 to 9203: it records the push, cancels
/// 9201's and doubles 9202's strength and turns its direction around.
fn hook(_: DamageLevel<'_>, pushed: Hurting<'_>, knock: &mut Knock) -> bool {
    let Hurting::Player(entity @ 9201..=9203, s) = pushed else {
        return false;
    };
    let source = (s.damage_type.to_owned(), s.causing, s.direct);
    PUSHES.lock().unwrap().push((entity, source, *knock));
    match entity {
        9201 => return true,
        9202 => {
            knock.strength *= 2.0;
            knock.z = -knock.z;
        }
        _ => {}
    }
    false
}

/// `LivingEntity.knockback` on a player: the hook runs before the
/// knockback resistance and the motion changes; a mod's strength or
/// direction pushes, a cancel pushes nothing, and without a mod in the way
/// the push is vanilla's. Needs the prepared data in
/// `.local/horizon-client/26.3`; skipped without it.
#[test]
fn the_native_knockback_hook_changes_or_cancels_a_players_push() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../.local/horizon-client")
        .join(horizon_core::VANILLA_VERSION);
    if !root.join("movement-properties.nbt").exists() {
        return;
    }
    let assets = horizon_client::assets::Assets::load(&root).unwrap();
    let resources = horizon_client::motion::MotionResources::load(&root, &assets).unwrap();
    let hooks = DamageHooks {
        knockback: Some(hook),
        ..DamageHooks::NONE
    };
    let mut test_level = horizon_client::health::TestLevel::with_hooks(1, hooks);
    let hit = HurtSource {
        damage_type: "minecraft:player_attack",
        causing: 5,
        direct: 5,
        position: None,
    };
    let mut push = |entity, resistance: f64| {
        let mut p = horizon_client::motion::PlayerMotion::new(&resources, Vec3::ZERO).unwrap();
        p.entity_id = entity;
        p.ground.on_ground = true;
        if resistance != 0.0 {
            let attribute = "minecraft:knockback_resistance";
            let set = p.set_modifier(attribute, "horizon:test", resistance, "add_value", true);
            set.unwrap();
        }
        knockback::apply_player_knockback(
            &mut p,
            &mut test_level.level(),
            0.4,
            (0.0, 3.0),
            hit,
            false,
        );
        (p.velocity.x, p.velocity.y, p.velocity.z)
    };
    // No mod in the way: vanilla's push, away from z = 3.
    assert_eq!(push(9200, 0.0), (0.0, 0.4, -0.4));
    // Canceled: no motion. The hook saw the push and the hit.
    assert_eq!(push(9201, 0.0), (0.0, 0.0, 0.0));
    let made = Knock {
        strength: 0.4,
        x: 0.0,
        z: 3.0,
    };
    let named = ("minecraft:player_attack".to_owned(), 5, 5);
    let seen = |entity| -> Vec<_> {
        let seen = PUSHES.lock().unwrap();
        seen.iter().filter(|p| p.0 == entity).cloned().collect()
    };
    assert_eq!(seen(9201), [(9201, named, made)]);
    // Twice the strength, direction turned around.
    assert_eq!(push(9202, 0.0), (0.0, 0.4, 0.8));
    // Before the resistance (`LivingKnockBackEvent`): a 0.5 resistance still hands the hook
    // the raw strength, and the push that follows is halved.
    assert_eq!(push(9203, 0.5), (0.0, 0.2, -0.2));
    assert_eq!(seen(9203)[0].2.strength, 0.4);
}
