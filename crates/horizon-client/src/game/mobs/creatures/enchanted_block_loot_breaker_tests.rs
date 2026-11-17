use super::super::native_entity_refs::tests::with_game;
use super::{MobEvent, Projectile, ProjectileKind};
use horizon_core::geometry::Vec3;
use mod_api::{EnchantedBlockLootEvent, Hooks, LevelHost, ModEntry};
use mod_platform::testing::with_mods;
use std::sync::{Arc, Mutex};

const POS: [i32; 3] = [1, 64, 1];

/// Records the entity of every read.
struct Heard(Arc<Mutex<Vec<Option<u32>>>>);
#[mod_api::__subscriptions]
impl Hooks for Heard {
    fn enchanted_block_loot(&self, _: &dyn LevelHost, event: &mut EnchantedBlockLootEvent<'_>) {
        self.0.lock().unwrap().push(event.entity.map(|e| e.id()));
    }
}

fn place(game: &mut super::Game<'_>, block: &str) {
    let state = game.pack.assets.registry.default_state(block).unwrap();
    game.world.set(POS[0], POS[1], POS[2], state).unwrap();
}
fn projectile(kind: ProjectileKind, id: u32) -> Projectile {
    let mut projectile = Projectile::new(kind, Vec3::ZERO, None);
    projectile.id = id;
    projectile
}

/// `destroyBlock(pos, true, entity)` reaches the listener as `getEntity`:
/// the projectile on a chorus flower or a decorated pot, the villager or
/// ravager on a crop. A speleothem's `destroyBlock(pos, true)` names
/// nobody. Vanilla's tables for those three read no level, so a data pack
/// makes them read one.
#[test]
#[ignore = "daily"]
fn the_entity_destroy_block_names_reaches_the_listener() {
    let root = std::env::temp_dir().join(format!("block-loot-{}", std::process::id()));
    let dir = root.join("data/minecraft/loot_table/blocks");
    std::fs::create_dir_all(&dir).unwrap();
    let table = r#"{"pools":[{"rolls":1,"conditions":[{"type":"minecraft:table_bonus",
        "enchantment":"minecraft:fortune","chances":[1.0]}],
        "entries":[{"type":"minecraft:item","name":"minecraft:stick"}]}]}"#;
    for block in [
        "chorus_flower",
        "decorated_pot",
        "pointed_dripstone",
        "wheat",
    ] {
        std::fs::write(dir.join(format!("{block}.json")), table).unwrap();
    }
    let tables =
        horizon_world::loot::shared_tables(&root, &horizon_world::loot::PackOverlay::NONE)
            .unwrap();
    let _ = std::fs::remove_dir_all(&root);
    with_game(|game| {
        game.options.dedicated = Some(Vec::new());
        game.world
            .chunks
            .insert((0, 0), Arc::new(horizon_client::chunk::Chunk::empty(0, 0)));
        let mut items = game.items().clone();
        items.set_loot_tables(tables);
        game.server_items = Some(Arc::new(items));
        let heard = Arc::new(Mutex::new(Vec::new()));
        let mods = vec![ModEntry::new("breaker", Box::new(Heard(heard.clone())))].into();
        with_mods(mods, || {
            let take = || std::mem::take(&mut *heard.lock().unwrap());

            place(game, "minecraft:chorus_flower");
            game.projectile_breaks_block(&projectile(ProjectileKind::Snowball, 77), POS);
            assert_eq!(take(), [Some(77)]);

            place(game, "minecraft:decorated_pot");
            game.projectile_breaks_block(&projectile(ProjectileKind::Egg, 78), POS);
            assert_eq!(take(), [Some(78)]);

            place(game, "minecraft:pointed_dripstone");
            let mut trident = projectile(ProjectileKind::Trident, 79);
            trident.arrow.as_mut().unwrap().hit_speed = 1.0;
            game.projectile_breaks_block(&trident, POS);
            assert_eq!(take(), [None]);

            place(game, "minecraft:wheat");
            let harvest = MobEvent::BreakBlock {
                pos: POS,
                drop: true,
                by: None,
                breaker: Some(80),
            };
            game.handle_mob_events(vec![harvest]).unwrap();
            assert_eq!(take(), [Some(80)]);
        });
    });
}
