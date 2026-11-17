use super::super::native_entity_refs::tests::with_game;
use super::super::native_item_places::tests::{guest, open_to_lan};
use horizon_client::chunk::Chunk;
use horizon_client::mobs::{Mob, enderman::stared_at};
use horizon_core::geometry::Vec3;
use std::sync::{Arc, Mutex};

/// Keeps enderman `self.0` calm, and logs who stared at whom.
struct Calm(u32, Arc<Mutex<Vec<(u32, u128)>>>);
#[mod_api::__subscriptions]
impl mod_api::Hooks for Calm {
    fn enderman_anger(
        &self,
        _: &dyn mod_api::Host,
        event: &mut mod_api::EndermanAngerEvent,
    ) -> mod_api::Flow {
        let asked = (event.enderman.id(), event.player.uuid());
        self.1.lock().unwrap().push(asked);
        if event.enderman.id() == self.0 {
            mod_api::Flow::Cancel
        } else {
            mod_api::Flow::Continue
        }
    }
}

/// `EnderManAngerEvent`: a guest stares straight at two endermen; the
/// mod keeps one calm, the other still counts the stare.
#[test]
#[ignore = "daily"]
fn a_mod_keeps_one_stared_at_enderman_calm() {
    let heard = Arc::default();
    let (calm, angry) = (900, 901);
    let mods = vec![mod_api::ModEntry::new(
        "calm",
        Box::new(Calm(calm, Arc::clone(&heard))),
    )]
    .into();
    mod_platform::testing::with_mods(mods, || {
        with_game(|game| {
            open_to_lan(game);
            game.world
                .chunks
                .insert((0, 0), Arc::new(Chunk::empty(0, 0)));
            let mut staring = guest(5);
            staring.loaded = true;
            staring.position = [8.0, 64.0, 2.0];
            game.lan.as_mut().unwrap().hub.guests.push(staring);
            let views = game.lan_guest_views();
            let mut view = views[0].1.clone();
            let kind = game
                .res
                .mob_types()
                .get("minecraft:enderman")
                .unwrap()
                .clone();
            let endermen = [calm, angry]
                .map(|id| Mob::new(id, kind.clone(), Vec3::new(8.0, 64.0, 10.0), 0.0));
            let eye = endermen[0].eye_y();
            view.position = Vec3::new(8.0, 64.0, 2.0);
            view.eye = [8.0, eye, 2.0];
            view.look = [0.0, 0.0, 1.0];
            view.head = None;
            let mut world = game.mob_world_base(&[]);
            assert!(world.mob_hooks.is_some(), "asked only while a mod listens");
            world.player = Some(view);
            assert!(!stared_at(&endermen[0], &world), "the mod keeps it calm");
            assert!(stared_at(&endermen[1], &world));
            assert_eq!(*heard.lock().unwrap(), [(calm, 127), (angry, 127)]);
        });
    });
}
