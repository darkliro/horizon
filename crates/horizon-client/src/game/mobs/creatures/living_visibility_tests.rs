use super::super::native_entity_refs::tests::with_game;
use super::super::native_item_places::tests::{guest, open_to_lan};
use std::sync::{Arc, Mutex};

/// Halves the player's visibility, and logs what it was asked.
struct Halve(Arc<Mutex<Vec<(u32, Option<u32>, f64)>>>);
#[mod_api::__subscriptions]
impl mod_api::Hooks for Halve {
    fn living_visibility(&self, _: &dyn mod_api::Host, event: &mut mod_api::LivingVisibilityEvent) {
        let looking = event.looking.map(|e| e.id());
        self.0
            .lock()
            .unwrap()
            .push((event.entity.entity().id(), looking, event.visibility));
        event.visibility *= 0.5;
    }
}

/// `LivingVisibilityEvent` for a guest a mob targets: the mod gets the
/// guest, the mob and vanilla's percent, and its answer is the percent.
#[test]
#[ignore = "daily"]
fn a_mod_halves_a_guests_visibility() {
    let heard = Arc::default();
    let mods = vec![mod_api::ModEntry::new(
        "halve",
        Box::new(Halve(Arc::clone(&heard))),
    )]
    .into();
    mod_platform::testing::with_mods(mods, || {
        with_game(|game| {
            open_to_lan(game);
            let mut seen = guest(5);
            seen.loaded = true;
            game.lan.as_mut().unwrap().hub.guests.push(seen);
            let view = game.lan_guest_views()[0].1.clone();
            let world = game.mob_world_base(&[]);
            assert!(world.mob_hooks.is_some(), "asked only while a mod listens");
            assert_eq!(world.player_visibility(&view, 900, 0.8), 0.4);
            assert_eq!(*heard.lock().unwrap(), [(24, Some(900), 0.8)]);
        });
    });
}
