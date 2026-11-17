use super::*;
use mod_api::{self, Hooks, LivingConversionPostEvent, ModEntry, ServerHost};
use std::sync::Mutex;

#[test]
#[ignore = "daily"]
fn post_only_listener_is_subscribed_without_a_pre_listener() {
    struct DefaultMod;
    #[mod_api::__subscriptions]
    impl Hooks for DefaultMod {}
    struct PostOnly(Arc<Mutex<Vec<(u32, u32)>>>);
    #[mod_api::__subscriptions]
    impl Hooks for PostOnly {
        fn living_conversion_post(
            &self,
            _: &mut dyn ServerHost,
            event: &mut LivingConversionPostEvent<'_>,
        ) {
            assert_eq!(event.kind, "minecraft:zombie");
            assert_eq!(event.converted_kind, "minecraft:drowned");
            self.0.lock().unwrap().push((event.entity, event.converted));
        }
    }
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mods = vec![
        ModEntry::new("default", Box::new(DefaultMod)),
        ModEntry::new("post", Box::new(PostOnly(seen.clone()))),
    ]
    .into();
    mod_platform::testing::with_mods(mods, || {
        super::super::native_entity_refs::tests::with_game(|game| {
            let zombie = game
                .res
                .mob_types()
                .get("minecraft:zombie")
                .unwrap()
                .clone();
            for _ in 0..2 {
                game.next_entity += 1;
                let from = Mob::new(
                    game.next_entity,
                    zombie.clone(),
                    Vec3::new(0.5, 64.0, 0.5),
                    0.0,
                );
                let converted = game
                    .convert_to(
                        &from,
                        "minecraft:drowned",
                        horizon_client::mobs::conversion::Params::KEEP_ALL,
                        false,
                        |_, _| {},
                    )
                    .unwrap();
                let events = seen.lock().unwrap();
                assert_eq!(events.last(), Some(&(from.id, converted)));
            }
            assert!(!mod_api::__private::listened(
                mod_api::HookId::LivingConversionPre
            ));
            assert_eq!(seen.lock().unwrap().len(), 2);
        });
    });
}
