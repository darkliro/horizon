#[test]
#[ignore = "daily"]
fn empty_mod_never_defers_taming_or_block_destruction() {
    use mod_api::{self, Hooks};
    struct Empty;
    #[mod_api::__subscriptions]
    impl Hooks for Empty {}
    let mods: mod_api::Mods = vec![mod_api::ModEntry::new("empty", Box::new(Empty))].into();
    mod_platform::testing::with_mods(mods, || {
        super::super::native_entity_refs::tests::with_game(|game| {
            let world = game.mob_world_base(&[]);
            assert!(!world.asks_animal_tame && !world.asks_living_destroy_block);
            let mut cancel = false;
            game.native_animal_tame(1, 1, 2, "minecraft:horse", &mut cancel);
            game.native_living_destroy_block(
                2,
                "minecraft:ravager",
                [0; 3],
                game.world.air,
                &mut cancel,
            );
            let world = game.mob_world_base(&[]);
            assert!(!world.asks_animal_tame && !world.asks_living_destroy_block);
            assert!(!cancel);
        });
    });
}
