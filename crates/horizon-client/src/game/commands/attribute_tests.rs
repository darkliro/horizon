//! `/attribute … modifier value get` must not create the instance it looks up.
//! `AttributeCommand.getAttributeModifier` checks `hasAttribute` (in
//! `getEntityWithAttribute`), then `hasModifier`, and then reads
//! `getModifierValue`; all three use the cached instance or the type's supplier,
//! so a failed lookup leaves the entity's attribute map — and the `attributes`
//! list `LivingEntity.addAdditionalSaveData` writes — untouched.
use crate::game::{Audio, Game, Options, Progress, Settings, load_resources};
use horizon_client::motion::PlayerMotion;
use horizon_core::geometry::Vec3;
use horizon_world::nbt::Tag;
use std::path::PathBuf;

fn with_game(test: impl FnOnce(&mut Game<'_>)) {
    let root = PathBuf::from(std::env::var_os("HORIZON_TEST_RESOURCES").unwrap());
    let settings = Settings::default();
    let loaded = load_resources(
        root.clone(),
        None,
        &root,
        &settings,
        std::time::Instant::now(),
        false,
        true,
        &Progress::new(),
        &mut |_, _| unreachable!(),
    )
    .unwrap();
    let mut options = Options::parse(std::iter::empty()).unwrap().unwrap();
    options.resources = root.clone();
    options.game_dir = Some(
        PathBuf::from(std::env::var_os("HORIZON_HOME").unwrap())
            .join(format!("h-attrs-{:?}", std::thread::current().id())),
    );
    options.execution_role = crate::game::execution_role::ExecutionRole::IntegratedServer;
    let mut game = Game::new(
        &loaded.resources,
        loaded.pack,
        loaded.gui.unwrap(),
        options,
        settings,
        Audio::silent(&root),
        None,
    )
    .unwrap();
    game.build_registries();
    game.set_allow_commands(true);
    test(&mut game);
}

/// What a command says, and whether it succeeded.
fn say(game: &mut Game<'_>, command: &str) -> (bool, String) {
    game.chat_capture = Some(Vec::new());
    let ok = game.run_command(command);
    let said = game.chat_capture.take().unwrap_or_default();
    (
        ok,
        said.iter().map(horizon_client::text::Text::plain).collect(),
    )
}

/// The `attributes` list the next save writes for the player.
fn saved_attributes(game: &mut Game<'_>) -> Tag {
    game.sync_level();
    game.level
        .player_extra
        .as_ref()
        .and_then(|extra| extra.get("attributes"))
        .cloned()
        .expect("the player's save carries its attributes")
}

#[test]
#[ignore = "needs prepared resources in HORIZON_TEST_RESOURCES"]
fn failed_modifier_value_get_leaves_the_attribute_store_and_its_save_alone() {
    with_game(|game| {
        game.player = Some(PlayerMotion::new(game.motion, Vec3::new(4.5, 100.0, 4.5)).unwrap());
        let key = game
            .player
            .as_ref()
            .unwrap()
            .attributes
            .key("minecraft:luck");
        assert!(key.is_some(), "a player has luck");
        assert!(
            game.player
                .as_ref()
                .unwrap()
                .attributes
                .lookup(key.as_ref())
                .is_none(),
            "an attribute the player never touched has no instance"
        );
        let before = saved_attributes(game);
        let (ok, said) = say(
            game,
            "attribute @s minecraft:luck modifier value get test:none",
        );
        assert!(!ok, "the lookup fails: {said}");
        assert!(
            said.contains("has no modifier test:none"),
            "the missing modifier is what fails, not the attribute: {said}"
        );
        assert_eq!(
            saved_attributes(game),
            before,
            "a failed lookup must not write a new attribute into the save"
        );
        assert!(
            game.player
                .as_ref()
                .unwrap()
                .attributes
                .lookup(key.as_ref())
                .is_none(),
            "nor create the instance"
        );
        // getEntityWithAttribute checks hasAttribute first: an attribute the
        // player's supplier does not have is the attribute error, not the
        // modifier one, and that check creates nothing either.
        let (ok, said) = say(
            game,
            "attribute @s minecraft:flying_speed modifier value get test:none",
        );
        assert!(!ok, "the lookup fails: {said}");
        assert!(
            said.contains("has no attribute"),
            "an attribute the player lacks is the attribute error: {said}"
        );
        assert_eq!(saved_attributes(game), before, "still no new attribute");
        // The same read on a modifier that exists still finds its amount.
        assert!(
            game.run_command("attribute @s minecraft:luck modifier add test:luck 0.25 add_value")
        );
        let (ok, said) = say(
            game,
            "attribute @s minecraft:luck modifier value get test:luck",
        );
        assert!(ok, "an existing modifier reads: {said}");
        assert!(said.contains("is 0.25"), "its amount: {said}");
    });
}
