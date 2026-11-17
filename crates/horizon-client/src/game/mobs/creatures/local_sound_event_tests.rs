use crate::game::{Audio, Game, Options, Progress, Settings, load_resources};
use horizon_client::mobs::MobEvent;

#[test]
#[ignore = "daily"]
fn local_mob_events_play_without_entering_the_shared_capture() {
    let root = std::path::PathBuf::from(std::env::var_os("HORIZON_TEST_RESOURCES").unwrap());
    let settings = Settings::default();
    let loaded = load_resources(
        horizon_assets::resource_source::ResourceRoot::directory(root.clone()),
        Vec::new(),
        None,
        &root,
        &settings,
        std::time::Instant::now(),
        false,
        crate::game::execution_role::ExecutionRole::IntegratedServer,
        &Progress::new(),
        &mut |_, _| unreachable!(),
    )
    .unwrap();
    let mut options = Options::parse(std::iter::empty()).unwrap().unwrap();
    options.resources = root.clone();
    options.game_dir =
        Some(std::env::temp_dir().join(format!("horizon-h069-unit-{}", std::process::id())));
    options.execution_role = crate::game::execution_role::ExecutionRole::IntegratedServer;
    let mut game = Game::new(
        &loaded.resources,
        loaded.pack,
        loaded.gui.unwrap(),
        options,
        settings,
        Audio::silent(&horizon_assets::resource_source::ResourceSource::directory(
            root.clone(),
        )),
        None,
    )
    .unwrap();
    game.audio.capture = Some(Vec::new());
    game.handle_mob_events(vec![
        MobEvent::Sound("entity.cow.hurt", [1.0, 100.0, 2.0], 0.5, 1.0),
        MobEvent::LocalSound("entity.breeze.whirl", [3.0, 100.0, 4.0], 0.8, 0.7),
        MobEvent::Sound("entity.zombie.hurt", [5.0, 100.0, 6.0], 1.0, 0.9),
    ])
    .unwrap();
    // A positive LocalSound must still take the playback path. An absent
    // producer cannot satisfy this actual Game adapter regression.
    assert_eq!(
        game.audio.heard.get("minecraft:entity.breeze.whirl"),
        Some(&1)
    );
    assert_eq!(game.audio.heard.get("minecraft:entity.cow.hurt"), Some(&1));
    assert_eq!(
        game.audio.heard.get("minecraft:entity.zombie.hurt"),
        Some(&1)
    );
    let shared = game.audio.capture.as_ref().unwrap();
    assert_eq!(shared.len(), 2, "LocalSound must not become a shared sound");
    assert_eq!(shared[0].event, "minecraft:entity.cow.hurt");
    assert_eq!(shared[1].event, "minecraft:entity.zombie.hurt");
    assert_eq!(
        (
            shared[0].position,
            shared[0].volume,
            shared[0].pitch,
            shared[0].source
        ),
        ([1.0, 100.0, 2.0], 0.5, 1.0, 6)
    );
    assert_eq!(
        (
            shared[1].position,
            shared[1].volume,
            shared[1].pitch,
            shared[1].source
        ),
        ([5.0, 100.0, 6.0], 1.0, 0.9, 5)
    );
}
