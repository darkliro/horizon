use super::*;
use crate::game::{Audio, Options, Progress, Settings, commands::Target, load_resources};
use horizon_core::nbt::Tag;

#[test]
#[ignore = "daily"]
fn converted_mobs_keep_command_tags_at_the_new_id_and_on_save() {
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
    game.build_registries();
    for (number, (from, to)) in [
        ("husk", "zombie"),
        ("zombie", "drowned"),
        ("skeleton", "stray"),
    ]
    .into_iter()
    .enumerate()
    {
        let uuid = [10374, 1, 0, number as i32];
        let tags = Tag::List(vec![
            Tag::String("subject".into()),
            Tag::String(from.into()),
        ]);
        let raw = Tag::compound([
            (
                "id",
                Tag::String(format!("minecraft:{from}").as_str().into()),
            ),
            ("UUID", Tag::IntArray(uuid.to_vec())),
            (
                "Pos",
                Tag::List(vec![
                    Tag::Double(0.5_f64.to_bits()),
                    Tag::Double(70.0_f64.to_bits()),
                    Tag::Double(0.5_f64.to_bits()),
                ]),
            ),
            ("NoAI", Tag::Byte(1)),
            ("NoGravity", Tag::Byte(1)),
            ("Tags", tags.clone()),
            ("horizon:old_entity_only", Tag::Byte(1)),
        ]);
        let old_id = game.load_mob(raw, None).unwrap();
        assert_eq!(game.tags_of(Target::Mob(old_id)), ["subject", from]);
        let source = game.mobs.iter().find(|m| m.id == old_id).unwrap().clone();
        let new_id = game
            .convert_to(
                &source,
                &format!("minecraft:{to}"),
                horizon_client::mobs::conversion::Params::KEEP_ALL,
                false,
                |_, _| {},
            )
            .unwrap();
        assert_ne!(old_id, new_id);
        assert_eq!(
            game.tags_of(Target::Mob(new_id)),
            ["subject", from],
            "converted ID must keep tags"
        );
        let index = game.mobs.iter().position(|m| m.id == new_id).unwrap();
        let saved = game.mob_tree_nbt(index);
        assert_eq!(saved.get("Tags"), Some(&tags));
        assert_eq!(
            horizon_client::entity_nbt::kind(&saved).as_deref(),
            Some(format!("minecraft:{to}").as_str())
        );
        assert_ne!(horizon_client::entity_nbt::uuid(&saved), Some(uuid));
        assert!(saved.get("horizon:old_entity_only").is_none());
        game.mobs.remove(index);
        let reloaded = game.load_mob(saved, None).unwrap();
        assert_eq!(game.tags_of(Target::Mob(reloaded)), ["subject", from]);
    }
}
