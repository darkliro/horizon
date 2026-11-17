#[cfg(test)]
mod owner_attachment_save_tests {
    use horizon_core::nbt::Tag;
    use horizon_mod_runtime::attachments::{
        AttachmentCodec, AttachmentError, AttachmentType, Attachments,
    };

    static PROGRESS: AttachmentType<i32> = AttachmentType {
        namespace: "test",
        name: "progress",
        default: || 0,
        codec: Some(AttachmentCodec {
            encode: |n: &i32| Ok([("n".into(), Tag::Int(*n))].into()),
            decode: |c| match c.get(&"n".into()) {
                Some(Tag::Int(n)) => Ok(*n),
                _ => Err(AttachmentError::Decode),
            },
        }),
        copy_on_death: true,
        sync: false,
    };

    /// The live value as the named owner file would hold it.
    fn saved_value(game: &crate::game::Game<'_>) -> Option<i32> {
        let Some(Tag::Compound(map)) = game.level.player_extra.as_ref() else {
            return None;
        };
        let data = horizon_mod_runtime::attachments::attachment_compound(
            map.get(&horizon_mod_runtime::attachments::ATTACHMENTS_KEY.into()),
        )
        .ok()
        .flatten()?;
        PROGRESS
            .get_existing(
                &mut Attachments::from_nbt(data.clone()),
                horizon_mod_runtime::attachments::Holder::Level,
            )
            .ok()
            .flatten()
    }

    #[test]
    #[ignore = "daily"]
    fn an_autosave_writes_the_owners_live_attachments_to_the_file_it_loads() {
        let mut resources = None;
        let mut game = crate::game::game_tests::tests::server_game(&mut resources);
        let _rx = super::super::world_spawn_sync_tests::host_for(&mut game);
        let uuid = 1;
        game.lan.as_mut().unwrap().owner = Some(crate::game::lan::Owner { uuid, saved: None });
        // While the owner plays, its attachments are the game's, not the guest's.
        PROGRESS
            .set(
                &mut game.player_state.attachments,
                horizon_mod_runtime::attachments::Holder::Level,
                7,
            )
            .unwrap();
        assert!(game.store_owner().is_some());
        assert_eq!(
            saved_value(&game),
            Some(7),
            "the autosave stores the live value"
        );
        let host = game.lan.as_ref().unwrap();
        let g = host.hub.guests.iter().find(|g| g.uuid == uuid).unwrap();
        assert!(
            g.attachments.to_nbt().is_empty(),
            "the map stays their one owner"
        );
        // A later change is what the next autosave writes.
        PROGRESS
            .set(
                &mut game.player_state.attachments,
                horizon_mod_runtime::attachments::Holder::Level,
                9,
            )
            .unwrap();
        game.store_owner();
        assert_eq!(saved_value(&game), Some(9));
    }
}

#[cfg(test)]
mod connected_statistics_save_tests {
    use std::fs;

    fn connected_stats_save_control(stale: bool) {
        use super::super::super::{execution_role::ExecutionRole, stats};
        use std::sync::{Arc, Mutex};
        let mut resources = None;
        let mut game = crate::game::game_tests::tests::server_game(&mut resources);
        game.options.execution_role = ExecutionRole::DedicatedServer;
        // The command's dedicated-only guard also checks the launch mode.
        game.options.dedicated = Some(Vec::new());
        let dir = std::env::current_dir().unwrap().join(if stale {
            "stats-stale"
        } else {
            "stats-missing"
        });
        fs::create_dir_all(&dir).unwrap();
        game.world_root = dir.clone();
        game.save = Arc::new(horizon_client::save::SaveDir::new(
            dir.clone(),
            horizon_client::save::LevelShape::OVERWORLD,
        ));
        use horizon_client::server::user_lists::{BanList, IpBanList, OpList, WhiteList};
        game.server_lists = Some(Arc::new(Mutex::new(super::super::super::dedicated::Lists {
            ops: OpList::new(dir.join("ops.json")),
            white: WhiteList::new(dir.join("whitelist.json")),
            bans: BanList::new(dir.join("banned-players.json")),
            ip_bans: IpBanList::new(dir.join("banned-ips.json")),
            use_whitelist: false,
            enforce_whitelist: false,
            op_level: 4,
            player_idle_timeout: 0,
            stop: false,
            autosave: true,
            whitelist_bypass: Vec::new(),
        })));
        game.set_allow_commands(true);
        let _rx = super::super::world_spawn_sync_tests::host_for(&mut game);
        let host = game.lan.as_mut().unwrap();
        host.owner = None;
        host.hub
            .guests
            .update(0, |guest| guest.dimension = "minecraft:overworld".into());
        assert!(host.hub.guests[0].placed);
        assert!(!host.hub.guests[0].released);
        assert_eq!(host.hub.guests[0].uuid, 1);
        let path = stats::player_json(&dir, "stats", [0, 0, 0, 1]);
        assert!(!path.exists());
        let play = host.play.entry(1).or_default();
        play.stats.loaded = true;
        if stale {
            play.stats.add("used", "minecraft:diamond_sword", 4);
            play.stats.write(&path);
        }
        play.stats.add("used", "minecraft:diamond_sword", 1);
        let expected = if stale { 5 } else { 1 };
        // Packet dirty state is not the file's authority: a stats response can
        // consume it before the connected player's next server save.
        assert!(!play.stats.take_dirty().is_empty());
        assert_eq!(play.stats.get("used", "minecraft:diamond_sword"), expected);
        assert!(play.stats.take_dirty().is_empty());
        for command in ["save-all", "save-all flush"] {
            game.chat_capture = Some(Vec::new());
            // Match dedicated registration without changing the process-wide
            // tree used by unrelated integrated-server tests.
            let dispatcher = horizon_client::command::vanilla::build_for(false, true);
            game.ui.registries.permission = 4;
            let parsed = dispatcher.parse(command, &game.ui.registries);
            let (ctx, _) = parsed.executable().expect("registered dedicated command");
            let (root, ctx) = (parsed.context.clone(), ctx.clone());
            assert_eq!(game.run_chain(&root, &ctx).unwrap(), 1);
            let messages = game.chat_capture.take().unwrap();
            assert!(
                messages
                    .iter()
                    .any(|m| m.translation_key() == Some("commands.save.success")),
                "actual save completion"
            );
            let host = game.lan.as_ref().unwrap();
            assert_eq!(host.hub.guests.len(), 1, "saved while still connected");
            assert_eq!(
                host.play[&1].stats.get("used", "minecraft:diamond_sword"),
                expected
            );
            let raw = fs::read_to_string(&path)
                .expect("save-all must write the connected guest's actual stats file");
            let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
            assert_eq!(
                json["stats"]["minecraft:used"]["minecraft:diamond_sword"], expected,
                "{command}: current live value, not stale disk data"
            );
        }
    }

    #[test]
    #[ignore = "daily"]
    fn save_all_creates_the_connected_guests_statistics_file() {
        connected_stats_save_control(false);
    }

    #[test]
    #[ignore = "daily"]
    fn save_all_replaces_stale_connected_guest_statistics() {
        connected_stats_save_control(true);
    }
}

#[cfg(test)]
mod connected_statistics_save_exclusion_tests {

    /// Source controls for the bounded eligibility gates; these excluded
    /// guests are not counted as successfully saved players.
    #[test]
    #[ignore = "daily"]
    fn excluded_guest_records_are_not_invented_or_written() {
        let mut resources = None;
        let mut game = crate::game::game_tests::tests::server_game(&mut resources);
        let _rx = super::super::world_spawn_sync_tests::host_for(&mut game);
        game.world_root = std::env::current_dir().unwrap().join("stats-excluded");
        std::fs::create_dir_all(&game.world_root).unwrap();
        let path = super::super::super::stats::player_json(&game.world_root, "stats", [0, 0, 0, 1]);
        let mut host = game.lan.take().unwrap();
        host.play
            .entry(1)
            .or_default()
            .stats
            .add("used", "minecraft:diamond_sword", 1);
        host.hub.guests.update(0, |guest| guest.placed = false);
        game.save_connected_guest_statistics(&host);
        assert!(!path.exists(), "not placed: no statistics file");
        host.hub.guests.update(0, |guest| guest.placed = true);
        host.hub.guests.update(0, |guest| guest.released = true);
        game.save_connected_guest_statistics(&host);
        assert!(!path.exists(), "released: no statistics file");
        host.hub.guests.update(0, |guest| guest.released = false);
        let play = host.play.remove(&1).unwrap();
        game.save_connected_guest_statistics(&host);
        assert!(!path.exists(), "no play record: no invented statistics");
        host.play.insert(1, play);
        game.lan = Some(host);
        game.options.dedicated = None;
        game.save_game_metadata().unwrap();
        assert!(!path.exists(), "the existing dedicated gate is preserved");
        game.options.dedicated = Some(Vec::new());
        game.options.execution_role = crate::game::execution_role::ExecutionRole::Client;
        game.save_game_metadata().unwrap();
        assert!(!path.exists(), "the existing nonclient gate is preserved");
        game.options.execution_role = crate::game::execution_role::ExecutionRole::DedicatedServer;
        let host = game.lan.take().unwrap();
        game.save_game_metadata().unwrap();
        assert!(!path.exists(), "no hub: no substituted record");
        assert_eq!(
            host.play[&1].stats.get("used", "minecraft:diamond_sword"),
            1
        );
    }
}
