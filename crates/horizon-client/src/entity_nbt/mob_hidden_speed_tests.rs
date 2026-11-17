use super::*;
use crate::mobs::{Species, sensor_scan_tests::kind};
fn types() -> HashMap<String, Arc<MobType>> {
    let mut pig = (*kind()).clone();
    pig.id = "minecraft:pig".into();
    pig.species = Species::Pig;
    HashMap::from([("minecraft:pig".to_owned(), Arc::new(pig))])
}
fn saved(m: &Mob) -> Tag {
    mob_to_nbt(m, [202, 0, 0, 1], None, None, |_| None)
}
#[test]
fn loaded_speed_fallback_survives_a_no_raw_save() {
    let input = horizon_core::nbt::parse_snbt(r#"{id:"minecraft:pig",UUID:[I;202,0,0,1],Pos:[2.5d,200d,4.5d],Health:10f,NoAI:1b,NoGravity:1b,active_effects:[{id:"minecraft:speed",amplifier:1b,duration:40,show_icon:1b,hidden_effect:{duration:200,ambient:1b,show_particles:0b,show_icon:0b}}]}"#).unwrap().0;
    let m = mob_from_nbt(&input, &types(), 7, None).unwrap();
    assert_eq!(m.id, 7);
    assert_eq!(m.kind.id, "minecraft:pig");
    assert!(m.no_ai && m.no_gravity);
    assert_eq!(m.health, 10.0);
    assert_eq!(
        m.extra
            .effects
            .iter()
            .map(|e| (e.id, e.duration, e.amplifier))
            .collect::<Vec<_>>(),
        [("minecraft:speed", 40, 1)]
    );
    let tag = saved(&m);
    let hidden = tag
        .get("active_effects")
        .and_then(|t| match t {
            Tag::List(v) => v.first(),
            _ => None,
        })
        .and_then(|e| e.get("hidden_effect"));
    assert_eq!(
        hidden.and_then(|t| t.get("duration")),
        Some(&Tag::Int(200)),
        "real decoder must retain weaker duration without raw input"
    );
}
// Added after the frozen old proof: modeled mutation and stale raw controls.
#[test]
fn fallback_model_wins_over_stale_raw_and_clears_without_resurrection() {
    let input = horizon_core::nbt::parse_snbt(r#"{id:"minecraft:pig",Pos:[2.5d,200d,4.5d],Tags:["untouched"],active_effects:[{id:"minecraft:speed",amplifier:1b,duration:40,hidden_effect:{duration:200,ambient:1b,show_particles:0b,show_icon:0b}}]}"#).unwrap().0;
    let mut m = mob_from_nbt(&input, &types(), 7, None).unwrap();
    let h = &m.extra.effect_chains[0];
    assert_eq!(
        (
            h.id,
            h.duration,
            h.amplifier,
            h.ambient,
            h.visible,
            h.show_icon
        ),
        ("minecraft:speed", 200, 0, true, false, false)
    );
    m.extra.effect_chains[0].duration = 123;
    m.extra.effects[0].duration = 30;
    let actual = mob_to_nbt(&m, [202, 0, 0, 1], Some(&input), None, |_| None);
    let first = |t: &Tag| match t.get("active_effects").unwrap() {
        Tag::List(v) => v[0].clone(),
        _ => panic!(),
    };
    assert_eq!(first(&actual).get("duration"), Some(&Tag::Int(30)));
    assert_eq!(
        first(&actual).get("hidden_effect").unwrap().get("duration"),
        Some(&Tag::Int(123))
    );
    assert_eq!(actual.get("Tags"), input.get("Tags"));
    m.extra.effect_chains.clear();
    for raw in [None, Some(&input)] {
        let actual = mob_to_nbt(&m, [202, 0, 0, 1], raw, None, |_| None);
        assert!(first(&actual).get("hidden_effect").is_none());
    }
}
