use horizon_hooks::{TooLate, actions, engine, screens};

struct Overlay(&'static str);

impl screens::OverlayScreen for Overlay {
    fn id(&self) -> &'static str {
        self.0
    }

    fn title(&self) -> &'static str {
        "test.overlay.title"
    }
}

static BEFORE: Overlay = Overlay("test.before");
static AFTER: Overlay = Overlay("test.after");
static MAPPINGS: &actions::Mappings = &[("key.test", "key.keyboard.k", "misc")];

// A separate integration-test process keeps these one-shot registries independent
// of the library's registration tests.
#[test]
fn engine_seal_freezes_actions_and_overlays() {
    assert_eq!(actions::register(MAPPINGS), Ok(()));
    assert_eq!(screens::register(&BEFORE), Ok(()));

    engine::seal();

    // No registry read before this: seal itself must refuse late registration.
    assert_eq!(actions::register(MAPPINGS), Err(TooLate));
    assert_eq!(screens::register(&AFTER), Err(TooLate));
    assert_eq!(actions::engine::all(), &[MAPPINGS]);
    let ids: Vec<_> = screens::engine::all().iter().map(|s| s.id()).collect();
    assert_eq!(ids, ["test.before"]);

    engine::seal();
    assert_eq!(screens::register(&AFTER), Err(TooLate));
}
