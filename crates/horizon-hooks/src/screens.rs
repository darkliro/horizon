//! Overlay screen registration (docs/layers.md, "Platform registries"): a crate above the
//! engine registers an interactive overlay screen over gameplay (such as a wheel menu) without an
//! engine file naming it.
use crate::{TooLate, registry::Registry};

/// An interactive overlay screen over the world that does not pause gameplay.
pub trait OverlayScreen: Sync {
    /// Stable and unique id: `<crate>.<screen>`, as `horizon-ctl` and logs name it.
    fn id(&self) -> &'static str;
    /// Translation key for the screen title, for narration and accessibility.
    fn title(&self) -> &'static str;
    /// The action or key mapping that opens this overlay (`key.emote`).
    fn trigger_action(&self) -> Option<&'static str> {
        None
    }
}

static SCREENS: Registry<dyn OverlayScreen> = Registry::new();

/// Registers an overlay screen before the game starts (as [`crate::register`]).
/// A later call is refused.
pub fn register(screen: &'static dyn OverlayScreen) -> Result<(), TooLate> {
    SCREENS.register(screen)
}

/// The engine's side. Only the engine calls it.
pub mod engine {
    use super::*;

    /// Every registered overlay screen, in registration order.
    pub fn all() -> &'static [&'static dyn OverlayScreen] {
        SCREENS.all()
    }

    /// Finds a registered overlay screen by id or trigger action.
    pub fn find_by_trigger(action: &str) -> Option<&'static dyn OverlayScreen> {
        all()
            .iter()
            .copied()
            .find(|s| s.trigger_action() == Some(action))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestOverlay;
    impl OverlayScreen for TestOverlay {
        fn id(&self) -> &'static str {
            "test.overlay"
        }
        fn title(&self) -> &'static str {
            "test.overlay.title"
        }
        fn trigger_action(&self) -> Option<&'static str> {
            Some("key.test")
        }
    }

    static TEST_OVERLAY: TestOverlay = TestOverlay;

    #[test]
    fn screen_registered_before_the_game_starts_is_searchable() {
        assert_eq!(register(&TEST_OVERLAY), Ok(()));
        let found = engine::find_by_trigger("key.test");
        assert!(found.is_some());
        assert_eq!(found.unwrap().id(), "test.overlay");
        assert_eq!(register(&TEST_OVERLAY), Err(TooLate));
    }
}
