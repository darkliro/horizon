//! Addresses a higher layer opens a tunnel for: `friend:<code>`, Play with Friends' invite codes
//! (a Horizon crate, docs/play-with-friends.md). The engine keeps the address form and the join
//! (`net.rs`); the [`Resolver`] a registered hook offers ([`Hooks::tunnel`](crate::Hooks::tunnel))
//! reads the codes and opens the tunnel, which the client then reaches on a loopback port like any
//! server. Without one (horizon-client's own executable, the server build) no code is one, and
//! such an address is refused, never retried.
//!
//! Hosting needs no hook here: the tunnel's host end connects to the world's LAN port over
//! loopback, so its guests are LAN guests.
/// What starts a tunnel address.
pub const PREFIX: &str = "friend:";

/// How a join is going, shown on the connect screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Progress {
    /// Finding the host from the code.
    Finding,
    /// The tunnel reached the host: `direct`ly, or through a relay.
    Connected { direct: bool },
}

impl Progress {
    /// Translation key for the progress stage shown to the player.
    pub fn key(self) -> &'static str {
        match self {
            Progress::Finding => "horizon.multiplayer.finding_friend",
            Progress::Connected { direct: true } => "horizon.multiplayer.connected_friend_direct",
            Progress::Connected { direct: false } => "horizon.multiplayer.connected_friend_relay",
        }
    }
}

/// The tunnel's local end: the loopback port to connect to, and the token to send on it first
/// (the port takes only the client that sends it).
pub struct Opened {
    pub port: u16,
    pub token: Vec<u8>,
}

/// Reads tunnel codes and opens tunnels. The join calls it on the network thread; the address
/// shown to players asks it on the game thread, never per frame.
pub trait Resolver: Send + Sync {
    /// Whether `code` is a code; with `full`, only a full one (a short relay code could also be a
    /// host name).
    fn is_code(&self, code: &str, full: bool) -> bool;
    /// A short code's name, the part players may see (`K7QM` of `K7QM-3XPD`); None for any other
    /// code, whose secret is all through it.
    fn nameplate(&self, code: &str) -> Option<String>;
    /// An address as players and logs see it without revealing secrets: the short code's
    /// nameplate or generic text.
    fn shown(&self, code: &str) -> String {
        match self.nameplate(code) {
            Some(nameplate) => format!("Friend's world ({nameplate}-****)"),
            None => "Friend's world".to_owned(),
        }
    }
    /// What tells one shared world from another, without any secret: native mods key a server's
    /// data by it.
    fn public_part(&self, code: &str) -> String;
    /// Reach the host from `code`, directly or through a relay (the resolver's own setting), and
    /// open the tunnel's local end.
    fn open(&self, code: &str, progress: &mut dyn FnMut(Progress)) -> Result<Opened, String>;
}
