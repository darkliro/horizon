import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from tools.ci.rust_changes import changed_paths, needs_launcher, needs_rust, needs_server_build


class RustChangesTests(unittest.TestCase):
    def test_only_content_changes_skip_rust(self):
        self.assertFalse(needs_rust([]))
        self.assertFalse(needs_rust([
            "docs/setup.md", "docs/media/banner.webp", "README.md",
        ]))
        # The Python tools: verify's Python steps run their tests, and no Rust build or test reads them.
        self.assertFalse(needs_rust([
            "tools/parity.py", "tools/test_tick_parity.py", "tests/test_ci_runners.py", "docs/setup.md",
        ]))
        for path in [
            "crates/horizon-client/src/game.rs", "xtask/src/main.rs", "Cargo.lock",
            ".cargo/config.toml", ".github/workflows/ci.yml", "tools/ci/rust_changes.py",
            "tools/release_train.py", "tools/ci/sign_release.sh", "tools/native-hook-fragments/a.json",
            "tools/parity_hosts.json", "mods/horizon/src/x.py",
            "research-lock.json", "crates/horizon-theme/tokens.css", "unknown-file",
        ]:
            with self.subTest(path=path):
                self.assertTrue(needs_rust(["README.md", path]))

    def test_launcher_checks_follow_the_crates_it_builds(self):
        for path in ["crates/horizon-client/src/game.rs", "launcher/links.json", "launcher/docs/a.md",
                     "launcher/README.md", "website/src/index.html", "tools/pr_ready.py"]:
            with self.subTest(path=path):
                self.assertFalse(needs_launcher([path]))
        for path in ["launcher/src/main.rs", "launcher/Cargo.lock", "crates/horizon-port-pack/src/net.rs",
                     "crates/horizon-mcp/src/lib.rs", "crates/horizon-mod-api-gen/src/injection.rs",
                     "crates/horizon-theme/src/lib.rs", "Cargo.toml", "Cargo.lock",
                     ".github/workflows/ci.yml"]:
            with self.subTest(path=path):
                self.assertTrue(needs_launcher(["README.md", path]))

    def test_the_server_build_clippy_follows_the_client_crates_and_manifests(self):
        for path in ["crates/horizon-world/src/lib.rs", "crates/horizon-perf/src/lib.rs", "tools/parity.py",
                     "docs/setup.md", "launcher/src/main.rs"]:
            with self.subTest(path=path):
                self.assertFalse(needs_server_build([path]))
        for path in ["crates/horizon-client/src/game.rs", "crates/horizon-client-render/src/hud.rs",
                     "crates/horizon-client-core/src/lang.rs", "crates/horizon-world/Cargo.toml", "Cargo.lock",
                     "Cargo.toml", ".cargo/config.toml", ".github/workflows/ci.yml", "tools/ci/rust_changes.py"]:
            with self.subTest(path=path):
                self.assertTrue(needs_server_build(["docs/setup.md", path]))

    def test_moving_source_into_docs_still_requires_rust(self):
        with tempfile.TemporaryDirectory() as directory:
            def git(*args):
                return subprocess.check_output(["git", "-C", directory, *args])

            git("init", "--quiet")
            source = Path(directory, "source.rs")
            source.write_text("fn main() {}\n")
            git("add", ".")
            git("-c", "user.name=CI test", "-c", "user.email=ci@example.invalid",
                "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "base")
            base = git("rev-parse", "HEAD").decode().strip()
            Path(directory, "docs").mkdir()
            source.rename(Path(directory, "docs/example.rs"))
            git("add", "-A")
            git("-c", "user.name=CI test", "-c", "user.email=ci@example.invalid",
                "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "move")
            # Exercise the same diff command as CI in the temporary repository.
            previous = Path.cwd()
            try:
                os.chdir(directory)
                paths = changed_paths(base, "HEAD")
            finally:
                os.chdir(previous)
            self.assertIn("source.rs", paths)
            self.assertTrue(needs_rust(paths))


if __name__ == "__main__":
    unittest.main()
