# Releases

LORA ships on two channels. Each comes from its own kind of branch, so a stable line can take fixes without
taking everything that has landed on `master` since.

| Channel | Branch | Version | GitHub release |
| --- | --- | --- | --- |
| Nightly | `master` | `<package.json version>-nightly.<CI run>` | Prerelease, e.g. `v0.3.0-nightly.57`, one per push |
| Stable | `release/X.Y` | `package.json` version, always `X.Y.Z` | `vX.Y.Z`, published when that version first lands |

`package.json` is the only place the version is written. CI adds the nightly suffix and never edits the file.
You can always see which version you have: in the app sidebar, with `lora --version`, in the `VERSION` file in
each download, and in the release title.

Published releases are permanent. Never move or delete a tag or asset; ship a fix as a new patch version.

## What CI does

`.github/workflows/release.yml` runs on every push to `master` and `release/**`, and on pull requests:

1. Works out the version and channel. On `release/X.Y` it fails if `package.json` isn't `X.Y.*`.
2. Builds each target on its own OS (Linux x64 and arm64, macOS arm64 and x64, Windows x64), runs the binary,
   checks `--version` and the `VERSION` file, starts it, and checks that the UI and API answer.
3. On pushes only: uploads every archive plus `SHA256SUMS` as a GitHub release. Pull requests publish nothing.
   If the stable tag already exists, it skips publishing and says why.

## Cut a stable release

1. Check that `package.json` on `master` is `X.Y.0`.
2. `git switch -c release/X.Y master && git push -u origin release/X.Y`. CI publishes `vX.Y.0`.
3. Open a PR on `master` that bumps `package.json` to `X.(Y+1).0`, so nightlies sort after the new stable.

## Fix a released version

1. `git switch -c fix/<topic> origin/release/X.Y`
2. Make the fix and bump `package.json` to `X.Y.(Z+1)` in the same PR.
3. Open the PR against `release/X.Y`. Merging publishes `vX.Y.(Z+1)`. A push that leaves the version unchanged
   builds but doesn't publish.
4. If `master` has the same bug, cherry-pick the fix onto `master` in a separate PR. Never merge a release branch
   back into `master`.

## Install

```sh
# macOS / Linux: stable (default), nightly, or a tag
curl -fsSL https://raw.githubusercontent.com/SSB-Coder/L.o.r.a/master/scripts/install.sh | sh
curl -fsSL https://raw.githubusercontent.com/SSB-Coder/L.o.r.a/master/scripts/install.sh | sh -s -- nightly
```

```powershell
# Windows
irm https://raw.githubusercontent.com/SSB-Coder/L.o.r.a/master/scripts/install.ps1 | iex
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/SSB-Coder/L.o.r.a/master/scripts/install.ps1))) -Channel nightly
```

`stable` is GitHub's latest release, which never includes prereleases. `nightly` is the newest prerelease. Both
installers check the archive against `SHA256SUMS` before installing.

Builds are not notarized. On macOS, if Gatekeeper blocks a downloaded binary, run
`xattr -d com.apple.quarantine ~/.lora/bin/lora` (the install script's `curl` download isn't quarantined).
