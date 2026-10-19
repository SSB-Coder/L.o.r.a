# Dependencies

| Kind of change | When it lands | Who does it |
| --- | --- | --- |
| Security fix (a Dependabot alert has a fixed version) | Right away; the PR merges itself once required checks pass | Dependabot security updates + `.github/workflows/dependabot-security.yml` |
| New release of a dependency | 14 days after it was published, never sooner | Dependabot version updates (`cooldown: 14`) |
| Override pinning a transitive dependency | By hand, only when nothing else fixes an alert; removed automatically | `.github/dependency-overrides.yml` + `.github/workflows/dependency-overrides.yml` |

`bunfig.toml` sets `minimumReleaseAge = 1209600` (14 days), so `bun install` and `bun add` never resolve a
version younger than that, transitive packages included. A security fix that needs a newer version can pass
`--minimum-release-age 0` for that one install; name the advisory in the PR title.

Security fixes that bump a major version open right away but don't auto-merge. They're labelled
`security-major` for same-day review.

## Packages that move together

- `effect` and every `@effect/*` package must be on matching versions. `@effect/platform-node-shared` is a direct
  dependency only to pin it to the same version as `effect`; without the pin, Bun can resolve a newer one that
  calls APIs the installed `effect` lacks.
- `@earendil-works/pi-durable`, `pi-ai`, and `chord` are experimental and pinned exactly. Upgrade them together and
  read the changelog first; their APIs change between releases.

Dependabot groups both sets.

## Overrides

An override is a last resort for an alert. Add it to `package.json` `"overrides"` with a range that starts at
the fixed version and stays on its major line (never `*` or `latest`), regenerate the lockfile, run the tests,
and add a ledger entry in the same commit:

```yaml
# .github/dependency-overrides.yml
overrides:
  - package: tar-lite            # exactly as written in package.json
    ecosystem: bun
    manifest: package.json
    override: "^3.1.4"
    fixed_in: 3.1.4
    advisory: GHSA-xxxx-xxxx-xxxx
    parents: [bundler-kit]
    reason: bundler-kit@2.4.0 pins tar-lite ~3.0.0, which has a path traversal bug fixed in 3.1.4.
    added: 2026-10-19
    owner: "@SSB-Coder"
```

Pull requests fail if overrides and ledger entries don't match. Every day, and after each lockfile change on
`master`, `scripts/check-bun-overrides.ts` tries removing each override in a scratch worktree. When upstream
resolves to a fixed version on its own (and nothing newly resolved is under 14 days old) it opens an auto-merging
PR that removes it. An override still needed after 90 days gets an issue. Nobody removes an override by hand.

## One-time repository setup

These are GitHub settings, not files, and need a repository admin:

```sh
repo=SSB-Coder/L.o.r.a
gh api -X PUT "repos/$repo/vulnerability-alerts"       # Dependabot alerts
gh api -X PUT "repos/$repo/automated-security-fixes"   # Dependabot security updates
gh repo edit "$repo" --enable-auto-merge --delete-branch-on-merge
```

- Protect `master` with a ruleset that requires the CI checks, so auto-merge can't merge untested code.
- Create a fine-grained token with **Dependabot alerts: read** and store it as a *Dependabot* secret:
  `gh secret set DEPENDABOT_ALERTS_TOKEN --app dependabot --repo $repo`.
- Create a fine-grained token with **Contents**, **Pull requests**, and **Issues** read/write and store it as an
  Actions secret named `DEPS_BOT_TOKEN`. PRs opened with `GITHUB_TOKEN` don't trigger CI.
