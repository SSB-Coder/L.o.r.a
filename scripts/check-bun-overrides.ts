#!/usr/bin/env bun
/**
 * Checks every Bun override in the ledger (.github/dependency-overrides.yml) against package.json "overrides".
 *
 *   --ledger-only  fail unless overrides and ledger entries match one to one, with complete entries
 *   --open-prs     open an auto-merging PR for each override that is no longer needed, and an issue for each
 *                  override still needed after STALE_DAYS
 *
 *   NEEDED     without the override, a vulnerable version would be installed
 *   WAITING    safe to remove, but it would pull in a version younger than MIN_AGE_DAYS
 *   REMOVABLE  safe to remove now
 */
import { mkdtempSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { $ } from "bun"

type Entry = {
  package: string
  ecosystem: string
  override: string
  fixed_in: string
  advisory: string
  reason: string
  added: string
  owner: string
  parents?: string[]
}

const LEDGER = ".github/dependency-overrides.yml"
const MIN_AGE_DAYS = Number(process.env.MIN_AGE_DAYS ?? 14)
const STALE_DAYS = Number(process.env.STALE_DAYS ?? 90)
const DAY = 86_400_000

const readLedger = async (dir = ".") =>
  (
    (Bun.YAML.parse(await Bun.file(join(dir, LEDGER)).text()) as { overrides?: Entry[] }).overrides ?? []
  ).filter((e) => e.ecosystem === "bun")
const readPackage = async (dir = ".") =>
  (await Bun.file(join(dir, "package.json")).json()) as { overrides?: Record<string, string> }

const entries = await readLedger()
const overrides = (await readPackage()).overrides ?? {}

// Ledger check: the two lists match, and every entry is complete.
const ledgerNames = entries.map((e) => e.package).sort()
const overrideNames = Object.keys(overrides).sort()
if (JSON.stringify(ledgerNames) !== JSON.stringify(overrideNames)) {
  console.error(
    `::error::Overrides ${JSON.stringify(overrideNames)} and ledger entries ${JSON.stringify(ledgerNames)} differ`,
  )
  process.exit(1)
}
const incomplete = entries.filter((e) => !(e.fixed_in && e.advisory && e.reason && e.owner && e.added))
if (incomplete.length) {
  console.error(
    `::error::Every ledger entry needs fixed_in, advisory, reason, owner and added: ${incomplete.map((e) => e.package)}`,
  )
  process.exit(1)
}
if (process.argv.includes("--ledger-only")) process.exit(0)
const openPrs = process.argv.includes("--open-prs")

/** Versions of `pkg` in a bun.lock (JSONC: trailing commas). */
async function resolved(dir: string, pkg: string): Promise<string[]> {
  const text = (await Bun.file(join(dir, "bun.lock")).text()).replace(/,(\s*[}\]])/g, "$1")
  const lock = JSON.parse(text) as { packages: Record<string, [string, ...unknown[]]> }
  return [
    ...new Set(
      Object.values(lock.packages)
        .map(([spec]) => spec)
        .filter((s) => s.startsWith(`${pkg}@`))
        .map((s) => s.slice(pkg.length + 1)),
    ),
  ]
}

const atLeast = (version: string, fixed: string) => Bun.semver.satisfies(version, `>=${fixed}`)

async function check(entry: Entry, dir: string): Promise<"NEEDED" | "WAITING" | "REMOVABLE"> {
  const manifest = await readPackage(dir)
  delete manifest.overrides?.[entry.package]
  if (manifest.overrides && Object.keys(manifest.overrides).length === 0) delete manifest.overrides
  await Bun.write(join(dir, "package.json"), `${JSON.stringify(manifest, null, 2)}\n`)
  const ledger = Bun.YAML.parse(await Bun.file(join(dir, LEDGER)).text()) as { overrides: Entry[] }
  ledger.overrides = ledger.overrides.filter((e) => !(e.ecosystem === "bun" && e.package === entry.package))
  await Bun.write(join(dir, LEDGER), Bun.YAML.stringify(ledger, null, 2))
  // Ignore the repo's minimum release age here: we want to see what upstream would pick, then judge its age.
  await $`bun install --lockfile-only --ignore-scripts --minimum-release-age 0`.cwd(dir).quiet()

  const audit = await $`bun audit --json`.cwd(dir).nothrow().quiet()
  if (audit.stdout.toString().includes(`"${entry.package}"`)) return "NEEDED"
  const versions = await resolved(dir, entry.package)
  if (versions.some((v) => !atLeast(v, entry.fixed_in))) return "NEEDED"
  const times = (await $`npm view ${entry.package} time --json`.quiet().json()) as Record<string, string>
  const young = versions.some((v) => times[v] && Date.now() - Date.parse(times[v]) < MIN_AGE_DAYS * DAY)
  return young ? "WAITING" : "REMOVABLE"
}

async function openPr(entry: Entry, dir: string) {
  const branch = `deps/remove-override-${entry.package.replace(/@/g, "")}`
  if ((await $`git ls-remote --exit-code --heads origin ${branch}`.nothrow().quiet()).exitCode === 0) {
    console.log(`  ${branch} already exists, skipping`)
    return
  }
  const title = `chore(deps): remove override for ${entry.package}`
  await $`git switch --quiet -c ${branch}`.cwd(dir)
  await $`git commit --quiet -am ${title} -m ${`Without the override, every installed ${entry.package} is >= ${entry.fixed_in}, bun audit reports nothing for it, and every newly resolved version is at least ${MIN_AGE_DAYS} days old.`}`.cwd(
    dir,
  )
  await $`git push --quiet origin ${branch}`.cwd(dir)
  await $`gh pr create --head ${branch} --title ${title} --body ${`Opened by the dependency-overrides workflow. Upstream now resolves \`${entry.package}\` to a fixed version (>= ${entry.fixed_in}) on its own, so the override and its ledger entry are no longer needed.`}`.cwd(
    dir,
  )
  await $`gh pr merge --auto --squash ${branch}`.cwd(dir)
}

async function flagStale(entry: Entry) {
  const title = `Stale dependency override: ${entry.package}`
  const open =
    await $`gh issue list --state open --search ${`"${title}" in:title`} --json number -q .[].number`.text()
  if (open.trim()) return
  await $`gh issue create --title ${title} --body ${`The override for \`${entry.package}\` was added on ${entry.added} and is still needed. Owner: ${entry.owner}. Check whether ${(entry.parents ?? []).join(", ")} has a fix upstream, or replace it.`}`
}

for (const entry of entries) {
  const dir = mkdtempSync(join(tmpdir(), "overrides-"))
  await $`git worktree add --quiet --detach ${dir} HEAD`.quiet()
  try {
    const status = await check(entry, dir)
    console.log(`${entry.package}: ${status} (override added ${entry.added})`)
    if (openPrs && status === "REMOVABLE") await openPr(entry, dir)
    if (openPrs && status === "NEEDED" && Date.now() - Date.parse(entry.added) >= STALE_DAYS * DAY)
      await flagStale(entry)
  } finally {
    await $`git worktree remove --force ${dir}`.quiet()
  }
}
