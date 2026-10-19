#!/usr/bin/env bun
/**
 * Build single-file LORA binaries (UI embedded) and release archives.
 *
 *   bun scripts/build.ts                 # every target
 *   bun scripts/build.ts linux-x64       # some targets
 *   LORA_VERSION=0.2.0-nightly.7 bun scripts/build.ts
 *
 * Writes dist/lora-<version>-<target>.(tar.gz|zip), each holding the binary, VERSION, README, and an example
 * config, plus dist/SHA256SUMS.
 */
import { createHash } from "node:crypto"
import { cp, mkdir, rm } from "node:fs/promises"
import { join } from "node:path"
import { $ } from "bun"
import pkg from "../package.json" with { type: "json" }

const TARGETS = ["linux-x64", "linux-arm64", "darwin-arm64", "darwin-x64", "windows-x64"] as const
const version = process.env.LORA_VERSION ?? pkg.version
const wanted = process.argv.slice(2)
const targets = wanted.length ? TARGETS.filter((t) => wanted.includes(t)) : TARGETS
if (wanted.length && targets.length !== wanted.length)
  throw new Error(`unknown target in ${wanted}; known: ${TARGETS}`)

await rm("dist", { recursive: true, force: true })
await mkdir("dist")

const archives: string[] = []
for (const target of targets) {
  const name = `lora-${version}-${target}`
  const dir = join("dist", name)
  const exe = target.startsWith("windows") ? "lora.exe" : "lora"
  console.log(`building ${name}`)
  await $`bun build src/main.ts --compile --minify --target=bun-${target} --define process.env.LORA_VERSION=${JSON.stringify(version)} --outfile ${join(dir, exe)}`.quiet()
  // Apple silicon refuses unsigned binaries; an ad-hoc signature is enough to run locally built ones.
  if (target.startsWith("darwin") && process.platform === "darwin")
    await $`codesign --force --sign - ${join(dir, exe)}`.quiet()
  await Bun.write(join(dir, "VERSION"), `${version}\n`)
  await cp("README.md", join(dir, "README.md"))
  await cp("lora.example.yaml", join(dir, "lora.example.yaml"))
  const archive = target.startsWith("windows") ? `${name}.zip` : `${name}.tar.gz`
  // Windows' bundled bsdtar writes zips with -a; elsewhere use zip.
  if (archive.endsWith(".zip") && process.platform === "win32")
    await $`tar -a -cf ${archive} ${name}`.cwd("dist")
  else if (archive.endsWith(".zip")) await $`zip -qr ${archive} ${name}`.cwd("dist")
  else await $`tar -czf ${archive} ${name}`.cwd("dist")
  archives.push(archive)
}

const sums = await Promise.all(
  archives.map(async (archive) => {
    const hash = createHash("sha256").update(await Bun.file(join("dist", archive)).bytes())
    return `${hash.digest("hex")}  ${archive}`
  }),
)
await Bun.write("dist/SHA256SUMS", `${sums.join("\n")}\n`)
console.log(`dist/: ${archives.join(", ")}, SHA256SUMS`)
