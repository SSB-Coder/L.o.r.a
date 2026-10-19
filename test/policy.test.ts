import { expect, test } from "bun:test"
import { mkdirSync, mkdtempSync, symlinkSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { blockReason, checkPath, isPrivateAddress, toolsFor } from "../src/policy.ts"

test("phones get only public-web tools; unknown tools never run", () => {
  expect(toolsFor("mobile").sort()).toEqual(["fetch_page", "search_papers", "web_search"])
  expect(blockReason("open", "mobile")).toContain("not available")
  expect(blockReason("open", "desktop")).toBeUndefined()
  expect(blockReason("bash", "desktop")).toContain("not an allowed tool")
})

// Creating symlinks needs extra privileges on Windows.
test.skipIf(process.platform === "win32")(
  "paths must stay inside the allowed folders, symlinks included",
  async () => {
    const base = mkdtempSync(join(tmpdir(), "lora-policy-"))
    const root = join(base, "docs")
    mkdirSync(root)
    writeFileSync(join(root, "a.txt"), "hi")
    symlinkSync("/etc", join(root, "escape"))
    expect(await checkPath(join(root, "a.txt"), [root])).toMatchObject({ ok: true })
    expect(await checkPath("a.txt", [root])).toMatchObject({ ok: true })
    expect(await checkPath(join(root, "../outside"), [root])).toMatchObject({ ok: false })
    expect(await checkPath(join(root, "escape/hostname"), [root])).toMatchObject({ ok: false })
    expect(await checkPath("\\\\server\\share\\x", [root])).toMatchObject({ ok: false })
    expect(await checkPath("C:\\docs\\a.txt:hidden", [root])).toMatchObject({ ok: false })
    expect(await checkPath("file:///etc/passwd", [root])).toMatchObject({ ok: false })
  },
)

test("private, loopback, and metadata addresses are private", () => {
  for (const ip of [
    "127.0.0.1",
    "10.1.2.3",
    "192.168.50.10",
    "172.20.0.1",
    "169.254.169.254",
    "100.64.0.1",
    "::1",
    "fd00::1",
    "::ffff:127.0.0.1",
    "0.0.0.0",
  ]) {
    expect(isPrivateAddress(ip)).toBe(true)
  }
  for (const ip of ["93.184.216.34", "2606:4700::1111", "8.8.8.8"]) expect(isPrivateAddress(ip)).toBe(false)
})
