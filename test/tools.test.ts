import { expect, test } from "bun:test"
import { decodeConfig } from "../src/config.ts"
import { opener, resolveOpen } from "../src/tools.ts"

const config = decodeConfig({})
const deny = async () => {
  throw new Error("outside the allowed folders")
}

test("open resolves apps, site searches, and links", async () => {
  expect(await resolveOpen("calc", undefined, config, deny)).toMatchObject({
    kind: "app",
    label: "calculator",
  })
  expect(await resolveOpen("wiki", "durable execution", config, deny)).toMatchObject({
    kind: "url",
    value: "https://en.wikipedia.org/wiki/Special:Search?search=durable%20execution",
  })
  expect(await resolveOpen("https://example.com/a?b=1&c=2", undefined, config, deny)).toMatchObject({
    kind: "url",
  })
})

test("open refuses other schemes, programs, and paths outside the allowed folders", async () => {
  await expect(resolveOpen("javascript:alert(1)", undefined, config, deny)).rejects.toThrow(
    "only http and https",
  )
  await expect(resolveOpen("file:///etc/passwd", undefined, config, deny)).rejects.toThrow(
    "only http and https",
  )
  await expect(resolveOpen("/etc/passwd", undefined, config, deny)).rejects.toThrow("outside")
  await expect(resolveOpen("/home/me/run.exe", undefined, config, async (p) => p)).rejects.toThrow(
    "can't be opened",
  )
})

test("on Windows, links never pass through cmd, where & would start a new command", () => {
  const url = { kind: "url" as const, value: "https://x.example/?a=1&calc.exe", label: "" }
  expect(opener(url, "win32")).toEqual(["rundll32", "url.dll,FileProtocolHandler", url.value])
  expect(opener({ kind: "app", value: "notepad.exe", label: "" }, "win32")).toEqual([
    "cmd",
    "/c",
    "start",
    "",
    "notepad.exe",
  ])
})

test("config rejects binding the phone listener to every interface", () => {
  expect(() => decodeConfig({ server: { mobile: { host: "0.0.0.0" } } })).toThrow("every interface")
  expect(decodeConfig({ server: { mobile: { host: "192.168.50.10" } } }).server.mobile?.port).toBe(8765)
})
