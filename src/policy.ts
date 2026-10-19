import { realpath } from "node:fs/promises"
import { isIP } from "node:net"
import { extname, isAbsolute, relative, resolve } from "node:path"

/** Where a chat was started. Phones get read-only research and nothing that touches this computer. */
export type Origin = "desktop" | "mobile"

/** 0: reads the public web. 1: reads or writes local files. 2: acts on this computer. */
export const TOOL_TIERS: Record<string, 0 | 1 | 2> = {
  web_search: 0,
  fetch_page: 0,
  search_papers: 0,
  list_dir: 1,
  read_file: 1,
  search_files: 1,
  save_note: 1,
  remember: 1,
  open: 2,
}

const MAX_TIER: Record<Origin, number> = { desktop: 2, mobile: 0 }

export const toolsFor = (origin: Origin) =>
  Object.keys(TOOL_TIERS).filter((name) => (TOOL_TIERS[name] ?? 99) <= MAX_TIER[origin])

/** Why `tool` may not run in a chat from `origin`, or undefined when it may. Unknown tools never run. */
export function blockReason(tool: string, origin: Origin): string | undefined {
  const tier = TOOL_TIERS[tool]
  if (tier === undefined) return `"${tool}" is not an allowed tool`
  if (tier > MAX_TIER[origin]) return `"${tool}" is not available from a ${origin} chat`
  return undefined
}

/** Never launched, even inside an allowed folder. */
export const BLOCKED_EXTENSIONS = new Set(
  ".exe .bat .cmd .com .ps1 .psm1 .vbs .vbe .js .jse .wsf .wsh .reg .dll .sys .scr .cpl .msi .msp .hta .pif .lnk .docm .xlsm .pptm .sh .app .command".split(
    " ",
  ),
)

export type PathCheck = { ok: true; path: string } | { ok: false; reason: string }

/** Resolve `input` and require it to sit inside one of `roots`, after following symlinks. */
export async function checkPath(input: string, roots: readonly string[]): Promise<PathCheck> {
  if (input.includes("\0")) return { ok: false, reason: "invalid path" }
  if (/^(\\\\|\/\/)/.test(input)) return { ok: false, reason: "network paths are not allowed" }
  if (/^[a-z][a-z0-9+.-]+:\/\//i.test(input)) return { ok: false, reason: "URLs are not file paths" }
  // NTFS alternate data streams: a colon anywhere but after the drive letter.
  if (input.slice(/^[a-z]:/i.test(input) ? 2 : 0).includes(":")) return { ok: false, reason: "invalid path" }
  const realRoots = await Promise.all(roots.map((root) => realpath(root).catch(() => resolve(root))))
  const absolute = isAbsolute(input) ? resolve(input) : resolve(realRoots[0] ?? ".", input)
  const real = await realpath(absolute).catch(() => absolute)
  const inside = realRoots.some((root) => {
    const rel = relative(fold(root), fold(real))
    return rel === "" || (!rel.startsWith("..") && !isAbsolute(rel))
  })
  return inside
    ? { ok: true, path: real }
    : { ok: false, reason: `outside the allowed folders: ${roots.join(", ")}` }
}

export const isBlockedFile = (path: string) => BLOCKED_EXTENSIONS.has(extname(path).toLowerCase())

const fold = (path: string) => (process.platform === "win32" ? path.toLowerCase() : path)

/** Loopback, private, link-local, CGNAT, and other addresses a web fetch must never reach. */
export function isPrivateAddress(address: string): boolean {
  const ip = address.replace(/^\[|\]$/g, "").toLowerCase()
  if (isIP(ip) === 4) {
    const [a = 0, b = 0] = ip.split(".").map(Number)
    return (
      a === 0 ||
      a === 10 ||
      a === 127 ||
      (a === 100 && b >= 64 && b <= 127) ||
      (a === 169 && b === 254) ||
      (a === 172 && b >= 16 && b <= 31) ||
      (a === 192 && b === 168) ||
      a >= 224
    )
  }
  if (isIP(ip) === 6) {
    const mapped = ip.match(/^::ffff:(\d+\.\d+\.\d+\.\d+)$/)
    if (mapped?.[1]) return isPrivateAddress(mapped[1])
    return ip === "::" || ip === "::1" || /^(fc|fd|fe[89ab])/.test(ip)
  }
  return true
}
