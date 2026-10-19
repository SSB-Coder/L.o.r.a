import { randomBytes, timingSafeEqual } from "node:crypto"
import { chmod } from "node:fs/promises"
import { join } from "node:path"

export type Tokens = { desktop: string; mobile: string }

/** Load the access tokens, creating them on first run. They persist so a paired phone stays paired. */
export async function loadTokens(dir: string): Promise<Tokens> {
  const file = Bun.file(join(dir, "tokens.json"))
  if (await file.exists()) {
    const tokens = (await file.json()) as Partial<Tokens>
    if (tokens.desktop && tokens.mobile) return tokens as Tokens
  }
  const tokens = { desktop: newToken(), mobile: newToken() }
  await Bun.write(file, JSON.stringify(tokens))
  await chmod(file.name ?? "", 0o600).catch(() => {})
  return tokens
}

const newToken = () => randomBytes(32).toString("base64url")

export function sameToken(given: string | null | undefined, expected: string): boolean {
  if (!given) return false
  const a = Buffer.from(given)
  const b = Buffer.from(expected)
  return a.length === b.length && timingSafeEqual(a, b)
}

/** Bans an address for `banMs` after `limit` failed logins within `windowMs`. */
export class LoginLimiter {
  #failures = new Map<string, number[]>()
  #bans = new Map<string, number>()
  constructor(
    readonly limit = 5,
    readonly windowMs = 60_000,
    readonly banMs = 15 * 60_000,
    readonly now = () => Date.now(),
  ) {}

  banned(ip: string): boolean {
    const until = this.#bans.get(ip)
    if (until === undefined) return false
    if (until > this.now()) return true
    this.#bans.delete(ip)
    return false
  }

  fail(ip: string) {
    const now = this.now()
    const recent = (this.#failures.get(ip) ?? []).filter((t) => t > now - this.windowMs)
    recent.push(now)
    if (recent.length >= this.limit) {
      this.#bans.set(ip, now + this.banMs)
      this.#failures.delete(ip)
    } else this.#failures.set(ip, recent)
    if (this.#failures.size > 1000) this.#failures.clear()
  }

  succeed(ip: string) {
    this.#failures.delete(ip)
  }
}

/**
 * Requests must name this listener in `Host` (stops DNS rebinding), and a browser `Origin`, when sent, must be this
 * listener too (stops other sites driving the API with the user's cookie).
 */
export function sameSite(request: Request, hosts: readonly string[]): boolean {
  const host = request.headers.get("host")?.toLowerCase()
  if (!host || !hosts.includes(host)) return false
  const origin = request.headers.get("origin")
  if (!origin) return true
  try {
    return hosts.includes(new URL(origin).host.toLowerCase())
  } catch {
    return false
  }
}

export const COOKIE = "lora_token"

export function cookieToken(request: Request): string | undefined {
  const header = request.headers.get("cookie") ?? ""
  return header.match(new RegExp(`(?:^|;\\s*)${COOKIE}=([^;]+)`))?.[1]
}
