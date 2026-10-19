import { afterEach, expect, test } from "bun:test"
import { fauxAssistantMessage, fauxText } from "@earendil-works/pi-ai"
import { Effect } from "effect"
import { LoginLimiter } from "../src/auth.ts"
import { createHandler, type Listener } from "../src/server.ts"
import { ctx, testAgent } from "./helpers.ts"

const tokens = { desktop: "desk-token", mobile: "phone-token" }
const desktop: Listener = { origin: "desktop", hosts: ["127.0.0.1:8765"], token: tokens.desktop }
const phone: Listener = { origin: "mobile", hosts: ["192.168.50.10:8765"], token: tokens.mobile }

let close: (() => Promise<void>) | undefined
afterEach(async () => {
  await close?.()
  close = undefined
})

async function setup() {
  const t = await testAgent()
  close = () => t.agent.harness.close(ctx)
  const ollama = { list: () => ["faux"], preferred: () => "faux", refresh: () => Effect.succeed(["faux"]) }
  const limiter = new LoginLimiter(3)
  const handle = createHandler({
    agent: t.agent,
    ollama: ollama as never,
    config: t.config,
    tokens,
    version: "test",
    limiter,
  })
  const call = (
    listener: Listener,
    path: string,
    init: RequestInit & { token?: string | null; host?: string } = {},
  ) => {
    const headers = new Headers(init.headers)
    headers.set("host", init.host ?? listener.hosts[0]!)
    const token = init.token === undefined ? listener.token : init.token
    if (token) headers.set("authorization", `Bearer ${token}`)
    if (init.body) headers.set("content-type", "application/json")
    return handle(
      new Request(`http://${listener.hosts[0]}${path}`, { ...init, headers }),
      listener,
      "10.0.0.2",
    )
  }
  return { ...t, call }
}

test("the API needs the listener's own token", async () => {
  const s = await setup()
  expect((await s.call(desktop, "/api/state", { token: null })).status).toBe(401)
  expect((await s.call(desktop, "/api/state", { token: tokens.mobile })).status).toBe(401)
  expect((await s.call(desktop, "/api/state")).status).toBe(200)
})

test("a pairing link sets an HttpOnly cookie that signs the browser in", async () => {
  const s = await setup()
  const response = await s.call(phone, `/auth?token=${tokens.mobile}`, { token: null })
  expect(response.status).toBe(302)
  const cookie = response.headers.get("set-cookie") ?? ""
  expect(cookie).toContain("HttpOnly")
  expect(cookie).toContain("SameSite=Strict")
  const state = await s.call(phone, "/api/state", { token: null, headers: { cookie: cookie.split(";")[0]! } })
  expect(state.status).toBe(200)
})

test("repeated bad logins ban the address", async () => {
  const s = await setup()
  for (let i = 0; i < 3; i++) await s.call(phone, "/auth?token=wrong", { token: null })
  expect((await s.call(phone, `/auth?token=${tokens.mobile}`, { token: null })).status).toBe(429)
})

test("requests naming another host, or sent from another site, are refused", async () => {
  const s = await setup()
  expect((await s.call(desktop, "/api/state", { host: "evil.example:8765" })).status).toBe(403)
  expect((await s.call(desktop, "/api/state", { headers: { origin: "https://evil.example" } })).status).toBe(
    403,
  )
  expect((await s.call(desktop, "/api/state", { headers: { origin: "http://127.0.0.1:8765" } })).status).toBe(
    200,
  )
})

test("phones see only phone chats and no desktop details", async () => {
  const s = await setup()
  const { id: desk } = await (await s.call(desktop, "/api/chats", { method: "POST", body: "{}" })).json()
  const { id: mine } = await (await s.call(phone, "/api/chats", { method: "POST", body: "{}" })).json()
  const state = await (await s.call(phone, "/api/state")).json()
  expect(state.chats.map((c: { id: string }) => c.id)).toEqual([mine])
  expect(state.pairing).toBeUndefined()
  expect(state.memory).toEqual([])
  expect((await s.call(phone, `/api/chats/${desk}/events`)).status).toBe(404)
  expect((await s.call(phone, "/api/memory", { method: "DELETE", body: '{"fact":"x"}' })).status).toBe(403)
  const all = await (await s.call(desktop, "/api/state")).json()
  expect(all.chats).toHaveLength(2)
})

test("a message streams its answer over server-sent events", async () => {
  const s = await setup()
  s.faux.setResponses([fauxAssistantMessage([fauxText("Hello from LORA")])])
  const { id } = await (await s.call(desktop, "/api/chats", { method: "POST", body: "{}" })).json()
  const events = await s.call(desktop, `/api/chats/${id}/events`)
  expect(events.headers.get("content-type")).toBe("text/event-stream")
  const sent = await s.call(desktop, `/api/chats/${id}/messages`, {
    method: "POST",
    body: JSON.stringify({ text: "hi", requestId: "1" }),
  })
  expect(sent.status).toBe(200)

  const reader = events.body!.getReader()
  let text = ""
  while (!text.includes('"type":"run_end"')) text += new TextDecoder().decode((await reader.read()).value)
  await reader.cancel()
  expect(text).toContain('"type":"snapshot"')
  expect(text).toContain("Hello from LORA")
})

test("invalid bodies and unknown models are rejected", async () => {
  const s = await setup()
  expect((await s.call(desktop, "/api/chats", { method: "POST", body: '{"model":"nope"}' })).status).toBe(400)
  const { id } = await (await s.call(desktop, "/api/chats", { method: "POST", body: "{}" })).json()
  expect(
    (await s.call(desktop, `/api/chats/${id}/messages`, { method: "POST", body: '{"text":""}' })).status,
  ).toBe(400)
})

test("hiding a chat removes it from the list, and a phone can't hide desktop chats", async () => {
  const s = await setup()
  const { id } = await (await s.call(desktop, "/api/chats", { method: "POST", body: "{}" })).json()
  expect((await s.call(phone, `/api/chats/${id}`, { method: "DELETE" })).status).toBe(404)
  expect((await s.call(desktop, `/api/chats/${id}`, { method: "DELETE" })).status).toBe(200)
  expect((await (await s.call(desktop, "/api/state")).json()).chats).toEqual([])
})
