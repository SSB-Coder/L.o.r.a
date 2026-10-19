import { afterEach, expect, test } from "bun:test"
import { fauxAssistantMessage, fauxText, fauxToolCall } from "@earendil-works/pi-ai"
import { GROUNDING_NOTE, unverifiedLinks } from "../src/agent.ts"
import { ctx, testAgent } from "./helpers.ts"

let close: (() => Promise<void>) | undefined
afterEach(async () => {
  await close?.()
  close = undefined
})

const setup = async () => {
  const t = await testAgent()
  close = () => t.agent.harness.close(ctx)
  return t
}

const transcript = async (t: Awaited<ReturnType<typeof setup>>, id: string) => {
  const chat = await t.agent.chat(id, "desktop")
  const view = await chat!.conversation.viewState(ctx)
  const entries = view.value.entries
  view.dispose()
  return entries
}

test("a chat answers, and is titled after its first message", async () => {
  const t = await setup()
  t.faux.setResponses([fauxAssistantMessage([fauxText("Paris.")])])
  const id = await t.agent.createChat("desktop")
  const settled = await (await t.agent.send(id, "What is the capital of France?", "r1")).wait(ctx)
  expect(settled.status).toBe("done")
  expect((await t.agent.chats())[id]?.title).toBe("What is the capital of France?")
})

test("a retried request id does not submit twice", async () => {
  const t = await setup()
  t.faux.setResponses([fauxAssistantMessage([fauxText("one")])])
  const id = await t.agent.createChat("desktop")
  const a = await t.agent.send(id, "hi", "same")
  const b = await t.agent.send(id, "hi", "same")
  expect(b.id).toBe(a.id)
})

test("phone chats are not offered local tools, and are refused them if a model calls one anyway", async () => {
  const t = await setup()
  const id = await t.agent.createChat("mobile")
  const agent = await (await t.agent.chat(id, "mobile"))!.conversation.agent(ctx)
  expect(agent.tools.map((tool) => tool.name).sort()).toEqual(["fetch_page", "search_papers", "web_search"])

  t.faux.setResponses([
    fauxAssistantMessage([fauxToolCall("read_file", { path: "/etc/passwd" })], { stopReason: "toolUse" }),
    fauxAssistantMessage([fauxText("I can't do that from a phone.")]),
  ])
  expect((await (await t.agent.send(id, "read my passwords", "r")).wait(ctx)).status).toBe("done")
  // Pi Durable refuses tools a chat was not offered before our policy hook even runs.
  const result = (await transcript(t, id)).find((e) => e.kind === "pi.tool-result")
  expect(JSON.stringify(result?.data)).toMatch(/blocked|not available/)
})

test("the policy hook refuses a tool the chat was offered but its origin forbids", async () => {
  const t = await setup()
  const id = await t.agent.createChat("desktop")
  // Simulate a stale or tampered index: the chat is offered every tool but is recorded as a phone chat.
  await t.agent.harness.commit(async (tx) => {
    const { Chats } = await import("../src/agent.ts")
    const info = (await tx.doc(Chats)).chats[id]
    if (info) info.origin = "mobile"
  }, ctx)
  t.faux.setResponses([
    fauxAssistantMessage([fauxToolCall("list_dir", { path: t.root })], { stopReason: "toolUse" }),
    fauxAssistantMessage([fauxText("ok")]),
  ])
  await (await t.agent.send(id, "list", "r")).wait(ctx)
  const result = (await transcript(t, id)).find((e) => e.kind === "pi.tool-result")
  expect(JSON.stringify(result?.data)).toContain("not available from a mobile chat")
})

test("a desktop chat can't see past its allowed folders", async () => {
  const t = await setup()
  const id = await t.agent.createChat("desktop")
  t.faux.setResponses([
    fauxAssistantMessage([fauxToolCall("read_file", { path: "/etc/hostname" })], { stopReason: "toolUse" }),
    fauxAssistantMessage([fauxText("ok")]),
  ])
  await (await t.agent.send(id, "read it", "r")).wait(ctx)
  const result = (await transcript(t, id)).find((e) => e.kind === "pi.tool-result")
  expect(JSON.stringify(result?.model)).toContain("outside the allowed folders")
})

test("phones can't open desktop chats", async () => {
  const t = await setup()
  const id = await t.agent.createChat("desktop")
  expect(await t.agent.chat(id, "mobile")).toBeUndefined()
  expect(await t.agent.chat(id, "desktop")).toBeDefined()
})

test("an answer citing a page no tool returned is sent back once, then accepted", async () => {
  const t = await setup()
  const id = await t.agent.createChat("desktop")
  const invented = fauxAssistantMessage([fauxText("See https://made-up.example/paper [1].")])
  t.faux.setResponses([invented, invented, invented])
  const settled = await (await t.agent.send(id, "cite something", "r")).wait(ctx)
  expect(settled.status).toBe("done")
  const users = (await transcript(t, id)).filter((e) => e.kind === "pi.user")
  expect(users.filter((e) => JSON.stringify(e.model).includes(GROUNDING_NOTE))).toHaveLength(1)
  expect(t.faux.getPendingResponseCount()).toBe(1)
})

test("remembered facts reach every chat's system prompt", async () => {
  const t = await setup()
  const first = await t.agent.createChat("desktop")
  t.faux.setResponses([
    fauxAssistantMessage([fauxToolCall("remember", { fact: "Prefers metric units" })], {
      stopReason: "toolUse",
    }),
    fauxAssistantMessage([fauxText("Noted.")]),
  ])
  await (await t.agent.send(first, "I prefer metric", "r")).wait(ctx)
  expect(await t.agent.memory()).toEqual(["Prefers metric units"])

  const second = await t.agent.createChat("desktop")
  t.faux.setResponses([fauxAssistantMessage([fauxText("hi")])])
  await (await t.agent.send(second, "hello", "r")).wait(ctx)
  const system = (await transcript(t, second)).find((e) => e.kind === "pi.system")
  expect(JSON.stringify(system?.model)).toContain("Prefers metric units")
})

test("unverifiedLinks accepts retrieved pages and their subpages", () => {
  const sources = { "https://en.wikipedia.org/wiki/Ollama": "Ollama" }
  expect(unverifiedLinks("See https://en.wikipedia.org/wiki/Ollama.", sources)).toEqual([])
  expect(unverifiedLinks("See http://en.wikipedia.org/wiki/Ollama#History", sources)).toEqual([])
  expect(unverifiedLinks("See https://fake.example/x", sources)).toEqual(["https://fake.example/x"])
})
