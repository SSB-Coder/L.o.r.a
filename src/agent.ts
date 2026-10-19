import { mkdirSync } from "node:fs"
import { dirname } from "node:path"
import { BACKGROUND_CONTEXT as ctx } from "@earendil-works/chord/context"
import type { Models } from "@earendil-works/pi-ai"
import {
  type ConversationId,
  createRegistry,
  defineDoc,
  defineExtension,
  GenerationTask,
  Harness,
  type HookApi,
  hook,
  section,
  ToolTask,
} from "@earendil-works/pi-durable"
import { openNodeSqliteStorage } from "@earendil-works/pi-durable/storage/sqlite/node"
import { Context, Effect, Layer } from "effect"
import type { Config } from "./config.ts"
import { PROVIDER } from "./models.ts"
import { blockReason, type Origin, toolsFor } from "./policy.ts"
import { createTools, Memory, normalizeUrl, Sources } from "./tools.ts"

export type ChatInfo = { title: string; origin: Origin; created: number }

/** Index of chats. Conversations themselves live in Pi Durable; this only names them and records their origin. */
export const Chats = defineDoc<{ chats: Record<string, ChatInfo> }>({
  kind: "lora.chats",
  version: 1,
  scope: "session",
  initial: () => ({ chats: {} }),
})

const PREAMBLE = `You are LORA, a local research assistant running on the user's own computer.
- Answer directly and briefly. Use Markdown.
- For anything factual you are not certain of, research it: web_search, then fetch_page the best results.
- Cite the sources you used inline as [n] and list them at the end as "[n] Title — URL".
- Only cite URLs that a tool returned in this chat. Never invent links, numbers, or quotes.
- If sources disagree or evidence is weak, say so.
- Speak in plain sentences without tables when the user is likely listening (short questions).`

export const GROUNDING_NOTE = "[grounding check]"

/** Links in `text` that no tool returned in this chat. */
export function unverifiedLinks(text: string, sources: Record<string, string>): string[] {
  const cited = new Set(
    (text.match(/https?:\/\/[^\s<>)\]"'`]+/g) ?? []).map((u) => u.replace(/[.,;:!?]+$/, "")),
  )
  const known = Object.keys(sources)
  return [...cited].filter((url) => {
    const n = normalizeUrl(url)
    return !known.some((k) => k === n || n.startsWith(`${k}/`) || k.startsWith(`${n}/`))
  })
}

const chatOrigin = async (api: Pick<HookApi, "snapshot">, id: ConversationId): Promise<Origin> => {
  const index = await api.snapshot(Chats, ctx)
  // A conversation missing from the index (a subagent, or one made by hand) gets the narrower permissions.
  return index?.chats[id]?.origin ?? "mobile"
}

export function loraExtension(config: Config) {
  const tools = createTools(config)
  const extension = defineExtension({
    name: "lora",
    tools: [...tools],
    sections: [
      section("preamble", () => PREAMBLE, { tag: false }),
      section("today", () => `Today is ${new Date().toDateString()}. The computer runs ${process.platform}.`),
      section("memory", async (input, context) => {
        const memory = await input.read.snapshot(Memory, context)
        return memory?.facts.length ? memory.facts.map((f) => `- ${f}`).join("\n") : undefined
      }),
    ],
    hooks: [
      // Tools a chat is not offered are also refused here, in case a model calls one anyway.
      hook(ToolTask, {
        beforeTool: async (call, api) => {
          const reason = blockReason(call.name, await chatOrigin(api, api.conversationId))
          return reason ? { block: reason } : undefined
        },
      }),
      // Ask once for a rewrite when the answer links to pages no tool returned. Each `continue` starts a new
      // generation task, so the generation answering a grounding note marks itself and is not asked again.
      hook(GenerationTask, {
        beforeRequest: async ({ messages }, api, context) => {
          const last = messages.findLast((m) => m.role === "user")
          if (last && JSON.stringify(last.content).includes(GROUNDING_NOTE))
            await api.memo("regrounded", true, context)
          return undefined
        },
        onYield: async (answer, api, context) => {
          if (await api.memo<boolean>("regrounded", context)) return undefined
          const body = answer.content.flatMap((b) => (b.type === "text" ? [b.text] : [])).join("\n")
          const sources = (await api.snapshot(Sources, api.conversationId, context))?.urls ?? {}
          const missing = unverifiedLinks(body, sources)
          if (missing.length === 0) return undefined
          return {
            continue: `${GROUNDING_NOTE} Your answer links to pages you never retrieved: ${missing.join(", ")}. Fetch them, or rewrite the answer without them. Reply with the full corrected answer only.`,
          }
        },
      }),
    ],
  })
  return { extension, tools }
}

/** The durable harness and the operations the server needs on chats. */
export class Agent extends Context.Service<Agent, Awaited<ReturnType<typeof makeAgent>>>()("lora/Agent") {
  static layer = (config: Config, models: Models, database: string, defaultModel: () => string) =>
    Layer.effect(
      Agent,
      Effect.acquireRelease(
        Effect.promise(async () => {
          mkdirSync(dirname(database), { recursive: true })
          return makeAgent(config, models, await openNodeSqliteStorage(database), defaultModel)
        }),
        (agent) => Effect.promise(() => agent.harness.close(ctx)),
      ),
    )
}

export async function makeAgent(
  config: Config,
  models: Models,
  storage: Parameters<typeof Harness.open>[0],
  defaultModel: () => string,
) {
  const { extension, tools } = loraExtension(config)
  const registry = createRegistry()
  registry.install(extension)
  const window = config.ollama.contextWindow
  const harness = await Harness.open(
    storage,
    {
      models,
      registry,
      settings: {
        stream: { timeoutMs: 600_000 },
        retry: { maxRetries: 2 },
        // Local models have small windows; keep compaction proportional instead of using cloud-sized defaults.
        compaction: {
          reserveTokens: Math.floor(window * 0.25),
          keepRecentTokens: Math.floor(window * 0.35),
          backgroundTokens: Math.floor(window * 0.1),
        },
        toolExecution: "parallel",
      },
      onReport: (error) => console.error("[lora]", error),
    },
    ctx,
  )
  harness.resume()

  const chats = async () => (await harness.snapshot(Chats, ctx))?.chats ?? {}
  // Pi Durable ids are numbers; the API and the chat index use their string form.
  const conversation = (id: string) => harness.conversation(Number(id) as unknown as ConversationId, ctx)

  return {
    harness,
    chats,
    async createChat(origin: Origin, model = defaultModel()) {
      const conversation = await harness.createConversation(
        {
          ownership: { kind: "ownerless" },
          agent: {
            model: { provider: PROVIDER, modelId: model },
            tools: tools.filter((t) => toolsFor(origin).includes(t.name)),
          },
          init: async (tx, id) => {
            ;(await tx.doc(Chats)).chats[id] = { title: "New chat", origin, created: Date.now() }
          },
        },
        ctx,
      )
      return String(conversation.id)
    },
    /** The chat when `origin` may see it: desktop sees every chat, a phone only phone chats. */
    async chat(id: string, origin: Origin) {
      const info = (await chats())[id]
      if (!info || (origin === "mobile" && info.origin !== "mobile")) return undefined
      const handle = await conversation(id)
      return handle && { conversation: handle, info }
    },
    async send(id: string, text: string, requestId: string) {
      const handle = await conversation(id)
      if (!handle) throw new Error("no such chat")
      await harness.commit(async (tx) => {
        const info = (await tx.doc(Chats)).chats[id]
        if (info?.title === "New chat") info.title = text.replace(/\s+/g, " ").trim().slice(0, 60)
      }, ctx)
      return handle.submit({ type: "input", content: text, requestId }, ctx)
    },
    /** Hide a chat from the list. Its transcript stays in storage. */
    async hide(id: string) {
      await harness.commit(async (tx) => {
        delete (await tx.doc(Chats)).chats[id]
      }, ctx)
    },
    async setModel(id: string, model: string) {
      await (await conversation(id))?.configure({ model: { provider: PROVIDER, modelId: model } }, ctx)
    },
    memory: async () => (await harness.snapshot(Memory, ctx))?.facts ?? [],
    async forget(fact: string) {
      await harness.commit(async (tx) => {
        const memory = await tx.doc(Memory)
        memory.facts = memory.facts.filter((f) => f !== fact)
      }, ctx)
    },
  }
}
