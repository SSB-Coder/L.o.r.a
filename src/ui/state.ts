import type { AgentEvent } from "@earendil-works/pi-durable"
import { Effect, Queue, Schema, Stream } from "effect"
import { Atom } from "effect/reactivity"
import { type ChatState, emptyChat, reduce } from "./chat.ts"

export type ChatSummary = { id: string; title: string; origin: "desktop" | "mobile"; created: number }
export type AppState = {
  version: string
  origin: "desktop" | "mobile"
  models: string[]
  defaultModel: string
  chats: ChatSummary[]
  memory: string[]
  voice: { stt: boolean; tts: boolean }
  pairing?: string
}

export class ApiError extends Schema.TaggedError<ApiError>()("ApiError", {
  status: Schema.Number,
  message: Schema.String,
}) {}

export const api = Effect.fn("api")(function* <A = { ok: true }>(
  path: string,
  init?: { method?: string; body?: unknown },
) {
  const response = yield* Effect.tryPromise({
    try: (signal) =>
      fetch(`/api/${path}`, {
        method: init?.method ?? (init?.body === undefined ? "GET" : "POST"),
        headers: init?.body === undefined ? {} : { "content-type": "application/json" },
        body: init?.body === undefined ? undefined : JSON.stringify(init.body),
        signal,
      }),
    catch: () => new ApiError({ status: 0, message: "LORA is not reachable" }),
  })
  const data = yield* Effect.promise(() => response.json().catch(() => ({})))
  if (!response.ok)
    return yield* new ApiError({ status: response.status, message: data.error ?? response.statusText })
  return data as A
})

/** Run an API call from an event handler. */
export const call = <A = { ok: true }>(path: string, init?: { method?: string; body?: unknown }) =>
  Effect.runPromise(api<A>(path, init))

export const appState = Atom.make(api<AppState>("state")).pipe(Atom.keepAlive)

export const selectedChat = Atom.make<string | undefined>(undefined)

/** Live state of one chat, folded from its server-sent event stream. Reconnects on its own after drops. */
export const chatState = Atom.family((id: string) =>
  Atom.make(
    Stream.callback<readonly AgentEvent[]>(
      Effect.fn(function* (queue) {
        yield* Effect.acquireRelease(
          Effect.sync(() => {
            const source = new EventSource(`/api/chats/${id}/events`)
            source.onmessage = (message) => Queue.offerUnsafe(queue, JSON.parse(message.data))
            return source
          }),
          (source) => Effect.sync(() => source.close()),
        )
      }),
    ).pipe(
      Stream.scan(
        () => emptyChat,
        (state: ChatState, events) => reduce(state, events),
      ),
    ),
  ),
)

export const speakReplies = Atom.make(localStorage.getItem("lora.speak") === "1")
