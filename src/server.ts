import { BACKGROUND_CONTEXT as ctx } from "@earendil-works/chord/context"
import { type AgentEvent, watchEvents } from "@earendil-works/pi-durable"
import type { HTMLBundle } from "bun"
import { Effect, Schema } from "effect"
import type { Agent } from "./agent.ts"
import { COOKIE, cookieToken, LoginLimiter, sameSite, sameToken, type Tokens } from "./auth.ts"
import type { Config } from "./config.ts"
import type { Ollama } from "./models.ts"
import type { Origin } from "./policy.ts"

export type Deps = {
  agent: Agent["Service"]
  ollama: Ollama
  config: Config
  tokens: Tokens
  version: string
  limiter?: LoginLimiter
}

/** One listening socket: who connects through it, under which Host names, with which token. */
export type Listener = { origin: Origin; hosts: string[]; token: string }

const Send = Schema.Struct({ text: Schema.NonEmptyString, requestId: Schema.NonEmptyString })
const Patch = Schema.Struct({ model: Schema.String })
const NewChat = Schema.Struct({ model: Schema.optional(Schema.String) })
const Forget = Schema.Struct({ fact: Schema.String })
const Speak = Schema.Struct({ text: Schema.NonEmptyString })

const json = (body: unknown, status = 200) => Response.json(body, { status })
const fail = (status: number, error: string) => json({ error }, status)

class HttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message)
  }
}

const body = async <T>(request: Request, schema: Schema.ConstraintDecoder<T>): Promise<T> => {
  const value = await request.json().catch(() => undefined)
  const exit = Schema.decodeUnknownExit(schema)(value)
  if (exit._tag === "Failure") throw new HttpError(400, "invalid request body")
  return exit.value
}

export function createHandler(deps: Deps) {
  const { agent, ollama, config, tokens } = deps
  const limiter = deps.limiter ?? new LoginLimiter()

  const chatList = async (origin: Origin) =>
    Object.entries(await agent.chats())
      .filter(([, info]) => origin === "desktop" || info.origin === "mobile")
      .map(([id, info]) => ({ id, ...info }))
      .sort((a, b) => b.created - a.created)

  const chatOr404 = async (id: string, origin: Origin) => {
    const chat = await agent.chat(id, origin)
    if (!chat) throw new HttpError(404, "no such chat")
    return chat
  }

  const desktopOnly = (origin: Origin) => {
    if (origin !== "desktop") throw new HttpError(403, "only available on the computer itself")
  }

  async function api(request: Request, listener: Listener, path: string[]): Promise<Response> {
    const { origin } = listener
    const method = request.method
    const [resource, id, action] = path

    if (resource === "state" && method === "GET") {
      return json({
        version: deps.version,
        origin,
        models: ollama.list(),
        defaultModel: ollama.preferred(),
        chats: await chatList(origin),
        memory: origin === "desktop" ? await agent.memory() : [],
        voice: { stt: Boolean(config.voice.stt), tts: Boolean(config.voice.tts) },
        pairing: origin === "desktop" ? pairingUrl(config, tokens) : undefined,
      })
    }
    if (resource === "models" && method === "POST") {
      return json({ models: await Effect.runPromise(ollama.refresh()) })
    }
    if (resource === "memory" && method === "DELETE") {
      desktopOnly(origin)
      await agent.forget((await body(request, Forget)).fact)
      return json({ ok: true })
    }
    if (resource === "voice" && id === "transcribe" && method === "POST") return transcribe(request)
    if (resource === "voice" && id === "speak" && method === "POST") return speak(request)

    if (resource === "chats" && id === undefined && method === "POST") {
      const { model } = await body(request, NewChat)
      if (model && !ollama.list().includes(model)) throw new HttpError(400, `model ${model} is not installed`)
      return json({ id: await agent.createChat(origin, model) })
    }
    if (resource !== "chats" || id === undefined) throw new HttpError(404, "not found")

    const { conversation } = await chatOr404(id, origin)
    if (action === "events" && method === "GET") return events(request, id)
    if (action === "messages" && method === "POST") {
      const { text, requestId } = await body(request, Send)
      const submission = await agent.send(id, text, requestId)
      return json({ submissionId: submission.id })
    }
    if (action === "abort" && method === "POST") {
      await conversation.abort(ctx)
      return json({ ok: true })
    }
    if (action === undefined && method === "PATCH") {
      const { model } = await body(request, Patch)
      if (!ollama.list().includes(model)) throw new HttpError(400, `model ${model} is not installed`)
      await agent.setModel(id, model)
      return json({ ok: true })
    }
    if (action === undefined && method === "DELETE") {
      await agent.hide(id)
      return json({ ok: true })
    }
    throw new HttpError(404, "not found")
  }

  /** Server-sent events: the chat's snapshot, then one batch of Pi Durable agent events per commit. */
  async function events(request: Request, id: string): Promise<Response> {
    const stream = await watchEvents(agent.harness, Number(id) as never, ctx)
    const encoder = new TextEncoder()
    let ping: ReturnType<typeof setInterval> | undefined
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        const send = (events: readonly AgentEvent[]) =>
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(events)}\n\n`))
        send([stream.snapshot])
        stream.start(async (events) => send(events))
        ping = setInterval(() => controller.enqueue(encoder.encode(": ping\n\n")), 15_000)
        void stream.closed
          .then(() => {
            clearInterval(ping)
            controller.close()
          })
          .catch(() => {})
      },
      cancel() {
        clearInterval(ping)
        void stream.stop()
      },
    })
    request.signal.addEventListener("abort", () => {
      clearInterval(ping)
      void stream.stop()
    })
    return new Response(body, {
      headers: { "content-type": "text/event-stream", "cache-control": "no-store", connection: "keep-alive" },
    })
  }

  async function transcribe(request: Request): Promise<Response> {
    if (!config.voice.stt) throw new HttpError(404, "no speech-to-text server configured")
    const form = new FormData()
    form.append(
      "file",
      new File([await request.arrayBuffer()], "speech.webm", {
        type: request.headers.get("content-type") ?? "audio/webm",
      }),
    )
    form.append("model", config.voice.sttModel)
    const response = await fetch(config.voice.stt, { method: "POST", body: form })
    if (!response.ok) throw new HttpError(502, `speech-to-text failed: ${response.status}`)
    const { text } = (await response.json()) as { text?: string }
    return json({ text: text?.trim() ?? "" })
  }

  async function speak(request: Request): Promise<Response> {
    if (!config.voice.tts) throw new HttpError(404, "no text-to-speech server configured")
    const { text } = await body(request, Speak)
    const response = await fetch(config.voice.tts, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: config.voice.ttsModel,
        voice: config.voice.voice,
        input: text,
        response_format: "mp3",
      }),
    })
    if (!response.ok || !response.body) throw new HttpError(502, `text-to-speech failed: ${response.status}`)
    return new Response(response.body, {
      headers: { "content-type": response.headers.get("content-type") ?? "audio/mpeg" },
    })
  }

  /** Every non-page request: the same-site check, then the token, then the API. */
  return async function handle(request: Request, listener: Listener, ip: string): Promise<Response> {
    if (!sameSite(request, listener.hosts)) return fail(403, "forbidden")
    if (limiter.banned(ip)) return fail(429, "too many failed logins; try again later")
    const url = new URL(request.url)

    if (url.pathname === "/auth") {
      if (!sameToken(url.searchParams.get("token"), listener.token)) {
        limiter.fail(ip)
        return fail(401, "invalid pairing link")
      }
      limiter.succeed(ip)
      return new Response(null, {
        status: 302,
        headers: {
          location: "/",
          "set-cookie": `${COOKIE}=${listener.token}; Path=/; HttpOnly; SameSite=Strict; Max-Age=31536000`,
        },
      })
    }

    const bearer = request.headers.get("authorization")?.replace(/^Bearer\s+/i, "")
    if (!sameToken(bearer ?? cookieToken(request), listener.token)) {
      limiter.fail(ip)
      return fail(401, "not signed in")
    }
    try {
      return await api(request, listener, url.pathname.split("/").slice(2).filter(Boolean))
    } catch (error) {
      if (error instanceof HttpError) return fail(error.status, error.message)
      console.error("[lora]", error)
      return fail(500, error instanceof Error ? error.message : "internal error")
    }
  }
}

export function pairingUrl(config: Config, tokens: Tokens): string | undefined {
  const mobile = config.server.mobile
  return mobile && `http://${mobile.host}:${mobile.port}/auth?token=${tokens.mobile}`
}

/** Bind the desktop listener on loopback, and the phone listener when configured. */
export function listen(deps: Deps, page: HTMLBundle) {
  const handle = createHandler(deps)
  const { port, mobile } = deps.config.server
  const listeners: (Listener & { hostname: string; port: number })[] = [
    {
      origin: "desktop",
      hostname: "127.0.0.1",
      port,
      hosts: [`127.0.0.1:${port}`, `localhost:${port}`],
      token: deps.tokens.desktop,
    },
  ]
  if (mobile) {
    listeners.push({
      origin: "mobile",
      hostname: mobile.host,
      port: mobile.port,
      hosts: [`${mobile.host}:${mobile.port}`],
      token: deps.tokens.mobile,
    })
  }
  return listeners.map((listener) =>
    Bun.serve({
      hostname: listener.hostname,
      port: listener.port,
      idleTimeout: 60,
      // LORA_DEV=1 serves the UI unminified with hot reload.
      development: process.env.LORA_DEV === "1",
      routes: { "/": page },
      fetch: (request, server) => handle(request, listener, server.requestIP(request)?.address ?? "unknown"),
    }),
  )
}
