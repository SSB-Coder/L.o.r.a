import { useAtom, useAtomRefresh, useAtomValue } from "@effect/atom-react"
import DOMPurify from "dompurify"
import { AsyncResult } from "effect/reactivity"
import { marked } from "marked"
import qrcode from "qrcode-generator"
import { type KeyboardEvent, useEffect, useMemo, useRef, useState } from "react"
import { type Turn, turns } from "./chat.ts"
import { type AppState, appState, call, chatState, selectedChat, speakReplies } from "./state.ts"
import { canListen, type Listening, listen, speak, stopSpeaking } from "./voice.ts"

type Mode = "idle" | "listening" | "thinking" | "speaking"

export function App() {
  const result = useAtomValue(appState)
  const refresh = useAtomRefresh(appState)
  if (AsyncResult.isFailure(result)) {
    const message = AsyncResult.error(result)
    const status = message._tag === "Some" ? message.value.status : 0
    return (
      <main className="center">
        <Orb mode="idle" />
        <h1>LORA</h1>
        <p>
          {status === 401
            ? "Open LORA from its app window, or scan the pairing QR code on your computer."
            : "LORA is not reachable."}
        </p>
        <button type="button" onClick={refresh}>
          Try again
        </button>
      </main>
    )
  }
  if (!AsyncResult.isSuccess(result))
    return (
      <main className="center">
        <Orb mode="thinking" />
      </main>
    )
  return <Shell state={result.value} refresh={refresh} />
}

function Shell({ state, refresh }: { state: AppState; refresh: () => void }) {
  const [selected, select] = useAtom(selectedChat)
  const [menu, setMenu] = useState(false)
  const current = state.chats.find((c) => c.id === selected) ?? state.chats[0]

  const newChat = async () => {
    const { id } = await call<{ id: string }>("chats", { body: {} })
    refresh()
    select(id)
    setMenu(false)
  }

  return (
    <div className={`shell ${menu ? "menu-open" : ""}`}>
      <aside>
        <button type="button" className="primary" onClick={newChat}>
          + New chat
        </button>
        <nav>
          {state.chats.map((chat) => (
            <div key={chat.id} className={`chat-item ${chat.id === current?.id ? "active" : ""}`}>
              <button
                type="button"
                onClick={() => {
                  select(chat.id)
                  setMenu(false)
                }}
              >
                {chat.origin === "mobile" ? "📱 " : ""}
                {chat.title}
              </button>
              <button
                type="button"
                className="hide"
                title="Remove from the list"
                onClick={async () => {
                  await call(`chats/${chat.id}`, { method: "DELETE" })
                  refresh()
                }}
              >
                ×
              </button>
            </div>
          ))}
        </nav>
        {state.origin === "desktop" && <Memory facts={state.memory} refresh={refresh} />}
        {state.pairing && <Pairing url={state.pairing} />}
        <small className="muted">LORA {state.version}</small>
      </aside>
      <section className="main">
        {current ? (
          <Chat
            key={current.id}
            id={current.id}
            title={current.title}
            state={state}
            refresh={refresh}
            onMenu={() => setMenu(!menu)}
          />
        ) : (
          <div className="center">
            <Orb mode="idle" />
            <h1>LORA</h1>
            <p className="muted">
              {state.models.length
                ? "Ask anything. I research the web and cite what I read."
                : "No Ollama models found. Start Ollama and pull a model."}
            </p>
            <button type="button" className="primary" onClick={newChat} disabled={!state.models.length}>
              Start a chat
            </button>
          </div>
        )}
      </section>
    </div>
  )
}

function Chat(props: {
  id: string
  title: string
  state: AppState
  refresh: () => void
  onMenu: () => void
}) {
  const { id, state, refresh } = props
  const result = useAtomValue(chatState(id))
  const chat = AsyncResult.isSuccess(result) ? result.value : undefined
  const items = useMemo(() => (chat ? turns(chat) : []), [chat])
  const [speakOn, setSpeak] = useAtom(speakReplies)
  const [mode, setMode] = useState<Mode>("idle")
  const [error, setError] = useState("")
  const bottom = useRef<HTMLDivElement>(null)
  const spoken = useRef<string | undefined>(undefined)

  // biome-ignore lint/correctness/useExhaustiveDependencies: scroll whenever the transcript changes
  useEffect(() => {
    // Block body on purpose: newer browsers return a Promise from scrollIntoView, which React would call as cleanup.
    bottom.current?.scrollIntoView({ block: "end" })
  }, [items])

  // Read each finished answer aloud once, when the user has turned speech on.
  const last = items.at(-1)
  useEffect(() => {
    if (!chat || chat.busy || last?.kind !== "assistant" || !last.text || last.streaming) return
    if (spoken.current === undefined) spoken.current = last.id // don't read history aloud on open
    if (spoken.current === last.id) return
    spoken.current = last.id
    if (!speakOn) {
      setMode("idle")
      return
    }
    setMode("speaking")
    void speak(last.text, state.voice.tts, () => setMode("idle"))
  }, [chat, last, speakOn, state.voice.tts])

  const send = async (text: string) => {
    if (!text.trim()) return
    setError("")
    try {
      await call(`chats/${id}/messages`, { body: { text, requestId: crypto.randomUUID() } })
      refresh()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const busy = chat?.busy ?? false
  const orb: Mode = mode === "idle" && busy ? "thinking" : mode

  return (
    <>
      <header>
        <button type="button" className="icon menu" onClick={props.onMenu} aria-label="Chats">
          ☰
        </button>
        <h2>{props.title}</h2>
        <select
          value={chat?.model ?? ""}
          onChange={(e) =>
            call(`chats/${id}`, { method: "PATCH", body: { model: e.target.value } }).catch((x) =>
              setError(x.message),
            )
          }
          aria-label="Model"
        >
          {[...new Set([chat?.model ?? "", ...state.models])].filter(Boolean).map((m) => (
            <option key={m} value={m}>
              {m}
              {state.models.includes(m) ? "" : " (not installed)"}
            </option>
          ))}
        </select>
        <button
          type="button"
          className={`icon ${speakOn ? "on" : ""}`}
          title="Read answers aloud"
          onClick={() => {
            localStorage.setItem("lora.speak", speakOn ? "0" : "1")
            if (speakOn) stopSpeaking()
            setSpeak(!speakOn)
          }}
        >
          🔊
        </button>
      </header>
      <div className="messages">
        {items.length === 0 && (
          <div className="center">
            <Orb mode={orb} />
          </div>
        )}
        {items.map((turn) => (
          <TurnView key={turn.id} turn={turn} />
        ))}
        {chat?.notice && <p className="notice">{chat.notice}</p>}
        {error && <p className="notice error">{error}</p>}
        <div ref={bottom} />
      </div>
      <Composer
        busy={busy}
        mode={orb}
        serverStt={state.voice.stt}
        onSend={send}
        onMode={setMode}
        onError={setError}
        onStop={async () => {
          stopSpeaking()
          setMode("idle")
          await call(`chats/${id}/abort`, { body: {} })
        }}
      />
    </>
  )
}

function TurnView({ turn }: { turn: Turn }) {
  if (turn.kind === "note") return <p className="note">{turn.text}</p>
  if (turn.kind === "user") return <div className="bubble user">{turn.text}</div>
  return (
    <div className="bubble assistant">
      {turn.tools.map((tool) => (
        <details key={tool.id} className={`tool ${tool.status}`}>
          <summary>
            {tool.status === "running" ? "⏳" : tool.status === "error" ? "⚠️" : "✓"} {tool.name}{" "}
            <span className="muted">{summarize(tool.args)}</span>
          </summary>
          <pre>{tool.output || "…"}</pre>
        </details>
      ))}
      {turn.text ? (
        <Markdown text={turn.text} />
      ) : (
        turn.streaming && turn.tools.length === 0 && <span className="dots" />
      )}
      {turn.error && <p className="notice error">{turn.error}</p>}
    </div>
  )
}

const summarize = (args: unknown) => {
  const values = Object.values((args ?? {}) as Record<string, unknown>).filter((v) => typeof v === "string")
  const text = values.join(" · ")
  return text.length > 80 ? `${text.slice(0, 80)}…` : text
}

function Markdown({ text }: { text: string }) {
  const html = useMemo(() => DOMPurify.sanitize(marked.parse(text, { async: false })), [text])
  // biome-ignore lint/security/noDangerouslySetInnerHtml: sanitized by DOMPurify above
  return <div className="markdown" dangerouslySetInnerHTML={{ __html: html }} />
}

// Links in answers open outside the app window.
DOMPurify.addHook("afterSanitizeAttributes", (node) => {
  if (node.tagName === "A") {
    node.setAttribute("target", "_blank")
    node.setAttribute("rel", "noopener noreferrer")
  }
})

function Composer(props: {
  busy: boolean
  mode: Mode
  serverStt: boolean
  onSend: (text: string) => Promise<void>
  onStop: () => void
  onMode: (mode: Mode) => void
  onError: (error: string) => void
}) {
  const [text, setText] = useState("")
  const listening = useRef<Listening | undefined>(undefined)
  const voice = canListen(props.serverStt)

  const submit = async () => {
    const value = text
    setText("")
    await props.onSend(value)
  }

  const toggleMic = async () => {
    if (listening.current) return listening.current.stop()
    stopSpeaking()
    try {
      listening.current = listen(props.serverStt)
      props.onMode("listening")
      const heard = await listening.current.text
      props.onMode("idle")
      if (heard) await props.onSend(heard)
    } catch (e) {
      props.onMode("idle")
      props.onError(`Microphone: ${(e as Error).message}`)
    } finally {
      listening.current = undefined
    }
  }

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      void submit()
    }
  }

  return (
    <footer>
      {voice && (
        <button
          type="button"
          className={`mic ${props.mode === "listening" ? "on" : ""}`}
          onClick={toggleMic}
          title="Talk"
        >
          <Orb mode={props.mode} small />
        </button>
      )}
      <textarea
        value={text}
        rows={1}
        placeholder={props.busy ? "Add to the request…" : "Ask LORA…"}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKey}
      />
      {props.busy ? (
        <button type="button" className="stop" onClick={props.onStop}>
          Stop
        </button>
      ) : (
        <button type="button" className="primary" onClick={submit} disabled={!text.trim()}>
          Send
        </button>
      )}
    </footer>
  )
}

function Memory({ facts, refresh }: { facts: string[]; refresh: () => void }) {
  if (facts.length === 0) return null
  return (
    <details className="memory">
      <summary>Remembered ({facts.length})</summary>
      {facts.map((fact) => (
        <p key={fact}>
          {fact}{" "}
          <button
            type="button"
            className="link"
            onClick={async () => {
              await call("memory", { method: "DELETE", body: { fact } })
              refresh()
            }}
          >
            forget
          </button>
        </p>
      ))}
    </details>
  )
}

function Pairing({ url }: { url: string }) {
  const svg = useMemo(() => {
    const qr = qrcode(0, "M")
    qr.addData(url)
    qr.make()
    return qr.createSvgTag({ cellSize: 4, margin: 2, scalable: true })
  }, [url])
  return (
    <details className="pairing">
      <summary>Pair a phone</summary>
      {/* biome-ignore lint/security/noDangerouslySetInnerHtml: SVG built by qrcode-generator from our own URL */}
      <div className="qr" dangerouslySetInnerHTML={{ __html: svg }} />
      <small className="muted">Scan on the LORA network. The link signs the phone in.</small>
    </details>
  )
}

function Orb({ mode, small }: { mode: Mode; small?: boolean }) {
  return <span className={`orb ${mode} ${small ? "small" : ""}`} role="img" aria-label={mode} />
}
