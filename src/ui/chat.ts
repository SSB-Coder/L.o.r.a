import type { AssistantMessage, Message, ToolResultMessage } from "@earendil-works/pi-ai"
import type { AgentEvent, EntryRecord } from "@earendil-works/pi-durable"

/** What the client keeps of one chat: committed entries plus the in-flight answer and tool output. */
export type ChatState = {
  entries: EntryRecord[]
  partial?: AssistantMessage | undefined
  live: Record<string, { name: string; args: unknown; output: string }>
  busy: boolean
  model?: string | undefined
  notice?: string | undefined
}

export const emptyChat: ChatState = { entries: [], live: {}, busy: false }

const append = (entries: EntryRecord[], entry: EntryRecord) =>
  entries.some((e) => e.id === entry.id) ? entries : [...entries, entry]

/** Apply one batch of agent events (one commit). */
export function reduce(state: ChatState, events: readonly AgentEvent[]): ChatState {
  let next = state
  for (const event of events) next = step(next, event)
  return next
}

function step(state: ChatState, event: AgentEvent): ChatState {
  switch (event.type) {
    case "snapshot":
      return {
        entries: [...event.entries],
        partial: event.generation?.message,
        live: Object.fromEntries(
          event.tools
            .filter((t) => t.status !== "done")
            .map((t) => [t.callId, { name: t.name, args: {}, output: t.output ?? "" }]),
        ),
        busy: event.run !== undefined,
        model: event.agent.model?.modelId,
        notice: event.generation?.retry && `Retrying: ${event.generation.retry.error}`,
      }
    case "run_start":
      return { ...state, busy: true, notice: undefined }
    case "run_end":
      return { ...state, busy: false, partial: undefined, live: {} }
    case "message_start":
      return event.message.role === "assistant" ? { ...state, partial: event.message } : state
    case "message_update":
      return state.partial ? { ...state, partial: applyChanges(state.partial, event.changes) } : state
    case "message_end":
      return { ...state, entries: append(state.entries, event.entry), partial: undefined, notice: undefined }
    case "entry_appended":
      if (event.entry.kind === "pi.reset") return { ...state, entries: [event.entry] }
      return { ...state, entries: append(state.entries, event.entry) }
    case "tool_execution_start":
      return {
        ...state,
        live: { ...state.live, [event.toolCallId]: { name: event.toolName, args: event.args, output: "" } },
      }
    case "tool_execution_update": {
      const tool = state.live[event.toolCallId]
      if (!tool || !event.output) return state
      const output =
        "set" in event.output
          ? event.output.set
          : tool.output.slice(event.output.trimStart ?? 0) + (event.output.append ?? "")
      return { ...state, live: { ...state.live, [event.toolCallId]: { ...tool, output } } }
    }
    case "tool_execution_end": {
      const { [event.toolCallId]: _, ...live } = state.live
      return { ...state, live, entries: event.entry ? append(state.entries, event.entry) : state.entries }
    }
    case "agent_changed":
      return { ...state, model: event.agent.model?.modelId }
    case "auto_retry_start":
      return { ...state, notice: `Retrying (${event.attempt}): ${event.errorMessage}` }
    case "auto_retry_end":
      return { ...state, notice: undefined }
    case "task_failed":
      return { ...state, notice: event.message }
    default:
      return state
  }
}

type Changes = Extract<AgentEvent, { type: "message_update" }>["changes"]

function applyChanges(message: AssistantMessage, changes: Changes): AssistantMessage {
  const content = [...message.content]
  for (const change of changes) {
    if (change.type === "message") return change.message
    if (change.type === "toolcall_delta") continue
    const block = content[change.contentIndex]
    if (change.type === "text_delta" && block?.type === "text") {
      content[change.contentIndex] = { ...block, text: block.text + change.delta }
    } else if (change.type === "thinking_delta" && block?.type === "thinking") {
      content[change.contentIndex] = { ...block, thinking: block.thinking + change.delta }
    } else if ("block" in change) {
      content[change.contentIndex] = change.block
    }
  }
  return { ...message, content }
}

export type ToolView = {
  id: string
  name: string
  args: unknown
  output: string
  status: "running" | "done" | "error"
}

export type Turn =
  | { kind: "user"; id: string; text: string }
  | {
      kind: "assistant"
      id: string
      text: string
      thinking: string
      tools: ToolView[]
      error?: string
      streaming: boolean
    }
  | { kind: "note"; id: string; text: string }

const GROUNDING = "[grounding check]"

const textOf = (content: Message["content"]) =>
  typeof content === "string" ? content : content.flatMap((b) => ("text" in b ? [b.text] : [])).join("")

/** Display turns: each user message, then one assistant turn holding its tool calls and final answer. */
export function turns(state: ChatState): Turn[] {
  const out: Turn[] = []
  const results = new Map<string, ToolResultMessage>()
  for (const entry of state.entries) {
    const message = entry.model?.[0]
    if (message?.role === "toolResult") results.set(message.toolCallId, message)
  }
  const assistantTurn = (id: string) => {
    const last = out.at(-1)
    if (last?.kind === "assistant") return last
    const turn: Turn = { kind: "assistant", id, text: "", thinking: "", tools: [], streaming: false }
    out.push(turn)
    return turn
  }
  const addAssistant = (id: string, message: AssistantMessage, streaming: boolean) => {
    const turn = assistantTurn(id)
    if (turn.kind !== "assistant") return
    let text = ""
    for (const block of message.content) {
      if (block.type === "text") text += block.text
      else if (block.type === "thinking") turn.thinking += block.thinking
      else if (block.type === "toolCall") {
        const result = results.get(block.id)
        const live = state.live[block.id]
        turn.tools.push({
          id: block.id,
          name: block.name,
          args: block.arguments,
          output: result ? textOf(result.content) : (live?.output ?? ""),
          status: result ? (result.isError ? "error" : "done") : "running",
        })
      }
    }
    // A later message in the same turn (the answer after tool calls, or a rewrite) replaces earlier text.
    if (text) turn.text = text
    if (message.stopReason === "error" || message.stopReason === "aborted") {
      turn.error =
        message.stopReason === "aborted" ? "Stopped." : (message.errorMessage ?? "The model failed.")
    }
    turn.streaming = streaming
  }

  for (const entry of state.entries) {
    const message = entry.model?.[0]
    const id = String(entry.id)
    if (entry.kind === "pi.compaction")
      out.push({ kind: "note", id, text: "Earlier messages were summarized." })
    else if (entry.kind === "pi.reset") out.push({ kind: "note", id, text: "Started a fresh context." })
    else if (message?.role === "user") {
      const text = textOf(message.content)
      if (text.startsWith(GROUNDING))
        out.push({ kind: "note", id, text: "Checking that every link was actually read…" })
      else out.push({ kind: "user", id, text })
    } else if (message?.role === "assistant") addAssistant(id, message, false)
  }
  if (state.partial) addAssistant("partial", state.partial, true)
  return out
}
