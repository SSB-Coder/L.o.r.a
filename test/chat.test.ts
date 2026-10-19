import { expect, test } from "bun:test"
import type { AgentEvent, EntryRecord } from "@earendil-works/pi-durable"
import { emptyChat, reduce, turns } from "../src/ui/chat.ts"

const entry = (id: number, model: unknown): EntryRecord =>
  ({ id, conversationId: 1, kind: "x", model }) as never
const assistant = (content: unknown[], extra = {}) => ({
  role: "assistant",
  content,
  stopReason: "stop",
  ...extra,
})

test("streaming deltas build the partial answer, and the committed entry replaces it", () => {
  let state = reduce(emptyChat, [
    { type: "run_start", inputs: [] },
    { type: "message_start", message: assistant([]) },
  ] as AgentEvent[])
  state = reduce(state, [
    {
      type: "message_update",
      usage: {},
      changes: [{ type: "text_start", contentIndex: 0, block: { type: "text", text: "" } }],
    },
    { type: "message_update", usage: {}, changes: [{ type: "text_delta", contentIndex: 0, delta: "Hel" }] },
    { type: "message_update", usage: {}, changes: [{ type: "text_delta", contentIndex: 0, delta: "lo" }] },
  ] as never)
  expect(turns(state)).toMatchObject([{ kind: "assistant", text: "Hello", streaming: true }])
  state = reduce(state, [
    { type: "message_end", entry: entry(3, [assistant([{ type: "text", text: "Hello" }])]) },
    { type: "run_end", inputs: [] },
  ] as never)
  expect(state.busy).toBe(false)
  expect(turns(state)).toMatchObject([{ kind: "assistant", text: "Hello", streaming: false }])
})

test("tool calls and their results join one assistant turn after the user's message", () => {
  const state = reduce(emptyChat, [
    {
      type: "snapshot",
      entries: [
        entry(1, [{ role: "user", content: "search it" }]),
        entry(2, [
          assistant([{ type: "toolCall", id: "c1", name: "web_search", arguments: { query: "q" } }]),
        ]),
        entry(3, [
          {
            role: "toolResult",
            toolCallId: "c1",
            content: [{ type: "text", text: "[1] result" }],
            isError: false,
          },
        ]),
        entry(4, [assistant([{ type: "text", text: "Answer [1]" }])]),
        entry(5, [{ role: "user", content: "[grounding check] fix links" }]),
      ],
      tools: [],
      compactions: [],
      inbox: [],
      agent: { model: { provider: "ollama", modelId: "qwen3:8b" } },
      usage: {},
    },
  ] as never)
  expect(state.model).toBe("qwen3:8b")
  expect(turns(state)).toMatchObject([
    { kind: "user", text: "search it" },
    {
      kind: "assistant",
      text: "Answer [1]",
      tools: [{ name: "web_search", output: "[1] result", status: "done" }],
    },
    { kind: "note" },
  ])
})

test("live tool output appends and trims", () => {
  let state = reduce(emptyChat, [
    { type: "tool_execution_start", toolCallId: "c", toolName: "t", args: {} },
  ] as never)
  state = reduce(state, [
    { type: "tool_execution_update", toolCallId: "c", toolName: "t", output: { append: "abcdef" } },
  ] as never)
  state = reduce(state, [
    { type: "tool_execution_update", toolCallId: "c", toolName: "t", output: { trimStart: 3, append: "g" } },
  ] as never)
  expect(state.live.c?.output).toBe("defg")
})
