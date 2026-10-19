import { createModels, createProvider, type Model, type MutableModels } from "@earendil-works/pi-ai"
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy"
import { Effect, Schema } from "effect"
import type { Config } from "./config.ts"

export const PROVIDER = "ollama"

const Tags = Schema.Struct({ models: Schema.Array(Schema.Struct({ name: Schema.String })) })

/** Names of the chat models Ollama has installed; empty when Ollama is not running. */
export const installedModels = Effect.fn("installedModels")(
  function* (ollamaUrl: string) {
    const response = yield* Effect.tryPromise((signal) => fetch(`${ollamaUrl}/api/tags`, { signal })).pipe(
      Effect.timeout("3 seconds"),
    )
    const { models } = yield* Schema.decodeUnknownEffect(Tags)(yield* Effect.promise(() => response.json()))
    return models.map((m) => m.name).filter((name) => !/embed|rerank/i.test(name))
  },
  Effect.orElseSucceed((): string[] => []),
)

const toModel = (config: Config["ollama"], id: string): Model<"openai-completions"> => ({
  id,
  name: id,
  api: "openai-completions",
  provider: PROVIDER,
  baseUrl: `${config.url}/v1`,
  reasoning: false,
  input: ["text"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: config.contextWindow,
  maxTokens: Math.min(4096, Math.floor(config.contextWindow / 2)),
  compat: {
    supportsDeveloperRole: false,
    supportsReasoningEffort: false,
    supportsStore: false,
    maxTokensField: "max_tokens",
  },
})

/** pi-ai model access backed by the local Ollama server. `refresh()` picks up models pulled while running. */
export function ollama(config: Config["ollama"]) {
  const models: MutableModels = createModels()
  let names: string[] = []
  const refresh = Effect.fn("ollama.refresh")(function* () {
    names = yield* installedModels(config.url)
    models.setProvider(
      createProvider({
        id: PROVIDER,
        name: "Ollama",
        baseUrl: `${config.url}/v1`,
        // Ollama ignores the key, but pi-ai's OpenAI client refuses to send a request without one.
        auth: { apiKey: { name: "Ollama", resolve: async () => ({ auth: { apiKey: "ollama" } }) } },
        models: names.map((id) => toModel(config, id)),
        api: openAICompletionsApi(),
      }),
    )
    return names
  })
  return {
    models,
    refresh,
    list: () => names,
    /** The configured model when installed, else the first installed one. */
    preferred: () => (names.includes(config.model) ? config.model : (names[0] ?? config.model)),
  }
}
export type Ollama = ReturnType<typeof ollama>
