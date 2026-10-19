import { homedir } from "node:os"
import { join, resolve } from "node:path"
import { Effect, Schema } from "effect"

// Every field has a default, so an empty or missing lora.yaml is a valid config.
const opt = <S extends Schema.Top>(schema: S, value: S["Encoded"]) =>
  schema.pipe(Schema.withDecodingDefaultKey(Effect.succeed(value)))

const Shortcut = Schema.Struct({
  /** Windows: passed to `start`; macOS: an app name for `open -a`; Linux: a command. */
  command: opt(Schema.String, ""),
  url: opt(Schema.String, ""),
  /** URL with `{query}`, for "search <site> for ...". */
  search: opt(Schema.String, ""),
  aliases: opt(Schema.Array(Schema.String), []),
})
export type Shortcut = typeof Shortcut.Type

const Host = Schema.String.pipe(
  Schema.check(
    Schema.makeFilter((host) =>
      host === "0.0.0.0" || host === "::" ? "never bind to every interface" : undefined,
    ),
  ),
)

export const Config = Schema.Struct({
  ollama: opt(
    Schema.Struct({
      url: opt(Schema.String, "http://127.0.0.1:11434"),
      /** Default model for new chats; empty picks the first installed one. */
      model: opt(Schema.String, ""),
      contextWindow: opt(Schema.Int, 8192),
    }),
    {},
  ),
  server: opt(
    Schema.Struct({
      port: opt(Schema.Int, 8765),
      /** Phone access. Bind it to a private, isolated interface (see docs/NETWORK_SETUP.md); absent means off. */
      mobile: Schema.optionalKey(Schema.Struct({ host: Host, port: opt(Schema.Int, 8765) })),
    }),
    {},
  ),
  files: opt(
    Schema.Struct({
      /** The only directories the assistant may read from. */
      roots: opt(Schema.Array(Schema.String), ["~/Documents", "~/Downloads", "~/Desktop"]),
      notes: opt(Schema.String, "~/Documents/LORA Notes"),
    }),
    {},
  ),
  research: opt(Schema.Struct({ searxng: opt(Schema.String, "") }), {}),
  voice: opt(
    Schema.Struct({
      /** OpenAI-compatible /v1/audio/transcriptions endpoint; empty uses the browser's speech recognition. */
      stt: opt(Schema.String, ""),
      sttModel: opt(Schema.String, "whisper-1"),
      /** OpenAI-compatible /v1/audio/speech endpoint (e.g. Kokoro-FastAPI); empty uses the browser's voices. */
      tts: opt(Schema.String, ""),
      ttsModel: opt(Schema.String, "kokoro"),
      voice: opt(Schema.String, "af_sky"),
    }),
    {},
  ),
  apps: opt(Schema.Record(Schema.String, Shortcut), defaultApps()),
  sites: opt(Schema.Record(Schema.String, Shortcut), defaultSites()),
})
export type Config = typeof Config.Type

export class ConfigError extends Schema.TaggedError<ConfigError>()("ConfigError", {
  path: Schema.String,
  message: Schema.String,
}) {}

export const home = () => process.env.LORA_HOME ?? join(homedir(), ".lora")

export const expandHome = (path: string) => resolve(path.replace(/^~(?=$|[\\/])/, homedir()))

export const decodeConfig = Schema.decodeUnknownSync(Config)

export const loadConfig = Effect.fn("loadConfig")(function* (path = join(home(), "lora.yaml")) {
  const file = Bun.file(path)
  const text = (yield* Effect.promise(() => file.exists())) ? yield* Effect.promise(() => file.text()) : ""
  const raw = yield* Effect.try({
    try: () => Bun.YAML.parse(text) ?? {},
    catch: (e) => new ConfigError({ path, message: String(e) }),
  })
  return yield* Schema.decodeUnknownEffect(Config)(raw).pipe(
    Effect.mapError((e) => new ConfigError({ path, message: e.message })),
  )
})

function defaultApps(): Record<string, typeof Shortcut.Encoded> {
  const win = process.platform === "win32"
  const mac = process.platform === "darwin"
  const pick = (w: string, m: string, l: string) => (win ? w : mac ? m : l)
  return {
    vscode: { command: pick("code", "Visual Studio Code", "code"), aliases: ["code", "vs code"] },
    notepad: { command: pick("notepad.exe", "TextEdit", "gedit"), aliases: ["text editor", "notes app"] },
    calculator: { command: pick("calc.exe", "Calculator", "gnome-calculator"), aliases: ["calc"] },
    files: { command: pick("explorer.exe", "Finder", "xdg-open ~"), aliases: ["file explorer", "finder"] },
    spotify: { command: pick("spotify:", "Spotify", "spotify"), aliases: ["music"] },
  }
}

function defaultSites(): Record<string, typeof Shortcut.Encoded> {
  return {
    wikipedia: {
      url: "https://en.wikipedia.org",
      search: "https://en.wikipedia.org/wiki/Special:Search?search={query}",
      aliases: ["wiki"],
    },
    arxiv: { url: "https://arxiv.org", search: "https://arxiv.org/search/?query={query}&searchtype=all" },
    youtube: {
      url: "https://www.youtube.com",
      search: "https://www.youtube.com/results?search_query={query}",
    },
    google: { url: "https://www.google.com", search: "https://www.google.com/search?q={query}" },
    github: { url: "https://github.com", search: "https://github.com/search?q={query}" },
    perplexity: { url: "https://www.perplexity.ai", search: "https://www.perplexity.ai/search?q={query}" },
    chatgpt: { url: "https://chatgpt.com", aliases: ["chat gpt"] },
    claude: { url: "https://claude.ai" },
    gemini: { url: "https://gemini.google.com" },
  }
}
