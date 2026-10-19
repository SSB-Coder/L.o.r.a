#!/usr/bin/env bun
import { existsSync } from "node:fs"
import { join } from "node:path"
import { BunRuntime, BunServices } from "@effect/platform-bun"
import { Console, Effect, Layer, Option } from "effect"
import { Command, Flag } from "effect/cli"
import { Agent } from "./agent.ts"
import { loadTokens } from "./auth.ts"
import { home, loadConfig } from "./config.ts"
import { ollama } from "./models.ts"
import { listen, pairingUrl } from "./server.ts"
import page from "./ui/index.html"
import { VERSION } from "./version.ts"

const configFlag = Flag.String("config").pipe(
  Flag.withDescription("Path to lora.yaml (default: ~/.lora/lora.yaml)"),
  Flag.optional,
)
const configPath = (flag: Option.Option<string>) => Option.getOrElse(flag, () => join(home(), "lora.yaml"))

const serve = Effect.fn("serve")(function* (options: { config: Option.Option<string>; open: boolean }) {
  const config = yield* loadConfig(configPath(options.config))
  const models = ollama(config.ollama)
  const installed = yield* models.refresh()
  if (installed.length === 0) {
    yield* Effect.logWarning(
      `No models found at ${config.ollama.url}. Start Ollama and pull a model, e.g. "ollama pull qwen3:8b".`,
    )
  }
  const tokens = yield* Effect.promise(() => loadTokens(home()))
  const desktopUrl = `http://127.0.0.1:${config.server.port}/auth?token=${tokens.desktop}`

  const app = Layer.effectDiscard(
    Effect.gen(function* () {
      const agent = yield* Agent
      const servers = yield* Effect.acquireRelease(
        Effect.try(() => listen({ agent, ollama: models, config, tokens, version: VERSION }, page)),
        (servers) => Effect.promise(() => Promise.all(servers.map((s) => s.stop(true)))),
      )
      yield* Effect.log(`LORA ${VERSION} on ${servers.map((s) => s.url.href).join(" and ")}`)
      if (options.open) openWindow(desktopUrl)
    }),
  ).pipe(Layer.provide(Agent.layer(config, models.models, join(home(), "lora.sqlite"), models.preferred)))

  return yield* Layer.launch(app).pipe(
    Effect.catchDefect((defect) =>
      String(defect).includes("EADDRINUSE")
        ? Effect.gen(function* () {
            // Another LORA owns the port, and the storage; just show its window.
            yield* Effect.log(`LORA is already running on port ${config.server.port}.`)
            if (options.open) openWindow(desktopUrl)
          })
        : Effect.die(defect),
    ),
  )
})

const lora = Command.make(
  "lora",
  {
    config: configFlag,
    noOpen: Flag.Boolean("no-open").pipe(
      Flag.withDescription("Don't open the app window"),
      Flag.withDefault(false),
    ),
  },
  ({ config, noOpen }) => serve({ config, open: !noOpen }),
).pipe(Command.withDescription("L.O.R.A — a local, durable research assistant on Ollama"))

const pair = Command.make("pair", { config: configFlag }, ({ config }) =>
  Effect.gen(function* () {
    const loaded = yield* loadConfig(configPath(config))
    const url = pairingUrl(loaded, yield* Effect.promise(() => loadTokens(home())))
    yield* Console.log(
      url ??
        "Phone access is off. Add `server: { mobile: { host: <this PC's isolated LAN IP> } }` to lora.yaml.",
    )
  }),
).pipe(Command.withDescription("Print the link that pairs a phone (also shown as a QR code in the app)"))

/** Open the UI as an app window (Edge/Chrome `--app`) when one of those browsers exists, else the default browser. */
function openWindow(url: string) {
  const spawn = (cmd: string[]) => Bun.spawn(cmd, { stdio: ["ignore", "ignore", "ignore"] }).unref()
  if (process.platform === "win32") return spawn(["cmd", "/c", "start", "", "msedge", `--app=${url}`])
  if (process.platform === "darwin") {
    const chrome = "/Applications/Google Chrome.app"
    return spawn(existsSync(chrome) ? ["open", "-na", chrome, "--args", `--app=${url}`] : ["open", url])
  }
  const browser = ["google-chrome", "chromium", "chromium-browser", "microsoft-edge"].find((b) =>
    Bun.which(b),
  )
  spawn(browser ? [browser, `--app=${url}`] : ["xdg-open", url])
}

lora.pipe(
  Command.withSubcommands([pair]),
  Command.run({ version: VERSION }),
  Effect.provide(BunServices.layer),
  BunRuntime.runMain,
)
