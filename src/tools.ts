import { mkdir, readdir, stat } from "node:fs/promises"
import { basename, join } from "node:path"
import type { Context } from "@earendil-works/chord"
import { Type } from "@earendil-works/pi-ai"
import {
  defineDoc,
  defineTool,
  type ToolExecutionApi,
  type ToolRegistration,
} from "@earendil-works/pi-durable"
import { Cause, Effect, Exit } from "effect"
import { type Config, expandHome, type Shortcut } from "./config.ts"
import { checkPath, isBlockedFile } from "./policy.ts"
import { fetchPage, searchPapers, webSearch } from "./web.ts"

/** URLs this chat has actually retrieved, so answers can be checked against them. */
export const Sources = defineDoc<{ urls: Record<string, string> }>({
  kind: "lora.sources",
  version: 1,
  scope: "conversation",
  history: "latest",
  fork: "current",
  initial: () => ({ urls: {} }),
})

/** Facts the user asked LORA to remember, shown to every chat. */
export const Memory = defineDoc<{ facts: string[] }>({
  kind: "lora.memory",
  version: 1,
  scope: "session",
  initial: () => ({ facts: [] }),
})

/** Run an Effect for a tool call; cancelling the call interrupts it, and failures become the tool's error text. */
const run = async <A, E>(effect: Effect.Effect<A, E>, context: Context): Promise<A> => {
  const exit = await Effect.runPromiseExit(effect, { signal: context.abortSignal })
  if (Exit.isSuccess(exit)) return exit.value
  const error = Cause.squash(exit.cause) as { message?: string }
  throw new Error(error?.message ?? String(error))
}

const text = (value: string) => ({ content: [{ type: "text" as const, text: value }] })

const remember = (api: ToolExecutionApi, context: Context, links: { url: string; title: string }[]) =>
  api.commit(async (tx) => {
    const doc = await tx.doc(Sources, api.conversationId)
    for (const link of links) doc.urls[normalizeUrl(link.url)] = link.title
  }, context)

export const normalizeUrl = (url: string) =>
  url
    .replace(/[#?].*$/, "")
    .replace(/\/+$/, "")
    .replace(/^http:/, "https:")

export function createTools(config: Config) {
  const roots = config.files.roots.map(expandHome)
  const allowed = async (path: string) => {
    const check = await checkPath(path, roots)
    if (!check.ok) throw new Error(check.reason)
    return check.path
  }

  const web_search = defineTool({
    name: "web_search",
    description: "Search the web. Returns numbered results with title, URL, and snippet.",
    parameters: Type.Object({ query: Type.String({ description: "Search terms" }) }),
    replay: "safe",
    execute: async ({ query }, api, context) => {
      const results = await run(webSearch(query, config.research.searxng), context)
      await remember(api, context, results)
      if (results.length === 0) return text(`No results for "${query}".`)
      return text(results.map((r, i) => `[${i + 1}] ${r.title}\n${r.url}\n${r.snippet}`).join("\n\n"))
    },
  })

  const fetch_page = defineTool({
    name: "fetch_page",
    description:
      "Read a web page as text. Pass `focus` to keep only the passages about it. Use this before citing a page.",
    parameters: Type.Object({
      url: Type.String(),
      focus: Type.Optional(Type.String({ description: "What you are looking for on the page" })),
    }),
    replay: "safe",
    execute: async ({ url, focus }, api, context) => {
      const page = await run(fetchPage(url, focus ?? ""), context)
      await remember(api, context, [
        { url, title: page.title },
        { url: page.url, title: page.title },
      ])
      return text(`# ${page.title || page.url}\n${page.url}\n\n${page.text}`)
    },
  })

  const search_papers = defineTool({
    name: "search_papers",
    description: "Search scholarly papers on arXiv (preprints) and Crossref (published work).",
    parameters: Type.Object({ query: Type.String() }),
    replay: "safe",
    execute: async ({ query }, api, context) => {
      const papers = await run(searchPapers(query), context)
      await remember(api, context, papers)
      if (papers.length === 0) return text(`No papers found for "${query}".`)
      return text(
        papers
          .map((p, i) => `[${i + 1}] ${p.title} (${p.year || "n.d."}) ${p.authors}\n${p.url}\n${p.snippet}`)
          .join("\n\n"),
      )
    },
  })

  const list_dir = defineTool({
    name: "list_dir",
    description: `List a folder. Allowed folders: ${roots.join(", ")}`,
    parameters: Type.Object({ path: Type.String() }),
    replay: "safe",
    execute: async ({ path }) => {
      const dir = await allowed(path)
      const entries = await readdir(dir, { withFileTypes: true })
      const lines = entries
        .filter((e) => !e.name.startsWith("."))
        .slice(0, 200)
        .map((e) => (e.isDirectory() ? `${e.name}/` : e.name))
      return text(`${dir}\n${lines.join("\n") || "(empty)"}${entries.length > 200 ? "\n…" : ""}`)
    },
  })

  const read_file = defineTool({
    name: "read_file",
    description: "Read a text file from an allowed folder.",
    parameters: Type.Object({ path: Type.String() }),
    replay: "safe",
    execute: async ({ path }) => {
      const file = await allowed(path)
      if ((await stat(file)).size > 5_000_000) throw new Error("file is larger than 5 MB")
      const bytes = new Uint8Array(await Bun.file(file).arrayBuffer())
      if (bytes.subarray(0, 8000).includes(0)) throw new Error("not a text file")
      const body = new TextDecoder().decode(bytes)
      return text(
        body.length > 20_000 ? `${body.slice(0, 20_000)}\n… (${body.length - 20_000} more characters)` : body,
      )
    },
  })

  const search_files = defineTool({
    name: "search_files",
    description: "Find files whose name contains `name`, in the allowed folders or under `path`.",
    parameters: Type.Object({ name: Type.String(), path: Type.Optional(Type.String()) }),
    replay: "safe",
    execute: async ({ name, path }) => {
      const bases = path ? [await allowed(path)] : roots
      const needle = name.toLowerCase()
      const found: string[] = []
      for (const base of bases) {
        const glob = new Bun.Glob("**/*")
        for await (const file of glob.scan({ cwd: base, onlyFiles: true, followSymlinks: false })) {
          if (/(^|[\\/])(node_modules|\.git|AppData)[\\/]/.test(file)) continue
          if (basename(file).toLowerCase().includes(needle)) found.push(join(base, file))
          if (found.length >= 50) break
        }
      }
      return text(found.length ? found.join("\n") : `No files named like "${name}".`)
    },
  })

  const save_note = defineTool({
    name: "save_note",
    description: "Save a Markdown note to the user's notes folder. Never overwrites an existing note.",
    parameters: Type.Object({ title: Type.String(), content: Type.String() }),
    execute: async ({ title, content }) => {
      const dir = expandHome(config.files.notes)
      await mkdir(dir, { recursive: true })
      const slug =
        title
          .replace(/[^\p{L}\p{N} _-]+/gu, "")
          .trim()
          .slice(0, 80) || "note"
      let file = join(dir, `${slug}.md`)
      for (let n = 2; await Bun.file(file).exists(); n++) file = join(dir, `${slug} ${n}.md`)
      await Bun.write(file, `# ${title}\n\n${content}\n`)
      return text(`Saved ${file}`)
    },
  })

  const rememberFact = defineTool({
    name: "remember",
    description: "Remember a lasting fact or preference about the user for future chats.",
    parameters: Type.Object({ fact: Type.String() }),
    execute: async ({ fact }, api, context) => {
      await api.commit(async (tx) => {
        const memory = await tx.doc(Memory)
        if (!memory.facts.includes(fact)) memory.facts.push(fact)
      }, context)
      return text("Remembered.")
    },
  })

  const open = defineTool({
    name: "open",
    description: `Open an app, a website, a web URL, or a file on this computer. Apps: ${Object.keys(config.apps).join(", ")}. Sites: ${Object.keys(config.sites).join(", ")}. Pass \`query\` to search a site.`,
    parameters: Type.Object({ target: Type.String(), query: Type.Optional(Type.String()) }),
    execute: async ({ target, query }) => {
      const launch = await resolveOpen(target, query, config, allowed)
      const proc = Bun.spawn(opener(launch), { stdio: ["ignore", "ignore", "ignore"] })
      proc.unref()
      return text(`Opened ${launch.label}.`)
    },
  })

  return [
    web_search,
    fetch_page,
    search_papers,
    list_dir,
    read_file,
    search_files,
    save_note,
    rememberFact,
    open,
  ] as const
}

type Launch = { kind: "app" | "url" | "file"; value: string; label: string }

const find = (shortcuts: Record<string, Shortcut>, name: string) => {
  const key = name.trim().toLowerCase()
  return Object.entries(shortcuts).find(
    ([id, s]) => id === key || s.aliases.some((a) => a.toLowerCase() === key),
  )
}

export async function resolveOpen(
  target: string,
  query: string | undefined,
  config: Config,
  allowed: (path: string) => Promise<string>,
): Promise<Launch> {
  const app = find(config.apps, target)
  if (app?.[1].command) return { kind: "app", value: app[1].command, label: app[0] }
  const site = find(config.sites, target)
  if (site) {
    const [id, s] = site
    const url = query && s.search ? s.search.replace("{query}", encodeURIComponent(query)) : s.url
    return { kind: "url", value: url, label: query ? `${id} search for "${query}"` : id }
  }
  if (/^[a-z][a-z0-9+.-]*:/i.test(target) && !/^[a-z]:[\\/]/i.test(target)) {
    if (!/^https?:\/\//i.test(target)) throw new Error("only http and https links can be opened")
    return { kind: "url", value: new URL(target).href, label: target }
  }
  const file = await allowed(target)
  if (isBlockedFile(file)) throw new Error("programs and scripts can't be opened")
  return { kind: "file", value: file, label: file }
}

/**
 * The argv that opens `launch`. URLs and paths may come from a model, so on Windows they never pass through
 * `cmd` (where `&` would start a new command); only app commands from the user's own config do.
 */
export function opener({ kind, value }: Launch, platform = process.platform): string[] {
  if (platform === "win32") {
    return kind === "app"
      ? ["cmd", "/c", "start", "", value]
      : ["rundll32", "url.dll,FileProtocolHandler", value]
  }
  if (platform === "darwin") return kind === "app" ? ["open", "-a", value] : ["open", value]
  if (process.platform === "darwin") return kind === "app" ? ["open", "-a", value] : ["open", value]
  return kind === "app" ? value.split(" ").map((part) => expandHome(part)) : ["xdg-open", value]
}

export type LoraTool = ToolRegistration
