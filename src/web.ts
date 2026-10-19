import { lookup } from "node:dns/promises"
import { Effect, Schema } from "effect"
import { focus, htmlToText, type Link, parseDuckDuckGo } from "./html.ts"
import { isPrivateAddress } from "./policy.ts"

export class WebError extends Schema.TaggedError<WebError>()("WebError", { message: Schema.String }) {}

const USER_AGENT = "Mozilla/5.0 (compatible; LORA/0.2; +https://github.com/SSB-Coder/L.o.r.a)"
const MAX_BYTES = 3_000_000

/** Refuse anything but public http(s) hosts, so a web page can't steer a fetch into this machine or the LAN. */
export const assertPublicUrl = Effect.fn("assertPublicUrl")(function* (raw: string) {
  const url = yield* Effect.try({
    try: () => new URL(raw),
    catch: () => new WebError({ message: `not a URL: ${raw}` }),
  })
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    return yield* new WebError({ message: `only http and https URLs can be fetched` })
  }
  if (url.username || url.password)
    return yield* new WebError({ message: "URLs with credentials are not allowed" })
  const host = url.hostname.replace(/^\[|\]$/g, "")
  const addresses = yield* Effect.tryPromise({
    try: () => lookup(host, { all: true }),
    catch: () => new WebError({ message: `could not resolve ${host}` }),
  })
  if (host === "localhost" || addresses.some((a) => isPrivateAddress(a.address))) {
    return yield* new WebError({ message: `${host} is a private address` })
  }
  return url
})

/** GET with every redirect hop re-checked, a timeout, and a size cap. */
export const get = Effect.fn("get")(function* (raw: string, accept = "text/html,*/*;q=0.8") {
  let url = yield* assertPublicUrl(raw)
  for (let hop = 0; hop < 5; hop++) {
    const target = url
    const response = yield* Effect.tryPromise({
      try: (signal) =>
        fetch(target, { redirect: "manual", signal, headers: { "user-agent": USER_AGENT, accept } }),
      catch: (e) => new WebError({ message: `${target.host}: ${e instanceof Error ? e.message : e}` }),
    }).pipe(
      Effect.timeoutOrElse({
        duration: "20 seconds",
        orElse: () => Effect.fail(new WebError({ message: `${target.host} timed out` })),
      }),
    )
    const location = response.headers.get("location")
    if (response.status >= 300 && response.status < 400 && location) {
      url = yield* assertPublicUrl(new URL(location, url).href)
      continue
    }
    if (!response.ok) return yield* new WebError({ message: `${url.host} answered ${response.status}` })
    const body = yield* Effect.promise(() => response.arrayBuffer())
    const text = new TextDecoder().decode(body.slice(0, MAX_BYTES))
    return { url: url.href, type: response.headers.get("content-type") ?? "", text }
  }
  return yield* new WebError({ message: "too many redirects" })
})

const json =
  <S extends Schema.Top & { readonly DecodingServices: never }>(schema: S) =>
  (text: string) =>
    Schema.decodeUnknownEffect(Schema.fromJsonString(schema))(text).pipe(
      Effect.mapError((e) => new WebError({ message: e.message })),
    )

export const webSearch = Effect.fn("webSearch")(function* (query: string, searxng: string) {
  if (searxng) {
    const { text } = yield* get(
      `${searxng.replace(/\/$/, "")}/search?format=json&q=${encodeURIComponent(query)}`,
    )
    const { results } = yield* json(SearxResults)(text)
    return results.slice(0, 8).map((r) => ({ title: r.title, url: r.url, snippet: r.content ?? "" }))
  }
  const ddg = yield* get(`https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`).pipe(
    Effect.flatMap(({ text }) => Effect.promise(() => parseDuckDuckGo(text))),
    Effect.orElseSucceed((): Link[] => []),
  )
  // DuckDuckGo sometimes serves a challenge page instead of results; Wikipedia always answers.
  return ddg.length > 0 ? ddg.slice(0, 8) : yield* wikipediaSearch(query)
})

const SearxResults = Schema.Struct({
  results: Schema.Array(
    Schema.Struct({ title: Schema.String, url: Schema.String, content: Schema.optional(Schema.String) }),
  ),
})

export const wikipediaSearch = Effect.fn("wikipediaSearch")(function* (query: string) {
  const { text } = yield* get(
    `https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=6&srsearch=${encodeURIComponent(query)}`,
    "application/json",
  )
  const data = yield* json(WikiResults)(text)
  return data.query.search.map((r) => ({
    title: r.title,
    url: `https://en.wikipedia.org/wiki/${encodeURIComponent(r.title.replaceAll(" ", "_"))}`,
    snippet: r.snippet.replace(/<[^>]+>/g, ""),
  }))
})

const WikiResults = Schema.Struct({
  query: Schema.Struct({
    search: Schema.Array(Schema.Struct({ title: Schema.String, snippet: Schema.String })),
  }),
})

export type Paper = Link & { year?: string; authors: string }

/** arXiv and Crossref together; either may fail without losing the other. */
export const searchPapers = Effect.fn("searchPapers")(function* (query: string) {
  const [arxiv, crossref] = yield* Effect.all(
    [
      arxivSearch(query).pipe(Effect.orElseSucceed(() => [])),
      crossrefSearch(query).pipe(Effect.orElseSucceed(() => [])),
    ],
    { concurrency: 2 },
  )
  return [...arxiv, ...crossref]
})

const arxivSearch = Effect.fn("arxivSearch")(function* (query: string) {
  const { text } = yield* get(
    `https://export.arxiv.org/api/query?max_results=5&search_query=all:${encodeURIComponent(query)}`,
    "application/atom+xml",
  )
  return [...text.matchAll(/<entry>([\s\S]*?)<\/entry>/g)].map(([, entry = ""]): Paper => {
    const tag = (name: string) =>
      entry
        .match(new RegExp(`<${name}[^>]*>([\\s\\S]*?)</${name}>`))?.[1]
        ?.replace(/\s+/g, " ")
        .trim() ?? ""
    return {
      title: tag("title"),
      url: tag("id"),
      snippet: `[preprint] ${tag("summary").slice(0, 400)}`,
      year: tag("published").slice(0, 4),
      authors: [...entry.matchAll(/<name>([^<]+)<\/name>/g)]
        .map((m) => m[1])
        .slice(0, 4)
        .join(", "),
    }
  })
})

const crossrefSearch = Effect.fn("crossrefSearch")(function* (query: string) {
  const { text } = yield* get(
    `https://api.crossref.org/works?rows=5&select=DOI,title,author,issued,container-title&query=${encodeURIComponent(query)}`,
    "application/json",
  )
  const { message } = yield* json(CrossrefResults)(text)
  return message.items.map(
    (item): Paper => ({
      title: item.title?.[0] ?? "(untitled)",
      url: `https://doi.org/${item.DOI}`,
      snippet: item["container-title"]?.[0] ?? "",
      year: String(item.issued?.["date-parts"]?.[0]?.[0] ?? ""),
      authors: (item.author ?? [])
        .slice(0, 4)
        .map((a) => [a.given, a.family].filter(Boolean).join(" "))
        .join(", "),
    }),
  )
})

const CrossrefResults = Schema.Struct({
  message: Schema.Struct({
    items: Schema.Array(
      Schema.Struct({
        DOI: Schema.String,
        title: Schema.optional(Schema.Array(Schema.String)),
        "container-title": Schema.optional(Schema.Array(Schema.String)),
        author: Schema.optional(
          Schema.Array(
            Schema.Struct({ given: Schema.optional(Schema.String), family: Schema.optional(Schema.String) }),
          ),
        ),
        issued: Schema.optional(
          Schema.Struct({
            "date-parts": Schema.optional(Schema.Array(Schema.Array(Schema.NullOr(Schema.Number)))),
          }),
        ),
      }),
    ),
  }),
})

/** A page as plain text, trimmed to the passages most relevant to `query`. */
export const fetchPage = Effect.fn("fetchPage")(function* (url: string, query: string, limit = 6000) {
  const page = yield* get(url)
  if (page.type.includes("pdf"))
    return yield* new WebError({ message: "PDFs can't be read yet; try the HTML version of the page" })
  const { title, text } = page.type.includes("html")
    ? yield* Effect.promise(() => htmlToText(page.text))
    : { title: "", text: page.text }
  return { url: page.url, title, text: focus(text, query, limit) }
})
