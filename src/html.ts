const SKIP = "head,script,style,noscript,nav,header,footer,aside,form,svg,iframe,template,button"
const BLOCK = "p,h1,h2,h3,h4,h5,h6,li,pre,blockquote,td,th,dd,dt,figcaption,div,section,article,main,tr"

/** Readable text of an HTML page: one line per block, chrome (nav, scripts, forms) dropped. */
export async function htmlToText(html: string): Promise<{ title: string; text: string }> {
  let skip = 0
  let title = ""
  let line = ""
  const lines: string[] = []
  const flush = () => {
    const text = decodeEntities(line).replace(/\s+/g, " ").trim()
    if (text && text !== lines.at(-1)) lines.push(text)
    line = ""
  }
  await new HTMLRewriter()
    .on("title", {
      text(t) {
        title += t.text
      },
    })
    .on(SKIP, {
      element(el) {
        skip++
        el.onEndTag(() => void skip--)
      },
    })
    .on(BLOCK, {
      element(el) {
        flush()
        el.onEndTag(flush)
      },
    })
    .on("br", { element: flush })
    .on("*", {
      text(t) {
        if (skip === 0) line += t.text
      },
    })
    .transform(new Response(html))
    .text()
  flush()
  return { title: decodeEntities(title).trim(), text: lines.join("\n") }
}

export type Link = { title: string; url: string; snippet: string }

/** Results from DuckDuckGo's HTML endpoint, with its redirect wrapper removed. */
export async function parseDuckDuckGo(html: string): Promise<Link[]> {
  const results: Link[] = []
  await new HTMLRewriter()
    .on("a.result__a", {
      element(el) {
        results.push({ title: "", url: unwrapDuckDuckGo(el.getAttribute("href") ?? ""), snippet: "" })
      },
      text(t) {
        const last = results.at(-1)
        if (last) last.title += t.text
      },
    })
    .on(".result__snippet", {
      text(t) {
        const last = results.at(-1)
        if (last) last.snippet += t.text
      },
    })
    .transform(new Response(html))
    .text()
  return results
    .map((r) => ({ url: r.url, title: clean(r.title), snippet: clean(r.snippet) }))
    .filter((r) => r.url.startsWith("http") && !r.url.includes("duckduckgo.com/y.js"))
}

function unwrapDuckDuckGo(href: string): string {
  try {
    const url = new URL(href, "https://duckduckgo.com")
    return url.searchParams.get("uddg") ?? url.href
  } catch {
    return ""
  }
}

const clean = (text: string) => decodeEntities(text).replace(/\s+/g, " ").trim()

const NAMED: Record<string, string> = {
  amp: "&",
  lt: "<",
  gt: ">",
  quot: '"',
  apos: "'",
  nbsp: " ",
  ndash: "–",
  mdash: "—",
  hellip: "…",
  deg: "°",
  copy: "©",
  reg: "®",
  trade: "™",
  times: "×",
  euro: "€",
  pound: "£",
  laquo: "«",
  raquo: "»",
  lsquo: "‘",
  rsquo: "’",
  ldquo: "“",
  rdquo: "”",
  middot: "·",
  bull: "•",
}

export const decodeEntities = (text: string) =>
  text.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (match, code: string) => {
    if (code[0] === "#") {
      const n =
        code[1] === "x" || code[1] === "X" ? Number.parseInt(code.slice(2), 16) : Number(code.slice(1))
      return Number.isFinite(n) && n > 0 && n <= 0x10ffff ? String.fromCodePoint(n) : match
    }
    return NAMED[code.toLowerCase()] ?? match
  })

/**
 * Keep the paragraphs that share the most words with `query`, in page order, up to `limit` characters.
 * Small local models do better with the relevant part of a page than with its first N characters.
 */
export function focus(text: string, query: string, limit: number): string {
  if (text.length <= limit) return text
  const terms = new Set(query.toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) ?? [])
  if (terms.size === 0) return `${text.slice(0, limit)}…`
  const paragraphs = text.split("\n").map((p, i) => ({ p, i, score: score(p, terms) }))
  const keep = new Set<number>()
  let used = 0
  for (const { p, i } of [...paragraphs].sort((a, b) => b.score - a.score || a.i - b.i)) {
    if (used + p.length > limit) continue
    keep.add(i)
    used += p.length + 1
  }
  return paragraphs
    .filter(({ i }) => keep.has(i))
    .map(({ p }) => p)
    .join("\n")
}

const score = (paragraph: string, terms: Set<string>) => {
  const words = paragraph.toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) ?? []
  return words.filter((w) => terms.has(w)).length / Math.sqrt(words.length + 1)
}
