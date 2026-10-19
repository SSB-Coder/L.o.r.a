import { expect, test } from "bun:test"
import { decodeEntities, focus, htmlToText, parseDuckDuckGo } from "../src/html.ts"

test("page text keeps the article and drops scripts, navigation, and forms", async () => {
  const { title, text } =
    await htmlToText(`<html><head><title>Tea &amp; Coffee</title><script>evil()</script></head>
    <body><nav>Home | About</nav><article><h1>Brewing</h1><p>Steep green tea at 80&deg;C&nbsp;for <b>2</b> minutes.</p>
    <ul><li>One</li><li>Two</li></ul></article><form><button>Subscribe</button></form><footer>©</footer></body></html>`)
  expect(title).toBe("Tea & Coffee")
  expect(text).toBe("Brewing\nSteep green tea at 80°C for 2 minutes.\nOne\nTwo")
})

test("DuckDuckGo results are unwrapped from its redirect links", async () => {
  const results = await parseDuckDuckGo(`
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&rut=x">Example <b>A</b></a>
    <a class="result__snippet">About &quot;A&quot;</a>
    <a class="result__a" href="https://duckduckgo.com/y.js?ad=1">Ad</a>`)
  expect(results).toEqual([{ url: "https://example.com/a", title: "Example A", snippet: 'About "A"' }])
})

test("entities decode, and focus keeps the passages about the query in page order", () => {
  expect(decodeEntities("&#39;&#x41;&lt;&unknown;")).toBe("'A<&unknown;")
  const text = [
    "intro filler words here",
    "ollama runs local models",
    "unrelated cooking tips",
    "models need memory",
  ].join("\n")
  expect(focus(text, "ollama models", 50)).toBe("ollama runs local models\nmodels need memory")
})
