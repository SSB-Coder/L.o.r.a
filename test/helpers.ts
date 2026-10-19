import { mkdtempSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { BACKGROUND_CONTEXT } from "@earendil-works/chord/context"
import { createModels, fauxProvider } from "@earendil-works/pi-ai"
import { MemoryStorage } from "@earendil-works/pi-durable"
import { makeAgent } from "../src/agent.ts"
import { decodeConfig } from "../src/config.ts"
import { PROVIDER } from "../src/models.ts"

export const ctx = BACKGROUND_CONTEXT

/** An agent on in-memory storage whose model answers with scripted faux responses. */
export async function testAgent(overrides: object = {}) {
  const root = mkdtempSync(join(tmpdir(), "lora-test-"))
  const config = decodeConfig({ files: { roots: [root], notes: join(root, "notes") }, ...overrides })
  const faux = fauxProvider({ provider: PROVIDER, models: [{ id: "faux" }] })
  const models = createModels()
  models.setProvider(faux.provider)
  const agent = await makeAgent(config, models, new MemoryStorage(), () => "faux")
  return { agent, faux, config, root }
}
