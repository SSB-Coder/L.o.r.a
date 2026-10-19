import pkg from "../package.json" with { type: "json" }

/** Release builds set LORA_VERSION (e.g. 0.2.0-nightly.57); everything else reports package.json. */
export const VERSION: string = process.env.LORA_VERSION ?? pkg.version
