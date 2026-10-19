type Recognition = {
  lang: string
  interimResults: boolean
  onresult: ((event: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null
  onerror: ((event: { error: string }) => void) | null
  onend: (() => void) | null
  start(): void
  stop(): void
}

const SpeechRecognition =
  (
    globalThis as unknown as {
      SpeechRecognition?: new () => Recognition
      webkitSpeechRecognition?: new () => Recognition
    }
  ).SpeechRecognition ??
  (globalThis as unknown as { webkitSpeechRecognition?: new () => Recognition }).webkitSpeechRecognition

/** Microphones need a secure context: localhost, or HTTPS. A phone on plain HTTP over the LAN can't record. */
export const canListen = (serverStt: boolean) =>
  globalThis.isSecureContext &&
  (serverStt ? Boolean(navigator.mediaDevices?.getUserMedia) : Boolean(SpeechRecognition))

export type Listening = { stop(): void; text: Promise<string> }

/** Start listening; `stop()` ends the utterance and `text` resolves with what was said. */
export function listen(serverStt: boolean): Listening {
  return serverStt ? record() : recognize()
}

function recognize(): Listening {
  if (!SpeechRecognition) throw new Error("This browser has no speech recognition")
  const recognition = new SpeechRecognition()
  recognition.lang = navigator.language || "en-US"
  recognition.interimResults = false
  const text = new Promise<string>((resolve, reject) => {
    let heard = ""
    recognition.onresult = (event) => {
      heard = Array.from(event.results, (r) => r[0]?.transcript ?? "").join(" ")
    }
    recognition.onerror = (event) =>
      event.error === "no-speech" ? resolve("") : reject(new Error(event.error))
    recognition.onend = () => resolve(heard.trim())
  })
  recognition.start()
  return { stop: () => recognition.stop(), text }
}

function record(): Listening {
  let stop = () => {}
  const text = (async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    const recorder = new MediaRecorder(stream)
    const chunks: Blob[] = []
    recorder.ondataavailable = (e) => chunks.push(e.data)
    const stopped = new Promise((resolve) => (recorder.onstop = resolve))
    stop = () => recorder.state !== "inactive" && recorder.stop()
    recorder.start()
    await stopped
    for (const track of stream.getTracks()) track.stop()
    const audio = new Blob(chunks, { type: recorder.mimeType })
    const response = await fetch("/api/voice/transcribe", {
      method: "POST",
      body: audio,
      headers: { "content-type": audio.type },
    })
    if (!response.ok) throw new Error((await response.json()).error ?? "transcription failed")
    return ((await response.json()) as { text: string }).text
  })()
  return { stop: () => stop(), text }
}

let audio: HTMLAudioElement | undefined

/** Read an answer aloud, without the Markdown, links, or source list. */
export async function speak(markdown: string, serverTts: boolean, onEnd: () => void) {
  stopSpeaking()
  const text = forSpeech(markdown)
  if (!text) return onEnd()
  if (serverTts) {
    const response = await fetch("/api/voice/speak", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text }),
    })
    if (response.ok) {
      audio = new Audio(URL.createObjectURL(await response.blob()))
      audio.onended = onEnd
      await audio.play()
      return
    }
  }
  const utterance = new SpeechSynthesisUtterance(text)
  utterance.onend = onEnd
  utterance.onerror = onEnd
  speechSynthesis.speak(utterance)
}

export function stopSpeaking() {
  audio?.pause()
  audio = undefined
  globalThis.speechSynthesis?.cancel()
}

export const forSpeech = (markdown: string) =>
  markdown
    .split(/\n(?:#+\s*)?(?:sources|references)\s*:?\s*\n/i)[0]!
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/\[(\d+)\]/g, "")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/https?:\/\/\S+/g, "")
    .replace(/[*_`#>|]/g, "")
    .replace(/\s+/g, " ")
    .trim()
