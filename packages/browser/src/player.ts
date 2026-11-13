import { openSound, type PcmStream } from "./wav-stream.js";

/** A sound the app hands over: its bytes (the player decodes a copy), or the URL of the app's own route that answers one.
 * A route answering 16-bit mono PCM `audio/wav` plays while its body still arrives; every other sound plays whole. */
export type Sound = ArrayBuffer | string | URL;

export type PlaybackEnd =
  | { readonly kind: "played" }
  | { readonly kind: "stopped" } // stop() or close()
  | { readonly kind: "muted" } // setMuted(true), or a sound handed over while muted
  | { readonly kind: "refused" } // no user activation, or resume() refused
  | { readonly kind: "failed" }; // the fetch, the decode or a streamed body failed

export interface Playback {
  /** Resolves once and never rejects: "playing" when the sound's source starts on a running context, "ended" when it ended first. */
  readonly started: Promise<"playing" | "ended">;
  /** Resolves once and never rejects, with how the sound ended. A streamed body that breaks off ends "failed" where it broke. */
  readonly ended: Promise<PlaybackEnd>;
}

export interface Player {
  /** Queues the sound behind every earlier one and starts fetching and decoding it at once. Never throws. */
  play(sound: Sound): Playback;
  /** Ends the playing sound at once and every waiting one as "stopped", aborting their fetches; later sounds play. Idempotent; never throws. */
  stop(): void;
  /** The app's mute switch. Muting ends the playing and every waiting sound as "muted" and aborts their fetches; while muted, a sound handed over ends "muted" at once, unfetched. Idempotent; never throws. */
  setMuted(muted: boolean): void;
  /** Ends every sound as "stopped" and closes the player's context; every later sound ends "stopped" at once. Idempotent; never throws. */
  close(): void;
}

/**
 * Creates the player: its context and its one GainNode, the only input of the output.
 *
 * Every sound plays through that node, in the order of the `play()` calls. The player holds no token, sends no
 * header, stores nothing and builds no tap. It throws only if the browser cannot create an AudioContext.
 */
export function createPlayer(): Player {
  const run = new PlayerRun();
  return {
    play: (sound) => run.play(sound),
    stop: () => run.stop(),
    setMuted: (muted) => run.setMuted(muted),
    close: () => run.close(),
  };
}

const STOPPED: PlaybackEnd = { kind: "stopped" };
const MUTED: PlaybackEnd = { kind: "muted" };
const NONE = -1;

type Loaded =
  | { readonly kind: "buffer"; readonly buffer: AudioBuffer }
  | { readonly kind: "stream"; readonly stream: PcmStream };

/** The sources one sound has scheduled; a whole sound has one, a streamed sound one per chunk. */
class Scheduled {
  readonly sources = new Set<AudioBufferSourceNode>();
  bodyScheduled = false;
  constructor(readonly item: Queued) {}
}

/** Detaches the sources, so that their late `ended` neither reports nor pumps, and stops them at once. */
function silence(scheduled: Scheduled): void {
  for (const source of scheduled.sources) {
    source.onended = null;
    source.stop();
  }
  scheduled.sources.clear();
}

class Queued {
  readonly abort = new AbortController();
  readonly playback: Playback;
  loaded: Promise<Loaded | undefined> = Promise.resolve(undefined);
  readonly #started = Promise.withResolvers<"playing" | "ended">();
  readonly #ended = Promise.withResolvers<PlaybackEnd>();
  #over = false;

  constructor() {
    this.playback = { started: this.#started.promise, ended: this.#ended.promise };
  }

  begin(): void {
    this.#started.resolve("playing");
  }

  /** The first end wins. */
  end(end: PlaybackEnd): void {
    if (!this.#over) {
      this.#over = true;
      this.#started.resolve("ended");
      this.#ended.resolve(end);
    }
  }
}

class PlayerRun {
  readonly #context = new AudioContext();
  readonly #gain = this.#context.createGain();
  #queue: Queued[] = [];
  #playing: Scheduled | undefined;
  // Stop, close and mute bump the generation; a pump run goes on only in its own.
  #generation = 0;
  #pumping = NONE;
  #muted = false;
  #closed = false;

  constructor() {
    this.#gain.connect(this.#context.destination);
  }

  play(sound: Sound): Playback {
    const item = new Queued();
    if (this.#closed) {
      item.end(STOPPED);
    } else if (this.#muted) {
      item.end(MUTED);
    } else {
      item.loaded = this.#load(item, sound);
      this.#queue.push(item);
      void this.#pump();
    }
    return item.playback;
  }

  stop(): void {
    this.#endAll(STOPPED);
  }

  setMuted(muted: boolean): void {
    this.#muted = muted;
    this.#gain.gain.value = muted ? 0 : 1;
    if (muted) {
      this.#endAll(MUTED);
    }
  }

  close(): void {
    if (!this.#closed) {
      this.#closed = true;
      this.#endAll(STOPPED);
      void this.#context.close();
    }
  }

  async #load(item: Queued, sound: Sound): Promise<Loaded | undefined> {
    try {
      let bytes: ArrayBuffer;
      if (typeof sound === "string" || sound instanceof URL) {
        const response = await fetch(sound, { signal: item.abort.signal });
        if (!response.ok) {
          throw new Error(`the route answered ${response.status}`);
        }
        const opened = await openSound(response);
        if (opened.kind === "stream") {
          return opened;
        }
        bytes = opened.bytes;
      } else {
        bytes = sound.slice(0);
      }
      return { kind: "buffer", buffer: await this.#context.decodeAudioData(bytes) };
    } catch {
      item.end({ kind: "failed" });
      return undefined;
    }
  }

  async #pump(): Promise<void> {
    if (this.#playing !== undefined || this.#pumping === this.#generation) {
      return;
    }
    const generation = this.#generation;
    this.#pumping = generation;
    try {
      for (let head = this.#queue[0]; head !== undefined; head = this.#queue[0]) {
        const loaded = await head.loaded;
        if (!this.#isHead(generation, head)) {
          return;
        }
        const running = loaded !== undefined && (await this.#resume());
        if (!this.#isHead(generation, head)) {
          return;
        }
        if (running) {
          this.#queue.shift();
          const scheduled = new Scheduled(head);
          this.#playing = scheduled;
          if (loaded.kind === "buffer") {
            this.#schedule(scheduled, loaded.buffer, 0);
            scheduled.bodyScheduled = true;
          } else {
            void this.#playStream(scheduled, loaded.stream, generation);
          }
          return;
        }
        // A failed sound has ended already; a refused one ends here. Neither blocks the next.
        head.end({ kind: "refused" });
        this.#queue.shift();
      }
    } finally {
      if (this.#pumping === generation) {
        this.#pumping = NONE;
      }
    }
  }

  #isHead(generation: number, item: Queued): boolean {
    return generation === this.#generation && this.#queue[0] === item;
  }

  async #resume(): Promise<boolean> {
    if (this.#context.state === "running") {
      return true;
    }
    // Without sticky activation a suspended context's resume() would stay pending forever.
    if (this.#context.state === "suspended" && !navigator.userActivation.hasBeenActive) {
      return false;
    }
    try {
      await this.#context.resume();
      return true;
    } catch {
      return false;
    }
  }

  /** Schedules the buffer on the one gain at `when` of the context's clock. */
  #schedule(scheduled: Scheduled, buffer: AudioBuffer, when: number): void {
    const source = this.#context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.#gain);
    source.onended = () => {
      scheduled.sources.delete(source);
      this.#settle(scheduled);
    };
    scheduled.sources.add(source);
    source.start(when);
    scheduled.item.begin();
  }

  /** Back to back on the context's clock: each chunk starts where the one before ends. */
  async #playStream(scheduled: Scheduled, stream: PcmStream, generation: number): Promise<void> {
    let nextAt = 0;
    try {
      for (let samples = await stream.next(); samples !== undefined; samples = await stream.next()) {
        if (generation !== this.#generation) {
          return;
        }
        const buffer = this.#context.createBuffer(1, samples.length, stream.sampleRate);
        buffer.copyToChannel(samples, 0);
        nextAt = Math.max(nextAt, this.#context.currentTime);
        this.#schedule(scheduled, buffer, nextAt);
        nextAt += buffer.duration;
      }
    } catch {
      if (generation === this.#generation) {
        this.#playing = undefined;
        silence(scheduled);
        scheduled.item.end({ kind: "failed" });
        void this.#pump();
      }
      return;
    }
    if (generation === this.#generation) {
      scheduled.bodyScheduled = true;
      this.#settle(scheduled);
    }
  }

  #settle(scheduled: Scheduled): void {
    if (scheduled.bodyScheduled && scheduled.sources.size === 0 && this.#playing === scheduled) {
      this.#playing = undefined;
      scheduled.item.end({ kind: "played" });
      void this.#pump();
    }
  }

  /** Silences the playing sound first, so that its late `ended` neither reports nor pumps. */
  #endAll(end: PlaybackEnd): void {
    this.#generation += 1;
    const playing = this.#playing;
    const waiting = this.#queue;
    this.#playing = undefined;
    this.#queue = [];
    if (playing !== undefined) {
      silence(playing);
      playing.item.abort.abort();
      playing.item.end(end);
    }
    for (const item of waiting) {
      item.abort.abort();
      item.end(end);
    }
  }
}
