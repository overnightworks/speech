/** A whole sound the app hands over: its bytes (the player decodes a copy), or the URL of the app's own route that answers one. */
export type Sound = ArrayBuffer | string | URL;

export type PlaybackEnd =
  | { readonly kind: "played" }
  | { readonly kind: "stopped" } // stop() or close()
  | { readonly kind: "muted" } // setMuted(true), or a sound handed over while muted
  | { readonly kind: "refused" } // no user activation, or resume() refused
  | { readonly kind: "failed" }; // the fetch or the decode failed

export interface Playback {
  /** Resolves once and never rejects: "playing" when the sound's source starts on a running context, "ended" when it ended first. */
  readonly started: Promise<"playing" | "ended">;
  /** Resolves once and never rejects, with how the sound ended. */
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

class Queued {
  readonly abort = new AbortController();
  readonly playback: Playback;
  buffer: Promise<AudioBuffer | undefined> = Promise.resolve(undefined);
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
  #playing: { readonly item: Queued; readonly source: AudioBufferSourceNode } | undefined;
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
      item.buffer = this.#load(item, sound);
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

  async #load(item: Queued, sound: Sound): Promise<AudioBuffer | undefined> {
    try {
      let bytes: ArrayBuffer;
      if (typeof sound === "string" || sound instanceof URL) {
        const response = await fetch(sound, { signal: item.abort.signal });
        if (!response.ok) {
          throw new Error(`the route answered ${response.status}`);
        }
        bytes = await response.arrayBuffer();
      } else {
        bytes = sound.slice(0);
      }
      return await this.#context.decodeAudioData(bytes);
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
        const buffer = await head.buffer;
        if (!this.#isHead(generation, head)) {
          return;
        }
        const running = buffer !== undefined && (await this.#resume());
        if (!this.#isHead(generation, head)) {
          return;
        }
        if (running) {
          this.#startSource(head, buffer);
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

  #startSource(item: Queued, buffer: AudioBuffer): void {
    this.#queue.shift();
    const source = this.#context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.#gain);
    source.onended = () => {
      this.#playing = undefined;
      item.end({ kind: "played" });
      void this.#pump();
    };
    this.#playing = { item, source };
    source.start();
    item.begin();
  }

  /** Detaches the playing source first, so that its late `ended` neither reports nor pumps. */
  #endAll(end: PlaybackEnd): void {
    this.#generation += 1;
    const playing = this.#playing;
    const waiting = this.#queue;
    this.#playing = undefined;
    this.#queue = [];
    if (playing !== undefined) {
      playing.source.onended = null;
      playing.source.stop();
      playing.item.end(end);
    }
    for (const item of waiting) {
      item.abort.abort();
      item.end(end);
    }
  }
}
