import { endOf, playedBy, type Chunk } from "./timeline.js";
import type { Held, Playback, PlaybackEnd, PlaybackStart } from "./player.js";
import type { PcmStream } from "./wav-stream.js";

export type Loaded =
  | { readonly kind: "buffer"; readonly buffer: AudioBuffer }
  | { readonly kind: "stream"; readonly stream: PcmStream };

/** Where a held sound was held: the context time its fade to silence ends and, when the player knows it ahead, the seconds read there. */
interface HoldPoint {
  readonly at: number;
  readonly seconds?: number;
}

/** The sources one sound has scheduled, each with the context time it starts at; a whole sound has one, a streamed sound one per chunk. */
export class Scheduled {
  readonly sources = new Map<AudioBufferSourceNode, number>();
  readonly chunks: Chunk[] = [];
  bodyScheduled = false;
  /** The context time at which a `stop()` ends the sound, once it has asked for it. */
  stoppingAt: number | undefined;
  /** Set while the sound is held, or a fade to a hold is on the gain's timeline. */
  held: HoldPoint | undefined;
  constructor(readonly item: Queued) {}

  /** The seconds the output has played by `now` of the context's clock, no further than a stop's edge and the hold point. */
  read(now: number): number {
    const until = Math.min(now, this.stoppingAt ?? Infinity);
    if (this.held !== undefined && until >= this.held.at) {
      return this.held.seconds ?? playedBy(this.chunks, this.held.at);
    }
    return playedBy(this.chunks, until);
  }

  /** Whether the output has fallen silent at the hold point by `now` of the context's clock. */
  heldBy(now: number): boolean {
    return this.held !== undefined && now >= this.held.at;
  }

  /** Whether the sound's sources run out after its hold point, so that a late page can end it before `resume()`. */
  runsOutHeld(): boolean {
    return this.held !== undefined && this.held.at < endOf(this.chunks);
  }
}

/** Detaches the sources, so that their late `ended` neither reports nor pumps, and stops them at once. */
export function silence(scheduled: Scheduled): void {
  for (const source of scheduled.sources.keys()) {
    source.onended = null;
    source.stop();
  }
  scheduled.sources.clear();
}

/** The queued sound behind each playback this module handed out. */
export const queuedOf = new WeakMap<Playback, Queued>();

export class Queued {
  readonly abort = new AbortController();
  readonly playback: Playback;
  /** Who is told, once and at once, how the sound ended. */
  onEnd: ((end: PlaybackEnd) => void) | undefined;
  loaded: Promise<Loaded | undefined> = Promise.resolve(undefined);
  readonly #started = Promise.withResolvers<PlaybackStart>();
  readonly #ended = Promise.withResolvers<PlaybackEnd>();
  #over = false;
  #end: PlaybackEnd | undefined;
  #playedSeconds = (): number => 0;

  constructor(readonly owner: object) {
    this.playback = {
      started: this.#started.promise,
      ended: this.#ended.promise,
      playedSeconds: () => this.#playedSeconds(),
    };
    queuedOf.set(this.playback, this);
  }

  /** How the sound ended, once it has. */
  get endedAs(): PlaybackEnd | undefined {
    return this.#end;
  }

  /** From now on, `playback.playedSeconds()` reads `playedSeconds`, until the sound ends. */
  follow(playedSeconds: () => number): void {
    if (!this.#over) {
      this.#playedSeconds = playedSeconds;
    }
  }

  /** Only the first call counts: a streamed sound begins with its first chunk. */
  begin(contextTime: number, context: AudioContext): void {
    this.#started.resolve({ kind: "playing", contextTime, context });
  }

  /** The first end wins. */
  end(end: PlaybackEnd): void {
    if (!this.#over) {
      this.#over = true;
      this.#end = end;
      // A failed end carries no seconds: the sound keeps what it had played, 0 when it never started.
      const played = "playedSeconds" in end ? end.playedSeconds : this.#playedSeconds();
      this.#playedSeconds = () => played;
      this.#started.resolve({ kind: "ended" });
      this.#ended.resolve(end);
      this.onEnd?.(end);
    }
  }
}

/** A `pauseAt` waiting for its hold, with the call's result once it has one. */
export class HoldCall {
  readonly result = Promise.withResolvers<Held>();
  /** The fade to the position is on the gain's timeline. */
  scheduled = false;
  /** The target ended on its own: the hold comes at the next point. */
  afterEnd = false;
  constructor(
    readonly item: Queued,
    readonly position: number,
  ) {}
}
