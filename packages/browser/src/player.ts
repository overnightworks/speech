import { classify, contextTimeAt, playedBy, type Chunk } from "./timeline.js";
import { openSound, type PcmStream } from "./wav-stream.js";

/** A sound the app hands over: its bytes (the player decodes a copy), or the URL of the app's own route that answers one.
 * A route answering 16-bit mono PCM `audio/wav` plays while its body still arrives; every other sound plays whole. */
export type Sound = ArrayBuffer | string | URL;

export type PlaybackEnd =
  /** `playedSeconds`: the seconds of the sound the output played, read from the player's AudioContext clock and never more than the sound's length. After a page stall around the start call (see `PlaybackStart.contextTime`) it reads ahead of what was heard by up to the blocked time. A "played" sound played its full length; one that never started played 0. */
  | { readonly kind: "played"; readonly playedSeconds: number }
  | { readonly kind: "stopped"; readonly playedSeconds: number } // stop() or close(); a stop() ends the playing sound once its edge is over
  | { readonly kind: "muted"; readonly playedSeconds: number } // setMuted(true), or a sound handed over while muted
  | { readonly kind: "refused" } // no user activation, or resume() refused
  | { readonly kind: "failed" }; // the fetch, the decode or a streamed body failed

export type PlaybackStart =
  /** `contextTime`: the time on `context`'s clock (`currentTime`) at which the sound's first sample plays, to within one render quantum, unless the page's task is blocked around the player's start call for longer than the output's lead (the larger of `baseLatency` and one render quantum); then the first sample plays later than `contextTime`, by up to the time the task was blocked. It places only the sound's first sample: a streamed sound's later chunk starts when it arrives if that is later than the one before ends, so the start plus a timing runs early after a stall. `Playback.playedSeconds()` reads how far the sound has played. `context` is the AudioContext the sound plays on. */
  | { readonly kind: "playing"; readonly contextTime: number; readonly context: AudioContext }
  | { readonly kind: "ended" }; // the sound ended first, so it has no start

/**
 * How `Player.pauseAt` ended. It resolves once, when the player is held (the suspend has landed) or at once when no hold comes:
 * - `held`: the player holds, silent, and `playedSeconds` is the sound's read at the hold point: the position the call gave
 *   when it was ahead, else where the hold landed. It is what `Playback.playedSeconds()` reads while held and what a
 *   `stop()`, `setMuted(true)` or `close()` after the hold reports;
 * - `held_after_end`: the target ended on its own before the position (it ended, failed or was refused, or its stream
 *   ended or failed first, or the position was at its end), and the player holds at the next point: before the next
 *   waiting sound, or at once inside the next sound when that already plays;
 * - `overtaken`: the hold the call asked for will not come: a newer `pauseAt`, `resume()`, `stop()`, `setMuted(true)` or
 *   `close()` came first, the target was stopped or muted, a `stop()` of it is in flight, or it is not this player's.
 */
export type Held =
  | { readonly kind: "held"; readonly playedSeconds: number }
  | { readonly kind: "held_after_end" }
  | { readonly kind: "overtaken" };

export interface Playback {
  /** Resolves once and never rejects: "playing" with the sound's start when its source starts on a running context, "ended" when it ended first. */
  readonly started: Promise<PlaybackStart>;
  /** Resolves once and never rejects, with how the sound ended. A streamed body that breaks off ends "failed" where it broke. */
  readonly ended: Promise<PlaybackEnd>;
  /**
   * The seconds of the sound the output has played by now, on the same clock and unit as `PlaybackEnd.playedSeconds`, so
   * an app adds no start to it. After a page stall around the start call (see `PlaybackStart.contextTime`) it reads ahead of what was heard by up to the blocked time. It is 0 before the sound starts, stands still while the output waits for a late chunk
   * and while paused, and reads the end's `playedSeconds` once the sound has ended (0 for a refused one, and for a failed one that never started; a failed one that played keeps what it played). While the sound is held (`pause()` or `pauseAt()`) it reads
   * the hold point: where the volume's fade to silence ended, which is the position a `pauseAt` gave when it was ahead. The
   * context runs on silently until the suspend lands, and the read does not follow it; after `resume()` it jumps forward by
   * that run-on and never steps back. Never throws.
   */
  playedSeconds(): number;
}

export interface Player {
  /**
   * The player's one GainNode, the only input of the output and alive from `createPlayer()` to `close()`. An app reads
   * a sentence's energy by connecting an analyser of this node's own context to it. Mute silences it. While paused or
   * after close the context stops, so an analyser holds its last reading: key on the player's state there, not on the
   * reading. Calling `disconnect()` on it without arguments cuts the player's output.
   */
  readonly tap: AudioNode;
  /** Queues the sound behind every earlier one and starts fetching and decoding it at once. Never throws. */
  play(sound: Sound): Playback;
  /**
   * Holds the output at once and the sounds where they are: the volume ramps to 0 over a click-free edge of a few tens
   * of milliseconds, then the context suspends. The context's clock, every scheduled sound and each start the app read
   * stay valid, a body that arrives meanwhile queues on the frozen clock, and no waiting sound starts until `resume()`; a streamed sound's first samples that arrive meanwhile report its start at `resume()`.
   * The sound that plays is held at the fade's end, its hold point (see `Playback.playedSeconds`): once that has passed
   * it does not end "played" before `resume()`, even if a late page lets its sources run out meanwhile, and no next sound
   * starts. Then `resume()` ends it "played" and starts the next, and `stop()`, `setMuted(true)` or `close()` end it at
   * the hold point. Idempotent; ignored while muted or closed; never throws.
   */
  pause(): void;
  /**
   * Holds the playing sound once its played position reaches `seconds`, as `pause()` holds: the volume ramps to 0 over
   * the same edge, ending exactly at the position on the audio clock, then the context suspends; the call resolves when
   * the suspend lands (see `Held` for each outcome). The player holds before the next waiting sound; a later
   * `resume()` plays the rest faded in, then the queue, and a later `stop()` ends the sound "stopped" at the hold point.
   * Never rejects and never throws.
   *
   * The clock is the sound's own seconds, the unit of `Playback.playedSeconds`, counted from its first sample; a word
   * mark of a speak answer counts from the answer's first sample across its sentences, so an app that plays one part of
   * an answer subtracts that part's start. A position inside a streamed sound's late chunk is its content time, not the
   * clock time the chunk's gap takes.
   * - A position well ahead holds at it: `held`, with `playedSeconds` equal to it.
   * - A position already passed, closer than one edge, negative or NaN holds at once: `held`, with where it landed.
   * - A position at or past the sound's length, within one edge of its end, or `Infinity` lets the sound end "played"
   *   and holds before the next waiting sound, which does not start: `held_after_end`. The end wins over a passed position.
   * - A position beyond the chunks that have arrived waits for them; a stream that ends or fails before it gives
   *   `held_after_end`.
   * - The target may be queued (the position counts once it plays, also while the player is paused, where the call stays
   *   pending through `resume()`); one that ended "played", "failed" or "refused" holds the sound that plays now (or the
   *   next point) and gives `held_after_end`; one that ended "stopped" or "muted", one whose `stop()` is in flight, and a
   *   playback of another player give `overtaken`.
   * - A newer `pauseAt` replaces a pending one; `resume()` cancels the pending call's fade and plays on; `stop()`,
   *   `setMuted(true)` and `close()` end the call; all give `overtaken`. `pause()` while a call waits on the playing sound
   *   holds now and the call gives `held` with where it held; a call for the held sound while paused or pausing gives
   *   `held` with the hold point's read.
   *
   * A `resume()` after the hold loses the content the context ran through, silent, between the position and the suspend:
   * at most as long as the page is late, faded in. A fade that has begun when `resume()` or `stop()` comes continues from
   * the volume it has reached.
   */
  pauseAt(playback: Playback, seconds: number): Promise<Held>;
  /**
   * Resumes the context and ramps the volume back up over the same edge: the sounds play on where they paused, and the
   * read of a held sound jumps forward by the silent run-on of a late page and never steps back. A held sound that ran out
   * meanwhile ends "played" now, and the queue plays on. Idempotent; never throws.
   */
  resume(): void;
  /**
   * Ramps the volume to 0 over the same edge, ends the playing sound as "stopped" when it is over and every waiting
   * one at once, aborting their fetches; later sounds play. A pause stays until `resume()`: while paused, the playing
   * sound ends at once, because the output is held already, at its hold point (see `Playback.playedSeconds`). Idempotent; never throws.
   */
  stop(): void;
  /** The app's mute switch. Muting ends the playing and every waiting sound as "muted" at once and aborts their fetches, and clears a pause; while muted, a sound handed over ends "muted" at once, unfetched. Idempotent; never throws. */
  setMuted(muted: boolean): void;
  /** Ends every sound as "stopped" at once and closes the player's context; every later sound ends "stopped" at once. Idempotent; never throws. */
  close(): void;
}

/**
 * Creates the player: its context and its one GainNode, the only input of the output.
 *
 * Every sound plays through that node, in the order of the `play()` calls. The player holds no token, sends no
 * header, stores nothing and builds no node for a reader: `tap` is the one GainNode. It throws only if the browser
 * cannot create an AudioContext.
 */
export function createPlayer(): Player {
  const run = new PlayerRun();
  return {
    tap: run.tap,
    play: (sound) => run.play(sound),
    pause: () => run.pause(),
    pauseAt: (playback, seconds) => run.pauseAt(playback, seconds),
    resume: () => run.resume(),
    stop: () => run.stop(),
    setMuted: (muted) => run.setMuted(muted),
    close: () => run.close(),
  };
}

type Interruption = "stopped" | "muted";

/** "pausing": the volume ramps to 0 and the context is not yet suspended; "paused": the context suspends or is suspended. */
type Flow = "running" | "pausing" | "paused";

/** The ramp the gain is on: where it ends up, and when on the context's clock it starts and ends. */
interface Ramp {
  readonly target: number;
  readonly startsAt: number;
  readonly endsAt: number;
}

const NONE = -1;
/**
 * The edge of every volume change but mute and close: a step of the volume is a click, a linear ramp of this length is
 * not heard as one, and it is short enough to read as "at once".
 */
const FADE_SECONDS = 0.03;
const MILLISECONDS_PER_SECOND = 1000;
/** The Web Audio render quantum; the audio graph advances in steps of this many frames. */
const RENDER_QUANTUM_FRAMES = 128;

type Loaded =
  | { readonly kind: "buffer"; readonly buffer: AudioBuffer }
  | { readonly kind: "stream"; readonly stream: PcmStream };

/** Where a held sound was held: the context time its fade to silence ends and, when the player knows it ahead, the seconds read there. */
interface HoldPoint {
  readonly at: number;
  readonly seconds?: number;
}

/** The sources one sound has scheduled, each with the context time it starts at; a whole sound has one, a streamed sound one per chunk. */
class Scheduled {
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

  /** Whether the sound's sources run out after its hold point, so that a late page can end it before `resume()`. */
  runsOutHeld(): boolean {
    return this.held !== undefined && this.held.at < Math.max(...this.chunks.map(({ start, seconds }) => start + seconds));
  }
}

/** Detaches the sources, so that their late `ended` neither reports nor pumps, and stops them at once. */
function silence(scheduled: Scheduled): void {
  for (const source of scheduled.sources.keys()) {
    source.onended = null;
    source.stop();
  }
  scheduled.sources.clear();
}

/** The queued sound behind each playback this module handed out. */
const queuedOf = new WeakMap<Playback, Queued>();

class Queued {
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

  constructor(readonly owner: PlayerRun) {
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
class HoldCall {
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
  #flow: Flow = "running";
  #ramp: Ramp | undefined;
  // The latest resume of the context, in flight or done.
  #resuming: Promise<boolean> = Promise.resolve(true);
  // The context time the volume is back to 1 from: no sound starts under the stop's ramp before it.
  #restoredAt = 0;
  // Every call that sets the volume bumps the fade; a wait that began in an older one is overtaken.
  #fade = 0;
  // The one pauseAt waiting for its hold: a newer call replaces it.
  #call: HoldCall | undefined;

  constructor() {
    this.#gain.connect(this.#context.destination);
  }

  get tap(): AudioNode {
    return this.#gain;
  }

  play(sound: Sound): Playback {
    const item = new Queued(this);
    if (this.#closed) {
      item.end({ kind: "stopped", playedSeconds: 0 });
    } else if (this.#muted) {
      item.end({ kind: "muted", playedSeconds: 0 });
    } else {
      item.loaded = this.#load(item, sound);
      this.#queue.push(item);
      void this.#pump();
    }
    return item.playback;
  }

  pause(): void {
    if (this.#closed || this.#muted || this.#flow !== "running") {
      return;
    }
    this.#holdNow();
  }

  pauseAt(playback: Playback, seconds: number): Promise<Held> {
    const item = queuedOf.get(playback);
    if (item?.owner !== this) {
      return Promise.resolve({ kind: "overtaken" });
    }
    if (this.#call !== undefined) {
      this.#overtake(this.#call, { restoreVolume: true });
    }
    const call = new HoldCall(item, seconds);
    this.#call = call;
    item.onEnd = (end) => this.#targetEnded(call, end);
    this.#advanceCall();
    return call.result.promise;
  }

  resume(): void {
    if (this.#closed) {
      return;
    }
    if (this.#call !== undefined && this.#call.item === this.#playing?.item) {
      this.#overtake(this.#call, { restoreVolume: true });
    }
    if (this.#flow === "running") {
      return;
    }
    if (this.#playing !== undefined) {
      this.#playing.held = undefined;
    }
    this.#flow = "running";
    void this.#resumeAndFadeIn(++this.#fade);
  }

  stop(): void {
    this.#endAll("stopped", { instant: false });
  }

  setMuted(muted: boolean): void {
    const wasMuted = this.#muted;
    this.#muted = muted;
    if (muted) {
      this.#flow = "running";
      this.#setVolume(0);
      this.#endAll("muted", { instant: true });
    } else if (wasMuted) {
      this.#setVolume(1);
    }
  }

  close(): void {
    if (!this.#closed) {
      this.#closed = true;
      this.#fade += 1;
      this.#endAll("stopped", { instant: true });
      void this.#context.close();
    }
  }

  /** The volume at once, for mute: no ramp, and no older ramp left to override it. */
  #setVolume(volume: number): void {
    this.#fade += 1;
    this.#ramp = undefined;
    this.#gain.gain.cancelScheduledValues(this.#context.currentTime);
    this.#gain.gain.value = volume;
  }

  /**
   * The volume back to 1 from `time` on, the end of a stop's edge. The stopped source's `ended` can reach the page
   * while the render thread is still on the quantum that holds the source's last samples; a volume set at once would
   * apply to them as a click, one set on the timeline at the ramp's end leaves them under the ramp.
   */
  #restoreVolumeAt(time: number): void {
    this.#fade += 1;
    this.#ramp = undefined;
    this.#restoredAt = time;
    this.#gain.gain.setValueAtTime(1, time);
  }

  /**
   * Ramps the gain from where it is to `target` and returns the context time the ramp ends at. A ramp already on its
   * way to `target` stays, so that a sound stopped at the end of it is cut where the volume is 0; one still ahead does not
   * count, so a `pause()` or `stop()` while a `pauseAt` waits gets its own edge. Firefox lacks
   * cancelAndHoldAtTime, so the ramp starts from the value the gain reads.
   */
  #rampTo(target: number): number {
    const now = this.#context.currentTime;
    if (this.#ramp?.target === target && this.#ramp.startsAt <= now && this.#ramp.endsAt > now) {
      return this.#ramp.endsAt;
    }
    const param = this.#gain.gain;
    const from = param.value;
    const endsAt = now + FADE_SECONDS;
    param.cancelScheduledValues(now);
    param.setValueAtTime(from, now);
    param.linearRampToValueAtTime(target, endsAt);
    this.#ramp = { target, startsAt: now, endsAt };
    return endsAt;
  }

  /** Waits on the context's clock, where the page's timers lag, until it passes `time`; false if a later fade overtook it. */
  async #untilContextTime(time: number, fade: number): Promise<boolean> {
    if (fade !== this.#fade) {
      return false;
    }
    const seconds = time - this.#context.currentTime;
    if (this.#context.state !== "running" || seconds <= 0) {
      return true;
    }
    await new Promise((resolve) => setTimeout(resolve, seconds * MILLISECONDS_PER_SECOND));
    return this.#untilContextTime(time, fade);
  }

  /** Ramps the volume up, unless a sound being stopped is to bring it back when it ends. */
  #fadeIn(): void {
    if (this.#playing?.stoppingAt === undefined) {
      this.#rampTo(1);
    }
  }

  async #suspendOnceFaded(endsAt: number, fade: number): Promise<void> {
    // A suspend asked of a context that is still to run again is lost: the resume then runs it on, with the volume at 0.
    await this.#resuming;
    if (await this.#untilContextTime(endsAt, fade)) {
      this.#flow = "paused";
      await this.#context.suspend();
      if (this.#flow === "paused") {
        this.#landed();
      }
    }
  }

  /**
   * The one hold: the volume ramps to 0 over the edge, the sound that plays is held at the ramp's end, and the context
   * suspends there. Every hold records its hold point, so the read while held and the end a late page leaves are the
   * point's, not the suspended clock's.
   */
  #holdNow(): void {
    this.#flow = "pausing";
    const endsAt = this.#rampTo(0);
    if (this.#playing !== undefined) {
      this.#playing.held = { at: endsAt };
    }
    if (this.#call !== undefined) {
      this.#call.scheduled = false;
    }
    void this.#suspendOnceFaded(endsAt, ++this.#fade);
  }

  /** The player is suspended: the call on the sound that plays is held where the sound was held, one after an end is held after it. */
  #landed(): void {
    const call = this.#call;
    if (call === undefined) {
      return;
    }
    const playing = this.#playing;
    if (playing?.item === call.item) {
      this.#settleCall(call, { kind: "held", playedSeconds: playing.read(this.#context.currentTime) });
    } else if (call.afterEnd) {
      this.#settleCall(call, { kind: "held_after_end" });
    }
  }

  #settleCall(call: HoldCall, held: Held): void {
    if (this.#call === call) {
      this.#call = undefined;
    }
    call.item.onEnd = undefined;
    call.result.resolve(held);
  }

  /**
   * Ends the waiting call, and takes its fade off the gain's timeline if it is on it, ramping the volume back up unless
   * the switch that overtakes it sets the volume itself.
   */
  #overtake(call: HoldCall, { restoreVolume }: { readonly restoreVolume: boolean }): void {
    if (call.scheduled) {
      call.scheduled = false;
      if (this.#playing?.item === call.item) {
        this.#playing.held = undefined;
      }
      this.#fade += 1;
      if (restoreVolume && this.#flow === "running") {
        this.#rampTo(1);
      }
    }
    this.#settleCall(call, { kind: "overtaken" });
  }

  #targetEnded(call: HoldCall, end: PlaybackEnd): void {
    if (this.#call !== call) {
      return;
    }
    if (end.kind === "stopped" || end.kind === "muted") {
      this.#settleCall(call, { kind: "overtaken" });
      return;
    }
    call.afterEnd = true;
    if (this.#flow === "running") {
      this.#holdNow();
    } else if (this.#flow === "paused") {
      this.#landed();
    }
  }

  /** Looks at the waiting call again: the target may now be playing, have more chunks, or have ended. */
  #advanceCall(): void {
    const call = this.#call;
    if (call === undefined || call.scheduled) {
      return;
    }
    const ended = call.item.endedAs;
    if (ended !== undefined) {
      this.#targetEnded(call, ended);
      return;
    }
    const playing = this.#playing;
    if (playing?.item !== call.item) {
      return;
    }
    if (playing.stoppingAt !== undefined) {
      this.#settleCall(call, { kind: "overtaken" });
    } else if (this.#flow === "paused") {
      this.#landed();
    } else if (this.#flow === "running") {
      const position = classify(playing.chunks, playing.bodyScheduled, call.position, this.#context.currentTime, FADE_SECONDS);
      if (position === "passed") {
        this.#holdNow();
      } else if (position === "ahead") {
        this.#fadeAhead(call, playing);
      }
    }
  }

  /** The fade to 0 that ends exactly at the position on the audio clock, then the suspend. */
  #fadeAhead(call: HoldCall, playing: Scheduled): void {
    const endsAt = contextTimeAt(playing.chunks, call.position);
    if (endsAt === undefined) {
      return;
    }
    const startsAt = endsAt - FADE_SECONDS;
    this.#gain.gain.setValueAtTime(1, startsAt);
    this.#gain.gain.linearRampToValueAtTime(0, endsAt);
    this.#ramp = { target: 0, startsAt, endsAt };
    playing.held = { at: endsAt, seconds: call.position };
    call.scheduled = true;
    void this.#suspendOnceFaded(endsAt, this.#fade);
  }

  async #resumeAndFadeIn(fade: number): Promise<void> {
    // A suspended context's clock stands still, so the ramp set on it starts the moment the context renders again, not
    // when the page's task that awaits the resume runs: the sounds play on where they paused, whatever the page's timing.
    this.#fadeIn();
    if (!(await this.#resume())) {
      if (fade === this.#fade) {
        this.#flow = "paused";
      }
      return;
    }
    if (fade === this.#fade) {
      if (this.#playing !== undefined) {
        this.#announceStart(this.#playing);
        this.#settle(this.#playing);
      }
      this.#advanceCall();
      void this.#pump();
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
        if (!this.#isHead(generation, head) || this.#flow !== "running") {
          return;
        }
        const running = loaded !== undefined && (await this.#resume());
        // A pause that came meanwhile leaves the head waiting: its resume() pumps again.
        if (!this.#isHead(generation, head) || this.#flow !== "running") {
          return;
        }
        if (running) {
          this.#queue.shift();
          const scheduled = new Scheduled(head);
          this.#playing = scheduled;
          head.follow(() => scheduled.read(this.#context.currentTime));
          if (loaded.kind === "buffer") {
            scheduled.bodyScheduled = true;
            this.#schedule(scheduled, loaded.buffer, 0);
          } else {
            void this.#playStream(scheduled, loaded.stream, generation);
          }
          return;
        }
        // A failed sound has ended already; a refused one ends here. Neither blocks the next.
        head.abort.abort();
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

  #resume(): Promise<boolean> {
    this.#resuming = this.#resumeContext();
    return this.#resuming;
  }

  async #resumeContext(): Promise<boolean> {
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

  /**
   * The earliest context time the player knows to lie ahead of the render thread: the page's `currentTime` lags it by
   * up to one callback buffer (`baseLatency`), so a start before this would play at a time the player cannot tell.
   */
  #earliestKnownStart(): number {
    const quantum = RENDER_QUANTUM_FRAMES / this.#context.sampleRate;
    return this.#context.currentTime + Math.max(this.#context.baseLatency, quantum);
  }

  /** Schedules the buffer on the one gain at `when` of the context's clock, or at the earliest known start if that is later; returns the start. */
  #schedule(scheduled: Scheduled, buffer: AudioBuffer, when: number): number {
    const source = this.#context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.#gain);
    source.onended = () => {
      scheduled.sources.delete(source);
      this.#settle(scheduled);
    };
    const start = Math.max(when, this.#earliestKnownStart(), this.#restoredAt);
    scheduled.sources.set(source, start);
    source.start(start);
    scheduled.chunks.push({ start, seconds: buffer.duration });
    this.#announceStart(scheduled);
    this.#advanceCall();
    return start;
  }

  /**
   * Reports the sound's start once its first sample can be heard: while the player runs, or while it pauses and the
   * sample starts before the pause ramp ends. Not once it is paused, where a streamed body's first samples queue until
   * `resume()` and the sound may still end before it has played.
   */
  #announceStart(scheduled: Scheduled): void {
    const first = scheduled.chunks[0];
    if (first !== undefined && this.#context.state === "running" && this.#heardFrom(first.start)) {
      scheduled.item.begin(first.start, this.#context);
    }
  }

  #heardFrom(start: number): boolean {
    if (this.#flow === "pausing") {
      return this.#ramp !== undefined && start < this.#ramp.endsAt;
    }
    return this.#flow === "running";
  }

  /**
   * Back to back on the context's clock: each chunk starts where the one before ends, counted from the start the
   * player asked for. A stall before the start call that moves the first chunk later makes the next overlap it by up
   * to the blocked time. The page cannot tell that stall from one after the call, which leaves the first chunk on
   * time, so counting from a later clock read would turn this overlap into a gap there.
   */
  async #playStream(scheduled: Scheduled, stream: PcmStream, generation: number): Promise<void> {
    try {
      await this.#scheduleChunks(scheduled, stream, generation, 0);
    } catch {
      if (generation === this.#generation) {
        this.#playing = undefined;
        silence(scheduled);
        scheduled.item.abort.abort();
        scheduled.item.end({ kind: "failed" });
        void this.#pump();
      }
      return;
    }
    if (generation === this.#generation) {
      scheduled.bodyScheduled = true;
      this.#advanceCall();
      this.#settle(scheduled);
    }
  }

  async #scheduleChunks(
    scheduled: Scheduled,
    stream: PcmStream,
    generation: number,
    startAt: number,
  ): Promise<void> {
    const samples = await stream.next();
    if (samples === undefined || generation !== this.#generation) {
      return;
    }
    const buffer = this.#context.createBuffer(1, samples.length, stream.sampleRate);
    buffer.copyToChannel(samples, 0);
    const when = this.#schedule(scheduled, buffer, startAt);
    return this.#scheduleChunks(scheduled, stream, generation, when + buffer.duration);
  }

  #settle(scheduled: Scheduled): void {
    if (scheduled.bodyScheduled && scheduled.sources.size === 0 && this.#playing === scheduled && !scheduled.runsOutHeld()) {
      this.#playing = undefined;
      scheduled.item.end({ kind: "played", playedSeconds: playedBy(scheduled.chunks, Infinity) });
      void this.#pump();
    }
  }

  /**
   * Ends the waiting sounds at once. The playing sound ends at once as well when `instant`, or when no one can hear
   * it: the output is held, or nothing of it is scheduled. Otherwise it ends where the volume ramp reaches 0.
   */
  #endAll(kind: Interruption, { instant }: { readonly instant: boolean }): void {
    if (this.#call !== undefined) {
      this.#overtake(this.#call, { restoreVolume: false });
    }
    this.#generation += 1;
    const playing = this.#playing;
    const waiting = this.#queue;
    this.#queue = [];
    if (playing !== undefined) {
      playing.item.abort.abort();
      if (instant || this.#flow === "paused" || this.#context.state !== "running" || playing.sources.size === 0) {
        this.#playing = undefined;
        silence(playing);
        playing.item.end({ kind, playedSeconds: playing.read(this.#context.currentTime) });
      } else if (playing.stoppingAt === undefined) {
        this.#fadeOut(playing);
      }
    }
    for (const item of waiting) {
      item.abort.abort();
      item.end({ kind, playedSeconds: 0 });
    }
  }

  /**
   * The sources that play at the ramp's end stop there on the audio clock, so that the cut is silent. One that starts
   * after it never plays, and Web Audio ends it only when its start passes, so it is dropped here.
   */
  #fadeOut(playing: Scheduled): void {
    const endsAt = this.#rampTo(0);
    playing.stoppingAt = endsAt;
    for (const [source, start] of playing.sources) {
      if (start < endsAt) {
        source.onended = () => {
          playing.sources.delete(source);
          this.#endStopped(playing, endsAt);
        };
        source.stop(endsAt);
      } else {
        source.onended = null;
        source.stop();
        playing.sources.delete(source);
      }
    }
    this.#endStopped(playing, endsAt);
  }

  #endStopped(playing: Scheduled, endsAt: number): void {
    if (playing.sources.size > 0 || this.#playing !== playing) {
      return;
    }
    this.#playing = undefined;
    playing.item.end({ kind: "stopped", playedSeconds: playing.read(endsAt) });
    if (this.#flow === "running") {
      this.#restoreVolumeAt(endsAt);
    }
    void this.#pump();
  }
}
