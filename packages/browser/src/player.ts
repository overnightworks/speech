import { openSound, type PcmStream } from "./wav-stream.js";

/** A sound the app hands over: its bytes (the player decodes a copy), or the URL of the app's own route that answers one.
 * A route answering 16-bit mono PCM `audio/wav` plays while its body still arrives; every other sound plays whole. */
export type Sound = ArrayBuffer | string | URL;

export type PlaybackEnd =
  /** `playedSeconds`: the seconds of the sound the output played, read from the player's AudioContext clock and never more than the sound's length. A "played" sound played its full length; one that never started played 0. */
  | { readonly kind: "played"; readonly playedSeconds: number }
  | { readonly kind: "stopped"; readonly playedSeconds: number } // stop() or close(); a stop() ends the playing sound once its edge is over
  | { readonly kind: "muted"; readonly playedSeconds: number } // setMuted(true), or a sound handed over while muted
  | { readonly kind: "refused" } // no user activation, or resume() refused
  | { readonly kind: "failed" }; // the fetch, the decode or a streamed body failed

export type PlaybackStart =
  /** `contextTime`: the time on `context`'s clock (`currentTime`) at which the sound's first sample plays, to within one render quantum. An app adds a timing relative to the sound's start to it to place that part on the clock it reads each frame. `context` is the AudioContext the sound plays on. */
  | { readonly kind: "playing"; readonly contextTime: number; readonly context: AudioContext }
  | { readonly kind: "ended" }; // the sound ended first, so it has no start

export interface Playback {
  /** Resolves once and never rejects: "playing" with the sound's start when its source starts on a running context, "ended" when it ended first. */
  readonly started: Promise<PlaybackStart>;
  /** Resolves once and never rejects, with how the sound ended. A streamed body that breaks off ends "failed" where it broke. */
  readonly ended: Promise<PlaybackEnd>;
}

export interface Player {
  /** Queues the sound behind every earlier one and starts fetching and decoding it at once. Never throws. */
  play(sound: Sound): Playback;
  /**
   * Holds the output at once and the sounds where they are: the volume ramps to 0 over a click-free edge of a few tens
   * of milliseconds, then the context suspends. The context's clock, every scheduled sound and each start the app read
   * stay valid, a body that arrives meanwhile queues on the frozen clock, and no waiting sound starts until `resume()`; a streamed sound's first samples that arrive meanwhile report its start at `resume()`.
   * Idempotent; ignored while muted or closed; never throws.
   */
  pause(): void;
  /** Resumes the context and ramps the volume back up over the same edge: the sounds play on where they paused. Idempotent; never throws. */
  resume(): void;
  /**
   * Ramps the volume to 0 over the same edge, ends the playing sound as "stopped" when it is over and every waiting
   * one at once, aborting their fetches; later sounds play. A pause stays until `resume()`: while paused, the playing
   * sound ends at once, because the output is held already. Idempotent; never throws.
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
 * header, stores nothing and builds no tap. It throws only if the browser cannot create an AudioContext.
 */
export function createPlayer(): Player {
  const run = new PlayerRun();
  return {
    play: (sound) => run.play(sound),
    pause: () => run.pause(),
    resume: () => run.resume(),
    stop: () => run.stop(),
    setMuted: (muted) => run.setMuted(muted),
    close: () => run.close(),
  };
}

type Interruption = "stopped" | "muted";

/** "pausing": the volume ramps to 0 and the context is not yet suspended; "paused": the context suspends or is suspended. */
type Flow = "running" | "pausing" | "paused";

/** The ramp the gain is on: where it ends up, and when on the context's clock. */
interface Ramp {
  readonly target: number;
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

/** The sources one sound has scheduled, each with the context time it starts at; a whole sound has one, a streamed sound one per chunk. */
class Scheduled {
  readonly sources = new Map<AudioBufferSourceNode, number>();
  readonly chunks: { readonly start: number; readonly seconds: number }[] = [];
  bodyScheduled = false;
  /** The context time at which a `stop()` ends the sound, once it has asked for it. */
  stoppingAt: number | undefined;
  constructor(readonly item: Queued) {}

  /** The seconds of the scheduled chunks the output has played by `now` of the context's clock. */
  playedBy(now: number): number {
    let played = 0;
    for (const { start, seconds } of this.chunks) {
      played += Math.min(Math.max(now - start, 0), seconds);
    }
    return played;
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

class Queued {
  readonly abort = new AbortController();
  readonly playback: Playback;
  loaded: Promise<Loaded | undefined> = Promise.resolve(undefined);
  readonly #started = Promise.withResolvers<PlaybackStart>();
  readonly #ended = Promise.withResolvers<PlaybackEnd>();
  #over = false;

  constructor() {
    this.playback = { started: this.#started.promise, ended: this.#ended.promise };
  }

  /** Only the first call counts: a streamed sound begins with its first chunk. */
  begin(contextTime: number, context: AudioContext): void {
    this.#started.resolve({ kind: "playing", contextTime, context });
  }

  /** The first end wins. */
  end(end: PlaybackEnd): void {
    if (!this.#over) {
      this.#over = true;
      this.#started.resolve({ kind: "ended" });
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
  #flow: Flow = "running";
  #ramp: Ramp | undefined;
  // Every call that sets the volume bumps the fade; a wait that began in an older one is overtaken.
  #fade = 0;

  constructor() {
    this.#gain.connect(this.#context.destination);
  }

  play(sound: Sound): Playback {
    const item = new Queued();
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
    this.#flow = "pausing";
    const fade = ++this.#fade;
    void this.#suspendOnceFaded(this.#rampTo(0), fade);
  }

  resume(): void {
    if (this.#closed || this.#flow === "running") {
      return;
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

  /** The volume at once, for mute and for the end of a stop's edge: no ramp, and no older ramp left to override it. */
  #setVolume(volume: number): void {
    this.#fade += 1;
    this.#ramp = undefined;
    this.#gain.gain.cancelScheduledValues(this.#context.currentTime);
    this.#gain.gain.value = volume;
  }

  /**
   * Ramps the gain from where it is to `target` and returns the context time the ramp ends at. A ramp already on its
   * way to `target` stays, so that a sound stopped at the end of it is cut where the volume is 0. Firefox lacks
   * cancelAndHoldAtTime, so the ramp starts from the value the gain reads.
   */
  #rampTo(target: number): number {
    const now = this.#context.currentTime;
    if (this.#ramp?.target === target && this.#ramp.endsAt > now) {
      return this.#ramp.endsAt;
    }
    const param = this.#gain.gain;
    const from = param.value;
    const endsAt = now + FADE_SECONDS;
    param.cancelScheduledValues(now);
    param.setValueAtTime(from, now);
    param.linearRampToValueAtTime(target, endsAt);
    this.#ramp = { target, endsAt };
    return endsAt;
  }

  /** Waits on the context's clock, where the page's timers lag, until it passes `time`; false if a later fade overtook it. */
  async #untilContextTime(time: number, fade: number): Promise<boolean> {
    while (fade === this.#fade && this.#context.state === "running" && this.#context.currentTime < time) {
      const seconds = time - this.#context.currentTime;
      await new Promise((resolve) => setTimeout(resolve, seconds * MILLISECONDS_PER_SECOND));
    }
    return fade === this.#fade;
  }

  async #suspendOnceFaded(endsAt: number, fade: number): Promise<void> {
    if (await this.#untilContextTime(endsAt, fade)) {
      this.#flow = "paused";
      void this.#context.suspend();
    }
  }

  async #resumeAndFadeIn(fade: number): Promise<void> {
    if (!(await this.#resume())) {
      if (fade === this.#fade) {
        this.#flow = "paused";
      }
      return;
    }
    if (fade === this.#fade) {
      // A sound being stopped brings the volume back when it ends.
      if (this.#playing?.stoppingAt === undefined) {
        this.#rampTo(1);
      }
      if (this.#playing !== undefined) {
        this.#announceStart(this.#playing);
      }
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
          if (loaded.kind === "buffer") {
            this.#schedule(scheduled, loaded.buffer, 0);
            scheduled.bodyScheduled = true;
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
    const start = Math.max(when, this.#earliestKnownStart());
    scheduled.sources.set(source, start);
    source.start(start);
    scheduled.chunks.push({ start, seconds: buffer.duration });
    this.#announceStart(scheduled);
    return start;
  }

  /**
   * Reports the sound's start once its first sample can be heard: not while the player is paused, where a streamed
   * body's first samples queue until `resume()` and the sound may still end before it has played.
   */
  #announceStart(scheduled: Scheduled): void {
    const first = scheduled.chunks[0];
    if (first !== undefined && this.#flow === "running" && this.#context.state === "running") {
      scheduled.item.begin(first.start, this.#context);
    }
  }

  /** Back to back on the context's clock: each chunk starts where the one before ends. */
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
    if (scheduled.bodyScheduled && scheduled.sources.size === 0 && this.#playing === scheduled) {
      this.#playing = undefined;
      scheduled.item.end({ kind: "played", playedSeconds: scheduled.playedBy(Infinity) });
      void this.#pump();
    }
  }

  /**
   * Ends the waiting sounds at once. The playing sound ends at once as well when `instant`, or when no one can hear
   * it: the output is held, or nothing of it is scheduled. Otherwise it ends where the volume ramp reaches 0.
   */
  #endAll(kind: Interruption, { instant }: { readonly instant: boolean }): void {
    this.#generation += 1;
    const playing = this.#playing;
    const waiting = this.#queue;
    this.#queue = [];
    if (playing !== undefined) {
      playing.item.abort.abort();
      if (instant || this.#flow === "paused" || this.#context.state !== "running" || playing.sources.size === 0) {
        this.#playing = undefined;
        silence(playing);
        playing.item.end({ kind, playedSeconds: playing.playedBy(this.#context.currentTime) });
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
    playing.item.end({ kind: "stopped", playedSeconds: playing.playedBy(endsAt) });
    if (this.#flow === "running") {
      this.#setVolume(1);
    }
    void this.#pump();
  }
}
