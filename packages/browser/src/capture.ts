import { HopClose } from "./hop.js";
import { Framer, Resampler } from "./resampler.js";

export { HOP_FRAME_BYTES, HOP_FRAME_SAMPLES, HopClose } from "./hop.js";

export interface CaptureOptions {
  /** The app's stream; the package uses its first audio track. */
  readonly stream: MediaStream;
  /** The app's own same-origin relay route. */
  readonly hopUrl: string | URL;
  /** Where the app serves the exported worklet file. */
  readonly workletUrl: string | URL;
  /**
   * Receives every message the app's server sends on the hop, as the browser delivers it: text as a string, binary as
   * an ArrayBuffer. It is called from the socket's creation, in arrival order, and never after the capture ends; the
   * package does not read a message, and a throw surfaces as an uncaught error of the page without ending the capture.
   */
  readonly onMessage?: (data: string | ArrayBuffer) => void;
}

export interface Capture {
  /** Resolves once and never rejects: "streaming" when the worklet runs into an open hop, "ended" when the capture ended first. */
  readonly started: Promise<"streaming" | "ended">;
  /** Resolves once and never rejects, with the first cause of the end. */
  readonly ended: Promise<CaptureEnd>;
  /** The app's switch. Idempotent; never throws. */
  stop(): void;
}

export type LocalFailure = "no-audio-track" | "worklet-refused" | "context-failed";

export type CaptureEnd =
  | { readonly kind: "hop"; readonly close: HopClose }
  | { readonly kind: "socket"; readonly code: number }
  | { readonly kind: "local"; readonly failure: LocalFailure };

const PROCESSOR = "speech-capture";
const HOP_CLOSE_CODES: ReadonlySet<number> = new Set(Object.values(HopClose));

function isHopClose(code: number): code is HopClose {
  return HOP_CLOSE_CODES.has(code);
}

function endOfSocket(code: number): CaptureEnd {
  return isHopClose(code) ? { kind: "hop", close: code } : { kind: "socket", code };
}

/**
 * Streams the first audio track of the app's stream to the app's relay route as PCM16 frames at the wire's rate.
 *
 * The package opens no device, never stops the app's track and interprets no message of the hop; it hands each one to
 * `onMessage` as it came. Audio is taken only once the hop is open, and everything is sent: there is no loudness
 * gate.
 */
export function startCapture(options: CaptureOptions): Capture {
  const run = new CaptureRun(options);
  run.start();
  return { started: run.started, ended: run.ended, stop: () => run.finish() };
}

class CaptureRun {
  readonly #startedGate = Promise.withResolvers<"streaming" | "ended">();
  readonly #endedGate = Promise.withResolvers<CaptureEnd>();
  readonly started = this.#startedGate.promise;
  readonly ended = this.#endedGate.promise;
  readonly #options: CaptureOptions;
  // The first cause; "socket" leaves it to the code the socket closes with.
  #cause: CaptureEnd | "socket" | undefined;
  #track: MediaStreamTrack | undefined;
  #context: AudioContext | undefined;
  #socket: WebSocket | undefined;
  #socketClosed: Promise<number> = Promise.resolve(0);
  #resumed = false;
  #workletLoaded = false;
  #source: MediaStreamAudioSourceNode | undefined;
  #node: AudioWorkletNode | undefined;
  #resampler: Resampler | undefined;
  readonly #framer = new Framer();
  readonly #onTrackEnded = (): void => this.finish();

  constructor(options: CaptureOptions) {
    this.#options = options;
  }

  start(): void {
    const { stream, hopUrl, workletUrl } = this.#options;
    const track = stream.getAudioTracks()[0];
    if (track === undefined || track.readyState === "ended") {
      this.finish({ kind: "local", failure: "no-audio-track" });
      return;
    }
    try {
      this.#context = new AudioContext();
    } catch {
      this.finish({ kind: "local", failure: "context-failed" });
      return;
    }
    // Without sticky activation a suspended context's resume() would stay pending forever.
    if (this.#context.state === "suspended" && !navigator.userActivation.hasBeenActive) {
      this.finish({ kind: "local", failure: "context-failed" });
      return;
    }
    try {
      this.#socket = new WebSocket(hopUrl);
      this.#socket.binaryType = "arraybuffer";
    } catch (error) {
      void this.#context.close();
      throw error;
    }
    this.#track = track;
    this.#begin(this.#context, this.#socket, workletUrl);
  }

  #begin(context: AudioContext, socket: WebSocket, workletUrl: string | URL): void {
    this.#track?.addEventListener("ended", this.#onTrackEnded);
    this.#socketClosed = new Promise((resolve) => {
      socket.addEventListener("close", (event) => {
        resolve(event.code);
        this.finish("socket");
      });
    });
    socket.addEventListener("open", () => this.#connectWhenReady());
    const { onMessage } = this.#options;
    if (onMessage !== undefined) {
      // The browser delivers no message to a socket that is not open, and every end closes it in `finish`.
      socket.addEventListener("message", (event: MessageEvent<string | ArrayBuffer>) =>
        onMessage(event.data),
      );
    }
    const failed = (failure: LocalFailure) => () => this.finish({ kind: "local", failure });
    Promise.resolve()
      .then(() => context.audioWorklet.addModule(workletUrl))
      .then(() => {
        this.#workletLoaded = true;
        this.#connectWhenReady();
      }, failed("worklet-refused"));
    this.#resumed = context.state === "running";
    if (!this.#resumed) {
      context.resume().then(() => {
        this.#resumed = true;
        this.#connectWhenReady();
      }, failed("context-failed"));
    }
  }

  #connectWhenReady(): void {
    const context = this.#context as AudioContext;
    const ready = this.#resumed && this.#workletLoaded && this.#socket?.readyState === WebSocket.OPEN;
    if (!ready || this.#cause !== undefined) {
      return;
    }
    try {
      this.#node = new AudioWorkletNode(context, PROCESSOR, { channelCount: 1, channelCountMode: "explicit" });
    } catch {
      this.finish({ kind: "local", failure: "worklet-refused" });
      return;
    }
    const resampler = new Resampler(context.sampleRate);
    this.#resampler = resampler;
    this.#node.port.onmessage = (event: MessageEvent<Float32Array>) => this.#send(resampler.push(event.data));
    this.#source = context.createMediaStreamSource(new MediaStream([this.#track as MediaStreamTrack]));
    this.#source.connect(this.#node);
    this.#startedGate.resolve("streaming");
  }

  #send(samples: Float32Array, last = false): void {
    const frames = this.#framer.push(samples);
    if (last) {
      frames.push(...this.#framer.flush());
    }
    for (const frame of frames) {
      this.#socket?.send(frame);
    }
  }

  /** Ends the capture once; the first cause wins, and without one the socket's close code decides. */
  finish(cause: CaptureEnd | "socket" = "socket"): void {
    if (this.#cause !== undefined) {
      return;
    }
    this.#cause = cause;
    this.#track?.removeEventListener("ended", this.#onTrackEnded);
    this.#source?.disconnect();
    if (this.#node !== undefined) {
      this.#node.port.onmessage = null;
    }
    if (this.#resampler !== undefined && this.#socket?.readyState === WebSocket.OPEN) {
      this.#send(this.#resampler.flush(), true);
    }
    this.#socket?.close(HopClose.NORMAL);
    void this.#settle();
  }

  async #settle(): Promise<void> {
    let code = 0;
    try {
      [code] = await Promise.all([this.#socketClosed, this.#context?.close()]);
    } finally {
      const cause = this.#cause === "socket" ? endOfSocket(code) : (this.#cause as CaptureEnd);
      this.#startedGate.resolve("ended");
      this.#endedGate.resolve(cause);
    }
  }
}
