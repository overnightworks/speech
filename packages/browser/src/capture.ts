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
  /**
   * Sends a text message of the app's own up the hop, in order with the audio frames, and returns whether it was
   * sent. It returns false, sends nothing and queues nothing before the socket is open and after the capture ended;
   * the message is not resent. A message that is not a string throws a `TypeError`, and the capture goes on. The
   * package does not read the message, and the app's relay route bounds it like every frame (a longer one ends the
   * hop as `HopClose.FRAME_TOO_LARGE`). It is the page's counterpart of `onMessage`.
   */
  send(message: string): boolean;
}

export type LocalFailure = "no-audio-track" | "worklet-refused" | "context-failed" | "context-stopped";

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
 * `onMessage` as it came and sends the app's own text through `send`. Audio is taken only once the hop is open, and
 * everything is sent: there is no loudness gate.
 *
 * It throws, after closing the context it opened, on a `hopUrl` the WebSocket constructor refuses, such as one with a
 * scheme other than `ws:`, `wss:`, `http:` or `https:`: a caller's programming error.
 */
export function startCapture(options: CaptureOptions): Capture {
  const run = new CaptureRun(options);
  run.start();
  return {
    started: run.started,
    ended: run.ended,
    stop: () => run.finish(),
    send: (message) => run.sendText(message),
  };
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
  // A context that leaves "running" after it streamed (suspended, interrupted or closed by the host) sends no more audio.
  // No resume() is tried: a browser resumes a context only under user activation or a policy, and an interruption by
  // the host's sound server ends at the host's time, so a retry would only leave the hop open and silent.
  readonly #onContextStateChange = (): void => {
    if (this.#context?.state !== "running") {
      this.finish({ kind: "local", failure: "context-stopped" });
    }
  };

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
    context.addEventListener("statechange", this.#onContextStateChange);
    this.#startedGate.resolve("streaming");
  }

  sendText(message: string): boolean {
    if (typeof message !== "string") {
      throw new TypeError("send takes a string");
    }
    if (this.#cause !== undefined || this.#socket?.readyState !== WebSocket.OPEN) {
      return false;
    }
    this.#socket.send(message);
    return true;
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
    this.#context?.removeEventListener("statechange", this.#onContextStateChange);
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
      // close() of a closed context rejects, and the host may have closed it already.
      const closing = this.#context?.state === "closed" ? undefined : this.#context?.close();
      [code] = await Promise.all([this.#socketClosed, closing]);
    } finally {
      const cause = this.#cause === "socket" ? endOfSocket(code) : (this.#cause as CaptureEnd);
      this.#startedGate.resolve("ended");
      this.#endedGate.resolve(cause);
    }
  }
}
