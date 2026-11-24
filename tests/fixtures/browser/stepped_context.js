// An AudioContext the test steps. A page script that runs after this one and creates `AudioContext` gets an
// OfflineAudioContext that renders only when the test calls `step`: its clock, its nodes and its worklet advance by
// whole render quanta, and nothing else moves them.
//
// What the context offers is the AudioContext surface a page uses: `state` and its `statechange`, `resume()`, `suspend()`
// without a time, `close()`, `baseLatency`, `currentTime`. A context starts suspended; it renders while its state is
// "running" and stands still while it is "suspended", as a context does on the audio device.
(() => {
  const RENDER_QUANTUM_FRAMES = 128;
  const SAMPLE_RATE = 48000;
  const OUTPUT_CHANNELS = 2;
  // How long a context can be stepped: the rendering buffer is allocated up front.
  const RENDERABLE_SECONDS = 60;
  // What the audio device's output buffer holds: the lag `currentTime` has behind the sound that is heard.
  const BASE_LATENCY_FRAMES = 512;
  const { currentTime: clock } = Object.getOwnPropertyDescriptors(BaseAudioContext.prototype);

  window.AudioContext = class SteppedAudioContext extends OfflineAudioContext {
    #state = "suspended";
    #changes = new EventTarget();
    #started = false;
    // Counts the calls that end a pending `resume`: a `suspend` or `close` that comes after it wins.
    #stops = 0;

    constructor() {
      super(OUTPUT_CHANNELS, RENDERABLE_SECONDS * SAMPLE_RATE, SAMPLE_RATE);
    }

    get state() {
      return this.#state;
    }

    get baseLatency() {
      return BASE_LATENCY_FRAMES / SAMPLE_RATE;
    }

    // The offline context's own state changes are its stepping, not the page's context changing state.
    addEventListener(type, listener, options) {
      if (type === "statechange") {
        this.#changes.addEventListener(type, listener, options);
      } else {
        super.addEventListener(type, listener, options);
      }
    }

    // The state is "running" when the promise resolves, and `suspend` and `close` change it at the call.
    async resume() {
      this.#refuseWhenClosed();
      const stopsAtCall = this.#stops;
      await null;
      if (this.#stops === stopsAtCall) {
        this.#become("running");
      }
    }

    async suspend() {
      this.#refuseWhenClosed();
      this.#stops += 1;
      this.#become("suspended");
    }

    async close() {
      this.#stops += 1;
      this.#become("closed");
    }

    #refuseWhenClosed() {
      if (this.#state === "closed") {
        throw new DOMException("the context is closed", "InvalidStateError");
      }
    }

    // The event follows the promise of the call that changed the state.
    #become(state) {
      if (state !== this.#state) {
        this.#state = state;
        queueMicrotask(() => queueMicrotask(() => this.#changes.dispatchEvent(new Event("statechange"))));
      }
    }

    // Renders `quanta` render quanta if the context is running, and returns the frames rendered so far.
    async step(quanta) {
      if (this.#state === "running") {
        const frames = Math.round(clock.get.call(this) * SAMPLE_RATE) + quanta * RENDER_QUANTUM_FRAMES;
        // A suspend time is rounded up to a render quantum: asking for the middle of the last one is exact.
        const reached = super.suspend((frames - RENDER_QUANTUM_FRAMES / 2) / SAMPLE_RATE);
        if (this.#started) {
          super.resume();
        } else {
          this.#started = true;
          super.startRendering();
        }
        await reached;
      }
      return Math.round(clock.get.call(this) * SAMPLE_RATE);
    }
  };
})();
