const HEARD_THRESHOLD = 0.005;

// Runs on the render thread, so `currentFrame` makes each onset exact to one sample of the context's clock.
// Each render quantum also reports its peak, with the context time it starts at: the output's envelope.
registerProcessor(
  "ear",
  class extends AudioWorkletProcessor {
    silent = true;

    process(inputs, outputs) {
      const samples = inputs[0][0] ?? [];
      const first = samples.findIndex((value) => Math.abs(value) > HEARD_THRESHOLD);
      if (first >= 0 && this.silent) this.port.postMessage({ onset: (currentFrame + first) / sampleRate });
      this.silent = first < 0;
      this.port.postMessage({ quantum: currentFrame / sampleRate, peak: samples.reduce((peak, value) => Math.max(peak, Math.abs(value)), 0) });
      inputs[0].forEach((channel, index) => outputs[0][index]?.set(channel));
      return true;
    }
  },
);
