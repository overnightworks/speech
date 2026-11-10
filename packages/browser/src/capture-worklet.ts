// The worklet's own global scope: it posts a transferred copy of each input block to the main thread.
class SpeechCapture extends AudioWorkletProcessor {
  process(inputs: Float32Array[][]): boolean {
    const block = inputs[0]?.[0];
    if (block !== undefined) {
      const copy = block.slice();
      this.port.postMessage(copy, [copy.buffer]);
    }
    return true;
  }
}

registerProcessor("speech-capture", SpeechCapture);
