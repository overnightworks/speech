import { HOP_FRAME_BYTES, HOP_FRAME_SAMPLES, SAMPLE_BYTES, SAMPLE_RATE } from "./hop.js";

// The filter's design: a Kaiser-windowed sinc whose stopband lies about 80 dB down,
// with its cutoff at 0.45 of the lower of the two rates and its transition band
// from 0.4 to 0.5 of it, so nothing above the lower rate's Nyquist frequency passes.
const STOPBAND_DB = 80;
const CUTOFF_SHARE = 0.45;
const TRANSITION_SHARE = 0.1;
const KAISER_BETA = 0.1102 * (STOPBAND_DB - 8.7);

function greatestCommonDivisor(a: number, b: number): number {
  return b === 0 ? a : greatestCommonDivisor(b, a % b);
}

function besselI0(x: number): number {
  let sum = 1;
  let term = 1;
  for (let k = 1; term > sum * 1e-12; k += 1) {
    term *= (x / (2 * k)) ** 2;
    sum += term;
  }
  return sum;
}

function windowedSinc(distance: number, cutoff: number, halfWidth: number): number {
  const position = distance / halfWidth;
  const window = besselI0(KAISER_BETA * Math.sqrt(Math.max(0, 1 - position * position))) / besselI0(KAISER_BETA);
  const argument = Math.PI * 2 * cutoff * distance;
  return 2 * cutoff * (argument === 0 ? 1 : Math.sin(argument) / argument) * window;
}

/**
 * A streaming rational resampler: a polyphase FIR filter with a Kaiser-windowed sinc.
 *
 * Output sample m lies at input time m * down / up, so the output is aligned with the input and carries no delay.
 * Each output waits for the input samples its filter reaches; `flush` treats the input after the last sample as
 * silence and returns the rest. Any split of the same input gives the same samples.
 */
export class Resampler {
  readonly #up: number;
  readonly #down: number;
  readonly #halfWidth: number;
  readonly #phases: Float32Array[];
  #history: Float32Array;
  #historyStart: number;
  #received = 0;
  #next = 0;

  constructor(inputRate: number, outputRate: number = SAMPLE_RATE) {
    const divisor = greatestCommonDivisor(inputRate, outputRate);
    this.#up = outputRate / divisor;
    this.#down = inputRate / divisor;
    const lowerRate = Math.min(inputRate, outputRate);
    const transition = (TRANSITION_SHARE * lowerRate) / inputRate;
    this.#halfWidth = Math.ceil((STOPBAND_DB - 7.95) / (2.285 * 2 * Math.PI * transition) / 2);
    const cutoff = (CUTOFF_SHARE * lowerRate) / inputRate;
    this.#phases = Array.from({ length: this.#up }, (_, phase) => {
      const taps = Float32Array.from({ length: 2 * this.#halfWidth }, (_, tap) =>
        windowedSinc(phase / this.#up - (tap - this.#halfWidth + 1), cutoff, this.#halfWidth),
      );
      const gain = taps.reduce((sum, tap) => sum + tap, 0);
      return taps.map((tap) => tap / gain);
    });
    // The input before the first sample is silence.
    this.#history = new Float32Array(this.#halfWidth - 1);
    this.#historyStart = 1 - this.#halfWidth;
  }

  /** The output samples this block completes. */
  push(block: Float32Array): Float32Array {
    this.#append(block);
    this.#received += block.length;
    return this.#produce(this.#outputsBefore(this.#received - this.#halfWidth));
  }

  /** The rest of the output, as if silence followed the last block. */
  flush(): Float32Array {
    this.#append(new Float32Array(this.#halfWidth));
    return this.#produce(this.#outputsBefore(this.#received));
  }

  #outputsBefore(inputSamples: number): number {
    return Math.ceil((Math.max(0, inputSamples) * this.#up) / this.#down);
  }

  #append(block: Float32Array): void {
    const joined = new Float32Array(this.#history.length + block.length);
    joined.set(this.#history);
    joined.set(block, this.#history.length);
    this.#history = joined;
  }

  #produce(end: number): Float32Array {
    const output = new Float32Array(end - this.#next);
    for (let index = 0; index < output.length; index += 1) {
      const position = (this.#next + index) * this.#down;
      const nearest = Math.floor(position / this.#up);
      const taps = this.#phases[position - nearest * this.#up] as Float32Array;
      const first = nearest - this.#halfWidth + 1 - this.#historyStart;
      let sum = 0;
      for (let tap = 0; tap < taps.length; tap += 1) {
        sum += (taps[tap] as number) * (this.#history[first + tap] as number);
      }
      output[index] = sum;
    }
    this.#next = end;
    const kept = Math.floor((end * this.#down) / this.#up) - this.#halfWidth + 1;
    this.#history = this.#history.subarray(kept - this.#historyStart);
    this.#historyStart = kept;
    return output;
  }
}

function pcm16(sample: number): number {
  const clamped = Math.max(-1, Math.min(1, sample));
  return Math.round(clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff);
}

/** Cuts samples into PCM16 little-endian frames of `HOP_FRAME_SAMPLES` samples each. */
export class Framer {
  #frame = new DataView(new ArrayBuffer(HOP_FRAME_BYTES));
  #filled = 0;

  /** The whole frames these samples complete. */
  push(samples: Float32Array): ArrayBuffer[] {
    const frames: ArrayBuffer[] = [];
    for (const sample of samples) {
      this.#frame.setInt16(this.#filled * SAMPLE_BYTES, pcm16(sample), true);
      this.#filled += 1;
      if (this.#filled === HOP_FRAME_SAMPLES) {
        frames.push(this.#frame.buffer);
        this.#frame = new DataView(new ArrayBuffer(HOP_FRAME_BYTES));
        this.#filled = 0;
      }
    }
    return frames;
  }

  /** The last partial frame, if any samples are left. */
  flush(): ArrayBuffer[] {
    const rest = this.#frame.buffer.slice(0, this.#filled * SAMPLE_BYTES);
    this.#filled = 0;
    return rest.byteLength === 0 ? [] : [rest];
  }
}
