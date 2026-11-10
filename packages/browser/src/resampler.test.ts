import { describe, expect, it } from "vitest";

import { HOP_FRAME_BYTES, SAMPLE_RATE } from "./hop.js";
import { Framer, Resampler } from "./resampler.js";

const RENDER_QUANTUM = 128;
const ODD_SIZES = [1, 7, 333, 129, 2];
// A tenth of a second at the output rate holds a whole number of cycles of every tone measured here.
const MEASURED = SAMPLE_RATE / 10;

function tone(rate: number, frequency: number, seconds: number, amplitude = 0.5): Float32Array {
  return Float32Array.from(
    { length: Math.round(rate * seconds) },
    (_, index) => amplitude * Math.sin((2 * Math.PI * frequency * index) / rate),
  );
}

function concatenated(parts: Float32Array[]): Float32Array {
  const whole = new Float32Array(parts.reduce((length, part) => length + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    whole.set(part, offset);
    offset += part.length;
  }
  return whole;
}

function resampled(rate: number, input: Float32Array, sizes: number[]): Float32Array {
  const resampler = new Resampler(rate);
  const parts: Float32Array[] = [];
  for (let start = 0, turn = 0; start < input.length; turn += 1) {
    const size = sizes[turn % sizes.length] as number;
    parts.push(resampler.push(input.subarray(start, start + size)));
    start += size;
  }
  parts.push(resampler.flush());
  return concatenated(parts);
}

// The middle tenth of a second, clear of the filter's ramp at either end.
function middle(output: Float32Array): Float32Array {
  return output.subarray(MEASURED, 2 * MEASURED);
}

function rms(samples: Float32Array): number {
  return Math.sqrt(samples.reduce((sum, sample) => sum + sample * sample, 0) / samples.length);
}

function amplitudeAt(samples: Float32Array, frequency: number): number {
  let real = 0;
  let imaginary = 0;
  samples.forEach((sample, index) => {
    const angle = (2 * Math.PI * frequency * index) / SAMPLE_RATE;
    real += sample * Math.cos(angle);
    imaginary -= sample * Math.sin(angle);
  });
  return (2 * Math.hypot(real, imaginary)) / samples.length;
}

function decibels(ratio: number): number {
  return 20 * Math.log10(ratio);
}

describe("Resampler", () => {
  it.each([48_000, 44_100, 96_000, 16_000])(
    "gives the same %i Hz input the same output in render quanta, odd sizes and one block",
    (rate) => {
      const input = concatenated([tone(rate, 1000, 0.5), tone(rate, 3100, 0.25, 0.2)]);

      const whole = resampled(rate, input, [input.length]);

      expect(whole.length).toBe(Math.ceil((input.length * SAMPLE_RATE) / rate));
      expect(resampled(rate, input, [RENDER_QUANTUM])).toEqual(whole);
      expect(resampled(rate, input, ODD_SIZES)).toEqual(whole);
    },
  );

  it.each([48_000, 44_100, 96_000, 16_000])("passes a 1 kHz tone at %i Hz input within 1 dB", (rate) => {
    const output = middle(resampled(rate, tone(rate, 1000, 0.5), [RENDER_QUANTUM]));

    expect(Math.abs(decibels(amplitudeAt(output, 1000) / 0.5))).toBeLessThan(1);
  });

  it.each([48_000, 44_100])("damps a 15 kHz tone at %i Hz input by at least 50 dB", (rate) => {
    const input = tone(rate, 15_000, 0.5);

    const output = middle(resampled(rate, input, [RENDER_QUANTUM]));

    expect(decibels(rms(output) / rms(input))).toBeLessThan(-50);
  });

  it("damps the 10 kHz image of a 6 kHz tone at 16 kHz input by at least 50 dB", () => {
    const output = middle(resampled(16_000, tone(16_000, 6000, 0.5), [RENDER_QUANTUM]));

    expect(decibels(amplitudeAt(output, 10_000) / 0.5)).toBeLessThan(-50);
  });
});

describe("Framer", () => {
  function bytesOf(frames: ArrayBuffer[]): number[] {
    return frames.flatMap((frame) => [...new Uint8Array(frame)]);
  }

  it("cuts frames of whole samples across blocks, every one but the last of the hop's frame size", () => {
    const framer = new Framer();
    const blocks = [new Float32Array(1000), new Float32Array(1401), new Float32Array(2400), new Float32Array(2403)];

    const frames = [...blocks.flatMap((block) => framer.push(block)), ...framer.flush()];

    expect(frames.map((frame) => frame.byteLength)).toEqual([HOP_FRAME_BYTES, HOP_FRAME_BYTES, HOP_FRAME_BYTES, 8]);
  });

  it("writes PCM16 little-endian, clamped to the full scale", () => {
    const framer = new Framer();
    framer.push(Float32Array.of(0.5, -1, 2, -2));

    expect(bytesOf(framer.flush())).toEqual([0x00, 0x40, 0x00, 0x80, 0xff, 0x7f, 0x00, 0x80]);
  });

  it("leaves no frame at flush when no sample is left", () => {
    const framer = new Framer();

    expect(framer.flush()).toEqual([]);
  });
});
