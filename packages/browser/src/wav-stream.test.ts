import { describe, expect, it } from "vitest";

import { openSound, WavPcmDecoder } from "./wav-stream.js";

const RATE = 22_050;
const UNKNOWN = 0xffffffff;

function ascii(text: string): number[] {
  return [...text].map((letter) => letter.charCodeAt(0));
}

function u16(value: number): number[] {
  return [value & 0xff, (value >> 8) & 0xff];
}

function u32(value: number): number[] {
  return [...u16(value & 0xffff), ...u16(Math.floor(value / 0x10000))];
}

function chunk(id: string, body: number[]): number[] {
  return [...ascii(id), ...u32(body.length), ...body, ...(body.length % 2 === 1 ? [0] : [])];
}

function format(channels = 1, bits = 16, tag = 1, extra: number[] = []): number[] {
  return chunk("fmt ", [...u16(tag), ...u16(channels), ...u32(RATE), ...u32(RATE * 2), ...u16(2), ...u16(bits), ...extra]);
}

function pcm(samples: number[]): number[] {
  return samples.flatMap((sample) => u16(sample & 0xffff));
}

/** A WAV with the AI mark before `data`; `dataSize` overrides the size the header states. */
function wav(samples: number[], options: { dataSize?: number; before?: number[]; after?: number[] } = {}): Uint8Array {
  const data = pcm(samples);
  const header = [...ascii("data"), ...u32(options.dataSize ?? data.length)];
  const body = [...ascii("WAVE"), ...format(), ...(options.before ?? chunk("LIST", [...ascii("INFO"), ...chunk("ICMT", ascii("AI"))])), ...header, ...data, ...(options.after ?? [])];
  return Uint8Array.from([...ascii("RIFF"), ...u32(options.dataSize === UNKNOWN ? UNKNOWN : body.length), ...body]);
}

function decodedIn(bytes: Uint8Array, sizes: number[]): { samples: number[]; decoder: WavPcmDecoder } {
  const decoder = new WavPcmDecoder();
  const samples: number[] = [];
  let offset = 0;
  for (let index = 0; offset < bytes.length; index += 1) {
    const size = sizes[index % sizes.length] ?? 1;
    samples.push(...decoder.push(bytes.subarray(offset, offset + size)));
    offset += size;
  }
  return { samples, decoder };
}

const SAMPLES = [0, 16384, -16384, -32768, 32767, 1];
const EXPECTED = [0, 0.5, -0.5, -1, 32767 / 32768, 1 / 32768];

function stream(parts: Uint8Array[], failure?: Error): ReadableStream<Uint8Array> {
  let next = 0;
  return new ReadableStream<Uint8Array>({
    pull(controller) {
      const part = parts[next];
      next += 1;
      if (part !== undefined) {
        controller.enqueue(part);
      } else if (failure !== undefined) {
        controller.error(failure);
      } else {
        controller.close();
      }
    },
  });
}

function answer(parts: Uint8Array[], type = "audio/wav", failure?: Error): Response {
  return new Response(stream(parts, failure), { headers: { "content-type": type } });
}

describe("WavPcmDecoder", () => {
  it.each([[[10_000]], [[1]], [[2]], [[3]], [[5, 1, 9, 2]], [[7]]])("reads the same samples whatever the chunk sizes %j", (sizes) => {
    const { samples, decoder } = decodedIn(wav(SAMPLES), sizes);

    expect(samples).toEqual(EXPECTED);
    expect(decoder.sampleRate).toBe(RATE);
    expect(() => decoder.finish()).not.toThrow();
  });

  it("carries an odd trailing byte into the next chunk", () => {
    const bytes = wav(SAMPLES);
    const decoder = new WavPcmDecoder();
    const cut = bytes.length - 3;

    const first = decoder.push(bytes.subarray(0, cut));
    const second = decoder.push(bytes.subarray(cut));

    expect([...first, ...second]).toEqual(EXPECTED);
    expect(first.length).toBe(SAMPLES.length - 2);
  });

  it("reports no rate before the data chunk begins", () => {
    const decoder = new WavPcmDecoder();
    const bytes = wav(SAMPLES);

    decoder.push(bytes.subarray(0, 30));

    expect(decoder.sampleRate).toBeUndefined();
    expect(decoder.unsupported).toBe(false);
  });

  it("takes the end of the body as the end of data when the size is the unknown marker", () => {
    const { samples, decoder } = decodedIn(wav(SAMPLES, { dataSize: UNKNOWN }), [4]);

    expect(samples).toEqual(EXPECTED);
    expect(() => decoder.finish()).not.toThrow();
  });

  it("stops at the stated size and ignores what follows the data", () => {
    const { samples, decoder } = decodedIn(wav(SAMPLES, { after: chunk("junk", [1, 2, 3]) }), [6]);

    expect(samples).toEqual(EXPECTED);
    expect(() => decoder.finish()).not.toThrow();
  });

  it("skips an odd-sized chunk and its pad byte before the data", () => {
    const { samples } = decodedIn(wav(SAMPLES, { before: chunk("odd ", [1, 2, 3]) }), [1]);

    expect(samples).toEqual(EXPECTED);
  });

  it("fails to finish when the body ends before the stated data size", () => {
    const { decoder } = decodedIn(wav(SAMPLES).subarray(0, -4), [8]);

    expect(() => decoder.finish()).toThrow();
  });

  it("fails to finish when the body ends inside the header", () => {
    const { decoder } = decodedIn(wav(SAMPLES).subarray(0, 20), [8]);

    expect(() => decoder.finish()).toThrow();
  });

  it.each([
    ["stereo", Uint8Array.from([...ascii("RIFF"), ...u32(0), ...ascii("WAVE"), ...format(2)])],
    ["eight-bit", Uint8Array.from([...ascii("RIFF"), ...u32(0), ...ascii("WAVE"), ...format(1, 8)])],
    ["float", Uint8Array.from([...ascii("RIFF"), ...u32(0), ...ascii("WAVE"), ...format(1, 16, 3)])],
    ["a short format chunk", Uint8Array.from([...ascii("RIFF"), ...u32(0), ...ascii("WAVE"), ...chunk("fmt ", [1, 0])])],
    ["another container", Uint8Array.from([...ascii("OggS"), ...u32(0), ...ascii("WAVE"), ...format()])],
    ["data before the format", Uint8Array.from([...ascii("RIFF"), ...u32(0), ...ascii("WAVE"), ...ascii("data"), ...u32(4)])],
  ])("calls %s unsupported", (_name, bytes) => {
    const { decoder } = decodedIn(bytes, [5]);

    expect(decoder.unsupported).toBe(true);
    expect(decoder.sampleRate).toBeUndefined();
  });
});

describe("openSound", () => {
  it("streams a WAV whose header and samples arrive in pieces", async () => {
    const bytes = wav(SAMPLES);
    const opened = await openSound(answer([bytes.subarray(0, 5), bytes.subarray(5, 45), bytes.subarray(45, 47), bytes.subarray(47)]));

    expect(opened.kind).toBe("stream");
    if (opened.kind !== "stream") {
      return;
    }
    const heard: number[] = [];
    for (let samples = await opened.stream.next(); samples !== undefined; samples = await opened.stream.next()) {
      heard.push(...samples);
    }
    expect(opened.stream.sampleRate).toBe(RATE);
    expect(heard).toEqual(EXPECTED);
  });

  it("hands over the samples that came with the header first and waits through chunks that hold none", async () => {
    const bytes = wav(SAMPLES);
    const header = bytes.length - SAMPLES.length * 2;
    const opened = await openSound(answer([bytes.subarray(0, header + 4), bytes.subarray(header + 4, header + 5), bytes.subarray(header + 5)]));
    if (opened.kind !== "stream") {
      throw new Error("not streamed");
    }

    const parts = [];
    for (let samples = await opened.stream.next(); samples !== undefined; samples = await opened.stream.next()) {
      parts.push(samples.length);
    }

    expect(parts).toEqual([2, 4]);
  });

  it("ends the stream with an error when the body breaks off after the header", async () => {
    const bytes = wav(SAMPLES);
    const opened = await openSound(answer([bytes.subarray(0, bytes.length - 4)], "audio/wav", new Error("cut")));
    if (opened.kind !== "stream") {
      throw new Error("not streamed");
    }

    await expect(async () => {
      for (let samples = await opened.stream.next(); samples !== undefined; samples = await opened.stream.next()) {
        // Reads to the break.
      }
    }).rejects.toThrow("cut");
  });

  it("ends the stream with an error when the body ends short of its stated data", async () => {
    const bytes = wav(SAMPLES);
    const opened = await openSound(answer([bytes.subarray(0, bytes.length - 4)]));
    if (opened.kind !== "stream") {
      throw new Error("not streamed");
    }

    await opened.stream.next();
    await expect(opened.stream.next()).rejects.toThrow();
  });

  it("throws when the body ends inside the header", async () => {
    await expect(openSound(answer([wav(SAMPLES).subarray(0, 20)]))).rejects.toThrow();
  });

  it.each([["audio/ogg"], ["application/octet-stream"]])("reads %j whole", async (type) => {
    const bytes = wav(SAMPLES);
    const opened = await openSound(answer([bytes], type));

    expect(opened).toEqual({ kind: "whole", bytes: bytes.buffer.slice(0) });
  });

  it("reads an answer that names no type whole", async () => {
    const bytes = wav(SAMPLES);
    const opened = await openSound(new Response(bytes));

    expect(opened.kind).toBe("whole");
  });

  it("starts streaming when the header ends exactly at a chunk boundary", async () => {
    const bytes = wav(SAMPLES);
    const header = bytes.length - SAMPLES.length * 2;
    const opened = await openSound(answer([bytes.subarray(0, header), bytes.subarray(header)]));
    if (opened.kind !== "stream") {
      throw new Error("not streamed");
    }

    expect([...((await opened.stream.next()) ?? [])]).toEqual(EXPECTED);
  });

  it("reads an answer without a body whole", async () => {
    const opened = await openSound(new Response(null, { headers: { "content-type": "audio/wav" } }));

    expect(opened).toEqual({ kind: "whole", bytes: new ArrayBuffer(0) });
  });

  it("reads a WAV that is not 16-bit mono whole, with the bytes already read", async () => {
    const stereo = Uint8Array.from([...ascii("RIFF"), ...u32(0), ...ascii("WAVE"), ...format(2), ...chunk("data", [1, 2, 3, 4])]);
    const opened = await openSound(answer([stereo.subarray(0, 14), stereo.subarray(14, 50), stereo.subarray(50)]));

    expect(opened.kind).toBe("whole");
    if (opened.kind === "whole") {
      expect(new Uint8Array(opened.bytes)).toEqual(stereo);
    }
  });
});
