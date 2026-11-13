const UNKNOWN_SIZE = 0xffffffff;
const RIFF_HEADER_BYTES = 12;
const CHUNK_HEADER_BYTES = 8;
const FORMAT_FIELDS_BYTES = 16;
const PCM_TAG = 1;
const MONO = 1;
const SAMPLE_BITS = 16;
const SAMPLE_BYTES = 2;
const FULL_SCALE = 32768;
const WAV_MEDIA_TYPE = /^audio\/(x-)?wav\b/i;

export type Samples = Float32Array<ArrayBuffer>;

type Phase = "riff" | "chunks" | "data" | "done" | "unsupported";

function tagAt(bytes: Uint8Array, offset: number): string {
  return String.fromCharCode(...bytes.subarray(offset, offset + 4));
}

/**
 * Turns the body of a 16-bit mono PCM WAV, pushed in chunks of any size, into float samples.
 *
 * The header may split anywhere; chunks before `data` (the AI mark's LIST/INFO among them) are skipped, a
 * sample split across chunks is carried, and a `data` size of 0xFFFFFFFF means the body's end is the end.
 */
export class WavPcmDecoder {
  #phase: Phase = "riff";
  #buffer = new Uint8Array(0);
  #skip = 0;
  #rate = 0;
  #remaining = 0;

  /** The rate once the `data` chunk has begun. */
  get sampleRate(): number | undefined {
    return this.#phase === "data" || this.#phase === "done" ? this.#rate : undefined;
  }

  /** True once the header shows anything but 16-bit mono PCM. */
  get unsupported(): boolean {
    return this.#phase === "unsupported";
  }

  push(bytes: Uint8Array): Samples {
    const joined = new Uint8Array(this.#buffer.length + bytes.length);
    joined.set(this.#buffer);
    joined.set(bytes, this.#buffer.length);
    this.#buffer = joined;
    const samples: Samples[] = [];
    while (this.#step(samples)) {
      // Every step consumes bytes or changes the phase.
    }
    const all = new Float32Array(samples.reduce((count, part) => count + part.length, 0));
    let offset = 0;
    for (const part of samples) {
      all.set(part, offset);
      offset += part.length;
    }
    return all;
  }

  /** Throws unless the body ended where the WAV says it ends. */
  finish(): void {
    const whole = this.#phase === "done" || (this.#phase === "data" && this.#remaining === Infinity);
    if (!whole) {
      throw new Error("the WAV body ended before its data did");
    }
  }

  #consume(count: number): void {
    this.#buffer = this.#buffer.subarray(count);
  }

  #step(samples: Samples[]): boolean {
    const buffer = this.#buffer;
    switch (this.#phase) {
      case "riff":
        if (buffer.length < RIFF_HEADER_BYTES) {
          return false;
        }
        this.#phase = tagAt(buffer, 0) === "RIFF" && tagAt(buffer, 8) === "WAVE" ? "chunks" : "unsupported";
        this.#consume(RIFF_HEADER_BYTES);
        return true;
      case "chunks":
        return this.#stepChunk();
      case "data": {
        const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.length);
        const count = Math.floor(Math.min(buffer.length, this.#remaining) / SAMPLE_BYTES);
        const part = new Float32Array(count);
        for (let index = 0; index < count; index += 1) {
          part[index] = view.getInt16(index * SAMPLE_BYTES, true) / FULL_SCALE;
        }
        samples.push(part);
        this.#consume(count * SAMPLE_BYTES);
        this.#remaining -= count * SAMPLE_BYTES;
        if (this.#remaining < SAMPLE_BYTES) {
          this.#phase = "done";
        }
        return false;
      }
      default:
        return false;
    }
  }

  #stepChunk(): boolean {
    const dropped = Math.min(this.#skip, this.#buffer.length);
    this.#consume(dropped);
    this.#skip -= dropped;
    const buffer = this.#buffer;
    if (this.#skip > 0 || buffer.length < CHUNK_HEADER_BYTES) {
      return false;
    }
    const id = tagAt(buffer, 0);
    const size = new DataView(buffer.buffer, buffer.byteOffset, buffer.length).getUint32(4, true);
    if (id === "data") {
      this.#phase = this.#rate === 0 ? "unsupported" : "data";
      this.#remaining = size === UNKNOWN_SIZE ? Infinity : size;
      this.#consume(CHUNK_HEADER_BYTES);
    } else if (id === "fmt ") {
      if (buffer.length < CHUNK_HEADER_BYTES + size) {
        return false;
      }
      const fields = new DataView(buffer.buffer, buffer.byteOffset + CHUNK_HEADER_BYTES, size);
      const plain =
        size >= FORMAT_FIELDS_BYTES &&
        fields.getUint16(0, true) === PCM_TAG &&
        fields.getUint16(2, true) === MONO &&
        fields.getUint16(14, true) === SAMPLE_BITS;
      this.#phase = plain ? "chunks" : "unsupported";
      this.#rate = plain ? fields.getUint32(4, true) : 0;
      this.#consume(CHUNK_HEADER_BYTES + size);
      this.#skip = size % 2;
    } else {
      this.#consume(CHUNK_HEADER_BYTES);
      this.#skip = size + (size % 2);
    }
    return true;
  }
}

/** The samples of a streamed WAV as they arrive. */
export class PcmStream {
  readonly sampleRate: number;
  #first: Samples | undefined;

  constructor(
    sampleRate: number,
    first: Samples,
    private readonly reader: ReadableStreamDefaultReader<Uint8Array>,
    private readonly decoder: WavPcmDecoder,
  ) {
    this.sampleRate = sampleRate;
    this.#first = first.length > 0 ? first : undefined;
  }

  /** The next samples, or undefined once the body ended as the WAV says. Throws when the body broke off. */
  async next(): Promise<Samples | undefined> {
    if (this.#first !== undefined) {
      const first = this.#first;
      this.#first = undefined;
      return first;
    }
    for (;;) {
      const { done, value } = await this.reader.read();
      if (done) {
        this.decoder.finish();
        return undefined;
      }
      const samples = this.decoder.push(value);
      if (samples.length > 0) {
        return samples;
      }
    }
  }
}

export type OpenedSound =
  | { readonly kind: "stream"; readonly stream: PcmStream }
  | { readonly kind: "whole"; readonly bytes: ArrayBuffer };

/**
 * Reads a response up to its first samples when it is a 16-bit mono PCM WAV; any other answer is read whole.
 * Throws when the body breaks off before the header is complete.
 */
export async function openSound(response: Response): Promise<OpenedSound> {
  if (response.body === null || !WAV_MEDIA_TYPE.test(response.headers.get("content-type") ?? "")) {
    return { kind: "whole", bytes: await response.arrayBuffer() };
  }
  const reader = response.body.getReader();
  const decoder = new WavPcmDecoder();
  const head: Uint8Array[] = [];
  let first: Samples = new Float32Array(0);
  while (decoder.sampleRate === undefined && !decoder.unsupported) {
    const { done, value } = await reader.read();
    if (done) {
      throw new Error("the WAV body ended inside its header");
    }
    head.push(value);
    first = decoder.push(value);
  }
  if (decoder.sampleRate !== undefined) {
    return { kind: "stream", stream: new PcmStream(decoder.sampleRate, first, reader, decoder) };
  }
  for (let next = await reader.read(); !next.done; next = await reader.read()) {
    head.push(next.value);
  }
  return { kind: "whole", bytes: await new Blob(head.map((part) => part.slice())).arrayBuffer() };
}
