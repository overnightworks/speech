/** One scheduled stretch of a sound: the context time its first sample plays at, and its length. */
export interface Chunk {
  readonly start: number;
  readonly seconds: number;
}

/**
 * What a position of a sound's own seconds is, against the sound as scheduled:
 * - `passed`: the output has played it, or will have within one edge, or it is NaN or negative;
 * - `ahead`: inside the scheduled chunks, past one edge;
 * - `beyond`: within one edge of the chunks that have arrived or past them, on a sound whose body is still to come: more
 *   may arrive, or the sound may end there;
 * - `end`: at or past the sound's length, or within one edge of it, on a sound whose body has arrived whole.
 */
export type PositionKind = "passed" | "ahead" | "beyond" | "end";

/** The seconds of the chunks the output has played by `now` of the context's clock. */
export function playedBy(chunks: readonly Chunk[], now: number): number {
  let played = 0;
  for (const { start, seconds } of chunks) {
    played += Math.min(Math.max(now - start, 0), seconds);
  }
  return played;
}

export function lengthOf(chunks: readonly Chunk[]): number {
  return chunks.reduce((length, { seconds }) => length + seconds, 0);
}

/**
 * The inverse of `playedBy`: the earliest context time at which it reaches `position` (above 0), inside the chunks'
 * content and never into a gap between them. Undefined past the chunks.
 */
export function contextTimeAt(chunks: readonly Chunk[], position: number): number | undefined {
  const bounds = [...new Set(chunks.flatMap(({ start, seconds }) => [start, start + seconds]))].sort((a, b) => a - b);
  for (let index = 1; index < bounds.length; index += 1) {
    const from = bounds[index - 1] as number;
    const to = bounds[index] as number;
    const reached = playedBy(chunks, to);
    if (reached >= position) {
      const before = playedBy(chunks, from);
      return from + ((position - before) / (reached - before)) * (to - from);
    }
  }
  return undefined;
}

/**
 * Classifies `position` for a hold asked at `now` of the context's clock. `complete` is whether the body has arrived
 * whole, so that the chunks are the sound's length; `edge` is the length of the fade. The end wins over a position
 * that is also passed.
 */
export function classify(
  chunks: readonly Chunk[],
  complete: boolean,
  position: number,
  now: number,
  edge: number,
): PositionKind {
  const length = lengthOf(chunks);
  if (complete && position >= length - edge) {
    return "end";
  }
  if (!(position > playedBy(chunks, now + edge))) {
    return "passed";
  }
  return position >= length - edge ? "beyond" : "ahead";
}
