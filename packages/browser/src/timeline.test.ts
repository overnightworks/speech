import { describe, expect, it } from "vitest";

import { classify, contextTimeAt, lengthOf, playedBy, type Chunk } from "./timeline.js";

const EDGE = 0.03;
// Two chunks of one second with a gap of half a second between them, as a late chunk leaves.
const WITH_GAP: Chunk[] = [
  { start: 10, seconds: 1 },
  { start: 11.5, seconds: 1 },
];

describe("playedBy", () => {
  it.each([
    ["before the first chunk", 9, 0],
    ["inside the first chunk", 10.25, 0.25],
    ["in the gap, where it stands still", 11.25, 1],
    ["inside the second chunk, counting content not clock time", 12, 1.5],
    ["after the last chunk", 20, 2],
  ])("reads %s", (_name, now, played) => {
    expect(playedBy(WITH_GAP, now)).toBeCloseTo(played, 12);
  });

  it("reads 0 for no chunk", () => {
    expect(playedBy([], 5)).toBe(0);
  });

  it("counts the overlap of two chunks twice, as the output plays both", () => {
    expect(
      playedBy(
        [
          { start: 0, seconds: 1 },
          { start: 0.5, seconds: 1 },
        ],
        1,
      ),
    ).toBeCloseTo(1.5, 12);
  });
});

describe("lengthOf", () => {
  it("adds the chunks", () => {
    expect(lengthOf(WITH_GAP)).toBe(2);
  });
});

describe("contextTimeAt", () => {
  it.each([
    ["inside the first chunk", 0.25, 10.25],
    ["at the first chunk's end", 1, 11],
    ["inside the second chunk, after the gap", 1.5, 12],
    ["at the end of the arrived chunks", 2, 12.5],
  ])("gives the context time of a position %s", (_name, position, time) => {
    expect(contextTimeAt(WITH_GAP, position)).toBeCloseTo(time, 12);
  });

  it("gives no time for a position beyond the arrived chunks", () => {
    expect(contextTimeAt(WITH_GAP, 2.01)).toBeUndefined();
  });

  it("gives no time for no chunks", () => {
    expect(contextTimeAt([], 0.5)).toBeUndefined();
  });

  it("is the earliest time at which two overlapping chunks have played the position", () => {
    const overlapping: Chunk[] = [
      { start: 0, seconds: 1 },
      { start: 0.5, seconds: 1 },
    ];
    expect(contextTimeAt(overlapping, 0.5)).toBeCloseTo(0.5, 12);
    // From 0.5 s on both chunks play: the position 1.25 is 0.5 played at 0.5 s and 0.75 more at twice the rate.
    expect(contextTimeAt(overlapping, 1.25)).toBeCloseTo(0.875, 12);
  });

  it.each([0.1, 0.9, 1.0, 1.7, 2.0])("inverts playedBy at %s", (position) => {
    const time = contextTimeAt(WITH_GAP, position) as number;
    expect(playedBy(WITH_GAP, time)).toBeCloseTo(position, 9);
    expect(playedBy(WITH_GAP, time - 1e-6)).toBeLessThan(position);
  });
});

describe("classify", () => {
  const WHILE_PLAYING = 10.5; // 0.5 s into the first chunk
  const whole = (position: number, now = WHILE_PLAYING) => classify(WITH_GAP, true, position, now, EDGE);
  const streaming = (position: number, now = WHILE_PLAYING) => classify(WITH_GAP, false, position, now, EDGE);

  it.each([
    ["a position played already", 0.2],
    ["a position closer than one edge", 0.5 + EDGE / 2],
    ["the position the output has reached", 0.5],
    ["a negative position", -3],
    ["NaN", Number.NaN],
  ])("calls %s passed", (_name, position) => {
    expect(whole(position)).toBe("passed");
    expect(streaming(position)).toBe("passed");
  });

  it("calls a position past one edge ahead", () => {
    expect(whole(0.5 + 2 * EDGE)).toBe("ahead");
    expect(streaming(1.9)).toBe("ahead");
  });

  it("counts the edge in content seconds, so a late chunk's gap is not passed through", () => {
    // 0.99 s played, the next chunk 0.5 s away: the edge ends inside the gap, at content 1 s.
    expect(whole(1.2, 10.99)).toBe("ahead");
    expect(whole(1.0, 10.99)).toBe("passed");
  });

  it.each([
    ["the sound's length", 2],
    ["past it", 5],
    ["Infinity", Infinity],
    ["within one edge of it", 2 - EDGE / 2],
  ])("calls %s the end for a whole sound", (_name, position) => {
    expect(whole(position)).toBe("end");
  });

  it("lets the end win over a passed position when the sound is almost over", () => {
    expect(whole(2 - EDGE / 2, 12.5)).toBe("end");
  });

  it("calls a position past the arrived chunks of a streamed sound beyond", () => {
    expect(streaming(2.5)).toBe("beyond");
    expect(streaming(Infinity)).toBe("beyond");
  });

  it("calls a position within one edge of the arrived length of a streamed sound beyond, not the end", () => {
    expect(streaming(2)).toBe("beyond");
    expect(streaming(2 - EDGE / 2)).toBe("beyond");
    expect(streaming(2 - 2 * EDGE)).toBe("ahead");
  });
});
