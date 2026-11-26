// The stand-in app's page for the player. `window.playerPage` is what the test drives and reads.
//
// The measuring instrument wraps the Web Audio entry points before the page creates its player, and keeps
// the originals: what it wraps it records (edges to the output, sources, decodes, resumes), and its own
// analyser, built with the originals, reads the energy of what reaches the output.
import * as entry from "/dist/index.js";

const Page = {
  AnalyserNode,
  AudioBufferSourceNode,
  AudioContext,
  AudioDestinationNode,
  GainNode,
  connect: AudioNode.prototype.connect,
  clock: Object.getOwnPropertyDescriptor(BaseAudioContext.prototype, "currentTime"),
  createAnalyser: BaseAudioContext.prototype.createAnalyser,
  rampTo: AudioParam.prototype.linearRampToValueAtTime,
  createBufferSource: BaseAudioContext.prototype.createBufferSource,
  decodeAudioData: BaseAudioContext.prototype.decodeAudioData,
  resume: AudioContext.prototype.resume,
  start: AudioBufferSourceNode.prototype.start,
};
const JUNK_BYTES = 512;
// How far past a start the real clock is let run in the stall around a source's start, in render quanta.
const STALL_QUANTA = 6;
const KINDS = [
  ["GainNode", Page.GainNode],
  ["AudioBufferSourceNode", Page.AudioBufferSourceNode],
  ["AudioDestinationNode", Page.AudioDestinationNode],
  ["AnalyserNode", Page.AnalyserNode],
];

const seen = {
  exportNames: Object.keys(entry),
  sounds: [],
  log: [],
  errors: [],
  edges: [],
  bytes: {},
  analysersCreated: 0,
  sourcesCreated: 0,
  sourcesStarted: 0,
  scheduledChunks: [],
  stalls: [],
  sourcesEnded: 0,
  instrumentStops: 0,
  decodesCalled: 0,
  decodesSettled: 0,
  resumesCalled: 0,
  endedAtAction: null,
  playedSeries: [],
  heardOnsets: [],
  heardPeaks: [],
  contextAtAction: null,
  actions: [],
  ramps: [],
  stateChanges: [],
  pauseAts: [],
};
const contexts = [];
const clockReads = new Map();
const frozenClocks = new Map();
const nodeIds = new WeakMap();
const trackedSources = new WeakSet();
const held = { decode: [], resume: [] };
const heldNow = [];
const byteStore = new Map();
const fedBodies = new Map();
const feeders = new Map();
const playbacks = [];
let nextNodeId = 0;
let stopOnSourceEnd = false;
let switchRunning = false;
let rejectNextResume = false;
// Where the next source's start stalls the page: "before" the call reaches the render thread, or "after" it.
let stallNextStart = null;
let outputGain = null;
let tapMeter = null;
let player = null;

addEventListener("error", (event) => seen.errors.push(String(event.message)));
addEventListener("unhandledrejection", (event) => seen.errors.push(String(event.reason)));

function idOf(node) {
  if (!nodeIds.has(node)) nodeIds.set(node, nextNodeId++);
  return nodeIds.get(node);
}

function kindOf(node) {
  return KINDS.find(([, constructor]) => node instanceof constructor)?.[0] ?? node?.constructor?.name;
}

const earsReady = [];
// The first frame of the latest render quantum the ear has reported, and who waits for a later one.
let heardFrame = -1;
const earWaiters = [];

function heardThrough(frame) {
  if (heardFrame >= frame) return Promise.resolve();
  return new Promise((resolve) => earWaiters.push({ frame, resolve }));
}

// The page's own ear: a worklet in the graph (`ear_processor.js`) that notes the context time of the first sample
// of each stretch of sound that reaches the output and the peak of each render quantum.
function listenTo(output) {
  const context = output.context;
  earsReady.push(
    context.audioWorklet.addModule("/ear_processor.js").then(() => {
      const ear = new AudioWorkletNode(context, "ear");
      ear.port.onmessage = ({ data }) => {
        if (data.onset !== undefined) seen.heardOnsets.push(data.onset);
        else {
          seen.heardPeaks.push([data.quantum, data.peak]);
          heardFrame = Math.round(data.quantum * context.sampleRate);
          earWaiters.splice(0).forEach((waiter) => (waiter.frame <= heardFrame ? waiter.resolve() : earWaiters.push(waiter)));
        }
      };
      Page.connect.call(output, ear);
      Page.connect.call(ear, context.destination);
    }),
  );
}

function trackSource(source) {
  if (trackedSources.has(source)) return source;
  trackedSources.add(source);
  seen.sourcesCreated += 1;
  // Added at creation, so it runs before the `onended` the package sets.
  source.addEventListener("ended", () => {
    seen.sourcesEnded += 1;
    if (stopOnSourceEnd) {
      const next = stopOnSourceEnd;
      stopOnSourceEnd = false;
      seen.instrumentStops += 1;
      player.stop();
      if (next !== true) handOver(next);
    }
  });
  return source;
}

function gateFor(kind, result) {
  const gate = held[kind].shift();
  if (gate === undefined) return result;
  heldNow.push(gate);
  return gate.promise.then(() => result);
}

// The page's task spins, as a task does that the machine does not run, until the render thread is `STALL_QUANTA` past
// `when`; the stall is recorded as the span of the context's clock the task was blocked.
function stallPast(context, when) {
  const from = Page.clock.get.call(context);
  const until = when + (STALL_QUANTA * context.renderQuantumSize) / context.sampleRate;
  while (Page.clock.get.call(context) < until);
  seen.stalls.push({ from, to: Page.clock.get.call(context) });
}

// The context clock advances on the audio thread, and under load two reads in one task straddle any number of its
// steps. While the app throws a switch (`act`) the instrument holds the first read of a context, so that the player
// and the page read the same instant when they act in one task. Every other read is the clock's own, which
// moves on as the page works.
function readClock(context) {
  if (frozenClocks.has(context)) return frozenClocks.get(context);
  if (!switchRunning) return Page.clock.get.call(context);
  if (!clockReads.has(context)) clockReads.set(context, Page.clock.get.call(context));
  return clockReads.get(context);
}

Object.defineProperty(BaseAudioContext.prototype, "currentTime", {
  configurable: true,
  get() {
    return readClock(this);
  },
});
window.AudioContext = class AudioContext extends Page.AudioContext {
  constructor(...parameters) {
    super(...parameters);
    contexts.push(this);
    this.addEventListener("statechange", () =>
      seen.stateChanges.push({ state: this.state, at: Page.clock.get.call(this) }),
    );
  }
};
window.AudioBufferSourceNode = class AudioBufferSourceNode extends Page.AudioBufferSourceNode {
  constructor(...parameters) {
    super(...parameters);
    trackSource(this);
  }
};
window.AnalyserNode = class AnalyserNode extends Page.AnalyserNode {
  constructor(...parameters) {
    super(...parameters);
    seen.analysersCreated += 1;
  }
};
BaseAudioContext.prototype.createBufferSource = function () {
  return trackSource(Page.createBufferSource.call(this));
};
BaseAudioContext.prototype.createAnalyser = function () {
  seen.analysersCreated += 1;
  return Page.createAnalyser.call(this);
};
// The volume's ramps as the player sets them: where each ends up and when, on the context's clock.
AudioParam.prototype.linearRampToValueAtTime = function (target, endsAt) {
  if (this === outputGain?.gain) seen.ramps.push({ target, endsAt });
  return Page.rampTo.call(this, target, endsAt);
};
Page.AudioBufferSourceNode.prototype.start = function (...parameters) {
  seen.sourcesStarted += 1;
  seen.scheduledChunks.push({ start: parameters[0], seconds: this.buffer.duration });
  const stall = stallNextStart;
  stallNextStart = null;
  if (stall === "before") stallPast(this.context, parameters[0]);
  const result = Page.start.apply(this, parameters);
  if (stall === "after") stallPast(this.context, parameters[0]);
  return result;
};
AudioNode.prototype.connect = function (target, ...parameters) {
  const toOutput = target instanceof Page.AudioDestinationNode;
  seen.edges.push({ from: idOf(this), fromKind: kindOf(this), to: target instanceof AudioNode ? idOf(target) : null, toKind: kindOf(target) });
  if (toOutput && outputGain === null) {
    outputGain = this;
    listenTo(this);
  }
  return Page.connect.call(this, target, ...parameters);
};
BaseAudioContext.prototype.decodeAudioData = function (...parameters) {
  seen.decodesCalled += 1;
  const result = Page.decodeAudioData.apply(this, parameters);
  const settle = () => (seen.decodesSettled += 1);
  result.then(settle, settle);
  return gateFor("decode", result);
};
AudioContext.prototype.resume = function () {
  seen.resumesCalled += 1;
  if (rejectNextResume) {
    rejectNextResume = false;
    return Promise.reject(new DOMException("refused", "NotAllowedError"));
  }
  return gateFor("resume", Page.resume.call(this));
};

function tapRms() {
  if (tapMeter === null) return null;
  const samples = new Float32Array(tapMeter.fftSize);
  tapMeter.getFloatTimeDomainData(samples);
  return Math.sqrt(samples.reduce((sum, value) => sum + value * value, 0) / samples.length);
}

// The loudest the tap's analyser read while each sound plays, after each step of the context: the analyser reflects
// the last rendered quantum.
function noteTap() {
  const level = tapRms();
  if (level === null) return;
  for (const sound of seen.sounds) {
    if (sound.startedValue === "playing" && sound.endedValue === null) sound.peakTapRms = Math.max(sound.peakTapRms, level);
  }
}

// What the first sound's read shows after each step of the context, each switch of the app and each end: a read that
// steps back between two of them shows. A moment that reads nothing new adds nothing, so the series stays as long as
// the read's own changes.
function notePlayed() {
  if (playbacks.length > 0) {
    const played = playbacks[0].playedSeconds();
    if (played !== seen.playedSeries.at(-1)) seen.playedSeries.push(played);
  }
}

// A body the page feeds itself: its first `lead` bytes are there at once and `feedWhile` hands on the rest, so that
// the first samples reach the player in the task of a switch, with no timing between them.
const pageFetch = window.fetch;
window.fetch = (input, init) => {
  const body = fedBodies.get(String(input));
  if (body === undefined) return pageFetch(input, init);
  return Promise.resolve(new Response(body, { headers: { "content-type": "audio/wav" } }));
};

function fedUrl({ fed, lead }) {
  const bytes = new Uint8Array(byteStore.get(fed));
  const address = new URL(`/fed/${fed}`, location.href).href;
  fedBodies.set(
    address,
    new ReadableStream({
      start(controller) {
        controller.enqueue(bytes.slice(0, lead));
        feeders.set(fed, () => {
          controller.enqueue(bytes.slice(lead));
          controller.close();
        });
      },
    }),
  );
  return address;
}

function argumentOf(spec) {
  if (spec.fed !== undefined) return fedUrl(spec);
  if (spec.url !== undefined) return spec.url;
  if (spec.urlObject !== undefined) return new URL(spec.urlObject, location.href);
  if (spec.bytes !== undefined) return byteStore.get(spec.bytes);
  return new Uint8Array(JUNK_BYTES).fill(7).buffer;
}

function handOver(spec) {
  const sound = { id: seen.sounds.length, startedValue: null, startedCount: 0, endedValue: null, endedCount: 0, endedAt: null, peakTapRms: 0, playedSeconds: null, start: null };
  seen.sounds.push(sound);
  const playback = player.play(argumentOf(spec));
  playbacks.push(playback);
  playback.started.then((value) => {
    sound.startedValue = value.kind;
    sound.start = value.kind === "playing" ? { contextTime: value.contextTime, onThePlayersContext: value.context === contexts[0] } : null;
    sound.startedCount += 1;
    seen.log.push({ event: "started", id: sound.id, value: value.kind, at: performance.now() });
  });
  playback.ended.then((end) => {
    sound.endedValue = end.kind;
    sound.endedCount += 1;
    sound.endedAt = performance.now();
    sound.playedSeconds = end.playedSeconds ?? null;
    seen.log.push({ event: "ended", id: sound.id, value: end.kind, at: sound.endedAt });
    if (spec.bytes !== undefined) seen.bytes[spec.bytes].after = byteStore.get(spec.bytes).byteLength;
    notePlayed();
    window.noteEnded?.(sound.id);
  });
  return sound.id;
}

// A switch of the app: it records when it happened and what had ended by the end of its own task. The held clock read
// is dropped first and the switch's time is the read the player made: a read from before, held across a slow page,
// would start the player's volume ramp in the past, and the ear would hear a ramp shorter than the player's edge.
async function act(name, action) {
  clockReads.clear();
  switchRunning = true;
  try {
    action();
    seen.contextAtAction = contexts[0].currentTime;
    seen.actions.push({ name, at: seen.contextAtAction });
    // The page's handlers of the sounds the switch ended run in this tick and read the clock the player read: a fresh
    // read could lie any number of audio steps past it.
    await Promise.resolve();
    seen.endedAtAction = seen.sounds.map((sound) => sound.endedValue);
    notePlayed();
  } finally {
    switchRunning = false;
    clockReads.clear();
  }
}

window.playerPage = {
  seen,
  play: (...specs) => specs.map(handOver),
  prepareBytes: async (key, url) => {
    const bytes = await (await fetch(url)).arrayBuffer();
    byteStore.set(key, bytes);
    seen.bytes[key] = { before: bytes.byteLength, after: null };
  },
  stop: () => act("stop", () => player.stop()),
  pause: () => act("pause", () => player.pause()),
  resume: () => act("resume", () => player.resume()),
  // The switches of one task: each runs after the one before, with no render quantum of its own between them.
  inOneTask: (...names) => act(names.join("+"), () => names.forEach((name) => player[name]())),
  // The rest of the fed body `key`, in one task.
  feed: (key) => act("feed", () => feeders.get(key)()),
  // The switch `name` and the rest of the fed body `key`, in one task.
  feedWhile: (key, name) => act(`${name}+feed`, () => { player[name](); feeders.get(key)(); }),
  mute: (muted) => act("mute", () => player.setMuted(muted)),
  // `pauseAt` of the sound `target` (its id, or "foreign" for a playback of another player, made here), recorded with its result.
  pauseAt: (target, seconds) =>
    act("pauseAt", () => {
      const call = { target, seconds, result: null, resolvedAt: null };
      seen.pauseAts.push(call);
      const playback = target === "foreign" ? entry.createPlayer().play(argumentOf({})) : playbacks[target];
      player.pauseAt(playback, seconds).then((result) => {
        call.result = result;
        call.resolvedAt = Page.clock.get.call(contexts[0]);
        notePlayed();
      });
    }),
  close: () => act("close", () => player.close()),
  // A stop, then `cut` once the context clock reads `secondsLater` on: the cut comes after the stop's edge, before the
  // player's own end of the sound.
  stopThenCutPastItsEdge: (cut, secondsLater) =>
    act("stop+cut", () => {
      player.stop();
      frozenClocks.set(contexts[0], contexts[0].currentTime + secondsLater);
      try {
        if (cut === "mute") player.setMuted(true);
        else player.close();
      } finally {
        frozenClocks.clear();
      }
    }),
  // A stop, then what the first sound reads once the context clock reads `secondsLater` on: the read in the stop's edge.
  readAfterStopPast: (secondsLater) =>
    act("stop+read", () => {
      player.stop();
      frozenClocks.set(contexts[0], contexts[0].currentTime + secondsLater);
      try {
        seen.readPastStop = playbacks[0].playedSeconds();
      } finally {
        frozenClocks.clear();
      }
    }),
  listening: () => Promise.all(earsReady),
  // Renders `quanta` render quanta of a stepped context, once the ear has reported them; the frames rendered so far.
  step: async (quanta) => {
    const frames = await contexts[0].step(quanta);
    await heardThrough(frames - contexts[0].renderQuantumSize);
    notePlayed();
    noteTap();
    return { frames, sampleRate: contexts[0].sampleRate };
  },
  // The app's reader: one analyser of the player's context, connected to `player.tap` from the outside.
  readTap: () => {
    tapMeter = new Page.AnalyserNode(player.tap.context);
    player.tap.connect(tapMeter);
  },
  hold: (kind) => held[kind].push(Promise.withResolvers()),
  release: () => heldNow.splice(0).forEach((gate) => gate.resolve()),
  rejectNextResume: () => (rejectNextResume = true),
  // The next source's start finds the page stalled: `where` is "before" the start reaches the render thread, which then
  // plays the source from the quantum it is on, or "after" it, which leaves the start asked for.
  stallNextStart: (where) => (stallNextStart = where),
  stopOnNextSourceEnd: (next) => (stopOnSourceEnd = next ?? true),
  // One read of the clock serves the whole snapshot, so that its times are of one instant.
  snapshot: () => {
    clockReads.clear();
    switchRunning = true;
    try {
      return {
        ...seen,
        contextStates: contexts.map((context) => context.state),
        sampleRate: contexts[0].sampleRate,
        contextNow: contexts[0].currentTime,
        playedNow: playbacks.map((playback) => playback.playedSeconds()),
        renderQuantumFrames: contexts[0].renderQuantumSize,
        gainValue: outputGain?.gain.value ?? null,
        tapRms: tapRms(),
        tapIsOutputGain: player.tap === outputGain,
        hasBeenActive: navigator.userActivation.hasBeenActive,
      };
    } finally {
      switchRunning = false;
      clockReads.clear();
    }
  },
};
player = entry.createPlayer();

// A sound handed over while the page loads, before any click: the autoplay policy refuses it.
// The test learns of its end through `window.noteEnded`, because any call of the test into the page counts as a click.
const atLoad = new URLSearchParams(location.search).get("handOver");
if (atLoad !== null) handOver({ url: atLoad });
