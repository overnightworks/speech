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
  createBufferSource: BaseAudioContext.prototype.createBufferSource,
  decodeAudioData: BaseAudioContext.prototype.decodeAudioData,
  resume: AudioContext.prototype.resume,
  start: AudioBufferSourceNode.prototype.start,
};
const SAMPLE_MS = 20;
const SILENCE_POLL_MS = 10;
const SILENCE_GIVE_UP_MS = 5000;
const JUNK_BYTES = 512;
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
  sourcesEnded: 0,
  instrumentStops: 0,
  decodesCalled: 0,
  decodesSettled: 0,
  resumesCalled: 0,
  lastActionAt: null,
  endedAtAction: null,
  peakRms: 0,
  heardOnsets: [],
  heardPeaks: [],
  contextAtAction: null,
  actions: [],
};
const contexts = [];
const clockReads = new Map();
const frozenClocks = new Map();
const meters = new WeakMap();
const nodeIds = new WeakMap();
const trackedSources = new WeakSet();
const held = { decode: [], resume: [] };
const heldNow = [];
const byteStore = new Map();
const fedBodies = new Map();
const feeders = new Map();
let nextNodeId = 0;
let stopOnSourceEnd = false;
let rejectNextResume = false;
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

function meterOf(context) {
  if (!meters.has(context)) meters.set(context, new Page.AnalyserNode(context));
  return meters.get(context);
}

const earsReady = [];

// The page's own ear: a worklet in the graph (`ear_processor.js`) that notes the context time of the first sample
// of each stretch of sound that reaches the output and the peak of each render quantum.
function listenTo(output) {
  const context = output.context;
  earsReady.push(
    context.audioWorklet.addModule("/ear_processor.js").then(() => {
      const ear = new AudioWorkletNode(context, "ear");
      ear.port.onmessage = ({ data }) => {
        if (data.onset !== undefined) seen.heardOnsets.push(data.onset);
        else seen.heardPeaks.push([data.quantum, data.peak]);
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
      stopOnSourceEnd = false;
      seen.instrumentStops += 1;
      player.stop();
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

// The context clock advances on the audio thread, and under load two reads in one task straddle any number of its
// steps. The instrument holds the first read of a context until the task is over, so that the player and the page
// read the same instant when they act in one task.
Object.defineProperty(BaseAudioContext.prototype, "currentTime", {
  configurable: true,
  get() {
    if (frozenClocks.has(this)) return frozenClocks.get(this);
    if (!clockReads.has(this)) {
      clockReads.set(this, Page.clock.get.call(this));
      setTimeout(() => clockReads.delete(this));
    }
    return clockReads.get(this);
  },
});
window.AudioContext = class AudioContext extends Page.AudioContext {
  constructor(...parameters) {
    super(...parameters);
    contexts.push(this);
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
Page.AudioBufferSourceNode.prototype.start = function (...parameters) {
  seen.sourcesStarted += 1;
  return Page.start.apply(this, parameters);
};
AudioNode.prototype.connect = function (target, ...parameters) {
  const toOutput = target instanceof Page.AudioDestinationNode;
  seen.edges.push({ from: idOf(this), fromKind: kindOf(this), to: target instanceof AudioNode ? idOf(target) : null, toKind: kindOf(target) });
  if (toOutput) {
    outputGain = this;
    Page.connect.call(this, meterOf(this.context));
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

function rms() {
  const meter = meters.get(contexts[0]);
  if (meter === undefined) return 0;
  const samples = new Float32Array(meter.fftSize);
  meter.getFloatTimeDomainData(samples);
  return Math.sqrt(samples.reduce((sum, value) => sum + value * value, 0) / samples.length);
}

function tapRms() {
  if (tapMeter === null) return null;
  const samples = new Float32Array(tapMeter.fftSize);
  tapMeter.getFloatTimeDomainData(samples);
  return Math.sqrt(samples.reduce((sum, value) => sum + value * value, 0) / samples.length);
}

setInterval(() => {
  const level = rms();
  const tapLevel = tapRms();
  seen.peakRms = Math.max(seen.peakRms, level);
  for (const sound of seen.sounds) {
    if (sound.startedValue === "playing" && sound.endedValue === null) {
      sound.peakRms = Math.max(sound.peakRms, level);
      if (tapLevel !== null) sound.peakTapRms = Math.max(sound.peakTapRms, tapLevel);
    }
  }
}, SAMPLE_MS);

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
  const sound = { id: seen.sounds.length, startedValue: null, startedCount: 0, endedValue: null, endedCount: 0, endedAt: null, peakRms: 0, peakTapRms: 0, playedSeconds: null, start: null, contextAtEnd: null };
  seen.sounds.push(sound);
  const playback = player.play(argumentOf(spec));
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
    sound.contextAtEnd = contexts[0].currentTime;
    seen.log.push({ event: "ended", id: sound.id, value: end.kind, at: sound.endedAt });
    if (spec.bytes !== undefined) seen.bytes[spec.bytes].after = byteStore.get(spec.bytes).byteLength;
    window.noteEnded?.(sound.id);
  });
  return sound.id;
}

// A switch of the app: it records when it happened and what had ended by the end of its own task. The held clock read
// is dropped first and the switch's time is the read the player made: a read from before, held across a slow page,
// would start the player's volume ramp in the past, and the ear would hear a ramp shorter than the player's edge.
async function act(name, action) {
  seen.lastActionAt = performance.now();
  const gain = outputGain?.gain.value ?? null;
  clockReads.delete(contexts[0]);
  action();
  seen.contextAtAction = contexts[0].currentTime;
  seen.actions.push({ name, at: seen.contextAtAction, gain });
  await Promise.resolve();
  seen.endedAtAction = seen.sounds.map((sound) => sound.endedValue);
}

function nextTurn() {
  return new Promise((resolve) => {
    const channel = new MessageChannel();
    channel.port1.onmessage = resolve;
    channel.port2.postMessage(null);
  });
}

// A switch of the app thrown once the volume has left the value it had when the call began, so that it reverses a ramp
// under way. The throw records the volume it found: a ramp that never began, or is over, shows there.
async function actWhileFading(name, action) {
  const from = outputGain.gain.value;
  const giveUp = performance.now() + SILENCE_GIVE_UP_MS;
  while (outputGain.gain.value === from && performance.now() < giveUp) await nextTurn();
  return act(name, action);
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
  // The switch `name` and the rest of the fed body `key`, in one task.
  feedWhile: (key, name) => act(`${name}+feed`, () => { player[name](); feeders.get(key)(); }),
  // The switch `first`, then `second` as soon as the volume it set moves: the second reverses the ramp of the first.
  reverse: async (first, second) => {
    await act(first, () => player[first]());
    return actWhileFading(second, () => player[second]());
  },
  // The context clock stands still, as the player and the page read it, until `thawClock`: what the player does before
  // the clock moves again happens inside the ramp it started, whatever the page's timing.
  freezeClock: () => contexts.forEach((context) => frozenClocks.set(context, context.currentTime)),
  thawClock: () => frozenClocks.clear(),
  mute: (muted) => act("mute", () => player.setMuted(muted)),
  close: () => act("close", () => player.close()),
  listening: () => Promise.all(earsReady),
  // The app's reader: one analyser of the player's context, connected to `player.tap` from the outside.
  readTap: () => {
    tapMeter = new Page.AnalyserNode(player.tap.context);
    player.tap.connect(tapMeter);
  },
  hold: (kind) => held[kind].push(Promise.withResolvers()),
  release: () => heldNow.splice(0).forEach((gate) => gate.resolve()),
  rejectNextResume: () => (rejectNextResume = true),
  stopOnNextSourceEnd: () => (stopOnSourceEnd = true),
  snapshot: () => ({
    ...seen,
    contextStates: contexts.map((context) => context.state),
    sampleRate: contexts[0].sampleRate,
    clockStep: Math.max(128 / contexts[0].sampleRate, contexts[0].baseLatency),
    gainValue: outputGain?.gain.value ?? null,
    tapRms: tapRms(),
    tapIsOutputGain: player.tap === outputGain,
    hasBeenActive: navigator.userActivation.hasBeenActive,
  }),
  async silentAfterAction(threshold) {
    const start = performance.now();
    while (rms() > threshold && performance.now() - start < SILENCE_GIVE_UP_MS) {
      await new Promise((resolve) => setTimeout(resolve, SILENCE_POLL_MS));
    }
    return performance.now() - seen.lastActionAt;
  },
};
player = entry.createPlayer();

// A sound handed over while the page loads, before any click: the autoplay policy refuses it.
// The test learns of its end through `window.noteEnded`, because any call of the test into the page counts as a click.
const atLoad = new URLSearchParams(location.search).get("handOver");
if (atLoad !== null) handOver({ url: atLoad });
