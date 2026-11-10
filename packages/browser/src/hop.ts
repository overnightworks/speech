// Generated from speech_client.wire and speech_client.relay by scripts/generate_browser_wire.py. Do not edit.
// Rewrite it with: uv run --locked python scripts/generate_browser_wire.py

export const SAMPLE_RATE = 24000;
export const SAMPLE_BYTES = 2;
export const CHANNELS = 1;
export const HOP_FRAME_SAMPLES = 2400;
export const HOP_FRAME_BYTES = 4800;

export const HopClose = {
  NORMAL: 1000,
  TEXT_FRAME: 4400,
  FRAME_TOO_LARGE: 4413,
  OVER_CAP: 4507,
  BUSY: 4429,
  PAST_DEADLINE: 4504,
  SERVICE_FAILED: 4502,
} as const;

export type HopClose = (typeof HopClose)[keyof typeof HopClose];
