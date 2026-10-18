import { describe, expect, it } from 'vitest';
import { captureSource, formatAmount, parsePortion, shouldSendCapture, visibleBoxes } from './utils';
import type { EvidenceEvent } from './types';

describe('explicit capture selection', () => {
  it('keeps follow-up questions on the current product after a capture is accepted', () => {
    expect(shouldSendCapture('live_phone', true, false)).toBe(true);
    expect(shouldSendCapture('live_phone', false, false)).toBe(false);
    expect(shouldSendCapture('recorded_glasses', false, false)).toBe(false);
  });
  it('allows a requested new frame while keeping fixtures independent of camera state', () => {
    expect(shouldSendCapture('live_phone', false, true)).toBe(true);
    expect(shouldSendCapture('recorded_glasses', false, true)).toBe(true);
    expect(shouldSendCapture('fixture', true, true)).toBe(false);
  });
  it('does not replace a loading new video frame with an old still', () => {
    expect(captureSource(true, false, true)).toBeNull();
    expect(captureSource(true, true, true)).toBe('video');
    expect(captureSource(false, false, true)).toBe('image');
    expect(captureSource(false, false, false)).toBeNull();
  });
});

describe('nutrition input boundaries', () => {
  it('keeps an unspecified portion distinct from a zero portion', () => {
    expect(parsePortion('')).toBeUndefined();
    expect(() => parsePortion('0')).toThrow();
    expect(() => parsePortion('Infinity')).toThrow();
    expect(parsePortion('60.5')).toBe(60.5);
  });
  it('never represents missing nutrition as zero', () => {
    expect(formatAmount(null)).toBe('Not available');
    expect(formatAmount(0)).toBe('0 g');
    expect(formatAmount(1.25)).toBe('1.25 g');
  });
});
describe('evidence-only overlays', () => {
  const event: EvidenceEvent = { id: '1', type: 'detected', at: '', stage: 'perception', message: '', media_time_s: 5, box: { x: 0.2, y: 0.2, w: 0.3, h: 0.4 } };
  it('does not extend a detection through unseen video time', () => {
    expect(visibleBoxes([event], 5)).toHaveLength(1);
    expect(visibleBoxes([event], 6)).toHaveLength(0);
    expect(visibleBoxes([event], null)).toHaveLength(0);
  });
  it('discards invalid geometry rather than drawing invented repairs', () => {
    expect(visibleBoxes([{ ...event, box: { x: 0.9, y: 0, w: 0.5, h: 0.5 } }], 5)).toHaveLength(0);
    expect(visibleBoxes([{ ...event, box: undefined }], 5)).toHaveLength(0);
  });
  it('accepts nullable optional fields from the API without inventing a video timestamp', () => {
    expect(visibleBoxes([{ ...event, media_time_s: null }], null)).toHaveLength(1);
    expect(visibleBoxes([{ ...event, media_time_s: null }], 0)).toHaveLength(0);
    expect(visibleBoxes([{ ...event, box: null }], null)).toHaveLength(0);
  });
});
