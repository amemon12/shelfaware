import type { EvidenceEvent, Mode } from './types';

export const modeLabels: Record<Mode, string> = { fixture: 'Synthetic demo', live_phone: 'Phone camera', recorded_glasses: 'Recorded glasses POV', live_glasses: 'Glasses bridge' };
export const sourceLabels = { synthetic: 'Synthetic fixture', package_ocr: 'Package OCR · awaiting review', reviewed_label: 'User-reviewed label', database: 'Product database', calculation: 'Calculated from label' };
export function shouldSendCapture(mode: Mode, pendingCapture: boolean, forceCapture: boolean): boolean {
  return mode !== 'fixture' && (pendingCapture || forceCapture);
}
export function captureSource(hasActiveMedia: boolean, mediaReady: boolean, hasStill: boolean): 'video' | 'image' | null {
  // A loading camera/video must never silently reuse a previously captured product.
  if (hasActiveMedia) return mediaReady ? 'video' : null;
  return hasStill ? 'image' : null;
}
export function parsePortion(value: string): number | undefined {
  if (!value.trim()) return undefined;
  const amount = Number(value);
  if (!Number.isFinite(amount) || amount <= 0 || amount > 5000) throw new Error('Enter a portion between 0 and 5,000 grams.');
  return amount;
}
export function formatAmount(value: number | null, unit = 'g'): string {
  return value === null ? 'Not available' : `${new Intl.NumberFormat('en', { maximumFractionDigits: 2 }).format(value)} ${unit}`;
}
export function timeLabel(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? 'Time unavailable' : date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
}
export function visibleBoxes(events: EvidenceEvent[], mediaTime: number | null): EvidenceEvent[] {
  return events.filter(event => {
    const box = event.box;
    if (!box || ![box.x, box.y, box.w, box.h].every(Number.isFinite) || box.x < 0 || box.y < 0 || box.w <= 0 || box.h <= 0 || box.x + box.w > 1.001 || box.y + box.h > 1.001) return false;
    if (mediaTime !== null) return event.media_time_s != null && Math.abs(event.media_time_s - mediaTime) < 0.22;
    return event.media_time_s == null;
  });
}
