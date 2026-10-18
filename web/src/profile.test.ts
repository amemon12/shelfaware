import { describe, expect, it } from 'vitest';
import { goalOptions, goalPresets, matchingPreset, profilePayload } from './profile';
import type { ProfileDraft } from './profile';

const draft: ProfileDraft = { name: 'Alex', goals: [], avoid: '', consent: true, mode: 'live_phone', useReviewedGuidance: false, guidanceConfirmed: false, sourceDate: '' };

describe('explicit profile goals and provenance', () => {
  it('keeps General shopper free of inherited carbohydrate or fiber goals', () => {
    const payload = profilePayload(draft);
    expect(payload.goals).toEqual([]);
    expect(matchingPreset(payload.goals)).toBe('general');
    expect(payload.goal_source).toEqual({ kind: 'self_selected', source_date: null, confirmed: true });
  });
  it('requires both profile consent and separate reviewed-advice confirmation', () => {
    expect(() => profilePayload({ ...draft, consent: false })).toThrow('consent');
    expect(() => profilePayload({ ...draft, useReviewedGuidance: true })).toThrow('Confirm');
    expect(profilePayload({ ...draft, useReviewedGuidance: true, guidanceConfirmed: true }).goal_source?.kind).toBe('reviewed_guidance');
  });
  it('does not leak a prior advice date into self-selected goals', () => {
    const payload = profilePayload({ ...draft, sourceDate: '2026-01-02', guidanceConfirmed: true });
    expect(payload.goal_source?.source_date).toBeNull();
  });
  it('rejects invalid or future advice dates instead of fabricating provenance', () => {
    const reviewed = { ...draft, useReviewedGuidance: true, guidanceConfirmed: true };
    expect(() => profilePayload({ ...reviewed, sourceDate: '2026-02-31' }, '2026-10-18')).toThrow('valid advice date');
    expect(() => profilePayload({ ...reviewed, sourceDate: '2026-10-19' }, '2026-10-18')).toThrow('valid advice date');
  });
  it('sends only explicit goal metadata and leaves confirmation timestamps to the server', () => {
    const input = { ...draft, useReviewedGuidance: true, guidanceConfirmed: true, sourceDate: '2026-01-02', lab_values: { triglycerides: 500 }, diagnosis: 'unverified', reviewed_at: 'made-up' };
    const payload = profilePayload(input);
    expect(Object.keys(payload).sort()).toEqual(['avoid_ingredients', 'consented', 'display_name', 'goal_source', 'goals']);
    expect(payload.goal_source).toEqual({ kind: 'reviewed_guidance', source_date: '2026-01-02', confirmed: true });
    expect(JSON.stringify(payload)).not.toContain('triglycerides');
  });
  it('isolates preset arrays across sessions and rejects unsupported clinical instructions', () => {
    const payload = profilePayload({ ...draft, goals: goalPresets[1].goals });
    payload.goals.pop();
    expect(goalPresets[1].goals).toEqual([goalOptions[0], goalOptions[1]]);
    expect(() => profilePayload({ ...draft, goals: ['Infer a diet from triglycerides'] })).toThrow('listed comparison goals');
  });
  it('labels fixture profiles as synthetic unless reviewed guidance was explicitly confirmed', () => {
    expect(profilePayload({ ...draft, mode: 'fixture' }).goal_source?.kind).toBe('synthetic');
  });
});
