import type { Mode, Profile } from './types';

export const goalOptions = ['Compare total carbohydrate at my portion', 'Prefer more fiber', 'Prefer less sodium', 'Compare protein'] as const;
export const goalPresets = [
  { id: 'general', title: 'General shopper', description: 'Start with the label and your portion.', goals: [] as string[] },
  { id: 'portion', title: 'Portion-conscious', description: 'Compare carbohydrate and fiber at your portion.', goals: [goalOptions[0], goalOptions[1]] as string[] },
  { id: 'fiber', title: 'Fiber-focused', description: 'Give fiber a closer look.', goals: [goalOptions[1]] as string[] },
] as const;

export function matchingPreset(goals: string[]): string | null {
  return goalPresets.find(preset => preset.goals.length === goals.length && preset.goals.every(goal => goals.includes(goal)))?.id ?? null;
}

export type ProfileDraft = {
  name: string; goals: string[]; avoid: string; consent: boolean; mode: Mode;
  useReviewedGuidance: boolean; guidanceConfirmed: boolean; sourceDate: string;
};

export function profilePayload(draft: ProfileDraft, today = new Date().toISOString().slice(0, 10)): Profile {
  if (!draft.consent) throw new Error('Review the profile and confirm consent before starting.');
  if (draft.goals.some(goal => !goalOptions.includes(goal as typeof goalOptions[number]))) throw new Error('Choose from the listed comparison goals.');
  if (draft.useReviewedGuidance && !draft.guidanceConfirmed) throw new Error('Confirm that these selected goals match the advice you reviewed.');
  const sourceDate = draft.useReviewedGuidance && draft.sourceDate ? draft.sourceDate : null;
  if (sourceDate) {
    const parsed = new Date(`${sourceDate}T00:00:00Z`);
    if (!/^\d{4}-\d{2}-\d{2}$/.test(sourceDate) || Number.isNaN(parsed.getTime()) || parsed.toISOString().slice(0, 10) !== sourceDate || sourceDate > today) throw new Error('Use a valid advice date that is today or earlier, or leave it blank.');
  }
  return {
    display_name: draft.name.trim() || 'Shopper', goals: [...new Set(draft.goals)],
    avoid_ingredients: [...new Set(draft.avoid.split(',').map(value => value.trim()).filter(Boolean))], consented: true,
    goal_source: { kind: draft.useReviewedGuidance ? 'reviewed_guidance' : draft.mode === 'fixture' ? 'synthetic' : 'self_selected', source_date: sourceDate, confirmed: true },
  };
}
