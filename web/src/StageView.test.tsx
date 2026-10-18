import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { StageView } from './StageView';
import type { Session, TurnResponse } from './types';

const session: Session = {
  id: 'a05b0d4d-4be8-4f0b-a20e-cbf53c9510f4', mode: 'live_glasses', recipient_id: 'household', portion_grams: 60, generation: 1,
  profile: { display_name: 'Alex', goals: [], avoid_ingredients: [], consented: true }, products: [], current_product_id: null,
  cart: [], pending_action: null, messages: [{ id: '1', role: 'user', text: 'What about this one?', created_at: '2026-10-18T06:00:00Z' }], created_at: '2026-10-18T06:00:00Z',
};

describe('silent presentation view', () => {
  it('does not expose the private goal source or advice date on stage', () => {
    const privateSession: Session = { ...session, profile: { ...session.profile, goal_source: { kind: 'reviewed_guidance', source_date: '2020-04-19', reviewed_at: '2026-10-18T04:02:01Z', confirmed: true } } };
    const html = renderToStaticMarkup(<StageView session={privateSession} result={null} image={null} capturedAt={null} connected error="" onExit={() => {}}/>);
    expect(html).not.toContain('reviewed_guidance');
    expect(html).not.toContain('2020-04-19');
    expect(html).not.toContain('2026-10-18T04:02:01Z');
  });
  it('shows the whole completed answer with recipient and portion, without action controls', () => {
    const result: TurnResponse = { session, reply: 'At 60 grams, the second product has 6 grams less total carbohydrate.', status: 'ready', comparison: null, evidence: [], events: [], pending_action: null };
    const html = renderToStaticMarkup(<StageView session={session} result={result} image={null} capturedAt={null} connected error="" onExit={() => {}}/>);
    expect(html).toContain(result.reply);
    expect(html).toContain('For my family');
    expect(html).toContain('60 g intended portion');
    expect(html).toContain('browser audio off');
    expect((html.match(/<button/g) ?? [])).toHaveLength(1); // Exit is the only available control.
    expect(html).not.toContain('<video');
  });

  it('labels a real still as a timestamped capture and keeps missing comparison values unknown', () => {
    const result: TurnResponse = { session, reply: 'Review the unavailable fiber field.', status: 'ready', evidence: [], events: [], pending_action: null, comparison: { portion_grams: 60, recipient_id: 'household', summary: '', items: [{ product_id: 'one', name: 'Reviewed product', carbohydrate_g: 1.25, fiber_g: null, sugars_g: null, protein_g: null, sodium_mg: null }] } };
    const html = renderToStaticMarkup(<StageView session={session} result={result} image="data:image/png;base64,aGVsbG8=" capturedAt="2026-10-18T06:00:00Z" connected error="" onExit={() => {}}/>);
    expect(html).toContain('Timestamped capture · not live video');
    expect(html).toContain('2026-10-18T06:00:00Z');
    expect(html).toContain('1.25 g');
    expect(html).toContain('Not available');
  });
});
