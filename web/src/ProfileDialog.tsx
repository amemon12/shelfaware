import { useState } from 'react';
import { Icon } from './Icon';
import { goalOptions, goalPresets, matchingPreset, profilePayload } from './profile';
import type { Mode, Profile } from './types';
import { modeLabels } from './utils';

type Props = { initial: Profile; mode: Mode; busy: boolean; hasSession: boolean; error: string; onClose: () => void; onStart: (profile: Profile, mode: Mode) => Promise<void> };

export function ProfileDialog({ initial, mode, busy, hasSession, error, onClose, onStart }: Props) {
  const [name, setName] = useState(initial.display_name);
  const [goals, setGoals] = useState(initial.goals.filter(goal => goalOptions.includes(goal as typeof goalOptions[number])));
  const [avoid, setAvoid] = useState(initial.avoid_ingredients.join(', '));
  const [consent, setConsent] = useState(false);
  const [useReviewedGuidance, setUseReviewedGuidance] = useState(initial.goal_source?.kind === 'reviewed_guidance');
  const [sourceDate, setSourceDate] = useState(initial.goal_source?.kind === 'reviewed_guidance' ? initial.goal_source.source_date ?? '' : '');
  const [guidanceConfirmed, setGuidanceConfirmed] = useState(false);
  const [localError, setLocalError] = useState('');
  const selectedPreset = matchingPreset(goals);
  const today = new Date().toISOString().slice(0, 10);

  function changeGoals(next: string[]) {
    setGoals([...next]); setGuidanceConfirmed(false); setConsent(false); setLocalError('');
  }
  async function submit(event: React.FormEvent) {
    event.preventDefault(); setLocalError('');
    try { await onStart(profilePayload({ name, goals, avoid, consent, mode, useReviewedGuidance, guidanceConfirmed, sourceDate }), mode); }
    catch (cause) { setLocalError(cause instanceof Error ? cause.message : 'Could not start the session.'); }
  }

  return <div className="modal-backdrop"><section className="modal profile-modal" role="dialog" aria-modal="true" aria-labelledby="profile-heading" aria-describedby="profile-intro">
    <button className="icon-button modal-close" aria-label="Close profile" onClick={onClose} disabled={busy}><Icon name="close"/></button>
    <div className="profile-heading-line"><span className="profile-mini-leaf"><Icon name="leaf" size={23}/></span><span className="eyebrow">MAKE IT YOURS, ON YOUR TERMS</span></div>
    <h2 id="profile-heading">Your goals. Your choice.</h2>
    <p id="profile-intro">Choose what matters for this trip. Every preset is editable; no diagnosis is needed.</p>
    {mode === 'fixture' && <div className="profile-demo-note"><span className="mode-badge synthetic">Synthetic practice</span><span>Use a fictional profile. Products and values are made up.</span></div>}
    <form onSubmit={submit}>
      <fieldset className="preset-fieldset" disabled={busy}><legend>Start with a shopping style</legend><div className="profile-presets">{goalPresets.map(preset => <label key={preset.id} className={`profile-preset ${selectedPreset === preset.id ? 'selected' : ''}`}><span className="preset-title"><input type="radio" name="profile-preset" value={preset.id} checked={selectedPreset === preset.id} onChange={() => changeGoals(preset.goals)}/><strong>{preset.title}</strong></span><span className="preset-description">{preset.description}</span></label>)}</div>{selectedPreset === null && <p className="preset-custom">You’ve customized the comparison goals below.</p>}</fieldset>
      <fieldset disabled={busy}><legend>Fine-tune your comparison goals</legend><div className="goal-options">{goalOptions.map(goal => <label key={goal} className={goals.includes(goal) ? 'selected' : ''}><input type="checkbox" checked={goals.includes(goal)} onChange={() => changeGoals(goals.includes(goal) ? goals.filter(value => value !== goal) : [...goals, goal])}/><span>{goal}</span></label>)}</div><p className="profile-small-note">With no goals selected, we explain the label and portion without a personal nutrient preference.</p></fieldset>
      <div className="profile-personal-fields"><label>{mode === 'fixture' ? 'Demo name' : 'Preferred name'}<input value={name} onChange={event => { setName(event.target.value); setConsent(false); }} maxLength={50} disabled={busy} autoComplete="off"/></label><label>Ingredients to avoid <span className="optional">optional</span><input value={avoid} onChange={event => { setAvoid(event.target.value); setConsent(false); }} placeholder="e.g. peanuts, sesame" disabled={busy}/></label></div>
      <p className="profile-boundary">Ingredient matching flags declared ingredients; it cannot guarantee allergy safety or cross-contact information.</p>
      <details className="reviewed-advice" open={useReviewedGuidance || undefined}><summary>Goals from reviewed advice <span>Optional</span></summary><div className="reviewed-advice-content"><p>If you already reviewed advice with a clinician, select the relevant goals above yourself. Shelf Aware does not interpret that advice or set dietary targets.</p><label className="check-label"><input type="checkbox" checked={useReviewedGuidance} disabled={busy} onChange={event => { setUseReviewedGuidance(event.target.checked); setGuidanceConfirmed(false); setConsent(false); }}/><span>These goals come from advice I have already reviewed.</span></label>{useReviewedGuidance && <><label>Advice date <span className="optional">optional</span><input type="date" value={sourceDate} max={today} disabled={busy} onChange={event => { setSourceDate(event.target.value); setGuidanceConfirmed(false); setConsent(false); }}/></label><label className="check-label"><input type="checkbox" checked={guidanceConfirmed} disabled={busy} onChange={event => { setGuidanceConfirmed(event.target.checked); setConsent(false); }}/><span>I confirm that the selected comparison goals match the advice I reviewed. I will correct them if needed.</span></label><p className="profile-small-note">We save only this source type, the optional date, and confirmation time. This source is not spoken aloud or shown in stage view.</p></>}<p className="report-import-note">Report import is not included in this prototype. Do not paste lab values or medical records here.</p></div></details>
      <label className="check-label profile-consent"><input type="checkbox" checked={consent} disabled={busy} onChange={event => setConsent(event.target.checked)}/><span>I reviewed these goals and ingredients, and agree to use this profile for the session. {mode !== 'fixture' && 'Submitted label images may be processed by the configured AI provider.'}</span></label>
      {hasSession && <p className="notice">Starting creates a fresh {modeLabels[mode].toLowerCase()} session and clears your current local trip.</p>}
      {(localError || error) && <p className="error-message" role="alert">{localError || error}</p>}
      <button className="button primary start-button" disabled={!consent || busy || (useReviewedGuidance && !guidanceConfirmed)}>{busy ? 'Starting…' : `Start ${mode === 'fixture' ? 'the synthetic demo' : 'this session'}`}<Icon name="arrow"/></button>
    </form>
  </section></div>;
}
