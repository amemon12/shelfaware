import { useState } from 'react';
import type { Nutrition, Product } from './types';
import { Icon } from './Icon';

const fields: { key: keyof Nutrition; label: string; unit: string }[] = [
  { key: 'basis_grams', label: 'Label serving size', unit: 'g' },
  { key: 'carbohydrate_g', label: 'Total carbohydrate', unit: 'g' },
  { key: 'fiber_g', label: 'Fiber', unit: 'g' },
  { key: 'sugars_g', label: 'Total sugars', unit: 'g' },
  { key: 'protein_g', label: 'Protein', unit: 'g' },
  { key: 'sodium_mg', label: 'Sodium', unit: 'mg' },
];
export function ReviewDialog({ product, image, busy, onClose, onConfirm }: { product: Product; image: string | null; busy: boolean; onClose: () => void; onConfirm: (corrections: { name: string; nutrition: Nutrition; ingredients: string[] }) => Promise<void> }) {
  const [name, setName] = useState(product.name);
  const [values, setValues] = useState(Object.fromEntries(fields.map(field => [field.key, product.nutrition[field.key]?.toString() ?? ''])) as Record<keyof Nutrition, string>);
  const [ingredients, setIngredients] = useState(product.ingredients.join(', '));
  const [error, setError] = useState('');
  const [checked, setChecked] = useState(false);
  async function submit(event: React.FormEvent) {
    event.preventDefault(); setError('');
    try {
      const nutrition = Object.fromEntries(fields.map(field => {
        const raw = values[field.key].trim(); const value = raw === '' ? null : Number(raw);
        if (value !== null && (!Number.isFinite(value) || value < 0)) throw new Error('Nutrition values must be zero or positive numbers. Leave an unreadable value blank.');
        if (field.key === 'basis_grams' && (value === null || value <= 0)) throw new Error('A readable serving size in grams is required. Retake the label if it is missing.');
        return [field.key, value];
      })) as Nutrition;
      if (!name.trim()) throw new Error('Enter the product name exactly as shown.');
      if (!checked) throw new Error('Please confirm you checked these values against the package.');
      await onConfirm({ name: name.trim(), nutrition, ingredients: ingredients.split(',').map(value => value.trim()).filter(Boolean) });
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not save this review.'); }
  }
  return <div className="modal-backdrop"><section className="modal review-modal" role="dialog" aria-modal="true" aria-labelledby="review-heading">
    <button className="icon-button modal-close" aria-label="Close label review" onClick={onClose} disabled={busy}><Icon name="close"/></button>
    <div className="eyebrow">YOUR EYES GET THE FINAL SAY</div><h2 id="review-heading">Check the label.</h2><p>OCR can misread a number or a serving column. Correct the values below before we use them.</p>
    <form onSubmit={submit}><div className="review-layout">{image && <img className="review-image" src={image} alt="Original label photo to check against extracted nutrition"/>}<div className="review-inputs"><label>Product name<input value={name} onChange={event => setName(event.target.value)} required/></label><div className="nutrition-fields">{fields.map(field => <label key={field.key}>{field.label}<div className="input-unit"><input type="number" min={field.key === 'basis_grams' ? '0.01' : '0'} step="any" value={values[field.key]} onChange={event => setValues({ ...values, [field.key]: event.target.value })} placeholder="Unreadable"/><span>{field.unit}</span></div></label>)}</div><label>Ingredients, as printed<textarea value={ingredients} onChange={event => setIngredients(event.target.value)} rows={2}/></label></div></div>
    {product.uncertain_fields.length > 0 && <p className="notice">Needs attention: {product.uncertain_fields.join(', ')}</p>}
    <label className="check-label"><input type="checkbox" checked={checked} onChange={event => setChecked(event.target.checked)}/><span>I checked this exact product and serving column against the package. Blank fields are unknown.</span></label>
    {error && <p className="error-message" role="alert">{error}</p>}<div className="modal-actions"><button type="button" className="button secondary" onClick={onClose} disabled={busy}>Retake instead</button><button className="button primary" disabled={!checked || busy}>{busy ? 'Saving review…' : 'Confirm these label values'}<Icon name="check"/></button></div></form>
  </section></div>;
}
