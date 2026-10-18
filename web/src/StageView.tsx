import { Icon } from './Icon';
import type { Session, TurnResponse } from './types';
import { formatAmount, modeLabels, sourceLabels } from './utils';

type Props = {
  session: Session;
  result: TurnResponse | null;
  image: string | null;
  capturedAt: string | null;
  connected: boolean;
  error: string;
  onExit: () => void;
};

export function StageView({ session, result, image, capturedAt, connected, error, onExit }: Props) {
  const product = session.products.find(item => item.id === session.current_product_id);
  const question = [...session.messages].reverse().find(message => message.role === 'user');
  const comparison = result?.comparison;
  const timestamp = capturedAt ? new Date(capturedAt).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', second: '2-digit' }) : null;

  return <div className="stage-shell">
    <header className="stage-header">
      <div className="brand"><span className="brand-icon"><Icon name="leaf" size={28}/></span><span>Shelf<span className="brand-aware">Aware</span><small>A LITTLE CLARITY IN EVERY AISLE</small></span></div>
      <div className="stage-header-status"><span className={`mode-badge ${session.mode === 'fixture' ? 'synthetic' : ''}`}>{modeLabels[session.mode]}</span><span className={`connection ${connected ? 'online' : ''}`} role="status"><span/>{connected ? 'Session connected' : 'Connection interrupted'}</span><button className="button secondary" onClick={onExit}>Exit stage view<Icon name="close" size={18}/></button></div>
    </header>
    <main className="stage-main">
      {error && <div className="alert error-alert" role="alert"><Icon name="info"/><span>{error}</span></div>}
      <div className="stage-context"><span><Icon name="user" size={22}/>{session.recipient_id === 'self' ? 'For me' : 'For my family'}</span><span>{session.portion_grams == null ? 'Portion not yet confirmed' : `${formatAmount(session.portion_grams)} intended portion`}</span><span className="stage-observer-label"><Icon name="eye" size={20}/>Read-only observer · browser audio off</span></div>
      <div className="stage-presentation">
        <section className="stage-capture" aria-label="Latest timestamped capture">
          <div className="stage-image">{image ? <img src={image} alt="Latest explicitly requested product capture"/> : <div className="stage-image-empty"><Icon name="camera" size={50}/><p>{session.mode === 'fixture' ? 'Synthetic practice session' : 'Awaiting a requested capture'}</p><span>{session.mode === 'fixture' ? 'No real product image in this mode.' : 'Captured images expire after two minutes.'}</span></div>}</div>
          <div className="stage-capture-caption"><span>Timestamped capture · not live video</span><time dateTime={capturedAt ?? undefined}>{timestamp ?? 'No capture available'}</time></div>
          <div className="stage-product"><span className="eyebrow">CURRENT PRODUCT</span><h2>{product?.name ?? 'Ready for the first product.'}</h2><span className={`mode-badge ${product?.provenance.kind === 'synthetic' ? 'synthetic' : ''}`}>{product ? sourceLabels[product.provenance.kind] : 'No product evidence yet'}</span></div>
        </section>
        <section className="stage-dialogue" aria-label="Completed answer transcript">
          <div className="eyebrow"><span className="small-line"/>THE CONVERSATION</div>
          {question && <div className="stage-question"><span>YOU ASKED</span><p>{question.text}</p></div>}
          <div className={`stage-answer ${(result?.reply.length ?? 0) > 300 ? 'long-answer' : ''}`} aria-live="polite" aria-atomic="true"><span className="stage-answer-label"><Icon name="leaf" size={22}/>SHELF AWARE</span><p>{result?.reply ?? 'Ready when you are. The next completed answer will appear here.'}</p></div>
          <p className="stage-caption-note">Full answers appear after processing. This transcript is not synchronized word by word with speech.</p>
        </section>
      </div>
      <section className="stage-facts" aria-label={comparison ? 'Current portion comparison' : 'Confirmed shopping list'}>
        {comparison ? <><div className="stage-facts-heading"><span className="eyebrow">SAME PORTION. A CLEARER PICTURE.</span><h2>{formatAmount(comparison.portion_grams)} of each product</h2></div><div className="stage-comparison">{comparison.items.map(item => <article key={item.product_id}><h3>{item.name}</h3><div><span><strong>{formatAmount(item.carbohydrate_g)}</strong>Total carbohydrate</span><span><strong>{formatAmount(item.fiber_g)}</strong>Fiber</span></div></article>)}</div></> : <><div className="stage-facts-heading"><span className="eyebrow">CHOSEN WITH CONFIDENCE</span><h2>Your confirmed list</h2></div>{session.cart.length ? <ul className="stage-list">{session.cart.map(item => <li key={item.id}><Icon name="check" size={22}/><span>{item.product_name}<small>{item.quantity} package{item.quantity === 1 ? '' : 's'} · {item.recipient_id === 'self' ? 'For me' : 'For my family'}</small></span></li>)}</ul> : <p className="stage-list-empty">Nothing added yet. Every list update needs confirmation.</p>}</>}
      </section>
    </main>
    <footer className="stage-footer"><span>Nutrition context · no glucose prediction</span><span>{session.mode === 'fixture' ? 'Fictional products and values' : session.mode === 'recorded_glasses' ? 'Recorded POV · frames only on request' : 'Evidence from this session'} · {session.id.slice(0, 8)}</span></footer>
  </div>;
}
