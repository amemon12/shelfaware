import { useCallback, useEffect, useRef, useState } from 'react';
import { api, ApiError } from './api';
import { Icon } from './Icon';
import { MediaStage } from './MediaStage';
import { ReviewDialog } from './ReviewDialog';
import { ProfileDialog } from './ProfileDialog';
import { StageView } from './StageView';
import { captureSource, formatAmount, modeLabels, parsePortion, shouldSendCapture, sourceLabels, timeLabel } from './utils';
import type { Comparison, Evidence, EvidenceEvent, Health, Mode, Nutrition, Product, Profile, Recipient, Session, TurnRequest, TurnResponse, VoiceRecognition } from './types';

const defaultProfile: Profile = { display_name: 'Alex', goals: [], avoid_ingredients: [], consented: false };

const modeOptions: Mode[] = ['fixture', 'live_phone', 'recorded_glasses', 'live_glasses'];
const activeSessionKey = 'shelf-aware:active-session-id';
function storedSessionId() { try { return sessionStorage.getItem(activeSessionKey); } catch { return null; } }
function rememberSession(id: string | null) { try { if (id) sessionStorage.setItem(activeSessionKey, id); else sessionStorage.removeItem(activeSessionKey); } catch { /* Storage can be disabled; the current session still works. */ } }

function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [fixtures, setFixtures] = useState<Product[]>([]);
  const [session, setSession] = useState<Session | null>(null);
  const [result, setResult] = useState<TurnResponse | null>(null);
  const [events, setEvents] = useState<EvidenceEvent[]>([]);
  const [selectedFixture, setSelectedFixture] = useState('oat-a');
  const [mode, setMode] = useState<Mode>('fixture');
  const [view, setView] = useState<'shop' | 'evidence'>('shop');
  const [recipient, setRecipient] = useState<Recipient>('self');
  const [portion, setPortion] = useState('');
  const [question, setQuestion] = useState('');
  const [error, setError] = useState('');
  const [hint, setHint] = useState('');
  const [phase, setPhase] = useState<'idle' | 'listening' | 'thinking' | 'speaking'>('idle');
  const [busy, setBusy] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const [profile, setProfile] = useState<Profile>(defaultProfile);
  const [profileMode, setProfileMode] = useState<Mode>('fixture');
  const [reviewOpen, setReviewOpen] = useState(false);
  const [summaryOpen, setSummaryOpen] = useState(false);
  const [attachOpen, setAttachOpen] = useState(false);
  const [attachId, setAttachId] = useState('');
  const [observer, setObserver] = useState(false);
  const [stageView, setStageView] = useState(() => new URLSearchParams(window.location.search).get('stage') === '1');
  const [image, setImage] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [captureTime, setCaptureTime] = useState<string | null>(null);
  const [pendingCapture, setPendingCapture] = useState(false);
  const [imageMediaTime, setImageMediaTime] = useState<number | null>(null);
  const [language, setLanguage] = useState('en-US');
  const [bridgeCopied, setBridgeCopied] = useState(false);
  const controller = useRef<AbortController | null>(null);
  const turnSequence = useRef(0);
  const speechSequence = useRef(0);
  const mediaSequence = useRef(0);
  const recognition = useRef<VoiceRecognition | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const mediaTime = useRef(0);
  const imageInput = useRef<HTMLInputElement>(null);
  const videoInput = useRef<HTMLInputElement>(null);
  const latestObserverReply = useRef('');
  const busyRef = useRef(false);

  const activeProduct = session?.products.find(product => product.id === session.current_product_id) ?? null;
  const fixture = fixtures.find(product => product.id === selectedFixture);
  const pending = session?.pending_action ?? null;
  const needsReview = activeProduct?.provenance.kind === 'package_ocr' && !activeProduct.provenance.verified;
  const speechAvailable = typeof window !== 'undefined' && !!(window.SpeechRecognition || window.webkitSpeechRecognition);
  const onTimeChange = useCallback((time: number) => { mediaTime.current = time; }, []);

  const connect = useCallback(async () => {
    try {
      const [status, products] = await Promise.all([api<Health>('/health'), api<{ products: Product[] }>('/fixtures')]);
      setHealth(status); setFixtures(products.products); setError('');
    } catch (cause) { setHealth(null); setError(cause instanceof Error ? cause.message : 'Could not connect to the server.'); }
  }, []);
  useEffect(() => { void connect(); }, [connect]);
  useEffect(() => () => { stream?.getTracks().forEach(track => track.stop()); }, [stream]);
  useEffect(() => () => { if (videoUrl) URL.revokeObjectURL(videoUrl); }, [videoUrl]);
  useEffect(() => () => { mediaSequence.current++; speechSequence.current++; controller.current?.abort(); const active = recognition.current; recognition.current = null; active?.abort(); window.speechSynthesis?.cancel(); }, []);
  useEffect(() => {
    if (!observer || !captureTime) return;
    const age = Date.now() - new Date(captureTime).getTime();
    const timeout = window.setTimeout(() => { setImage(null); setCaptureTime(null); setImageMediaTime(null); }, Math.max(0, 120000 - (Number.isFinite(age) ? age : 0)));
    return () => window.clearTimeout(timeout);
  }, [observer, captureTime]);
  useEffect(() => {
    if (!profileOpen && !reviewOpen && !attachOpen && !summaryOpen) return;
    const previous = document.activeElement as HTMLElement | null;
    const dialog = document.querySelector<HTMLElement>('[role="dialog"]');
    if (!dialog) return;
    const selector = 'button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex="0"]';
    if (!dialog.contains(document.activeElement)) (dialog.querySelector<HTMLElement>('input:not(:disabled)') ?? dialog.querySelector<HTMLElement>(selector))?.focus();
    const keydown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !busy) { setProfileOpen(false); setReviewOpen(false); setAttachOpen(false); setSummaryOpen(false); }
      if (event.key !== 'Tab') return;
      const elements = Array.from(dialog.querySelectorAll<HTMLElement>(selector));
      const first = elements[0]; const last = elements.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    };
    document.addEventListener('keydown', keydown);
    return () => { document.removeEventListener('keydown', keydown); previous?.focus(); };
  }, [profileOpen, reviewOpen, attachOpen, summaryOpen, busy]);

  function ingest(response: TurnResponse, append = true) {
    setSession(response.session); setResult(response); setRecipient(response.session.recipient_id);
    setPortion(response.session.portion_grams?.toString() ?? ''); setMode(response.session.mode);
    if (response.session.mode === 'fixture' && response.session.current_product_id && response.status !== 'unknown') setSelectedFixture(response.session.current_product_id);
    setEvents(old => {
      const all = append ? [...old, ...response.events] : response.events;
      return Array.from(new Map(all.map(event => [event.id, event])).values()).slice(-80);
    });
  }
  const attach = useCallback(async (id: string) => {
    if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id.trim())) { setError('Paste the session ID shown by the glasses app.'); return; }
    try {
      const attached = await api<Session>(`/sessions/${encodeURIComponent(id.trim())}`);
      turnSequence.current++; mediaSequence.current++; controller.current?.abort(); stopRecognition(); stopSpeaking();
      setStream(null); setSession(attached); setMode(attached.mode); setRecipient(attached.recipient_id); setPortion(attached.portion_grams?.toString() ?? '');
      setProfile(attached.profile); setObserver(true); setResult(null); setEvents([]); setImage(null); setVideoUrl(null); setCaptureTime(null); setImageMediaTime(null); latestObserverReply.current = ''; setView('evidence'); setAttachOpen(false); setProfileOpen(false); setError(''); setPhase('idle'); setBusy(false); busyRef.current = false;
      const stageRequested = new URLSearchParams(window.location.search).get('stage') === '1';
      setStageView(stageRequested);
      window.history.replaceState({}, '', `${window.location.pathname}?session=${attached.id}${stageRequested ? '&stage=1' : ''}`);
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not attach to that session.'); }
  }, []);
  useEffect(() => {
    const observedId = new URLSearchParams(window.location.search).get('session');
    if (observedId) { void attach(observedId); return; }
    const savedId = storedSessionId();
    if (!savedId) return;
    let disposed = false;
    async function restore() {
      try {
        const restored = await api<Session>(`/sessions/${encodeURIComponent(savedId!)}`);
        if (disposed) return;
        setSession(restored); setMode(restored.mode); setRecipient(restored.recipient_id); setPortion(restored.portion_grams?.toString() ?? ''); setProfile(restored.profile);
        if (restored.mode === 'fixture' && restored.current_product_id) setSelectedFixture(restored.current_product_id);
        try { const latest = await api<TurnResponse>(`/sessions/${restored.id}/latest`); if (!disposed) ingest(latest, false); }
        catch { /* A session can exist before its first completed answer. */ }
      } catch (cause) {
        if (disposed) return;
        if (cause instanceof ApiError && cause.status === 404) rememberSession(null);
        else setError('Your session could not be restored. Check the server connection and reload.');
      }
    }
    void restore();
    return () => { disposed = true; };
  }, [attach]);
  useEffect(() => {
    if (!observer || !session?.id) return;
    let disposed = false; let inFlight = false;
    async function poll() {
      if (inFlight) return; inFlight = true;
      try {
        const latest = await api<TurnResponse>(`/sessions/${session!.id}/latest`);
        if (!disposed) {
          const key = `${latest.session.generation}:${latest.session.messages.at(-1)?.id ?? latest.reply}`;
          if (key !== latestObserverReply.current) {
            latestObserverReply.current = key; ingest(latest);
            try {
              const frame = await api<{ image_data_url: string; captured_at: string; media_time_s?: number | null }>(`/sessions/${session!.id}/frame`);
              if (!disposed) { setImage(frame.image_data_url); setCaptureTime(frame.captured_at); setImageMediaTime(frame.media_time_s ?? null); }
            } catch { if (!disposed) { setImage(null); setCaptureTime(null); setImageMediaTime(null); } }
          }
          setError('');
        }
      } catch (cause) {
        if (cause instanceof ApiError && cause.status === 404) {
          try { const current = await api<Session>(`/sessions/${session!.id}`); if (!disposed) { setSession(current); setRecipient(current.recipient_id); setPortion(current.portion_grams?.toString() ?? ''); setResult(null); setImage(null); setCaptureTime(null); setError(''); latestObserverReply.current = ''; } }
          catch { if (!disposed) setError('This session is no longer available. Attach a new glasses session.'); }
        } else if (!disposed) setError('Connection interrupted. Reconnecting to the glasses session…');
      } finally { inFlight = false; }
    }
    void poll(); const interval = window.setInterval(() => { void poll(); }, 500);
    return () => { disposed = true; clearInterval(interval); };
  }, [observer, session?.id]);

  function stopRecognition() {
    const active = recognition.current; recognition.current = null; active?.abort();
  }
  function stopSpeaking() {
    speechSequence.current++; window.speechSynthesis?.cancel();
  }
  function speak(text: string) {
    if (!voiceEnabled || observer || !('speechSynthesis' in window)) return;
    stopSpeaking();
    const sequence = speechSequence.current;
    const utterance = new SpeechSynthesisUtterance(text); utterance.lang = language; utterance.rate = 0.96;
    utterance.onstart = () => { if (sequence === speechSequence.current) setPhase('speaking'); };
    utterance.onend = utterance.onerror = () => { if (sequence === speechSequence.current) setPhase('idle'); };
    window.speechSynthesis.speak(utterance);
  }
  async function stop(cancelServer = true) {
    turnSequence.current++; mediaSequence.current++; controller.current?.abort(); stopRecognition(); stopSpeaking();
    setPhase('idle'); setBusy(false); busyRef.current = false;
    if (session && cancelServer && !observer) {
      const sequence = turnSequence.current; setBusy(true); busyRef.current = true;
      try { const cancelled = await api<Session>(`/sessions/${session.id}/cancel`, { method: 'POST' }); if (sequence === turnSequence.current) setSession(cancelled); }
      catch { setHint('Stopped locally. Reconnect before making another change.'); }
      finally { if (sequence === turnSequence.current) { setBusy(false); busyRef.current = false; } }
    }
  }
  async function startSession(nextProfile: Profile, nextMode: Mode) {
    setBusy(true); setError('');
    try {
      await stop();
      setBusy(true); busyRef.current = true;
      const created = await api<Session>('/sessions', { method: 'POST', body: { mode: nextMode, profile: nextProfile } });
      rememberSession(created.id);
      if (session && !observer) await api(`/sessions/${session.id}`, { method: 'DELETE' }).catch(() => {});
      setSession(created); setProfile(created.profile); setMode(created.mode); setResult(null); setEvents([]); setRecipient('self'); setPortion('');
      setProfileOpen(false); setObserver(false); setImage(null); setVideoUrl(null); setStream(null); setCaptureTime(null); setPendingCapture(false); setHint(''); setQuestion('');
      window.history.replaceState({}, '', window.location.pathname);
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not start this session.'); }
    finally { setBusy(false); busyRef.current = false; }
  }
  function requestMode(nextMode: Mode) { setProfileMode(nextMode); setProfileOpen(true); }
  async function changeRecipient(next: Recipient) {
    if (!session || observer || next === recipient) return;
    await stop(); setBusy(true); busyRef.current = true; setError('');
    const sequence = ++turnSequence.current; controller.current?.abort(); controller.current = new AbortController();
    try {
      const updated = await api<Session>(`/sessions/${session.id}/context`, { method: 'POST', body: { recipient_id: next }, signal: controller.current.signal });
      if (sequence !== turnSequence.current) return;
      setSession(updated); setRecipient(updated.recipient_id); setPortion(''); setResult(null);
      setHint(next === 'household' ? 'Shopping for family. Your personal goals are not applied; tell us the intended portion.' : 'Back to your profile. Tell us the portion for this choice.');
    } catch (cause) { if (sequence === turnSequence.current) setError(cause instanceof Error ? cause.message : 'Could not change recipient.'); }
    finally { if (sequence === turnSequence.current) { setBusy(false); busyRef.current = false; } }
  }
  async function startCamera() {
    setError('');
    if (!navigator.mediaDevices?.getUserMedia) { setError('Camera access needs localhost or HTTPS and a supported browser. You can upload a label photo instead.'); return; }
    const sequence = ++mediaSequence.current; stopRecognition(); stopSpeaking(); setPhase('idle');
    try {
      const media = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' }, width: { ideal: 1600 } }, audio: false });
      if (sequence !== mediaSequence.current) { media.getTracks().forEach(track => track.stop()); return; }
      setStream(media); setVideoUrl(null); setImage(null); setCaptureTime(null); setPendingCapture(true);
      setHint('Camera preview is local. Press Ask to capture and analyze one frame.');
    } catch { if (sequence === mediaSequence.current) setError('Camera access was unavailable. Check permission or upload a clear label photo.'); }
  }
  function captureFrame(): { image_data_url: string; captured_at: string; media_time_s?: number } | null {
    const video = videoRef.current;
    const source = captureSource(!!(stream || videoUrl), !!video && video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0, !!image);
    if (source === 'video' && video) {
      const factor = Math.min(1, 1600 / Math.max(video.videoWidth, video.videoHeight));
      const canvas = document.createElement('canvas'); canvas.width = Math.round(video.videoWidth * factor); canvas.height = Math.round(video.videoHeight * factor);
      canvas.getContext('2d')!.drawImage(video, 0, 0, canvas.width, canvas.height);
      const data = canvas.toDataURL('image/jpeg', 0.88); const at = new Date().toISOString(); setImage(data); setCaptureTime(at); setImageMediaTime(videoUrl ? video.currentTime : null);
      return { image_data_url: data, captured_at: at, ...(videoUrl ? { media_time_s: video.currentTime } : {}) };
    }
    return source === 'image' && image ? { image_data_url: image, captured_at: captureTime ?? new Date().toISOString(), ...(imageMediaTime == null ? {} : { media_time_s: imageMediaTime }) } : null;
  }
  async function uploadImage(file?: File) {
    if (!file) return; setError('');
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) { setError('Choose a JPEG, PNG, or WebP label photo.'); return; }
    if (file.size > 15 * 1024 * 1024) { setError('Choose a photo under 15 MB.'); return; }
    const sequence = ++mediaSequence.current; stopRecognition(); stopSpeaking(); setPhase('idle');
    const url = URL.createObjectURL(file);
    try {
      const img = new Image(); img.src = url; await img.decode();
      if (sequence !== mediaSequence.current) return;
      const factor = Math.min(1, 1600 / Math.max(img.naturalWidth, img.naturalHeight));
      const canvas = document.createElement('canvas'); canvas.width = Math.round(img.naturalWidth * factor); canvas.height = Math.round(img.naturalHeight * factor);
      canvas.getContext('2d')!.drawImage(img, 0, 0, canvas.width, canvas.height);
      setImage(canvas.toDataURL('image/jpeg', 0.88)); setCaptureTime(new Date().toISOString()); setImageMediaTime(null); setStream(null); setVideoUrl(null); setResult(null); setPendingCapture(true);
      setHint('Photo ready. Ask a question to send this label for analysis.');
    } catch { if (sequence === mediaSequence.current) setError('That image could not be opened. Try another photo.'); }
    finally { URL.revokeObjectURL(url); }
  }
  function uploadVideo(file?: File) {
    if (!file) return;
    if (!file.type.startsWith('video/')) { setError('Choose a video file supported by this browser.'); return; }
    mediaSequence.current++; stopRecognition(); stopSpeaking(); setPhase('idle');
    setStream(null); setImage(null); setCaptureTime(null); setVideoUrl(URL.createObjectURL(file)); setResult(null); setError(''); setPendingCapture(true);
    setHint('Recorded POV stays on this device. Pause on a clear label and ask; only that captured frame is sent.');
  }
  async function sendTurn(text: string, forceCapture = false) {
    if (observer) return;
    if (!session) { setProfileMode(mode); setProfileOpen(true); return; }
    if (!text.trim() || busyRef.current) return;
    let amount: number | undefined;
    try { amount = parsePortion(portion); } catch (cause) { setError((cause as Error).message); return; }
    stopSpeaking(); stopRecognition();
    const sequence = ++turnSequence.current; controller.current?.abort(); controller.current = new AbortController();
    setBusy(true); busyRef.current = true; setPhase('thinking'); setError(''); setHint('');
    const body: TurnRequest = { text: text.trim(), client_turn_id: crypto.randomUUID(), recipient_id: recipient, ...(amount === undefined ? {} : { portion_grams: amount }) };
    if (mode === 'fixture') body.fixture_product_id = selectedFixture;
    else if (shouldSendCapture(mode, pendingCapture, forceCapture)) {
      setPendingCapture(true);
      let capture;
      try { capture = captureFrame(); }
      catch { setError('This frame could not be captured. Try a clear label photo instead.'); setBusy(false); busyRef.current = false; setPhase('idle'); return; }
      if (capture) Object.assign(body, capture);
      else { setError('The new frame is not ready. Wait for the camera or video, or upload a label photo.'); setBusy(false); busyRef.current = false; setPhase('idle'); return; }
    }
    try {
      const response = await api<TurnResponse>(`/sessions/${session.id}/turn`, { method: 'POST', body, signal: controller.current.signal });
      if (sequence !== turnSequence.current) return;
      if (body.image_data_url && response.status !== 'needs_image' && response.status !== 'cancelled') setPendingCapture(false);
      ingest(response); setQuestion(''); setPhase('idle'); speak(response.reply);
    } catch (cause) {
      if (sequence !== turnSequence.current || (cause instanceof Error && cause.name === 'AbortError')) return;
      setError(cause instanceof Error ? cause.message : 'Something interrupted the request. Please try again.'); setPhase('idle');
    } finally { if (sequence === turnSequence.current) { setBusy(false); busyRef.current = false; } }
  }
  function toggleListening() {
    if (!session) { setProfileMode(mode); setProfileOpen(true); return; }
    if (phase === 'listening') { recognition.current?.stop(); return; }
    const Constructor = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Constructor) { setHint('Voice recognition is not supported here. Type your question below.'); inputRef.current?.focus(); return; }
    if (busy) return;
    stopSpeaking(); stopRecognition();
    const speech = new Constructor(); speech.lang = language; speech.continuous = false; speech.interimResults = true;
    let finalText = ''; let failed = false;
    speech.onresult = event => {
      if (recognition.current !== speech) return;
      const pieces = Array.from(event.results); const text = pieces.map(piece => piece[0].transcript).join(' '); setQuestion(text);
      finalText = pieces.filter(piece => piece.isFinal).map(piece => piece[0].transcript).join(' ');
    };
    speech.onerror = event => { failed = true; if (recognition.current !== speech) return; setPhase('idle'); if (event.error !== 'aborted') setHint(`Microphone recognition stopped (${event.error}). You can type your question.`); };
    speech.onend = () => { if (recognition.current !== speech) return; setPhase('idle'); recognition.current = null; if (finalText && !failed) void sendTurn(finalText); };
    recognition.current = speech;
    try { speech.start(); setPhase('listening'); setHint('Listening until you pause or press the microphone again. Browser speech may use its own recognition service.'); }
    catch { if (recognition.current === speech) recognition.current = null; setPhase('idle'); setHint('Could not start speech recognition. Type your question instead.'); }
  }
  async function confirmAction() {
    if (!session || !pending || observer || busyRef.current) return;
    const sequence = ++turnSequence.current; controller.current?.abort(); controller.current = new AbortController(); stopRecognition(); stopSpeaking();
    setBusy(true); busyRef.current = true; setPhase('thinking'); setError('');
    try { const response = await api<TurnResponse>(`/sessions/${session.id}/confirm`, { method: 'POST', body: { action_id: pending.id }, signal: controller.current.signal }); if (sequence !== turnSequence.current) return; ingest(response); setPhase('idle'); speak(response.reply); }
    catch (cause) { if (sequence === turnSequence.current) { setPhase('idle'); setError(cause instanceof Error ? cause.message : 'Could not confirm. Ask to add the item again.'); } }
    finally { if (sequence === turnSequence.current) { setBusy(false); busyRef.current = false; } }
  }
  async function review(corrections: { name: string; nutrition: Nutrition; ingredients: string[] }) {
    if (!session || !activeProduct || observer || busyRef.current) return;
    const sequence = ++turnSequence.current; controller.current?.abort(); controller.current = new AbortController(); stopRecognition(); stopSpeaking();
    setBusy(true); busyRef.current = true; setPhase('idle');
    try { const response = await api<TurnResponse>(`/sessions/${session.id}/review`, { method: 'POST', body: { product_id: activeProduct.id, confirmed: true, corrections }, signal: controller.current.signal }); if (sequence !== turnSequence.current) return; ingest(response); setReviewOpen(false); speak(response.reply); }
    finally { if (sequence === turnSequence.current) { setBusy(false); busyRef.current = false; } }
  }
  async function deleteSession() {
    if (!session) return;
    await stop();
    try {
      if (!observer) await api(`/sessions/${session.id}`, { method: 'DELETE' });
      if (!observer && storedSessionId() === session.id) rememberSession(null);
      setSession(null); setResult(null); setEvents([]); setSummaryOpen(false); setObserver(false); setImage(null); setVideoUrl(null); setStream(null); setCaptureTime(null); setPortion(''); setQuestion(''); setHint(observer ? 'Viewer detached. The original session remains available on the server.' : 'Session cleared from this device and local server. Provider retention follows their policies.');
      window.history.replaceState({}, '', window.location.pathname);
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not delete the session.'); }
  }
  async function copyBridge() {
    if (!session) return;
    try { await navigator.clipboard.writeText(session.id); setBridgeCopied(true); window.setTimeout(() => setBridgeCopied(false), 1800); }
    catch { setHint(`Session ID: ${session.id}`); }
  }
  async function reconnect() {
    await connect();
    if (!session || observer || busyRef.current) return;
    try {
      const restored = await api<Session>(`/sessions/${session.id}`);
      setSession(restored); setRecipient(restored.recipient_id); setPortion(restored.portion_grams?.toString() ?? '');
      try { ingest(await api<TurnResponse>(`/sessions/${session.id}/latest`)); }
      catch (cause) { if (cause instanceof ApiError && cause.status === 404) setResult(null); else throw cause; }
      setError(''); setHint('Reconnected. The confirmed list now matches the server.');
    } catch (cause) {
      if (cause instanceof ApiError && cause.status === 404) {
        rememberSession(null); setSession(null); setResult(null); setEvents([]); setStream(null); setImage(null); setVideoUrl(null); setCaptureTime(null); setPendingCapture(false);
        setError('The previous session is no longer on the server. Start a new shopping session.');
      } else setError(cause instanceof Error ? cause.message : 'Could not restore the session.');
    }
  }
  async function openStageView() {
    if (!session) return;
    window.history.replaceState({}, '', `${window.location.pathname}?session=${session.id}&stage=1`);
    if (observer) { setStageView(true); window.speechSynthesis?.cancel(); }
    else await attach(session.id);
  }
  function exitStageView() {
    setStageView(false);
    window.history.replaceState({}, '', `${window.location.pathname}${session && observer ? `?session=${session.id}` : ''}`);
  }

  if (stageView && observer && session) return <StageView session={session} result={result} image={image} capturedAt={captureTime} connected={!!health && !error} error={error} onExit={exitStageView}/>;

  return <div className="app-shell">
    <a className="skip-link" href="#main">Skip to shopping</a>
    <header className="site-header"><a className="brand" href={window.location.pathname} aria-label="Shelf Aware home"><span className="brand-icon"><Icon name="leaf" size={26}/></span><span>Shelf<span className="brand-aware">Aware</span><small>YOUR EVERYDAY SHOPPING COMPANION</small></span></a><nav className="view-switch" aria-label="View"><button className={view === 'shop' ? 'active' : ''} onClick={() => setView('shop')}>The shopping trip</button><button className={view === 'evidence' ? 'active' : ''} onClick={() => setView('evidence')}><Icon name="eye" size={16}/>Behind the answer</button></nav><div className="header-actions"><span className={`connection ${health ? 'online' : ''}`}><span/>{health ? 'Local server connected' : 'Server offline'}</span><button className="avatar" disabled={busy} onClick={() => { setProfileMode(mode); setProfileOpen(true); }} aria-label="Review profile">{profile.display_name.charAt(0).toUpperCase()}</button></div></header>

    <main id="main">
      <section className="page-intro"><div><div className="eyebrow"><span className="small-line"/> A LITTLE CLARITY IN EVERY AISLE</div><h1>{view === 'shop' ? <>Good choices start with<br/><em>better questions.</em></> : <>A clear answer.<br/><em>A visible reason.</em></>}</h1><p>{view === 'shop' ? 'Hold a product. Ask what matters. Compare the portion you actually eat.' : 'Follow the product, the source, and the calculation behind each response.'}</p></div><div className="intro-note"><span className="note-number">01—03</span><span>See the label.<br/>Understand your portion.<br/><strong>Choose for yourself.</strong></span><svg viewBox="0 0 92 50" aria-hidden="true"><path d="M4 10c35-17 20 39 77 21m-13-9 15 9-9 12"/></svg></div></section>

      {error && <div className="alert error-alert" role="alert"><Icon name="info"/><span>{error}</span><button onClick={() => { void reconnect(); }}><Icon name="refresh" size={16}/>Reconnect</button><button className="icon-button" aria-label="Dismiss error" onClick={() => setError('')}><Icon name="close" size={16}/></button></div>}
      <div className="session-strip"><div className="source-controls"><span className={`mode-badge ${mode === 'fixture' ? 'synthetic' : ''}`}><span/>{modeLabels[mode]}</span><select aria-label="Input source" value={mode} disabled={busy || observer} onChange={event => requestMode(event.target.value as Mode)}>{modeOptions.map(option => <option key={option} value={option}>{modeLabels[option]}</option>)}</select></div><div className="session-tools"><span>{observer ? 'Observer only · no audio or state changes' : mode === 'fixture' ? 'Fictional products & values · no live AI' : mode === 'recorded_glasses' ? 'Recorded footage · frame analysis on request' : mode === 'live_glasses' ? 'Native bridge required · hardware unverified' : 'Phone input · glasses integration not implied'}</span>{session && <button className="text-button" disabled={busy} onClick={() => void openStageView()}>Stage view<Icon name="eye" size={15}/></button>}<button className="text-button" disabled={busy} onClick={() => setAttachOpen(true)}>{observer ? 'Change session' : 'Attach glasses session'}<Icon name="arrow" size={15}/></button></div></div>

      <div className={`workspace ${view === 'evidence' ? 'evidence-view' : ''}`}>
        <section className="product-column">
          <div className="camera-card"><div className="card-topline"><div><span className="eyebrow">{mode === 'fixture' ? 'ON THE PRACTICE SHELF' : 'YOUR VISUAL CONTEXT'}</span><h2>{mode === 'fixture' ? fixture?.name ?? 'Choose a product' : mode === 'recorded_glasses' ? 'A moment from your aisle.' : 'What’s in your hand?'}</h2></div><span className="view-mark"><Icon name="eye" size={16}/>{mode === 'fixture' ? 'Demo' : 'POV'}</span></div>
            <MediaStage fixture={fixture} fixtureMode={mode === 'fixture'} imageUrl={image} videoUrl={videoUrl} stream={stream} videoRef={videoRef} events={result?.events ?? []} hasCapturedImage={!!captureTime} onTimeChange={onTimeChange} imageMediaTime={imageMediaTime} observer={observer}/>
            <div className="media-caption"><span><Icon name="info" size={14}/>{mode === 'fixture' ? 'Fictional labels for practicing the complete flow.' : observer ? 'Latest captured frame · not a live video stream.' : 'Companion visualization · audio uses the selected device output.'}</span>{captureTime && <span>Captured {timeLabel(captureTime)}</span>}</div>
            {mode === 'fixture' ? <div className="fixture-picker" aria-label="Choose a fictional product">{fixtures.map((product, index) => <button key={product.id} className={selectedFixture === product.id ? 'selected' : ''} disabled={busy || observer} onClick={() => { setSelectedFixture(product.id); setResult(null); }}><span className={`mini-package mini-${index}`}>{index < 2 ? String.fromCharCode(65+index) : '?'}</span><span>{product.name}<small>{index < 2 ? `${product.nutrition.basis_grams} g label serving` : 'Practice an unknown label'}</small></span>{selectedFixture === product.id && <Icon name="check" size={16}/>}</button>)}</div> : <div className="capture-controls">{mode === 'live_phone' && <button className="button secondary" disabled={observer || busy} onClick={() => stream ? (mediaSequence.current++, setStream(null)) : void startCamera()}><Icon name={stream ? 'stop' : 'camera'} size={17}/>{stream ? 'Close camera' : 'Open camera'}</button>}{mode === 'recorded_glasses' && <button className="button secondary" disabled={observer || busy} onClick={() => videoInput.current?.click()}><Icon name="play" size={16}/>Choose POV video</button>}{mode !== 'live_glasses' && <button className="button secondary" disabled={observer || busy} onClick={() => imageInput.current?.click()}><Icon name="upload" size={16}/>Upload label</button>}{(stream || videoUrl) && <button className="button secondary" disabled={observer || busy || !session} onClick={() => void sendTurn(question.trim() || 'Look at this product.', true)}><Icon name="camera" size={16}/>Look &amp; ask · new frame</button>}{mode === 'live_glasses' && <button className="button secondary" onClick={() => void copyBridge()} disabled={!session}><Icon name="check" size={16}/>{bridgeCopied ? 'Session ID copied' : 'Copy bridge session ID'}</button>}<input ref={imageInput} type="file" accept="image/jpeg,image/png,image/webp" hidden onChange={event => { void uploadImage(event.target.files?.[0]); event.target.value = ''; }}/><input ref={videoInput} type="file" accept="video/*" hidden onChange={event => { uploadVideo(event.target.files?.[0]); event.target.value = ''; }}/></div>}
          </div>

          {view === 'evidence' ? <EvidencePanel evidence={result?.evidence ?? []} events={events} product={activeProduct} health={health} observer={observer}/> : <div className="how-it-works"><div><span>01</span><h3>Your portion matters.</h3><p>A serving on the label may be different from what you eat.</p></div><div><span>02</span><h3>Every answer has a source.</h3><p>Review the facts. Keep the choice in your hands.</p></div></div>}
        </section>

        <aside className="conversation-column" aria-label="Shopping companion">
          <section className="context-card"><div className="section-heading"><h2>Your shopping context</h2><Icon name="user" size={18}/></div><div className="recipient-switch" aria-label="Who is this for?"><button disabled={!session || busy || observer} className={recipient === 'self' ? 'selected' : ''} onClick={() => void changeRecipient('self')}>For me</button><button disabled={!session || busy || observer} className={recipient === 'household' ? 'selected' : ''} onClick={() => void changeRecipient('household')}>For my family</button></div><div className="portion-row"><label htmlFor="portion">How much will you eat?<small>Use the same amount to compare fairly.</small></label><div className="input-unit portion-input"><input id="portion" type="number" min="0.1" max="5000" step="any" placeholder="e.g. 60" value={portion} disabled={busy || observer} onChange={event => setPortion(event.target.value)}/><span>g</span></div></div><div className="profile-summary"><span className="tiny-leaf"><Icon name="leaf" size={13}/></span>{recipient === 'self' ? session ? session.profile.goals[0] ?? 'General shopper · label and portion context' : 'A profile you review and control.' : 'Family context · personal goals not applied'}</div></section>

          <section className="assistant-card"><div className="assistant-header"><span className={`assistant-orb ${phase !== 'idle' ? 'animated' : ''}`}><Icon name="leaf" size={21}/></span><div><strong>Your aisle companion</strong><span aria-live="polite">{observer ? 'Watching your glasses session' : phase === 'listening' ? 'Listening to your question…' : phase === 'thinking' ? 'Checking the facts…' : phase === 'speaking' ? 'Speaking · you can stop anytime' : 'Here when you need a little clarity'}</span></div><button className="icon-button" aria-label={voiceEnabled ? 'Turn spoken replies off' : 'Enable spoken replies'} aria-pressed={voiceEnabled} disabled={observer} onClick={() => { if (voiceEnabled) { stopSpeaking(); if (phase === 'speaking') setPhase('idle'); } setVoiceEnabled(!voiceEnabled); }}><Icon name={voiceEnabled ? 'sound' : 'mute'} size={19}/></button></div>
            <div className="reply-area" aria-live="polite" aria-atomic="true">{phase === 'thinking' ? <div className="thinking"><i/><i/><i/><span>Finding an answer with evidence.</span></div> : <><span className="quote-mark">“</span><p>{result?.reply ?? (observer ? 'Your next glasses interaction will appear here, with its evidence.' : !session ? 'A label tells you what’s inside. Let’s work out what it means for your portion.' : activeProduct ? `Ready for the next question. We’ll keep ${activeProduct.name} in mind.` : 'Pick a product and ask, “Is this a good option for me?”')}</p></>}{result?.status === 'needs_review' && <button className="button review-button" onClick={() => setReviewOpen(true)} disabled={observer}><Icon name="eye" size={17}/>Review extracted label</button>}{result?.status === 'needs_portion' && <button className="text-button" onClick={() => document.getElementById('portion')?.focus()}>Set your portion above<Icon name="arrow" size={15}/></button>}</div>
            {mode !== 'fixture' && !observer && session && <p className="inline-hint">{pendingCapture ? 'Your next question sends this photo or captures one frame.' : activeProduct ? `Ask about ${activeProduct.name}. To change products, ${stream || videoUrl ? 'choose “Look & ask · new frame”.' : 'upload a new label.'}` : 'Open a camera or upload a label to begin.'}</p>}
            {hint && <p className="inline-hint" role="status">{hint}</p>}
            {!observer && <div className="ask-controls">{!session ? <button className="button primary start-button" onClick={() => { setProfileMode(mode); setProfileOpen(true); }}>Start a shopping session<Icon name="arrow"/></button> : <><div className="voice-controls"><button className={`talk-button ${phase === 'listening' ? 'listening' : ''}`} onClick={toggleListening} disabled={busy} aria-label={phase === 'listening' ? 'Finish recording your question' : 'Ask with your voice'}><Icon name="mic"/>{phase === 'listening' ? 'Finish question' : 'Ask Shelf Aware'}<span className="wave-bars" aria-hidden="true"><i/><i/><i/><i/><i/></span></button><button className="stop-button" onClick={() => void stop()} disabled={phase === 'idle' && !busy && !pending} aria-label="Stop speech and cancel pending request"><Icon name="stop" size={16}/><span>Stop</span></button></div><form className="question-form" onSubmit={event => { event.preventDefault(); void sendTurn(question); }}><input ref={inputRef} aria-label="Type a shopping question" placeholder={speechAvailable ? 'Or type your question…' : 'Type your question here…'} value={question} disabled={busy} onChange={event => setQuestion(event.target.value)}/><button aria-label="Send question" disabled={busy || !question.trim()}><Icon name="arrow" size={20}/></button></form><div className="suggestions"><button onClick={() => void sendTurn('Is this a good option?')} disabled={busy}>Is this a good option?</button><button onClick={() => void sendTurn('Compare this with the previous product.')} disabled={busy}>Compare these</button><button onClick={() => void sendTurn('Add this to my shopping list.')} disabled={busy || needsReview}>Add to my list</button></div></>}{session && <div className="audio-footnote"><span>{voiceEnabled ? 'Spoken replies on · system audio output' : 'Spoken replies off'}</span><label><span className="sr-only">Speech language</span><select value={language} onChange={event => setLanguage(event.target.value)}><option value="en-US">English</option><option value="es-US">Español</option><option value="hi-IN">हिन्दी</option></select></label></div>}</div>}
          </section>

          {pending && <section className="confirmation-card" aria-labelledby="confirm-title"><span className="eyebrow">ONE QUICK CONFIRMATION</span><h3 id="confirm-title">{pending.label}</h3><p>{pending.quantity} package{pending.quantity === 1 ? '' : 's'} · {pending.recipient_id === 'self' ? 'For you' : 'For your family'}</p><button className="button primary" onClick={() => void confirmAction()} disabled={busy || observer}><Icon name="check" size={17}/>Yes, add to my list</button><span>No purchase or payment is made.</span></section>}
          <section className="basket-card"><div className="section-heading"><h2><Icon name="basket" size={19}/>Your confirmed list</h2><span className="count-badge">{session?.cart.length ?? 0}</span></div>{session?.cart.length ? <ul className="cart-list">{session.cart.map(item => <li key={item.id}><span className="cart-check"><Icon name="check" size={14}/></span><div><strong>{item.product_name}</strong><span>{item.recipient_id === 'self' ? 'For you' : 'For your family'} · {item.quantity} package{item.quantity === 1 ? '' : 's'}</span></div></li>)}</ul> : <p className="empty-list">A little less uncertainty.<br/>Add your first choice when you’re ready.</p>}{session && <button className="text-button finish-button" onClick={() => { void stop(); setStream(null); setSummaryOpen(true); }}>End & review this trip<Icon name="arrow" size={16}/></button>}</section>
        </aside>
      </div>

      {result?.comparison && <ComparisonCard comparison={result.comparison} synthetic={mode === 'fixture'}/>}
      {activeProduct && view === 'shop' && <section className="source-summary"><div><span className="eyebrow">LAST ANALYZED LABEL</span><h3>{activeProduct.name}</h3><p>{sourceLabels[activeProduct.provenance.kind]} · {activeProduct.nutrition.basis_grams} g label serving · {activeProduct.market}</p></div><button className="button secondary" onClick={() => setView('evidence')}>Show the evidence<Icon name="arrow" size={17}/></button></section>}
      {session && <details className="conversation-history"><summary>Conversation & session details <span>{session.messages.length} messages</span></summary><div className="history-meta"><span>Session {session.id}</span><button className="text-button" onClick={() => void copyBridge()}>{bridgeCopied ? 'Copied' : 'Copy session ID'}</button></div>{session.messages.length === 0 ? <p>Your conversation will appear here.</p> : <ol>{session.messages.map(message => <li key={message.id}><strong>{message.role === 'user' ? 'You' : 'Shelf Aware'}</strong><p>{message.text}</p><time>{timeLabel(message.created_at)}</time></li>)}</ol>}</details>}
    </main>
    <footer className="site-footer"><span className="footer-brand"><Icon name="leaf" size={17}/> A little clarity goes a long way.</span><span>Nutrition context, not glucose prediction or treatment advice.</span><span>Built for DubHacks ’26</span></footer>

    {profileOpen && <ProfileDialog initial={profile} mode={profileMode} busy={busy} hasSession={!!session} error={error} onClose={() => setProfileOpen(false)} onStart={startSession}/>}
    {reviewOpen && activeProduct && <ReviewDialog key={activeProduct.id} product={activeProduct} image={image} busy={busy} onClose={() => setReviewOpen(false)} onConfirm={review}/>}
    {attachOpen && <div className="modal-backdrop"><section className="modal compact-modal" role="dialog" aria-modal="true" aria-labelledby="attach-heading"><button className="icon-button modal-close" aria-label="Close attach session" onClick={() => setAttachOpen(false)}><Icon name="close"/></button><div className="eyebrow">THE JUDGES’ WINDOW</div><h2 id="attach-heading">See the same session.</h2><p>Paste the session ID from the native glasses app. This view observes real results without sending requests or playing a second voice.</p><form onSubmit={event => { event.preventDefault(); void attach(attachId); }}><label>Session ID<input value={attachId} onChange={event => setAttachId(event.target.value)} placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" required autoFocus/></label><button className="button primary" type="submit">Attach evidence view<Icon name="arrow"/></button></form></section></div>}
    {summaryOpen && session && <div className="modal-backdrop"><section className="modal compact-modal" role="dialog" aria-modal="true" aria-labelledby="summary-heading"><button className="icon-button modal-close" aria-label="Close trip summary" onClick={() => setSummaryOpen(false)}><Icon name="close"/></button><div className="eyebrow">YOUR TRIP, WITH A LITTLE MORE CLARITY</div><h2 id="summary-heading">Thoughtfully chosen.</h2><p>{session.products.length} product{session.products.length === 1 ? '' : 's'} considered. {session.cart.length} confirmed choice{session.cart.length === 1 ? '' : 's'}.</p><span className="mode-badge synthetic">{modeLabels[session.mode]}</span><ul className="summary-list">{session.cart.length ? session.cart.map(item => <li key={item.id}><Icon name="check" size={18}/><span>{item.product_name}<small>{item.recipient_id === 'self' ? 'For you' : 'For your family'} · {item.quantity} package{item.quantity === 1 ? '' : 's'}</small></span></li>) : <li>No items confirmed. Your choices remain yours.</li>}</ul><p className="muted">Camera and speech are stopped. The list stays in this local session until you clear it or the server restarts.</p><div className="modal-actions"><button className="button secondary" onClick={() => setSummaryOpen(false)}>Keep shopping</button><button className="button primary" onClick={() => void deleteSession()}>{observer ? 'Detach this viewer' : 'Clear session & finish'}</button></div></section></div>}
  </div>;
}

function ComparisonCard({ comparison, synthetic }: { comparison: Comparison; synthetic: boolean }) {
  const rows: { key: 'carbohydrate_g' | 'fiber_g' | 'sugars_g' | 'protein_g' | 'sodium_mg'; title: string; unit: string }[] = [{ key: 'carbohydrate_g', title: 'Total carbohydrate', unit: 'g' }, { key: 'fiber_g', title: 'Fiber', unit: 'g' }, { key: 'sugars_g', title: 'Total sugars', unit: 'g' }, { key: 'protein_g', title: 'Protein', unit: 'g' }, { key: 'sodium_mg', title: 'Sodium', unit: 'mg' }];
  return <section className="comparison-card"><div className="comparison-intro"><span className="eyebrow">A FAIRER COMPARISON</span><h2>Same portion.<br/><em>A clearer picture.</em></h2><p>Each product at <strong>{comparison.portion_grams} grams</strong> · {comparison.recipient_id === 'self' ? 'For you' : 'For your family'}</p>{synthetic && <span className="mode-badge synthetic">Synthetic values</span>}</div><div className="comparison-data"><div className="table-scroll"><table><caption className="sr-only">Nutrition compared at the same intended portion</caption><thead><tr><th scope="col">At your portion</th>{comparison.items.map(item => <th scope="col" key={item.product_id}>{item.name}</th>)}</tr></thead><tbody>{rows.map(row => <tr key={row.key}><th scope="row">{row.title}</th>{comparison.items.map(item => <td key={item.product_id}>{formatAmount(item[row.key], row.unit)}</td>)}</tr>)}</tbody></table></div><p className="comparison-summary"><Icon name="info" size={17}/>{comparison.summary}</p></div></section>;
}

function EvidencePanel({ evidence, events, product, health, observer }: { evidence: Evidence[]; events: EvidenceEvent[]; product: Product | null; health: Health | null; observer: boolean }) {
  return <section className="evidence-panel"><div className="section-heading"><h2>The answer, opened up.</h2><span className="eyebrow">{observer ? 'OBSERVER VIEW' : 'EVIDENCE VIEW'}</span></div><div className="provider-status"><span>Perceptron <strong>{health?.providers.perceptron ? 'configured' : 'not configured'}</strong></span><span>Gemini <strong>{health?.providers.gemini ? 'configured' : 'not configured'}</strong></span><small>Configured is not proof of a successful API call. Check the actual events below.</small></div>{evidence.length ? <div className="evidence-grid">{evidence.map((item, index) => <article key={`${item.label}-${index}`}><span>{item.label}</span><strong>{item.value}</strong><small>{sourceLabels[item.kind]} · {item.verified ? 'reviewed / verified' : 'unverified'}</small><p>{item.source}</p></article>)}</div> : <div className="evidence-empty"><Icon name="eye" size={24}/><p>Ask about a product to see its supporting facts.<br/>No evidence has been generated yet.</p></div>}{product && <details className="product-facts"><summary>Label record · {product.name}</summary><dl><div><dt>Source</dt><dd>{product.provenance.source}</dd></div><div><dt>Captured</dt><dd>{new Date(product.provenance.captured_at).toLocaleString()}</dd></div><div><dt>Serving basis</dt><dd>{product.nutrition.basis_grams} g</dd></div><div><dt>Barcode</dt><dd>{product.barcode ?? 'Not available'}</dd></div><div><dt>Ingredients</dt><dd>{product.ingredients.join(', ') || 'Not available'}</dd></div>{product.uncertain_fields.length > 0 && <div><dt>Uncertain</dt><dd>{product.uncertain_fields.join(', ')}</dd></div>}</dl></details>}<div className="event-heading"><h3>What actually happened</h3><span>{events.length} events</span></div>{events.length ? <ol className="event-list">{events.slice().reverse().map(event => <li key={event.id}><span className="event-dot"/><div><span className="event-stage">{event.stage} <time>{timeLabel(event.at)}</time></span><p>{event.message}</p>{event.media_time_s != null && <small>Recorded media time {event.media_time_s.toFixed(2)}s</small>}</div></li>)}</ol> : <p className="muted">Pipeline events will appear after an interaction.</p>}</section>;
}

export default App;
