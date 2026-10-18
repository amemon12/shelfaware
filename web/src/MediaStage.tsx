import { useEffect, useRef, useState } from 'react';
import type { RefObject } from 'react';
import type { EvidenceEvent, Product } from './types';
import { visibleBoxes } from './utils';
import { Icon } from './Icon';

type Props = {
  fixture?: Product; fixtureMode: boolean; imageUrl: string | null; videoUrl: string | null;
  stream: MediaStream | null; videoRef: RefObject<HTMLVideoElement | null>;
  events: EvidenceEvent[]; hasCapturedImage: boolean; onTimeChange: (time: number) => void;
  imageMediaTime?: number | null; observer?: boolean;
};
export function MediaStage({ fixture, fixtureMode, imageUrl, videoUrl, stream, videoRef, events, hasCapturedImage, onTimeChange, imageMediaTime = null, observer = false }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 600, h: 350 });
  const [natural, setNatural] = useState({ w: 16, h: 10 });
  const [mediaTime, setMediaTime] = useState(0);
  useEffect(() => {
    if (!container.current) return;
    const observer = new ResizeObserver(([entry]) => setSize({ w: entry.contentRect.width, h: entry.contentRect.height }));
    observer.observe(container.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    const video = videoRef.current;
    if (video && stream) { video.srcObject = stream; void video.play().catch(() => {}); }
    return () => { if (video?.srcObject) video.srcObject = null; };
  }, [stream, videoRef]);
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !videoUrl || !('requestVideoFrameCallback' in video)) return;
    let callbackId = 0;
    const tick: VideoFrameRequestCallback = (_now, metadata) => {
      setMediaTime(metadata.mediaTime);
      onTimeChange(metadata.mediaTime);
      callbackId = video.requestVideoFrameCallback(tick);
    };
    callbackId = video.requestVideoFrameCallback(tick);
    return () => video.cancelVideoFrameCallback(callbackId);
  }, [videoUrl, videoRef, onTimeChange]);
  const scale = Math.min(size.w / natural.w, size.h / natural.h);
  const display = { w: natural.w * scale, h: natural.h * scale };
  // A live camera preview has moved beyond the analyzed still; never imply tracking.
  const boxes = fixtureMode || stream || !hasCapturedImage || (!imageUrl && !videoUrl) ? [] : visibleBoxes(events, videoUrl ? mediaTime : imageMediaTime);
  return <div className={`media-stage ${fixtureMode ? 'fixture-stage' : ''}`} ref={container}>
    {fixtureMode ? <>
      <span className="stage-caption">A SMALL PRACTICE AISLE</span>
      {fixture?.id === 'unknown' ? <div className="unknown-package"><span>?</span><p>Unfamiliar product</p><small>Some labels need another look.</small></div> : <div className={`package-scene ${fixture?.id === 'oat-b' ? 'package-b' : ''}`}>
        <div className="grain grain-one"/><div className="grain grain-two"/><div className="grain grain-three"/>
        <div className="oat-package"><div className="package-seam"/><div className="package-label"><span className="package-eyebrow">THE PRACTICE PANTRY</span><Icon name="leaf" size={34}/><strong>{fixture?.name ?? 'An everyday choice'}</strong><span className="package-rule"/><span>FICTIONAL PRODUCT</span></div><div className="package-foot">For a clearer kind of shopping.</div></div>
        <span className="stage-note">Illustration only.<br/>No camera or model inference.</span>
      </div>}
    </> : (videoUrl || stream) ? <video ref={videoRef} src={videoUrl ?? undefined} muted playsInline controls={!!videoUrl} onLoadedMetadata={event => { setNatural({ w: event.currentTarget.videoWidth || 16, h: event.currentTarget.videoHeight || 10 }); }} onTimeUpdate={event => { setMediaTime(event.currentTarget.currentTime); onTimeChange(event.currentTarget.currentTime); }} /> : imageUrl ? <img src={imageUrl} alt="Your captured product label, used as the source for extraction" onLoad={event => setNatural({ w: event.currentTarget.naturalWidth, h: event.currentTarget.naturalHeight })} /> : <div className="empty-camera"><div className="viewfinder"><Icon name="camera" size={35}/></div><h3>{observer ? 'Ready for the next capture.' : 'Let’s take a closer look.'}</h3><p>{observer ? <>The latest requested glasses frame will appear here.<br/>This is an evidence view, not a live video stream.</> : <>Open your camera or choose a label photo.<br/>Images are analyzed only when you ask.</>}</p></div>}
    {boxes.length > 0 && <svg className="evidence-boxes" width={size.w} height={size.h} aria-label="Model-observed product locations" style={{ pointerEvents: 'none' }}>
      {boxes.map(event => { const box = event.box!; return <g key={event.id} transform={`translate(${(size.w-display.w)/2},${(size.h-display.h)/2})`}><rect x={box.x*display.w} y={box.y*display.h} width={box.w*display.w} height={box.h*display.h} rx="6"/><text x={box.x*display.w+7} y={box.y*display.h+20}>Model observation</text></g>; })}
    </svg>}
  </div>;
}
