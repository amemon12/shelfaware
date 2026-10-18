export type IconName = 'leaf' | 'mic' | 'stop' | 'arrow' | 'camera' | 'upload' | 'check' | 'close' | 'basket' | 'sound' | 'mute' | 'eye' | 'refresh' | 'play' | 'user' | 'info' | 'chevron';
const paths: Record<IconName, React.ReactNode> = {
  leaf: <><path d="M5 15C5 7 11 4 20 4c0 9-3 15-11 15"/><path d="m5 19 10-10"/></>,
  mic: <><rect x="9" y="3" width="6" height="12" rx="3"/><path d="M5 11v1a7 7 0 0 0 14 0v-1M12 19v3m-4 0h8"/></>,
  stop: <rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none"/>,
  arrow: <path d="M4 12h15m-6-6 6 6-6 6"/>,
  camera: <><path d="M3 7h4l2-3h6l2 3h4v13H3z"/><circle cx="12" cy="13" r="4"/></>,
  upload: <><path d="M12 16V3m-5 5 5-5 5 5M4 16v5h16v-5"/></>,
  check: <path d="m5 12 4 4 10-10"/>,
  close: <path d="m6 6 12 12M18 6 6 18"/>,
  basket: <><path d="m7 9 5-6 5 6M3 9h18l-2 12H5zM9 13v4m6-4v4"/></>,
  sound: <><path d="M4 9h4l5-5v16l-5-5H4zM17 8a6 6 0 0 1 0 8m3-11a10 10 0 0 1 0 14"/></>,
  mute: <><path d="M4 9h4l5-5v16l-5-5H4zM17 9l5 6m0-6-5 6"/></>,
  eye: <><path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/></>,
  refresh: <><path d="M20 8a8 8 0 1 0 0 8M20 3v5h-5"/></>,
  play: <path d="m8 4 12 8-12 8z"/>,
  user: <><circle cx="12" cy="7" r="4"/><path d="M4 21v-2a8 8 0 0 1 16 0v2"/></>,
  info: <><circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v.1"/></>,
  chevron: <path d="m8 5 7 7-7 7"/>,
};
export function Icon({ name, size = 20 }: { name: IconName; size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}
