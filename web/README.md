# Shelf Aware browser companion

The shopper dashboard and silent stage observer share the local backend on port 8000. Source files and the lockfile are portable; install dependencies on the target Mac instead of copying Windows `node_modules`.

## Mac commands

Use **Node 22.12 or later within Node 22, Node 24, or Node 26+**. Node 24 is a straightforward choice. The installed Vitest 5 dependency does not support early Node 22 releases. Run from this `web` directory:

```sh
node --version
npm ci
npm test
npm run build
npm run dev
```

Open `http://localhost:5173`. Start the Python backend separately using the project-level instructions. `npm run preview` serves the built bundle at `http://localhost:4173` and inherits the same `/api` proxy. Use the root validation commands to test the browser source. Browser tests do not establish native glasses or live provider behavior.

Provider keys belong in `../.env` or the backend process environment. An optional `SHELF_API_TOKEN` in `../.env` is read by the local Vite server and injected into proxied requests. Never put a provider key or this token in a `VITE_` variable. Restart Vite when its server-side environment changes. The frontend build alone is not a deployed API server.

## Presentation and recovery

- Shopper dashboard: `http://localhost:5173`.
- Silent dashboard observer: `http://localhost:5173/?session=SESSION_ID`.
- Silent stage projection: `http://localhost:5173/?session=SESSION_ID&stage=1`.
- Stage text is the completed backend reply, not a word-aligned or streaming speech transcript. Images are timestamped requested stills, not a live video stream. No browser TTS runs in observer mode.
- A real photo is sent once; follow-up questions reuse the current product. Use **Look & ask · new frame** for another camera/video capture, or upload the next label.
- OCR candidates require label review. List changes require explicit confirmation.
- Stop cancels local speech and in-flight requests. It cannot undo a list confirmation already received by the server. Reconnect refreshes the confirmed list after an uncertain response.
- Observer GET requests time out after 8 seconds; other requests after 45 seconds. No requests or list changes are automatically replayed.
- Browser voice recognition depends on browser/OS permission and service availability. Typed questions are the fallback. Browser speech uses the computer's configured audio output and does not prove glasses audio routing.
- Camera capture needs localhost or HTTPS. A plain HTTP LAN address is suitable for displaying the stage observer, but browsers generally block camera/microphone capture there. The native glasses app uses its own API connection.

Only the active session ID is stored in browser session storage. Images, profiles, and conversation text are not persisted there. Sessions disappear when the local backend restarts.
