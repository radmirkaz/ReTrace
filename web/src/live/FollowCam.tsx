import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { Tracks } from '../data'

interface Box { x: number; y: number; w: number; h: number }

/** A second view of the feed that zooms and pans to keep one tracked vehicle centred. */
export default function FollowCam({ src, tracks, track }: { src: string; tracks: Tracks; track: number }) {
  const video = useRef<HTMLVideoElement>(null)
  const frameEl = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ w: 400, h: 300 })
  const [box, setBox] = useState<Box | null>(null)
  const [view, setView] = useState<Box | null>(null)

  // Frames where this vehicle is on screen, with its box; the follow-cam loops over that span.
  const path = useMemo(() => {
    const out = new Map<number, Box>()
    tracks.frames.forEach((f, i) => {
      const d = f.find((x) => x[0] === track)
      if (d) out.set(i, { x: d[1], y: d[2], w: d[3] - d[1], h: d[4] - d[2] })
    })
    return out
  }, [tracks, track])
  const frames = [...path.keys()]
  const first = frames.length ? Math.min(...frames) : 0
  const last = frames.length ? Math.max(...frames) : tracks.n - 1

  useLayoutEffect(() => {
    const el = frameEl.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    const v = video.current
    if (!v) return
    let alive = true
    let handle = 0
    let smooth: Box | null = null
    const t0 = first / tracks.fps
    const t1 = (last + 1) / tracks.fps
    const start = () => { v.currentTime = t0; v.play().catch(() => {}) }
    const tick = (_: number, meta?: VideoFrameCallbackMetadata) => {
      if (!alive) return
      const t = meta ? meta.mediaTime : v.currentTime
      if (t >= t1 || t < t0 - 0.2) v.currentTime = t0
      const f = Math.min(tracks.n - 1, Math.round(t * tracks.fps))
      let target: Box | undefined = path.get(f)
      if (!target) for (let k = 1; k < 8 && !target; k++) target = path.get(f - k) ?? path.get(f + k)
      if (target) {
        // The camera framing is smoothed so it doesn't jitter; the box itself is drawn exactly.
        const a = smooth ? 0.45 : 1
        smooth = smooth
          ? { x: smooth.x + (target.x - smooth.x) * a, y: smooth.y + (target.y - smooth.y) * a, w: smooth.w + (target.w - smooth.w) * a, h: smooth.h + (target.h - smooth.h) * a }
          : target
        setView(smooth)
        setBox(target)
      }
      handle = 'requestVideoFrameCallback' in v ? v.requestVideoFrameCallback(tick) : requestAnimationFrame(() => tick(0))
    }
    if (v.readyState >= 1) start()
    else v.addEventListener('loadedmetadata', start, { once: true })
    handle = 'requestVideoFrameCallback' in v ? v.requestVideoFrameCallback(tick) : requestAnimationFrame(() => tick(0))
    return () => {
      alive = false
      if ('cancelVideoFrameCallback' in v) v.cancelVideoFrameCallback(handle)
      else cancelAnimationFrame(handle)
    }
  }, [src, path, first, last, tracks])

  // Zoom so the vehicle fills ~65% of the view, capped so tiny boxes don't turn into mush.
  const cv = view ?? { x: 0, y: 0, w: tracks.w, h: tracks.h }
  const b = box ?? cv
  const k = Math.min(0.65 * size.w / cv.w, 0.65 * size.h / cv.h, (size.w / tracks.w) * 6)
  const left = size.w / 2 - (cv.x + cv.w / 2) * k
  const top = size.h / 2 - (cv.y + cv.h / 2) * k

  return (
    <div ref={frameEl} style={{ position: 'relative', aspectRatio: '4 / 3', overflow: 'hidden', borderRadius: 8, background: '#000', border: '1px solid var(--amber)' }}>
      <video ref={video} src={src} muted playsInline preload="auto" style={{ position: 'absolute', left, top, width: tracks.w * k, height: tracks.h * k, maxWidth: 'none', transition: 'none' }} />
      {box && (
        <div style={{ position: 'absolute', left: left + b.x * k, top: top + b.y * k, width: b.w * k, height: b.h * k, border: '2px solid var(--amber)', borderRadius: 3, pointerEvents: 'none' }} />
      )}
      <span className="chip" style={{ position: 'absolute', left: 8, top: 8, color: 'var(--amber)' }}>FOLLOW · #{track}</span>
      <span className="chip" style={{ position: 'absolute', right: 8, top: 8 }}>{(k / (size.w / tracks.w)).toFixed(1)}× ZOOM</span>
    </div>
  )
}
