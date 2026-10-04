import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import type { Det, Tracks } from '../data'

interface Props {
  src: string
  poster?: string
  tracks?: Tracks | null
  /** 0..1. When set, the video follows scroll instead of playing. */
  progress?: number
  /** Track to draw in amber; every other box is dimmed by `othersOpacity`. */
  highlight?: number
  othersOpacity?: number
  /** Per-box reveal 0..1 (index = order of first appearance); omit to show all boxes. */
  reveal?: (order: number) => number
  onFrame?: (frame: number, dets: Det[]) => void
  onPick?: (track: number) => void
  /** Seconds to jump to once the clip loads (e.g. when opening a sighting). */
  startAt?: number
  /** Show a fullscreen button; boxes stay aligned because the overlay goes fullscreen with the video. */
  fullscreen?: boolean
  /** Custom box label per track (defaults to detector class + confidence). */
  labelFor?: (track: number) => string | undefined
  className?: string
  children?: React.ReactNode
}

/** Plays (or scroll-scrubs) a clip and overlays the detector's per-frame tracks. */
export default function TrackedVideo({ src, poster, tracks, progress, highlight, othersOpacity = 1, reveal, onFrame, onPick, startAt, fullscreen, labelFor, className, children }: Props) {
  const outer = useRef<HTMLDivElement>(null)
  const video = useRef<HTMLVideoElement>(null)
  const [frame, setFrame] = useState(0)
  const wanted = useRef(0)
  const scrub = progress !== undefined

  // Playback mode: boxes are painted on a canvas inside the video-frame callback, together with the
  // frame itself. Going through React state here adds a render behind the video and the boxes trail.
  const canvas = useRef<HTMLCanvasElement>(null)
  const shown = useRef<{ f: number; dets: Det[] }>({ f: -1, dets: [] })
  const latest = useRef({ highlight, othersOpacity, labelFor, onFrame })
  latest.current = { highlight, othersOpacity, labelFor, onFrame }

  const paint = () => {
    const c = canvas.current
    if (!c || !tracks) return
    const ctx = c.getContext('2d')
    if (!ctx) return
    const { highlight: hl, othersOpacity: oo, labelFor: lf } = latest.current
    const sx = c.width / tracks.w
    const sy = c.height / tracks.h
    const dpr = window.devicePixelRatio || 1
    ctx.clearRect(0, 0, c.width, c.height)
    ctx.font = `600 ${11 * dpr}px "JetBrains Mono", monospace`
    ctx.textBaseline = 'top'
    const ordered = [...shown.current.dets].sort((a, b) => Number(a[0] === hl) - Number(b[0] === hl))
    for (const [tid, x1, y1, x2, y2, conf, cls] of ordered) {
      const target = tid === hl
      ctx.globalAlpha = target ? 1 : oo
      const colour = target ? '#fbbf24' : '#5eead4'
      const x = x1 * sx, y = y1 * sy, w = (x2 - x1) * sx, h = (y2 - y1) * sy
      ctx.fillStyle = target ? 'rgba(251,191,36,.10)' : 'rgba(94,234,212,.07)'
      ctx.fillRect(x, y, w, h)
      ctx.lineWidth = (target ? 2.5 : 1.5) * dpr
      ctx.strokeStyle = colour
      ctx.strokeRect(x, y, w, h)
      const text = `#${tid} ${lf?.(tid) ?? `${tracks.names[String(cls)] ?? 'vehicle'} ${conf.toFixed(2)}`}`
      const tw = ctx.measureText(text).width + 8 * dpr
      const th = 15 * dpr
      const ty = y - th - 2 * dpr < 0 ? y + 2 * dpr : y - th - 2 * dpr
      ctx.fillStyle = colour
      ctx.fillRect(x, ty, tw, th)
      ctx.fillStyle = target ? '#1f1300' : '#042f2c'
      ctx.fillText(text, x + 4 * dpr, ty + 2 * dpr)
    }
    ctx.globalAlpha = 1
  }

  useEffect(() => {
    const v = video.current
    if (!v || scrub || !tracks) return
    let handle = 0
    let alive = true
    const schedule = () => { handle = 'requestVideoFrameCallback' in v ? v.requestVideoFrameCallback(tick) : requestAnimationFrame(() => tick(0)) }
    const tick = (_: number, meta?: VideoFrameCallbackMetadata) => {
      if (!alive) return
      const t = meta ? meta.mediaTime : v.currentTime
      const f = Math.min(tracks.n - 1, Math.round(t * tracks.fps)) // round: mediaTime k/fps can land just below k
      if (f !== shown.current.f) {
        shown.current = { f, dets: tracks.frames[f] ?? [] }
        paint()
        latest.current.onFrame?.(f, shown.current.dets)
      }
      schedule()
    }
    schedule()
    return () => {
      alive = false
      if ('cancelVideoFrameCallback' in v) v.cancelVideoFrameCallback(handle)
      else cancelAnimationFrame(handle)
    }
  }, [scrub, tracks, src]) // eslint-disable-line react-hooks/exhaustive-deps

  // Keep the canvas at device resolution and repaint when the selection changes.
  useLayoutEffect(() => {
    const c = canvas.current
    if (!c || scrub) return
    const fit = () => {
      const dpr = window.devicePixelRatio || 1
      c.width = Math.round(c.clientWidth * dpr)
      c.height = Math.round(c.clientHeight * dpr)
      paint()
    }
    fit()
    const ro = new ResizeObserver(fit)
    ro.observe(c)
    return () => ro.disconnect()
  }, [scrub, tracks]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (!scrub) paint() }, [highlight, othersOpacity, labelFor]) // eslint-disable-line react-hooks/exhaustive-deps

  const pickAt = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onPick || !tracks) return
    const r = e.currentTarget.getBoundingClientRect()
    const x = ((e.clientX - r.left) / r.width) * tracks.w
    const y = ((e.clientY - r.top) / r.height) * tracks.h
    const hits = shown.current.dets.filter((d) => x >= d[1] && x <= d[3] && y >= d[2] && y <= d[4])
    hits.sort((a, b) => (a[3] - a[1]) * (a[4] - a[2]) - (b[3] - b[1]) * (b[4] - b[2])) // smallest box wins when boxes overlap
    if (hits[0]) onPick(hits[0][0])
  }

  // Scrub mode: scroll position drives the clip.
  useEffect(() => {
    const v = video.current
    if (!v || !scrub || !tracks) return
    const f = Math.min(tracks.n - 1, Math.max(0, Math.round(progress! * (tracks.n - 1))))
    setFrame(f)
    wanted.current = f / tracks.fps
    if (v.readyState >= 1) v.currentTime = wanted.current
  }, [progress, scrub, tracks])

  const dets = tracks?.frames[frame] ?? []
  useEffect(() => { if (scrub) onFrame?.(frame, dets) }, [frame]) // eslint-disable-line react-hooks/exhaustive-deps

  // Order tracks by first appearance so reveal() can stagger them.
  const order = useRef(new Map<number, number>())
  useEffect(() => {
    order.current = new Map()
    tracks?.frames.forEach((f) => f.forEach((d) => { if (!order.current.has(d[0])) order.current.set(d[0], order.current.size) }))
  }, [tracks])

  const ar = tracks ? `${tracks.w} / ${tracks.h}` : '16 / 9'
  const inner = (
    <div className={fullscreen ? 'tv-inner' : className} style={{ position: 'relative', aspectRatio: ar, overflow: 'hidden', background: '#0a0f16', ['--ar' as string]: tracks ? tracks.w / tracks.h : 16 / 9 }}>
      <video
        ref={video}
        src={src}
        poster={poster}
        muted
        playsInline
        loop={!scrub}
        autoPlay={!scrub}
        preload="auto"
        onLoadedData={(e) => {
          if (scrub) e.currentTarget.currentTime = wanted.current + 0.001
          else if (startAt) e.currentTarget.currentTime = startAt
        }}
        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
      />
      {tracks && !scrub && (
        <canvas ref={canvas} onClick={pickAt} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', cursor: onPick ? 'pointer' : undefined }} />
      )}
      {tracks && scrub && dets.map((d) => {
        const [tid, x1, y1, x2, y2, conf, cls] = d
        const isTarget = highlight === tid
        const r = reveal ? reveal(order.current.get(tid) ?? 0) : 1
        const o = isTarget ? 1 : r * othersOpacity
        if (o <= 0.01) return null
        return (
          <div
            key={tid}
            className={`bbox${isTarget ? ' target' : ''}`}
            onClick={onPick ? () => onPick(tid) : undefined}
            style={{
              left: `${(x1 / tracks.w) * 100}%`, top: `${(y1 / tracks.h) * 100}%`,
              width: `${((x2 - x1) / tracks.w) * 100}%`, height: `${((y2 - y1) / tracks.h) * 100}%`,
              opacity: o, transform: `scale(${1.25 - 0.25 * Math.min(1, r)})`, cursor: onPick ? 'pointer' : undefined,
            }}
          >
            <span>#{tid} {labelFor?.(tid) ?? `${tracks.names[String(cls)] ?? 'vehicle'} ${conf.toFixed(2)}`}</span>
          </div>
        )
      })}
      {children}
    </div>
  )
  if (!fullscreen) return inner
  const toggle = () => (document.fullscreenElement ? document.exitFullscreen() : outer.current?.requestFullscreen())
  return (
    <div ref={outer} className={`tv-outer ${className ?? ''}`} style={{ position: 'relative' }}>
      {inner}
      <button className="tv-fs" onClick={toggle} aria-label="Toggle fullscreen" title="Fullscreen">⛶</button>
    </div>
  )
}
