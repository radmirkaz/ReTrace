import { useMotionValueEvent, useScroll } from 'framer-motion'
import { useEffect, useRef, useState } from 'react'

/** 0..1 progress through a tall section whose child is `position: sticky`. */
export function usePinProgress<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start start', 'end end'] })
  const [p, setP] = useState(0)
  useMotionValueEvent(scrollYProgress, 'change', setP)
  return [ref, p] as const
}

/** 0..1 as an element scrolls into view (for count-ups). */
export function useRevealProgress<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start end', 'start 0.35'] })
  const [p, setP] = useState(0)
  useMotionValueEvent(scrollYProgress, 'change', setP)
  return [ref, p] as const
}

/** Ticks every `ms` while mounted — drives ambient loops like waveforms. */
export function useTicker(ms = 100) {
  const [t, setT] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setT((x) => x + 1), ms)
    return () => clearInterval(id)
  }, [ms])
  return t
}

export const clamp = (v: number, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v))
export const ease = (v: number) => { const k = clamp(v); return k * k * (3 - 2 * k) }
/** Maps p from [a, b] onto 0..1. */
export const span = (p: number, a: number, b: number) => clamp((p - a) / (b - a))
