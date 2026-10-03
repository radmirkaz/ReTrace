import { useEffect, useState } from 'react'
import Nav from './components/Nav'
import Nerd from './components/Nerd'
import Landing from './landing/Landing'
import Live from './live/Live'

const route = () => (location.hash.startsWith('#/live') ? 'live' : 'landing')

export default function App() {
  const [page, setPage] = useState<'landing' | 'live'>(route)
  const [nerd, setNerd] = useState(false)

  useEffect(() => {
    const on = () => { if (location.hash.startsWith('#/') || location.hash === '') { setPage(route()); window.scrollTo(0, 0) } }
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])

  // Demo shortcut: press "~" to open the technical dossier.
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setNerd(false)
      else if ((e.key === '~' || e.key === '`') && !(e.target instanceof HTMLInputElement)) setNerd((v) => !v)
    }
    window.addEventListener('keydown', on)
    return () => window.removeEventListener('keydown', on)
  }, [])

  return (
    <>
      <Nav page={page} nerd={nerd} onNerd={() => setNerd((v) => !v)} />
      {page === 'live' ? <Live /> : <Landing />}
      <Nerd open={nerd} onClose={() => setNerd(false)} />
    </>
  )
}
