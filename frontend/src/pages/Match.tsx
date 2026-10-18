import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState, type ChangeEvent } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { MapView } from '../components/MapView'
import { PetModal } from '../components/PetModals'
import { Icon, Seg, Spinner, StatusBadge } from '../components/ui'
import { api, type MatchRun, type Pet } from '../lib/api'
import { useAuth } from '../lib/auth'
import { ago, fmtKm, img } from '../lib/format'
import { useAIInfo, usePet, usePets, useRequireAuth } from '../lib/queries'
import { useToast } from '../lib/toast'

const AREAS = [['ennasr', 'Ennasr, Ariana'], ['marsa', 'La Marsa, Tunis'], ['centre', 'Tunis centre'], ['sousse', 'Sousse'], ['sfax', 'Sfax']] as const
const SAMPLE_NAMES = ['Mishmish', 'Rex', 'Kahwa']
const areaFor = (gov: string) => (gov === 'Ariana' ? 'ennasr' : gov === 'Sousse' ? 'sousse' : gov === 'Sfax' ? 'sfax' : 'marsa')
const band = (s: number) => (s >= 80 ? ['Strong similarity', '#115850', '#D7ECE8'] : s >= 60 ? ['Moderate similarity', '#6B500D', '#F4E8CC'] : ['Weak similarity', '#68747D', '#EEE7DF'])
const zoomFor = (r: number) => (r <= 6 ? 13 : r <= 15 ? 12 : r <= 40 ? 11 : r <= 90 ? 10 : 8)

export default function Match() {
  const [sp, setSp] = useSearchParams()
  const { user } = useAuth()
  const toast = useToast()
  const qc = useQueryClient()
  const requireAuth = useRequireAuth()
  const petParam = sp.get('pet')
  const [petId, setPetId] = useState<number | null>(petParam ? Number(petParam) : null)
  const [photo, setPhoto] = useState<string | null>(null)
  const [species, setSpecies] = useState<'Cat' | 'Dog'>('Cat')
  const [status, setStatus] = useState<'found' | 'lost' | 'both'>('found')
  const [area, setArea] = useState('ennasr')
  const [radius, setRadius] = useState(20)
  const [threshold, setThreshold] = useState(50)
  const [selected, setSelected] = useState<number | null>(sp.get('selected') ? Number(sp.get('selected')) : null)
  const [dismissed, setDismissed] = useState<number[]>([])
  const [scanning, setScanning] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [contact, setContact] = useState<Pet | null>(null)
  const scanTimer = useRef<number | undefined>(undefined)

  const reports = usePets({ status: 'all', page_size: 100 })
  const samples = useMemo(() => SAMPLE_NAMES.map((n) => reports.data?.items.find((p) => p.name === n)).filter((p): p is Pet => !!p), [reports.data])
  const queryPet = usePet(petId ?? NaN)
  const qp = petId ? queryPet.data : undefined

  // Default to the first sample when nothing was requested.
  useEffect(() => {
    if (!petId && !photo && samples[0]) setPetId(samples[0].id)
  }, [samples, petId, photo])

  // Align the search with the query report: same species, opposite status, nearby area.
  useEffect(() => {
    if (!qp) return
    if (qp.species === 'Cat' || qp.species === 'Dog') setSpecies(qp.species)
    setStatus(qp.status === 'found' ? 'lost' : 'found')
    setArea(areaFor(qp.gov))
    setRadius(qp.species === 'Cat' ? 20 : 25)
  }, [qp?.id]) // eslint-disable-line react-hooks/exhaustive-deps

  const [settled, setSettled] = useState({ radius, threshold })
  useEffect(() => {
    const t = window.setTimeout(() => setSettled({ radius, threshold }), 250)
    return () => window.clearTimeout(t)
  }, [radius, threshold])

  const params = { pet_id: photo ? null : petId, photo, species, status, area, radius: settled.radius, threshold: settled.threshold }
  const ready = !!(photo || (petId && qp))
  const { data: aiInfo } = useAIInfo()
  const run = useQuery({
    queryKey: ['match', params],
    queryFn: () => api.post<MatchRun>('/api/ai/match', params),
    enabled: ready,
    placeholderData: (prev) => prev,
  })
  const results = (run.data?.results ?? []).filter((h) => !dismissed.includes(h.pet.id))
  const sel = results.find((h) => h.pet.id === selected) ?? results[0]
  const busy = scanning || (run.isFetching && !run.data)

  const runNow = async () => {
    window.clearTimeout(scanTimer.current)
    setScanning(true)
    const started = Date.now()
    const res = await run.refetch()
    scanTimer.current = window.setTimeout(() => {
      setScanning(false)
      const n = (res.data?.results ?? []).filter((h) => !dismissed.includes(h.pet.id)).length
      toast(n ? `${n} possible ${n === 1 ? 'match' : 'matches'} found` : 'No possible matches above your threshold')
    }, Math.max(0, 1600 - (Date.now() - started)))
  }
  useEffect(() => () => window.clearTimeout(scanTimer.current), [])

  const pickSample = (p: Pet) => {
    setPhoto(null)
    setPetId(p.id)
    setSelected(null)
    setDismissed([])
    setSp(new URLSearchParams({ pet: String(p.id) }), { replace: true })
  }
  const onFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    e.target.value = ''
    if (!f) return
    setUploading(true)
    try {
      const { urls } = await api.upload([f])
      setPhoto(urls[0])
      setPetId(null)
      setSelected(null)
      setDismissed([])
      setSp(new URLSearchParams(), { replace: true })
    } catch (err) {
      toast(err instanceof Error ? err.message : 'Upload failed.', true)
    } finally {
      setUploading(false)
    }
  }
  const dismiss = async (p: Pet) => {
    setDismissed((d) => [...d, p.id])
    if (qp?.is_mine) {
      try {
        await api.post(`/api/ai/match/${qp.id}/dismiss/${p.id}`)
        qc.invalidateQueries({ queryKey: ['dashboard'] })
      } catch {
        /* still hidden locally */
      }
    }
  }
  const openContact = (p: Pet) => {
    if (requireAuth('Sign in to contact the owner or finder.')) setContact(p)
  }
  const contactMsg = (p: Pet) => `Hi ${p.contact_name.split(' ')[0]}, PawConnect suggested your ${p.status === 'found' ? 'found' : 'lost'} report as a possible match for ${qp ? qp.name : 'my pet'}. Could you check for `

  const qImg = photo ?? (qp ? img(qp.photos[0], 1100, 1000) : '')
  const sig = run.data?.signature ?? []
  const model = run.data?.model ?? aiInfo?.provider
  const isDemo = !model || model === 'demo'
  const areaInfo = run.data ?? { area_lat: 36.8625, area_lng: 10.1647, area_label: AREAS.find((a) => a[0] === area)?.[1] ?? '' }

  return (
    <main>
      <section style={{ background: 'var(--night)', color: '#EEF3F1', backgroundImage: 'linear-gradient(rgba(127,240,222,.045) 1px,transparent 1px),linear-gradient(90deg,rgba(127,240,222,.045) 1px,transparent 1px)', backgroundSize: '32px 32px' }}>
        <div className="wrap" style={{ paddingTop: 44, paddingBottom: 48 }}>
          <div className="wrap-row between" style={{ alignItems: 'flex-end', gap: 20, marginBottom: 28 }}>
            <div style={{ maxWidth: 680 }}>
              <div className="row mono" style={{ gap: 8, fontSize: 12, color: 'var(--mint)', marginBottom: 12 }}><span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--mint)' }} />M2 · SIAMESE CNN · {isDemo ? 'DEMO MODE' : model.toUpperCase()}</div>
              <h1 style={{ margin: '0 0 10px', fontSize: 'clamp(34px,4vw,56px)', fontWeight: 800, letterSpacing: '-.04em', lineHeight: 1 }}>AI Lost Pet Matching</h1>
              <p style={{ fontSize: 16, lineHeight: 1.55, color: '#AEBBB8' }}>Upload a photo or pick a report, and we compare it with open reports nearby. Results are ranked by visual similarity.</p>
            </div>
            <div className="row" style={{ gap: 10, padding: '10px 14px', borderRadius: 12, border: '1px solid rgba(244,169,136,.35)', background: 'rgba(244,169,136,.08)', fontSize: 13, color: '#F4C3AC', maxWidth: 420, lineHeight: 1.45 }}><Icon name="info" size={20} />Suggested possible matches, not verified identities.</div>
          </div>

          <div className="wrap-row" style={{ gap: 20, alignItems: 'stretch' }}>
            <div className="col" style={{ flex: '1.4 1 460px', minWidth: 0, gap: 12 }}>
              <div style={{ position: 'relative', borderRadius: 22, overflow: 'hidden', background: '#0C1316', aspectRatio: '16/11', border: '1px solid rgba(127,240,222,.18)' }}>
                {qImg && <img src={qImg} alt="Query photo" style={{ width: '100%', height: '100%', objectFit: 'cover', opacity: 0.94 }} />}
                {busy && <div style={{ position: 'absolute', left: 0, right: 0, height: 2, background: 'var(--mint)', boxShadow: '0 0 22px 6px rgba(95,224,207,.55)', animation: 'pcScan 1.8s ease-in-out infinite', zIndex: 3 }} />}
                <div style={{ position: 'absolute', left: '20%', top: '12%', right: '20%', bottom: '10%', border: '1.5px solid rgba(127,240,222,.85)', borderRadius: 6, boxShadow: '0 0 0 9999px rgba(12,19,22,.28)' }}>
                  <span className="mono nowrap" style={{ position: 'absolute', left: -2, top: -24, padding: '3px 8px', borderRadius: 5, background: 'var(--mint)', color: '#0C1316', fontSize: 11, fontWeight: 600 }}>pet region · M6 mask</span>
                  {([{ left: -6, top: -6 }, { right: -6, top: -6 }, { left: -6, bottom: -6 }, { right: -6, bottom: -6 }] as const).map((pos, i) => (
                    <span key={i} style={{ position: 'absolute', width: 18, height: 18, ...pos, [`border${'left' in pos ? 'Left' : 'Right'}`]: '3px solid #7FF0DE', [`border${'top' in pos ? 'Top' : 'Bottom'}`]: '3px solid #7FF0DE' }} />
                  ))}
                </div>
                <div className="row between" style={{ position: 'absolute', left: 14, bottom: 14, right: 14, alignItems: 'flex-end', gap: 12 }}>
                  <div style={{ padding: '10px 12px', borderRadius: 12, background: 'rgba(12,19,22,.78)', backdropFilter: 'blur(8px)' }}>
                    <div style={{ fontSize: 15, fontWeight: 800 }}>{qp ? `${qp.name} · ${qp.status === 'lost' ? 'Lost' : qp.status === 'found' ? 'Found' : 'Adoption'} ${qp.city}` : photo ? 'Your uploaded photo' : 'Choose a photo'}</div>
                    <div style={{ fontSize: 12, color: '#AEBBB8' }}>{qp ? `${qp.breed} · ${qp.color}` : photo ? 'Uploaded just now' : 'Upload or pick a sample'}</div>
                  </div>
                  <label className="row" style={{ gap: 6, height: 40, padding: '0 14px', borderRadius: 10, background: '#fff', color: 'var(--night)', fontSize: 13, fontWeight: 700, cursor: 'pointer', flexShrink: 0 }}>
                    <input type="file" accept="image/*" onChange={onFile} className="sr-only" disabled={uploading} />
                    {uploading ? <Spinner variant="dark" /> : <Icon name="upload" size={18} />}Upload photo
                  </label>
                </div>
              </div>
              <div className="row" style={{ gap: 14, padding: '12px 14px', borderRadius: 16, background: 'rgba(255,255,255,.04)', border: '1px solid rgba(255,255,255,.08)' }}>
                <div className="mono nowrap" style={{ fontSize: 11, color: 'var(--mint)' }}>VISUAL SIGNATURE</div>
                <div className="row grow" style={{ height: 30, alignItems: 'flex-end', gap: 2 }}>
                  {(sig.length ? sig : Array(48).fill(0.3)).map((v, i) => (
                    <span key={i} style={{ flex: 1, height: `${18 + v * 82}%`, background: v > 0.7 ? '#7FF0DE' : v > 0.4 ? '#4FB3A5' : '#2E6E67', borderRadius: 1, opacity: sig.length ? 1 : 0.3 }} />
                  ))}
                </div>
                <div className="mono nowrap" style={{ fontSize: 11, color: '#AEBBB8' }}>{run.data ? `${run.data.signature_dim}-d` : '—'}</div>
              </div>
            </div>

            <div className="col" style={{ flex: '1 1 320px', minWidth: 0, gap: 18, padding: 22, borderRadius: 22, background: 'rgba(255,255,255,.04)', border: '1px solid rgba(255,255,255,.08)' }}>
              <div>
                <div style={{ fontSize: 12, fontWeight: 700, color: '#AEBBB8', marginBottom: 10 }}>Try a sample photo</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,minmax(0,1fr))', gap: 8 }}>
                  {samples.map((s) => {
                    const on = !photo && petId === s.id
                    return (
                      <button key={s.id} onClick={() => pickSample(s)} aria-pressed={on} className="col" style={{ alignItems: 'center', gap: 6, padding: '10px 6px', borderRadius: 14, border: `1.5px solid ${on ? '#7FF0DE' : 'rgba(255,255,255,.12)'}`, background: on ? 'rgba(127,240,222,.08)' : 'transparent', color: '#EEF3F1', minWidth: 0 }}>
                        <img src={img(s.photos[0], 120, 120)} alt="" style={{ width: 44, height: 44, borderRadius: '50%', objectFit: 'cover' }} />
                        <span style={{ fontSize: 13, fontWeight: 700 }}>{s.name}</span><span style={{ fontSize: 11, color: '#AEBBB8' }}>{s.status === 'lost' ? 'Lost' : 'Found'} {s.species.toLowerCase()}</span>
                      </button>
                    )
                  })}
                </div>
              </div>
              <div>
                <div style={{ fontSize: 12, fontWeight: 700, color: '#AEBBB8', marginBottom: 8 }}>Species</div>
                <Seg className="night" value={species} onChange={(v) => { setSpecies(v); setSelected(null) }} options={[{ value: 'Cat', label: 'Cats' }, { value: 'Dog', label: 'Dogs' }]} />
              </div>
              <div>
                <div style={{ fontSize: 12, fontWeight: 700, color: '#AEBBB8', marginBottom: 8 }}>Search in</div>
                <Seg className="night" value={status} onChange={(v) => { setStatus(v); setSelected(null) }} options={[{ value: 'found', label: 'Found reports' }, { value: 'lost', label: 'Lost reports' }, { value: 'both', label: 'Both' }]} />
              </div>
              <label className="field">
                <span style={{ display: 'block', fontSize: 12, fontWeight: 700, color: '#AEBBB8', marginBottom: 8 }}>Around</span>
                <select value={area} onChange={(e) => { setArea(e.target.value); setSelected(null) }} style={{ width: '100%', height: 44, padding: '0 12px', borderRadius: 12, border: '1px solid rgba(255,255,255,.14)', background: '#1C272C', color: '#EEF3F1', fontSize: 14, fontWeight: 600, outline: 'none' }}>
                  {AREAS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                </select>
              </label>
              <div>
                <div className="row between" style={{ fontSize: 12, fontWeight: 700, color: '#AEBBB8', marginBottom: 6 }}><span>Search radius</span><span className="mono" style={{ color: 'var(--mint)' }}>{radius} km</span></div>
                <input type="range" min={1} max={300} value={radius} onChange={(e) => setRadius(+e.target.value)} aria-label="Search radius" style={{ width: '100%', accentColor: '#7FF0DE' }} />
              </div>
              <div>
                <div className="row between" style={{ fontSize: 12, fontWeight: 700, color: '#AEBBB8', marginBottom: 6 }}><span>Minimum similarity</span><span className="mono" style={{ color: 'var(--mint)' }}>{threshold}%</span></div>
                <input type="range" min={0} max={95} step={5} value={threshold} onChange={(e) => setThreshold(+e.target.value)} aria-label="Minimum similarity" style={{ width: '100%', accentColor: '#7FF0DE' }} />
              </div>
              <button onClick={runNow} disabled={!ready || busy} className="row" style={{ marginTop: 'auto', height: 52, borderRadius: 14, border: 0, background: 'var(--mint)', color: '#0C1316', fontSize: 15, fontWeight: 800, justifyContent: 'center', gap: 10 }}>
                {busy ? <><Spinner variant="teal" />Comparing with {run.data?.pool ?? '…'} reports…</> : <><Icon name="image_search" size={20} />Run matching</>}
              </button>
            </div>
          </div>
        </div>
      </section>

      <section className="wrap" style={{ paddingTop: 32, paddingBottom: 88 }}>
        <div className="note" style={{ marginBottom: 28, alignItems: 'flex-start' }}>
          <Icon name="verified_user" size={22} />
          <div><strong>These are possible matches, not confirmed identities.</strong> Similarity is calculated from the photo alone. Before arranging a handover, compare collars, markings and size, and ask for vet records or a microchip scan. {isDemo ? 'All scores shown are demo outputs while the model is being trained.' : `Scores are computed by the M2 model (${model}), which was trained on dog photos only, so scores for cats are less reliable.`}</div>
        </div>
        <div className="wrap-row" style={{ gap: 24, alignItems: 'flex-start' }}>
          <div className="col" style={{ flex: '1.3 1 480px', minWidth: 0, gap: 14 }}>
            <div className="wrap-row between" style={{ alignItems: 'baseline', gap: 12 }}>
              <h2 style={{ fontSize: 24, fontWeight: 800, letterSpacing: '-.02em' }}>Ranked results</h2>
              <span style={{ fontSize: 14, color: 'var(--muted)' }}>{results.length} possible {results.length === 1 ? 'match' : 'matches'} at {settled.threshold}% similarity or higher, within {settled.radius} km of {areaInfo.area_label}</span>
            </div>
            {run.isError && <div className="alert"><Icon name="error" />{(run.error as Error).message}</div>}
            <div className="col" style={{ gap: 12, opacity: busy ? 0.35 : 1, transition: 'opacity .3s' }}>
              {results.map((h, i) => {
                const p = h.pet
                const on = sel?.pet.id === p.id
                const [bl, bf, bb] = band(h.score)
                return (
                  <div key={p.id} onClick={() => setSelected(p.id)} className="wrap-row" style={{ gap: 16, alignItems: 'center', padding: 12, background: '#fff', border: `1.5px solid ${on ? '#176B63' : '#F0E4D6'}`, boxShadow: on ? '0 0 0 3px rgba(23,107,99,.12)' : 'none', borderRadius: 20, cursor: 'pointer', transition: 'border-color .2s' }}>
                    <div style={{ position: 'relative', flexShrink: 0 }}>
                      <img src={img(p.photos[0], 300, 300)} alt={p.name} style={{ width: 120, height: 120, borderRadius: 14, objectFit: 'cover' }} />
                      <span className="mono" style={{ position: 'absolute', left: 8, top: 8, padding: '2px 7px', borderRadius: 6, background: 'var(--ink)', color: '#fff', fontSize: 12, fontWeight: 600 }}>#{i + 1}</span>
                    </div>
                    <div className="col" style={{ flex: '1 1 200px', minWidth: 0, gap: 6 }}>
                      <div className="wrap-row" style={{ alignItems: 'center', gap: 8 }}><span style={{ fontSize: 18, fontWeight: 800 }}>{p.name}</span><StatusBadge status={p.status} small /></div>
                      <div style={{ fontSize: 13.5, color: 'var(--body)' }}>{p.breed} · {p.color}</div>
                      <div style={{ fontSize: 13, color: 'var(--muted)' }}>{p.city} · {ago(p.days_ago)} · {fmtKm(h.distance_from_area)} from {areaInfo.area_label.split(',')[0]}</div>
                      <div className="wrap-row" style={{ gap: 8, marginTop: 4 }}>
                        <Link to={`/pets/${p.id}`} onClick={(e) => e.stopPropagation()} className="btn btn-outline btn-sm">View details</Link>
                        <button className="btn btn-primary btn-sm" onClick={(e) => { e.stopPropagation(); openContact(p) }}>{p.status === 'found' ? 'Contact finder' : 'Contact owner'}</button>
                        <button className="btn btn-ghost btn-sm" style={{ fontWeight: 600 }} onClick={(e) => { e.stopPropagation(); dismiss(p) }}>Not a match</button>
                      </div>
                    </div>
                    <div className="col" style={{ alignItems: 'center', gap: 6, padding: '0 8px', flexShrink: 0 }}>
                      <div className="ring" style={{ width: 76, height: 76, background: `conic-gradient(#176B63 ${Math.round(h.score * 3.6)}deg, #E6EEEC 0)` }}><div style={{ width: 62, height: 62, fontSize: 17 }}>{h.score}%</div></div>
                      <span className="nowrap" style={{ padding: '3px 8px', borderRadius: 999, background: bb, color: bf, fontSize: 11, fontWeight: 700 }}>{bl}</span>
                    </div>
                  </div>
                )
              })}
            </div>
            {run.data && !busy && results.length === 0 && (
              <div className="empty" style={{ padding: '44px 24px', borderRadius: 22, gap: 10 }}>
                <Icon name="travel_explore" size={40} style={{ color: 'var(--rust)' }} />
                <div style={{ fontSize: 19, fontWeight: 800 }}>No possible matches with these settings</div>
                <p style={{ maxWidth: 420, fontSize: 14, lineHeight: 1.55, color: 'var(--muted)' }}>Lower the minimum similarity, widen the radius, or switch between found and lost reports. {qp?.is_mine ? 'We’ll keep checking new reports for you.' : ''}</p>
              </div>
            )}
          </div>

          <div className="col" style={{ flex: '1 1 360px', minWidth: 0, gap: 16, position: 'sticky', top: 96 }}>
            <MapView height={320} center={[areaInfo.area_lat, areaInfo.area_lng]} zoom={zoomFor(settled.radius)} radiusKm={settled.radius} centerDot
              pins={results.map((h, i) => ({ kind: 'number' as const, id: h.pet.id, lat: h.pet.lat, lng: h.pet.lng, label: String(i + 1), bg: h.pet.id === sel?.pet.id ? '#202A32' : '#176B63', size: h.pet.id === sel?.pet.id ? 34 : 28, z: h.pet.id === sel?.pet.id ? 5 : 1, onClick: () => setSelected(h.pet.id) }))}>
              <div className="map-overlay map-label" style={{ fontSize: 12, left: 54 }}><span style={{ width: 10, height: 10, borderRadius: '50%', background: 'var(--orange)' }} />{areaInfo.area_label} · {settled.radius} km</div>
            </MapView>
            {sel && !busy && (
              <div className="panel" style={{ padding: 18, borderRadius: 20 }}>
                <div className="row between" style={{ marginBottom: 12 }}><span style={{ fontSize: 15, fontWeight: 800 }}>Side-by-side</span><span className="mono" style={{ fontSize: 13, fontWeight: 600, color: 'var(--teal)' }}>{sel.score}% similar</span></div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  <div><div style={{ aspectRatio: '1', borderRadius: 14, overflow: 'hidden', background: 'var(--sand)' }}>{qImg && <img src={qImg} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />}</div><div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 6, textAlign: 'center' }}>{qp ? qp.name : 'Your photo'}</div></div>
                  <div><div style={{ aspectRatio: '1', borderRadius: 14, overflow: 'hidden', background: 'var(--sand)' }}><img src={img(sel.pet.photos[0], 400, 400)} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} /></div><div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 6, textAlign: 'center' }}>{sel.pet.name} · {sel.pet.status === 'found' ? 'Found' : 'Lost'}</div></div>
                </div>
                <div style={{ marginTop: 14, fontSize: 13, fontWeight: 800 }}>Check these yourself</div>
                <div className="col" style={{ gap: 6, marginTop: 8, fontSize: 13, color: 'var(--body)' }}>
                  {['Collar, tag or harness', 'Unique markings, scars or patches', 'Size, ear and tail shape', 'Microchip or vet records'].map((t) => (
                    <div key={t} className="row" style={{ gap: 8 }}><Icon name="check_box_outline_blank" size={17} style={{ color: 'var(--teal)' }} />{t}</div>
                  ))}
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 14 }}>
                  <Link to={`/pets/${sel.pet.id}`} className="btn btn-outline" style={{ height: 42, fontSize: 13 }}>View details</Link>
                  <button className="btn btn-primary" style={{ height: 42, fontSize: 13 }} onClick={() => openContact(sel.pet)}>{sel.pet.status === 'found' ? 'Contact finder' : 'Contact owner'}</button>
                </div>
              </div>
            )}
            {!user && <div style={{ fontSize: 13, color: 'var(--muted)' }}><Link to="/report?kind=lost">Publish a report</Link> to get alerts automatically when a similar animal is reported.</div>}
          </div>
        </div>
      </section>
      {contact && <PetModal kind="contact" pet={contact} message={contactMsg(contact)} onClose={() => setContact(null)} />}
    </main>
  )
}
