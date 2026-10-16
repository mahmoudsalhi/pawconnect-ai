import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState, type ChangeEvent } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { MapView } from '../components/MapView'
import { BreedBars, FieldError, Icon, Seg, Spinner } from '../components/ui'
import { api, ApiError, type AnalyzeResult, type Pet } from '../lib/api'
import { useAuth } from '../lib/auth'
import { img } from '../lib/format'
import { useToast } from '../lib/toast'

type Kind = 'lost' | 'found'
interface Draft {
  step: number
  kind: Kind
  photos: string[]
  sample: boolean
  bg: boolean
  name: string
  species: 'Dog' | 'Cat' | 'Other'
  breed: string
  breedUsed: boolean
  sex: 'Male' | 'Female' | 'Unknown'
  age: string
  size: 'Small' | 'Medium' | 'Large'
  color: string
  chip: 'Yes' | 'No' | 'Unknown'
  desc: string
  accepted: string[]
  dismissed: string[]
  pin: { lat: number; lng: number } | null
  address: string
  date: string
  time: string
  cname: string
  phone: string
  email: string
  contact: 'Phone call' | 'WhatsApp' | 'In-app messages'
  showPhone: boolean
  consent: boolean
}

const DRAFT_KEY = 'pc_report_draft'
const today = () => new Date().toISOString().slice(0, 10)
const blank = (): Draft => ({
  step: 1, kind: 'lost', photos: [], sample: false, bg: false, name: '', species: 'Dog', breed: '', breedUsed: false, sex: 'Male',
  age: 'Adult (1–7 years)', size: 'Medium', color: '', chip: 'Unknown', desc: '', accepted: [], dismissed: [], pin: null, address: '',
  date: today(), time: '19:30', cname: '', phone: '', email: '', contact: 'Phone call', showPhone: false, consent: false,
})
const LABELS = ['Type', 'Photos', 'Details', 'Location', 'Contact', 'Review']
const SAMPLES: Record<Kind, string> = { lost: '1543466835-00a7907e9de1', found: '1561037404-61cd46aa615b' }
const CROPS = ['', '?crop=focalpoint&fp-x=.5&fp-y=.4&fp-z=1.7', '?crop=focalpoint&fp-x=.32&fp-y=.55&fp-z=1.4']
const MAP_CENTER: [number, number] = [36.845, 10.24]

function loadDraft(): Draft | null {
  try {
    const raw = localStorage.getItem(DRAFT_KEY)
    return raw ? { ...blank(), ...JSON.parse(raw) } : null
  } catch {
    return null
  }
}
function saveDraftLocal(d: Draft) {
  try {
    localStorage.setItem(DRAFT_KEY, JSON.stringify(d))
    return true
  } catch {
    return false
  }
}
function clearDraft() {
  try {
    localStorage.removeItem(DRAFT_KEY)
  } catch {
    /* ignore */
  }
}

function validate(r: Draft): Record<string, string> {
  const e: Record<string, string> = {}
  if (r.step === 2 && !r.photos.length) e.photos = 'Add at least one photo so the AI can analyse and match it.'
  if (r.step === 3) {
    if (r.kind === 'lost' && !r.name.trim()) e.name = 'Enter your pet’s name.'
    if (!r.color.trim()) e.color = 'Describe the main colour.'
    if (r.desc.trim().length < 20) e.desc = 'Add a few more details (at least 20 characters).'
  }
  if (r.step === 4) {
    if (!r.pin) e.pin = 'Click the map to mark the location.'
    if (!r.date) e.date = 'Choose a date.'
    else if (r.date > today()) e.date = 'The date can’t be in the future.'
  }
  if (r.step === 5) {
    if (r.cname.trim().length < 2) e.cname = 'Enter your name.'
    const d = r.phone.replace(/\D/g, '').replace(/^216/, '')
    if (d.length !== 8) e.phone = 'Enter an 8-digit Tunisian number, e.g. 22 418 903.'
    if (!/^\S+@\S+\.\S+$/.test(r.email)) e.email = 'Enter a valid email address.'
  }
  if (r.step === 6 && !r.consent) e.consent = 'Please confirm the information is accurate.'
  return e
}

// Maps API field names back to the wizard step that owns them.
const FIELD_STEP: Record<string, [number, string]> = {
  photos: [2, 'photos'], name: [3, 'name'], color: [3, 'color'], description: [3, 'desc'], lat: [4, 'pin'], lng: [4, 'pin'],
  event_date: [4, 'date'], contact_name: [5, 'cname'], contact_phone: [5, 'phone'], contact_email: [5, 'email'],
}

export default function Report() {
  const { user } = useAuth()
  const navigate = useNavigate()
  const toast = useToast()
  const qc = useQueryClient()
  const [sp] = useSearchParams()
  const [r, setR] = useState<Draft>(() => {
    const d = loadDraft()
    if (d) return d
    const kind = sp.get('kind')
    return kind === 'lost' || kind === 'found' ? { ...blank(), kind, step: 2 } : blank()
  })
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [uploading, setUploading] = useState(false)
  const [ai, setAi] = useState<AnalyzeResult | null>(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [tags, setTags] = useState<{ label: string; confidence: number }[]>([])
  const [done, setDone] = useState<{ pet: Pet; pool: number; match_count: number } | null>(null)
  const lost = r.kind === 'lost'

  const set = (patch: Partial<Draft>) => {
    setR((s) => ({ ...s, ...patch }))
    setErrors((e) => {
      const n = { ...e }
      for (const k of Object.keys(patch)) delete n[k]
      return n
    })
  }
  const top = () => window.scrollTo(0, 0)

  // Prefill contact details from the account.
  useEffect(() => {
    if (user) setR((s) => ({ ...s, cname: s.cname || user.name, email: s.email || user.email, phone: s.phone || user.phone }))
  }, [user])

  // Photo analysis (M1 breed, M5 quality, M6 background) whenever the photo set changes.
  const photoKey = r.photos.join('|')
  useEffect(() => {
    if (!r.photos.length) {
      setAi(null)
      return
    }
    let alive = true
    setAnalyzing(true)
    api.post<AnalyzeResult>('/api/ai/analyze', { photos: r.photos, species: r.species === 'Other' ? 'Dog' : r.species })
      .then((a) => alive && setAi(a))
      .catch(() => alive && setAi(null))
      .finally(() => alive && setAnalyzing(false))
    return () => {
      alive = false
    }
  }, [photoKey, r.species]) // eslint-disable-line react-hooks/exhaustive-deps

  // Description tagging (M4), debounced.
  useEffect(() => {
    const t = window.setTimeout(() => {
      if (r.desc.trim().length < 4) return setTags([])
      api.post<{ label: string; confidence: number }[]>('/api/ai/tags', { text: r.desc }).then(setTags).catch(() => setTags([]))
    }, 400)
    return () => window.clearTimeout(t)
  }, [r.desc])

  const addFiles = async (list: File[]) => {
    const files = list.filter((f) => f.type.startsWith('image/')).slice(0, 6 - (r.sample ? 0 : r.photos.length))
    if (!files.length) return
    const tooBig = files.find((f) => f.size > 10 * 1024 * 1024)
    if (tooBig) return setErrors({ photos: `${tooBig.name} is larger than 10 MB.` })
    setUploading(true)
    try {
      const { urls } = await api.upload(files)
      set({ photos: [...(r.sample ? [] : r.photos), ...urls].slice(0, 6), sample: false })
    } catch (err) {
      setErrors({ photos: err instanceof Error ? err.message : 'Upload failed.' })
    } finally {
      setUploading(false)
    }
  }
  const onFiles = (e: ChangeEvent<HTMLInputElement>) => {
    const files = [...(e.target.files ?? [])]
    e.target.value = ''
    addFiles(files)
  }
  const useSample = () => set({ photos: CROPS.map((c) => `https://images.unsplash.com/photo-${SAMPLES[r.kind]}${c}`), sample: true, species: 'Dog' })

  const submit = useMutation({
    mutationFn: () => {
      const accepted = r.accepted.filter((l) => tags.some((t) => t.label === l))
      return api.post<{ pet: Pet; pool: number; match_count: number }>('/api/pets', {
        status: r.kind, name: r.name, species: r.species, breed: r.breed, age_label: r.age, sex: r.sex, size: r.size, color: r.color,
        chip: r.chip, description: r.desc, tags: accepted, photos: r.photos, background_removed: r.bg, lat: r.pin!.lat, lng: r.pin!.lng,
        address: r.address, event_date: r.date, event_time: r.time, contact_name: r.cname, contact_phone: r.phone,
        contact_email: r.email, contact_pref: r.contact, show_phone: r.showPhone,
      })
    },
    onSuccess: (res) => {
      clearDraft()
      setDone(res)
      qc.invalidateQueries()
      top()
    },
    onError: (e: ApiError) => {
      const first = Object.keys(e.errors).find((k) => FIELD_STEP[k])
      if (first) {
        const [step, field] = FIELD_STEP[first]
        set({ step })
        setErrors({ [field]: e.errors[first] })
      } else setErrors({ form: e.message })
    },
  })

  const next = () => {
    const e = validate(r)
    if (Object.keys(e).length) return setErrors(e)
    if (r.step < 6) {
      set({ step: r.step + 1 })
      return top()
    }
    if (!user) {
      saveDraftLocal(r)
      navigate('/login', { state: { from: '/report', note: 'Sign in to publish your report. Your draft is saved.' } })
      return
    }
    submit.mutate()
  }
  const saveDraft = () => {
    const d = new Date()
    if (saveDraftLocal(r)) toast(`Draft saved at ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`)
    else toast('Couldn’t save the draft in this browser.', true)
  }
  const restart = () => {
    clearDraft()
    setDone(null)
    setR(blank())
    setErrors({})
    top()
  }
  const useMyLoc = () => {
    if (!navigator.geolocation) return toast('Location isn’t available in this browser.', true)
    navigator.geolocation.getCurrentPosition(
      (p) => set({ pin: { lat: p.coords.latitude, lng: p.coords.longitude } }),
      () => toast('Couldn’t get your location. Click the map instead.', true),
      { timeout: 8000 },
    )
  }

  const breeds = ai?.breeds ?? []
  const topBreed = breeds[0]?.[0]
  const breedUsed = r.breedUsed && r.breed === topBreed
  const quality = ai?.quality ?? []
  const avg = quality.length ? Math.round(quality.reduce((s, q) => s + q.score, 0) / quality.length) : 0
  const best = quality.reduce((b, q, i) => (q.score > quality[b].score ? i : b), 0)
  const visibleTags = tags.filter((t) => !r.dismissed.includes(t.label))
  const bgMask = r.bg ? 'radial-gradient(ellipse 40% 47% at 50% 50%,#000 70%,transparent 72%)' : 'none'
  const cover = ai?.cutout && r.bg ? ai.cutout : r.photos[0]

  if (done) {
    const n = done.match_count
    return (
      <main className="wrap" style={{ paddingTop: 40, paddingBottom: 88 }}>
        <div className="card col" style={{ maxWidth: 720, margin: '24px auto 0', padding: 'clamp(28px,5vw,56px)', borderRadius: 28, textAlign: 'center', alignItems: 'center', gap: 16, boxShadow: '0 30px 60px -40px rgba(32,42,50,.4)' }}>
          <div style={{ width: 80, height: 80, borderRadius: '50%', background: '#D7ECE8', display: 'grid', placeItems: 'center' }}><Icon name="task_alt" size={42} style={{ color: 'var(--teal)' }} /></div>
          <div className="mono muted" style={{ fontSize: 13 }}>{done.pet.ref}</div>
          <h1 style={{ fontSize: 'clamp(28px,3vw,38px)', fontWeight: 800, letterSpacing: '-.03em' }}>{lost ? `${done.pet.name}’s report is live` : 'Your found report is live'}</h1>
          <p style={{ maxWidth: 520, fontSize: 16, lineHeight: 1.6, color: 'var(--body)' }}>
            {n
              ? `AI matching compared your photos with ${done.pool} open ${lost ? 'found' : 'lost'} reports and found ${n} possible ${n === 1 ? 'match' : 'matches'} above 50% similarity. Review ${n === 1 ? 'it' : 'them'} carefully before contacting anyone.`
              : `AI matching compared your photos with ${done.pool} open reports. There are no strong matches yet. We’ll notify you as soon as a new report looks similar.`}
          </p>
          <div className="wrap-row" style={{ gap: 10, justifyContent: 'center', marginTop: 8 }}>
            <Link to={`/match?pet=${done.pet.id}`} className="btn btn-teal" style={{ height: 52, padding: '0 22px', borderRadius: 14, fontSize: 15 }}><Icon name="image_search" />See possible matches</Link>
            <Link to={`/pets/${done.pet.id}`} className="btn btn-outline" style={{ height: 52, padding: '0 22px', borderRadius: 14, fontSize: 15 }}>View listing</Link>
            <button className="btn btn-ghost" style={{ height: 52, fontSize: 15 }} onClick={restart}>Start another report</button>
          </div>
        </div>
      </main>
    )
  }

  const opt = <T extends string>(...v: [T, string][]) => v.map(([value, label]) => ({ value, label }))

  return (
    <main className="wrap" style={{ paddingTop: 40, paddingBottom: 88 }}>
      <div className="wrap-row between" style={{ alignItems: 'flex-end', gap: 20, marginBottom: 28 }}>
        <div style={{ maxWidth: 640 }}>
          <div className="mono muted" style={{ fontSize: 12, marginBottom: 8 }}>Step {r.step} of 6</div>
          <h1 className="h-page" style={{ marginBottom: 8 }}>{lost ? 'Report a lost pet' : 'Report a found animal'}</h1>
          <p className="lede">{lost ? 'The more detail you give, the better our matching works. You can edit the report at any time.' : 'Thank you for helping. Add a photo and the place you found the animal, and we will look for its owner.'}</p>
        </div>
        <button className="btn btn-outline" onClick={saveDraft}><Icon name="save" size={19} />Save draft</button>
      </div>

      <ol className="row" style={{ gap: 0, marginBottom: 28, overflowX: 'auto', scrollbarWidth: 'none', paddingBottom: 4, listStyle: 'none', paddingLeft: 0, margin: '0 0 28px' }}>
        {LABELS.map((l, i) => {
          const n = i + 1
          const isDone = n < r.step
          const cur = n === r.step
          return (
            <li key={l} style={{ flex: '1 0 110px' }}>
              <button onClick={() => isDone && set({ step: n })} disabled={!isDone && !cur} aria-current={cur ? 'step' : undefined} className="col" style={{ width: '100%', gap: 8, border: 0, background: 'none', padding: 0, textAlign: 'left', opacity: 1, cursor: isDone ? 'pointer' : 'default' }}>
                <div className="row" style={{ width: '100%' }}>
                  <span className="mono" style={{ width: 32, height: 32, borderRadius: '50%', border: `2px solid ${isDone ? '#176B63' : cur ? '#D65A31' : '#E5D6C6'}`, background: isDone ? '#176B63' : cur ? '#D65A31' : '#fff', color: isDone || cur ? '#fff' : 'var(--muted)', display: 'grid', placeItems: 'center', fontSize: 13, fontWeight: 600, flexShrink: 0 }}>
                    {isDone ? <Icon name="check" size={18} /> : n}
                  </span>
                  <span style={{ flex: 1, height: 2, background: isDone ? '#176B63' : '#E5D6C6', margin: '0 8px' }} />
                </div>
                <span style={{ fontSize: 13, fontWeight: 700, color: cur ? 'var(--ink)' : isDone ? 'var(--teal)' : 'var(--muted)' }}>{l}</span>
              </button>
            </li>
          )
        })}
      </ol>

      <div className="wrap-row" style={{ gap: 24, alignItems: 'flex-start' }}>
        <div className="card" style={{ flex: '1.7 1 540px', minWidth: 0, borderRadius: 24, boxShadow: '0 20px 50px -40px rgba(32,42,50,.4)' }}>
          <div style={{ padding: 'clamp(20px,3vw,36px)' }}>
            {errors.form && <div className="alert" style={{ marginBottom: 18 }}><Icon name="error" />{errors.form}</div>}

            {r.step === 1 && (
              <>
                <h2 className="h-card" style={{ marginBottom: 6 }}>What happened?</h2>
                <p className="lede" style={{ fontSize: 15, marginBottom: 22 }}>Choose the type of report. You can switch before you submit.</p>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(240px,1fr))', gap: 14 }}>
                  {([['lost', 'I lost my pet', 'Create a missing-pet report and get alerts when similar found animals are reported.', 'campaign', '#FBE1D5', '#9E3C19'], ['found', 'I found an animal', 'Report an animal you found or are looking after so its family can find it.', 'volunteer_activism', '#D7ECE8', '#115850']] as const).map(([v, t, d, i, bg, fg]) => {
                    const on = r.kind === v
                    return (
                      <button key={v} aria-pressed={on} onClick={() => set({ kind: v, sample: false, photos: r.sample ? [] : r.photos })} className="col" style={{ textAlign: 'left', padding: 22, borderRadius: 20, border: `2px solid ${on ? '#202A32' : '#E5D6C6'}`, boxShadow: on ? '0 0 0 3px rgba(32,42,50,.08)' : 'none', background: '#fff', gap: 14, position: 'relative' }}>
                        <Icon name="check_circle" size={24} fill={on} style={{ position: 'absolute', top: 18, right: 18, color: on ? '#202A32' : '#CDBFAF' }} />
                        <span style={{ width: 52, height: 52, borderRadius: 16, background: bg, color: fg, display: 'grid', placeItems: 'center' }}><Icon name={i} size={28} /></span>
                        <span style={{ fontSize: 19, fontWeight: 800 }}>{t}</span>
                        <span style={{ fontSize: 14, lineHeight: 1.5, color: 'var(--muted)' }}>{d}</span>
                      </button>
                    )
                  })}
                </div>
              </>
            )}

            {r.step === 2 && (
              <>
                <h2 className="h-card" style={{ marginBottom: 6 }}>Add photos</h2>
                <p className="lede" style={{ fontSize: 15, marginBottom: 22 }}>Up to 6 photos. A clear face shot and one full-body photo give the best matching results.</p>
                <label className="col" style={{ alignItems: 'center', justifyContent: 'center', gap: 10, padding: '32px 20px', border: `2px dashed ${errors.photos ? '#C23B22' : '#D9C7B4'}`, borderRadius: 20, background: 'var(--cream)', cursor: r.photos.length >= 6 ? 'not-allowed' : 'pointer', textAlign: 'center' }}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => { e.preventDefault(); if (!uploading) addFiles([...e.dataTransfer.files]) }}>
                  <input type="file" accept="image/*" multiple disabled={uploading || r.photos.length >= 6} onChange={onFiles} className="sr-only" />
                  <span style={{ width: 56, height: 56, borderRadius: '50%', background: '#fff', display: 'grid', placeItems: 'center', boxShadow: '0 6px 16px rgba(32,42,50,.1)' }}>
                    {uploading ? <Spinner variant="teal" /> : <Icon name="add_photo_alternate" size={28} style={{ color: 'var(--teal)' }} />}
                  </span>
                  <span style={{ fontSize: 16, fontWeight: 800 }}>{uploading ? 'Uploading…' : 'Drop photos here or click to upload'}</span>
                  <span style={{ fontSize: 13, color: 'var(--muted)' }}>JPG, PNG or WebP · up to 10 MB each</span>
                </label>
                <div className="row" style={{ justifyContent: 'center', marginTop: 12 }}>
                  <button className="btn-link" style={{ fontSize: 14 }} onClick={useSample}><Icon name="photo_library" size={18} />No photo handy? Use demo photos</button>
                </div>
                <FieldError msg={errors.photos} />
                {r.photos.length > 0 && (
                  <>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(140px,1fr))', gap: 12, marginTop: 20 }}>
                      {r.photos.map((src, i) => {
                        const q = quality[i]?.score
                        return (
                          <div key={src} onClick={() => set({ photos: [src, ...r.photos.filter((_, j) => j !== i)] })} style={{ position: 'relative', aspectRatio: '1', borderRadius: 16, overflow: 'hidden', background: 'var(--sand)', cursor: 'pointer' }}>
                            <img src={img(src, 400, 400)} alt={`Photo ${i + 1}`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                            {i === 0 && <span style={{ position: 'absolute', left: 8, top: 8, padding: '3px 8px', borderRadius: 6, background: 'var(--ink)', color: '#fff', fontSize: 11, fontWeight: 700 }}>Cover</span>}
                            {q !== undefined && !analyzing && <span className="mono" style={{ position: 'absolute', left: 8, bottom: 8, padding: '3px 7px', borderRadius: 6, background: q >= 75 ? '#176B63' : q >= 65 ? '#C99A2E' : '#C9522B', color: '#fff', fontSize: 11, fontWeight: 600 }}>{q}/100</span>}
                            <button aria-label="Remove photo" onClick={(e) => { e.stopPropagation(); set({ photos: r.photos.filter((_, j) => j !== i) }) }} style={{ position: 'absolute', right: 8, top: 8, width: 30, height: 30, borderRadius: '50%', border: 0, background: 'rgba(255,255,255,.94)', display: 'grid', placeItems: 'center' }}><Icon name="close" size={18} /></button>
                          </div>
                        )
                      })}
                    </div>
                    <div style={{ fontSize: 12.5, color: 'var(--muted)', marginTop: 10 }}>Click a photo to make it the cover.</div>
                  </>
                )}
              </>
            )}

            {r.step === 3 && (
              <>
                <h2 className="h-card" style={{ marginBottom: 6 }}>Pet details</h2>
                <p className="lede" style={{ fontSize: 15, marginBottom: 22 }}>AI suggestions are marked. You can accept, edit or ignore every one.</p>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(240px,1fr))', gap: 18 }}>
                  <label className="field"><span className="field-label">{lost ? 'Pet’s name' : 'Nickname (optional)'}</span>
                    <input className={`input${errors.name ? ' invalid' : ''}`} value={r.name} onChange={(e) => set({ name: e.target.value })} placeholder={lost ? 'e.g. Rex' : 'e.g. the name you’re using for now'} />
                    <FieldError msg={errors.name} /></label>
                  <div><span className="field-label">Species</span>
                    <Seg value={r.species} onChange={(v) => set({ species: v })} options={opt<Draft['species']>(['Dog', 'Dog'], ['Cat', 'Cat'], ['Other', 'Other'])} /></div>
                  <label className="field" style={{ gridColumn: '1/-1' }}>
                    <span className="field-label"><span>Breed</span><span className="demo-tag">AI · M1 DEMO</span></span>
                    <input className="input" value={r.breed} onChange={(e) => set({ breed: e.target.value, breedUsed: false })} placeholder="Type a breed, or leave blank if unsure" />
                    {topBreed && (
                      <div className="wrap-row" style={{ alignItems: 'center', gap: 8, marginTop: 8 }}>
                        <span style={{ fontSize: 13, color: 'var(--muted)' }}>Suggested from your photos:</span>
                        {breedUsed
                          ? <span className="row" style={{ gap: 6, height: 32, padding: '0 12px', borderRadius: 999, background: 'var(--teal)', color: '#fff', fontSize: 13, fontWeight: 700 }}><Icon name="check" size={16} />Using AI suggestion. Edit the field to override.</span>
                          : <button type="button" onClick={() => set({ breed: topBreed, breedUsed: true })} className="row" style={{ gap: 6, height: 32, padding: '0 12px', borderRadius: 999, border: '1px dashed #176B63', background: 'var(--teal-soft)', color: 'var(--teal-dd)', fontSize: 13, fontWeight: 700 }}><Icon name="auto_awesome" size={16} />{topBreed} · {breeds[0][1]}% · Use</button>}
                      </div>
                    )}
                  </label>
                  <div><span className="field-label">Sex</span><Seg value={r.sex} onChange={(v) => set({ sex: v })} options={opt<Draft['sex']>(['Male', 'Male'], ['Female', 'Female'], ['Unknown', 'Not sure'])} /></div>
                  <div><span className="field-label">Size</span><Seg value={r.size} onChange={(v) => set({ size: v })} options={opt<Draft['size']>(['Small', 'Small'], ['Medium', 'Medium'], ['Large', 'Large'])} /></div>
                  <label className="field"><span className="field-label">Age</span>
                    <select className="select" value={r.age} onChange={(e) => set({ age: e.target.value })}>
                      {['Puppy / kitten (under 1 year)', 'Adult (1–7 years)', 'Senior (8+ years)', 'Not sure'].map((a) => <option key={a}>{a}</option>)}
                    </select></label>
                  <label className="field"><span className="field-label">Main colour</span>
                    <input className={`input${errors.color ? ' invalid' : ''}`} value={r.color} onChange={(e) => set({ color: e.target.value })} placeholder="e.g. Tricolour, orange tabby" />
                    <FieldError msg={errors.color} /></label>
                  <div style={{ gridColumn: '1/-1' }}><span className="field-label">Microchipped?</span>
                    <Seg style={{ maxWidth: 360 }} value={r.chip} onChange={(v) => set({ chip: v })} options={opt<Draft['chip']>(['Yes', 'Yes'], ['No', 'No'], ['Unknown', 'Not sure'])} /></div>
                  <label className="field" style={{ gridColumn: '1/-1' }}>
                    <span className="field-label"><span>Description</span><span className="mono" style={{ fontWeight: 500, color: 'var(--muted)' }}>{r.desc.length} / 600</span></span>
                    <textarea className={`textarea${errors.desc ? ' invalid' : ''}`} rows={5} maxLength={600} value={r.desc} onChange={(e) => set({ desc: e.target.value })} placeholder="Personality, collar, markings, what happened and anything that helps someone recognise them…" />
                    <FieldError msg={errors.desc} /></label>
                  <div style={{ gridColumn: '1/-1', padding: 16, borderRadius: 16, background: '#F4FAF9', border: '1px solid #D7ECE8' }}>
                    <div className="wrap-row between" style={{ alignItems: 'center', gap: 8, marginBottom: 10 }}>
                      <span className="row" style={{ gap: 6, fontSize: 14, fontWeight: 800 }}><Icon name="sell" size={19} style={{ color: 'var(--teal)' }} />Suggested tags from your description</span>
                      <span className="demo-tag">M4 · TF-IDF + SVM · DEMO</span>
                    </div>
                    {visibleTags.length ? (
                      <div className="wrap-row" style={{ gap: 8 }}>
                        {visibleTags.map((t) => {
                          const a = r.accepted.includes(t.label)
                          return (
                            <span key={t.label} className="row" style={{ borderRadius: 999, border: `1px solid ${a ? '#176B63' : '#CFE3DF'}`, background: a ? '#176B63' : '#fff', color: a ? '#fff' : 'var(--ink)', overflow: 'hidden' }}>
                              <button type="button" aria-pressed={a} onClick={() => set({ accepted: a ? r.accepted.filter((x) => x !== t.label) : [...r.accepted, t.label] })} className="row" style={{ gap: 5, height: 32, padding: '0 6px 0 10px', border: 0, background: 'none', color: 'inherit', fontSize: 13, fontWeight: 700 }}>
                                <Icon name={a ? 'check' : 'add'} size={16} />{t.label}<span className="mono" style={{ fontSize: 11, fontWeight: 500, opacity: 0.75 }}>{t.confidence.toFixed(2)}</span>
                              </button>
                              <button type="button" aria-label={`Dismiss ${t.label}`} onClick={() => set({ dismissed: [...r.dismissed, t.label], accepted: r.accepted.filter((x) => x !== t.label) })} style={{ height: 32, width: 28, border: 0, background: 'none', color: 'inherit', display: 'grid', placeItems: 'center', opacity: 0.7 }}><Icon name="close" size={15} /></button>
                            </span>
                          )
                        })}
                      </div>
                    ) : <div style={{ fontSize: 13, color: 'var(--muted)' }}>Tags appear as you write. Try mentioning a collar, microchip, temperament or markings.</div>}
                  </div>
                </div>
              </>
            )}

            {r.step === 4 && (
              <>
                <h2 className="h-card" style={{ marginBottom: 6 }}>{lost ? 'Where was your pet last seen?' : 'Where did you find the animal?'}</h2>
                <p className="lede" style={{ fontSize: 15, marginBottom: 18 }}>Click the map to drop a pin. Only an approximate area is shown publicly.</p>
                <MapView height={380} center={r.pin ? [r.pin.lat, r.pin.lng] : MAP_CENTER} zoom={r.pin ? 14 : 12} cursor="crosshair"
                  border={`2px solid ${errors.pin ? '#C23B22' : '#F0E4D6'}`}
                  onPick={(lat, lng) => set({ pin: { lat, lng } })}
                  pins={r.pin ? [{ kind: 'drop', id: 'pin', lat: r.pin.lat, lng: r.pin.lng, color: lost ? '#D65A31' : '#176B63' }] : []}>
                  {!r.pin && <div className="map-overlay nowrap" style={{ left: '50%', top: 16, transform: 'translateX(-50%)', padding: '8px 14px', borderRadius: 999, background: 'var(--ink)', color: '#fff', fontSize: 13, fontWeight: 700, pointerEvents: 'none' }}>Click to place a pin</div>}
                  {r.pin && <div className="map-overlay mono" style={{ left: 12, bottom: 26, padding: '8px 12px', borderRadius: 10, background: '#fff', fontSize: 12, boxShadow: '0 6px 16px rgba(32,42,50,.15)', pointerEvents: 'none' }}>{r.pin.lat.toFixed(4)}° N, {r.pin.lng.toFixed(4)}° E</div>}
                </MapView>
                <FieldError msg={errors.pin} />
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(180px,1fr))', gap: 16, marginTop: 18 }}>
                  <label className="field" style={{ gridColumn: '1/-1' }}>
                    <span className="field-label"><span>Street or landmark</span><button type="button" className="btn-link" style={{ fontSize: 13, gap: 4 }} onClick={useMyLoc}><Icon name="my_location" size={16} />Use my location</button></span>
                    <input className="input" value={r.address} onChange={(e) => set({ address: e.target.value })} placeholder="e.g. Corniche, La Marsa" />
                  </label>
                  <label className="field"><span className="field-label">{lost ? 'Date last seen' : 'Date found'}</span>
                    <input type="date" max={today()} className={`input${errors.date ? ' invalid' : ''}`} style={{ padding: '0 12px' }} value={r.date} onChange={(e) => set({ date: e.target.value })} />
                    <FieldError msg={errors.date} /></label>
                  <label className="field"><span className="field-label">Approximate time</span>
                    <input type="time" className="input" style={{ padding: '0 12px' }} value={r.time} onChange={(e) => set({ time: e.target.value })} /></label>
                </div>
              </>
            )}

            {r.step === 5 && (
              <>
                <h2 className="h-card" style={{ marginBottom: 6 }}>How can people reach you?</h2>
                <p className="lede" style={{ fontSize: 15, marginBottom: 22 }}>Your phone number stays hidden unless you choose to show it.</p>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(240px,1fr))', gap: 18 }}>
                  <label className="field" style={{ gridColumn: '1/-1' }}><span className="field-label">Your name</span>
                    <input className={`input${errors.cname ? ' invalid' : ''}`} value={r.cname} onChange={(e) => set({ cname: e.target.value })} autoComplete="name" />
                    <FieldError msg={errors.cname} /></label>
                  <label className="field"><span className="field-label">Phone</span>
                    <div className="row" style={{ border: `1.5px solid ${errors.phone ? '#C23B22' : '#E5D6C6'}`, borderRadius: 12, background: '#fff', overflow: 'hidden', height: 48 }}>
                      <span className="row" style={{ alignSelf: 'stretch', padding: '0 12px', background: 'var(--cream)', borderRight: '1px solid var(--line-2)', fontSize: 14, fontWeight: 700 }}>+216</span>
                      <input type="tel" value={r.phone} onChange={(e) => set({ phone: e.target.value })} placeholder="22 418 903" autoComplete="tel" style={{ flex: 1, minWidth: 0, border: 0, padding: '0 12px', fontSize: 15, outline: 'none', height: '100%' }} />
                    </div>
                    <FieldError msg={errors.phone} /></label>
                  <label className="field"><span className="field-label">Email</span>
                    <input type="email" className={`input${errors.email ? ' invalid' : ''}`} value={r.email} onChange={(e) => set({ email: e.target.value })} placeholder="name@example.com" autoComplete="email" />
                    <FieldError msg={errors.email} /></label>
                  <div style={{ gridColumn: '1/-1' }}><span className="field-label">Preferred contact</span>
                    <Seg style={{ maxWidth: 480 }} value={r.contact} onChange={(v) => set({ contact: v })} options={opt<Draft['contact']>(['Phone call', 'Phone call'], ['WhatsApp', 'WhatsApp'], ['In-app messages', 'In-app only'])} /></div>
                  <label className="check" style={{ gridColumn: '1/-1', alignItems: 'center' }}>
                    <input type="checkbox" checked={r.showPhone} onChange={(e) => set({ showPhone: e.target.checked })} style={{ marginTop: 0 }} />Show my phone number on the public listing
                  </label>
                </div>
              </>
            )}

            {r.step === 6 && (
              <>
                <h2 className="h-card" style={{ marginBottom: 6 }}>Review and submit</h2>
                <p className="lede" style={{ fontSize: 15, marginBottom: 22 }}>Check everything once more. You can edit any section.</p>
                <div className="wrap-row" style={{ gap: 20 }}>
                  <div style={{ flex: '0 0 180px', aspectRatio: '1', borderRadius: 18, overflow: 'hidden', background: 'var(--sand)' }}>{r.photos[0] && <img src={img(r.photos[0], 400, 400)} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />}</div>
                  <div className="col" style={{ flex: '1 1 300px', minWidth: 0, border: '1px solid var(--line)', borderRadius: 16, overflow: 'hidden' }}>
                    {([
                      ['Report type', lost ? 'Lost pet' : 'Found animal', 1],
                      ['Photos', `${r.photos.length} photo${r.photos.length === 1 ? '' : 's'}${r.bg ? ' · background removed' : ''}`, 2],
                      ['Name', r.name || '—', 3],
                      ['Species & breed', `${r.species}${r.breed ? ` · ${r.breed}` : ''}`, 3],
                      ['Sex · Size · Age', `${r.sex} · ${r.size} · ${r.age}`, 3],
                      ['Colour', r.color || '—', 3],
                      ['Tags', r.accepted.filter((l) => tags.some((t) => t.label === l)).join(', ') || 'None selected', 3],
                      ['Location', `${r.address || 'Pinned on map'}${r.pin ? ` (${r.pin.lat.toFixed(3)}, ${r.pin.lng.toFixed(3)})` : ''}`, 4],
                      ['Date & time', `${r.date} · ${r.time}`, 4],
                      ['Contact', `${r.cname} · +216 ${r.phone.replace(/^\+?216\s?/, '')} · ${r.contact}`, 5],
                    ] as const).map(([k, v, s]) => (
                      <div key={k} className="kv">
                        <span style={{ flex: '0 0 130px', fontSize: 13, color: 'var(--muted)' }}>{k}</span>
                        <span style={{ flex: 1, fontSize: 14, fontWeight: 600, minWidth: 0, overflowWrap: 'anywhere' }}>{v}</span>
                        <button className="btn-link" style={{ fontSize: 13 }} onClick={() => set({ step: s })}>Edit</button>
                      </div>
                    ))}
                  </div>
                </div>
                <label className="check" style={{ marginTop: 20 }}>
                  <input type="checkbox" checked={r.consent} onChange={(e) => set({ consent: e.target.checked })} />
                  I confirm this information is accurate, and I understand AI matches are suggestions that I need to verify.
                </label>
                <FieldError msg={errors.consent} />
                {!user && <div className="info" style={{ marginTop: 16 }}><Icon name="info" size={20} />You’ll be asked to sign in before the report is published. Your draft is kept.</div>}
              </>
            )}
          </div>
          <div className="wrap-row between" style={{ alignItems: 'center', gap: 12, padding: '18px clamp(20px,3vw,36px)', borderTop: '1px solid var(--line)', background: 'var(--cream-2)', borderRadius: '0 0 24px 24px' }}>
            <div>{r.step > 1 && <button className="btn btn-outline" style={{ height: 48, fontSize: 15 }} onClick={() => { set({ step: r.step - 1 }); setErrors({}); top() }}><Icon name="arrow_back" size={19} />Back</button>}</div>
            <div className="row" style={{ gap: 10 }}>
              <button className="btn btn-ghost" style={{ height: 48, fontSize: 15 }} onClick={saveDraft}>Save draft</button>
              <button className="btn btn-primary" style={{ height: 48, padding: '0 22px', fontSize: 15 }} disabled={submit.isPending || uploading} onClick={next}>
                {submit.isPending ? <Spinner /> : <>{r.step === 6 ? 'Submit report' : 'Continue'}<Icon name={r.step === 6 ? 'send' : 'arrow_forward'} size={19} /></>}
              </button>
            </div>
          </div>
        </div>

        <aside className="col" style={{ flex: '1 1 320px', minWidth: 0, position: 'sticky', top: 96, gap: 14 }}>
          <div style={{ padding: 20, borderRadius: 22, background: 'var(--forest)', color: 'var(--cream)' }}>
            <div className="row between" style={{ gap: 8, marginBottom: 6 }}>
              <span className="row" style={{ gap: 8, fontSize: 16, fontWeight: 800 }}><Icon name="auto_awesome" size={21} style={{ color: 'var(--mint)' }} />AI assistant</span>
              <span className="dash-tag">DEMO RESULTS</span>
            </div>
            <div style={{ fontSize: 13, lineHeight: 1.5, color: 'var(--sage)' }}>Suggestions only. Nothing is applied to your report without your approval.</div>
            {!r.photos.length && (
              <div className="col" style={{ gap: 12, marginTop: 18 }}>
                {[['pets', 'Breed suggestion from your photos'], ['photo_camera', 'Photo quality feedback and tips'], ['background_replace', 'Optional background removal'], ['sell', 'Tags extracted from your description']].map(([i, t]) => (
                  <div key={t} className="row" style={{ gap: 10, fontSize: 13, color: 'var(--sage)' }}><Icon name={i} size={19} style={{ color: 'var(--mint)' }} />{t}</div>
                ))}
              </div>
            )}
            {analyzing && <div className="row" style={{ gap: 12, marginTop: 18, padding: 14, borderRadius: 14, background: 'rgba(255,255,255,.06)' }}><Spinner /><span style={{ fontSize: 14, fontWeight: 600 }}>Analysing your photos…</span></div>}
          </div>
          {ai && !analyzing && r.photos.length > 0 && (
            <>
              <div className="panel" style={{ padding: 18, borderRadius: 20 }}>
                <div className="row between" style={{ marginBottom: 14, gap: 8 }}><span style={{ fontSize: 14, fontWeight: 800 }}>Photo quality</span><span className="demo-tag">M5 · PAWPULARITY</span></div>
                <div className="row" style={{ gap: 14, marginBottom: 14 }}>
                  <div className="ring" style={{ width: 68, height: 68, background: `conic-gradient(#176B63 ${Math.round(avg * 3.6)}deg, #EEF1F0 0)` }}><div style={{ width: 54, height: 54, fontSize: 18 }}>{avg}</div></div>
                  <div><div style={{ fontSize: 14, fontWeight: 700 }}>{avg >= 75 ? 'Strong listing photos' : avg >= 65 ? 'Good, with room to improve' : 'Consider retaking a photo'}</div><div style={{ fontSize: 13, color: 'var(--muted)' }}>Best cover: Photo {best + 1}</div></div>
                </div>
                <div className="col" style={{ gap: 8 }}>
                  {quality.map((q, i) => (
                    <div key={q.url} className="row" style={{ gap: 10 }}>
                      <img src={img(r.photos[i], 100, 100)} alt="" style={{ width: 36, height: 36, borderRadius: 8, objectFit: 'cover' }} />
                      <div className="grow" style={{ fontSize: 12.5, lineHeight: 1.35, color: 'var(--body)' }}>{q.tip}</div>
                      <span className="mono" style={{ fontSize: 12, fontWeight: 600 }}>{q.score}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="panel" style={{ padding: 18, borderRadius: 20 }}>
                <div className="row between" style={{ marginBottom: 12, gap: 8 }}><span style={{ fontSize: 14, fontWeight: 800 }}>Breed suggestion</span><span className="demo-tag">M1 · BREED CNN</span></div>
                <BreedBars breeds={breeds} compact />
                {!breedUsed && topBreed && <button className="btn btn-soft btn-block" style={{ marginTop: 12, height: 40, borderRadius: 10, fontSize: 13 }} onClick={() => set({ breed: topBreed, breedUsed: true })}>Use “{topBreed}” in my report</button>}
              </div>
              <div className="panel" style={{ padding: 18, borderRadius: 20 }}>
                <div className="row between" style={{ marginBottom: 12, gap: 8 }}><span style={{ fontSize: 14, fontWeight: 800 }}>Remove background</span><button type="button" role="switch" aria-checked={r.bg} aria-label="Toggle background removal" className={`toggle${r.bg ? ' on' : ''}`} onClick={() => set({ bg: !r.bg })} /></div>
                <div style={{ position: 'relative', aspectRatio: '4/3', borderRadius: 14, overflow: 'hidden', background: 'repeating-conic-gradient(#ECE5DB 0 25%, #FFFFFF 0 50%) 0 0/16px 16px' }}>
                  <img src={img(cover, 600, 450)} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover', WebkitMaskImage: ai.cutout ? 'none' : bgMask, maskImage: ai.cutout ? 'none' : bgMask, transition: 'all .3s' }} />
                </div>
                <div className="row between" style={{ marginTop: 10, fontSize: 12, color: 'var(--muted)' }}><span>M6 · U-Net segmentation · {ai.cutout ? 'model output' : 'demo preview'}</span><span style={{ fontWeight: 700 }}>Original kept</span></div>
              </div>
            </>
          )}
          <div className="note" style={{ fontSize: 13 }}><Icon name="image_search" size={19} /><span>After you submit, AI Matching compares your photos with every open {lost ? 'found' : 'lost'} report and notifies you of possible matches.</span></div>
        </aside>
      </div>
    </main>
  )
}
