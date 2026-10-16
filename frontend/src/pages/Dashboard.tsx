import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { Link, Navigate, useLocation, useNavigate, useParams } from 'react-router-dom'
import { PetGrid } from '../components/PetCard'
import { FieldError, Icon, LogoMark, Seg, Skeleton, Spinner, Toggle } from '../components/ui'
import { api, ApiError, type Application, type Dashboard as DashData, type User } from '../lib/api'
import { useAuth } from '../lib/auth'
import { fmtDate, fmtKm, GOVERNORATES, img, initials, relTime } from '../lib/format'
import { useDashboard } from '../lib/queries'
import { useToast } from '../lib/toast'

const TABS = [
  ['overview', 'space_dashboard', 'Overview'],
  ['listings', 'list_alt', 'My reports'],
  ['matches', 'auto_awesome', 'Possible matches'],
  ['applications', 'assignment', 'Applications'],
  ['saved', 'favorite', 'Saved pets'],
  ['notifications', 'notifications', 'Notifications'],
  ['settings', 'settings', 'Account settings'],
] as const
type Tab = (typeof TABS)[number][0]
const TITLES: Record<Tab, string> = { overview: 'Overview', listings: 'My lost & found reports', matches: 'Possible matches', applications: 'Adoption applications', saved: 'Saved pets', notifications: 'Notifications', settings: 'Account settings' }
const STAGES = ['Submitted', 'Under review', 'Home visit', 'Decision']

function greeting() {
  const h = new Date().getHours()
  return h < 12 ? 'Good morning' : h < 18 ? 'Good afternoon' : 'Good evening'
}

function Panel({ title, tag, action, children }: { title: React.ReactNode; tag?: string; action?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="panel">
      <div className="wrap-row between" style={{ alignItems: 'center', gap: 8, marginBottom: 14 }}>
        <span className="panel-title">{title}</span>
        {tag && <span className="demo-tag">{tag}</span>}
        {action}
      </div>
      {children}
    </div>
  )
}

function MatchesPanel({ d }: { d: DashData }) {
  return (
    <Panel title="Recent AI matching results" tag="M2 · DEMO">
      {d.matches.length ? (
        <div className="col" style={{ gap: 8 }}>
          {d.matches.slice(0, 6).map((m) => (
            <Link key={m.id} to={`/match?pet=${m.query.id}&selected=${m.candidate.id}`} className="row" style={{ gap: 14, padding: 10, borderRadius: 14, border: '1px solid var(--line-3)', color: 'var(--ink)' }}>
              <div className="row" style={{ flexShrink: 0 }}>
                <img src={img(m.query.photos[0], 160, 160)} alt="" style={{ width: 44, height: 44, borderRadius: '50%', objectFit: 'cover', border: '2px solid #fff' }} />
                <img src={img(m.candidate.photos[0], 160, 160)} alt="" style={{ width: 44, height: 44, borderRadius: '50%', objectFit: 'cover', border: '2px solid #fff', marginLeft: -12 }} />
              </div>
              <div className="grow">
                <div className="row" style={{ gap: 6 }}><span style={{ fontSize: 14, fontWeight: 800 }}>{m.query.name} ↔ {m.candidate.name}</span>{m.is_new && <span style={{ padding: '2px 6px', borderRadius: 5, background: 'var(--orange)', color: '#fff', fontSize: 10, fontWeight: 800 }}>NEW</span>}</div>
                <div style={{ fontSize: 12.5, color: 'var(--muted)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{m.candidate.status === 'found' ? 'Found' : 'Lost'} in {m.candidate.city} · {fmtKm(m.candidate.distance_km)} · {relTime(m.created_at)}</div>
              </div>
              <div className="ring" style={{ width: 50, height: 50, background: `conic-gradient(#176B63 ${Math.round(m.score * 3.6)}deg, #E6EEEC 0)` }}><div style={{ width: 40, height: 40, fontSize: 12 }}>{m.score}%</div></div>
            </Link>
          ))}
        </div>
      ) : <div style={{ fontSize: 14, color: 'var(--muted)' }}>No possible matches yet. We compare every new report with yours automatically.</div>}
      <div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 12 }}>Possible matches only. Confirm markings before arranging a handover.</div>
    </Panel>
  )
}

function NotificationsPanel({ d }: { d: DashData }) {
  const qc = useQueryClient()
  const navigate = useNavigate()
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ['dashboard'] })
    qc.invalidateQueries({ queryKey: ['notifications'] })
  }
  const readAll = useMutation({ mutationFn: () => api.post('/api/me/notifications/read-all'), onSuccess: refresh })
  const open = async (id: number, link: string, unread: boolean) => {
    if (unread) api.post(`/api/me/notifications/${id}/read`).then(refresh).catch(() => {})
    if (link) navigate(link)
  }
  return (
    <Panel title={<>Notifications <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--muted)' }}>· {d.unread ? `${d.unread} unread` : 'All caught up'}</span></>}
      action={d.unread > 0 && <button className="btn-link" style={{ fontSize: 13 }} onClick={() => readAll.mutate()}>Mark all read</button>}>
      {d.notifications.length ? (
        <div className="col">
          {d.notifications.map((n) => (
            <button key={n.id} onClick={() => open(n.id, n.link, n.unread)} className="row" style={{ gap: 12, alignItems: 'flex-start', padding: '12px 8px', border: 0, borderBottom: '1px solid var(--line-3)', background: n.unread ? '#FFFBF7' : '#fff', textAlign: 'left', borderRadius: 8 }}>
              <span style={{ width: 36, height: 36, borderRadius: 10, background: 'var(--peach-2)', color: 'var(--rust)', display: 'grid', placeItems: 'center', flexShrink: 0 }}><Icon name={n.icon} size={19} /></span>
              <span className="grow">
                <span style={{ display: 'block', fontSize: 14, fontWeight: n.unread ? 800 : 600 }}>{n.title}</span>
                {n.body && <span style={{ display: 'block', fontSize: 13, color: 'var(--muted)', lineHeight: 1.4, marginTop: 2 }}>{n.body}</span>}
                <span style={{ display: 'block', fontSize: 12, color: 'var(--muted-2)', marginTop: 4 }}>{relTime(n.created_at)}</span>
              </span>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: n.unread ? 'var(--orange)' : 'transparent', marginTop: 6, flexShrink: 0 }} />
            </button>
          ))}
        </div>
      ) : <div style={{ fontSize: 14, color: 'var(--muted)' }}>No notifications yet.</div>}
    </Panel>
  )
}

function AppCard({ a }: { a: Application }) {
  return (
    <div>
      <Link to={`/pets/${a.pet_id}`} className="row" style={{ gap: 12, marginBottom: 14, color: 'var(--ink)' }}>
        <img src={img(a.pet_photo, 160, 160)} alt="" style={{ width: 48, height: 48, borderRadius: 12, objectFit: 'cover' }} />
        <div className="grow"><div style={{ fontSize: 15, fontWeight: 800 }}>{a.pet_name} <span className="mono" style={{ fontSize: 12, fontWeight: 500, color: 'var(--muted)' }}>{a.code}</span></div><div style={{ fontSize: 12.5, color: 'var(--muted)' }}>{a.shelter_name} · {fmtDate(a.created_at)}</div></div>
        <span className="nowrap" style={{ padding: '4px 10px', borderRadius: 999, background: '#FBE1D5', color: 'var(--rust-dd)', fontSize: 12, fontWeight: 700 }}>{a.stage_label}</span>
      </Link>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,minmax(0,1fr))', gap: 4 }}>
        {STAGES.map((l, i) => (
          <div key={l} className="col" style={{ alignItems: 'center', gap: 6, textAlign: 'center' }}>
            <span className="mono" style={{ width: 26, height: 26, borderRadius: '50%', background: i < a.stage ? '#176B63' : i === a.stage ? '#D65A31' : '#EADFD2', color: i <= a.stage ? '#fff' : 'var(--muted)', display: 'grid', placeItems: 'center', fontSize: 11, fontWeight: 600 }}>
              {i < a.stage ? <Icon name="check" size={15} /> : i + 1}
            </span>
            <span style={{ fontSize: 11.5, fontWeight: 700, color: i <= a.stage ? 'var(--ink)' : 'var(--muted)' }}>{l}</span>
          </div>
        ))}
      </div>
      <div style={{ marginTop: 12, padding: '10px 12px', borderRadius: 10, background: 'var(--cream)', fontSize: 13, color: 'var(--body)' }}>{a.next_step}</div>
    </div>
  )
}

function IncomingApplications() {
  const qc = useQueryClient()
  const toast = useToast()
  const { data } = useQuery({ queryKey: ['incoming'], queryFn: () => api.get<Application[]>('/api/adoptions/incoming') })
  const update = useMutation({
    mutationFn: (v: { id: number; stage: number }) => api.patch(`/api/adoptions/${v.id}`, { stage: v.stage }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['incoming'] })
      toast('Application updated. The adopter has been notified.')
    },
    onError: (e: Error) => toast(e.message, true),
  })
  return (
    <Panel title="Incoming applications" tag="SHELTER">
      {data?.length ? (
        <div className="col" style={{ gap: 12 }}>
          {data.map((a) => (
            <div key={a.id} className="wrap-row" style={{ gap: 12, alignItems: 'center', padding: 10, borderRadius: 14, border: '1px solid var(--line-3)' }}>
              <img src={img(a.pet_photo, 120, 120)} alt="" style={{ width: 44, height: 44, borderRadius: 10, objectFit: 'cover' }} />
              <div className="grow"><div style={{ fontSize: 14, fontWeight: 800 }}>{a.pet_name} · <span className="mono" style={{ fontWeight: 500 }}>{a.code}</span></div><div style={{ fontSize: 12.5, color: 'var(--muted)' }}>Received {fmtDate(a.created_at)}</div></div>
              <select className="select" style={{ width: 'auto', height: 40, fontSize: 14 }} value={a.stage} aria-label={`Stage for ${a.code}`} onChange={(e) => update.mutate({ id: a.id, stage: +e.target.value })}>
                {STAGES.map((s, i) => <option key={s} value={i}>{s}</option>)}
              </select>
            </div>
          ))}
        </div>
      ) : <div style={{ fontSize: 14, color: 'var(--muted)' }}>No applications for your animals yet.</div>}
    </Panel>
  )
}

function ReportsPanel({ d }: { d: DashData }) {
  const qc = useQueryClient()
  const toast = useToast()
  const resolve = useMutation({
    mutationFn: (id: number) => api.post(`/api/pets/${id}/resolve`),
    onSuccess: () => {
      qc.invalidateQueries()
      toast('Marked as reunited. Wonderful news!')
    },
    onError: (e: Error) => toast(e.message, true),
  })
  return (
    <Panel title="My lost & found reports" action={<Link to="/report" className="btn-link" style={{ fontSize: 13 }}>New report</Link>}>
      {d.reports.length ? (
        <div className="col" style={{ gap: 10 }}>
          {d.reports.map(({ pet: p, sightings, matches }) => (
            <div key={p.id} className="wrap-row" style={{ gap: 14, alignItems: 'center', padding: 12, borderRadius: 16, border: '1px solid var(--line-3)' }}>
              <Link to={`/pets/${p.id}`}><img src={img(p.photos[0], 200, 200)} alt="" style={{ width: 64, height: 64, borderRadius: 14, objectFit: 'cover' }} /></Link>
              <div style={{ flex: '1 1 180px', minWidth: 0 }}>
                <div className="row" style={{ gap: 8 }}>
                  <span style={{ fontSize: 15, fontWeight: 800 }}>{p.name}</span>
                  <span style={{ padding: '3px 8px', borderRadius: 999, background: p.resolved ? '#D7ECE8' : '#FBE1D5', color: p.resolved ? '#115850' : '#9E3C19', fontSize: 11, fontWeight: 700 }}>{p.resolved ? (p.status === 'adopt' ? 'Adopted' : 'Reunited') : 'Active'}</span>
                </div>
                <div style={{ fontSize: 12.5, color: 'var(--muted)', margin: '2px 0 6px' }}>{p.status === 'lost' ? 'Lost' : p.status === 'found' ? 'Found' : 'Adoption'} · {p.city} · {p.status === 'lost' ? 'since' : ''} {fmtDate(p.event_date)}</div>
                <div className="wrap-row" style={{ gap: 10 }}>
                  {(p.resolved ? [['favorite', 'Closed listing']] : [['visibility', `${p.views} views`], ['pin_drop', `${sightings} sighting${sightings === 1 ? '' : 's'}`], ['auto_awesome', `${matches} match${matches === 1 ? '' : 'es'}`]]).map(([i, l]) => (
                    <span key={l} className="row" style={{ gap: 4, fontSize: 12, color: 'var(--body)' }}><Icon name={i} size={15} style={{ color: 'var(--teal)' }} />{l}</span>
                  ))}
                </div>
              </div>
              <div className="row" style={{ gap: 8 }}>
                <Link to={`/pets/${p.id}`} className="btn btn-outline btn-sm">View</Link>
                {!p.resolved && p.status !== 'adopt' && <button className="btn btn-soft btn-sm" disabled={resolve.isPending} onClick={() => resolve.mutate(p.id)}>Mark reunited</button>}
              </div>
            </div>
          ))}
        </div>
      ) : <div style={{ fontSize: 14, color: 'var(--muted)' }}>You haven’t published a report yet. <Link to="/report">Report a lost or found pet</Link>.</div>}
    </Panel>
  )
}

function SettingsPanel({ user }: { user: User }) {
  const { setUser, logout } = useAuth()
  const toast = useToast()
  const [s, setS] = useState(user)
  const [errors, setErrors] = useState<Record<string, string>>({})
  useEffect(() => setS(user), [user])
  const save = useMutation({
    mutationFn: () => api.patch<User>('/api/me', {
      name: s.name, email: s.email, phone: s.phone || null, gov: s.gov, lang: s.lang, notify_email: s.notify_email,
      notify_sms: s.notify_sms, notify_push: s.notify_push, show_phone: s.show_phone, min_match_score: s.min_match_score,
    }),
    onSuccess: (u) => {
      setUser(u)
      setErrors({})
      toast('Settings saved')
    },
    onError: (e: ApiError) => setErrors(Object.keys(e.errors).length ? e.errors : { form: e.message }),
  })
  const set = <K extends keyof User>(k: K, v: User[K]) => {
    setS((x) => ({ ...x, [k]: v }))
    setErrors((e) => ({ ...e, [k]: '' }))
  }
  const toggles: [keyof User, string, string][] = [
    ['notify_email', 'Email notifications', 'Matches, sightings and application updates'],
    ['notify_sms', 'SMS alerts', 'Only for new possible matches above your threshold'],
    ['notify_push', 'Browser notifications', 'Real-time alerts while PawConnect is open'],
    ['show_phone', 'Show my phone number on listings', 'Otherwise people contact you through in-app messages'],
  ]
  return (
    <div className="col" style={{ gap: 20, maxWidth: 820 }}>
      {errors.form && <div className="alert"><Icon name="error" />{errors.form}</div>}
      <div className="panel" style={{ padding: 24 }}>
        <div className="panel-title" style={{ marginBottom: 16 }}>Profile</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(240px,1fr))', gap: 16 }}>
          <label className="field"><span className="field-label">Full name</span><input className={`input${errors.name ? ' invalid' : ''}`} style={{ height: 46 }} value={s.name} onChange={(e) => set('name', e.target.value)} /><FieldError msg={errors.name} /></label>
          <label className="field"><span className="field-label">Email</span><input type="email" className={`input${errors.email ? ' invalid' : ''}`} style={{ height: 46 }} value={s.email} onChange={(e) => set('email', e.target.value)} /><FieldError msg={errors.email} /></label>
          <label className="field"><span className="field-label">Phone</span><input type="tel" className={`input${errors.phone ? ' invalid' : ''}`} style={{ height: 46 }} value={s.phone} onChange={(e) => set('phone', e.target.value)} placeholder="+216 22 418 903" /><FieldError msg={errors.phone} /></label>
          <label className="field"><span className="field-label">Governorate</span><select className="select" style={{ height: 46 }} value={s.gov} onChange={(e) => set('gov', e.target.value)}>{GOVERNORATES.map((g) => <option key={g}>{g}</option>)}</select></label>
        </div>
        <div style={{ marginTop: 18 }}><span className="field-label">Language</span>
          <Seg style={{ maxWidth: 420 }} value={s.lang} onChange={(v) => set('lang', v)} options={['English', 'Français', 'العربية'].map((v) => ({ value: v, label: v }))} /></div>
      </div>
      <div className="panel" style={{ padding: 24 }}>
        <div className="panel-title" style={{ marginBottom: 8 }}>Notifications &amp; privacy</div>
        <div className="col">
          {toggles.map(([k, l, d]) => (
            <div key={k} className="row between" style={{ gap: 16, padding: '14px 0', borderBottom: '1px solid var(--line-3)' }}>
              <div><div style={{ fontSize: 14, fontWeight: 700 }}>{l}</div><div style={{ fontSize: 13, color: 'var(--muted)' }}>{d}</div></div>
              <Toggle label={l} on={!!s[k]} onChange={(v) => set(k, v as never)} />
            </div>
          ))}
        </div>
        <div style={{ marginTop: 18 }}>
          <div className="row between" style={{ fontSize: 14, fontWeight: 700, marginBottom: 8, maxWidth: 420 }}><span>Alert me for matches above</span><span className="mono" style={{ color: 'var(--teal)' }}>{s.min_match_score}%</span></div>
          <input type="range" min={30} max={95} step={5} value={s.min_match_score} onChange={(e) => set('min_match_score', +e.target.value)} aria-label="Match alert threshold" style={{ width: '100%', maxWidth: 420 }} />
        </div>
      </div>
      <div className="wrap-row" style={{ gap: 10 }}>
        <button className="btn btn-primary" style={{ height: 48, padding: '0 22px', fontSize: 15 }} disabled={save.isPending} onClick={() => save.mutate()}>{save.isPending ? <Spinner /> : 'Save changes'}</button>
        <button className="btn btn-outline" style={{ height: 48, padding: '0 20px', fontSize: 15 }} onClick={logout}>Sign out</button>
      </div>
    </div>
  )
}

export default function Dashboard() {
  const { user, ready, logout } = useAuth()
  const location = useLocation()
  const navigate = useNavigate()
  const tab = (useParams().tab ?? 'overview') as Tab
  const { data: d, isError } = useDashboard()

  if (!ready) return <div className="row" style={{ minHeight: '100vh', justifyContent: 'center' }}><Spinner variant="teal" /></div>
  if (!user) return <Navigate to="/login" replace state={{ from: location.pathname, note: 'Sign in to open your dashboard.' }} />
  if (!TITLES[tab]) return <Navigate to="/dashboard" replace />

  const active = d?.reports.find((r) => !r.pet.resolved && r.pet.status === 'lost')
  const newMatches = active ? d!.matches.filter((m) => m.query.id === active.pet.id && m.is_new).length : 0
  const badges: Partial<Record<Tab, number>> = d ? {
    listings: d.reports.length, matches: d.matches.length, applications: d.applications.length, saved: d.saved.length, notifications: d.unread,
  } : {}
  const show = (k: Tab) => tab === 'overview' || tab === k
  const signOut = () => {
    logout()
    navigate('/')
  }

  return (
    <div className="wrap-row" style={{ minHeight: '100vh', background: 'var(--cream)' }}>
      <aside className="col dash-aside">
        <Link to="/" className="logo" style={{ padding: '6px 10px 22px' }}>
          <LogoMark size={34} /><span className="logo-word" style={{ fontSize: 18 }}>PawConnect</span><span className="logo-ai" style={{ fontSize: 10, padding: '3px 5px' }}>AI</span>
        </Link>
        <nav className="col" style={{ gap: 2 }} aria-label="Dashboard">
          {TABS.map(([k, i, l]) => {
            const on = tab === k
            return (
              <Link key={k} to={k === 'overview' ? '/dashboard' : `/dashboard/${k}`} aria-current={on ? 'page' : undefined} className="row" style={{ gap: 12, height: 44, padding: '0 12px', borderRadius: 12, background: on ? 'var(--peach)' : 'transparent', color: on ? 'var(--rust-dd)' : 'var(--ink)', fontSize: 14, fontWeight: 700 }}>
                <Icon name={i} size={21} fill={on} /><span className="grow">{l}</span>
                {!!badges[k] && <span className="count">{badges[k]}</span>}
              </Link>
            )
          })}
        </nav>
        {active && (
          <div style={{ marginTop: 20, padding: 16, borderRadius: 16, background: 'var(--forest)', color: 'var(--cream)' }}>
            <div style={{ fontSize: 14, fontWeight: 800, marginBottom: 4 }}>{active.pet.name} is still missing</div>
            <div style={{ fontSize: 12.5, lineHeight: 1.45, color: 'var(--sage)', marginBottom: 12 }}>Share a poster in {active.pet.city} neighbourhood groups to reach more people.</div>
            <Link to={`/pets/${active.pet.id}/poster`} className="btn btn-block" style={{ height: 36, borderRadius: 10, background: 'var(--cream)', color: 'var(--forest)', fontSize: 13 }}>Get printable poster</Link>
          </div>
        )}
        <div className="col" style={{ marginTop: 'auto', paddingTop: 20, gap: 4 }}>
          <Link to="/" className="row" style={{ gap: 12, height: 40, padding: '0 12px', borderRadius: 10, fontSize: 13, fontWeight: 600, color: 'var(--muted)' }}><Icon name="public" size={19} />Back to website</Link>
          <button onClick={signOut} className="row" style={{ gap: 12, height: 40, padding: '0 12px', border: 0, borderRadius: 10, background: 'none', fontSize: 13, fontWeight: 600, color: 'var(--muted)', textAlign: 'left' }}><Icon name="logout" size={19} />Sign out</button>
        </div>
      </aside>

      <main style={{ flex: '999 1 600px', minWidth: 0 }}>
        <div style={{ position: 'sticky', top: 0, zIndex: 20, background: 'rgba(255,249,242,.92)', backdropFilter: 'blur(12px)', borderBottom: '1px solid var(--line)' }}>
          <div className="row" style={{ padding: '0 clamp(16px,3vw,36px)', height: 72, gap: 14 }}>
            <h1 className="grow" style={{ fontSize: 20, fontWeight: 800, letterSpacing: '-.02em', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{TITLES[tab]}</h1>
            <Link to="/pets" className="btn btn-outline hide-sm" style={{ height: 42, color: 'var(--muted)', fontWeight: 500 }}><Icon name="search" />Search pets</Link>
            <Link to="/report" className="btn btn-primary" style={{ height: 42 }}><Icon name="add" />New report</Link>
            <span className="avatar" style={{ width: 42, height: 42, borderRadius: 12, fontSize: 13 }}>{initials(user.name)}</span>
          </div>
        </div>

        <div className="col" style={{ padding: 'clamp(20px,3vw,36px)', gap: 24 }}>
          {isError && <div className="alert"><Icon name="error" />Couldn’t load your dashboard. Try again in a moment.</div>}
          {!d && !isError && <><Skeleton height={150} /><Skeleton height={300} /></>}

          {d && tab === 'overview' && (
            <>
              <div className="wrap-row" style={{ gap: 20, alignItems: 'stretch' }}>
                <div className="card wrap-row" style={{ flex: '2 1 460px', minWidth: 0, gap: 20, alignItems: 'center', padding: 22, borderRadius: 24 }}>
                  {active && <img src={img(active.pet.photos[0], 300, 300)} alt={active.pet.name} style={{ width: 108, height: 108, borderRadius: 20, objectFit: 'cover' }} />}
                  <div style={{ flex: '1 1 260px', minWidth: 0 }}>
                    <div style={{ fontSize: 26, fontWeight: 800, letterSpacing: '-.03em', marginBottom: 6 }}>{greeting()}, {user.name.split(' ')[0]}</div>
                    {active ? (
                      <>
                        <div style={{ fontSize: 15, lineHeight: 1.55, color: 'var(--body)' }}>
                          {active.pet.name} has been missing for {active.pet.days_ago} day{active.pet.days_ago === 1 ? '' : 's'}. There {newMatches === 1 ? 'is' : 'are'} <strong style={{ color: 'var(--ink)' }}>{newMatches} new possible match{newMatches === 1 ? '' : 'es'}</strong> and <strong style={{ color: 'var(--ink)' }}>{active.sightings} sighting{active.sightings === 1 ? '' : 's'}</strong> near {active.pet.city}.
                        </div>
                        <div className="wrap-row" style={{ gap: 8, marginTop: 14 }}>
                          <Link to={`/match?pet=${active.pet.id}`} className="btn btn-teal btn-md"><Icon name="auto_awesome" size={18} />Review matches</Link>
                          <Link to={`/pets/${active.pet.id}`} className="btn btn-outline btn-md">See sightings</Link>
                        </div>
                      </>
                    ) : <div style={{ fontSize: 15, lineHeight: 1.55, color: 'var(--body)' }}>Your reports, matches and adoption applications all live here. Start by reporting a pet or browsing animals waiting for a home.</div>}
                  </div>
                </div>
                <div className="card" style={{ flex: '1 1 280px', minWidth: 0, padding: 20, borderRadius: 24 }}>
                  <div style={{ fontSize: 15, fontWeight: 800, marginBottom: 12 }}>Quick actions</div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                    {([['/report?kind=lost', 'campaign', 'Report lost pet', '#C9522B'], ['/report?kind=found', 'volunteer_activism', 'Report found pet', '#176B63'], ['/match', 'image_search', 'Run AI match', '#176B63'], ['/adopt', 'favorite', 'Browse adoption', '#6B500D']] as const).map(([to, i, l, c]) => (
                      <Link key={to} to={to} className="col" style={{ alignItems: 'flex-start', gap: 8, padding: 12, borderRadius: 14, border: '1px solid var(--line)', background: 'var(--cream)', color: 'var(--ink)' }}>
                        <Icon name={i} size={22} style={{ color: c }} /><span style={{ fontSize: 13, fontWeight: 700 }}>{l}</span>
                      </Link>
                    ))}
                  </div>
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(180px,1fr))', gap: 14 }}>
                {([
                  ['list_alt', 'My listings', d.reports.filter((r) => !r.pet.resolved).length, active ? `${active.pet.name} · active for ${active.pet.days_ago} days` : 'No active reports', '#FBE1D5', '#9E3C19', 'listings'],
                  ['assignment', 'Adoption applications', d.applications.length, `${d.applications.filter((a) => a.stage < 3).length} in progress`, '#F4E8CC', '#6B500D', 'applications'],
                  ['favorite', 'Saved pets', d.saved.length, 'Across adoption & reports', '#FCE8DB', '#C9522B', 'saved'],
                  ['auto_awesome', 'Possible matches', d.matches.length, `${d.matches.filter((m) => m.score >= 60).length} above 60% similarity`, '#D7ECE8', '#115850', 'matches'],
                  ['pets', 'Lost / found reports', d.reports.length, `${d.reports.filter((r) => !r.pet.resolved).length} active · ${d.reports.filter((r) => r.pet.resolved).length} reunited`, '#E6ECEF', '#202A32', 'listings'],
                ] as const).map(([icon, label, value, sub, ib, ic, k]) => (
                  <Link key={label} to={`/dashboard/${k}`} className="card col" style={{ alignItems: 'flex-start', gap: 10, padding: 18, color: 'var(--ink)' }}>
                    <span style={{ width: 38, height: 38, borderRadius: 11, background: ib, color: ic, display: 'grid', placeItems: 'center' }}><Icon name={icon} size={21} /></span>
                    <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--muted)' }}>{label}</span>
                    <span style={{ fontSize: 32, fontWeight: 800, letterSpacing: '-.03em', lineHeight: 1 }}>{value}</span>
                    <span style={{ fontSize: 12.5, color: 'var(--muted)' }}>{sub}</span>
                  </Link>
                ))}
              </div>
            </>
          )}

          {d && tab !== 'saved' && tab !== 'settings' && (
            <div style={{ display: 'grid', gridTemplateColumns: tab === 'overview' ? 'repeat(auto-fit,minmax(min(100%,400px),1fr))' : 'minmax(0,1fr)', gap: 20, alignItems: 'start' }}>
              {show('matches') && <MatchesPanel d={d} />}
              {show('notifications') && <NotificationsPanel d={d} />}
              {show('applications') && (
                <Panel title="Adoption applications" action={<Link to="/adopt" className="btn-link" style={{ fontSize: 13 }}>Find more pets</Link>}>
                  {d.applications.length ? <div className="col" style={{ gap: 18 }}>{d.applications.map((a) => <AppCard key={a.id} a={a} />)}</div>
                    : <div style={{ fontSize: 14, color: 'var(--muted)' }}>No applications yet. <Link to="/adopt">Meet the animals waiting for a home</Link>.</div>}
                </Panel>
              )}
              {tab === 'applications' && user.shelter_id && <IncomingApplications />}
              {show('listings') && <ReportsPanel d={d} />}
              {tab === 'overview' && (
                <Panel title="Recent activity">
                  {d.activity.length ? (
                    <div className="col">
                      {d.activity.map((a) => (
                        <div key={a.id} className="row" style={{ gap: 12, alignItems: 'stretch' }}>
                          <div className="col" style={{ alignItems: 'center' }}>
                            <span style={{ width: 30, height: 30, borderRadius: '50%', background: 'var(--teal-soft)', color: 'var(--teal)', display: 'grid', placeItems: 'center', flexShrink: 0 }}><Icon name={a.icon} size={16} /></span>
                            <span style={{ flex: 1, width: 1.5, background: 'var(--line)', minHeight: 12 }} />
                          </div>
                          <div style={{ padding: '4px 0 16px', minWidth: 0 }}><div style={{ fontSize: 13.5, fontWeight: 600, lineHeight: 1.4 }}>{a.text}</div><div style={{ fontSize: 12, color: 'var(--muted-2)', marginTop: 2 }}>{relTime(a.created_at)}</div></div>
                        </div>
                      ))}
                    </div>
                  ) : <div style={{ fontSize: 14, color: 'var(--muted)' }}>Nothing here yet.</div>}
                </Panel>
              )}
            </div>
          )}

          {d && tab === 'saved' && (d.saved.length ? <PetGrid pets={d.saved} min={230} gap={18} /> : (
            <div className="empty" style={{ padding: '48px 24px', borderRadius: 22 }}>
              <Icon name="favorite" size={40} style={{ color: 'var(--rust)' }} />
              <div style={{ fontSize: 19, fontWeight: 800 }}>No saved pets yet</div>
              <div style={{ fontSize: 14, color: 'var(--muted)' }}>Tap the heart on any listing to keep it here.</div>
              <Link to="/pets" className="btn btn-dark">Browse pets</Link>
            </div>
          ))}
          {tab === 'settings' && <SettingsPanel user={user} />}
        </div>
      </main>
    </div>
  )
}
