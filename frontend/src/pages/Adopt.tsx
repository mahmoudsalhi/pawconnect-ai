import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { FavButton } from '../components/PetCard'
import { PetModal } from '../components/PetModals'
import { Empty, Icon, Pills, Skeleton } from '../components/ui'
import type { Application, Pet } from '../lib/api'
import { useAuth } from '../lib/auth'
import { fmtDate, img, SPEED, speedColor } from '../lib/format'
import { useInsights, useMyApplications, usePets, useRequireAuth, useShelters } from '../lib/queries'

const U = 'https://images.unsplash.com/photo-'

export function StageTrack({ app }: { app: Application }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,minmax(0,1fr))', gap: 6 }}>
      {['Submitted', 'Under review', 'Home visit', 'Decision'].map((l, i) => (
        <div key={l}>
          <div className={`stage-bar${i < app.stage ? ' done' : i === app.stage ? ' cur' : ''}`} />
          <div style={{ fontSize: 12, fontWeight: 700, color: i <= app.stage ? 'var(--ink)' : 'var(--muted)', marginTop: 6 }}>{l}</div>
        </div>
      ))}
    </div>
  )
}

function AdoptCard({ p, applied, onApply }: { p: Pet; applied: boolean; onApply: () => void }) {
  return (
    <Link to={`/pets/${p.id}`} className="pet-card" style={{ borderRadius: 20, boxShadow: 'none' }}>
      <div className="pet-card-media" style={{ aspectRatio: '4/3' }}>
        <img src={img(p.photos[0], 640, 480)} alt={p.name} loading="lazy" />
        <FavButton pet={p} />
        <span style={{ position: 'absolute', left: 12, bottom: 12, padding: '4px 9px', borderRadius: 8, background: 'rgba(32,42,50,.82)', color: '#fff', fontSize: 12, fontWeight: 600 }}>{p.shelter?.name ?? p.contact_name}</span>
      </div>
      <div className="col" style={{ padding: '16px 18px 18px', gap: 10, flex: 1 }}>
        <div className="row between" style={{ alignItems: 'baseline', gap: 8 }}>
          <span style={{ fontSize: 20, fontWeight: 800, letterSpacing: '-.01em' }}>{p.name}</span>
          <span style={{ fontSize: 13, color: 'var(--muted)' }}>{p.sex} · {p.age_label}</span>
        </div>
        <div style={{ fontSize: 14, color: 'var(--body)' }}>{p.breed} · {p.size}</div>
        <div className="wrap-row" style={{ gap: 6 }}>{p.traits.map((t) => <span key={t} className="trait">{t}</span>)}</div>
        <div className="row between" style={{ gap: 8, marginTop: 'auto', paddingTop: 8 }}>
          <span style={{ fontSize: 13, color: 'var(--muted)' }}>Fee: {p.fee ?? '—'}</span>
          {applied
            ? <span className="row" style={{ gap: 5, height: 38, padding: '0 12px', borderRadius: 10, background: '#D7ECE8', color: 'var(--teal-dd)', fontSize: 13, fontWeight: 700 }}><Icon name="check" size={17} />Applied</span>
            : <button className="btn btn-primary" style={{ height: 38, padding: '0 14px', borderRadius: 10, fontSize: 13 }} onClick={(e) => { e.preventDefault(); e.stopPropagation(); onApply() }}>Apply to adopt</button>}
        </div>
      </div>
    </Link>
  )
}

export default function Adopt() {
  const { user } = useAuth()
  const requireAuth = useRequireAuth()
  const [f, setF] = useState({ q: '', species: 'any', age: 'any', size: 'any', shelter: 'any' })
  const [dq, setDq] = useState('')
  useEffect(() => {
    const t = window.setTimeout(() => setDq(f.q), 300)
    return () => window.clearTimeout(t)
  }, [f.q])
  const list = usePets({ status: 'adopt', page_size: 100, q: dq, species: f.species, age: f.age, size: f.size, shelter: f.shelter })
  const shelters = useShelters()
  const apps = useMyApplications()
  const [insight, setInsight] = useState('carthage')
  const insights = useInsights(insight)
  const [applyTo, setApplyTo] = useState<Pet | null>(null)
  const appliedIds = new Set((apps.data ?? []).map((a) => a.pet_id))
  const set = (k: keyof typeof f, v: string) => setF((s) => ({ ...s, [k]: v }))

  return (
    <main>
      <section style={{ background: 'var(--peach)' }}>
        <div className="wrap" style={{ paddingTop: 'clamp(40px,5vw,72px)', paddingBottom: 'clamp(40px,5vw,72px)', display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(min(100%,460px),1fr))', gap: 40, alignItems: 'center' }}>
          <div className="col" style={{ gap: 22 }}>
            <div className="eyebrow" style={{ color: 'var(--rust-dd)', margin: 0 }}>Adoption Center</div>
            <h1 style={{ fontSize: 'clamp(38px,4.6vw,64px)', fontWeight: 800, letterSpacing: '-.04em', lineHeight: 1, textWrap: 'balance' }}>Give a rescued animal their next chapter.</h1>
            <p style={{ maxWidth: 520, fontSize: 17, lineHeight: 1.6, color: 'var(--body)' }}>Every animal here is cared for by a partner shelter in Tunisia. Shelters write the profiles, answer your questions and guide you through each step.</p>
            <div className="wrap-row" style={{ gap: 6, padding: 6, background: '#fff', borderRadius: 16, maxWidth: 560, boxShadow: '0 20px 40px -30px rgba(32,42,50,.4)' }}>
              <label className="row" style={{ flex: '1 1 220px', gap: 10, padding: '0 12px', height: 50, minWidth: 0 }}>
                <Icon name="search" size={21} style={{ color: 'var(--muted)' }} />
                <input value={f.q} onChange={(e) => set('q', e.target.value)} aria-label="Search animals" placeholder="Name, breed or trait, e.g. good with kids" style={{ flex: 1, minWidth: 0, border: 0, outline: 'none', background: 'transparent', fontSize: 15 }} />
              </label>
              <select value={f.shelter} onChange={(e) => set('shelter', e.target.value)} aria-label="Shelter" style={{ flex: '1 1 170px', height: 50, padding: '0 12px', border: 0, borderRadius: 12, background: 'var(--cream)', fontSize: 14, fontWeight: 600, outline: 'none' }}>
                <option value="any">All shelters</option>
                {shelters.data?.map((s) => <option key={s.slug} value={s.slug}>{s.name}</option>)}
              </select>
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gridTemplateRows: '200px 200px', gap: 12 }}>
            <div style={{ gridRow: '1/3', borderRadius: 24, overflow: 'hidden', background: 'var(--sand)' }}><img src={img(U + '1576201836106-db1758fd1c97', 900)} alt="A puppy running" style={{ width: '100%', height: '100%', objectFit: 'cover' }} /></div>
            <div style={{ borderRadius: 24, overflow: 'hidden', background: 'var(--sand)' }}><img src={img(U + '1526336024174-e58f5cdd8e13', 600)} alt="A long-haired cat" style={{ width: '100%', height: '100%', objectFit: 'cover' }} /></div>
            <div style={{ borderRadius: 24, overflow: 'hidden', background: 'var(--sand)' }}><img src={img(U + '1517849845537-4d257902454a', 600)} alt="A senior pug" style={{ width: '100%', height: '100%', objectFit: 'cover' }} /></div>
          </div>
        </div>
      </section>

      <section className="wrap" style={{ paddingTop: 48 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(220px,1fr))', gap: 14 }}>
          {[['01', 'Browse and save', 'Read the shelter’s notes on personality, health and daily needs.'], ['02', 'Apply online', 'A short form about your home and routine. Takes about five minutes.'], ['03', 'Meet and home visit', 'The shelter arranges a meeting and a short visit to your home.']].map(([n, t, d]) => (
            <div key={n} className="card" style={{ padding: 20 }}><div className="mono" style={{ fontSize: 12, color: 'var(--rust)', marginBottom: 12 }}>{n}</div><div style={{ fontSize: 16, fontWeight: 800, marginBottom: 6 }}>{t}</div><div style={{ fontSize: 14, lineHeight: 1.5, color: 'var(--muted)' }}>{d}</div></div>
          ))}
          <div style={{ padding: 20, borderRadius: 20, background: 'var(--ink)', color: 'var(--cream)' }}><div className="mono" style={{ fontSize: 12, color: '#F4A988', marginBottom: 12 }}>TIP</div><div style={{ fontSize: 16, fontWeight: 800, marginBottom: 6 }}>Before you apply</div><div style={{ fontSize: 14, lineHeight: 1.5, color: '#C9CFD3' }}>Check your building allows pets and plan who cares for them when you travel.</div></div>
        </div>
      </section>

      {user && !!apps.data?.length && (
        <section className="wrap" style={{ paddingTop: 40 }}>
          <div className="panel" style={{ padding: 22 }}>
            <div className="wrap-row between" style={{ alignItems: 'center', gap: 12, marginBottom: 16 }}>
              <h2 style={{ fontSize: 20, fontWeight: 800 }}>Your applications</h2>
              <Link to="/dashboard/applications" className="btn-link" style={{ fontSize: 14 }}>Open dashboard</Link>
            </div>
            <div className="col" style={{ gap: 14 }}>
              {apps.data.map((a) => (
                <div key={a.id} className="wrap-row" style={{ gap: 18, alignItems: 'center' }}>
                  <Link to={`/pets/${a.pet_id}`} className="row" style={{ gap: 12, flex: '1 1 240px', minWidth: 0, color: 'var(--ink)' }}>
                    <img src={img(a.pet_photo, 160, 160)} alt="" style={{ width: 56, height: 56, borderRadius: 14, objectFit: 'cover' }} />
                    <div style={{ minWidth: 0 }}><div style={{ fontSize: 16, fontWeight: 800 }}>{a.pet_name} <span className="mono" style={{ fontSize: 12, fontWeight: 500, color: 'var(--muted)' }}>{a.code}</span></div><div style={{ fontSize: 13, color: 'var(--muted)' }}>{a.shelter_name} · sent {fmtDate(a.created_at)}</div></div>
                  </Link>
                  <div style={{ flex: '2 1 380px', minWidth: 0 }}>
                    <StageTrack app={a} />
                    <div style={{ fontSize: 13, color: 'var(--body)', marginTop: 6 }}>{a.next_step}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}

      <section className="wrap" style={{ paddingTop: 48 }}>
        <div className="wrap-row between" style={{ alignItems: 'center', gap: 16, marginBottom: 22 }}>
          <h2 style={{ fontSize: 'clamp(26px,2.6vw,34px)', fontWeight: 800, letterSpacing: '-.03em' }}>{list.data ? list.data.total : '–'} animals waiting</h2>
          <div className="wrap-row" style={{ gap: 16 }}>
            <Pills value={f.species} onChange={(v) => set('species', v)} options={[{ value: 'any', label: 'All' }, { value: 'Dog', label: 'Dogs' }, { value: 'Cat', label: 'Cats' }]} />
            <Pills value={f.age} onChange={(v) => set('age', v)} options={[{ value: 'any', label: 'Any age' }, { value: 'young', label: 'Puppy / kitten' }, { value: 'adult', label: 'Adult' }, { value: 'senior', label: 'Senior' }]} />
            <Pills value={f.size} onChange={(v) => set('size', v)} options={[{ value: 'any', label: 'Any size' }, { value: 'Small', label: 'Small' }, { value: 'Medium', label: 'Medium' }, { value: 'Large', label: 'Large' }]} />
          </div>
        </div>
        {!list.data && <div className="pet-grid" style={{ gridTemplateColumns: 'repeat(auto-fill,minmax(270px,1fr))', gap: 22 }}>{[0, 1, 2].map((i) => <Skeleton key={i} height={420} />)}</div>}
        {list.data && list.data.total > 0 && (
          <div className="pet-grid" style={{ gridTemplateColumns: 'repeat(auto-fill,minmax(270px,1fr))', gap: 22 }}>
            {list.data.items.map((p) => <AdoptCard key={p.id} p={p} applied={appliedIds.has(p.id)} onApply={() => requireAuth('Sign in to apply for adoption.') && setApplyTo(p)} />)}
          </div>
        )}
        {list.data && list.data.total === 0 && (
          <Empty icon="pets" title="No animals match these filters" actions={<button className="btn btn-dark" onClick={() => { setF({ q: '', species: 'any', age: 'any', size: 'any', shelter: 'any' }) }}>Clear filters</button>} />
        )}
      </section>

      <section className="wrap" style={{ paddingTop: 72 }}>
        <h2 style={{ margin: '0 0 22px', fontSize: 'clamp(26px,2.6vw,34px)', fontWeight: 800, letterSpacing: '-.03em' }}>Partner shelters</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(270px,1fr))', gap: 18 }}>
          {shelters.data?.map((s) => (
            <div key={s.slug} className="card col" style={{ overflow: 'hidden' }}>
              <div style={{ height: 140, background: 'var(--sand)' }}><img src={img(s.photo_url, 700, 400)} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} /></div>
              <div className="col" style={{ padding: '16px 18px 18px', gap: 8, flex: 1 }}>
                <div style={{ fontSize: 16, fontWeight: 800 }}>{s.name}</div>
                <div className="row" style={{ gap: 4, fontSize: 13, color: 'var(--muted)' }}><Icon name="location_on" size={16} />{s.city} · Since {s.since}</div>
                <div style={{ fontSize: 13, color: 'var(--body)' }}>{s.focus} · {s.animal_count} animals in care</div>
                <div className="row" style={{ gap: 5, fontSize: 12.5, color: 'var(--teal)', fontWeight: 600 }}><Icon name="schedule" size={16} />{s.reply_note}</div>
                <button className="btn btn-outline" style={{ marginTop: 'auto', height: 40, borderRadius: 10, fontSize: 13 }} onClick={() => { set('shelter', s.slug); window.scrollTo({ top: 0, behavior: 'smooth' }) }}>View animals · {s.listed} on PawConnect</button>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="wrap" style={{ paddingTop: 72, paddingBottom: 96 }}>
        <div className="card" style={{ borderRadius: 28, overflow: 'hidden' }}>
          <div className="wrap-row between" style={{ padding: 'clamp(22px,3vw,36px)', alignItems: 'flex-end', gap: 20, borderBottom: '1px solid var(--line)' }}>
            <div style={{ maxWidth: 620 }}>
              <div className="wrap-row" style={{ alignItems: 'center', gap: 8, marginBottom: 10 }}><span className="eyebrow teal" style={{ margin: 0 }}>For shelters</span><span className="demo-tag">M3 · XGBOOST · PETFINDER</span></div>
              <h2 style={{ margin: '0 0 8px', fontSize: 'clamp(24px,2.4vw,32px)', fontWeight: 800, letterSpacing: '-.03em' }}>Adoption speed insights</h2>
              <p style={{ fontSize: 15, lineHeight: 1.55, color: 'var(--muted)' }}>Predicted time to adoption for each listing, using the PetFinder speed classes. Predictions come from listing features such as age, photo count, description length and fee, and help you decide which profiles to improve first.</p>
            </div>
            <div className="wrap-row" style={{ gap: 6 }}>
              {shelters.data?.map((s) => (
                <button key={s.slug} onClick={() => setInsight(s.slug)} aria-pressed={insight === s.slug} style={{ height: 36, padding: '0 12px', borderRadius: 10, border: `1px solid ${insight === s.slug ? '#202A32' : '#E5D6C6'}`, background: insight === s.slug ? '#202A32' : '#fff', color: insight === s.slug ? '#fff' : 'var(--ink)', fontSize: 13, fontWeight: 700 }}>
                  {s.name.replace('Refuge ', '').replace(' Collective', '')}
                </button>
              ))}
            </div>
          </div>
          <div className="col">
            {insights.data?.map((r) => (
              <div key={r.pet_id} className="wrap-row" style={{ gap: 18, alignItems: 'center', padding: '16px clamp(22px,3vw,36px)', borderBottom: '1px solid var(--line-3)' }}>
                <Link to={`/pets/${r.pet_id}`} className="row" style={{ gap: 12, flex: '1 1 220px', minWidth: 0, color: 'var(--ink)' }}>
                  <img src={img(r.photo, 120, 120)} alt="" style={{ width: 48, height: 48, borderRadius: 12, objectFit: 'cover' }} />
                  <div style={{ minWidth: 0 }}><div style={{ fontSize: 15, fontWeight: 800 }}>{r.name}</div><div style={{ fontSize: 12.5, color: 'var(--muted)' }}>{r.species} · {r.age_label} · listed {r.days_listed} days</div></div>
                </Link>
                <div style={{ flex: '1 1 240px', minWidth: 0 }}>
                  <div className="row between" style={{ fontSize: 13, marginBottom: 6 }}><span style={{ fontWeight: 800, color: speedColor(r.speed_class) }}>{SPEED[r.speed_class]}</span><span className="mono muted" style={{ fontSize: 11.5 }}>class {r.speed_class}</span></div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: 3 }}>{[0, 1, 2, 3, 4].map((i) => <span key={i} style={{ height: 8, borderRadius: 3, background: i === r.speed_class ? speedColor(r.speed_class) : '#EFE6DB' }} />)}</div>
                </div>
                <div className="wrap-row" style={{ flex: '1.4 1 260px', gap: 6 }}>
                  {r.tips.map((t) => <span key={t} className="row" style={{ gap: 5, padding: '5px 10px', borderRadius: 999, background: 'var(--cream)', border: '1px solid var(--line)', fontSize: 12.5, fontWeight: 600 }}><Icon name="tips_and_updates" size={15} style={{ color: 'var(--rust)' }} />{t}</span>)}
                </div>
              </div>
            ))}
            {insights.data?.length === 0 && <div style={{ padding: '24px clamp(22px,3vw,36px)', color: 'var(--muted)', fontSize: 14 }}>No open listings for this shelter.</div>}
          </div>
          <div style={{ padding: '14px clamp(22px,3vw,36px)', fontSize: 12.5, color: 'var(--muted)', background: 'var(--cream-2)' }}>Classes: 0 same day · 1 within a week · 2 within a month · 3 within 1–3 months · 4 not adopted after 100 days. Demo predictions; the model has not been evaluated yet.</div>
        </div>
      </section>
      {applyTo && <PetModal kind="apply" pet={applyTo} onClose={() => setApplyTo(null)} />}
    </main>
  )
}
