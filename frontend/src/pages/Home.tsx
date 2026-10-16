import { useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { PetGrid } from '../components/PetCard'
import { Icon, SectionHead, Seg, Skeleton } from '../components/ui'
import { ago, GOVERNORATES, img } from '../lib/format'
import { usePets, useStats } from '../lib/queries'

const HOW = {
  lost: [
    ['campaign', 'Report your pet', 'Add photos, a description and where you last saw them. It takes about three minutes.'],
    ['pets', 'AI suggests the breed', 'The breed model proposes a breed and the text model suggests tags. You can change anything.'],
    ['public', 'Publish your listing', 'Your report appears in search and on the map for people nearby.'],
    ['image_search', 'AI scans found reports', 'New found reports are compared with your photos and ranked by visual similarity.'],
    ['call', 'Contact the finder', 'Review each possible match, check the markings yourself, then message the finder.'],
  ],
  found: [
    ['add_a_photo', 'Report a found animal', 'Take a photo and mark where you found the animal on the map.'],
    ['image_search', 'AI compares lost reports', 'Your photo is compared with open lost reports within the radius you choose.'],
    ['fact_check', 'Review suggested matches', 'Each result shows a similarity score. Compare collars, markings and size.'],
    ['chat', 'Contact the owner', 'Message the owner through PawConnect. Your number stays private until you share it.'],
    ['home', 'Hand over safely', 'Ask for photos or vet records and meet in a public place.'],
  ],
  adopt: [
    ['travel_explore', 'Browse the Adoption Center', 'Filter by species, age, size and shelter across Tunisia.'],
    ['pets', 'Open a pet profile', 'Read the shelter’s notes on personality, health and daily needs.'],
    ['edit_note', 'Apply in minutes', 'Tell the shelter about your home and routine. One form, no paperwork.'],
    ['dashboard', 'Track your application', 'Follow each step from review to home visit in your dashboard.'],
    ['favorite', 'Welcome them home', 'Meet your new companion and finalise the adoption with the shelter.'],
  ],
} as const
type HowTab = keyof typeof HOW
const HOW_COLORS: Record<HowTab, [string, string]> = { lost: ['#FBE1D5', '#9E3C19'], found: ['#D7ECE8', '#115850'], adopt: ['#F4E8CC', '#6B500D'] }
const HOW_CTA: Record<HowTab, [string, string]> = {
  lost: ['Report a lost pet', '/report?kind=lost'],
  found: ['Report a found animal', '/report?kind=found'],
  adopt: ['Open the Adoption Center', '/adopt'],
}

const MODELS = [
  { code: 'M1 · CNN · OXFORD-IIIT PET', title: 'Breed identification', text: 'Suggests the most likely breed from a photo, with a confidence score you can accept or override.' },
  { code: 'M2 · SIAMESE CNN · PETFINDER PAIRS', title: 'Lost & found photo matching', text: 'Compares a new photo with every open report and ranks visually similar animals as possible matches.' },
  { code: 'M3 · XGBOOST · PETFINDER ADOPTION', title: 'Adoption speed insights', text: 'Helps shelters see which listings may wait longer, based on features like age, photo count and description length.' },
  { code: 'M4 · TF-IDF + SVM · PETFINDER TEXT', title: 'Listing text tagging', text: 'Reads a description and suggests structured attributes and tags that make listings easier to search.' },
  { code: 'M5 · CNN · PETFINDER PAWPULARITY', title: 'Photo engagement score', text: 'Estimates how engaging a listing photo is and offers practical tips for taking a better one.' },
  { code: 'M6 · U-NET · OXFORD-IIIT PET', title: 'Background removal', text: 'Separates the animal from its background for cleaner listings and more focused matching.' },
]
const U = 'https://images.unsplash.com/photo-'

function MiniBar({ label, pct, dim }: { label: string; pct: number; dim?: boolean }) {
  return (
    <div>
      <div className="row between" style={{ color: dim ? '#A9C4BF' : undefined }}>
        <span>{label}</span>
        <span className="mono" style={{ color: dim ? undefined : 'var(--mint-2)' }}>{pct}%</span>
      </div>
      <div style={{ height: 5, borderRadius: 3, background: 'rgba(255,255,255,.1)', marginTop: 4 }}>
        <div style={{ width: `${pct}%`, height: '100%', borderRadius: 3, background: '#5CC8B8' }} />
      </div>
    </div>
  )
}

function ModelVisual({ i }: { i: number }) {
  const box = { height: 156, borderRadius: 14, background: 'var(--forest-3)' }
  const hl = { background: 'rgba(92,200,184,.25)', color: 'var(--cream)', borderRadius: 4, padding: '0 3px' }
  switch (i) {
    case 0:
      return (
        <div className="row" style={{ ...box, padding: 18, gap: 16 }}>
          <img src={img(U + '1517849845537-4d257902454a', 240, 240)} alt="" style={{ width: 96, height: 96, borderRadius: 12, objectFit: 'cover', flexShrink: 0 }} />
          <div className="col grow" style={{ gap: 9, fontSize: 12 }}>
            <MiniBar label="Pug" pct={88} /><MiniBar label="Boxer" pct={4} dim /><MiniBar label="Japanese chin" pct={3} dim />
          </div>
        </div>
      )
    case 1:
      return (
        <div className="row" style={{ ...box, justifyContent: 'center', gap: 14 }}>
          <img src={img(U + '1573865526739-10659fec78a5', 240, 240)} alt="" style={{ width: 92, height: 92, borderRadius: '50%', objectFit: 'cover', border: '2px solid #5CC8B8' }} />
          <div className="col" style={{ alignItems: 'center', gap: 4 }}>
            <span className="mono" style={{ fontSize: 22, fontWeight: 600, color: 'var(--mint-2)' }}>87%</span>
            <div style={{ width: 70, borderTop: '2px dashed rgba(127,217,203,.6)' }} />
            <span style={{ fontSize: 11, color: '#A9C4BF' }}>similarity</span>
          </div>
          <img src={img(U + '1511044568932-338cba0ad803', 240, 240)} alt="" style={{ width: 92, height: 92, borderRadius: '50%', objectFit: 'cover', border: '2px solid #5CC8B8' }} />
        </div>
      )
    case 2:
      return (
        <div className="col" style={{ ...box, padding: 20, justifyContent: 'center', gap: 12 }}>
          <div className="row between" style={{ fontSize: 12, color: '#A9C4BF' }}><span>Predicted adoption speed</span><span className="mono" style={{ color: 'var(--mint-2)' }}>class 2</span></div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: 4 }}>
            {[0, 1, 2, 3, 4].map((k) => <div key={k} style={{ height: 28, borderRadius: 6, background: k === 2 ? '#5CC8B8' : 'rgba(255,255,255,.08)' }} />)}
          </div>
          <div className="mono" style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: 4, fontSize: 10, color: '#A9C4BF', textAlign: 'center' }}>
            <span>0 d</span><span>1–7</span><span style={{ color: 'var(--cream)' }}>8–30</span><span>31–90</span><span>100+</span>
          </div>
        </div>
      )
    case 3:
      return (
        <div className="col" style={{ ...box, padding: 18, gap: 12, justifyContent: 'center' }}>
          <div style={{ fontSize: 13, lineHeight: 1.5, color: 'var(--sage)' }}>“…very <span style={hl}>friendly</span>, <span style={hl}>neutered</span>, wears a <span style={hl}>blue collar</span>…”</div>
          <div className="wrap-row" style={{ gap: 6 }}>
            {['Friendly · 0.86', 'Neutered · 0.88', 'Wears a collar · 0.91'].map((t) => (
              <span key={t} style={{ padding: '5px 9px', borderRadius: 999, background: '#5CC8B8', color: 'var(--forest-3)', fontSize: 12, fontWeight: 700 }}>{t}</span>
            ))}
          </div>
        </div>
      )
    case 4:
      return (
        <div className="row" style={{ ...box, padding: 18, gap: 18 }}>
          <div className="ring" style={{ width: 96, height: 96, background: 'conic-gradient(#F4A988 0deg 281deg, rgba(255,255,255,.1) 281deg)' }}>
            <div style={{ width: 76, height: 76, background: 'var(--forest-3)', color: 'var(--cream)', fontSize: 24 }}>78</div>
          </div>
          <div style={{ fontSize: 13, lineHeight: 1.5, color: 'var(--sage)' }}>
            <div style={{ color: 'var(--cream)', fontWeight: 700, marginBottom: 4 }}>Engagement score</div>Get down to eye level and use daylight from a window.
          </div>
        </div>
      )
    default: {
      const src = img(U + '1507146426996-ef05306b995a', 400, 320)
      const mask = 'radial-gradient(ellipse 42% 50% at 50% 52%,#000 72%,transparent 74%)'
      return (
        <div style={{ ...box, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2, overflow: 'hidden' }}>
          <img src={src} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
          <div style={{ background: 'repeating-conic-gradient(#E9E2D8 0 25%, #FFFFFF 0 50%) 0 0/14px 14px' }}>
            <img src={src} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover', WebkitMaskImage: mask, maskImage: mask }} />
          </div>
        </div>
      )
    }
  }
}

export default function Home() {
  const navigate = useNavigate()
  const [q, setQ] = useState('')
  const [gov, setGov] = useState('all')
  const [species, setSpecies] = useState<'any' | 'Dog' | 'Cat'>('any')
  const [how, setHow] = useState<HowTab>('lost')
  const lost = usePets({ status: 'lost', page_size: 4 })
  const adopt = usePets({ status: 'adopt', page_size: 4 })
  const stats = useStats()
  const hero = lost.data?.items.find((p) => p.reward) ?? lost.data?.items[0]

  const search = (e: FormEvent) => {
    e.preventDefault()
    const p = new URLSearchParams()
    if (q.trim()) p.set('q', q.trim())
    if (gov !== 'all') p.set('gov', gov)
    if (species !== 'any') p.set('species', species)
    navigate(`/pets?${p}`)
  }

  return (
    <main>
      <section className="wrap" style={{ paddingTop: 'clamp(36px,5vw,72px)', paddingBottom: 'clamp(48px,6vw,88px)', display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(min(100%,500px),1fr))', gap: 'clamp(40px,5vw,72px)', alignItems: 'center' }}>
        <div className="col" style={{ gap: 28, minWidth: 0 }}>
          <div className="row" style={{ alignSelf: 'flex-start', gap: 10, padding: '7px 14px 7px 10px', borderRadius: 999, background: '#fff', border: '1px solid var(--line)', fontSize: 13, fontWeight: 600, color: 'var(--teal)' }}>
            <span className="pulse" /> AI-assisted lost &amp; found, built for Tunisia
          </div>
          <h1 className="h-display">Every Paw Deserves a <span style={{ color: 'var(--orange)' }}>Way Home.</span></h1>
          <p style={{ maxWidth: 540, fontSize: 'clamp(17px,1.4vw,20px)', lineHeight: 1.55, color: 'var(--body)', textWrap: 'pretty' }}>
            Report a missing or found animal in minutes. Our computer-vision models compare your photos with every open report across Tunisia and suggest possible matches, so families and finders can reach each other sooner.
          </p>
          <div className="wrap-row" style={{ gap: 12 }}>
            <Link to="/pets" className="btn btn-primary btn-xl glow">Find a Pet<Icon name="arrow_forward" /></Link>
            <Link to="/report?kind=lost" className="btn btn-outline btn-xl"><Icon name="campaign" style={{ color: 'var(--orange)' }} />Report a Lost Pet</Link>
          </div>
          <form onSubmit={search} className="wrap-row" style={{ alignItems: 'stretch', gap: 6, padding: 6, background: '#fff', border: '1px solid var(--line)', borderRadius: 18, boxShadow: '0 20px 44px -28px rgba(32,42,50,.35)', maxWidth: 640 }}>
            <label className="row" style={{ flex: '1 1 150px', gap: 10, padding: '0 12px', height: 52, minWidth: 0 }}>
              <Icon name="search" size={21} style={{ color: 'var(--muted)' }} />
              <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Name, breed or colour" aria-label="Search" style={{ flex: 1, minWidth: 0, border: 0, outline: 'none', background: 'transparent', fontSize: 15, fontWeight: 500 }} />
            </label>
            <label className="row" style={{ flex: '1 1 140px', gap: 8, padding: '0 12px', height: 52, borderRadius: 12, background: 'var(--cream)' }}>
              <Icon name="location_on" size={20} style={{ color: 'var(--teal)' }} />
              <select value={gov} onChange={(e) => setGov(e.target.value)} aria-label="Governorate" style={{ flex: 1, border: 0, outline: 'none', background: 'transparent', fontSize: 14, fontWeight: 600, minWidth: 0 }}>
                <option value="all">All of Tunisia</option>
                {GOVERNORATES.map((g) => <option key={g}>{g}</option>)}
              </select>
            </label>
            <Seg className="cream compact" style={{ flex: '0 1 auto', gap: 2 }} value={species} onChange={setSpecies} options={[{ value: 'any', label: 'Any' }, { value: 'Dog', label: 'Dogs' }, { value: 'Cat', label: 'Cats' }]} />
            <button type="submit" className="btn btn-teal" style={{ flex: '1 0 auto', height: 52, padding: '0 22px', fontSize: 15 }}>Search</button>
          </form>
        </div>

        <div style={{ position: 'relative', padding: '28px 0 28px 28px', minWidth: 0 }}>
          <div style={{ position: 'absolute', top: 0, right: -14, bottom: 60, left: 80, background: 'var(--peach)', borderRadius: 36 }} />
          <div style={{ position: 'relative', borderRadius: 28, overflow: 'hidden', height: 'clamp(440px,46vw,620px)', boxShadow: '0 44px 80px -44px rgba(32,42,50,.55)', background: 'var(--sand)' }}>
            {hero ? <img src={img(hero.photos[0], 1400)} alt={`${hero.name}, reported lost in ${hero.city}`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} /> : <Skeleton height="100%" />}
            <div style={{ position: 'absolute', inset: 'auto 0 0 0', height: '45%', background: 'linear-gradient(180deg,rgba(20,26,30,0),rgba(20,26,30,.62))' }} />
            {hero && (
              <div className="row between" style={{ position: 'absolute', left: 24, right: 24, bottom: 22, alignItems: 'flex-end', gap: 12, color: '#fff' }}>
                <div>
                  <span className="badge lost"><span className="d" />Lost · {ago(hero.days_ago).toLowerCase()}</span>
                  <div style={{ fontSize: 26, fontWeight: 800, letterSpacing: '-.02em', marginTop: 8 }}>{hero.name}, {hero.age_label}</div>
                  <div style={{ fontSize: 14, opacity: 0.92 }}>Last seen in {hero.city}</div>
                </div>
                <Link to={`/pets/${hero.id}`} className="btn" style={{ height: 44, flexShrink: 0 }}>View report</Link>
              </div>
            )}
          </div>
          {hero && hero.ai_breeds.length > 1 && (
            <div className="hide-sm" style={{ position: 'absolute', top: 70, left: 0, width: 228, background: '#fff', borderRadius: 18, padding: 16, boxShadow: '0 24px 50px -24px rgba(32,42,50,.45)', border: '1px solid var(--line)' }}>
              <div className="row between" style={{ marginBottom: 10 }}>
                <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--muted)' }}>Breed estimate</span>
                <span className="demo-tag" style={{ fontSize: 10 }}>DEMO · M1</span>
              </div>
              <div className="row between" style={{ alignItems: 'baseline' }}>
                <span style={{ fontSize: 16, fontWeight: 800 }}>{hero.ai_breeds[0][0]}</span>
                <span className="mono" style={{ fontSize: 15, fontWeight: 600, color: 'var(--teal)' }}>{hero.ai_breeds[0][1]}%</span>
              </div>
              <div className="bar" style={{ margin: '6px 0 10px' }}><div style={{ width: `${hero.ai_breeds[0][1]}%` }} /></div>
              <div className="row between" style={{ fontSize: 13, color: 'var(--muted)' }}>
                <span>{hero.ai_breeds[1][0]}</span><span className="mono">{hero.ai_breeds[1][1]}%</span>
              </div>
            </div>
          )}
          <Link to={hero ? `/match?pet=${hero.id}` : '/match'} className="row hide-sm" style={{ position: 'absolute', right: -6, bottom: 118, width: 270, background: '#fff', borderRadius: 18, padding: 14, boxShadow: '0 24px 50px -24px rgba(32,42,50,.45)', border: '1px solid var(--line)', gap: 12, color: 'var(--ink)' }}>
            <div className="row" style={{ flexShrink: 0 }}>
              <img src={img(hero?.photos[0] ?? U + '1543466835-00a7907e9de1', 160, 160)} alt="" style={{ width: 44, height: 44, borderRadius: '50%', objectFit: 'cover', border: '3px solid #fff' }} />
              <img src={img(U + '1561037404-61cd46aa615b', 160, 160)} alt="" style={{ width: 44, height: 44, borderRadius: '50%', objectFit: 'cover', border: '3px solid #fff', marginLeft: -14 }} />
            </div>
            <div className="grow">
              <div style={{ fontSize: 13, fontWeight: 800 }}>Possible match</div>
              <div style={{ fontSize: 12, color: 'var(--muted)', lineHeight: 1.35 }}>Run AI matching on this report</div>
            </div>
            <Icon name="image_search" size={22} style={{ color: 'var(--teal)' }} />
          </Link>
        </div>
      </section>

      <section className="wrap" style={{ paddingTop: 24, paddingBottom: 80 }}>
        <SectionHead eyebrow="Missing right now" title="Help bring them home" action={<Link to="/pets?status=lost" className="btn-link">See all lost pets<Icon name="arrow_forward" /></Link>} />
        {lost.data ? <PetGrid pets={lost.data.items} /> : <div className="pet-grid">{[0, 1, 2, 3].map((i) => <Skeleton key={i} height={360} />)}</div>}
      </section>

      <section style={{ background: '#fff', borderTop: '1px solid var(--line)', borderBottom: '1px solid var(--line)' }}>
        <div className="wrap" style={{ paddingTop: 'clamp(56px,7vw,96px)', paddingBottom: 'clamp(56px,7vw,96px)' }}>
          <div className="section-head" style={{ gap: 24, marginBottom: 48 }}>
            <div style={{ maxWidth: 560 }}>
              <div className="eyebrow teal">How PawConnect works</div>
              <h2 className="h-section">Three journeys, one connected platform</h2>
            </div>
            <Seg className="cream dark-thumb" style={{ border: '1px solid var(--line)', borderRadius: 14 }} value={how} onChange={setHow}
              options={[{ value: 'lost', label: 'Lost pet' }, { value: 'found', label: 'Found pet' }, { value: 'adopt', label: 'Adoption' }]} />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(190px,1fr))', gap: '32px 24px' }}>
            {HOW[how].map(([icon, title, text], i) => (
              <div key={title} className="col" style={{ gap: 14 }}>
                <div className="row" style={{ gap: 12 }}>
                  <div style={{ width: 48, height: 48, borderRadius: 14, background: HOW_COLORS[how][0], color: HOW_COLORS[how][1], display: 'grid', placeItems: 'center', flexShrink: 0 }}><Icon name={icon} size={24} /></div>
                  <div style={{ flex: 1, height: 1, background: '#EADCCD' }} />
                  <span className="mono muted" style={{ fontSize: 12 }}>0{i + 1}</span>
                </div>
                <div style={{ fontSize: 17, fontWeight: 800, letterSpacing: '-.01em' }}>{title}</div>
                <div style={{ fontSize: 14, lineHeight: 1.55, color: 'var(--muted)', textWrap: 'pretty' }}>{text}</div>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 40 }}>
            <Link to={HOW_CTA[how][1]} className="btn btn-outline" style={{ height: 48, padding: '0 20px', fontSize: 15 }}>{HOW_CTA[how][0]}<Icon name="arrow_forward" size={19} /></Link>
          </div>
        </div>
      </section>

      <section style={{ background: 'var(--forest)', color: 'var(--cream)' }}>
        <div className="wrap" style={{ paddingTop: 'clamp(64px,8vw,112px)', paddingBottom: 'clamp(64px,8vw,112px)' }}>
          <div className="section-head" style={{ gap: 24, marginBottom: 48 }}>
            <div style={{ maxWidth: 640 }}>
              <div className="eyebrow" style={{ color: '#F4A988', marginBottom: 12 }}>Six models, trained by our team</div>
              <h2 style={{ fontSize: 'clamp(30px,3.4vw,46px)', fontWeight: 800, letterSpacing: '-.03em', lineHeight: 1.08, textWrap: 'balance' }}>Computer vision and machine learning, built for rescue work</h2>
            </div>
            <p style={{ maxWidth: 420, fontSize: 15, lineHeight: 1.6, color: 'var(--sage)' }}>Each model is trained by one member of our six-person team on a public dataset. Until evaluation is complete, every prediction in the product is labelled as a demo result.</p>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(min(100%,340px),1fr))', gap: 20 }}>
            {MODELS.map((m, i) => (
              <div key={m.code} className="col" style={{ background: 'var(--forest-2)', border: '1px solid rgba(255,255,255,.08)', borderRadius: 22, padding: 14, gap: 18 }}>
                <ModelVisual i={i} />
                <div style={{ padding: '0 8px 8px' }}>
                  <div className="mono" style={{ fontSize: 11, color: 'var(--mint-2)', marginBottom: 8 }}>{m.code}</div>
                  <div style={{ fontSize: 19, fontWeight: 800, marginBottom: 6 }}>{m.title}</div>
                  <div style={{ fontSize: 14, lineHeight: 1.55, color: 'var(--sage)' }}>{m.text}</div>
                </div>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 36 }}>
            <Link to="/match" className="btn" style={{ height: 52, padding: '0 22px', borderRadius: 14, background: 'var(--cream)', color: 'var(--forest)', fontSize: 15 }}><Icon name="image_search" />Try AI Matching</Link>
          </div>
        </div>
      </section>

      <section className="wrap" style={{ paddingTop: 'clamp(64px,7vw,96px)', paddingBottom: 40 }}>
        <SectionHead eyebrow="From our partner shelters" title="Ready for a new home" action={<Link to="/adopt" className="btn-link">Visit the Adoption Center<Icon name="arrow_forward" /></Link>} />
        {adopt.data ? <PetGrid pets={adopt.data.items} /> : <div className="pet-grid">{[0, 1, 2, 3].map((i) => <Skeleton key={i} height={360} />)}</div>}
      </section>

      <section className="wrap" style={{ paddingTop: 56, paddingBottom: 'clamp(64px,7vw,96px)' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(min(100%,440px),1fr))', gap: 'clamp(32px,5vw,72px)', alignItems: 'center' }}>
          <div style={{ borderRadius: 28, overflow: 'hidden', aspectRatio: '5/4', background: 'var(--sand)' }}>
            <img src={img(U + '1504595403659-9088ce801e29', 1200)} alt="Two rescued dogs" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
          </div>
          <div className="col" style={{ gap: 26 }}>
            <div>
              <div className="eyebrow teal">Community</div>
              <h2 className="h-section" style={{ marginBottom: 14, textWrap: 'balance' }}>Built with Tunisia’s rescue volunteers</h2>
              <p style={{ fontSize: 16, lineHeight: 1.6, color: 'var(--body)', maxWidth: 520 }}>Lost and found posts are scattered across neighbourhood groups and shelter pages. PawConnect brings them into one searchable place, with photos that can be compared automatically.</p>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,minmax(0,1fr))', gap: 12 }}>
              {[
                ['24', 'governorates covered', 'var(--rust)'],
                [String(stats.data?.shelters ?? 4), 'pilot partner shelters', 'var(--teal)'],
                ['0 TND', 'for owners, finders and shelters', 'var(--ink)'],
              ].map(([v, l, c]) => (
                <div key={l} className="card" style={{ padding: 18, borderRadius: 18 }}>
                  <div style={{ fontSize: 'clamp(26px,2.6vw,36px)', fontWeight: 800, letterSpacing: '-.03em', color: c }}>{v}</div>
                  <div style={{ fontSize: 13, color: 'var(--muted)', lineHeight: 1.4, marginTop: 4 }}>{l}</div>
                </div>
              ))}
            </div>
            <figure style={{ margin: 0, padding: '22px 24px', borderRadius: 20, background: 'var(--peach)' }}>
              <blockquote style={{ margin: 0, fontSize: 17, lineHeight: 1.55, fontWeight: 600, textWrap: 'pretty' }}>“Most of our calls are about animals someone has found and can’t identify. One shared place with photos saves our volunteers hours every week.”</blockquote>
              <figcaption style={{ marginTop: 12, fontSize: 13, color: 'var(--muted)' }}>Amira H., volunteer coordinator · Refuge Les Amis de Carthage</figcaption>
            </figure>
          </div>
        </div>
      </section>

      <section className="wrap" style={{ paddingBottom: 88 }}>
        <div className="wrap-row between" style={{ borderRadius: 28, background: 'var(--ink)', color: 'var(--cream)', padding: 'clamp(32px,5vw,56px)', alignItems: 'center', gap: 24 }}>
          <div style={{ maxWidth: 620 }}>
            <h2 style={{ margin: '0 0 10px', fontSize: 'clamp(26px,3vw,38px)', fontWeight: 800, letterSpacing: '-.03em', lineHeight: 1.1 }}>Lost a pet today?</h2>
            <p style={{ fontSize: 16, lineHeight: 1.55, color: '#C9CFD3' }}>The first hours matter. Publish a report with photos and we start comparing it with found reports right away.</p>
          </div>
          <div className="wrap-row" style={{ gap: 12 }}>
            <Link to="/report?kind=lost" className="btn btn-lg" style={{ background: 'var(--orange)', color: '#fff', fontSize: 15 }}>Report a lost pet</Link>
            <Link to="/report?kind=found" className="btn btn-lg" style={{ background: 'transparent', color: 'var(--cream)', border: '1px solid rgba(255,255,255,.25)', fontSize: 15 }}>I found an animal</Link>
          </div>
        </div>
      </section>
    </main>
  )
}
