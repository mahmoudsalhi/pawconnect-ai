import { useMemo, useState } from 'react'
import type { AssistantAnswer, Pet, RecoveryOutlook, ReportFacts, SearchPlan } from '../lib/api'
import { img, STATUS_DOT } from '../lib/format'
import { useAssistant, useLocationPrediction } from '../lib/queries'
import { useToast } from '../lib/toast'
import { MapView, type ZoneBand } from './MapView'
import { Icon, Skeleton, Spinner } from './ui'

// Three clearly different colours, hottest = search first. Bands are exclusive, so 50 + 80 make the 80% zone.
const BAND_STYLE: Record<number, { color: string; opacity: number; label: string }> = {
  50: { color: '#C81E1E', opacity: 0.7, label: 'Search first' },
  80: { color: '#F97316', opacity: 0.5, label: 'Then here' },
  95: { color: '#FACC15', opacity: 0.42, label: 'Wider area' },
}

function km2(cells: number, cellM: number) {
  const v = (cells * cellM * cellM) / 1e6
  if (v < 0.01) return `${Math.round(v * 1e6).toLocaleString()} m²`
  return `${v.toFixed(v < 0.1 ? 2 : v < 10 ? 1 : 0)} km²`
}

const pct = (v: number) => `${Math.round(v * 100)}%`

function duration(hours: number) {
  if (hours < 1.5) return 'about an hour'
  if (hours < 36) return `about ${Math.round(hours)} hours`
  const d = hours / 24
  return d < 14 ? `about ${Math.round(d)} days` : `about ${Math.round(d / 7)} weeks`
}

function Outlook({ r, name, missing }: { r: RecoveryOutlook; name: string; missing: string }) {
  if (r.beyond_horizon) {
    return (
      <div className="note" style={{ marginBottom: 16 }}>
        <Icon name="hourglass_bottom" />
        <span>{name} has been missing longer than the {r.horizon_days} days this estimate covers. Pets are still found after
          that: keep posters up, check shelters and vets regularly, and keep sharing the report.</span>
      </div>
    )
  }
  const tone = r.found_alive >= 0.6 ? '#176B63' : r.found_alive >= 0.35 ? '#B7791F' : '#C81E1E'
  const tiles: [string, string][] = [
    ['Found in the next 24 hours', pct(r.within_24h)],
    ['Found in the next 7 days', pct(r.within_7d)],
    ['Typical wait from now, if found', r.median_hours_if_found == null ? '—' : duration(r.median_hours_if_found)],
  ]
  return (
    <div className="card" style={{ padding: 18, borderRadius: 18, marginBottom: 16 }}>
      <div className="row" style={{ gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
        <div style={{ fontSize: 40, fontWeight: 800, letterSpacing: '-.03em', color: tone, lineHeight: 1 }}>{pct(r.found_alive)}</div>
        <div style={{ flex: '1 1 200px', fontSize: 14, lineHeight: 1.45 }}>
          <strong>Chance {name} is still found alive</strong>
          <div style={{ color: 'var(--muted)' }}>
            Missing {missing} so far. It was {pct(r.found_alive_at_start)} when {name} went missing and goes down a little
            each day without news.
          </div>
        </div>
      </div>
      {r.by_lifestyle && (
        <div style={{ fontSize: 13, color: 'var(--muted)', marginTop: 10, lineHeight: 1.5 }}>
          This is an average, because the report doesn’t say whether {name} goes outside:{' '}
          <strong style={{ color: 'var(--ink)' }}>{pct(r.by_lifestyle.indoor_only)}</strong> if indoor-only,{' '}
          <strong style={{ color: 'var(--ink)' }}>{pct(r.by_lifestyle.outdoor_access)}</strong> if {name} usually goes out.
        </div>
      )}
      <div className="row" style={{ gap: 10, marginTop: 14, flexWrap: 'wrap' }}>
        {tiles.map(([label, value]) => (
          <div key={label} style={{ flex: '1 1 120px', padding: '10px 12px', borderRadius: 12, background: 'var(--peach)' }}>
            <div style={{ fontSize: 12, color: 'var(--muted)' }}>{label}</div>
            <div style={{ fontSize: 18, fontWeight: 800 }}>{value}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

const TEMPERAMENT: Record<string, string> = {
  curious: 'Curious, approaches people', cautious: 'Cautious, hides', xenophobic: 'Fearful of strangers',
  gregarious: 'Friendly with people', aloof: 'Wary, keeps distance',
}
const CIRCUMSTANCE: Record<string, string> = {
  door_window_escape: 'Escaped through a door or window', balcony_fall: 'Fell or jumped from a balcony',
  moved_house: 'Recently moved house', transport_escape: 'Escaped during transport', fireworks_noise: 'Scared by fireworks or noise',
  did_not_return: 'Didn’t come home', escaped_home_yard: 'Escaped from home or yard', slipped_leash_walk: 'Slipped the lead on a walk',
  car_accident: 'Lost after a car accident', followed_dogs: 'Followed other dogs',
}

function Facts({ f, name }: { f: ReportFacts; name: string }) {
  const chips = [
    f.goes_outside !== 'unknown' && (f.goes_outside === 'no' ? 'Indoor-only' : 'Usually goes outside'),
    TEMPERAMENT[f.temperament],
    CIRCUMSTANCE[f.circumstance],
    f.collar_or_tag !== 'unknown' && (f.collar_or_tag === 'yes' ? 'Wearing a collar or tag' : 'No collar or tag'),
  ].filter(Boolean) as string[]
  if (!chips.length) return null
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 13, color: 'var(--muted)', marginBottom: 8 }}>
        <Icon name="auto_awesome" size={14} style={{ verticalAlign: '-2px', marginRight: 4 }} />
        Read from {name}’s description and used in the estimate:
      </div>
      <div className="wrap-row" style={{ gap: 8 }}>{chips.map((c) => <span key={c} className="chip">{c}</span>)}</div>
    </div>
  )
}

function CopyButton({ text }: { text: string }) {
  const toast = useToast()
  return (
    <button type="button" className="btn-link" style={{ fontSize: 13 }}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text)
          toast('Copied')
        } catch {
          toast('Couldn’t copy')
        }
      }}>
      <Icon name="content_copy" size={16} />Copy
    </button>
  )
}

function Plan({ p, model }: { p: SearchPlan; model: string | null }) {
  const [lang, setLang] = useState<'fr' | 'ar'>('fr')
  const poster = lang === 'fr' ? p.poster_fr : p.poster_ar
  const posterText = `${poster.headline}\n\n${poster.text}`
  return (
    <div className="card" style={{ padding: 18, borderRadius: 18, marginTop: 18 }}>
      <div className="row between" style={{ alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
        <h3 style={{ fontSize: 18, fontWeight: 800 }}>Search plan</h3>
        <span style={{ fontSize: 12, color: 'var(--muted)' }}>Written by AI{model ? ` (${model})` : ''} from the zones above</span>
      </div>
      <p style={{ fontSize: 14, lineHeight: 1.55, marginTop: 8 }}>{p.summary}</p>
      <ol style={{ margin: '12px 0 0', paddingLeft: 20, display: 'grid', gap: 10 }}>
        {p.steps.map((s, i) => (
          <li key={i} style={{ fontSize: 14, lineHeight: 1.5 }}>
            <strong>{s.when}</strong> · <span style={{ color: 'var(--muted)' }}>{s.where}</span>
            <div>{s.what}</div>
          </li>
        ))}
      </ol>
      <div className="row between" style={{ alignItems: 'center', marginTop: 18, gap: 8, flexWrap: 'wrap' }}>
        <strong style={{ fontSize: 15 }}>Poster</strong>
        <div className="row" style={{ gap: 6 }}>
          {(['fr', 'ar'] as const).map((l) => (
            <button key={l} type="button" className="chip" onClick={() => setLang(l)}
              style={{ cursor: 'pointer', background: lang === l ? 'var(--ink)' : undefined, color: lang === l ? '#fff' : undefined }}>
              {l === 'fr' ? 'Français' : 'العربية'}
            </button>
          ))}
        </div>
      </div>
      <div dir={lang === 'ar' ? 'rtl' : 'ltr'} style={{ marginTop: 10, padding: 14, borderRadius: 12, background: 'var(--peach)' }}>
        <div style={{ fontSize: 20, fontWeight: 800, marginBottom: 6 }}>{poster.headline}</div>
        <div style={{ fontSize: 14, lineHeight: 1.55, whiteSpace: 'pre-wrap' }}>{poster.text}</div>
      </div>
      <div className="row" style={{ justifyContent: 'flex-end', marginTop: 6 }}><CopyButton text={posterText} /></div>
      <strong style={{ fontSize: 15, display: 'block', marginTop: 12 }}>Post to share</strong>
      <div style={{ marginTop: 6, fontSize: 14, lineHeight: 1.5 }}>{p.share_text}</div>
      <div className="row" style={{ justifyContent: 'flex-end', marginTop: 6 }}><CopyButton text={p.share_text} /></div>
    </div>
  )
}

function AssistantStatus({ ai }: { ai: { isLoading: boolean; data?: AssistantAnswer } }) {
  if (ai.isLoading) {
    return (
      <div className="row" style={{ gap: 10, alignItems: 'center', fontSize: 14, color: 'var(--muted)', marginTop: 16 }}>
        <Spinner variant="teal" />AI assistant is reading the description and writing a search plan…
      </div>
    )
  }
  if (ai.data?.status === 'error') {
    return <div style={{ fontSize: 13, color: 'var(--muted)', marginTop: 14 }}>Search plan not available right now: {ai.data.message}</div>
  }
  return null
}

function missingFor(hours: number) {
  if (hours < 24) return `${Math.round(hours)} h`
  const d = hours / 24
  return `${d < 10 ? d.toFixed(1) : Math.round(d)} days`
}

/** M7: where a lost pet is most likely to be now. Shown only for open lost reports. */
export function SearchZones({ pet }: { pet: Pet }) {
  const open = pet.status === 'lost' && !pet.resolved
  const q = useLocationPrediction(pet.id, open)
  const ai = useAssistant(pet.id, open)
  // Once the assistant has read the description, show M7's answer that uses what it found.
  const d = ai.data?.location ?? q.data
  const facts = ai.data?.status === 'ok' ? ai.data.facts : null
  const plan = ai.data?.status === 'ok' ? ai.data.plan : null

  const view = useMemo(() => {
    if (!d) return null
    const bands: ZoneBand[] = [...d.zones].sort((a, b) => b.share - a.share)
      .map((z) => ({ cells: z.cells, color: BAND_STYLE[z.share].color, opacity: BAND_STYLE[z.share].opacity }))
    // Frame the "search first" zone (zoom out for the rest); a sample of its cells is enough for the bounds.
    const inner = d.zones.filter((z) => z.share === 50).flatMap((z) => z.cells)
    const step = Math.max(1, Math.floor(inner.length / 400))
    const fitTo = inner.filter((_, i) => i % step === 0)
    let cum = 0
    const legend = [...d.zones].sort((a, b) => a.share - b.share).map((z) => {
      cum += z.cells.length
      return { share: z.share, area: km2(cum, d.cell_m), ...BAND_STYLE[z.share] }
    })
    return { bands, fitTo: fitTo.length > 1 ? fitTo : undefined, legend }
  }, [d])

  if (q.isError && !ai.data) return null
  const species = pet.species.toLowerCase() === 'cat' ? 'cats' : 'dogs'

  return (
    <div style={{ marginTop: 36 }}>
      <div className="row" style={{ gap: 10, marginBottom: 12, alignItems: 'center', flexWrap: 'wrap' }}>
        <h2 className="h-card">Where to search</h2>
        <span className="chip" style={{ fontSize: 12, padding: '4px 10px' }}>
          <Icon name="auto_awesome" size={14} style={{ verticalAlign: '-2px', marginRight: 4 }} />AI estimate · experimental
        </span>
      </div>
      {!d || !view ? (
        <Skeleton height={360} style={{ borderRadius: 18 }} />
      ) : (
        <>
          {facts && <Facts f={facts} name={pet.name} />}
          {d.recovery && <Outlook r={d.recovery} name={pet.name} missing={missingFor(d.hours_missing)} />}
          <MapView height={360} center={[pet.lat, pet.lng]} zoom={14} zones={view.bands} cellM={d.cell_m} fitTo={view.fitTo} maxZoom={16}
            pins={[{ kind: 'photo', id: `zones-${pet.id}`, lat: pet.lat, lng: pet.lng, photo: img(pet.photos[0], 200, 200), ring: STATUS_DOT[pet.status] }]}>
            <div className="map-overlay map-label"><Icon name="schedule" size={16} />Missing {missingFor(d.hours_missing)}</div>
          </MapView>
          <div className="wrap-row" style={{ gap: 8, marginTop: 12 }}>
            {view.legend.map((l) => (
              <span key={l.share} className="chip row" style={{ gap: 8, alignItems: 'center' }}>
                <span style={{ width: 12, height: 12, borderRadius: 3, background: l.color, opacity: Math.min(1, l.opacity + 0.35) }} />
                {l.label} · {l.share}% · {l.area}
              </span>
            ))}
          </div>
          <p style={{ fontSize: 14, color: 'var(--muted)', marginTop: 12, lineHeight: 1.55 }}>
            Each zone holds {pet.name} with that likelihood, based on how lost {species} move through the streets,
            parks and hiding places around where they were last seen, and how long they have been missing.
            {d.self_return_share > 0 && <> About {Math.round(d.self_return_share * 100)}% of similar {species} come home on their own,
            so keep a door or window open and leave out food and something that smells of home.</>}
          </p>
          {d.missing_inputs.length > 0 && (
            <div style={{ fontSize: 13, color: 'var(--muted)', marginTop: 10, lineHeight: 1.5 }}>
              <strong style={{ color: 'var(--ink)' }}>Sharper with:</strong> {d.missing_inputs.join(' · ')}
            </div>
          )}
          {plan ? <Plan p={plan} model={ai.data?.model ?? null} /> : <AssistantStatus ai={ai} />}
          <div className="note" style={{ marginTop: 14 }}>
            <Icon name="science" />
            <span>Experimental model trained on simulated cases calibrated to published lost-pet studies, not yet on real
              reunions. Use it to decide where to search and put up posters first. Zones use the same approximate location as
              this page.</span>
          </div>
        </>
      )}
    </div>
  )
}
