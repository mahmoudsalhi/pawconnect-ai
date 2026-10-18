import { useMutation } from '@tanstack/react-query'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Empty, Icon, Pills, ScoreRing, Spinner } from '../components/ui'
import { api, type CompatResult, type Questionnaire } from '../lib/api'
import { img } from '../lib/format'

type Key = Exclude<keyof Questionnaire, 'ideal'>

const QUESTIONS: { key: Key; label: string; options: [string, string][] }[] = [
  { key: 'species', label: 'Are you looking for a dog or a cat?', options: [['Any', 'Either'], ['Dog', 'Dog'], ['Cat', 'Cat']] },
  { key: 'home', label: 'Where do you live?', options: [['apartment', 'Apartment'], ['house', 'House, no garden'], ['house_garden', 'House with garden']] },
  { key: 'children', label: 'Children at home?', options: [['none', 'None'], ['older', 'Older children (6+)'], ['young', 'Young children (under 6)']] },
  { key: 'other_pets', label: 'Other pets at home?', options: [['none', 'None'], ['dogs', 'Dogs'], ['cats', 'Cats'], ['both', 'Dogs and cats']] },
  { key: 'activity', label: 'How active is your lifestyle?', options: [['relaxed', 'Relaxed'], ['moderate', 'Moderate'], ['active', 'Very active']] },
  { key: 'hours_alone', label: 'Hours the pet would be alone on a typical day', options: [['under4', 'Under 4'], ['4to8', '4 to 8'], ['over8', 'More than 8']] },
  { key: 'experience', label: 'Your experience with pets', options: [['first', 'First pet'], ['some', 'Some'], ['experienced', 'Experienced']] },
  { key: 'grooming', label: 'How much grooming are you happy to do?', options: [['minimal', 'Minimal'], ['some', 'Some'], ['happy', 'Happy to groom daily']] },
  { key: 'age', label: 'Preferred age', options: [['any', 'No preference'], ['young', 'Young'], ['adult', 'Adult'], ['senior', 'Senior']] },
  { key: 'size', label: 'Preferred size', options: [['any', 'No preference'], ['Small', 'Small'], ['Medium', 'Medium'], ['Large', 'Large']] },
]

const START: Questionnaire = {
  species: 'Any', home: 'apartment', children: 'none', other_pets: 'none', activity: 'moderate', hours_alone: '4to8',
  experience: 'some', grooming: 'some', age: 'any', size: 'any', ideal: '',
}

const scoreColor = (s: number) => (s >= 75 ? '#176B63' : s >= 60 ? '#C99A2E' : '#C9522B')

export default function Compat() {
  const [q, setQ] = useState<Questionnaire>(START)
  const run = useMutation({ mutationFn: (body: Questionnaire) => api.post<CompatResult>('/api/ai/compatibility', body) })
  const results = run.data?.results ?? []

  return (
    <main className="wrap" style={{ paddingTop: 40, paddingBottom: 88 }}>
      <div style={{ maxWidth: 680, marginBottom: 32 }}>
        <div className="wrap-row" style={{ alignItems: 'center', gap: 8, marginBottom: 10 }}><span className="eyebrow" style={{ margin: 0 }}>Find your match</span><span className="demo-tag">M9 · SENTENCE EMBEDDINGS + RIDGE · DOG BREED DATA</span></div>
        <h1 style={{ fontSize: 'clamp(34px,4vw,52px)', fontWeight: 800, letterSpacing: '-.04em', lineHeight: 1.05, textWrap: 'balance' }}>Which animal fits your life?</h1>
        <p style={{ fontSize: 16, lineHeight: 1.6, color: 'var(--body)' }}>Answer a few questions about your home and routine. Our AI reads every adoption profile, estimates each animal’s energy, needs and temperament, and explains how well they fit.</p>
      </div>

      <div className="wrap-row" style={{ gap: 28, alignItems: 'flex-start' }}>
        <form className="card col" style={{ flex: '1 1 380px', minWidth: 0, padding: 'clamp(18px,2.4vw,28px)', gap: 20, borderRadius: 22 }}
          onSubmit={(e) => { e.preventDefault(); run.mutate(q) }}>
          {QUESTIONS.map(({ key, label, options }) => (
            <div key={key}>
              <div className="field-label"><span>{label}</span></div>
              <Pills options={options.map(([value, l]) => ({ value, label: l }))} value={q[key]} onChange={(v) => setQ({ ...q, [key]: v })} />
            </div>
          ))}
          <div>
            <label className="field-label" htmlFor="ideal"><span>Describe your ideal companion</span><span style={{ color: 'var(--muted)', fontWeight: 600 }}>Optional</span></label>
            <textarea id="ideal" className="textarea" rows={3} maxLength={300} value={q.ideal} onChange={(e) => setQ({ ...q, ideal: e.target.value })} placeholder="e.g. a calm cuddly companion who enjoys quiet evenings" style={{ width: '100%' }} />
          </div>
          <button className="btn btn-primary btn-lg" disabled={run.isPending}>{run.isPending ? <Spinner /> : <Icon name="auto_awesome" />}Find my matches</button>
        </form>

        <section className="col" style={{ flex: '1.4 1 420px', minWidth: 0, gap: 14 }} aria-live="polite">
          {run.isError && <div className="note" style={{ color: 'var(--error)' }}><Icon name="error" /><span>Matching failed. Please try again.</span></div>}
          {!run.data && !run.isPending && !run.isError && (
            <Empty icon="favorite" title="Your matches will appear here">Each animal gets a compatibility score from 0 to 100, with the reasons behind it.</Empty>
          )}
          {run.data && results.length === 0 && <Empty icon="pets" title="No animals match these answers">Try “Either” for dog or cat, or loosen your preferences.</Empty>}
          {results.map(({ pet, score, reasons }) => (
            <Link key={pet.id} to={`/pets/${pet.id}`} className="card wrap-row" style={{ gap: 16, padding: 16, borderRadius: 18, color: 'var(--ink)', alignItems: 'flex-start' }}>
              <img src={img(pet.photos[0], 200, 200)} alt={pet.name} style={{ width: 88, height: 88, borderRadius: 14, objectFit: 'cover', flexShrink: 0 }} />
              <div className="col grow" style={{ flex: '1 1 200px', minWidth: 0, gap: 8 }}>
                <div>
                  <div style={{ fontSize: 18, fontWeight: 800 }}>{pet.name}</div>
                  <div style={{ fontSize: 13, color: 'var(--muted)' }}>{pet.species} · {pet.breed} · {pet.age_label} · {pet.shelter?.name ?? pet.contact_name}</div>
                </div>
                <ul className="col" style={{ gap: 4, margin: 0, padding: 0, listStyle: 'none' }}>
                  {reasons.map((r) => (
                    <li key={r.text} className="row" style={{ gap: 6, fontSize: 13.5, lineHeight: 1.4, alignItems: 'flex-start', color: r.good ? 'var(--ink)' : 'var(--rust-dd)' }}>
                      <Icon name={r.good ? 'check_circle' : 'error'} size={17} style={{ color: r.good ? 'var(--teal)' : 'var(--rust)', flexShrink: 0, marginTop: 1 }} />{r.text}
                    </li>
                  ))}
                </ul>
              </div>
              <ScoreRing score={score} size={64} fontSize={15} color={scoreColor(score)} />
            </Link>
          ))}
          {run.data && results.length > 0 && <div style={{ fontSize: 12.5, color: 'var(--muted)', lineHeight: 1.5 }}>Scores are a guide to start a conversation with the shelter, not a guarantee. Traits are estimated from each profile by a model trained on 618 dog breeds; estimates for cats are less certain.</div>}
        </section>
      </div>
    </main>
  )
}
