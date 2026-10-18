import { useMutation, useQuery } from '@tanstack/react-query'
import { useState, type ChangeEvent } from 'react'
import { Link } from 'react-router-dom'
import { BreedBars, Icon, Seg, Spinner } from '../components/ui'
import { api, type BreedInfo, type BreedResult, type LookalikeResult } from '../lib/api'
import { img } from '../lib/format'
import { useToast } from '../lib/toast'

type Species = 'Any' | 'Dog' | 'Cat'

function confidence(top: number) {
  if (top >= 70) return { label: 'Confident', color: 'var(--teal-dd)', bg: 'var(--teal-soft)', note: 'The model is fairly sure about this breed.' }
  if (top >= 40) return { label: 'Likely', color: '#8A6410', bg: '#F6EBCF', note: 'A reasonable guess. Check the other suggestions too.' }
  return { label: 'Uncertain', color: 'var(--rust-dd)', bg: 'var(--peach)', note: 'Probably a mixed breed or a breed the model was not trained on.' }
}

export default function BreedId() {
  const toast = useToast()
  const [photo, setPhoto] = useState<string | null>(null)
  const [species, setSpecies] = useState<Species>('Any')
  const [uploading, setUploading] = useState(false)
  const info = useQuery({ queryKey: ['breed-info'], queryFn: () => api.get<BreedInfo>('/api/ai/breed/info'), staleTime: Infinity })
  const run = useMutation({ mutationFn: (body: { photos: string[]; species: Species }) => api.post<BreedResult>('/api/ai/breed', body) })
  const looks = useMutation({ mutationFn: (body: { photos: string[]; species: Species }) => api.post<LookalikeResult>('/api/ai/breed/lookalikes', body) })

  const identify = (url: string, s: Species) => {
    run.mutate({ photos: [url], species: s })
    looks.mutate({ photos: [url], species: s })
  }

  async function onFile(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    setUploading(true)
    try {
      const { urls } = await api.upload([file])
      setPhoto(urls[0])
      identify(urls[0], species)
    } catch {
      toast('That photo could not be uploaded. Try a JPEG or PNG under 10 MB.')
    } finally {
      setUploading(false)
    }
  }

  const preds = run.data?.predictions ?? []
  const conf = preds.length ? confidence(preds[0].confidence) : null
  const m = info.data?.metrics
  const dogs = info.data?.breeds.filter((b) => b.species === 'Dog') ?? []
  const cats = info.data?.breeds.filter((b) => b.species === 'Cat') ?? []

  return (
    <main className="wrap" style={{ paddingTop: 40, paddingBottom: 88 }}>
      <div style={{ maxWidth: 700, marginBottom: 28 }}>
        <div className="wrap-row" style={{ alignItems: 'center', gap: 8, marginBottom: 10 }}><span className="eyebrow" style={{ margin: 0 }}>Breed identification</span><span className="demo-tag">M1 · CONVNEXT · OXFORD-IIIT PET</span></div>
        <h1 style={{ fontSize: 'clamp(34px,4vw,52px)', fontWeight: 800, letterSpacing: '-.04em', lineHeight: 1.05, textWrap: 'balance' }}>What breed is this pet?</h1>
        <p style={{ fontSize: 16, lineHeight: 1.6, color: 'var(--body)' }}>Upload a clear photo of a dog or cat. Our image model compares it with thousands of labelled photos and suggests the five most likely breeds.</p>
        {m && (
          <div className="wrap-row" style={{ gap: 8, marginTop: 14 }}>
            {[[`${(m.top1_accuracy * 100).toFixed(1)}%`, 'first guess right'], [`${(m.top3_accuracy * 100).toFixed(1)}%`, 'right breed in top 3'], [m.test_photos.toLocaleString(), 'test photos']].map(([v, l]) => (
              <span key={l} className="row" style={{ gap: 6, padding: '6px 12px', borderRadius: 999, background: '#fff', border: '1px solid var(--line)', fontSize: 13 }}><b className="mono">{v}</b><span style={{ color: 'var(--muted)' }}>{l}</span></span>
            ))}
          </div>
        )}
      </div>

      <div className="wrap-row" style={{ gap: 28, alignItems: 'flex-start' }}>
        <section className="card col" style={{ flex: '1 1 340px', minWidth: 0, padding: 'clamp(18px,2.4vw,26px)', gap: 18, borderRadius: 22 }}>
          <div>
            <div className="field-label"><span>Dog or cat?</span></div>
            <Seg value={species} onChange={(v) => { setSpecies(v); if (photo) identify(photo, v) }} options={[{ value: 'Any', label: 'Not sure' }, { value: 'Dog', label: 'Dog' }, { value: 'Cat', label: 'Cat' }]} />
          </div>
          <label className="col" style={{ alignItems: 'center', justifyContent: 'center', gap: 10, minHeight: 260, borderRadius: 18, border: '2px dashed var(--line-2)', background: 'var(--cream)', cursor: uploading ? 'wait' : 'pointer', overflow: 'hidden', position: 'relative', textAlign: 'center', padding: photo ? 0 : 20 }}>
            <input type="file" accept="image/*" className="sr-only" disabled={uploading} onChange={onFile} />
            {photo
              ? <img src={photo} alt="Your pet" style={{ width: '100%', maxHeight: 380, objectFit: 'contain', background: '#fff' }} />
              : <>
                  {uploading ? <Spinner variant="teal" /> : <Icon name="add_photo_alternate" size={34} style={{ color: 'var(--teal)' }} />}
                  <span style={{ fontSize: 16, fontWeight: 800 }}>{uploading ? 'Uploading…' : 'Click to upload a photo'}</span>
                  <span style={{ fontSize: 13, color: 'var(--muted)' }}>One animal, face visible, good light</span>
                </>}
          </label>
          {photo && <label className="btn btn-outline" style={{ cursor: 'pointer' }}><input type="file" accept="image/*" className="sr-only" disabled={uploading} onChange={onFile} /><Icon name="photo_camera" />{uploading ? 'Uploading…' : 'Try another photo'}</label>}
        </section>

        <section className="col" style={{ flex: '1.2 1 380px', minWidth: 0, gap: 16 }} aria-live="polite">
          <div className="card" style={{ padding: 'clamp(18px,2.4vw,26px)', borderRadius: 22 }}>
            <div className="row" style={{ gap: 8, fontSize: 16, fontWeight: 800, marginBottom: 16 }}><Icon name="auto_awesome" size={20} style={{ color: 'var(--teal)' }} />Breed suggestions</div>
            {run.isPending && <div className="row" style={{ gap: 10, color: 'var(--muted)', fontSize: 14 }}><Spinner variant="teal" />Analysing the photo… the first one takes a few seconds.</div>}
            {run.isError && <div style={{ color: 'var(--error)', fontSize: 14 }}>The photo could not be analysed. Please try another one.</div>}
            {!run.isPending && !run.data && !run.isError && <div style={{ color: 'var(--muted)', fontSize: 14 }}>Upload a photo to see the five most likely breeds.</div>}
            {!run.isPending && run.data && preds.length === 0 && <div style={{ color: 'var(--muted)', fontSize: 14 }}>The photo could not be read. Please try a different image.</div>}
            {!run.isPending && preds.length > 0 && conf && (
              <>
                <div style={{ fontSize: 26, fontWeight: 800, letterSpacing: '-.02em' }}>{preds[0].label}</div>
                <div className="wrap-row" style={{ gap: 8, alignItems: 'center', margin: '6px 0 18px' }}>
                  <span style={{ padding: '4px 9px', borderRadius: 999, background: conf.bg, color: conf.color, fontSize: 12, fontWeight: 700 }}>{conf.label} · {preds[0].confidence}%</span>
                  <span style={{ fontSize: 13, color: 'var(--muted)' }}>{conf.note}</span>
                </div>
                <BreedBars breeds={preds.map((p) => [`${p.label} (${p.species.toLowerCase()})`, p.confidence])} />
              </>
            )}
          </div>

          {(looks.isPending || (looks.data?.results.length ?? 0) > 0) && (
            <div className="card" style={{ padding: 'clamp(18px,2.4vw,26px)', borderRadius: 22 }}>
              <div className="wrap-row" style={{ gap: 8, alignItems: 'center', marginBottom: 4 }}>
                <Icon name="pets" size={20} style={{ color: 'var(--rust)' }} />
                <span style={{ fontSize: 16, fontWeight: 800 }}>Look-alikes up for adoption</span>
                <span className="demo-tag">M1 · IMAGE FEATURES</span>
              </div>
              <p style={{ fontSize: 13, color: 'var(--muted)', lineHeight: 1.5, margin: '0 0 14px' }}>Shelter pets whose photos look most like yours: coat, colour and face shape, compared by the same image model.</p>
              {looks.isPending && <div className="row" style={{ gap: 10, color: 'var(--muted)', fontSize: 14 }}><Spinner variant="teal" />Looking through the adoption listings…</div>}
              {!looks.isPending && looks.data && (
                <div className="col" style={{ gap: 10 }}>
                  {looks.data.results.map(({ pet, similarity }) => (
                    <Link key={pet.id} to={`/pets/${pet.id}`} className="row" style={{ gap: 12, padding: 8, borderRadius: 14, border: '1px solid var(--line)', background: '#fff', alignItems: 'center' }}>
                      <img src={img(pet.photos[0], 128, 128)} alt={pet.name} loading="lazy" style={{ width: 64, height: 64, borderRadius: 10, objectFit: 'cover', flex: 'none' }} />
                      <div style={{ minWidth: 0, flex: 1 }}>
                        <div style={{ fontSize: 15, fontWeight: 800 }}>{pet.name}</div>
                        <div style={{ fontSize: 13, color: 'var(--muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{pet.breed || pet.species} · {pet.age_label} · {pet.city}</div>
                      </div>
                      <span className="mono nowrap" style={{ padding: '4px 9px', borderRadius: 999, background: 'var(--teal-soft)', color: 'var(--teal-dd)', fontSize: 12, fontWeight: 700 }}>{similarity}% alike</span>
                    </Link>
                  ))}
                </div>
              )}
            </div>
          )}

          <div className="card" style={{ padding: 'clamp(18px,2.4vw,26px)', borderRadius: 22 }}>
            <div style={{ fontSize: 15, fontWeight: 800, marginBottom: 6 }}>Breeds the model knows ({info.data?.breeds.length ?? 37})</div>
            <p style={{ fontSize: 13, color: 'var(--muted)', lineHeight: 1.5, margin: '0 0 12px' }}>Trained on the Oxford-IIIT Pet dataset. Other breeds, such as German shepherds, Labradors, huskies, mixed breeds and domestic shorthair cats, get the closest breed it knows.</p>
            {[['Dogs', dogs], ['Cats', cats]].map(([title, list]) => (
              <div key={title as string} style={{ marginBottom: 10 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--muted)', marginBottom: 6 }}>{title as string}</div>
                <div className="wrap-row" style={{ gap: 6 }}>{(list as BreedInfo['breeds']).map((b) => <span key={b.label} className="trait">{b.label}</span>)}</div>
              </div>
            ))}
          </div>
        </section>
      </div>
    </main>
  )
}
