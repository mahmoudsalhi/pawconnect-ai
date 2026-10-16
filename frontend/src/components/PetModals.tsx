import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, ApiError, type Application, type Pet } from '../lib/api'
import { useAuth } from '../lib/auth'
import { firstName, GOVERNORATES, img, metaLabel } from '../lib/format'
import { useToast } from '../lib/toast'
import { Modal } from './Modal'
import { FieldError, Icon, Pills, Spinner } from './ui'

export type ModalKind = 'contact' | 'sighting' | 'apply'

export function contactDraft(p: Pet): string {
  const who = firstName(p.contact_name)
  if (p.status === 'lost') return `Hi ${who}, I think I may have seen ${p.name}. `
  if (p.status === 'found') return `Hi ${who}, I think this may be my pet. `
  return `Hello, I’d like to know more about ${p.name}. `
}

export function contactTitle(p: Pet): string {
  return p.status === 'adopt' ? `Message ${p.shelter?.name ?? p.contact_name}` : `Contact ${firstName(p.contact_name)}`
}

function ContactForm({ pet, initial, onDone }: { pet: Pet; initial?: string; onDone: () => void }) {
  const toast = useToast()
  const [msg, setMsg] = useState(initial ?? contactDraft(pet))
  const [err, setErr] = useState('')
  const send = useMutation({
    mutationFn: () => api.post(`/api/pets/${pet.id}/messages`, { body: msg }),
    onSuccess: () => {
      toast(`Message sent to ${pet.contact_name}`)
      onDone()
    },
    onError: (e: ApiError) => setErr(e.errors.body ?? e.message),
  })
  return (
    <div className="col" style={{ gap: 14 }}>
      <div style={{ fontSize: 14, color: 'var(--body)' }}>To <strong style={{ color: 'var(--ink)' }}>{pet.contact_name}</strong>. Your number is shared only if you both agree.</div>
      <textarea className={`textarea${err ? ' invalid' : ''}`} rows={5} value={msg} aria-label="Message" onChange={(e) => { setMsg(e.target.value); setErr('') }} />
      <FieldError msg={err} />
      {pet.contact_phone && <div className="info" style={{ fontSize: 13 }}><Icon name="call" size={18} />{pet.contact_name} prefers {pet.contact_pref.toLowerCase()}: {pet.contact_phone}</div>}
      <div style={{ padding: '12px 14px', borderRadius: 12, background: 'var(--peach)', fontSize: 13, lineHeight: 1.5 }}>Meet in a public place and ask for proof such as earlier photos, vet records or a microchip number.</div>
      <div className="row" style={{ justifyContent: 'flex-end', gap: 10 }}>
        <button className="btn btn-outline" onClick={onDone}>Cancel</button>
        <button className="btn btn-primary" disabled={send.isPending || msg.trim().length < 5} onClick={() => send.mutate()}>
          {send.isPending ? <Spinner /> : <><Icon name="send" size={18} />Send message</>}
        </button>
      </div>
    </div>
  )
}

function SightingForm({ pet, onDone }: { pet: Pet; onDone: () => void }) {
  const toast = useToast()
  const qc = useQueryClient()
  const [f, setF] = useState({ where: '', when: '', note: '' })
  const [errors, setErrors] = useState<Record<string, string>>({})
  const save = useMutation({
    mutationFn: () => api.post(`/api/pets/${pet.id}/sightings`, {
      place: f.where.trim(),
      seen_at: f.when ? new Date(f.when).toISOString() : null,
      note: f.note.trim(),
    }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['pet', pet.id] })
      toast(`Sighting shared with ${firstName(pet.contact_name)}. Thank you!`)
      onDone()
    },
    onError: (e: ApiError) => setErrors(Object.keys(e.errors).length ? { where: e.errors.place ?? e.message } : { where: e.message }),
  })
  const submit = () => {
    if (f.where.trim().length < 2) return setErrors({ where: `Add where you saw ${pet.name}.` })
    if (f.when && new Date(f.when) > new Date()) return setErrors({ when: 'The time can’t be in the future.' })
    save.mutate()
  }
  return (
    <div className="col" style={{ gap: 14 }}>
      <label className="field"><span className="field-label">Where did you see {pet.name}?</span>
        <input className={`input${errors.where ? ' invalid' : ''}`} value={f.where} onChange={(e) => { setF({ ...f, where: e.target.value }); setErrors({}) }} placeholder="Street, landmark or neighbourhood" />
        <FieldError msg={errors.where} /></label>
      <label className="field"><span className="field-label">When?</span>
        <input type="datetime-local" className={`input${errors.when ? ' invalid' : ''}`} style={{ padding: '0 12px' }} value={f.when} onChange={(e) => { setF({ ...f, when: e.target.value }); setErrors({}) }} />
        <FieldError msg={errors.when} /></label>
      <label className="field"><span className="field-label">Details (optional)</span>
        <textarea className="textarea" rows={3} value={f.note} onChange={(e) => setF({ ...f, note: e.target.value })} placeholder="Direction, behaviour, collar…" /></label>
      <div className="row" style={{ justifyContent: 'flex-end', gap: 10 }}>
        <button className="btn btn-outline" onClick={onDone}>Cancel</button>
        <button className="btn btn-teal" disabled={save.isPending} onClick={submit}>{save.isPending ? <Spinner /> : 'Submit sighting'}</button>
      </div>
    </div>
  )
}

const APPLY_STEPS = ['About you', 'Your home', 'Motivation']

function ApplyForm({ pet, onDone }: { pet: Pet; onDone: () => void }) {
  const { user } = useAuth()
  const qc = useQueryClient()
  const navigate = useNavigate()
  const [step, setStep] = useState(1)
  const [f, setF] = useState({
    full_name: user?.name ?? '', phone: user?.phone ?? '', gov: user?.gov || 'Tunis', housing: 'Apartment',
    other_pets: 'None', hours_alone: '4–6 hours', motivation: '', home_visit_ok: false,
  })
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [done, setDone] = useState<Application | null>(null)
  const set = (k: keyof typeof f, v: string | boolean) => { setF((s) => ({ ...s, [k]: v })); setErrors((e) => ({ ...e, [k]: '' })) }
  const shelter = pet.shelter?.name ?? pet.contact_name

  const submit = useMutation({
    mutationFn: () => api.post<Application>('/api/adoptions', { pet_id: pet.id, ...f }),
    onSuccess: (a) => {
      setDone(a)
      qc.invalidateQueries({ queryKey: ['applications'] })
      qc.invalidateQueries({ queryKey: ['dashboard'] })
      qc.invalidateQueries({ queryKey: ['notifications'] })
    },
    onError: (e: ApiError) => {
      setErrors(Object.keys(e.errors).length ? e.errors : { form: e.message })
      if (e.errors.full_name || e.errors.phone) setStep(1)
    },
  })

  const next = () => {
    const er: Record<string, string> = {}
    if (step === 1) {
      if (f.full_name.trim().length < 3) er.full_name = 'Enter your full name.'
      const d = f.phone.replace(/\D/g, '').replace(/^216/, '')
      if (d.length !== 8) er.phone = 'Enter an 8-digit Tunisian number, e.g. 22 418 903.'
    }
    if (step === 3) {
      if (f.motivation.trim().length < 30) er.motivation = 'Tell the shelter a little more (at least 30 characters).'
      if (!f.home_visit_ok) er.home_visit_ok = 'Shelters need to agree a home visit before an adoption.'
    }
    if (Object.keys(er).length) return setErrors(er)
    if (step < 3) setStep(step + 1)
    else submit.mutate()
  }

  if (done) {
    return (
      <div className="col center" style={{ alignItems: 'center', gap: 12, padding: '12px 0' }}>
        <div style={{ width: 68, height: 68, borderRadius: '50%', background: '#D7ECE8', display: 'grid', placeItems: 'center' }}><Icon name="task_alt" size={36} style={{ color: 'var(--teal)' }} /></div>
        <div style={{ fontSize: 22, fontWeight: 800 }}>Application sent</div>
        <div className="mono muted" style={{ fontSize: 13 }}>{done.code}</div>
        <div style={{ fontSize: 14, lineHeight: 1.55, color: 'var(--body)', maxWidth: 400 }}>{shelter} usually replies within 2 days. You can follow each step in your dashboard.</div>
        <button className="btn btn-teal" style={{ marginTop: 6, height: 48, padding: '0 22px', fontSize: 15 }} onClick={() => { onDone(); navigate('/dashboard/applications') }}>Track in dashboard</button>
      </div>
    )
  }

  return (
    <>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 6, marginBottom: 20 }}>
        {APPLY_STEPS.map((l, i) => (
          <div key={l}>
            <div className={`stage-bar${i + 1 < step ? ' done' : i + 1 === step ? ' cur' : ''}`} style={{ height: 5 }} />
            <div style={{ fontSize: 12, fontWeight: 700, color: i + 1 <= step ? 'var(--ink)' : 'var(--muted)', marginTop: 6 }}>{l}</div>
          </div>
        ))}
      </div>
      {errors.form && <div className="alert" style={{ marginBottom: 14 }}><Icon name="error" />{errors.form}</div>}
      {step === 1 && (
        <div className="col" style={{ gap: 14 }}>
          <label className="field"><span className="field-label">Full name</span>
            <input className={`input${errors.full_name ? ' invalid' : ''}`} value={f.full_name} onChange={(e) => set('full_name', e.target.value)} autoComplete="name" />
            <FieldError msg={errors.full_name} /></label>
          <label className="field"><span className="field-label">Phone</span>
            <input type="tel" className={`input${errors.phone ? ' invalid' : ''}`} value={f.phone} onChange={(e) => set('phone', e.target.value)} placeholder="+216 22 418 903" autoComplete="tel" />
            <FieldError msg={errors.phone} /></label>
          <label className="field"><span className="field-label">Governorate</span>
            <select className="select" value={f.gov} onChange={(e) => set('gov', e.target.value)}>{GOVERNORATES.map((g) => <option key={g}>{g}</option>)}</select></label>
        </div>
      )}
      {step === 2 && (
        <div className="col" style={{ gap: 18 }}>
          <div><span className="field-label" style={{ marginBottom: 8 }}>Housing</span>
            <Pills value={f.housing} onChange={(v) => set('housing', v)} options={['Apartment', 'House with garden', 'House without garden'].map((v) => ({ value: v, label: v }))} /></div>
          <div><span className="field-label" style={{ marginBottom: 8 }}>Other pets at home</span>
            <Pills value={f.other_pets} onChange={(v) => set('other_pets', v)} options={['None', 'Dogs', 'Cats', 'Dogs and cats'].map((v) => ({ value: v, label: v }))} /></div>
          <label className="field"><span className="field-label">Hours alone on a typical day</span>
            <select className="select" value={f.hours_alone} onChange={(e) => set('hours_alone', e.target.value)}>
              {['Less than 2 hours', '2–4 hours', '4–6 hours', 'More than 6 hours'].map((h) => <option key={h}>{h}</option>)}
            </select></label>
        </div>
      )}
      {step === 3 && (
        <div className="col" style={{ gap: 14 }}>
          <label className="field"><span className="field-label">Why would {pet.name} be happy with you?</span>
            <textarea className={`textarea${errors.motivation ? ' invalid' : ''}`} rows={5} value={f.motivation} onChange={(e) => set('motivation', e.target.value)} placeholder="Your routine, experience with animals, who lives at home…" />
            <FieldError msg={errors.motivation} /></label>
          <label className="check"><input type="checkbox" checked={f.home_visit_ok} onChange={(e) => set('home_visit_ok', e.target.checked)} />I agree to a short home visit by {shelter}.</label>
          <FieldError msg={errors.home_visit_ok} />
        </div>
      )}
      <div className="row between" style={{ gap: 10, marginTop: 22 }}>
        <div>{step > 1 && <button className="btn btn-outline" onClick={() => { setStep(step - 1); setErrors({}) }}>Back</button>}</div>
        <button className="btn btn-primary" style={{ padding: '0 22px' }} disabled={submit.isPending} onClick={next}>
          {submit.isPending ? <Spinner /> : step === 3 ? 'Submit application' : 'Continue'}
        </button>
      </div>
    </>
  )
}

export function PetModal({ kind, pet, message, onClose }: { kind: ModalKind; pet: Pet; message?: string; onClose: () => void }) {
  const title = kind === 'sighting' ? `Report a sighting of ${pet.name}` : kind === 'apply' ? `Apply to adopt ${pet.name}` : contactTitle(pet)
  return (
    <Modal title={title} subtitle={metaLabel(pet)} image={img(pet.photos[0], 200, 200)} onClose={onClose}>
      {kind === 'contact' && <ContactForm pet={pet} initial={message} onDone={onClose} />}
      {kind === 'sighting' && <SightingForm pet={pet} onDone={onClose} />}
      {kind === 'apply' && <ApplyForm pet={pet} onDone={onClose} />}
    </Modal>
  )
}
