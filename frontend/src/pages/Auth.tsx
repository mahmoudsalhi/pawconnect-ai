import { useState, type FormEvent } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { FaceLoginModal } from '../components/FaceLogin'
import { Logo } from '../components/Layout'
import { FieldError, Icon, Seg, Spinner } from '../components/ui'
import { ApiError } from '../lib/api'
import { useAuth } from '../lib/auth'
import { cameraSupported } from '../lib/face'
import { img } from '../lib/format'
import { useToast } from '../lib/toast'

type Mode = 'login' | 'register'
type AccountType = 'owner' | 'adopter' | 'shelter'
const TYPES: [AccountType, string, string, string][] = [
  ['owner', 'Pet owner', 'Report and find lost pets', 'pets'],
  ['adopter', 'Adopter', 'Apply to adopt animals', 'favorite'],
  ['shelter', 'Shelter / Rescue', 'List animals and see insights', 'home_work'],
]
const EMAIL_RX = /^\S+@\S+\.\S+$/

function strength(pw: string) {
  const sc = [pw.length >= 8, /[A-Z]/.test(pw), /\d/.test(pw), /[^A-Za-z0-9]/.test(pw)].filter(Boolean).length
  const level = pw.length < 8 ? 0 : sc
  return {
    label: ['Too short', 'Weak', 'Fair', 'Good', 'Strong'][level],
    color: ['#C9522B', '#C9522B', '#C99A2E', '#176B63', '#176B63'][level],
    bars: pw.length < 8 ? 1 : sc,
  }
}

export default function Auth({ mode: initial }: { mode: Mode }) {
  const { login, signInWithToken, register } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const toast = useToast()
  const state = (location.state ?? {}) as { from?: string; note?: string }
  const [mode, setMode] = useState<Mode>(initial)
  const [f, setF] = useState({ name: '', email: '', password: '', remember: true, terms: false, type: 'owner' as AccountType })
  const [showPw, setShowPw] = useState(false)
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [loading, setLoading] = useState(false)
  const [faceOpen, setFaceOpen] = useState(false)
  const reg = mode === 'register'

  const set = <K extends keyof typeof f>(k: K, v: (typeof f)[K]) => {
    setF((s) => ({ ...s, [k]: v }))
    setErrors((e) => ({ ...e, [k]: '' }))
  }
  const switchMode = (m: Mode) => {
    setMode(m)
    setErrors({})
    window.history.replaceState(window.history.state, '', m === 'login' ? '/login' : '/register')
  }

  const submit = async (e?: FormEvent, creds?: { email: string; password: string }) => {
    e?.preventDefault()
    const email = creds?.email ?? f.email
    const password = creds?.password ?? f.password
    const er: Record<string, string> = {}
    if (reg && !creds && f.name.trim().length < 2) er.name = 'Enter your full name.'
    if (!EMAIL_RX.test(email)) er.email = 'Enter a valid email address, like name@example.com.'
    if (password.length < 8) er.password = 'Use at least 8 characters.'
    if (reg && !creds && !f.terms) er.terms = 'Please accept the terms to continue.'
    if (Object.keys(er).length) return setErrors(er)
    setLoading(true)
    try {
      if (reg && !creds) {
        await register({ name: f.name.trim(), email, password, account_type: f.type })
        toast('Account created. Welcome to PawConnect')
      } else {
        await login(email, password, f.remember)
        toast('Welcome back')
      }
      navigate(state.from ?? '/dashboard', { replace: true })
    } catch (err) {
      if (err instanceof ApiError) setErrors(Object.keys(err.errors).length ? err.errors : { form: err.message })
      else setErrors({ form: 'Something went wrong. Please try again.' })
    } finally {
      setLoading(false)
    }
  }

  const openFaceLogin = () => {
    if (!EMAIL_RX.test(f.email)) return setErrors({ email: 'Enter your email first, then sign in with your face.' })
    setErrors({})
    setFaceOpen(true)
  }

  const st = strength(f.password)
  const border = (k: string) => (errors[k] ? 'invalid' : '')

  return (
    <main style={{ minHeight: '100vh', display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(min(100%,460px),1fr))' }}>
      <div style={{ position: 'relative', minHeight: 320, background: 'var(--sand)', overflow: 'hidden' }}>
        <img src={img('https://images.unsplash.com/photo-1415369629372-26f2fe60c467', 1400)} alt="A person holding a cat’s paw" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' }} />
        <div style={{ position: 'absolute', inset: 0, background: 'linear-gradient(180deg,rgba(18,63,59,.15),rgba(18,63,59,.72))' }} />
        <div className="col between" style={{ position: 'relative', height: '100%', padding: 'clamp(24px,4vw,48px)', color: 'var(--cream)', gap: 40 }}>
          <div style={{ alignSelf: 'flex-start' }}><Logo dark /></div>
          <div style={{ maxWidth: 460 }}>
            <h2 style={{ margin: '0 0 18px', fontSize: 'clamp(30px,3.2vw,44px)', fontWeight: 800, letterSpacing: '-.03em', lineHeight: 1.05, textWrap: 'balance' }}>One account for owners, finders, adopters and shelters.</h2>
            <div className="col" style={{ gap: 12, fontSize: 15 }}>
              {['Get alerts when a similar animal is reported', 'Track adoption applications in one place', 'Your phone number stays private until you share it'].map((t) => (
                <div key={t} className="row" style={{ gap: 10 }}><Icon name="check_circle" size={20} style={{ color: 'var(--mint)' }} />{t}</div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="row" style={{ justifyContent: 'center', padding: 'clamp(28px,5vw,64px) clamp(16px,4vw,48px)' }}>
        <div className="col" style={{ width: '100%', maxWidth: 460, gap: 22 }}>
          <Link to="/" className="row" style={{ alignSelf: 'flex-start', gap: 6, fontSize: 14, fontWeight: 600, color: 'var(--muted)' }}><Icon name="arrow_back" size={18} />Back to home</Link>
          <Seg value={mode} onChange={switchMode} style={{ borderRadius: 14 }} options={[{ value: 'login', label: 'Sign in' }, { value: 'register', label: 'Create account' }]} />
          <div>
            <h1 style={{ margin: '0 0 6px', fontSize: 32, fontWeight: 800, letterSpacing: '-.03em' }}>{reg ? 'Create your account' : 'Welcome back'}</h1>
            <p style={{ fontSize: 15, color: 'var(--muted)', lineHeight: 1.5 }}>{reg ? 'Join owners, finders, adopters and shelters across Tunisia.' : 'Sign in to manage your reports, matches and applications.'}</p>
          </div>
          {state.note && <div className="info"><Icon name="info" size={20} />{state.note}</div>}
          {errors.form && <div className="alert" role="alert"><Icon name="error" size={20} />{errors.form}</div>}

          <form onSubmit={submit} noValidate className="col" style={{ gap: 16 }}>
            {reg && (
              <>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 8 }}>I’m joining as</div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,minmax(0,1fr))', gap: 8 }}>
                    {TYPES.map(([v, l, d, i]) => {
                      const on = f.type === v
                      return (
                        <button key={v} type="button" aria-pressed={on} onClick={() => set('type', v)} className="col" style={{ textAlign: 'left', padding: 12, borderRadius: 14, border: `1.5px solid ${on ? '#176B63' : '#E5D6C6'}`, background: on ? '#EEF6F4' : '#fff', boxShadow: on ? '0 0 0 3px rgba(23,107,99,.14)' : 'none', gap: 6, minWidth: 0 }}>
                          <Icon name={i} size={22} style={{ color: on ? '#176B63' : '#68747D' }} />
                          <span style={{ fontSize: 13, fontWeight: 800 }}>{l}</span>
                          <span style={{ fontSize: 11.5, color: 'var(--muted)', lineHeight: 1.35 }}>{d}</span>
                        </button>
                      )
                    })}
                  </div>
                </div>
                <label className="field">
                  <span className="field-label">Full name</span>
                  <input className={`input ${border('name')}`} style={{ height: 50 }} value={f.name} onChange={(e) => set('name', e.target.value)} placeholder="Yasmine Ben Salah" autoComplete="name" />
                  <FieldError msg={errors.name} />
                </label>
              </>
            )}
            <label className="field">
              <span className="field-label">Email</span>
              <input type="email" className={`input ${border('email')}`} style={{ height: 50 }} value={f.email} onChange={(e) => set('email', e.target.value)} placeholder="name@example.com" autoComplete="email" />
              <FieldError msg={errors.email} />
            </label>
            <div className="field">
              <span className="field-label">
                <label htmlFor="pw">Password</label>
                {!reg && <button type="button" className="btn-link" style={{ fontSize: 13 }} onClick={() => toast('Password reset isn’t available yet. Contact contact@pawconnect.tn for help.')}>Forgot password?</button>}
              </span>
              <div style={{ position: 'relative' }}>
                <input id="pw" type={showPw ? 'text' : 'password'} className={`input ${border('password')}`} style={{ height: 50, paddingRight: 50 }} value={f.password} onChange={(e) => set('password', e.target.value)} placeholder="At least 8 characters" autoComplete={reg ? 'new-password' : 'current-password'} />
                <button type="button" onClick={() => setShowPw(!showPw)} aria-label={showPw ? 'Hide password' : 'Show password'} style={{ position: 'absolute', right: 6, top: 6, width: 38, height: 38, border: 0, borderRadius: 9, background: 'transparent', display: 'grid', placeItems: 'center', color: 'var(--muted)' }}>
                  <Icon name={showPw ? 'visibility_off' : 'visibility'} size={21} />
                </button>
              </div>
              {reg && f.password && (
                <div className="row" style={{ gap: 10, marginTop: 8 }}>
                  <div style={{ flex: 1, display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 4 }}>
                    {[0, 1, 2, 3].map((i) => <div key={i} style={{ height: 4, borderRadius: 2, background: i < st.bars ? st.color : '#EADFD2' }} />)}
                  </div>
                  <span style={{ fontSize: 12, fontWeight: 700, color: st.color }}>{st.label}</span>
                </div>
              )}
              <FieldError msg={errors.password} />
            </div>
            {!reg && (
              <label className="check" style={{ alignItems: 'center' }}>
                <input type="checkbox" checked={f.remember} onChange={(e) => set('remember', e.target.checked)} style={{ marginTop: 0 }} />Remember me on this device
              </label>
            )}
            {reg && (
              <div>
                <label className="check">
                  <input type="checkbox" checked={f.terms} onChange={(e) => set('terms', e.target.checked)} />
                  I agree to the Terms and understand that AI results are suggestions, not verified identities.
                </label>
                <FieldError msg={errors.terms} />
              </div>
            )}
            <button type="submit" className="btn btn-primary btn-lg glow" disabled={loading}>
              {loading ? <Spinner /> : reg ? 'Create account' : 'Sign in'}
            </button>
          </form>
          {!reg && cameraSupported() && (
            <button type="button" className="btn btn-outline btn-lg" disabled={loading} onClick={openFaceLogin}>
              <Icon name="face" />Sign in with your face (AI)
            </button>
          )}
          {faceOpen && (
            <FaceLoginModal email={f.email.trim()} remember={f.remember} onClose={() => setFaceOpen(false)}
              onSignedIn={(r) => {
                signInWithToken(r, f.remember)
                setFaceOpen(false)
                toast('Welcome back')
                navigate(state.from ?? '/dashboard', { replace: true })
              }} />
          )}

          <div className="row" style={{ gap: 12, fontSize: 13, color: 'var(--muted)' }}>
            <div style={{ flex: 1, height: 1, background: '#EADCCD' }} />or<div style={{ flex: 1, height: 1, background: '#EADCCD' }} />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(180px,1fr))', gap: 10 }}>
            <button type="button" className="btn btn-outline" style={{ height: 48 }} onClick={() => toast('Google sign-in isn’t connected yet.')}>Continue with Google</button>
            <button type="button" className="btn btn-soft" style={{ height: 48 }} disabled={loading} onClick={() => submit(undefined, { email: 'yasmine@example.com', password: 'demo-password' })}>Use demo account</button>
          </div>
          <div style={{ fontSize: 14, color: 'var(--muted)', textAlign: 'center' }}>
            {reg ? 'Already have an account?' : 'New to PawConnect?'}{' '}
            <button type="button" className="btn-link" style={{ fontSize: 14, display: 'inline' }} onClick={() => switchMode(reg ? 'login' : 'register')}>{reg ? 'Sign in' : 'Create an account'}</button>
          </div>
        </div>
      </div>
    </main>
  )
}
