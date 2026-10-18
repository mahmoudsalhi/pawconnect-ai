import { useEffect, useState } from 'react'
import { ApiError, type User } from '../lib/api'
import { faceChallenge, faceLogin, sleep, type TurnDirection } from '../lib/face'
import { CameraView } from './FaceCamera'
import { useCamera } from './useCamera'
import { Modal } from './Modal'
import { Icon } from './ui'

type Phase = 'intro' | 'front' | 'turn' | 'checking' | 'error'

/** Sign in with the face: one photo facing the camera, then a few while turning the head the asked way. */
export function FaceLoginModal({ email, remember, onSignedIn, onClose }: {
  email: string
  remember: boolean
  onSignedIn: (r: { access_token: string; user: User }) => void
  onClose: () => void
}) {
  const cam = useCamera()
  const [phase, setPhase] = useState<Phase>('intro')
  const [count, setCount] = useState(3)
  const [direction, setDirection] = useState<TurnDirection>('left')
  const [error, setError] = useState('')
  const { start } = cam
  useEffect(() => { start() }, [start])  // preview straight away, so the person can get into the oval

  const close = () => {
    cam.stop()
    onClose()
  }

  const run = async () => {
    setError('')
    if (!cam.on && !(await cam.start())) return
    try {
      const ch = await faceChallenge(email)
      setDirection(ch.direction)
      setPhase('front')
      for (let i = 3; i > 0; i--) {
        setCount(i)
        await sleep(700)
      }
      const front = cam.capture()
      setPhase('turn')
      await sleep(1300)
      const turned: string[] = []
      for (let i = 0; i < 3; i++) {
        turned.push(cam.capture())
        await sleep(280)
      }
      setPhase('checking')
      const r = await faceLogin({ email, state: ch.state, front, turned, remember })
      cam.stop()
      onSignedIn(r)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Something went wrong. Please try again.')
      setPhase('error')
    }
  }

  const arrow = direction === 'left' ? '←' : '→'  // the preview is mirrored, so the arrow points the way to turn
  const prompt = {
    intro: cam.on ? 'Ready when you are' : undefined,
    front: `Look straight at the camera… ${count}`,
    turn: <span>{arrow} Now slowly turn your head to your <u>{direction.toUpperCase()}</u> {arrow}</span>,
    checking: 'Checking your face…',
    error: undefined,
  }[phase]

  return (
    <Modal title="Sign in with your face" subtitle={email} onClose={close}>
      <div className="col" style={{ gap: 14 }}>
        <CameraView videoRef={cam.videoRef} on={cam.on} prompt={prompt} busy={phase === 'checking'} />
        {(cam.error || error) && <div className="alert" role="alert"><Icon name="error" size={20} />{cam.error || error}</div>}
        {(phase === 'intro' || phase === 'error') && (
          <>
            <div style={{ fontSize: 13.5, color: 'var(--muted)', lineHeight: 1.5 }}>
              First look straight at the camera, then turn your head the way we ask. Good light helps. The photos are checked by PawConnect’s face model and are not saved.
            </div>
            <button type="button" className="btn btn-primary btn-lg" onClick={run}><Icon name="face" />{phase === 'error' ? 'Try again' : 'Start'}</button>
          </>
        )}
      </div>
    </Modal>
  )
}
