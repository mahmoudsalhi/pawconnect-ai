import type { ReactNode } from 'react'
import { Icon, Spinner } from './ui'

/** Mirrored live preview (like a mirror, which feels natural) with an oval guide and a prompt. */
export function CameraView({ videoRef, on, prompt, busy }: { videoRef: React.RefObject<HTMLVideoElement | null>; on: boolean; prompt?: ReactNode; busy?: boolean }) {
  return (
    <div style={{ position: 'relative', width: '100%', aspectRatio: '4 / 3', borderRadius: 18, overflow: 'hidden', background: '#141D21' }}>
      <video ref={videoRef} muted playsInline style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)', opacity: on ? 1 : 0 }} />
      {!on && <div className="col" style={{ position: 'absolute', inset: 0, alignItems: 'center', justifyContent: 'center', gap: 8, color: '#C5D8D4' }}><Icon name="photo_camera" size={36} /><span style={{ fontSize: 13 }}>Camera is off</span></div>}
      {on && <div aria-hidden style={{ position: 'absolute', left: '50%', top: '47%', width: '42%', height: '72%', transform: 'translate(-50%,-50%)', border: '3px dashed rgba(255,255,255,.7)', borderRadius: '50%' }} />}
      {prompt && (
        <div className="row" role="status" style={{ position: 'absolute', left: 12, right: 12, bottom: 12, gap: 10, justifyContent: 'center', padding: '10px 14px', borderRadius: 14, background: 'rgba(18,63,59,.88)', color: '#fff', fontSize: 15, fontWeight: 800, textAlign: 'center' }}>
          {busy && <Spinner />}{prompt}
        </div>
      )}
    </div>
  )
}
