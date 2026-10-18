import { useCallback, useEffect, useRef, useState } from 'react'
import { cameraErrorMessage } from '../lib/face'

/** Webcam access: start/stop, and grab a frame as a JPEG data URL (not mirrored, as the camera sees it). */
export function useCamera() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [on, setOn] = useState(false)
  const [error, setError] = useState('')

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    setOn(false)
  }, [])

  // One camera at a time: concurrent start() calls share the same request (React runs effects twice in
  // development), and a camera that opens after the component is gone is switched off at once.
  const startingRef = useRef<Promise<boolean> | null>(null)
  const mountedRef = useRef(false)
  useEffect(() => {
    mountedRef.current = true
    return () => { mountedRef.current = false }
  }, [])

  const start = useCallback((): Promise<boolean> => {
    if (streamRef.current) return Promise.resolve(true)
    if (!startingRef.current) {
      startingRef.current = (async () => {
        setError('')
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false })
          if (!mountedRef.current) {
            stream.getTracks().forEach((t) => t.stop())
            return false
          }
          streamRef.current = stream
          if (videoRef.current) {
            videoRef.current.srcObject = stream
            await videoRef.current.play()
          }
          setOn(true)
          return true
        } catch (e) {
          setError(cameraErrorMessage(e))
          return false
        } finally {
          startingRef.current = null
        }
      })()
    }
    return startingRef.current
  }, [])

  const capture = useCallback((): string => {
    const v = videoRef.current
    if (!v || !v.videoWidth) throw new Error('camera not ready')
    const scale = Math.min(1, 640 / v.videoWidth)
    const c = document.createElement('canvas')
    c.width = Math.round(v.videoWidth * scale)
    c.height = Math.round(v.videoHeight * scale)
    c.getContext('2d')!.drawImage(v, 0, 0, c.width, c.height)
    return c.toDataURL('image/jpeg', 0.88)
  }, [])

  useEffect(() => stop, [stop])
  return { videoRef, on, error, start, stop, capture }
}
