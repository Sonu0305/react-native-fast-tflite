/* eslint-disable @typescript-eslint/no-var-requires */
import * as React from 'react'

import {
  ActivityIndicator,
  Image,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native'
import { useTensorflowModel } from 'react-native-fast-tflite'
import {
  Camera,
  PhotoFile,
  useCameraDevice,
  useCameraFormat,
  useCameraPermission,
} from 'react-native-vision-camera'

import { CHAR_DICT } from './charDict'
import { decodeImageToArrayBuffer } from './imageDecoder'
import { InferenceResult, RgbImage, runInference } from './inference'

interface DetectLatencyBreakdown {
  decodeImageMs: number
  byteArrayMs: number
  imageBuildMs: number
  inputPrepareMs: number
  inferenceCallMs: number
  endToEndMs: number
}

function toFileUri(path: string): string {
  return path.startsWith('file://') ? path : `file://${path}`
}

function formatError(error: unknown): string {
  if (error instanceof Error) {
    return error.message
  }
  return String(error)
}

function modelStateToMessage(state: 'loading' | 'loaded' | 'error'): string {
  if (state === 'loading') {
    return 'Loading...'
  }
  if (state === 'loaded') {
    return 'Loaded'
  }
  return 'Error'
}

function nowMs(): number {
  if (globalThis.performance?.now != null) {
    return globalThis.performance.now()
  }
  return Date.now()
}

function formatMs(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) {
    return '-'
  }
  return `${value.toFixed(1)} ms`
}

const MAX_RECOGNITION_CROPS = 2
const SHOW_DEBUG_IMAGES: boolean = false

export default function App(): React.ReactNode {
  const cameraRef = React.useRef<Camera>(null)
  const detectRunIdRef = React.useRef(0)

  const { hasPermission, requestPermission } = useCameraPermission()
  const device = useCameraDevice('back')
  const preferredFormat = useCameraFormat(device, [
    { photoResolution: { width: 1280, height: 720 } },
    { fps: 30 },
  ])
  const fallbackFormat = useCameraFormat(device, [
    { photoResolution: { width: 640, height: 480 } },
    { fps: 30 },
  ])
  const detPlugin = useTensorflowModel(
    require('../assets/yolov26n_no_nms_float16.tflite'),
    'android-gpu'
  )
  const recPlugin = useTensorflowModel(
    require('../assets/rec_model.tflite'),
    'android-gpu'
  )
  const detModel = detPlugin.state === 'loaded' ? detPlugin.model : undefined
  const recModel = recPlugin.state === 'loaded' ? recPlugin.model : undefined

  // React.useEffect(() => {
  //   if (detModel !== undefined && recModel !== undefined) {
  //     console.log('DET input dtype:', detModel.inputs?.[0]?.dataType)
  //     console.log('DET output dtype:', detModel.outputs?.[0]?.dataType)
  //     console.log('REC input dtype:', recModel.inputs?.[0]?.dataType)
  //     console.log('REC output dtype:', recModel.outputs?.[0]?.dataType)
  //   }
  // }, [detModel, recModel])

  // React.useEffect(() => {
  //   if (detModel != null && recModel != null) {
  //     const dummy = new Float32Array(1 * 416 * 416 * 3).fill(0)
  //     detModel.run([dummy]).catch(() => {})
  //   }
  // }, [detModel, recModel])

  // React.useEffect(() => {
  //   if (detModel != null) {
  //     console.log('DET output shape:', JSON.stringify(detModel.outputs))
  //   }
  // }, [detModel])

  const [capturedPhoto, setCapturedPhoto] = React.useState<PhotoFile | null>(
    null
  )
  const [isDetecting, setIsDetecting] = React.useState(false)
  const [useCameraFallback, setUseCameraFallback] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [result, setResult] = React.useState<InferenceResult | null>(null)
  const [detectLatency, setDetectLatency] =
    React.useState<DetectLatencyBreakdown | null>(null)

  React.useEffect(() => {
    requestPermission()
  }, [requestPermission])
  const modelsReady = detModel != null && recModel != null
  const debugImagesEnabled = SHOW_DEBUG_IMAGES === true
  const modelError =
    detPlugin.state === 'error'
      ? `Detection model: ${detPlugin.error.message}`
      : recPlugin.state === 'error'
        ? `Recognition model: ${recPlugin.error.message}`
        : null

  const onTakePhoto = React.useCallback(async () => {
    if (!hasPermission) {
      await requestPermission()
      return
    }

    if (cameraRef.current == null) {
      setError('Camera is not initialized yet. Try again in a second.')
      return
    }

    try {
      setError(null)
      setResult(null)
      setDetectLatency(null)
      const photo = await cameraRef.current.takePhoto({
        enableShutterSound: true,
      })
      setCapturedPhoto(photo)
    } catch (e) {
      setError(`Failed to capture photo: ${formatError(e)}`)
    }
  }, [hasPermission, requestPermission])

  const onRetake = React.useCallback(() => {
    detectRunIdRef.current += 1
    setCapturedPhoto(null)
    setResult(null)
    setDetectLatency(null)
    setError(null)
    setIsDetecting(false)
  }, [])

  const onDetect = React.useCallback(async () => {
    if (capturedPhoto == null) {
      setError('Take a photo first.')
      return
    }
    if (!modelsReady || detModel == null || recModel == null) {
      setError('Models are not ready yet. Please wait.')
      return
    }
    try {
      const runId = ++detectRunIdRef.current
      const detectStartedAt = nowMs()
      setIsDetecting(true)
      setError(null)
      setResult(null)
      setDetectLatency(null)
      await yieldToUi(runId, detectRunIdRef)

      const inputSize = 416

      const decodeStartedAt = nowMs()
      const decoded = await decodeImageToArrayBuffer(
        capturedPhoto.path,
        inputSize
      )
      if (runId !== detectRunIdRef.current) {
        return
      }
      const decodeImageMs = nowMs() - decodeStartedAt
      const byteArrayMs = 0
      const imageBuildMs = 0

      const decodedBytes = new Uint8Array(decoded.rgbBytes)
      const image: RgbImage = {
        width: decoded.width,
        height: decoded.height,
        data: decodedBytes,
      }

      const inferenceStartedAt = nowMs()
      const inference = await runInference(
        image,
        detModel,
        recModel,
        CHAR_DICT,
        0.25,
        0.5,
        MAX_RECOGNITION_CROPS,
        debugImagesEnabled
      )
      if (runId !== detectRunIdRef.current) {
        return
      }
      const inferenceCallMs = nowMs() - inferenceStartedAt
      const endToEndMs = nowMs() - detectStartedAt

      setResult(inference)
      setDetectLatency({
        decodeImageMs,
        byteArrayMs,
        imageBuildMs,
        inputPrepareMs: decodeImageMs + byteArrayMs + imageBuildMs,
        inferenceCallMs,
        endToEndMs,
      })
    } catch (e) {
      const message = formatError(e)
      if (message !== 'Detection canceled.') {
        setError(`Detection failed: ${message}`)
      }
    } finally {
      setIsDetecting(false)
    }
  }, [capturedPhoto, modelsReady, detModel, recModel])

  function ensureActive(
    runId: number,
    runRef: React.MutableRefObject<number>
  ): void {
    if (runId !== runRef.current) {
      throw new Error('Detection canceled.')
    }
  }

  async function yieldToUi(
    runId: number,
    runRef: React.MutableRefObject<number>
  ): Promise<void> {
    await new Promise<void>((resolve) => {
      requestAnimationFrame(() => resolve())
    })
    ensureActive(runId, runRef)
  }

  if (!hasPermission) {
    return (
      <SafeAreaView style={styles.root}>
        <View style={styles.centered}>
          <Text style={styles.title}>Camera permission is required.</Text>
          <Pressable style={styles.primaryButton} onPress={requestPermission}>
            <Text style={styles.buttonText}>Grant Camera Permission</Text>
          </Pressable>
        </View>
      </SafeAreaView>
    )
  }

  if (device == null) {
    return (
      <SafeAreaView style={styles.root}>
        <View style={styles.centered}>
          <Text style={styles.title}>No back camera device found.</Text>
        </View>
      </SafeAreaView>
    )
  }

  const photoUri =
    capturedPhoto != null ? toFileUri(capturedPhoto.path) : undefined
  const parsedWeight =
    result?.value != null
      ? `${result.value}${result.unit != null && result.unit.length > 0 ? ` ${result.unit}` : ''}`
      : 'No reading'

  return (
    <SafeAreaView style={styles.root}>
      <View style={styles.previewCard}>
        {capturedPhoto == null ? (
          <Camera
            key={useCameraFallback ? 'camera-fallback' : 'camera-default'}
            ref={cameraRef}
            style={StyleSheet.absoluteFill}
            device={device}
            format={useCameraFallback ? fallbackFormat : preferredFormat}
            isActive={true}
            photo={true}
            onError={(e) => {
              const msg = e.message ?? ''
              const low = msg.toLowerCase()
              const isInvalidOutputConfig =
                low.includes('invalid-output-configuration') ||
                low.includes('output/stream configurations are invalid')

              if (isInvalidOutputConfig && !useCameraFallback) {
                setUseCameraFallback(true)
                setError(
                  'Switched to safe camera format due to stream config error.'
                )
                return
              }

              setError(`Camera error: ${msg}`)
            }}
          />
        ) : (
          <Image
            source={{ uri: photoUri }}
            style={StyleSheet.absoluteFill}
            resizeMode="contain"
          />
        )}

        {isDetecting && (
          <View style={styles.loadingOverlay}>
            <ActivityIndicator size="large" color="#ffffff" />
            <Text style={styles.loadingText}>Running detection...</Text>
          </View>
        )}
      </View>

      <View style={styles.controlsRow}>
        {capturedPhoto == null ? (
          <Pressable
            style={[
              styles.primaryButton,
              styles.buttonSpacing,
              isDetecting && styles.disabledButton,
            ]}
            disabled={isDetecting}
            onPress={onTakePhoto}
          >
            <Text style={styles.buttonText}>Take Photo</Text>
          </Pressable>
        ) : (
          <>
            <Pressable
              style={[
                styles.secondaryButton,
                styles.buttonSpacing,
                isDetecting && styles.disabledButton,
              ]}
              disabled={false}
              onPress={onRetake}
            >
              <Text style={styles.secondaryButtonText}>Retake</Text>
            </Pressable>
            <Pressable
              style={[
                styles.primaryButton,
                styles.buttonSpacing,
                (!modelsReady || isDetecting) && styles.disabledButton,
              ]}
              disabled={!modelsReady || isDetecting}
              onPress={onDetect}
            >
              <Text style={styles.buttonText}>Detect</Text>
            </Pressable>
          </>
        )}
      </View>

      <ScrollView contentContainerStyle={styles.resultsContainer}>
        <Text style={styles.statusLabel}>
          Detection model: {modelStateToMessage(detPlugin.state)}
        </Text>
        <Text style={styles.statusLabel}>
          Recognition model: {modelStateToMessage(recPlugin.state)}
        </Text>

        {modelError != null && (
          <Text style={styles.errorText}>{modelError}</Text>
        )}
        {error != null && <Text style={styles.errorText}>{error}</Text>}

        {capturedPhoto != null && (
          <Text style={styles.metaText}>
            Captured image: {capturedPhoto.width}x{capturedPhoto.height}
          </Text>
        )}

        {result != null && (
          <View style={styles.resultCard}>
            <Text style={styles.resultTitle}>Detected Weight</Text>
            <Text style={styles.resultWeight}>{parsedWeight}</Text>
            <Text style={styles.resultLine}>Raw OCR: {result.combined}</Text>
            <Text style={styles.resultLine}>
              Detected boxes: {result.boxes.length}
            </Text>
            <Text style={styles.resultLine}>
              OCR confidence: {result.recConf.toFixed(3)}
            </Text>
            {debugImagesEnabled && (
              <>
                <Text style={styles.resultSectionTitle}>Model Inputs</Text>
                <View style={styles.modelInputRow}>
                  <View style={styles.modelInputCard}>
                    <Text style={styles.modelInputLabel}>
                      Detection (320x320)
                    </Text>
                    {result.detectionInputBase64 != null ? (
                      <Image
                        source={{
                          uri: `data:image/png;base64,${result.detectionInputBase64}`,
                        }}
                        style={styles.modelInputImage}
                      />
                    ) : (
                      <Text style={styles.modelInputEmpty}>
                        No detection input
                      </Text>
                    )}
                  </View>
                  <View style={styles.modelInputCard}>
                    <Text style={styles.modelInputLabel}>
                      Recognition crop (first)
                    </Text>
                    {result.recognitionInputsBase64.length > 0 ? (
                      <Image
                        source={{
                          uri: `data:image/png;base64,${result.recognitionInputsBase64[0]}`,
                        }}
                        style={styles.modelInputImage}
                        resizeMode="contain"
                      />
                    ) : (
                      <Text style={styles.modelInputEmpty}>No crop</Text>
                    )}
                  </View>
                </View>
                <View style={styles.modelInputRow}>
                  <View
                    style={[styles.modelInputCard, styles.modelInputCardWide]}
                  >
                    <Text style={styles.modelInputLabel}>
                      Detection crop (first box, 320x320)
                    </Text>
                    {result.detectionCrop320Base64 != null ? (
                      <Image
                        source={{
                          uri: `data:image/png;base64,${result.detectionCrop320Base64}`,
                        }}
                        style={styles.modelInputImage}
                        resizeMode="contain"
                      />
                    ) : (
                      <Text style={styles.modelInputEmpty}>
                        No detection crop
                      </Text>
                    )}
                  </View>
                </View>
              </>
            )}
            <Text style={styles.resultSectionTitle}>Latency Breakdown</Text>
            <Text style={styles.resultLine}>
              End-to-end detect(): {formatMs(detectLatency?.endToEndMs)}
            </Text>
            <Text style={styles.resultLine}>
              decodeImage (native): {formatMs(detectLatency?.decodeImageMs)}
            </Text>
            <Text style={styles.resultLine}>
              JS byte array build: {formatMs(detectLatency?.byteArrayMs)}
            </Text>
            <Text style={styles.resultLine}>
              RGB object build: {formatMs(detectLatency?.imageBuildMs)}
            </Text>
            <Text style={styles.resultLine}>
              Input prep total: {formatMs(detectLatency?.inputPrepareMs)}
            </Text>
            <Text style={styles.resultLine}>
              runInference() call: {formatMs(detectLatency?.inferenceCallMs)}
            </Text>
            <Text style={styles.resultLine}>
              Pipeline total: {formatMs(result.latency.totalMs)}
            </Text>
            <Text style={styles.resultLine}>
              Detection total: {formatMs(result.latency.detectionTotalMs)}
            </Text>
            <Text style={styles.resultLine}>
              Detection preprocess:{' '}
              {formatMs(result.latency.detectionPreprocessMs)}
            </Text>
            <Text style={styles.resultLine}>
              Detection inference:{' '}
              {formatMs(result.latency.detectionInferenceMs)}
            </Text>
            <Text style={styles.resultLine}>
              Detection postprocess:{' '}
              {formatMs(result.latency.detectionPostprocessMs)}
            </Text>
            <Text style={styles.resultLine}>
              Recognition total loop:{' '}
              {formatMs(result.latency.recognitionTotalMs)}
            </Text>
            <Text style={styles.resultLine}>
              Recognition crop: {formatMs(result.latency.recognitionCropMs)}
            </Text>
            <Text style={styles.resultLine}>
              Recognition preprocess:{' '}
              {formatMs(result.latency.recognitionPreprocessMs)}
            </Text>
            <Text style={styles.resultLine}>
              Recognition inference:{' '}
              {formatMs(result.latency.recognitionInferenceMs)}
            </Text>
            <Text style={styles.resultLine}>
              Recognition decode: {formatMs(result.latency.recognitionDecodeMs)}
            </Text>
            <Text style={styles.resultLine}>
              Recognition other/overhead:{' '}
              {formatMs(result.latency.recognitionOtherMs)}
            </Text>
            <Text style={styles.resultLine}>
              Parse weight: {formatMs(result.latency.parseMs)}
            </Text>
            <Text style={styles.resultLine}>
              Crops processed: {result.latency.cropsProcessed}/
              {result.latency.boxesDetected}
            </Text>
            <Text style={styles.resultLine}>
              Legacy total latency: {result.timeMs.toFixed(1)} ms
            </Text>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: '#0f172a',
  },
  centered: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 20,
  },
  title: {
    color: '#e2e8f0',
    fontSize: 18,
    textAlign: 'center',
    marginBottom: 16,
  },
  previewCard: {
    height: '52%',
    margin: 12,
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: '#020617',
    borderWidth: 1,
    borderColor: '#1e293b',
  },
  controlsRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 12,
  },
  buttonSpacing: {
    marginHorizontal: 5,
  },
  primaryButton: {
    minWidth: 130,
    paddingVertical: 12,
    paddingHorizontal: 18,
    borderRadius: 10,
    backgroundColor: '#0284c7',
    alignItems: 'center',
  },
  secondaryButton: {
    minWidth: 130,
    paddingVertical: 12,
    paddingHorizontal: 18,
    borderRadius: 10,
    backgroundColor: '#cbd5e1',
    alignItems: 'center',
  },
  buttonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '700',
  },
  secondaryButtonText: {
    color: '#0f172a',
    fontSize: 16,
    fontWeight: '700',
  },
  disabledButton: {
    opacity: 0.45,
  },
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(15, 23, 42, 0.78)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  loadingText: {
    color: '#e2e8f0',
    fontSize: 14,
    marginTop: 8,
  },
  resultsContainer: {
    paddingHorizontal: 14,
    paddingTop: 14,
    paddingBottom: 28,
  },
  statusLabel: {
    color: '#cbd5e1',
    fontSize: 14,
  },
  errorText: {
    color: '#fca5a5',
    fontSize: 13,
  },
  metaText: {
    color: '#93c5fd',
    fontSize: 13,
  },
  resultCard: {
    marginTop: 8,
    borderRadius: 12,
    padding: 14,
    backgroundColor: '#111827',
    borderWidth: 1,
    borderColor: '#1f2937',
  },
  resultTitle: {
    color: '#cbd5e1',
    fontSize: 14,
    fontWeight: '700',
  },
  resultWeight: {
    color: '#f8fafc',
    fontSize: 28,
    fontWeight: '800',
  },
  resultSectionTitle: {
    color: '#93c5fd',
    fontSize: 13,
    fontWeight: '700',
    marginTop: 8,
  },
  modelInputRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 8,
  },
  modelInputCard: {
    width: '48%',
    borderRadius: 10,
    backgroundColor: '#0b1220',
    borderWidth: 1,
    borderColor: '#1f2937',
    padding: 8,
  },
  modelInputCardWide: {
    width: '100%',
  },
  modelInputLabel: {
    color: '#cbd5e1',
    fontSize: 12,
    marginBottom: 6,
  },
  modelInputImage: {
    width: '100%',
    aspectRatio: 1,
    borderRadius: 6,
    backgroundColor: '#020617',
  },
  modelInputEmpty: {
    color: '#94a3b8',
    fontSize: 12,
    paddingVertical: 18,
    textAlign: 'center',
  },
  resultLine: {
    color: '#e2e8f0',
    fontSize: 13,
  },
})
