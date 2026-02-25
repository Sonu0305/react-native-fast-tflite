/* eslint-disable @typescript-eslint/no-var-requires */
import * as React from 'react'

import {
  ActivityIndicator,
  Image,
  NativeModules,
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
  useCameraPermission,
} from 'react-native-vision-camera'

import { CHAR_DICT } from './charDict'
import { InferenceResult, RgbImage, runInference } from './inference'

interface DecodedImagePayload {
  width: number
  height: number
  rgbBase64: string
}

interface ImageDecoderSpec {
  decodeImage(path: string, maxDimension: number): Promise<DecodedImagePayload>
}

interface DetectLatencyBreakdown {
  decodeImageMs: number
  base64DecodeMs: number
  imageBuildMs: number
  inputPrepareMs: number
  inferenceCallMs: number
  endToEndMs: number
}

const ImageDecoder = NativeModules.ImageDecoder as ImageDecoderSpec | undefined

const BASE64_ALPHABET =
  'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'

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

function decodeBase64ToBytes(base64: string): Uint8Array {
  const clean = base64.replace(/\s/g, '')
  const padding = clean.endsWith('==') ? 2 : clean.endsWith('=') ? 1 : 0
  const outputLength = (clean.length * 3) / 4 - padding
  const output = new Uint8Array(outputLength)

  const readChar = (char: string): number => {
    if (char === '=') {
      return 0
    }
    const idx = BASE64_ALPHABET.indexOf(char)
    if (idx < 0) {
      throw new Error('Invalid base64 data from image decoder.')
    }
    return idx
  }

  let outIndex = 0

  for (let i = 0; i < clean.length; i += 4) {
    const c0 = readChar(clean.charAt(i))
    const c1 = readChar(clean.charAt(i + 1))
    const c2 = readChar(clean.charAt(i + 2))
    const c3 = readChar(clean.charAt(i + 3))

    const packed = (c0 << 18) | (c1 << 12) | (c2 << 6) | c3

    if (outIndex < outputLength) {
      output[outIndex] = (packed >> 16) & 255
      outIndex += 1
    }
    if (outIndex < outputLength) {
      output[outIndex] = (packed >> 8) & 255
      outIndex += 1
    }
    if (outIndex < outputLength) {
      output[outIndex] = packed & 255
      outIndex += 1
    }
  }

  return output
}

export default function App(): React.ReactNode {
  const cameraRef = React.useRef<Camera>(null)

  const { hasPermission, requestPermission } = useCameraPermission()
  const device = useCameraDevice('back')

  const detPlugin = useTensorflowModel(require('../assets/det_model.tflite'))
  const recPlugin = useTensorflowModel(require('../assets/rec_model.tflite'))

  const detModel = detPlugin.state === 'loaded' ? detPlugin.model : undefined
  const recModel = recPlugin.state === 'loaded' ? recPlugin.model : undefined

  const [capturedPhoto, setCapturedPhoto] = React.useState<PhotoFile | null>(null)
  const [isDetecting, setIsDetecting] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [result, setResult] = React.useState<InferenceResult | null>(null)
  const [detectLatency, setDetectLatency] = React.useState<DetectLatencyBreakdown | null>(null)

  React.useEffect(() => {
    requestPermission()
  }, [requestPermission])

  const modelsReady = detModel != null && recModel != null

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
    setCapturedPhoto(null)
    setResult(null)
    setDetectLatency(null)
    setError(null)
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
    if (ImageDecoder == null) {
      setError('Native ImageDecoder module is unavailable.')
      return
    }

    try {
      const detectStartedAt = nowMs()
      setIsDetecting(true)
      setError(null)
      setResult(null)
      setDetectLatency(null)

      const decodeStartedAt = nowMs()
      const decoded = await ImageDecoder.decodeImage(capturedPhoto.path, 1280)
      const decodeImageMs = nowMs() - decodeStartedAt

      const base64DecodeStartedAt = nowMs()
      const decodedBytes = decodeBase64ToBytes(decoded.rgbBase64)
      const base64DecodeMs = nowMs() - base64DecodeStartedAt

      const imageBuildStartedAt = nowMs()
      const image: RgbImage = {
        width: decoded.width,
        height: decoded.height,
        data: decodedBytes,
      }
      const imageBuildMs = nowMs() - imageBuildStartedAt

      const inferenceStartedAt = nowMs()
      const inference = await runInference(image, detModel, recModel, CHAR_DICT)
      const inferenceCallMs = nowMs() - inferenceStartedAt
      const endToEndMs = nowMs() - detectStartedAt

      setResult(inference)
      setDetectLatency({
        decodeImageMs,
        base64DecodeMs,
        imageBuildMs,
        inputPrepareMs: decodeImageMs + base64DecodeMs + imageBuildMs,
        inferenceCallMs,
        endToEndMs,
      })
    } catch (e) {
      setError(`Detection failed: ${formatError(e)}`)
    } finally {
      setIsDetecting(false)
    }
  }, [capturedPhoto, modelsReady, detModel, recModel])

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

  const photoUri = capturedPhoto != null ? toFileUri(capturedPhoto.path) : undefined
  const parsedWeight =
    result?.value != null
      ? `${result.value}${result.unit != null && result.unit.length > 0 ? ` ${result.unit}` : ''}`
      : 'No reading'

  return (
    <SafeAreaView style={styles.root}>
      <View style={styles.previewCard}>
        {capturedPhoto == null ? (
          <Camera
            ref={cameraRef}
            style={StyleSheet.absoluteFill}
            isActive={true}
            device={device}
            photo={true}
            onError={(cameraError) => {
              setError(`Camera error: ${cameraError.message}`)
            }}
          />
        ) : (
          <Image source={{ uri: photoUri }} style={StyleSheet.absoluteFill} resizeMode="contain" />
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
              disabled={isDetecting}
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

        {modelError != null && <Text style={styles.errorText}>{modelError}</Text>}
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
            <Text style={styles.resultLine}>Detected boxes: {result.boxes.length}</Text>
            <Text style={styles.resultLine}>
              OCR confidence: {result.recConf.toFixed(3)}
            </Text>
            <Text style={styles.resultSectionTitle}>Latency Breakdown</Text>
            <Text style={styles.resultLine}>
              End-to-end detect(): {formatMs(detectLatency?.endToEndMs)}
            </Text>
            <Text style={styles.resultLine}>
              decodeImage (native): {formatMs(detectLatency?.decodeImageMs)}
            </Text>
            <Text style={styles.resultLine}>
              base64 -> RGB bytes: {formatMs(detectLatency?.base64DecodeMs)}
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
              Detection preprocess: {formatMs(result.latency.detectionPreprocessMs)}
            </Text>
            <Text style={styles.resultLine}>
              Detection inference: {formatMs(result.latency.detectionInferenceMs)}
            </Text>
            <Text style={styles.resultLine}>
              Detection postprocess: {formatMs(result.latency.detectionPostprocessMs)}
            </Text>
            <Text style={styles.resultLine}>
              Recognition total loop: {formatMs(result.latency.recognitionTotalMs)}
            </Text>
            <Text style={styles.resultLine}>
              Recognition crop: {formatMs(result.latency.recognitionCropMs)}
            </Text>
            <Text style={styles.resultLine}>
              Recognition preprocess: {formatMs(result.latency.recognitionPreprocessMs)}
            </Text>
            <Text style={styles.resultLine}>
              Recognition inference: {formatMs(result.latency.recognitionInferenceMs)}
            </Text>
            <Text style={styles.resultLine}>
              Recognition decode: {formatMs(result.latency.recognitionDecodeMs)}
            </Text>
            <Text style={styles.resultLine}>
              Recognition other/overhead: {formatMs(result.latency.recognitionOtherMs)}
            </Text>
            <Text style={styles.resultLine}>Parse weight: {formatMs(result.latency.parseMs)}</Text>
            <Text style={styles.resultLine}>
              Crops processed: {result.latency.cropsProcessed}/{result.latency.boxesDetected}
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
  resultLine: {
    color: '#e2e8f0',
    fontSize: 13,
  },
})
