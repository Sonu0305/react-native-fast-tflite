// @ts-nocheck
import { TensorflowModel } from 'react-native-fast-tflite'

export interface RgbImage {
  data: Uint8Array
  width: number
  height: number
}

export interface DetectionBox {
  x1: number
  y1: number
  x2: number
  y2: number
  conf: number
}

const DETECTION_PAD_LEFT_RATIO = 0.2
const DETECTION_PAD_RIGHT_RATIO = 0.08
const DETECTION_PAD_TOP_RATIO = 0.1
const DETECTION_PAD_BOTTOM_RATIO = 0.1

export interface RecognitionEntry {
  text: string
  detConf: number
  recConf: number
}

export interface InferenceLatencyBreakdown {
  detectionPreprocessMs: number
  detectionInferenceMs: number
  detectionPostprocessMs: number
  detectionTotalMs: number
  recognitionCropMs: number
  recognitionPreprocessMs: number
  recognitionInferenceMs: number
  recognitionDecodeMs: number
  recognitionLoopMs: number
  recognitionOtherMs: number
  recognitionTotalMs: number
  parseMs: number
  totalMs: number
  boxesDetected: number
  cropsProcessed: number
}

export interface InferenceResult {
  boxes: DetectionBox[]
  texts: RecognitionEntry[]
  combined: string
  value: string | null
  unit: string | null
  recConf: number
  detectionInputBase64: string | null
  detectionCrop320Base64: string | null
  recognitionInputsBase64: string[]
  timeMs: number
  latency: InferenceLatencyBreakdown
}

interface DetPreprocessResult {
  tensor: Float32Array
  scale: number
  padW: number
  padH: number
}

interface DecodedText {
  text: string
  confidence: number
}

const EMPTY_STRINGS: string[] = []

let detTensorCache: Float32Array | null = null
let detTensorCacheSize = 0

let recTensorCache: Float32Array | null = null
let recTensorCacheSize = 0

let recResizeCache: Uint8Array | null = null
let recResizeCacheSize = 0

function nowMs(): number {
  if (globalThis.performance?.now != null) {
    return globalThis.performance.now()
  }
  return Date.now()
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function clampByte(value: number): number {
  return value < 0 ? 0 : value > 255 ? 255 : value
}

function toFloat32Array(tensor: ArrayLike<number>): Float32Array {
  if (tensor instanceof Float32Array) {
    return tensor
  }
  return Float32Array.from(tensor)
}

function tensorToRgbBytes(
  tensor: Float32Array,
  width: number,
  height: number,
  mode: 'zeroOne' | 'minusOneOne'
): Uint8Array {
  const out = new Uint8Array(width * height * 3)
  if (mode === 'zeroOne') {
    for (let i = 0; i < out.length; i += 1) {
      out[i] = clampByte(Math.round(tensor[i] * 255))
    }
  } else {
    for (let i = 0; i < out.length; i += 1) {
      out[i] = clampByte(Math.round((tensor[i] + 1) * 127.5))
    }
  }
  return out
}

function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff
  for (let i = 0; i < bytes.length; i += 1) {
    crc ^= bytes[i]
    for (let j = 0; j < 8; j += 1) {
      const mask = -(crc & 1)
      crc = (crc >>> 1) ^ (0xedb88320 & mask)
    }
  }
  return (crc ^ 0xffffffff) >>> 0
}

function adler32(bytes: Uint8Array): number {
  let a = 1
  let b = 0
  for (let i = 0; i < bytes.length; i += 1) {
    a = (a + bytes[i]) % 65521
    b = (b + a) % 65521
  }
  return ((b << 16) | a) >>> 0
}

function pushUint32BE(target: number[], value: number): void {
  target.push((value >>> 24) & 0xff)
  target.push((value >>> 16) & 0xff)
  target.push((value >>> 8) & 0xff)
  target.push(value & 0xff)
}

function pushBytes(target: number[], bytes: Uint8Array): void {
  for (let i = 0; i < bytes.length; i += 1) {
    target.push(bytes[i])
  }
}

function encodePngBase64Rgb(
  rgb: Uint8Array,
  width: number,
  height: number
): string {
  const stride = width * 3
  const raw = new Uint8Array((stride + 1) * height)
  let offset = 0
  for (let y = 0; y < height; y += 1) {
    raw[offset] = 0
    offset += 1
    raw.set(rgb.subarray(y * stride, (y + 1) * stride), offset)
    offset += stride
  }

  const zlib: number[] = []
  zlib.push(0x78, 0x01)
  let pos = 0
  while (pos < raw.length) {
    const blockSize = Math.min(65535, raw.length - pos)
    const isFinal = pos + blockSize >= raw.length
    zlib.push(isFinal ? 0x01 : 0x00)
    zlib.push(blockSize & 0xff, (blockSize >>> 8) & 0xff)
    const nlen = ~blockSize & 0xffff
    zlib.push(nlen & 0xff, (nlen >>> 8) & 0xff)
    for (let i = 0; i < blockSize; i += 1) {
      zlib.push(raw[pos + i])
    }
    pos += blockSize
  }
  const adler = adler32(raw)
  pushUint32BE(zlib, adler)
  const zlibBytes = Uint8Array.from(zlib)

  const png: number[] = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]

  const ihdr: number[] = []
  pushUint32BE(ihdr, width)
  pushUint32BE(ihdr, height)
  ihdr.push(8, 2, 0, 0, 0)

  const ihdrChunk = new Uint8Array(4 + ihdr.length)
  ihdrChunk[0] = 0x49
  ihdrChunk[1] = 0x48
  ihdrChunk[2] = 0x44
  ihdrChunk[3] = 0x52
  for (let i = 0; i < ihdr.length; i += 1) {
    ihdrChunk[4 + i] = ihdr[i]
  }
  pushUint32BE(png, ihdr.length)
  pushBytes(png, ihdrChunk)
  pushUint32BE(png, crc32(ihdrChunk))

  const idatChunk = new Uint8Array(4 + zlibBytes.length)
  idatChunk[0] = 0x49
  idatChunk[1] = 0x44
  idatChunk[2] = 0x41
  idatChunk[3] = 0x54
  idatChunk.set(zlibBytes, 4)
  pushUint32BE(png, zlibBytes.length)
  pushBytes(png, idatChunk)
  pushUint32BE(png, crc32(idatChunk))

  const iendChunk = Uint8Array.from([0x49, 0x45, 0x4e, 0x44])
  pushUint32BE(png, 0)
  pushBytes(png, iendChunk)
  pushUint32BE(png, crc32(iendChunk))

  const bytes = Uint8Array.from(png)
  const base64Chars =
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
  let base64 = ''
  for (let i = 0; i < bytes.length; i += 3) {
    const a = bytes[i]
    const b = i + 1 < bytes.length ? bytes[i + 1] : 0
    const c = i + 2 < bytes.length ? bytes[i + 2] : 0
    const triple = (a << 16) | (b << 8) | c
    base64 += base64Chars[(triple >>> 18) & 0x3f]
    base64 += base64Chars[(triple >>> 12) & 0x3f]
    base64 += i + 1 < bytes.length ? base64Chars[(triple >>> 6) & 0x3f] : '='
    base64 += i + 2 < bytes.length ? base64Chars[triple & 0x3f] : '='
  }
  return base64
}

function resizeBilinearRgb(
  src: Uint8Array,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number
): Uint8Array {
  if (dstW <= 0 || dstH <= 0) {
    return new Uint8Array(0)
  }

  const dst = new Uint8Array(dstW * dstH * 3)
  resizeBilinearRgbInto(src, srcW, srcH, dstW, dstH, dst)
  return dst
}

function resizeBilinearRgbInto(
  src: Uint8Array,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number,
  dst: Uint8Array
): void {
  const xScale = srcW / dstW
  const yScale = srcH / dstH

  for (let dy = 0; dy < dstH; dy += 1) {
    const fy = (dy + 0.5) * yScale - 0.5
    const y0 = clamp(Math.floor(fy), 0, srcH - 1)
    const y1 = clamp(y0 + 1, 0, srcH - 1)
    const wy = fy - y0

    for (let dx = 0; dx < dstW; dx += 1) {
      const fx = (dx + 0.5) * xScale - 0.5
      const x0 = clamp(Math.floor(fx), 0, srcW - 1)
      const x1 = clamp(x0 + 1, 0, srcW - 1)
      const wx = fx - x0

      const topLeft = (y0 * srcW + x0) * 3
      const topRight = (y0 * srcW + x1) * 3
      const botLeft = (y1 * srcW + x0) * 3
      const botRight = (y1 * srcW + x1) * 3

      const outIndex = (dy * dstW + dx) * 3
      for (let c = 0; c < 3; c += 1) {
        const v00 = src[topLeft + c]
        const v01 = src[topRight + c]
        const v10 = src[botLeft + c]
        const v11 = src[botRight + c]

        const top = v00 + (v01 - v00) * wx
        const bottom = v10 + (v11 - v10) * wx
        const value = top + (bottom - top) * wy

        dst[outIndex + c] = clamp(Math.round(value), 0, 255)
      }
    }
  }
}

function preprocessDetection(
  image: RgbImage,
  inputSize: number
): DetPreprocessResult {
  const { width: origW, height: origH, data } = image

  const scale = Math.min(inputSize / origW, inputSize / origH)
  const newW = Math.max(1, Math.floor(origW * scale))
  const newH = Math.max(1, Math.floor(origH * scale))

  const padW = Math.floor((inputSize - newW) / 2)
  const padH = Math.floor((inputSize - newH) / 2)

  const tensorSize = inputSize * inputSize * 3
  if (detTensorCache == null || detTensorCacheSize !== tensorSize) {
    detTensorCache = new Float32Array(tensorSize)
    detTensorCacheSize = tensorSize
  }
  const tensor = detTensorCache
  const padValue = 114 / 255
  tensor.fill(padValue)

  const inv255 = 1 / 255

  for (let dy = 0; dy < newH; dy += 1) {
    const srcY = ((dy * origH) / newH) | 0

    const dstY = dy + padH
    const dstRow = dstY * inputSize * 3
    const srcRow = srcY * origW * 3

    for (let dx = 0; dx < newW; dx += 1) {
      const srcX = ((dx * origW) / newW) | 0
      const srcIdx = srcRow + srcX * 3

      const dstIdx = dstRow + (dx + padW) * 3
      tensor[dstIdx] = data[srcIdx] * inv255
      tensor[dstIdx + 1] = data[srcIdx + 1] * inv255
      tensor[dstIdx + 2] = data[srcIdx + 2] * inv255
    }
  }

  return { tensor, scale, padW, padH }
}

function computeIoU(a: number[], b: number[]): number {
  const xx1 = Math.max(a[0], b[0])
  const yy1 = Math.max(a[1], b[1])
  const xx2 = Math.min(a[2], b[2])
  const yy2 = Math.min(a[3], b[3])

  const interW = Math.max(0, xx2 - xx1)
  const interH = Math.max(0, yy2 - yy1)
  const inter = interW * interH

  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1])
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1])

  return inter / (areaA + areaB - inter + 1e-6)
}

function nms(
  boxes: number[][],
  scores: number[],
  iouThreshold: number
): number[] {
  if (boxes.length === 0) {
    return []
  }

  const order = scores
    .map((_, idx) => idx)
    .sort((a, b) => scores[b] - scores[a])

  const keep: number[] = []

  while (order.length > 0) {
    const current = order[0]
    keep.push(current)
    if (order.length === 1) {
      break
    }

    const remaining: number[] = []
    for (let i = 1; i < order.length; i += 1) {
      const next = order[i]
      const iou = computeIoU(boxes[current], boxes[next])
      if (iou <= iouThreshold) {
        remaining.push(next)
      }
    }

    order.splice(0, order.length, ...remaining)
  }

  return keep
}

function postprocessDetection(
  rawOutput: Float32Array,
  outputShape: number[],
  image: RgbImage,
  confThresh: number,
  inputSize: number,
  scale: number,
  padW: number,
  padH: number
): DetectionBox[] {
  // New format: [1, 3549, 5] — raw anchors (x1, y1, x2, y2, conf), normalized to inputSize
  // Old format: [1, 300, 6] — post-NMS boxes from in-model postprocess head
  const isRawAnchors = outputShape[1] !== 300

  const rows = outputShape[1]
  const stride = isRawAnchors ? 5 : 6
  const rawBoxes: number[][] = []
  const rawScores: number[] = []

  for (let i = 0; i < rows; i++) {
    const offset = i * stride
    const score = rawOutput[offset + 4]

    if (score < confThresh) continue

    const x1_416 = rawOutput[offset + 0] * inputSize
    const y1_416 = rawOutput[offset + 1] * inputSize
    const x2_416 = rawOutput[offset + 2] * inputSize
    const y2_416 = rawOutput[offset + 3] * inputSize

    const rawX1 = clamp(Math.round((x1_416 - padW) / scale), 0, image.width)
    const rawY1 = clamp(Math.round((y1_416 - padH) / scale), 0, image.height)
    const rawX2 = clamp(Math.round((x2_416 - padW) / scale), 0, image.width)
    const rawY2 = clamp(Math.round((y2_416 - padH) / scale), 0, image.height)

    if (rawX2 <= rawX1 || rawY2 <= rawY1) continue

    rawBoxes.push([rawX1, rawY1, rawX2, rawY2])
    rawScores.push(score)
  }

  const keepIndices = isRawAnchors ? nms(rawBoxes, rawScores, 0.5) : rawBoxes.map((_, idx) => idx)
  const results: DetectionBox[] = []

  for (const idx of keepIndices) {
    const [rawX1, rawY1, rawX2, rawY2] = rawBoxes[idx]
    const boxW = rawX2 - rawX1
    const boxH = rawY2 - rawY1
    const padLeft = Math.max(2, Math.round(boxW * DETECTION_PAD_LEFT_RATIO))
    const padRight = Math.max(2, Math.round(boxW * DETECTION_PAD_RIGHT_RATIO))
    const padTop = Math.max(2, Math.round(boxH * DETECTION_PAD_TOP_RATIO))
    const padBottom = Math.max(2, Math.round(boxH * DETECTION_PAD_BOTTOM_RATIO))

    const x1 = clamp(rawX1 - padLeft, 0, image.width)
    const y1 = clamp(rawY1 - padTop, 0, image.height)
    const x2 = clamp(rawX2 + padRight, 0, image.width)
    const y2 = clamp(rawY2 + padBottom, 0, image.height)

    if (x2 <= x1 || y2 <= y1) continue

    results.push({ x1, y1, x2, y2, conf: rawScores[idx] })
  }

  return results
}

function cropRgb(image: RgbImage, box: DetectionBox): RgbImage | null {
  const x1 = clamp(Math.floor(box.x1), 0, image.width)
  const y1 = clamp(Math.floor(box.y1), 0, image.height)
  const x2 = clamp(Math.floor(box.x2), 0, image.width)
  const y2 = clamp(Math.floor(box.y2), 0, image.height)

  const width = x2 - x1
  const height = y2 - y1

  if (width <= 0 || height <= 0) {
    return null
  }

  const data = new Uint8Array(width * height * 3)
  for (let y = 0; y < height; y += 1) {
    const srcOffset = ((y + y1) * image.width + x1) * 3
    const dstOffset = y * width * 3
    data.set(image.data.subarray(srcOffset, srcOffset + width * 3), dstOffset)
  }

  return { data, width, height }
}

function preprocessRecognition(
  crop: RgbImage,
  targetH: number,
  targetW: number
): Float32Array {
  const ratio = targetH / crop.height
  const newW = Math.max(1, Math.min(Math.floor(crop.width * ratio), targetW))

  const resizedSize = targetH * newW * 3
  if (recResizeCache == null || recResizeCacheSize !== resizedSize) {
    recResizeCache = new Uint8Array(resizedSize)
    recResizeCacheSize = resizedSize
  }
  const resized = recResizeCache
  resizeBilinearRgbInto(crop.data, crop.width, crop.height, newW, targetH, resized)

  const tensorSize = targetH * targetW * 3
  if (recTensorCache == null || recTensorCacheSize !== tensorSize) {
    recTensorCache = new Float32Array(tensorSize)
    recTensorCacheSize = tensorSize
  }
  const tensor = recTensorCache
  tensor.fill(-1)

  for (let y = 0; y < targetH; y += 1) {
    for (let x = 0; x < newW; x += 1) {
      const srcIdx = (y * newW + x) * 3
      const dstIdx = (y * targetW + x) * 3

      tensor[dstIdx] = resized[srcIdx] / 127.5 - 1.0
      tensor[dstIdx + 1] = resized[srcIdx + 1] / 127.5 - 1.0
      tensor[dstIdx + 2] = resized[srcIdx + 2] / 127.5 - 1.0
    }
  }

  return tensor
}

function ctcGreedyDecode(
  rawOutput: Float32Array,
  outputShape: number[],
  charDict: readonly string[]
): DecodedText {
  let seqLen = 0
  let numClasses = 0

  if (outputShape.length === 3 && outputShape[0] === 1) {
    seqLen = outputShape[1]
    numClasses = outputShape[2]
  } else if (outputShape.length === 2) {
    seqLen = outputShape[0]
    numClasses = outputShape[1]
  } else {
    throw new Error(
      `Unsupported recognition output shape: [${outputShape.join(', ')}]`
    )
  }

  const rowSums = new Float32Array(seqLen)
  let alreadySoftmax = true

  for (let row = 0; row < seqLen; row += 1) {
    let sum = 0
    for (let col = 0; col < numClasses; col += 1) {
      sum += rawOutput[row * numClasses + col]
    }
    rowSums[row] = sum
    if (Math.abs(sum - 1.0) > 0.05) {
      alreadySoftmax = false
    }
  }

  const indices = new Int32Array(seqLen)
  const maxProbs = new Float32Array(seqLen)

  for (let row = 0; row < seqLen; row += 1) {
    let maxIdx = 0
    let maxVal = Number.NEGATIVE_INFINITY

    for (let col = 0; col < numClasses; col += 1) {
      const value = rawOutput[row * numClasses + col]
      if (value > maxVal) {
        maxVal = value
        maxIdx = col
      }
    }

    indices[row] = maxIdx

    if (alreadySoftmax) {
      maxProbs[row] = maxVal
    } else {
      let expSum = 0
      for (let col = 0; col < numClasses; col += 1) {
        expSum += Math.exp(rawOutput[row * numClasses + col] - maxVal)
      }
      maxProbs[row] = expSum > 0 ? 1 / expSum : 0
    }
  }

  const collapsed: number[] = []
  const collapsedProbs: number[] = []
  let prev = -1

  for (let i = 0; i < seqLen; i += 1) {
    const idx = indices[i]
    if (idx !== prev) {
      collapsed.push(idx)
      collapsedProbs.push(maxProbs[i])
    }
    prev = idx
  }

  const chars: string[] = []
  const charProbs: number[] = []

  for (let i = 0; i < collapsed.length; i += 1) {
    const idx = collapsed[i]
    const prob = collapsedProbs[i]

    if (idx === 0) {
      continue
    }

    const charIdx = idx - 1
    if (charIdx >= 0 && charIdx < charDict.length) {
      chars.push(charDict[charIdx])
      charProbs.push(prob)
    }
  }

  const text = chars.join('')
  const confidence =
    charProbs.length > 0
      ? charProbs.reduce((sum, value) => sum + value, 0) / charProbs.length
      : 0

  return { text, confidence }
}

export function parseWeight(text: string): {
  value: string | null
  unit: string | null
} {
  const match = /(\d+\.?\d*)\s*(lb|kg|oz|jin|g|l|b|k|j|i|n)?/i.exec(text)
  if (match == null) {
    return { value: null, unit: null }
  }

  let rawValue = match[1]
  const rawUnit = (match[2] ?? '').toLowerCase()

  const digitsOnly = rawValue.replace('.', '')
  if (digitsOnly.length > 4) {
    const kept = digitsOnly.slice(0, 4)
    if (rawValue.includes('.')) {
      const dotPos = rawValue.indexOf('.')
      rawValue =
        dotPos >= 4 ? kept : `${kept.slice(0, dotPos)}.${kept.slice(dotPos)}`
    } else {
      rawValue = kept
    }
  }

  const unitMap: Record<string, string> = {
    lb: 'lb',
    kg: 'kg',
    oz: 'oz',
    jin: 'jin',
    g: 'g',
    l: 'lb',
    b: 'lb',
    k: 'kg',
    j: 'jin',
    i: 'jin',
    n: 'jin',
  }

  return {
    value: rawValue,
    unit: unitMap[rawUnit] ?? rawUnit,
  }
}

export async function runInference(
  image: RgbImage,
  detModel: TensorflowModel,
  recModel: TensorflowModel,
  charDict: readonly string[],
  confThresh = 0.25,
  iouThresh = 0.7,
  maxCrops = Number.POSITIVE_INFINITY,
  includeDebugImages = false,
  detTensor?: Float32Array
): Promise<InferenceResult> {
  const debugEnabled = includeDebugImages === true
  const startedAt = nowMs()

  if (detModel.inputs.length === 0 || detModel.outputs.length === 0) {
    throw new Error('Detection model input/output tensors are missing.')
  }
  if (recModel.inputs.length === 0 || recModel.outputs.length === 0) {
    throw new Error('Recognition model input/output tensors are missing.')
  }

  const detInputShape = detModel.inputs[0].shape
  const detOutputShape = detModel.outputs[0].shape
  const recInputShape = recModel.inputs[0].shape
  const recOutputShape = recModel.outputs[0].shape

  if (detInputShape.length < 3) {
    throw new Error(
      `Unexpected detection input shape: [${detInputShape.join(', ')}]`
    )
  }
  if (recInputShape.length < 3) {
    throw new Error(
      `Unexpected recognition input shape: [${recInputShape.join(', ')}]`
    )
  }

  const detInputSize = detInputShape[1]
  const recTargetH = recInputShape[1]
  const recTargetW = recInputShape[2]

  const detPreprocessStartedAt = nowMs()
  const detPrep = detTensor != null
    ? { tensor: detTensor, scale: Math.min(detInputSize / image.width, detInputSize / image.height), padW: Math.floor((detInputSize - Math.max(1, Math.floor(image.width * Math.min(detInputSize / image.width, detInputSize / image.height)))) / 2), padH: Math.floor((detInputSize - Math.max(1, Math.floor(image.height * Math.min(detInputSize / image.width, detInputSize / image.height)))) / 2) }
    : preprocessDetection(image, detInputSize)
  const detectionPreprocessMs = nowMs() - detPreprocessStartedAt
  const detectionInputBase64 = debugEnabled
    ? encodePngBase64Rgb(
        tensorToRgbBytes(detPrep.tensor, detInputSize, detInputSize, 'zeroOne'),
        detInputSize,
        detInputSize
      )
    : null

  const detInferenceStartedAt = nowMs()
  const detOutputs = await detModel.run([detPrep.tensor])
  const detectionInferenceMs = nowMs() - detInferenceStartedAt

  if (detOutputs.length === 0) {
    throw new Error('Detection model returned no outputs.')
  }

  const detRaw = toFloat32Array(detOutputs[0])

  const detPostprocessStartedAt = nowMs()
  const boxes = postprocessDetection(
    detRaw,
    detOutputShape,
    image,
    confThresh,
    detInputSize,
    detPrep.scale,
    detPrep.padW,
    detPrep.padH
  )
  const detectionPostprocessMs = nowMs() - detPostprocessStartedAt
  const detectionTotalMs =
    detectionPreprocessMs + detectionInferenceMs + detectionPostprocessMs
  const firstDetectionCrop =
    debugEnabled && boxes.length > 0 ? cropRgb(image, boxes[0]) : null
  const detectionCrop320Base64 =
    debugEnabled && firstDetectionCrop != null
      ? encodePngBase64Rgb(
          resizeBilinearRgb(
            firstDetectionCrop.data,
            firstDetectionCrop.width,
            firstDetectionCrop.height,
            320,
            320
          ),
          320,
          320
        )
      : null

  const texts: RecognitionEntry[] = []
  let boxesForRecognition: DetectionBox[] = boxes
  if (Number.isFinite(maxCrops) && boxes.length > maxCrops) {
    if (maxCrops === 1) {
      let best = boxes[0]
      for (let i = 1; i < boxes.length; i += 1) {
        if (boxes[i].conf > best.conf) {
          best = boxes[i]
        }
      }
      boxesForRecognition = [best]
    } else {
      boxesForRecognition = boxes
        .slice()
        .sort((a, b) => b.conf - a.conf)
        .slice(0, maxCrops)
    }
  }
  let recognitionCropMs = 0
  let recognitionPreprocessMs = 0
  let recognitionInferenceMs = 0
  let recognitionDecodeMs = 0
  let cropsProcessed = 0
  const recognitionInputsBase64: string[] = debugEnabled ? [] : EMPTY_STRINGS
  let bestRecConf = 0
  const recognitionLoopStartedAt = nowMs()

  for (let i = 0; i < boxesForRecognition.length; i += 1) {
    const cropStartedAt = nowMs()
    const crop = cropRgb(image, boxesForRecognition[i])
    recognitionCropMs += nowMs() - cropStartedAt
    if (crop == null) {
      continue
    }
    cropsProcessed += 1
    if (debugEnabled && recognitionInputsBase64 !== EMPTY_STRINGS) {
      recognitionInputsBase64.push(
        encodePngBase64Rgb(crop.data, crop.width, crop.height)
      )
    }

    const recPreprocessStartedAt = nowMs()
    const recTensor = preprocessRecognition(crop, recTargetH, recTargetW)
    recognitionPreprocessMs += nowMs() - recPreprocessStartedAt

    const recInferenceStartedAt = nowMs()
    const recOutputs = await recModel.run([recTensor])
    recognitionInferenceMs += nowMs() - recInferenceStartedAt
    if (recOutputs.length === 0) {
      continue
    }

    const recDecodeStartedAt = nowMs()
    const recRaw = toFloat32Array(recOutputs[0])
    const decoded = ctcGreedyDecode(recRaw, recOutputShape, charDict)
    recognitionDecodeMs += nowMs() - recDecodeStartedAt

    texts.push({
      text: decoded.text,
      detConf: boxesForRecognition[i].conf,
      recConf: decoded.confidence,
    })

    if (decoded.confidence >= bestRecConf) {
      bestRecConf = decoded.confidence
    }
  }
  const recognitionLoopMs = nowMs() - recognitionLoopStartedAt
  const recognitionAccountedMs =
    recognitionCropMs +
    recognitionPreprocessMs +
    recognitionInferenceMs +
    recognitionDecodeMs
  const recognitionOtherMs = Math.max(
    0,
    recognitionLoopMs - recognitionAccountedMs
  )
  const recognitionTotalMs = recognitionLoopMs

  const combined =
    texts.length > 0
      ? texts.map((item) => item.text).join(' | ')
      : '(no detection)'
  if (texts.length === 0) {
    bestRecConf = 0
  }

  const parseStartedAt = nowMs()
  const parsed =
    texts.length > 0 ? parseWeight(combined) : { value: null, unit: null }
  const parseMs = nowMs() - parseStartedAt
  const totalMs = nowMs() - startedAt

  return {
    boxes,
    texts,
    combined,
    value: parsed.value,
    unit: parsed.unit,
    recConf: bestRecConf,
    detectionInputBase64,
    detectionCrop320Base64,
    recognitionInputsBase64,
    timeMs: totalMs,
    latency: {
      detectionPreprocessMs,
      detectionInferenceMs,
      detectionPostprocessMs,
      detectionTotalMs,
      recognitionCropMs,
      recognitionPreprocessMs,
      recognitionInferenceMs,
      recognitionDecodeMs,
      recognitionLoopMs,
      recognitionOtherMs,
      recognitionTotalMs,
      parseMs,
      totalMs,
      boxesDetected: boxes.length,
      cropsProcessed,
    },
  }
}
