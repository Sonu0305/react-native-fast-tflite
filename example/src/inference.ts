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

export interface RecognitionEntry {
  text: string
  detConf: number
  recConf: number
}

export interface InferenceResult {
  boxes: DetectionBox[]
  texts: RecognitionEntry[]
  combined: string
  value: string | null
  unit: string | null
  recConf: number
  timeMs: number
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

function nowMs(): number {
  if (globalThis.performance?.now != null) {
    return globalThis.performance.now()
  }
  return Date.now()
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function toFloat32Array(tensor: ArrayLike<number>): Float32Array {
  if (tensor instanceof Float32Array) {
    return tensor
  }
  return Float32Array.from(tensor)
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

  return dst
}

function preprocessDetection(image: RgbImage, inputSize: number): DetPreprocessResult {
  const { width: origW, height: origH, data } = image

  const scale = Math.min(inputSize / origW, inputSize / origH)
  const newW = Math.max(1, Math.floor(origW * scale))
  const newH = Math.max(1, Math.floor(origH * scale))

  const resized = resizeBilinearRgb(data, origW, origH, newW, newH)

  const padW = Math.floor((inputSize - newW) / 2)
  const padH = Math.floor((inputSize - newH) / 2)

  const tensor = new Float32Array(inputSize * inputSize * 3)
  const padValue = 114 / 255
  tensor.fill(padValue)

  for (let y = 0; y < newH; y += 1) {
    for (let x = 0; x < newW; x += 1) {
      const srcIdx = (y * newW + x) * 3
      const dstX = x + padW
      const dstY = y + padH
      const dstIdx = (dstY * inputSize + dstX) * 3

      tensor[dstIdx] = resized[srcIdx] / 255
      tensor[dstIdx + 1] = resized[srcIdx + 1] / 255
      tensor[dstIdx + 2] = resized[srcIdx + 2] / 255
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

function nms(boxes: number[][], scores: number[], iouThreshold: number): number[] {
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
  scale: number,
  padW: number,
  padH: number,
  inputSize: number,
  confThresh: number,
  iouThresh: number
): DetectionBox[] {
  let rows = 0
  let cols = 0

  if (outputShape.length === 3 && outputShape[0] === 1) {
    rows = outputShape[1]
    cols = outputShape[2]
  } else if (outputShape.length === 2) {
    rows = outputShape[0]
    cols = outputShape[1]
  } else {
    throw new Error(`Unsupported detection output shape: [${outputShape.join(', ')}]`)
  }

  const transposed = rows <= 6
  const predRows = transposed ? cols : rows
  const predCols = transposed ? rows : cols

  const read = (r: number, c: number): number => {
    return transposed ? rawOutput[c * cols + r] : rawOutput[r * cols + c]
  }

  const filteredBoxes: number[][] = []
  const filteredScores: number[] = []

  let maxCoord = 0

  for (let r = 0; r < predRows; r += 1) {
    let score = 0

    if (predCols > 5) {
      let maxCls = Number.NEGATIVE_INFINITY
      for (let c = 4; c < predCols; c += 1) {
        const cls = read(r, c)
        if (cls > maxCls) {
          maxCls = cls
        }
      }
      score = maxCls
    } else {
      score = read(r, 4)
    }

    if (score <= confThresh) {
      continue
    }

    const cx = read(r, 0)
    const cy = read(r, 1)
    const w = read(r, 2)
    const h = read(r, 3)

    maxCoord = Math.max(maxCoord, cx, cy, w, h)

    filteredBoxes.push([cx, cy, w, h])
    filteredScores.push(score)
  }

  if (filteredBoxes.length === 0) {
    return []
  }

  const coordScale = maxCoord <= 2.0 ? inputSize : 1.0

  const xyxy: number[][] = filteredBoxes.map((box) => {
    const cx = box[0] * coordScale
    const cy = box[1] * coordScale
    const w = box[2] * coordScale
    const h = box[3] * coordScale

    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
  })

  const keep = nms(xyxy, filteredScores, iouThresh)

  const results: DetectionBox[] = []
  for (let i = 0; i < keep.length; i += 1) {
    const index = keep[i]
    const box = xyxy[index]

    const x1 = clamp(Math.round((box[0] - padW) / scale), 0, image.width)
    const y1 = clamp(Math.round((box[1] - padH) / scale), 0, image.height)
    const x2 = clamp(Math.round((box[2] - padW) / scale), 0, image.width)
    const y2 = clamp(Math.round((box[3] - padH) / scale), 0, image.height)

    if (x2 <= x1 || y2 <= y1) {
      continue
    }

    results.push({
      x1,
      y1,
      x2,
      y2,
      conf: filteredScores[index],
    })
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

function preprocessRecognition(crop: RgbImage, targetH: number, targetW: number): Float32Array {
  const ratio = targetH / crop.height
  const newW = Math.max(1, Math.min(Math.floor(crop.width * ratio), targetW))

  const resized = resizeBilinearRgb(crop.data, crop.width, crop.height, newW, targetH)

  const tensor = new Float32Array(targetH * targetW * 3)
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
    throw new Error(`Unsupported recognition output shape: [${outputShape.join(', ')}]`)
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

export function parseWeight(text: string): { value: string | null; unit: string | null } {
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
      rawValue = dotPos >= 4 ? kept : `${kept.slice(0, dotPos)}.${kept.slice(dotPos)}`
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
  iouThresh = 0.5
): Promise<InferenceResult> {
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
    throw new Error(`Unexpected detection input shape: [${detInputShape.join(', ')}]`)
  }
  if (recInputShape.length < 3) {
    throw new Error(`Unexpected recognition input shape: [${recInputShape.join(', ')}]`)
  }

  const detInputSize = detInputShape[1]
  const recTargetH = recInputShape[1]
  const recTargetW = recInputShape[2]

  const detPrep = preprocessDetection(image, detInputSize)
  const detOutputs = await detModel.run([detPrep.tensor])

  if (detOutputs.length === 0) {
    throw new Error('Detection model returned no outputs.')
  }

  const detRaw = toFloat32Array(detOutputs[0])
  const boxes = postprocessDetection(
    detRaw,
    detOutputShape,
    image,
    detPrep.scale,
    detPrep.padW,
    detPrep.padH,
    detInputSize,
    confThresh,
    iouThresh
  )

  const texts: RecognitionEntry[] = []

  for (let i = 0; i < boxes.length; i += 1) {
    const crop = cropRgb(image, boxes[i])
    if (crop == null) {
      continue
    }

    const recTensor = preprocessRecognition(crop, recTargetH, recTargetW)
    const recOutputs = await recModel.run([recTensor])
    if (recOutputs.length === 0) {
      continue
    }

    const recRaw = toFloat32Array(recOutputs[0])
    const decoded = ctcGreedyDecode(recRaw, recOutputShape, charDict)

    texts.push({
      text: decoded.text,
      detConf: boxes[i].conf,
      recConf: decoded.confidence,
    })
  }

  const combined = texts.length > 0 ? texts.map((item) => item.text).join(' | ') : '(no detection)'
  const bestRecConf = texts.reduce((best, item) => Math.max(best, item.recConf), 0)

  const parsed = texts.length > 0 ? parseWeight(combined) : { value: null, unit: null }

  return {
    boxes,
    texts,
    combined,
    value: parsed.value,
    unit: parsed.unit,
    recConf: bestRecConf,
    timeMs: nowMs() - startedAt,
  }
}
