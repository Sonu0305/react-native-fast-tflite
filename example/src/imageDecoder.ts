import { NativeModules } from 'react-native'

interface NativeBlobPayload {
  blobId: string
  offset: number
  size: number
  type?: string
}

interface NativeDecodedImagePayload {
  width: number
  height: number
  rgbBlob: NativeBlobPayload
}

export interface DecodedImagePayload {
  width: number
  height: number
  rgbBytes: ArrayBuffer
}

interface NativeImageDecoderSpec {
  decodeImage(
    path: string,
    maxDimension: number
  ): Promise<NativeDecodedImagePayload>,
  decodeImageAsDetectionTensor(
    path: string,
    inputSize: number,
    padValue: number
  ): Promise<NativeBlobPayload>
}

const NativeImageDecoder = NativeModules.ImageDecoder as
  | NativeImageDecoderSpec
  | undefined

// eslint-disable-next-line @typescript-eslint/no-var-requires
const BlobManager = require(
  'react-native/Libraries/Blob/BlobManager'
) as {
  createFromOptions: (options: NativeBlobPayload) => any
}

/**
 * Fast native blob -> ArrayBuffer transfer.
 * Uses RN FileReader polyfill (required for current RN versions).
 */
function readBlobAsArrayBuffer(options: NativeBlobPayload): Promise<ArrayBuffer> {
  return new Promise<ArrayBuffer>((resolve, reject) => {
    const blob = BlobManager.createFromOptions(options)
    const reader = new FileReader()
    reader.onload = () => {
      resolve(reader.result as ArrayBuffer)
      blob.close()
    }
    reader.onerror = () => {
      blob.close()
      reject(new Error('Failed to read blob.'))
    }
    reader.readAsArrayBuffer(blob)
  })
}

/**
 * Decodes image natively and returns RGB ArrayBuffer.
 * Assumes modern native implementation returning rgbBlob.
 */
export async function decodeImageToArrayBuffer(
  path: string,
  maxDimension: number
): Promise<DecodedImagePayload> {
  if (!NativeImageDecoder) {
    throw new Error('Native ImageDecoder module unavailable.')
  }

  const decoded = await NativeImageDecoder.decodeImage(
    path,
    maxDimension
  )

  if (!decoded?.rgbBlob?.blobId) {
    throw new Error(
      'Native decoder did not return rgbBlob. Rebuild native app.'
    )
  }

  const buffer = await readBlobAsArrayBuffer(decoded.rgbBlob)

  return {
    width: decoded.width,
    height: decoded.height,
    rgbBytes: buffer,
  }
}

export async function decodeImageAsDetectionTensor(
  path: string,
  inputSize: number,
  padValue: number = 114 / 255
): Promise<Float32Array> {
  if (NativeImageDecoder == null) {
    throw new Error('Native ImageDecoder module is unavailable.')
  }
  const blob = await NativeImageDecoder.decodeImageAsDetectionTensor(
    path,
    inputSize,
    padValue
  )
  const buffer = await readBlobAsArrayBuffer(blob)
  return new Float32Array(buffer)
}
