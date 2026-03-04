//
//  File.swift
//  TfliteExample
//

import Foundation
import UIKit
import React

@objc(ImageDecoder)
class ImageDecoder: NSObject, RCTBridgeModule {
  @objc var bridge: RCTBridge!

  @objc static func requiresMainQueueSetup() -> Bool {
    return false
  }

  @objc(decodeImageAsDetectionTensor:inputSize:padValue:resolver:rejecter:)
  func decodeImageAsDetectionTensor(
    _ path: String,
    inputSize: NSNumber,
    padValue: NSNumber,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    DispatchQueue.global(qos: .userInitiated).async {
      do {
        let normalizedPath = path.hasPrefix("file://") ? String(path.dropFirst(7)) : path

        guard let image = UIImage(contentsOfFile: normalizedPath),
              let cgImage = image.cgImage else {
          throw NSError(domain: "ImageDecoder", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to decode image file: \(normalizedPath)"])
        }

        let origW = cgImage.width
        let origH = cgImage.height
        let input = max(1, inputSize.intValue)
        let scale = min(Double(input) / Double(origW), Double(input) / Double(origH))
        let newW = max(1, Int(floor(Double(origW) * scale)))
        let newH = max(1, Int(floor(Double(origH) * scale)))
        let padW = Int(floor(Double(input - newW) / 2.0))
        let padH = Int(floor(Double(input - newH) / 2.0))

        guard let provider = cgImage.dataProvider,
              let dataRef = provider.data,
              let src = CFDataGetBytePtr(dataRef) else {
          throw NSError(domain: "ImageDecoder", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to read image pixels"])
        }

        let bytesPerRow = cgImage.bytesPerRow
        var tensor = [Float](repeating: Float(truncating: padValue), count: input * input * 3)

        for dy in 0..<newH {
          let srcY = (dy * origH) / newH
          let srcRowBase = srcY * bytesPerRow
          let dstRow = (dy + padH) * input
          for dx in 0..<newW {
            let srcX = (dx * origW) / newW
            let srcIdx = srcRowBase + srcX * 4
            let dstIdx = ((dstRow + (dx + padW)) * 3)
            tensor[dstIdx] = Float(src[srcIdx]) / 255.0
            tensor[dstIdx + 1] = Float(src[srcIdx + 1]) / 255.0
            tensor[dstIdx + 2] = Float(src[srcIdx + 2]) / 255.0
          }
        }

        let tensorData = tensor.withUnsafeBufferPointer { buffer in
          Data(buffer: buffer)
        }

        guard let blobManager = self.bridge.module(forName: "BlobModule") as? RCTBlobManager else {
          throw NSError(domain: "ImageDecoder", code: 3, userInfo: [NSLocalizedDescriptionKey: "BlobModule is unavailable."])
        }
        let blobId = UUID().uuidString
        blobManager.store(tensorData, withId: blobId)

        resolve([
          "blobId": blobId,
          "offset": 0,
          "size": tensorData.count,
          "type": "application/octet-stream",
        ])
      } catch let error {
        reject("decode_error", error.localizedDescription, error)
      }
    }
  }
}
