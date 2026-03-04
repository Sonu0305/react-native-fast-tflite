package com.tfliteexample;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import androidx.annotation.NonNull;
import androidx.exifinterface.media.ExifInterface;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.modules.blob.BlobModule;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class ImageDecoderModule extends ReactContextBaseJavaModule {
  private static final String MODULE_NAME = "ImageDecoder";
  private static final int DEFAULT_MAX_DIMENSION = 1280;

  public ImageDecoderModule(ReactApplicationContext reactContext) {
    super(reactContext);
  }

  @NonNull
  @Override
  public String getName() {
    return MODULE_NAME;
  }

  @ReactMethod
  public void decodeImage(String filePath, double maxDimensionInput, Promise promise) {
    Bitmap bitmap = null;
    Bitmap orientedBitmap = null;

    try {
      String normalizedPath = filePath;
      if (normalizedPath.startsWith("file://")) {
        normalizedPath = normalizedPath.substring(7);
      }

      int maxDimension = (int) Math.max(1, maxDimensionInput);
      if (maxDimension <= 1) {
        maxDimension = DEFAULT_MAX_DIMENSION;
      }

      BitmapFactory.Options bounds = new BitmapFactory.Options();
      bounds.inJustDecodeBounds = true;
      BitmapFactory.decodeFile(normalizedPath, bounds);

      if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
        throw new IOException("Invalid image dimensions for file: " + normalizedPath);
      }

      BitmapFactory.Options options = new BitmapFactory.Options();
      options.inSampleSize = calculateInSampleSize(bounds.outWidth, bounds.outHeight, maxDimension);
      options.inPreferredConfig = Bitmap.Config.ARGB_8888;

      bitmap = BitmapFactory.decodeFile(normalizedPath, options);
      if (bitmap == null) {
        throw new IOException("Failed to decode image file: " + normalizedPath);
      }

      orientedBitmap = applyExifOrientation(bitmap, normalizedPath);
      int width = orientedBitmap.getWidth();
      int height = orientedBitmap.getHeight();

      int pixelCount = width * height;
      int[] pixels = new int[pixelCount];
      orientedBitmap.getPixels(pixels, 0, width, 0, 0, width, height);

      byte[] rgb = new byte[pixelCount * 3];
      for (int i = 0, j = 0; i < pixelCount; i++) {
        int pixel = pixels[i];
        rgb[j++] = (byte) ((pixel >> 16) & 0xFF); // R
        rgb[j++] = (byte) ((pixel >> 8) & 0xFF);  // G
        rgb[j++] = (byte) (pixel & 0xFF);         // B
      }

      BlobModule blobModule = getReactApplicationContext().getNativeModule(BlobModule.class);
      if (blobModule == null) {
        throw new IOException("BlobModule is unavailable.");
      }
      String blobId = blobModule.store(rgb);

      WritableMap rgbBlob = Arguments.createMap();
      rgbBlob.putString("blobId", blobId);
      rgbBlob.putInt("offset", 0);
      rgbBlob.putInt("size", rgb.length);
      rgbBlob.putString("type", "application/octet-stream");

      WritableMap result = Arguments.createMap();
      result.putInt("width", width);
      result.putInt("height", height);
      result.putMap("rgbBlob", rgbBlob);
      promise.resolve(result);
    } catch (Exception e) {
      promise.reject("decode_error", e.getMessage(), e);
    } finally {
      if (orientedBitmap != null && orientedBitmap != bitmap && !orientedBitmap.isRecycled()) {
        orientedBitmap.recycle();
      }
      if (bitmap != null && !bitmap.isRecycled()) {
        bitmap.recycle();
      }
    }
  }

  @ReactMethod
  public void decodeImageAsDetectionTensor(String filePath, int inputSize, double padValue, Promise promise) {
    new Thread(() -> {
      Bitmap bitmap = null;
      Bitmap orientedBitmap = null;

      try {
        String normalizedPath = filePath;
        if (normalizedPath.startsWith("file://")) {
          normalizedPath = normalizedPath.substring(7);
        }

        bitmap = BitmapFactory.decodeFile(normalizedPath);
        if (bitmap == null) {
          throw new IOException("Failed to decode image file: " + normalizedPath);
        }

        orientedBitmap = applyExifOrientation(bitmap, normalizedPath);
        int origW = orientedBitmap.getWidth();
        int origH = orientedBitmap.getHeight();

        double scale = Math.min((double) inputSize / (double) origW, (double) inputSize / (double) origH);
        int newW = Math.max(1, (int) Math.floor(origW * scale));
        int newH = Math.max(1, (int) Math.floor(origH * scale));
        int padW = (int) Math.floor((inputSize - newW) / 2.0);
        int padH = (int) Math.floor((inputSize - newH) / 2.0);

        int[] srcPixels = new int[origW * origH];
        orientedBitmap.getPixels(srcPixels, 0, origW, 0, 0, origW, origH);

        int tensorSize = inputSize * inputSize * 3;
        ByteBuffer tensorBuffer = ByteBuffer.allocate(tensorSize * 4).order(ByteOrder.nativeOrder());
        float pad = (float) padValue;
        for (int i = 0; i < tensorSize; i++) {
          tensorBuffer.putFloat(pad);
        }

        for (int dy = 0; dy < newH; dy++) {
          int srcY = (dy * origH) / newH;
          int srcRow = srcY * origW;
          int dstRow = (dy + padH) * inputSize;
          for (int dx = 0; dx < newW; dx++) {
            int srcX = (dx * origW) / newW;
            int pixel = srcPixels[srcRow + srcX];
            int dstIdx = ((dstRow + (dx + padW)) * 3) * 4;
            tensorBuffer.putFloat(dstIdx, ((pixel >> 16) & 0xFF) / 255.0f);
            tensorBuffer.putFloat(dstIdx + 4, ((pixel >> 8) & 0xFF) / 255.0f);
            tensorBuffer.putFloat(dstIdx + 8, (pixel & 0xFF) / 255.0f);
          }
        }

        BlobModule blobModule = getReactApplicationContext().getNativeModule(BlobModule.class);
        if (blobModule == null) {
          throw new IOException("BlobModule is unavailable.");
        }
        String blobId = blobModule.store(tensorBuffer.array());

        WritableMap tensorBlob = Arguments.createMap();
        tensorBlob.putString("blobId", blobId);
        tensorBlob.putInt("offset", 0);
        tensorBlob.putInt("size", tensorBuffer.capacity());
        tensorBlob.putString("type", "application/octet-stream");
        promise.resolve(tensorBlob);
      } catch (Exception e) {
        promise.reject("decode_error", e.getMessage(), e);
      } finally {
        if (orientedBitmap != null && orientedBitmap != bitmap && !orientedBitmap.isRecycled()) {
          orientedBitmap.recycle();
        }
        if (bitmap != null && !bitmap.isRecycled()) {
          bitmap.recycle();
        }
      }
    }).start();
  }

  private static int calculateInSampleSize(int width, int height, int maxDimension) {
    int inSampleSize = 1;
    int maxSide = Math.max(width, height);

    while ((maxSide / inSampleSize) > maxDimension) {
      inSampleSize *= 2;
    }

    return Math.max(1, inSampleSize);
  }

  private static Bitmap applyExifOrientation(Bitmap bitmap, String path) {
    try {
      ExifInterface exif = new ExifInterface(path);
      int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);

      Matrix matrix = new Matrix();
      switch (orientation) {
        case ExifInterface.ORIENTATION_ROTATE_90:
          matrix.postRotate(90);
          break;
        case ExifInterface.ORIENTATION_ROTATE_180:
          matrix.postRotate(180);
          break;
        case ExifInterface.ORIENTATION_ROTATE_270:
          matrix.postRotate(270);
          break;
        case ExifInterface.ORIENTATION_FLIP_HORIZONTAL:
          matrix.preScale(-1, 1);
          break;
        case ExifInterface.ORIENTATION_FLIP_VERTICAL:
          matrix.preScale(1, -1);
          break;
        case ExifInterface.ORIENTATION_TRANSPOSE:
          matrix.preScale(-1, 1);
          matrix.postRotate(270);
          break;
        case ExifInterface.ORIENTATION_TRANSVERSE:
          matrix.preScale(-1, 1);
          matrix.postRotate(90);
          break;
        default:
          return bitmap;
      }

      Bitmap transformed = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
      return transformed == null ? bitmap : transformed;
    } catch (Exception ignored) {
      return bitmap;
    }
  }
}
