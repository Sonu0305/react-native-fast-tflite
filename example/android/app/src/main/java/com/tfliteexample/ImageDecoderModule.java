package com.tfliteexample;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.util.Base64;

import androidx.annotation.NonNull;
import androidx.exifinterface.media.ExifInterface;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.WritableMap;

import java.io.IOException;

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

      String rgbBase64 = Base64.encodeToString(rgb, Base64.NO_WRAP);

      WritableMap result = Arguments.createMap();
      result.putInt("width", width);
      result.putInt("height", height);
      result.putString("rgbBase64", rgbBase64);
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
