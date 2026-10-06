import 'dart:typed_data';

/// float32 simd and int32 arithmetic kernels
final class SimdOps {
  // float32 fast path; offset=0; full buffer

  static void addFloat(Float32List a, Float32List b, Float32List out) {
    final int len = a.length;
    final int simdChunks = len >> 2;
    final aSimd = a.buffer.asFloat32x4List(a.offsetInBytes, simdChunks);
    final bSimd = b.buffer.asFloat32x4List(b.offsetInBytes, simdChunks);
    final outSimd = out.buffer.asFloat32x4List(out.offsetInBytes, simdChunks);

    for (int i = 0; i < simdChunks; i++) {
      outSimd[i] = aSimd[i] + bSimd[i];
    }
    for (int i = simdChunks << 2; i < len; i++) {
      out[i] = a[i] + b[i];
    }
  }

  static void subFloat(Float32List a, Float32List b, Float32List out) {
    final int len = a.length;
    final int simdChunks = len >> 2;
    final aSimd = a.buffer.asFloat32x4List(a.offsetInBytes, simdChunks);
    final bSimd = b.buffer.asFloat32x4List(b.offsetInBytes, simdChunks);
    final outSimd = out.buffer.asFloat32x4List(out.offsetInBytes, simdChunks);

    for (int i = 0; i < simdChunks; i++) {
      outSimd[i] = aSimd[i] - bSimd[i];
    }
    for (int i = simdChunks << 2; i < len; i++) {
      out[i] = a[i] - b[i];
    }
  }

  static void scaleFloat(Float32List a, double scalar, Float32List out) {
    final int len = a.length;
    final int simdChunks = len >> 2;
    final scalarSimd = Float32x4.splat(scalar);
    final aSimd = a.buffer.asFloat32x4List(a.offsetInBytes, simdChunks);
    final outSimd = out.buffer.asFloat32x4List(out.offsetInBytes, simdChunks);

    for (int i = 0; i < simdChunks; i++) {
      outSimd[i] = aSimd[i] * scalarSimd;
    }
    for (int i = simdChunks << 2; i < len; i++) {
      out[i] = a[i] * scalar;
    }
  }

  static double sumFloat(Float32List a) {
    final int len = a.length;
    if (len == 0) return 0.0;

    final int simdChunks = len >> 2;
    final aSimd = a.buffer.asFloat32x4List(a.offsetInBytes, simdChunks);
    Float32x4 acc = Float32x4.zero();

    for (int i = 0; i < simdChunks; i++) {
      acc += aSimd[i];
    }

    double total = (acc.x + acc.y) + (acc.z + acc.w);
    for (int i = simdChunks << 2; i < len; i++) {
      total += a[i];
    }
    return total;
  }

  static (double min, double max) minMaxFloat(Float32List a) {
    if (a.isEmpty) return (0.0, 0.0);
    double minVal = a[0];
    double maxVal = a[0];
    for (int i = 1; i < a.length; i++) {
      final val = a[i];
      if (val < minVal) minVal = val;
      if (val > maxVal) maxVal = val;
    }
    return (minVal, maxVal);
  }

  // float32 offset path; peeled simd

  static void addFloatOffset(
    Float32List a,
    int aOffset,
    Float32List b,
    int bOffset,
    Float32List out,
    int outOffset,
    int len,
  ) {
    if (len <= 0) return;

    final aPhase = aOffset & 3;
    final bPhase = bOffset & 3;
    final outPhase = outOffset & 3;

    // phase match; peel loop to align
    if (aPhase == bPhase && bPhase == outPhase) {
      int prologue = (4 - aPhase) & 3;
      if (prologue > len) prologue = len;

      for (int i = 0; i < prologue; i++) {
        out[outOffset + i] = a[aOffset + i] + b[bOffset + i];
      }

      final remaining = len - prologue;
      final simdChunks = remaining >> 2;
      if (simdChunks > 0) {
        final aSimd = a.buffer.asFloat32x4List(
            a.offsetInBytes + ((aOffset + prologue) << 2), simdChunks);
        final bSimd = b.buffer.asFloat32x4List(
            b.offsetInBytes + ((bOffset + prologue) << 2), simdChunks);
        final outSimd = out.buffer.asFloat32x4List(
            out.offsetInBytes + ((outOffset + prologue) << 2), simdChunks);

        for (int i = 0; i < simdChunks; i++) {
          outSimd[i] = aSimd[i] + bSimd[i];
        }
      }

      final processed = prologue + (simdChunks << 2);
      for (int i = processed; i < len; i++) {
        out[outOffset + i] = a[aOffset + i] + b[bOffset + i];
      }
    } else {
      // scalar fallback; unrolled
      for (int i = 0; i < len; i++) {
        out[outOffset + i] = a[aOffset + i] + b[bOffset + i];
      }
    }
  }

  static double sumFloatOffset(Float32List a, int offset, int len) {
    if (len <= 0) return 0.0;

    int prologue = (4 - (offset & 3)) & 3;
    if (prologue > len) prologue = len;

    double total = 0.0;
    for (int i = 0; i < prologue; i++) {
      total += a[offset + i];
    }

    final remaining = len - prologue;
    final simdChunks = remaining >> 2;
    if (simdChunks > 0) {
      final aSimd = a.buffer.asFloat32x4List(
          a.offsetInBytes + ((offset + prologue) << 2), simdChunks);
      Float32x4 acc = Float32x4.zero();
      for (int i = 0; i < simdChunks; i++) {
        acc += aSimd[i];
      }
      total += (acc.x + acc.y) + (acc.z + acc.w);
    }

    final processed = prologue + (simdChunks << 2);
    for (int i = processed; i < len; i++) {
      total += a[offset + i];
    }
    return total;
  }

  // int32 path; four-way scalar unroll

  static void addInt(Int32List a, Int32List b, Int32List out) {
    final int len = a.length;
    final int unrolledLen = len & ~3;
    for (int i = 0; i < unrolledLen; i += 4) {
      out[i] = a[i] + b[i];
      out[i + 1] = a[i + 1] + b[i + 1];
      out[i + 2] = a[i + 2] + b[i + 2];
      out[i + 3] = a[i + 3] + b[i + 3];
    }
    for (int i = unrolledLen; i < len; i++) {
      out[i] = a[i] + b[i];
    }
  }

  static void subInt(Int32List a, Int32List b, Int32List out) {
    final int len = a.length;
    final int unrolledLen = len & ~3;
    for (int i = 0; i < unrolledLen; i += 4) {
      out[i] = a[i] - b[i];
      out[i + 1] = a[i + 1] - b[i + 1];
      out[i + 2] = a[i + 2] - b[i + 2];
      out[i + 3] = a[i + 3] - b[i + 3];
    }
    for (int i = unrolledLen; i < len; i++) {
      out[i] = a[i] - b[i];
    }
  }

  static void mulInt(Int32List a, Int32List b, Int32List out) {
    final int len = a.length;
    final int unrolledLen = len & ~3;
    for (int i = 0; i < unrolledLen; i += 4) {
      out[i] = a[i] * b[i];
      out[i + 1] = a[i + 1] * b[i + 1];
      out[i + 2] = a[i + 2] * b[i + 2];
      out[i + 3] = a[i + 3] * b[i + 3];
    }
    for (int i = unrolledLen; i < len; i++) {
      out[i] = a[i] * b[i];
    }
  }

  static int sumInt(Int32List a) {
    int total = 0;
    for (int i = 0; i < a.length; i++) {
      total += a[i];
    }
    return total;
  }

  // int32 offset path; scalar unroll

  static void addIntOffset(
    Int32List a,
    int aOffset,
    Int32List b,
    int bOffset,
    Int32List out,
    int outOffset,
    int len,
  ) {
    if (len <= 0) return;

    final int unrolledLen = len & ~3;
    for (int i = 0; i < unrolledLen; i += 4) {
      out[outOffset + i] = a[aOffset + i] + b[bOffset + i];
      out[outOffset + i + 1] = a[aOffset + i + 1] + b[bOffset + i + 1];
      out[outOffset + i + 2] = a[aOffset + i + 2] + b[bOffset + i + 2];
      out[outOffset + i + 3] = a[aOffset + i + 3] + b[bOffset + i + 3];
    }
    for (int i = unrolledLen; i < len; i++) {
      out[outOffset + i] = a[aOffset + i] + b[bOffset + i];
    }
  }
}
