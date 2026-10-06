import 'dart:typed_data';

/// float32 simd and int32 arithmetic kernels
///
/// Full-buffer kernels delegate to the offset kernels, which process exactly
/// `len` elements starting at each offset.
final class SimdOps {
  // float32x4 lanes per float32 partial; caps rounding drift on large sums
  static const int _sumBlock = 256;

  // full-buffer float32 path

  static void addFloat(Float32List a, Float32List b, Float32List out) =>
      addFloatOffset(a, 0, b, 0, out, 0, a.length);

  static void subFloat(Float32List a, Float32List b, Float32List out) =>
      subFloatOffset(a, 0, b, 0, out, 0, a.length);

  static void mulFloat(Float32List a, Float32List b, Float32List out) =>
      mulFloatOffset(a, 0, b, 0, out, 0, a.length);

  static void scaleFloat(Float32List a, double scalar, Float32List out) =>
      scaleFloatOffset(a, 0, scalar, out, 0, a.length);

  static double sumFloat(Float32List a) => sumFloatOffset(a, 0, a.length);

  static (double min, double max) minMaxFloat(Float32List a) =>
      minMaxFloatOffset(a, 0, a.length);

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

    final prologue =
        _sharedPrologue(a, aOffset, b, bOffset, out, outOffset, len);

    // phase mismatch; scalar fallback
    if (prologue < 0) {
      for (int i = 0; i < len; i++) {
        out[outOffset + i] = a[aOffset + i] + b[bOffset + i];
      }
      return;
    }

    for (int i = 0; i < prologue; i++) {
      out[outOffset + i] = a[aOffset + i] + b[bOffset + i];
    }

    final simdChunks = (len - prologue) >> 2;
    if (simdChunks > 0) {
      final aSimd = _float32x4(a, aOffset + prologue, simdChunks);
      final bSimd = _float32x4(b, bOffset + prologue, simdChunks);
      final outSimd = _float32x4(out, outOffset + prologue, simdChunks);

      for (int i = 0; i < simdChunks; i++) {
        outSimd[i] = aSimd[i] + bSimd[i];
      }
    }

    for (int i = prologue + (simdChunks << 2); i < len; i++) {
      out[outOffset + i] = a[aOffset + i] + b[bOffset + i];
    }
  }

  static void subFloatOffset(
    Float32List a,
    int aOffset,
    Float32List b,
    int bOffset,
    Float32List out,
    int outOffset,
    int len,
  ) {
    if (len <= 0) return;

    final prologue =
        _sharedPrologue(a, aOffset, b, bOffset, out, outOffset, len);

    if (prologue < 0) {
      for (int i = 0; i < len; i++) {
        out[outOffset + i] = a[aOffset + i] - b[bOffset + i];
      }
      return;
    }

    for (int i = 0; i < prologue; i++) {
      out[outOffset + i] = a[aOffset + i] - b[bOffset + i];
    }

    final simdChunks = (len - prologue) >> 2;
    if (simdChunks > 0) {
      final aSimd = _float32x4(a, aOffset + prologue, simdChunks);
      final bSimd = _float32x4(b, bOffset + prologue, simdChunks);
      final outSimd = _float32x4(out, outOffset + prologue, simdChunks);

      for (int i = 0; i < simdChunks; i++) {
        outSimd[i] = aSimd[i] - bSimd[i];
      }
    }

    for (int i = prologue + (simdChunks << 2); i < len; i++) {
      out[outOffset + i] = a[aOffset + i] - b[bOffset + i];
    }
  }

  static void mulFloatOffset(
    Float32List a,
    int aOffset,
    Float32List b,
    int bOffset,
    Float32List out,
    int outOffset,
    int len,
  ) {
    if (len <= 0) return;

    final prologue =
        _sharedPrologue(a, aOffset, b, bOffset, out, outOffset, len);

    if (prologue < 0) {
      for (int i = 0; i < len; i++) {
        out[outOffset + i] = a[aOffset + i] * b[bOffset + i];
      }
      return;
    }

    for (int i = 0; i < prologue; i++) {
      out[outOffset + i] = a[aOffset + i] * b[bOffset + i];
    }

    final simdChunks = (len - prologue) >> 2;
    if (simdChunks > 0) {
      final aSimd = _float32x4(a, aOffset + prologue, simdChunks);
      final bSimd = _float32x4(b, bOffset + prologue, simdChunks);
      final outSimd = _float32x4(out, outOffset + prologue, simdChunks);

      for (int i = 0; i < simdChunks; i++) {
        outSimd[i] = aSimd[i] * bSimd[i];
      }
    }

    for (int i = prologue + (simdChunks << 2); i < len; i++) {
      out[outOffset + i] = a[aOffset + i] * b[bOffset + i];
    }
  }

  static void scaleFloatOffset(
    Float32List a,
    int aOffset,
    double scalar,
    Float32List out,
    int outOffset,
    int len,
  ) {
    if (len <= 0) return;

    final prologue =
        _sharedPrologue(a, aOffset, a, aOffset, out, outOffset, len);

    if (prologue < 0) {
      for (int i = 0; i < len; i++) {
        out[outOffset + i] = a[aOffset + i] * scalar;
      }
      return;
    }

    for (int i = 0; i < prologue; i++) {
      out[outOffset + i] = a[aOffset + i] * scalar;
    }

    final simdChunks = (len - prologue) >> 2;
    if (simdChunks > 0) {
      final scalarSimd = Float32x4.splat(scalar);
      final aSimd = _float32x4(a, aOffset + prologue, simdChunks);
      final outSimd = _float32x4(out, outOffset + prologue, simdChunks);

      for (int i = 0; i < simdChunks; i++) {
        outSimd[i] = aSimd[i] * scalarSimd;
      }
    }

    for (int i = prologue + (simdChunks << 2); i < len; i++) {
      out[outOffset + i] = a[aOffset + i] * scalar;
    }
  }

  static double sumFloatOffset(Float32List a, int offset, int len) {
    if (len <= 0) return 0.0;

    int prologue = (4 - _phase(a, offset)) & 3;
    if (prologue > len) prologue = len;

    double total = 0.0;
    for (int i = 0; i < prologue; i++) {
      total += a[offset + i];
    }

    final simdChunks = (len - prologue) >> 2;
    if (simdChunks > 0) {
      final aSimd = _float32x4(a, offset + prologue, simdChunks);

      // blocked float32 partials; double total
      for (int block = 0; block < simdChunks; block += _sumBlock) {
        final blockEnd =
            block + _sumBlock < simdChunks ? block + _sumBlock : simdChunks;
        Float32x4 acc = Float32x4.zero();
        for (int i = block; i < blockEnd; i++) {
          acc += aSimd[i];
        }
        total += (acc.x + acc.y) + (acc.z + acc.w);
      }
    }

    for (int i = prologue + (simdChunks << 2); i < len; i++) {
      total += a[offset + i];
    }
    return total;
  }

  static (double min, double max) minMaxFloatOffset(
    Float32List a,
    int offset,
    int len,
  ) {
    if (len <= 0) return (0.0, 0.0);

    double minVal = a[offset];
    double maxVal = a[offset];

    int prologue = (4 - _phase(a, offset)) & 3;
    if (prologue > len) prologue = len;

    for (int i = 1; i < prologue; i++) {
      final val = a[offset + i];
      if (val < minVal) minVal = val;
      if (val > maxVal) maxVal = val;
    }

    final simdChunks = (len - prologue) >> 2;
    if (simdChunks > 0) {
      final aSimd = _float32x4(a, offset + prologue, simdChunks);
      Float32x4 minAcc = aSimd[0];
      Float32x4 maxAcc = aSimd[0];

      for (int i = 1; i < simdChunks; i++) {
        final val = aSimd[i];
        minAcc = minAcc.min(val);
        maxAcc = maxAcc.max(val);
      }

      for (final val in [minAcc.x, minAcc.y, minAcc.z, minAcc.w]) {
        if (val < minVal) minVal = val;
      }
      for (final val in [maxAcc.x, maxAcc.y, maxAcc.z, maxAcc.w]) {
        if (val > maxVal) maxVal = val;
      }
    }

    for (int i = prologue + (simdChunks << 2); i < len; i++) {
      final val = a[offset + i];
      if (val < minVal) minVal = val;
      if (val > maxVal) maxVal = val;
    }
    return (minVal, maxVal);
  }

  // full-buffer int32 path

  static void addInt(Int32List a, Int32List b, Int32List out) =>
      addIntOffset(a, 0, b, 0, out, 0, a.length);

  static void subInt(Int32List a, Int32List b, Int32List out) =>
      subIntOffset(a, 0, b, 0, out, 0, a.length);

  static void mulInt(Int32List a, Int32List b, Int32List out) =>
      mulIntOffset(a, 0, b, 0, out, 0, a.length);

  static int sumInt(Int32List a) => sumIntOffset(a, 0, a.length);

  // int32 offset path; int32x4 add/sub, scalar unroll mul/sum

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

    final prologue =
        _sharedPrologue(a, aOffset, b, bOffset, out, outOffset, len);

    if (prologue < 0) {
      for (int i = 0; i < len; i++) {
        out[outOffset + i] = a[aOffset + i] + b[bOffset + i];
      }
      return;
    }

    for (int i = 0; i < prologue; i++) {
      out[outOffset + i] = a[aOffset + i] + b[bOffset + i];
    }

    final simdChunks = (len - prologue) >> 2;
    if (simdChunks > 0) {
      final aSimd = _int32x4(a, aOffset + prologue, simdChunks);
      final bSimd = _int32x4(b, bOffset + prologue, simdChunks);
      final outSimd = _int32x4(out, outOffset + prologue, simdChunks);

      for (int i = 0; i < simdChunks; i++) {
        outSimd[i] = aSimd[i] + bSimd[i];
      }
    }

    for (int i = prologue + (simdChunks << 2); i < len; i++) {
      out[outOffset + i] = a[aOffset + i] + b[bOffset + i];
    }
  }

  static void subIntOffset(
    Int32List a,
    int aOffset,
    Int32List b,
    int bOffset,
    Int32List out,
    int outOffset,
    int len,
  ) {
    if (len <= 0) return;

    final prologue =
        _sharedPrologue(a, aOffset, b, bOffset, out, outOffset, len);

    if (prologue < 0) {
      for (int i = 0; i < len; i++) {
        out[outOffset + i] = a[aOffset + i] - b[bOffset + i];
      }
      return;
    }

    for (int i = 0; i < prologue; i++) {
      out[outOffset + i] = a[aOffset + i] - b[bOffset + i];
    }

    final simdChunks = (len - prologue) >> 2;
    if (simdChunks > 0) {
      final aSimd = _int32x4(a, aOffset + prologue, simdChunks);
      final bSimd = _int32x4(b, bOffset + prologue, simdChunks);
      final outSimd = _int32x4(out, outOffset + prologue, simdChunks);

      for (int i = 0; i < simdChunks; i++) {
        outSimd[i] = aSimd[i] - bSimd[i];
      }
    }

    for (int i = prologue + (simdChunks << 2); i < len; i++) {
      out[outOffset + i] = a[aOffset + i] - b[bOffset + i];
    }
  }

  static void mulIntOffset(
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
      out[outOffset + i] = a[aOffset + i] * b[bOffset + i];
      out[outOffset + i + 1] = a[aOffset + i + 1] * b[bOffset + i + 1];
      out[outOffset + i + 2] = a[aOffset + i + 2] * b[bOffset + i + 2];
      out[outOffset + i + 3] = a[aOffset + i + 3] * b[bOffset + i + 3];
    }
    for (int i = unrolledLen; i < len; i++) {
      out[outOffset + i] = a[aOffset + i] * b[bOffset + i];
    }
  }

  static int sumIntOffset(Int32List a, int offset, int len) {
    int total = 0;
    final end = offset + len;
    for (int i = offset; i < end; i++) {
      total += a[i];
    }
    return total;
  }

  // alignment helpers

  /// Returns the element phase of [offset] within a 16-byte lane for [list].
  static int _phase(TypedData list, int offset) =>
      ((list.offsetInBytes >> 2) + offset) & 3;

  /// Returns the scalar prologue that aligns all three operands, or -1 when
  /// their lane phases differ and SIMD loads cannot be aligned together.
  static int _sharedPrologue(
    TypedData a,
    int aOffset,
    TypedData b,
    int bOffset,
    TypedData out,
    int outOffset,
    int len,
  ) {
    final phase = _phase(a, aOffset);
    if (phase != _phase(b, bOffset) || phase != _phase(out, outOffset)) {
      return -1;
    }
    final prologue = (4 - phase) & 3;
    return prologue > len ? len : prologue;
  }

  static Float32x4List _float32x4(Float32List list, int offset, int chunks) =>
      list.buffer.asFloat32x4List(list.offsetInBytes + (offset << 2), chunks);

  static Int32x4List _int32x4(Int32List list, int offset, int chunks) =>
      list.buffer.asInt32x4List(list.offsetInBytes + (offset << 2), chunks);
}
