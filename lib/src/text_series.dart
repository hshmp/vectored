import 'dart:convert';
import 'dart:typed_data';
import 'int_tensor.dart';

/// text series; utf8 + int32 offsets
class TextSeries {
  final String name;
  final Uint8List _bytes;
  final Int32List _offsets;
  final List<Object>? index;

  TextSeries._(this.name, this._bytes, this._offsets, {this.index});

  /// Builds a text series from [strings] using a compact UTF-8 buffer.
  ///
  /// The optional [index] must match the number of rows in [strings].
  factory TextSeries.fromStrings(
    String name,
    List<String> strings, {
    List<Object>? index,
  }) {
    if (index != null) {
      assert(index.length == strings.length, 'Index length mismatch');
    }

    final int rowCount = strings.length;
    final offsets = Int32List(rowCount + 1);
    final builder = BytesBuilder(copy: false);

    int currentOffset = 0;
    for (int i = 0; i < rowCount; i++) {
      offsets[i] = currentOffset;
      final encoded = utf8.encode(strings[i]);
      builder.add(encoded);
      currentOffset += encoded.length;
    }
    offsets[rowCount] = currentOffset;

    return TextSeries._(name, builder.takeBytes(), offsets, index: index);
  }

  int get length => _offsets.length - 1;

  /// Returns the decoded row at index [i].
  String operator [](int i) {
    assert(i >= 0 && i < length, 'Index out of range: $i');
    return utf8.decode(
      Uint8List.sublistView(_bytes, _offsets[i], _offsets[i + 1]),
    );
  }

  /// Returns the byte length of the row at index [i].
  int byteLength(int i) => _offsets[i + 1] - _offsets[i];

  /// Returns a mask of rows that contain [pattern].
  IntTensor contains(String pattern) {
    final patternBytes = Uint8List.fromList(utf8.encode(pattern));
    final result = IntTensor.vector(length);
    _scanContains(patternBytes, (row) => result[row] = 1);
    return result;
  }

  /// Returns the row indices containing [pattern] without allocating a mask.
  List<int> containingIndices(String pattern) {
    final indices = <int>[];
    _scanContains(Uint8List.fromList(utf8.encode(pattern)), indices.add);
    return indices;
  }

  void _scanContains(Uint8List patternBytes, void Function(int) onMatch) {
    final int m = patternBytes.length;
    // empty pattern matches all
    if (m == 0) {
      for (int i = 0; i < length; i++) {
        onMatch(i);
      }
      return;
    }

    if (m > _bytes.length) return;

    // fast path; m==1
    // linear scan; no table
    if (m == 1) {
      final int target = patternBytes[0];
      for (int row = 0; row < length; row++) {
        final int start = _offsets[row];
        final int end = _offsets[row + 1];
        for (int i = start; i < end; i++) {
          if (_bytes[i] == target) {
            onMatch(row);
            break;
          }
        }
      }
      return;
    }

    // prefix table; linear-time fallback
    final prefix = Int32List(m);
    for (int i = 1, matched = 0; i < m; i++) {
      while (matched > 0 && patternBytes[i] != patternBytes[matched]) {
        matched = prefix[matched - 1];
      }
      if (patternBytes[i] == patternBytes[matched]) {
        matched++;
      }
      prefix[i] = matched;
    }

    final int lastByte = patternBytes[m - 1];
    final shift = Int32List(256)..fillRange(0, 256, m);
    for (int i = 0; i < m - 1; i++) {
      shift[patternBytes[i]] = m - 1 - i;
    }

    // bounded bmh; kmp fallback caps repetitive-input work
    for (int row = 0; row < length; row++) {
      final int rowStart = _offsets[row];
      final int rowEnd = _offsets[row + 1];
      final int rowLen = rowEnd - rowStart;
      if (rowLen < m) continue;

      final int workBudget = rowLen * 2;
      var work = 0;
      var textIdx = rowStart + m - 1;
      var fallback = false;

      while (textIdx < rowEnd) {
        work++;
        final int tailByte = _bytes[textIdx];
        if (work > workBudget) {
          fallback = true;
          break;
        }

        if (tailByte == lastByte) {
          work++;
          if (work > workBudget) {
            fallback = true;
            break;
          }
          if (_bytes[textIdx - m + 1] == patternBytes[0]) {
            var patIdx = m - 2;
            var curTextIdx = textIdx - 1;
            while (patIdx > 0) {
              work++;
              if (work > workBudget) {
                fallback = true;
                break;
              }
              if (_bytes[curTextIdx] != patternBytes[patIdx]) break;
              patIdx--;
              curTextIdx--;
            }

            if (fallback) break;
            if (patIdx == 0) {
              onMatch(row);
              break;
            }
          }
        }

        textIdx += shift[tailByte];
      }

      if (fallback) {
        var matched = 0;
        for (int i = rowStart; i < rowEnd; i++) {
          while (matched > 0 && _bytes[i] != patternBytes[matched]) {
            matched = prefix[matched - 1];
          }
          if (_bytes[i] == patternBytes[matched]) {
            matched++;
          }
          if (matched == m) {
            onMatch(row);
            break;
          }
        }
      }
    }
  }

  /// Returns a mask of rows where [pattern] matches any substring.
  ///
  /// Pass a precompiled [RegExp] to reuse its compiled form across searches.
  IntTensor match(RegExp pattern) {
    final result = IntTensor.vector(length);
    for (int row = 0; row < length; row++) {
      if (pattern.hasMatch(this[row])) {
        result[row] = 1;
      }
    }
    return result;
  }

  /// Returns a new series containing the rows at [indices].
  TextSeries take(List<int> indices) {
    final offsets = Int32List(indices.length + 1);
    final builder = BytesBuilder(copy: false);
    var byteOffset = 0;
    for (var outputRow = 0; outputRow < indices.length; outputRow++) {
      final row = indices[outputRow];
      final start = _offsets[row];
      final end = _offsets[row + 1];
      offsets[outputRow] = byteOffset;
      builder.add(Uint8List.sublistView(_bytes, start, end));
      byteOffset += end - start;
    }
    offsets[indices.length] = byteOffset;
    return TextSeries._(name, builder.takeBytes(), offsets);
  }

  /// Returns matching row indices for [pattern] without allocating a mask.
  List<int> matchingIndices(RegExp pattern) {
    final indices = <int>[];
    for (var row = 0; row < length; row++) {
      if (pattern.hasMatch(this[row])) indices.add(row);
    }
    return indices;
  }
}
