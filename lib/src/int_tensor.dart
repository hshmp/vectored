import 'dart:typed_data';
import 'simd_ops.dart';

/// contiguous int32 tensor; simd-accelerated
class IntTensor {
  final Int32List _data;
  final List<int> _shape;
  final List<int> _strides;
  final int _offset;
  int _runningSum = 0;
  final bool _tracked;

  IntTensor._(this._data, this._shape, this._strides, this._offset,
      {bool tracked = false})
      : _tracked = tracked {
    if (_tracked) {
      _runningSum = SimdOps.sumInt(_data);
    }
  }

  /// Creates a vector tensor with [length] elements initialized to zero.
  factory IntTensor.vector(int length, {bool tracked = false}) {
    return IntTensor._(Int32List(length), [length], [1], 0, tracked: tracked);
  }

  /// Creates a tensor from the values in [values].
  factory IntTensor.fromList(List<int> values, {bool tracked = false}) {
    return IntTensor._(
      Int32List.fromList(values),
      [values.length],
      [1],
      0,
      tracked: tracked,
    );
  }

  int get length => _shape.fold(1, (acc, dim) => acc * dim);
  int get count => length;
  List<int> get shape => _shape;
  List<int> get strides => _strides;
  int get rank => _shape.length;
  int get offset => _offset;
  Int32List get buffer => _data;

  @pragma('vm:prefer-inline')
  int operator [](int i) => _data[_offset + i];

  @pragma('vm:prefer-inline')
  void operator []=(int i, int val) {
    if (_tracked) {
      _runningSum += (val - _data[_offset + i]);
    }
    _data[_offset + i] = val;
  }

  /// Reads the element at the multidimensional coordinate [indices].
  @pragma('vm:prefer-inline')
  int at(List<int> indices) {
    assert(indices.length == _shape.length, 'Index rank mismatch');
    int idx = _offset;
    for (int i = 0; i < indices.length; i++) {
      idx += indices[i] * _strides[i];
    }
    return _data[idx];
  }

  /// Writes [val] to the multidimensional coordinate [indices].
  @pragma('vm:prefer-inline')
  void setAt(List<int> indices, int val) {
    assert(indices.length == _shape.length, 'Index rank mismatch');
    int idx = _offset;
    for (int i = 0; i < indices.length; i++) {
      idx += indices[i] * _strides[i];
    }
    if (_tracked) {
      _runningSum += (val - _data[idx]);
    }
    _data[idx] = val;
  }

  /// Returns a zero-copy view over the window from [start] to [end].
  IntTensor slice(int start, [int? end]) {
    final actualEnd = end ?? length;
    assert(start >= 0 && start <= length, 'Start out of bounds: $start');
    assert(actualEnd >= start && actualEnd <= length,
        'End out of bounds: $actualEnd');

    final sliceLen = actualEnd - start;
    return IntTensor._(_data, [sliceLen], [1], _offset + start, tracked: false);
  }

  /// Adds [other] into this tensor in-place and returns the mutated tensor.
  IntTensor add_(IntTensor other) {
    assert(length == other.length, 'Size mismatch');
    if (_offset == 0 && other._offset == 0) {
      SimdOps.addInt(_data, other._data, _data);
    } else {
      SimdOps.addIntOffset(
          _data, _offset, other._data, other._offset, _data, _offset, length);
    }
    if (_tracked) _runningSum = SimdOps.sumInt(_data);
    return this;
  }

  /// Subtracts [other] from this tensor in-place and returns the mutated tensor.
  IntTensor sub_(IntTensor other) {
    assert(length == other.length, 'Size mismatch');
    SimdOps.subInt(_data, other._data, _data);
    if (_tracked) _runningSum = SimdOps.sumInt(_data);
    return this;
  }

  /// Returns a new tensor containing the element-wise sum of this tensor and [other].
  IntTensor operator +(IntTensor other) {
    assert(length == other.length, 'Size mismatch');
    final out = IntTensor.vector(length);
    if (_offset == 0 && other._offset == 0) {
      SimdOps.addInt(_data, other._data, out._data);
    } else {
      SimdOps.addIntOffset(
          _data, _offset, other._data, other._offset, out._data, 0, length);
    }
    return out;
  }

  /// Returns a new tensor containing the element-wise difference of this tensor and [other].
  IntTensor operator -(IntTensor other) {
    assert(length == other.length, 'Size mismatch');
    final out = IntTensor.vector(length);
    SimdOps.subInt(_data, other._data, out._data);
    return out;
  }

  /// Returns a new tensor containing the element-wise product of this tensor and [other].
  IntTensor operator *(IntTensor other) {
    assert(length == other.length, 'Size mismatch');
    final out = IntTensor.vector(length);
    SimdOps.mulInt(_data, other._data, out._data);
    return out;
  }

  /// Returns the sum of all values in this tensor.
  int sum() => _tracked ? _runningSum : SimdOps.sumInt(_data);

  /// Returns the arithmetic mean of all values in this tensor.
  double mean() => count == 0 ? 0.0 : sum() / count;
}
