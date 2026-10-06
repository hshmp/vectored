import 'dart:typed_data';
import 'simd_ops.dart';

/// contiguous float32 tensor; simd-accelerated
class FloatTensor {
  final Float32List _data;
  final List<int> _shape;
  final List<int> _strides;
  final int _offset;

  FloatTensor._(this._data, this._shape, this._strides, this._offset);

  /// Creates a vector tensor with [length] elements initialized to zero.
  factory FloatTensor.vector(int length) {
    return FloatTensor._(Float32List(length), [length], [1], 0);
  }

  /// Creates a tensor from the values in [values].
  factory FloatTensor.fromList(List<double> values) {
    return FloatTensor._(Float32List.fromList(values), [values.length], [1], 0);
  }

  /// Creates a zero-initialized tensor with the given [shape].
  factory FloatTensor.zeros(List<int> shape) {
    final int size = shape.fold(1, (acc, dim) => acc * dim);
    final List<int> strides = _computeRowMajorStrides(shape);
    return FloatTensor._(
        Float32List(size), List.unmodifiable(shape), strides, 0);
  }

  /// Creates a tensor whose values vary evenly from [start] to [stop].
  factory FloatTensor.linspace(double start, double stop, int count) {
    assert(count > 1, 'Count must be at least 2');
    final tensor = FloatTensor.vector(count);
    final step = (stop - start) / (count - 1);
    for (int i = 0; i < count; i++) {
      tensor[i] = start + (step * i);
    }
    return tensor;
  }

  /// Creates a tensor whose values begin at [start] and step by [step].
  factory FloatTensor.arange(double start, double stop, [double step = 1.0]) {
    assert(step > 0, 'Step must be positive');
    final int count = ((stop - start) / step).ceil();
    final tensor = FloatTensor.vector(count);
    for (int i = 0; i < count; i++) {
      tensor[i] = start + (step * i);
    }
    return tensor;
  }

  static List<int> _computeRowMajorStrides(List<int> shape) {
    final strides = List<int>.filled(shape.length, 1);
    for (int i = shape.length - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
    return List.unmodifiable(strides);
  }

  int get length => _shape.fold(1, (acc, dim) => acc * dim);
  int get count => length;
  List<int> get shape => _shape;
  List<int> get strides => _strides;
  int get rank => _shape.length;
  int get offset => _offset;
  Float32List get buffer => _data;

  @pragma('vm:prefer-inline')
  double operator [](int i) => _data[_offset + i];

  @pragma('vm:prefer-inline')
  void operator []=(int i, double val) => _data[_offset + i] = val;

  /// Reads the element at the multidimensional coordinate [indices].
  @pragma('vm:prefer-inline')
  double at(List<int> indices) {
    assert(indices.length == _shape.length, 'Index rank mismatch');
    int idx = _offset;
    for (int i = 0; i < indices.length; i++) {
      idx += indices[i] * _strides[i];
    }
    return _data[idx];
  }

  /// Writes [val] to the multidimensional coordinate [indices].
  @pragma('vm:prefer-inline')
  void setAt(List<int> indices, double val) {
    assert(indices.length == _shape.length, 'Index rank mismatch');
    int idx = _offset;
    for (int i = 0; i < indices.length; i++) {
      idx += indices[i] * _strides[i];
    }
    _data[idx] = val;
  }

  /// Returns a zero-copy view over the window from [start] to [end].
  FloatTensor slice(int start, [int? end]) {
    final actualEnd = end ?? length;
    assert(start >= 0 && start <= length, 'Start out of bounds: $start');
    assert(actualEnd >= start && actualEnd <= length,
        'End out of bounds: $actualEnd');

    final sliceLen = actualEnd - start;
    return FloatTensor._(_data, [sliceLen], [1], _offset + start);
  }

  /// Adds [other] into this tensor in-place and returns the mutated tensor.
  FloatTensor add_(FloatTensor other) {
    assert(length == other.length, 'Size mismatch');
    if (_offset == 0 && other._offset == 0) {
      SimdOps.addFloat(_data, other._data, _data); // fast path
    } else {
      SimdOps.addFloatOffset(_data, _offset, other._data, other._offset, _data,
          _offset, length); // peeled path
    }
    return this;
  }

  /// Subtracts [other] from this tensor in-place and returns the mutated tensor.
  FloatTensor sub_(FloatTensor other) {
    assert(length == other.length, 'Size mismatch');
    SimdOps.subFloat(_data, other._data, _data);
    return this;
  }

  /// Scales this tensor in-place by [scalar] and returns the mutated tensor.
  FloatTensor scale_(double scalar) {
    SimdOps.scaleFloat(_data, scalar, _data);
    return this;
  }

  /// Returns a new tensor containing the element-wise sum of this tensor and [other].
  FloatTensor operator +(FloatTensor other) {
    assert(length == other.length, 'Size mismatch');
    final out = FloatTensor.zeros(_shape);
    if (_offset == 0 && other._offset == 0) {
      SimdOps.addFloat(_data, other._data, out._data);
    } else {
      SimdOps.addFloatOffset(
          _data, _offset, other._data, other._offset, out._data, 0, length);
    }
    return out;
  }

  /// Returns a new tensor containing the element-wise difference of this tensor and [other].
  FloatTensor operator -(FloatTensor other) {
    assert(length == other.length, 'Size mismatch');
    final out = FloatTensor.zeros(_shape);
    SimdOps.subFloat(_data, other._data, out._data);
    return out;
  }

  /// Returns a new tensor containing the element-wise product of this tensor and [other].
  FloatTensor operator *(dynamic other) {
    final out = FloatTensor.zeros(_shape);
    if (other is double) {
      SimdOps.scaleFloat(_data, other, out._data);
    } else if (other is FloatTensor) {
      assert(length == other.length, 'Size mismatch');
      // elem-wise
      for (int i = 0; i < length; i++) {
        out[i] = this[i] * other[i];
      }
    } else {
      throw ArgumentError('Unsupported operand: ${other.runtimeType}');
    }
    return out;
  }

  /// Returns the sum of all values in this tensor.
  double sum() {
    return _offset == 0
        ? SimdOps.sumFloat(_data)
        : SimdOps.sumFloatOffset(_data, _offset, length);
  }

  /// Returns the arithmetic mean of all values in this tensor.
  double mean() {
    if (count == 0) throw StateError('Empty tensor');
    return sum() / count;
  }

  /// Returns the most frequent value in this tensor.
  double mode() {
    if (count == 0) throw StateError('Empty tensor');
    final frequency = <double, int>{};
    double maxVal = this[0];
    int maxCount = 0;

    for (int i = 0; i < count; i++) {
      final val = this[i];
      final c = (frequency[val] ?? 0) + 1;
      frequency[val] = c;
      if (c > maxCount) {
        maxCount = c;
        maxVal = val;
      }
    }
    return maxVal;
  }

  /// Returns the minimum and maximum values in this tensor as a tuple.
  (double min, double max) get bounds => SimdOps.minMaxFloat(_data);
}
