import 'float_tensor.dart';
import 'int_tensor.dart';
import 'text_series.dart';

/// 1d labeled column view
sealed class Series<T> {
  final String name;
  final List<Object>? index;

  Series(this.name, {this.index});

  int get length;
  T operator [](int i);

  /// Creates a float series from [data] using a float tensor backing store.
  ///
  /// The optional [index] must match the number of rows in [data].
  static FloatSeries fromFloats(String name, List<double> data,
      {List<Object>? index}) {
    return FloatSeries(name, FloatTensor.fromList(data), index: index);
  }

  /// Creates an int series from [data] using an int tensor backing store.
  ///
  /// The optional [index] must match the number of rows in [data].
  static IntSeries fromInts(String name, List<int> data,
      {List<Object>? index}) {
    return IntSeries(name, IntTensor.fromList(data), index: index);
  }

  /// Creates a string series from [data].
  ///
  /// The optional [index] must match the number of rows in [data].
  static StringSeries fromStrings(String name, List<String> data,
      {List<Object>? index}) {
    return StringSeries(name, data, index: index);
  }

  /// Creates a general-purpose series from [data].
  static ObjectSeries fromObjects(String name, List<Object?> data,
      {List<Object>? index}) {
    return ObjectSeries(name, data, index: index);
  }
}

/// float column; simd-backed
class FloatSeries extends Series<double> {
  final FloatTensor tensor;

  FloatSeries(super.name, this.tensor, {super.index}) {
    _checkIndex(index, tensor.length);
  }

  @override
  int get length => tensor.length;

  @override
  double operator [](int i) => tensor[i];

  double sum() => tensor.sum();
  double mean() => tensor.mean();
  (double min, double max) get bounds => tensor.bounds;

  /// Copies selected values into a new typed series.
  FloatSeries take(List<int> indices) {
    final output = FloatTensor.vector(indices.length);
    for (var i = 0; i < indices.length; i++) {
      output[i] = tensor[indices[i]];
    }
    return FloatSeries(name, output);
  }

  // alloc

  FloatSeries operator +(FloatSeries other) {
    return FloatSeries(name, tensor + other.tensor, index: index);
  }

  FloatSeries operator -(FloatSeries other) {
    return FloatSeries(name, tensor - other.tensor, index: index);
  }

  FloatSeries operator *(double scalar) {
    return FloatSeries(name, tensor * scalar, index: index);
  }

  // in-place

  /// Adds [other] to this series in-place and returns the mutated series.
  ///
  /// The [other] series must have the same length as this series.
  FloatSeries add_(FloatSeries other) {
    tensor.add_(other.tensor);
    return this;
  }

  /// Subtracts [other] from this series in-place and returns the mutated series.
  ///
  /// The [other] series must have the same length as this series.
  FloatSeries sub_(FloatSeries other) {
    tensor.sub_(other.tensor);
    return this;
  }

  /// Scales this series in-place by [scalar] and returns the mutated series.
  FloatSeries scale_(double scalar) {
    tensor.scale_(scalar);
    return this;
  }
}

/// int column; simd-backed
class IntSeries extends Series<int> {
  final IntTensor tensor;

  IntSeries(super.name, this.tensor, {super.index}) {
    _checkIndex(index, tensor.length);
  }

  @override
  int get length => tensor.length;

  @override
  int operator [](int i) => tensor[i];

  int sum() => tensor.sum();
  double mean() => tensor.mean();

  /// Copies selected values into a new typed series.
  IntSeries take(List<int> indices) {
    final output = IntTensor.vector(indices.length);
    for (var i = 0; i < indices.length; i++) {
      output[i] = tensor[indices[i]];
    }
    return IntSeries(name, output);
  }

  // alloc

  IntSeries operator +(IntSeries other) {
    return IntSeries(name, tensor + other.tensor, index: index);
  }

  IntSeries operator -(IntSeries other) {
    return IntSeries(name, tensor - other.tensor, index: index);
  }

  // in-place

  /// Adds [other] to this series in-place and returns the mutated series.
  ///
  /// The [other] series must have the same length as this series.
  IntSeries add_(IntSeries other) {
    tensor.add_(other.tensor);
    return this;
  }

  /// Subtracts [other] from this series in-place and returns the mutated series.
  ///
  /// The [other] series must have the same length as this series.
  IntSeries sub_(IntSeries other) {
    tensor.sub_(other.tensor);
    return this;
  }
}

/// string column
class StringSeries extends Series<String> {
  final TextSeries _text;

  StringSeries(super.name, List<String> data, {super.index})
      : _text = TextSeries.fromStrings(name, data, index: index);

  StringSeries._(super.name, this._text);

  @override
  int get length => _text.length;

  @override
  String operator [](int i) => _text[i];

  /// Returns a materialized list of the decoded strings.
  List<String> get data =>
      List<String>.unmodifiable(List.generate(length, (i) => this[i]));

  /// Returns a mask of rows containing [pattern].
  IntTensor contains(String pattern) => _text.contains(pattern);

  /// Returns a mask of rows matching [pattern].
  IntTensor match(RegExp pattern) => _text.match(pattern);

  /// Returns row indices containing [pattern] without allocating a mask.
  List<int> containingIndices(String pattern) =>
      _text.containingIndices(pattern);

  /// Returns matching row indices without allocating a mask.
  List<int> matchingIndices(RegExp pattern) => _text.matchingIndices(pattern);

  /// Copies selected rows while preserving their packed UTF-8 representation.
  StringSeries take(List<int> indices) =>
      StringSeries._(name, _text.take(indices));
}

/// A general-purpose column for values without a specialized series type.
class ObjectSeries extends Series<Object?> {
  final List<Object?> _data;

  ObjectSeries(super.name, List<Object?> data, {super.index})
      : _data = List<Object?>.unmodifiable(data) {
    _checkIndex(index, data.length);
  }

  /// Copies selected values into a new object series.
  ObjectSeries take(List<int> indices) =>
      ObjectSeries(name, [for (final index in indices) _data[index]]);

  @override
  int get length => _data.length;

  @override
  Object? operator [](int i) => _data[i];

  /// Returns the immutable values in this series.
  List<Object?> get data => _data;
}

void _checkIndex(List<Object>? index, int length) {
  if (index != null && index.length != length) {
    throw ArgumentError.value(
        index.length, 'index', 'Index length must match data length');
  }
}
