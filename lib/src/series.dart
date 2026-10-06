import 'bitset.dart';
import 'float_tensor.dart';
import 'int_tensor.dart';
import 'text_series.dart';

/// 1d labeled column view
///
/// Typed series mark missing values in a [nulls] bitset and keep a
/// placeholder (0, 0.0, '' or false) in the value slot, so the values stay
/// in one compact typed array.
sealed class Series<T> {
  final String name;
  final List<Object>? index;
  Bitset? _nulls;

  Series(this.name, {this.index, Bitset? nulls})
      : _nulls = nulls != null && nulls.any ? nulls : null;

  int get length;

  /// Returns the stored value at [i]; a placeholder when [isNull] is true.
  T operator [](int i);

  /// Flags rows that hold no value, or `null` when every row has a value.
  Bitset? get nulls => _nulls;

  /// Returns whether row [i] holds no value.
  bool isNull(int i) => _nulls?[i] ?? false;

  /// The number of rows that hold no value.
  int get nullCount => _nulls?.count ?? 0;

  /// Returns the value at [i], or `null` when the row holds no value.
  Object? valueAt(int i) => isNull(i) ? null : this[i];

  /// Creates a float series from [data] using a float tensor backing store.
  ///
  /// `null` entries are recorded in [nulls]. The optional [index] must match
  /// the number of rows in [data].
  static FloatSeries fromFloats(String name, List<double?> data,
      {List<Object>? index}) {
    final (values, nulls) = _split(data, 0.0);
    return FloatSeries(name, FloatTensor.fromList(values),
        index: index, nulls: nulls);
  }

  /// Creates an int series from [data] using an int tensor backing store.
  ///
  /// `null` entries are recorded in [nulls]. The optional [index] must match
  /// the number of rows in [data].
  static IntSeries fromInts(String name, List<int?> data,
      {List<Object>? index}) {
    final (values, nulls) = _split(data, 0);
    return IntSeries(name, IntTensor.fromList(values),
        index: index, nulls: nulls);
  }

  /// Creates a string series from [data].
  ///
  /// `null` entries are recorded in [nulls]. The optional [index] must match
  /// the number of rows in [data].
  static StringSeries fromStrings(String name, List<String?> data,
      {List<Object>? index}) {
    final (values, nulls) = _split(data, '');
    return StringSeries(name, values, index: index, nulls: nulls);
  }

  /// Creates a boolean series from [data], packed one bit per row.
  ///
  /// `null` entries are recorded in [nulls].
  static BoolSeries fromBools(String name, List<bool?> data,
      {List<Object>? index}) {
    final (values, nulls) = _split(data, false);
    return BoolSeries(name, Bitset.fromBools(values),
        index: index, nulls: nulls);
  }

  /// Creates a general-purpose series from [data].
  static ObjectSeries fromObjects(String name, List<Object?> data,
      {List<Object>? index}) {
    return ObjectSeries(name, data, index: index);
  }

  /// Splits nullable [data] into placeholder-filled values and a null mask.
  static (List<T>, Bitset?) _split<T extends Object>(
      List<T?> data, T placeholder) {
    if (data is List<T>) return (data, null);
    Bitset? nulls;
    final values = List<T>.generate(data.length, (i) {
      final value = data[i];
      if (value != null) return value;
      (nulls ??= Bitset(data.length)).set(i);
      return placeholder;
    });
    return (values, nulls);
  }

  /// Returns [nulls] gathered at [indices], for typed `take` results.
  Bitset? _takeNulls(List<int> indices) {
    final source = _nulls;
    if (source == null) return null;
    final output = Bitset(indices.length);
    for (var i = 0; i < indices.length; i++) {
      if (source[indices[i]]) output.set(i);
    }
    return output;
  }

  void _checkNulls() {
    final nulls = _nulls;
    if (nulls != null && nulls.length != length) {
      throw ArgumentError.value(
          nulls.length, 'nulls', 'Null mask length must match data length');
    }
  }
}

/// Returns the union of two null masks, or `null` when neither has nulls.
Bitset? _unionNulls(Bitset? left, Bitset? right) {
  if (left == null) return right?.copy();
  if (right == null) return left.copy();
  return left | right;
}

/// float column; simd-backed
class FloatSeries extends Series<double> {
  final FloatTensor tensor;

  FloatSeries(super.name, this.tensor, {super.index, super.nulls}) {
    _checkIndex(index, tensor.length);
    _checkNulls();
  }

  @override
  int get length => tensor.length;

  @override
  double operator [](int i) => tensor[i];

  /// Returns the sum of non-null values.
  double sum() => tensor.sum();

  /// Returns the mean of non-null values.
  double mean() =>
      _nulls == null ? tensor.mean() : sum() / (length - nullCount);

  /// Returns the smallest and largest non-null values.
  (double min, double max) get bounds {
    final nulls = _nulls;
    if (nulls == null) return tensor.bounds;
    double? low;
    double? high;
    for (var i = 0; i < length; i++) {
      if (nulls[i]) continue;
      final value = tensor[i];
      if (low == null || value < low) low = value;
      if (high == null || value > high) high = value;
    }
    return (low ?? 0.0, high ?? 0.0);
  }

  /// Copies selected values into a new typed series.
  FloatSeries take(List<int> indices) {
    final output = FloatTensor.vector(indices.length);
    for (var i = 0; i < indices.length; i++) {
      output[i] = tensor[indices[i]];
    }
    return FloatSeries(name, output, nulls: _takeNulls(indices));
  }

  // alloc; null if either side null

  FloatSeries operator +(FloatSeries other) =>
      FloatSeries(name, tensor + other.tensor,
          index: index, nulls: _unionNulls(_nulls, other._nulls));

  FloatSeries operator -(FloatSeries other) =>
      FloatSeries(name, tensor - other.tensor,
          index: index, nulls: _unionNulls(_nulls, other._nulls));

  FloatSeries operator *(double scalar) =>
      FloatSeries(name, tensor * scalar, index: index, nulls: _nulls?.copy());

  // in-place

  /// Adds [other] to this series in-place and returns the mutated series.
  ///
  /// The [other] series must have the same length as this series.
  FloatSeries add_(FloatSeries other) {
    tensor.add_(other.tensor);
    _nulls = _unionNulls(_nulls, other._nulls);
    return this;
  }

  /// Subtracts [other] from this series in-place and returns the mutated series.
  ///
  /// The [other] series must have the same length as this series.
  FloatSeries sub_(FloatSeries other) {
    tensor.sub_(other.tensor);
    _nulls = _unionNulls(_nulls, other._nulls);
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

  IntSeries(super.name, this.tensor, {super.index, super.nulls}) {
    _checkIndex(index, tensor.length);
    _checkNulls();
  }

  @override
  int get length => tensor.length;

  @override
  int operator [](int i) => tensor[i];

  /// Returns the sum of non-null values.
  int sum() => tensor.sum();

  /// Returns the mean of non-null values.
  double mean() =>
      _nulls == null ? tensor.mean() : sum() / (length - nullCount);

  /// Copies selected values into a new typed series.
  IntSeries take(List<int> indices) {
    final output = IntTensor.vector(indices.length);
    for (var i = 0; i < indices.length; i++) {
      output[i] = tensor[indices[i]];
    }
    return IntSeries(name, output, nulls: _takeNulls(indices));
  }

  // alloc; null if either side null

  IntSeries operator +(IntSeries other) =>
      IntSeries(name, tensor + other.tensor,
          index: index, nulls: _unionNulls(_nulls, other._nulls));

  IntSeries operator -(IntSeries other) =>
      IntSeries(name, tensor - other.tensor,
          index: index, nulls: _unionNulls(_nulls, other._nulls));

  // in-place

  /// Adds [other] to this series in-place and returns the mutated series.
  ///
  /// The [other] series must have the same length as this series.
  IntSeries add_(IntSeries other) {
    tensor.add_(other.tensor);
    _nulls = _unionNulls(_nulls, other._nulls);
    return this;
  }

  /// Subtracts [other] from this series in-place and returns the mutated series.
  ///
  /// The [other] series must have the same length as this series.
  IntSeries sub_(IntSeries other) {
    tensor.sub_(other.tensor);
    _nulls = _unionNulls(_nulls, other._nulls);
    return this;
  }
}

/// string column
class StringSeries extends Series<String> {
  final TextSeries _text;

  StringSeries(super.name, List<String> data, {super.index, super.nulls})
      : _text = TextSeries.fromStrings(name, data, index: index) {
    _checkNulls();
  }

  StringSeries._(super.name, this._text, {super.nulls});

  @override
  int get length => _text.length;

  @override
  String operator [](int i) => _text[i];

  /// The packed UTF-8 storage behind this series.
  TextSeries get text => _text;

  /// Returns a materialized list of the decoded strings; nulls stay `null`.
  List<String?> get data => List<String?>.unmodifiable(
      List.generate(length, (i) => valueAt(i) as String?));

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
      StringSeries._(name, _text.take(indices), nulls: _takeNulls(indices));
}

/// boolean column; one bit per row
class BoolSeries extends Series<bool> {
  /// The packed flags; rows marked in [nulls] hold `false`.
  final Bitset values;

  BoolSeries(super.name, this.values, {super.index, super.nulls}) {
    _checkIndex(index, values.length);
    _checkNulls();
  }

  @override
  int get length => values.length;

  @override
  bool operator [](int i) => values[i];

  /// The number of rows that are `true`.
  int get trueCount => values.count;

  /// Copies selected values into a new boolean series.
  BoolSeries take(List<int> indices) {
    final output = Bitset(indices.length);
    for (var i = 0; i < indices.length; i++) {
      if (values[indices[i]]) output.set(i);
    }
    return BoolSeries(name, output, nulls: _takeNulls(indices));
  }
}

/// A general-purpose column for values without a specialized series type.
///
/// `null` values are stored directly, so [nulls] is always `null` here and
/// [isNull] inspects the value.
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

  @override
  bool isNull(int i) => _data[i] == null;

  @override
  int get nullCount => _data.where((value) => value == null).length;

  @override
  Object? valueAt(int i) => _data[i];

  /// Returns the immutable values in this series.
  List<Object?> get data => _data;
}

void _checkIndex(List<Object>? index, int length) {
  if (index != null && index.length != length) {
    throw ArgumentError.value(
        index.length, 'index', 'Index length must match data length');
  }
}
