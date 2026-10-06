part of 'data_frame.dart';

/// Rows of a [DataFrame] grouped by the values of one or more key columns.
///
/// Created by [DataFrame.groupBy]. Groups appear in the order their keys are
/// first seen, and rows whose key is `null` form a group of their own.
final class GroupBy {
  final DataFrame _frame;
  final List<String> _keys;

  GroupBy._(this._frame, this._keys);

  /// Returns one row per group with the key columns and the results of
  /// the aggregations built by [build].
  ///
  /// ```dart
  /// frame.groupBy(['city']).agg((g) => [g.count(), g.mean('age')]);
  /// ```
  DataFrame agg(List<Agg> Function(Aggs g) build) {
    final aggs = build(const Aggs._());
    if (aggs.isEmpty) {
      throw ArgumentError.value(aggs, 'build', 'Return at least one agg');
    }
    for (final agg in aggs) {
      if (agg.column case final column?) _frame._requireColumn(column);
    }

    final frame = _frame.._flushPending();
    final groups = _GroupIndex(_keys.length);
    final accumulators = [for (final agg in aggs) agg._accumulator()];

    for (final chunk in frame._chunks) {
      final ids = groups.assign(chunk, _keys, frame._deleted[chunk.id]);
      for (var i = 0; i < aggs.length; i++) {
        final column = aggs[i].column;
        accumulators[i].add(
          column == null ? null : chunk.columns[column]!,
          ids,
          groups.length,
        );
      }
    }

    final output = <String, List<Object?>>{
      for (var k = 0; k < _keys.length; k++) _keys[k]: groups.keyColumn(k),
    };
    for (var i = 0; i < aggs.length; i++) {
      final name = aggs[i].name;
      if (output.containsKey(name)) {
        throw ArgumentError.value(name, 'as', 'Duplicate output column');
      }
      output[name] = accumulators[i].finish(groups.length);
    }
    return DataFrame.fromColumns(output, chunkSize: frame._chunkSize);
  }

  /// Returns one row per group with the key columns and a `count` column.
  DataFrame count() => agg((g) => [g.count()]);
}

/// Builds aggregations inside a [GroupBy.agg] callback.
///
/// Each method names its output `column_op` (for example `age_mean`) unless
/// [as] is given. Null values are skipped.
final class Aggs {
  const Aggs._();

  /// The number of rows in each group.
  Agg count({String as = 'count'}) => Agg._(_AggOp.count, null, as);

  /// The sum of [column] in each group.
  Agg sum(String column, {String? as}) => Agg._(_AggOp.sum, column, as);

  /// The mean of [column] in each group, or `null` when it has no values.
  Agg mean(String column, {String? as}) => Agg._(_AggOp.mean, column, as);

  /// The smallest value of [column] in each group.
  Agg min(String column, {String? as}) => Agg._(_AggOp.min, column, as);

  /// The largest value of [column] in each group.
  Agg max(String column, {String? as}) => Agg._(_AggOp.max, column, as);

  /// The first value of [column] seen in each group.
  Agg first(String column, {String? as}) => Agg._(_AggOp.first, column, as);

  /// The last value of [column] seen in each group.
  Agg last(String column, {String? as}) => Agg._(_AggOp.last, column, as);
}

enum _AggOp { count, sum, mean, min, max, first, last }

/// One aggregation requested from [Aggs].
final class Agg {
  final _AggOp _op;

  /// The input column, or `null` for [Aggs.count].
  final String? column;

  /// The output column name.
  final String name;

  Agg._(this._op, this.column, String? as)
      : name = as ?? '${column}_${_op.name}';

  _Accumulator _accumulator() => switch (_op) {
        _AggOp.count => _CountAccumulator(),
        _AggOp.sum => _SumAccumulator(column!, mean: false),
        _AggOp.mean => _SumAccumulator(column!, mean: true),
        _AggOp.min => _ExtremeAccumulator(column!, keepSmaller: true),
        _AggOp.max => _ExtremeAccumulator(column!, keepSmaller: false),
        _AggOp.first => _PickAccumulator(keepFirst: true),
        _AggOp.last => _PickAccumulator(keepFirst: false),
      };
}

/// Maps key values to dense group ids in first-seen order.
final class _GroupIndex {
  final int keyCount;
  final Map<Object?, int> _ids = {};
  final List<List<Object?>> _keyValues;

  _GroupIndex(this.keyCount)
      : _keyValues = [for (var k = 0; k < keyCount; k++) <Object?>[]];

  int get length => _keyValues.first.length;

  List<Object?> keyColumn(int k) => _keyValues[k];

  /// Returns the group id of each physical row in [chunk]; -1 when deleted.
  Int32List assign(_FrameSegment chunk, List<String> keys, Bitset? deleted) {
    final columns = [for (final key in keys) chunk.columns[key]!];
    final ids = Int32List(chunk.length);
    for (var row = 0; row < chunk.length; row++) {
      if (deleted != null && deleted[row]) {
        ids[row] = -1;
        continue;
      }
      final Object? key = keyCount == 1
          ? columns.first.valueAt(row)
          : _CompositeKey([for (final c in columns) c.valueAt(row)]);
      ids[row] = _ids.putIfAbsent(key, () {
        for (var k = 0; k < keyCount; k++) {
          _keyValues[k].add(columns[k].valueAt(row));
        }
        return _ids.length;
      });
    }
    return ids;
  }
}

/// A multi-column group key with value equality.
final class _CompositeKey {
  final List<Object?> values;
  final int _hash;

  _CompositeKey(this.values) : _hash = Object.hashAll(values);

  @override
  int get hashCode => _hash;

  @override
  bool operator ==(Object other) {
    if (other is! _CompositeKey || other.values.length != values.length) {
      return false;
    }
    for (var i = 0; i < values.length; i++) {
      if (other.values[i] != values[i]) return false;
    }
    return true;
  }
}

sealed class _Accumulator {
  /// Folds the rows of [values] into their groups; [ids] of -1 are skipped.
  void add(Series<dynamic>? values, Int32List ids, int groupCount);

  /// Returns one result per group.
  List<Object?> finish(int groupCount);
}

final class _CountAccumulator extends _Accumulator {
  final List<int> _counts = [];

  @override
  void add(Series<dynamic>? values, Int32List ids, int groupCount) {
    while (_counts.length < groupCount) {
      _counts.add(0);
    }
    for (final id in ids) {
      if (id >= 0) _counts[id]++;
    }
  }

  @override
  List<Object?> finish(int groupCount) => _counts;
}

final class _SumAccumulator extends _Accumulator {
  final String column;
  final bool mean;
  final List<num> _sums = [];
  final List<int> _counts = [];

  _SumAccumulator(this.column, {required this.mean});

  @override
  void add(Series<dynamic>? values, Int32List ids, int groupCount) {
    while (_sums.length < groupCount) {
      _sums.add(0);
      _counts.add(0);
    }
    final series = values!;
    final nulls = series.nulls;
    switch (series) {
      // typed fast paths read the buffer directly
      case IntSeries(:final tensor):
        final data = tensor.buffer;
        final offset = tensor.offset;
        for (var row = 0; row < ids.length; row++) {
          final id = ids[row];
          if (id < 0 || (nulls != null && nulls[row])) continue;
          _sums[id] += data[offset + row];
          _counts[id]++;
        }
      case FloatSeries(:final tensor):
        final data = tensor.buffer;
        final offset = tensor.offset;
        for (var row = 0; row < ids.length; row++) {
          final id = ids[row];
          if (id < 0 || (nulls != null && nulls[row])) continue;
          _sums[id] += data[offset + row];
          _counts[id]++;
        }
      default:
        for (var row = 0; row < ids.length; row++) {
          final id = ids[row];
          final value = series.valueAt(row);
          if (id < 0 || value == null) continue;
          if (value is! num) {
            throw ArgumentError.value(
                column, 'column', 'Column is not numeric');
          }
          _sums[id] += value;
          _counts[id]++;
        }
    }
  }

  @override
  List<Object?> finish(int groupCount) {
    if (!mean) return _sums;
    return [
      for (var id = 0; id < groupCount; id++)
        _counts[id] == 0 ? null : _sums[id] / _counts[id],
    ];
  }
}

final class _ExtremeAccumulator extends _Accumulator {
  final String column;
  final bool keepSmaller;
  final List<Comparable<Object?>?> _best = [];

  _ExtremeAccumulator(this.column, {required this.keepSmaller});

  @override
  void add(Series<dynamic>? values, Int32List ids, int groupCount) {
    while (_best.length < groupCount) {
      _best.add(null);
    }
    final series = values!;
    for (var row = 0; row < ids.length; row++) {
      final id = ids[row];
      final value = series.valueAt(row);
      if (id < 0 || value == null) continue;
      if (value is! Comparable<Object?>) {
        throw ArgumentError.value(
            column, 'column', 'Column values cannot be ordered');
      }
      final best = _best[id];
      if (best == null) {
        _best[id] = value;
        continue;
      }
      final order = value.compareTo(best);
      if (keepSmaller ? order < 0 : order > 0) _best[id] = value;
    }
  }

  @override
  List<Object?> finish(int groupCount) => _best;
}

final class _PickAccumulator extends _Accumulator {
  final bool keepFirst;
  final List<Object?> _picked = [];

  _PickAccumulator({required this.keepFirst});

  @override
  void add(Series<dynamic>? values, Int32List ids, int groupCount) {
    while (_picked.length < groupCount) {
      _picked.add(null);
    }
    final series = values!;
    for (var row = 0; row < ids.length; row++) {
      final id = ids[row];
      final value = series.valueAt(row);
      if (id < 0 || value == null) continue;
      if (!keepFirst || _picked[id] == null) _picked[id] = value;
    }
  }

  @override
  List<Object?> finish(int groupCount) => _picked;
}
