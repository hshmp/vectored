part of 'data_frame.dart';

/// Builds a [Condition] from the columns handed to it as [c].
///
/// Used by [DataFrame.filter], [DataFrame.view] and [DataFrame.mask]:
///
/// ```dart
/// frame.filter((c) => c('age').gt(30) & c('name').contains('a'));
/// ```
typedef Where = Condition Function(Columns c);

/// Looks up columns by name inside a [Where] callback.
final class Columns {
  const Columns._();

  /// Returns a reference to the column called [name].
  Col call(String name) => Col._(name);
}

/// A column reference whose methods build row conditions.
///
/// Every condition except [isNull] skips rows that hold no value.
final class Col {
  /// The referenced column name.
  final String name;

  const Col._(this.name);

  // text

  /// Rows whose text contains [pattern].
  ///
  /// Pass a [String] for a fast literal search or a [RegExp] for a pattern,
  /// just like [String.contains].
  Condition contains(Pattern pattern) =>
      _TextCondition(name, _TextOp.contains, pattern);

  /// Rows whose text starts with [prefix].
  Condition startsWith(String prefix) =>
      _TextCondition(name, _TextOp.startsWith, prefix);

  /// Rows whose text ends with [suffix].
  Condition endsWith(String suffix) =>
      _TextCondition(name, _TextOp.endsWith, suffix);

  // equality

  /// Rows equal to [value]; `eq(null)` is the same as [isNull].
  Condition eq(Object? value) =>
      value == null ? isNull : _EqualCondition(name, value);

  /// Rows that hold a value other than [value].
  Condition neq(Object? value) =>
      value == null ? isNotNull : ~_EqualCondition(name, value) & isNotNull;

  /// Rows whose value is one of [values]; include `null` to keep nulls.
  Condition isIn(Iterable<Object?> values) =>
      _InCondition(name, values.toSet());

  // numbers

  /// Rows greater than [value].
  Condition gt(num value) => _CompareCondition(name, _Compare.gt, value);

  /// Rows greater than or equal to [value].
  Condition gte(num value) => _CompareCondition(name, _Compare.gte, value);

  /// Rows less than [value].
  Condition lt(num value) => _CompareCondition(name, _Compare.lt, value);

  /// Rows less than or equal to [value].
  Condition lte(num value) => _CompareCondition(name, _Compare.lte, value);

  /// Rows from [low] to [high], inclusive.
  Condition between(num low, num high) => gte(low) & lte(high);

  // nulls and flags

  /// Rows that hold no value.
  Condition get isNull => _NullCondition(name);

  /// Rows that hold a value.
  Condition get isNotNull => ~_NullCondition(name);

  /// Rows of a boolean column that are `true`.
  Condition get isTrue => _FlagCondition(name, true);

  /// Rows of a boolean column that are `false`.
  Condition get isFalse => _FlagCondition(name, false);

  // custom

  /// Rows whose value passes [test]; the slow but flexible escape hatch.
  Condition satisfies(bool Function(Object value) test) =>
      _TestCondition(name, test);
}

/// A row selection built inside a [Where] callback.
///
/// Combine conditions with `&` (and), `|` (or) and `~` (not). `~` selects
/// every row the condition did not, including rows without a value.
sealed class Condition {
  const Condition();

  /// Rows selected by both this and [other].
  Condition operator &(Condition other) => _AndCondition(this, other);

  /// Rows selected by either this or [other].
  Condition operator |(Condition other) => _OrCondition(this, other);

  /// Rows not selected by this condition.
  Condition operator ~() => _NotCondition(this);

  /// Throws when a referenced column is missing from [frame].
  void _validate(DataFrame frame);

  /// Returns the physical rows of [chunk] selected by this condition.
  Bitset _evaluate(_FrameSegment chunk);
}

final class _AndCondition extends Condition {
  final Condition left;
  final Condition right;

  const _AndCondition(this.left, this.right);

  @override
  void _validate(DataFrame frame) {
    left._validate(frame);
    right._validate(frame);
  }

  @override
  Bitset _evaluate(_FrameSegment chunk) =>
      left._evaluate(chunk).and(right._evaluate(chunk));
}

final class _OrCondition extends Condition {
  final Condition left;
  final Condition right;

  const _OrCondition(this.left, this.right);

  @override
  void _validate(DataFrame frame) {
    left._validate(frame);
    right._validate(frame);
  }

  @override
  Bitset _evaluate(_FrameSegment chunk) =>
      left._evaluate(chunk).or(right._evaluate(chunk));
}

final class _NotCondition extends Condition {
  final Condition inner;

  const _NotCondition(this.inner);

  @override
  void _validate(DataFrame frame) => inner._validate(frame);

  @override
  Bitset _evaluate(_FrameSegment chunk) => inner._evaluate(chunk).invert();
}

/// A condition on one column's values; null rows never match.
sealed class _ColumnCondition extends Condition {
  final String column;

  const _ColumnCondition(this.column);

  @override
  void _validate(DataFrame frame) => frame._requireColumn(column);

  @override
  Bitset _evaluate(_FrameSegment chunk) {
    final series = chunk.columns[column]!;
    final out = Bitset(series.length);
    _mark(series, out);
    final nulls = series.nulls;
    if (nulls != null) out.andNot(nulls);
    return out;
  }

  /// Sets [out] for rows of [series] that match.
  void _mark(Series<dynamic> series, Bitset out);

  /// Tests each non-null value; used for untyped columns.
  void _markEach(
      Series<dynamic> series, Bitset out, bool Function(Object value) test) {
    for (var row = 0; row < series.length; row++) {
      final value = series.valueAt(row);
      if (value != null && test(value)) out.set(row);
    }
  }

  Never _wrongType(String expected) =>
      throw ArgumentError.value(column, 'column', 'Column is not $expected');
}

enum _TextOp { contains, startsWith, endsWith }

final class _TextCondition extends _ColumnCondition {
  final _TextOp op;
  final Pattern pattern;

  const _TextCondition(super.column, this.op, this.pattern);

  @override
  void _mark(Series<dynamic> series, Bitset out) {
    switch (series) {
      case StringSeries(:final text):
        final pattern = this.pattern;
        switch (op) {
          case _TextOp.contains when pattern is String:
            text.markContains(pattern, out);
          case _TextOp.contains:
            for (var row = 0; row < series.length; row++) {
              if (series[row].contains(pattern)) out.set(row);
            }
          case _TextOp.startsWith:
            text.markStartsWith(pattern as String, out);
          case _TextOp.endsWith:
            text.markEndsWith(pattern as String, out);
        }
      case ObjectSeries():
        _markEach(series, out, (value) {
          if (value is! String) _wrongType('text');
          return _test(value);
        });
      default:
        _wrongType('text');
    }
  }

  bool _test(String value) => switch (op) {
        _TextOp.contains => value.contains(pattern),
        _TextOp.startsWith => value.startsWith(pattern as String),
        _TextOp.endsWith => value.endsWith(pattern as String),
      };
}

enum _Compare { gt, gte, lt, lte }

final class _CompareCondition extends _ColumnCondition {
  final _Compare op;
  final num value;

  const _CompareCondition(super.column, this.op, this.value);

  @override
  void _mark(Series<dynamic> series, Bitset out) {
    switch (series) {
      case IntSeries(:final tensor):
        _markInts(tensor.buffer, tensor.offset, tensor.length, out);
      case FloatSeries(:final tensor):
        _markFloats(tensor.buffer, tensor.offset, tensor.length, out);
      case ObjectSeries():
        _markEach(series, out, (value) {
          if (value is! num) _wrongType('numeric');
          return _test(value, this.value);
        });
      default:
        _wrongType('numeric');
    }
  }

  // tight typed loops; op switch hoisted out of the row loop
  void _markInts(Int32List data, int offset, int length, Bitset out) {
    final target = value;
    switch (op) {
      case _Compare.gt:
        for (var i = 0; i < length; i++) {
          if (data[offset + i] > target) out.set(i);
        }
      case _Compare.gte:
        for (var i = 0; i < length; i++) {
          if (data[offset + i] >= target) out.set(i);
        }
      case _Compare.lt:
        for (var i = 0; i < length; i++) {
          if (data[offset + i] < target) out.set(i);
        }
      case _Compare.lte:
        for (var i = 0; i < length; i++) {
          if (data[offset + i] <= target) out.set(i);
        }
    }
  }

  void _markFloats(Float32List data, int offset, int length, Bitset out) {
    // compare at stored precision so gt(0.1) skips a stored 0.1
    final target = _toFloat32(value);
    switch (op) {
      case _Compare.gt:
        for (var i = 0; i < length; i++) {
          if (data[offset + i] > target) out.set(i);
        }
      case _Compare.gte:
        for (var i = 0; i < length; i++) {
          if (data[offset + i] >= target) out.set(i);
        }
      case _Compare.lt:
        for (var i = 0; i < length; i++) {
          if (data[offset + i] < target) out.set(i);
        }
      case _Compare.lte:
        for (var i = 0; i < length; i++) {
          if (data[offset + i] <= target) out.set(i);
        }
    }
  }

  bool _test(num left, num right) => switch (op) {
        _Compare.gt => left > right,
        _Compare.gte => left >= right,
        _Compare.lt => left < right,
        _Compare.lte => left <= right,
      };
}

final class _EqualCondition extends _ColumnCondition {
  final Object value;

  const _EqualCondition(super.column, this.value);

  @override
  void _mark(Series<dynamic> series, Bitset out) {
    final value = this.value;
    switch (series) {
      case StringSeries(:final text) when value is String:
        text.markEquals(value, out);
      case IntSeries(:final tensor) when value is num:
        final data = tensor.buffer;
        final offset = tensor.offset;
        for (var i = 0; i < tensor.length; i++) {
          if (data[offset + i] == value) out.set(i);
        }
      case FloatSeries(:final tensor) when value is num:
        final data = tensor.buffer;
        final offset = tensor.offset;
        final target = _toFloat32(value);
        for (var i = 0; i < tensor.length; i++) {
          if (data[offset + i] == target) out.set(i);
        }
      case BoolSeries(:final values) when value is bool:
        out.or(values);
        if (!value) out.invert();
      case ObjectSeries():
        _markEach(series, out, (stored) => stored == value);
      default:
      // mismatched types never compare equal
    }
  }
}

final class _InCondition extends _ColumnCondition {
  final Set<Object?> values;

  const _InCondition(super.column, this.values);

  @override
  Bitset _evaluate(_FrameSegment chunk) {
    final out = super._evaluate(chunk);
    if (!values.contains(null)) return out;
    return out.or(_NullCondition(column)._evaluate(chunk));
  }

  @override
  void _mark(Series<dynamic> series, Bitset out) {
    // floats compare at stored precision
    final lookup = series is FloatSeries
        ? {for (final value in values) value is num ? _toFloat32(value) : value}
        : values;
    _markEach(series, out, lookup.contains);
  }
}

final class _NullCondition extends Condition {
  final String column;

  const _NullCondition(this.column);

  @override
  void _validate(DataFrame frame) => frame._requireColumn(column);

  @override
  Bitset _evaluate(_FrameSegment chunk) {
    final series = chunk.columns[column]!;
    final nulls = series.nulls;
    if (nulls != null) return nulls.copy();
    final out = Bitset(series.length);
    if (series is ObjectSeries) {
      for (var row = 0; row < series.length; row++) {
        if (series.isNull(row)) out.set(row);
      }
    }
    return out;
  }
}

final class _FlagCondition extends _ColumnCondition {
  final bool expected;

  const _FlagCondition(super.column, this.expected);

  @override
  void _mark(Series<dynamic> series, Bitset out) {
    switch (series) {
      case BoolSeries(:final values):
        out.or(values);
        if (!expected) out.invert();
      case ObjectSeries():
        _markEach(series, out, (value) {
          if (value is! bool) _wrongType('boolean');
          return value == expected;
        });
      default:
        _wrongType('boolean');
    }
  }
}

final class _TestCondition extends _ColumnCondition {
  final bool Function(Object value) test;

  const _TestCondition(super.column, this.test);

  @override
  void _mark(Series<dynamic> series, Bitset out) =>
      _markEach(series, out, test);
}

final _float32Scratch = Float32List(1);

/// Rounds [value] to the nearest float32, the precision of [FloatSeries].
double _toFloat32(num value) {
  _float32Scratch[0] = value.toDouble();
  return _float32Scratch[0];
}
