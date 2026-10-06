import 'dart:collection';
import 'dart:typed_data';

import 'bitset.dart';
import 'series.dart';

part 'condition.dart';
part 'group_by.dart';

/// A mutable, chunked collection of named [Series] columns.
///
/// Appends are buffered into column series, deletes are logical until
/// [compact], and concatenation links series chunks without copying values.
class DataFrame {
  final List<String> _columns;
  final int _chunkSize;
  final List<_FrameSegment> _chunks;
  late final Map<String, List<Object?>> _pendingColumns = {
    for (final column in _columns) column: <Object?>[],
  };
  int _pendingLength = 0;
  // deleted rows per segment id; absent when a segment has none
  final Map<int, Bitset> _deleted;
  int _deletedCount;
  final Map<String, _NumericStats> _numericStats;
  final Set<String> _uncacheableColumns;
  final Set<String> _nonNumericColumns;
  final bool _aggregateCaching;
  List<({int chunk, int row})>? _visibleRows;

  static int _nextRowId = 0;

  DataFrame._(
    this._columns,
    this._chunkSize,
    this._chunks,
    this._deleted,
    this._numericStats,
    this._uncacheableColumns,
    this._nonNumericColumns,
    this._aggregateCaching,
  ) : _deletedCount =
            _deleted.values.fold(0, (total, rows) => total + rows.count);

  /// Creates an empty frame with the named [columns].
  factory DataFrame.empty(
    List<String> columns, {
    int chunkSize = 1024,
  }) {
    _validateColumns(columns);
    _validateChunkSize(chunkSize);
    return DataFrame._(
      List.unmodifiable(columns),
      chunkSize,
      [],
      {},
      {},
      {},
      {},
      true,
    );
  }

  /// Creates a frame from column-oriented [data].
  ///
  /// Each column is converted to its matching [Series] type. Values are
  /// copied into frame-owned series chunks.
  factory DataFrame.fromColumns(
    Map<String, List<Object?>> data, {
    int chunkSize = 1024,
  }) {
    final columns = data.keys.toList(growable: false);
    _validateColumns(columns);
    _validateChunkSize(chunkSize);

    final rowCounts = data.values.map((values) => values.length).toSet();
    if (rowCounts.length > 1) {
      throw ArgumentError('All columns must have the same length');
    }

    return _fromOwnedColumns(
      columns,
      data,
      chunkSize: chunkSize,
    );
  }

  /// Creates a frame from existing [columns] without copying their series.
  ///
  /// Column names must be unique and all series must have equal lengths.
  /// Mutations to the supplied series are visible through this frame.
  factory DataFrame.fromSeries(
    Iterable<Series<dynamic>> columns, {
    int chunkSize = 1024,
  }) {
    final series = columns.toList(growable: false);
    final names = series.map((column) => column.name).toList(growable: false);
    _validateColumns(names);
    _validateChunkSize(chunkSize);

    final lengths = series.map((column) => column.length).toSet();
    if (lengths.length > 1) {
      throw ArgumentError('All series must have the same length');
    }

    final frame = DataFrame._(
      List.unmodifiable(names),
      chunkSize,
      [],
      {},
      {},
      names.toSet(),
      {
        for (final column in series)
          if (column is! IntSeries && column is! FloatSeries) column.name,
      },
      false,
    );
    if (series.isNotEmpty) {
      frame._chunks.add(
        _FrameSegment(
          _SeriesChunk({for (final column in series) column.name: column}),
          _newSegmentId(),
        ),
      );
    }
    return frame;
  }

  /// The number of rows that have not been deleted.
  int get length => _physicalLength - _deletedCount;

  /// The declared column names in insertion order.
  List<String> get columns => _columns;

  /// The number of stored rows, including rows marked for deletion.
  int get physicalLength =>
      _chunks.fold(0, (total, chunk) => total + chunk.length) + _pendingLength;

  /// The number of stored chunks, including a pending append buffer.
  int get physicalChunkCount => _chunks.length + (_pendingLength == 0 ? 0 : 1);

  int get _physicalLength => physicalLength;

  /// Returns a lazy view of [name] over the frame's live rows.
  DataFrameColumn operator [](String name) {
    _requireColumn(name);
    return DataFrameColumn._(this, name);
  }

  /// Returns the live row at logical index [index].
  Map<String, Object?> rowAt(int index) {
    _flushPending();
    final location = _locationAt(index);
    final chunk = _chunks[location.chunk];
    return Map.unmodifiable({
      for (final column in _columns)
        column: chunk.columns[column]!.valueAt(location.row),
    });
  }

  Object? _valueAt(String column, int index) {
    _flushPending();
    final location = _locationAt(index);
    return _chunks[location.chunk].columns[column]!.valueAt(location.row);
  }

  /// Returns a materialized [Series] containing this column's live values.
  Series<dynamic> toSeries(String name) {
    _requireColumn(name);
    _flushPending();
    return _seriesFromValues(name, [for (final value in this[name]) value]);
  }

  /// Appends [row] to the pending tail buffer.
  ///
  /// [row] must provide every declared column and no others. Values are
  /// converted to series when the buffer reaches [chunkSize] or is read;
  /// cached aggregates are updated from the stored values at that point.
  void appendRow(Map<String, Object?> row) {
    _validateRow(row);
    for (final column in _columns) {
      _pendingColumns[column]!.add(row[column]);
    }
    _pendingLength++;
    _visibleRows = null;
    if (_pendingLength >= _chunkSize) {
      _flushPending();
    }
  }

  /// Appends a batch of [rows] using bounded series chunks.
  void appendRows(Iterable<Map<String, Object?>> rows) {
    for (final row in rows) {
      appendRow(row);
    }
  }

  /// Marks the live row at logical index [index] as deleted.
  ///
  /// Row data remains in its series chunk until [compact] is called.
  void deleteRow(int index) {
    _flushPending();
    final location = _locationForDelete(index);
    final chunk = _chunks[location.chunk];
    (_deleted[chunk.id] ??= Bitset(chunk.length)).set(location.row);
    _deletedCount++;
    for (final column in _columns) {
      final value = chunk.columns[column]!.valueAt(location.row);
      final stats = _numericStats[column];
      if (value is! num || stats == null) continue;

      // extrema lost; row already marked deleted
      if (!stats.remove(value)) {
        _numericStats[column] = _rebuildNumericStats(column);
      }
    }
    _visibleRows = null;
  }

  /// Returns one flag per live row: set where [where] selects the row.
  ///
  /// ```dart
  /// final adults = frame.mask((c) => c('age').gte(18));
  /// adults.count; // how many rows matched
  /// ```
  Bitset mask(Where where) {
    final condition = _prepare(where);
    final output = Bitset(length);
    var logicalRow = 0;
    for (final chunk in _chunks) {
      final selected = condition._evaluate(chunk);
      final deleted = _deleted[chunk.id];
      if (deleted == null) {
        for (final row in selected.toIndices()) {
          output.set(logicalRow + row);
        }
        logicalRow += chunk.length;
        continue;
      }
      for (var row = 0; row < chunk.length; row++) {
        if (deleted[row]) continue;
        if (selected[row]) output.set(logicalRow);
        logicalRow++;
      }
    }
    return output;
  }

  /// Returns a new frame with only the rows selected by [where].
  ///
  /// ```dart
  /// frame.filter((c) => c('name').contains('lin'));
  /// frame.filter((c) => c('name').contains(RegExp(r'^a')) | c('age').gt(60));
  /// ```
  DataFrame filter(Where where) {
    final condition = _prepare(where);
    final selectedChunks = <_FrameSegment>[];
    for (final chunk in _chunks) {
      _appendSelectedChunk(
        chunk,
        _liveRows(condition, chunk).toIndices(),
        selectedChunks,
      );
    }
    return _fromSelectedChunks(selectedChunks);
  }

  /// Returns a lightweight view of the rows selected by [where].
  ///
  /// Nothing is copied: the view points at this frame's rows, which makes it
  /// faster than [filter] when you only need to read or total the results.
  DataFrameView view(Where where) {
    final condition = _prepare(where);
    final rows = <_RowPointer>[];
    for (final chunk in _chunks) {
      for (final row in _liveRows(condition, chunk).toIndices()) {
        rows.add(_RowPointer(chunk, row));
      }
    }
    return DataFrameView._(this, rows);
  }

  /// Groups rows by the values of the [keys] columns.
  ///
  /// ```dart
  /// frame.groupBy(['city']).agg((g) => [g.count(), g.mean('age')]);
  /// ```
  GroupBy groupBy(List<String> keys) {
    if (keys.isEmpty) {
      throw ArgumentError.value(keys, 'keys', 'Group by at least one column');
    }
    keys.forEach(_requireColumn);
    return GroupBy._(this, List.unmodifiable(keys));
  }

  Condition _prepare(Where where) {
    _flushPending();
    final condition = where(const Columns._());
    condition._validate(this);
    return condition;
  }

  /// Returns the live rows of [chunk] selected by [condition].
  Bitset _liveRows(Condition condition, _FrameSegment chunk) {
    final selected = condition._evaluate(chunk);
    final deleted = _deleted[chunk.id];
    return deleted == null ? selected : selected.andNot(deleted);
  }

  void _appendSelectedChunk(
    _FrameSegment source,
    List<int> selectedRows,
    List<_FrameSegment> output,
  ) {
    if (selectedRows.isEmpty) return;
    for (var start = 0; start < selectedRows.length; start += _chunkSize) {
      final end = (start + _chunkSize).clamp(0, selectedRows.length).toInt();
      final indices = selectedRows.sublist(start, end);
      final columns = <String, Series<dynamic>>{
        for (final column in _columns)
          column: _takeSeries(source.columns[column]!, indices),
      };
      output.add(_FrameSegment(_SeriesChunk(columns), _newSegmentId()));
    }
  }

  DataFrame _fromSelectedChunks(List<_FrameSegment> chunks) {
    final stats = <String, _NumericStats>{};
    final uncacheable = <String>{};
    final nonNumeric = <String>{};
    for (final column in _columns) {
      // no selected rows; fall back to the source column's type
      final numeric = chunks.isEmpty
          ? !_nonNumericColumns.contains(column) &&
              (_chunks.isEmpty ||
                  _isNumericSeries(_chunks.first.columns[column]!))
          : chunks.every((chunk) => _isNumericSeries(chunk.columns[column]!));
      if (!numeric) {
        uncacheable.add(column);
        nonNumeric.add(column);
        continue;
      }

      final columnStats = _NumericStats(trackFrequencies: false);
      final cacheable = chunks.every(
        (chunk) => _addSeriesToStats(columnStats, chunk.columns[column]!),
      );
      if (cacheable) {
        stats[column] = columnStats;
      } else {
        uncacheable.add(column);
      }
    }
    return DataFrame._(
      _columns,
      _chunkSize,
      chunks,
      {},
      stats,
      uncacheable,
      nonNumeric,
      true,
    );
  }

  static Series<dynamic> _takeSeries(
      Series<dynamic> series, List<int> indices) {
    return switch (series) {
      IntSeries() => series.take(indices),
      FloatSeries() => series.take(indices),
      StringSeries() => series.take(indices),
      BoolSeries() => series.take(indices),
      ObjectSeries() => series.take(indices),
    };
  }

  /// Returns the sum of the numeric [column] over live rows.
  ///
  /// Owned frames return the running aggregate; shared external series are
  /// scanned so mutations remain visible.
  num sum(String column) {
    _requireColumn(column);
    _flushPending();
    final cached = _aggregateCaching ? _numericStats[column] : null;
    if (cached != null) return cached.sum;
    if (_aggregateCaching && _nonNumericColumns.contains(column)) {
      throw ArgumentError.value(column, 'column', 'Column is not numeric');
    }
    num total = 0;

    for (final chunk in _chunks) {
      final values = chunk.columns[column]!;
      final hasDeletedRows = _hasDeletedRows(chunk);
      if (!hasDeletedRows && values is IntSeries) {
        total += values.sum();
      } else if (!hasDeletedRows && values is FloatSeries) {
        total += values.sum();
      } else {
        final deleted = _deleted[chunk.id];
        for (var row = 0; row < chunk.length; row++) {
          if (deleted != null && deleted[row]) continue;
          final value = values.valueAt(row);
          if (value == null) continue;
          if (value is! num) {
            throw ArgumentError.value(
                column, 'column', 'Column is not numeric');
          }
          total += value;
        }
      }
    }
    return total;
  }

  /// Returns the number of live non-null values in numeric [column].
  ///
  /// Returns in O(1) for owned frames. Shared external series require a scan.
  int count(String column) => _numericStatsFor(column).count;

  /// Returns the mean of numeric [column], or `null` when it has no live rows.
  ///
  /// Owned frames use cached sum and count; shared Series are scanned.
  double? mean(String column) {
    final stats = _numericStatsFor(column);
    if (stats.count == 0) return null;
    return stats.sum / stats.count;
  }

  /// Returns the minimum live value, or `null` when [column] is empty.
  ///
  /// Ordered value frequencies maintain the extrema across inserts/deletes.
  num? min(String column) => _numericStatsFor(column).minimum;

  /// Returns the maximum live value, or `null` when [column] is empty.
  num? max(String column) => _numericStatsFor(column).maximum;

  /// Combines this frame and [other] by linking their series chunks.
  ///
  /// Both frames must have the same column names. Chunks are shared without
  /// copying values; future appends and deletes remain local to each frame.
  DataFrame concat(DataFrame other) {
    _flushPending();
    other._flushPending();
    if (!_sameSchema(other)) {
      throw ArgumentError('Frames must have the same column names');
    }
    final (leftChunks, leftDeletes) = _copySegmentsForConcat(this);
    final (rightChunks, rightDeletes) = _copySegmentsForConcat(other);
    final aggregateCaching = _aggregateCaching && other._aggregateCaching;
    final numericStats = <String, _NumericStats>{};
    if (aggregateCaching) {
      for (final column in _columns) {
        if (_uncacheableColumns.contains(column) ||
            other._uncacheableColumns.contains(column)) {
          continue;
        }
        final left = _numericStats[column];
        final right = other._numericStats[column];
        if (left != null || right != null) {
          numericStats[column] = (left ?? _NumericStats()).copy()
            ..merge(right ?? _NumericStats());
        }
      }
    }
    return DataFrame._(
      _columns,
      _chunkSize,
      [...leftChunks, ...rightChunks],
      {...leftDeletes, ...rightDeletes},
      numericStats,
      {..._uncacheableColumns, ...other._uncacheableColumns},
      {..._nonNumericColumns, ...other._nonNumericColumns},
      aggregateCaching,
    );
  }

  /// Rebuilds the frame from live rows and clears delete markers.
  ///
  /// This copies selected values into fresh series chunks and is useful when
  /// deleted rows or many small chunks consume too much memory.
  void compact() {
    _flushPending();
    if (_deletedCount == 0 && _chunks.length <= 1) return;

    final values = [for (final _ in _columns) <Object?>[]];
    for (final chunk in _chunks) {
      final sourceColumns = [
        for (final column in _columns) chunk.columns[column]!,
      ];
      final deleted = _deleted[chunk.id];
      for (var row = 0; row < chunk.length; row++) {
        if (deleted != null && deleted[row]) continue;
        for (var index = 0; index < _columns.length; index++) {
          values[index].add(sourceColumns[index].valueAt(row));
        }
      }
    }
    final compacted = _fromOwnedColumns(
      _columns,
      _columnsToMap(values),
      chunkSize: _chunkSize,
      numericStats: {
        for (final entry in _numericStats.entries)
          entry.key: entry.value.copy(),
      },
      uncacheableColumns: _uncacheableColumns,
      nonNumericColumns: _nonNumericColumns,
    );
    _chunks
      ..clear()
      ..addAll(compacted._chunks);
    _deleted.clear();
    _deletedCount = 0;
    _numericStats
      ..clear()
      ..addAll(compacted._numericStats);
    _uncacheableColumns
      ..clear()
      ..addAll(compacted._uncacheableColumns);
    _nonNumericColumns
      ..clear()
      ..addAll(compacted._nonNumericColumns);
    _visibleRows = null;
  }

  /// Creates a frame from row-oriented [rows].
  factory DataFrame.fromRows(
    Iterable<Map<String, Object?>> rows, {
    required List<String> columns,
    int chunkSize = 1024,
  }) {
    final frame = DataFrame.empty(columns, chunkSize: chunkSize);
    frame.appendRows(rows);
    frame._flushPending();
    return frame;
  }

  void _flushPending() {
    if (_pendingLength == 0) return;
    final series = <String, Series<dynamic>>{
      for (final column in _columns)
        column: _seriesFromValues(column, _pendingColumns[column]!),
    };
    _chunks.add(_FrameSegment(_SeriesChunk(series), _newSegmentId()));
    _recordChunkStats(series);
    for (final values in _pendingColumns.values) {
      values.clear();
    }
    _pendingLength = 0;
    _visibleRows = null;
  }

  /// Folds the stored values of a new chunk's [series] into the cached
  /// aggregates, marking columns uncacheable or non-numeric as needed.
  void _recordChunkStats(Map<String, Series<dynamic>> series) {
    for (final column in _columns) {
      final values = series[column]!;
      if (!_isNumericSeries(values)) {
        _nonNumericColumns.add(column);
        _uncacheableColumns.add(column);
        _numericStats.remove(column);
        continue;
      }
      if (!_aggregateCaching || _uncacheableColumns.contains(column)) continue;

      final stats = _numericStats[column] ??= _NumericStats();
      if (!_addSeriesToStats(stats, values)) {
        _uncacheableColumns.add(column);
        _numericStats.remove(column);
      }
    }
  }

  /// Returns whether every value in [series] is a number.
  static bool _isNumericSeries(Series<dynamic> series) => switch (series) {
        IntSeries() || FloatSeries() => true,
        ObjectSeries(:final data) =>
          data.every((value) => value == null || value is num),
        StringSeries() || BoolSeries() => false,
      };

  /// Adds every non-null stored value of numeric [series] to [stats].
  ///
  /// Returns false when a value is not finite and cannot be cached.
  static bool _addSeriesToStats(_NumericStats stats, Series<dynamic> series) {
    for (var row = 0; row < series.length; row++) {
      final raw = series.valueAt(row);
      if (raw == null) continue;
      final value = _cacheableNumber(raw);
      if (value == null) return false;
      stats.add(value);
    }
    return true;
  }

  Map<String, List<Object?>> _columnsToMap(List<List<Object?>> values) => {
        for (var index = 0; index < _columns.length; index++)
          _columns[index]: values[index],
      };

  List<({int chunk, int row})> _buildVisibleRows() {
    final rows = <({int chunk, int row})>[];
    for (var chunkIndex = 0; chunkIndex < _chunks.length; chunkIndex++) {
      final chunk = _chunks[chunkIndex];
      final deleted = _deleted[chunk.id];
      for (var rowIndex = 0; rowIndex < chunk.length; rowIndex++) {
        if (deleted == null || !deleted[rowIndex]) {
          rows.add((chunk: chunkIndex, row: rowIndex));
        }
      }
    }
    _visibleRows = rows;
    return rows;
  }

  ({int chunk, int row}) _locationAt(int index) {
    if (index < 0 || index >= length) {
      throw RangeError.index(index, this, 'index');
    }
    return (_visibleRows ?? _buildVisibleRows())[index];
  }

  ({int chunk, int row}) _locationForDelete(int index) {
    if (index < 0 || index >= length) {
      throw RangeError.index(index, this, 'index');
    }

    var remaining = index;
    for (var chunkIndex = 0; chunkIndex < _chunks.length; chunkIndex++) {
      final chunk = _chunks[chunkIndex];
      final deleted = _deleted[chunk.id];
      final liveCount = chunk.length - (deleted?.count ?? 0);
      if (remaining >= liveCount) {
        remaining -= liveCount;
        continue;
      }
      for (var row = 0; row < chunk.length; row++) {
        if (deleted != null && deleted[row]) continue;
        if (remaining-- == 0) return (chunk: chunkIndex, row: row);
      }
    }
    throw StateError('Live row index could not be resolved');
  }

  _NumericStats _numericStatsFor(String column) {
    _requireColumn(column);
    _flushPending();
    if (_aggregateCaching) {
      final cached = _numericStats[column];
      if (cached != null) return cached;
    }

    final stats = _NumericStats();
    for (final chunk in _chunks) {
      final values = chunk.columns[column]!;
      final deleted = _deleted[chunk.id];
      for (var row = 0; row < chunk.length; row++) {
        if (deleted != null && deleted[row]) continue;
        final value = values.valueAt(row);
        if (value == null) continue;
        if (value is! num) {
          throw ArgumentError.value(column, 'column', 'Column is not numeric');
        }
        stats.add(value);
      }
    }
    return stats;
  }

  _NumericStats _rebuildNumericStats(String column) {
    final stats = _NumericStats(trackFrequencies: false);
    for (final segment in _chunks) {
      final values = segment.columns[column]!;
      final deleted = _deleted[segment.id];
      for (var row = 0; row < segment.length; row++) {
        if (deleted != null && deleted[row]) continue;
        final value = values.valueAt(row);
        if (value != null) stats.add(value as num);
      }
    }
    return stats;
  }

  void _validateRow(Map<String, Object?> row) {
    final actual = row.keys.toSet();
    final expected = _columns.toSet();
    if (actual.length != row.length ||
        actual.length != expected.length ||
        !actual.containsAll(expected)) {
      throw ArgumentError.value(
        row.keys,
        'row',
        'Row keys must match the frame columns',
      );
    }
  }

  void _requireColumn(String name) {
    if (!_columns.contains(name)) {
      throw ArgumentError.value(name, 'name', 'Unknown column');
    }
  }

  bool _hasDeletedRows(_FrameSegment chunk) => _deleted.containsKey(chunk.id);

  bool _sameSchema(DataFrame other) =>
      _columns.length == other._columns.length &&
      _columns.toSet().containsAll(other._columns);

  static int _newSegmentId() => _nextRowId++;

  static (List<_FrameSegment>, Map<int, Bitset>) _copySegmentsForConcat(
      DataFrame frame) {
    final newSegments = <_FrameSegment>[];
    final deletes = <int, Bitset>{};
    for (final segment in frame._chunks) {
      final newId = _newSegmentId();
      newSegments.add(_FrameSegment(segment.data, newId));
      final deleted = frame._deleted[segment.id];
      if (deleted != null) deletes[newId] = deleted.copy();
    }
    return (newSegments, deletes);
  }

  /// Picks the most compact series type that stores [values] without
  /// silently changing them; `null` entries become null flags.
  ///
  /// Ints outside int32 range, or ints too large for float32 in a mixed
  /// numeric column, fall back to an exact [ObjectSeries], as do chunks that
  /// are entirely `null`.
  static Series<dynamic> _seriesFromValues(String name, List<Object?> values) {
    final present = values.where((value) => value != null);
    if (values.isNotEmpty && present.isEmpty) {
      return Series.fromObjects(name, values);
    }
    if (present.every((value) => value is int)) {
      if (present.every((value) => _fitsInt32(value as int))) {
        return Series.fromInts(name, values.cast<int?>());
      }
      return Series.fromObjects(name, values);
    }
    if (present.every((value) => value is num)) {
      if (present.any((value) => value is int && !_fitsFloat32(value))) {
        return Series.fromObjects(name, values);
      }
      return Series.fromFloats(
        name,
        [for (final value in values) (value as num?)?.toDouble()],
      );
    }
    if (present.every((value) => value is String)) {
      return Series.fromStrings(name, values.cast<String?>());
    }
    if (present.every((value) => value is bool)) {
      return Series.fromBools(name, values.cast<bool?>());
    }
    return Series.fromObjects(name, values);
  }

  static DataFrame _fromOwnedColumns(
    List<String> columns,
    Map<String, List<Object?>> data, {
    required int chunkSize,
    Map<String, _NumericStats>? numericStats,
    Set<String>? uncacheableColumns,
    Set<String>? nonNumericColumns,
  }) {
    final frame = DataFrame.empty(columns, chunkSize: chunkSize);
    if (numericStats == null) {
      // empty columns; element type is the only hint
      for (final column in columns) {
        final values = data[column]!;
        if (values.isNotEmpty) continue;
        if (values is List<num>) {
          frame._numericStats[column] = _NumericStats();
        } else if (values is List<String> || values is List<bool>) {
          frame._nonNumericColumns.add(column);
          frame._uncacheableColumns.add(column);
        }
      }
    } else {
      frame._numericStats.addAll(numericStats);
      frame._uncacheableColumns.addAll(uncacheableColumns ?? const {});
      frame._nonNumericColumns.addAll(nonNumericColumns ?? const {});
      for (final column in columns) {
        if (!frame._numericStats.containsKey(column) &&
            !frame._uncacheableColumns.contains(column)) {
          frame._numericStats[column] = _NumericStats();
        }
      }
    }

    final rowCount = columns.isEmpty ? 0 : data[columns.first]!.length;
    final wholeNumericColumns = <String, Series<dynamic>>{};
    for (final column in columns) {
      final values = data[column]!;
      if (values.every((value) => value is num)) {
        wholeNumericColumns[column] = _seriesFromValues(column, values);
      }
    }
    for (var start = 0; start < rowCount; start += chunkSize) {
      final end = (start + chunkSize).clamp(0, rowCount).toInt();
      final series = <String, Series<dynamic>>{
        for (final column in columns)
          column: switch (wholeNumericColumns[column]) {
            IntSeries(:final tensor) =>
              IntSeries(column, tensor.slice(start, end)),
            FloatSeries(:final tensor) =>
              FloatSeries(column, tensor.slice(start, end)),
            _ => _seriesFromValues(column, data[column]!.sublist(start, end)),
          },
      };
      frame._chunks.add(_FrameSegment(_SeriesChunk(series), _newSegmentId()));
      if (numericStats == null) frame._recordChunkStats(series);
    }
    return frame;
  }

  static bool _fitsInt32(int value) =>
      value >= -0x80000000 && value <= 0x7fffffff;

  // float32 mantissa holds ints exactly up to 2^24
  static bool _fitsFloat32(int value) => value.abs() <= 0x1000000;

  static num? _cacheableNumber(Object? value) {
    if (value is int) return value;
    if (value is double && value.isFinite) return value;
    return null;
  }

  static void _validateColumns(List<String> columns) {
    if (columns.any((column) => column.isEmpty)) {
      throw ArgumentError('Column names must not be empty');
    }
    if (columns.toSet().length != columns.length) {
      throw ArgumentError('Column names must be unique');
    }
  }

  static void _validateChunkSize(int chunkSize) {
    if (chunkSize <= 0) {
      throw ArgumentError.value(chunkSize, 'chunkSize', 'Must be positive');
    }
  }
}

/// A lazy, indexed view of a [DataFrame] column.
class DataFrameColumn extends IterableBase<Object?> {
  final DataFrame _frame;
  final String name;

  DataFrameColumn._(this._frame, this.name);

  @override
  int get length => _frame.length;

  /// Returns the value at live row [index].
  Object? operator [](int index) => _frame._valueAt(name, index);

  @override
  Iterator<Object?> get iterator => _iterate().iterator;

  Iterable<Object?> _iterate() sync* {
    for (var index = 0; index < length; index++) {
      yield this[index];
    }
  }
}

final class _SeriesChunk {
  final Map<String, Series<dynamic>> columns;

  _SeriesChunk(this.columns);

  int get length => columns.values.first.length;
}

final class _FrameSegment {
  final _SeriesChunk data;
  final int id;

  _FrameSegment(this.data, this.id);

  Map<String, Series<dynamic>> get columns => data.columns;
  int get length => data.length;
}

final class _RowPointer {
  final _FrameSegment segment;
  final int row;

  const _RowPointer(this.segment, this.row);
}

/// A non-materialized snapshot of selected DataFrame row pointers.
///
/// Column reads and aggregation operate on the original Series storage.
class DataFrameView extends IterableBase<Map<String, Object?>> {
  final DataFrame _frame;
  final List<_RowPointer> _rows;

  DataFrameView._(this._frame, List<_RowPointer> rows)
      : _rows = List.unmodifiable(rows);

  /// The columns retained from the source frame.
  List<String> get columns => _frame.columns;

  @override
  int get length => _rows.length;

  /// Returns a lazy view of [name] over selected rows.
  DataFrameViewColumn operator [](String name) {
    _frame._requireColumn(name);
    return DataFrameViewColumn._(this, name);
  }

  /// Returns selected row [index] as a named map.
  Map<String, Object?> rowAt(int index) {
    final pointer = _rows[index];
    return Map.unmodifiable({
      for (final column in columns)
        column: pointer.segment.columns[column]!.valueAt(pointer.row),
    });
  }

  /// Sums the non-null values of numeric [column] in the selected rows.
  num sum(String column) => _numeric(column).$1;

  /// Returns the mean of numeric [column], or `null` when it has no values.
  double? mean(String column) {
    final (total, count) = _numeric(column);
    return count == 0 ? null : total / count;
  }

  (num, int) _numeric(String column) {
    _frame._requireColumn(column);
    num total = 0;
    var count = 0;
    for (final pointer in _rows) {
      final value = pointer.segment.columns[column]!.valueAt(pointer.row);
      if (value == null) continue;
      if (value is! num) {
        throw ArgumentError.value(column, 'column', 'Column is not numeric');
      }
      total += value;
      count++;
    }
    return (total, count);
  }

  /// Narrows this view to the rows also selected by [where].
  DataFrameView filter(Where where) {
    final condition = where(const Columns._()).._validate(_frame);
    final selected = <_FrameSegment, Bitset>{};
    return DataFrameView._(_frame, [
      for (final pointer in _rows)
        if (selected.putIfAbsent(pointer.segment,
            () => condition._evaluate(pointer.segment))[pointer.row])
          pointer,
    ]);
  }

  /// Materializes this view into independent typed Series chunks.
  DataFrame materialize() {
    final chunks = <_FrameSegment>[];
    _FrameSegment? segment;
    var selectedRows = <int>[];
    for (final pointer in _rows) {
      if (segment != null &&
          !identical(pointer.segment, segment) &&
          selectedRows.isNotEmpty) {
        _frame._appendSelectedChunk(segment, selectedRows, chunks);
        selectedRows = <int>[];
      }
      segment = pointer.segment;
      selectedRows.add(pointer.row);
    }
    if (segment != null && selectedRows.isNotEmpty) {
      _frame._appendSelectedChunk(segment, selectedRows, chunks);
    }
    return _frame._fromSelectedChunks(chunks);
  }

  @override
  Iterator<Map<String, Object?>> get iterator => _iterate().iterator;

  Iterable<Map<String, Object?>> _iterate() sync* {
    for (var index = 0; index < length; index++) {
      yield rowAt(index);
    }
  }
}

/// A lazy column view over [DataFrameView] row pointers.
class DataFrameViewColumn extends IterableBase<Object?> {
  final DataFrameView _view;
  final String name;

  DataFrameViewColumn._(this._view, this.name);

  @override
  int get length => _view.length;

  Object? operator [](int index) {
    final pointer = _view._rows[index];
    return pointer.segment.columns[name]!.valueAt(pointer.row);
  }

  @override
  Iterator<Object?> get iterator => _iterate().iterator;

  Iterable<Object?> _iterate() sync* {
    for (var index = 0; index < length; index++) {
      yield this[index];
    }
  }
}

final class _NumericStats {
  final SplayTreeMap<num, int> _frequencies = SplayTreeMap<num, int>();
  bool _tracksFrequencies;
  num sum = 0;
  int count = 0;
  num? _minimum;
  num? _maximum;

  _NumericStats({bool trackFrequencies = true})
      : _tracksFrequencies = trackFrequencies;

  bool get tracksFrequencies => _tracksFrequencies;
  num? get minimum => _minimum;
  num? get maximum => _maximum;

  void add(num value) {
    if (count == 0 || value < _minimum!) _minimum = value;
    if (count == 0 || value > _maximum!) _maximum = value;
    sum += value;
    count++;
    if (_tracksFrequencies) {
      _frequencies.update(
        value,
        (frequency) => frequency + 1,
        ifAbsent: () => 1,
      );
    }
  }

  /// Removes [value] and returns whether [minimum] and [maximum] are still
  /// exact; callers rebuild the stats from live rows when this is false.
  bool remove(num value) {
    sum -= value;
    count--;
    if (!_tracksFrequencies) {
      return value != _minimum && value != _maximum;
    }
    final frequency = _frequencies[value];
    if (frequency == null) return false;
    if (frequency == 1) {
      _frequencies.remove(value);
      if (_frequencies.isEmpty) {
        _minimum = null;
        _maximum = null;
      } else {
        if (value == _minimum) _minimum = _frequencies.firstKey();
        if (value == _maximum) _maximum = _frequencies.lastKey();
      }
    } else {
      _frequencies[value] = frequency - 1;
    }
    return true;
  }

  _NumericStats copy() {
    final result = _NumericStats(trackFrequencies: _tracksFrequencies)
      ..sum = sum
      ..count = count
      .._minimum = _minimum
      .._maximum = _maximum;
    result._frequencies.addAll(_frequencies);
    return result;
  }

  void merge(_NumericStats other) {
    sum += other.sum;
    count += other.count;
    if (!_tracksFrequencies || !other._tracksFrequencies) {
      _tracksFrequencies = false;
      _frequencies.clear();
      if (_minimum == null ||
          other._minimum != null && other._minimum! < _minimum!) {
        _minimum = other._minimum;
      }
      if (_maximum == null ||
          other._maximum != null && other._maximum! > _maximum!) {
        _maximum = other._maximum;
      }
      return;
    }
    other._frequencies.forEach((value, frequency) {
      _frequencies.update(value, (current) => current + frequency,
          ifAbsent: () => frequency);
    });
    if (_frequencies.isNotEmpty) {
      _minimum = _frequencies.firstKey();
      _maximum = _frequencies.lastKey();
    }
  }
}
