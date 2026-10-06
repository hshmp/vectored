import 'dart:collection';

import 'int_tensor.dart';
import 'series.dart';

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
  final Set<({int segment, int row})> _deletedRows;
  final Map<int, int> _deletedCountsBySegment;
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
    this._deletedRows,
    this._deletedCountsBySegment,
    this._numericStats,
    this._uncacheableColumns,
    this._nonNumericColumns,
    this._aggregateCaching,
  );

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
      <({int segment, int row})>{},
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
      <({int segment, int row})>{},
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
  int get length => _physicalLength - _deletedRows.length;

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
        column: chunk.columns[column]![location.row],
    });
  }

  Object? _valueAt(String column, int index) {
    _flushPending();
    final location = _locationAt(index);
    return _chunks[location.chunk].columns[column]![location.row];
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
    _deletedRows.add((segment: chunk.id, row: location.row));
    _deletedCountsBySegment.update(
      chunk.id,
      (count) => count + 1,
      ifAbsent: () => 1,
    );
    for (final column in _columns) {
      final value = chunk.columns[column]![location.row];
      final stats = _numericStats[column];
      if (value is! num || stats == null) continue;

      // extrema lost; row already marked deleted
      if (!stats.remove(value)) {
        _numericStats[column] = _rebuildNumericStats(column);
      }
    }
    _visibleRows = null;
  }

  /// Returns a mask of live rows whose string [column] contains [pattern].
  ///
  /// Uses the packed [StringSeries] matcher on each chunk and omits deleted
  /// rows. Null values never match.
  IntTensor contains(String column, String pattern) => _stringMask(
        column,
        (values) => values.containingIndices(pattern),
        (text) => text.contains(pattern),
      );

  /// Returns a mask of live rows matching [pattern] in [column].
  ///
  /// Null values never match.
  IntTensor match(String column, RegExp pattern) => _stringMask(
        column,
        (values) => values.matchingIndices(pattern),
        pattern.hasMatch,
      );

  IntTensor _stringMask(
    String column,
    List<int> Function(StringSeries) matcher,
    bool Function(String) rowTest,
  ) {
    _requireColumn(column);
    _flushPending();
    final output = IntTensor.vector(length);
    var logicalRow = 0;

    for (final chunk in _chunks) {
      final hits = _matchingRows(chunk, column, matcher, rowTest);
      var next = 0;
      for (var row = 0; row < chunk.length; row++) {
        if (_deletedRows.contains((segment: chunk.id, row: row))) continue;
        while (next < hits.length && hits[next] < row) {
          next++;
        }
        if (next < hits.length && hits[next] == row) output[logicalRow] = 1;
        logicalRow++;
      }
    }
    return output;
  }

  /// Returns the ascending rows of [chunk] whose string [column] matches.
  ///
  /// Packed string chunks use [matcher]; chunks holding strings and nulls
  /// (stored as [ObjectSeries]) test each string with [rowTest].
  static List<int> _matchingRows(
    _FrameSegment chunk,
    String column,
    List<int> Function(StringSeries) matcher,
    bool Function(String) rowTest,
  ) {
    final values = chunk.columns[column];
    if (values is StringSeries) return matcher(values);
    if (values is ObjectSeries &&
        values.data.every((value) => value == null || value is String)) {
      return [
        for (var row = 0; row < values.length; row++)
          if (values[row] case final String text when rowTest(text)) row,
      ];
    }
    throw ArgumentError.value(
        column, 'column', 'Column is not a string series');
  }

  /// Returns a new frame containing rows selected by [mask].
  ///
  /// [mask] must have one entry per live row; nonzero entries are selected.
  DataFrame filter(IntTensor mask) {
    _flushPending();
    if (mask.length != length) {
      throw ArgumentError.value(mask.length, 'mask', 'Mask length mismatch');
    }

    final selectedChunks = <_FrameSegment>[];
    var logicalRow = 0;
    for (final chunk in _chunks) {
      final selectedRows = <int>[];
      for (var row = 0; row < chunk.length; row++) {
        if (_deletedRows.contains((segment: chunk.id, row: row))) continue;
        if (mask[logicalRow++] != 0) selectedRows.add(row);
      }
      _appendSelectedChunk(chunk, selectedRows, selectedChunks);
    }
    return _fromSelectedChunks(selectedChunks);
  }

  /// Returns a new frame containing rows where [column] contains [pattern].
  DataFrame filterContains(String column, String pattern) => _filterString(
        column,
        (values) => values.containingIndices(pattern),
        (text) => text.contains(pattern),
      );

  /// Returns a new frame containing rows where [pattern] matches [column].
  DataFrame filterMatch(String column, RegExp pattern) => _filterString(
        column,
        (values) => values.matchingIndices(pattern),
        pattern.hasMatch,
      );

  /// Returns a snapshot view of rows whose [column] contains [pattern].
  ///
  /// The view stores source segment and row indices; it does not materialize
  /// columns or allocate a row mask.
  DataFrameView filterViewContains(String column, String pattern) =>
      _filterView(
        column,
        (values) => values.containingIndices(pattern),
        (text) => text.contains(pattern),
      );

  /// Returns a snapshot view of rows matching [pattern] in [column].
  DataFrameView filterViewMatch(String column, RegExp pattern) => _filterView(
        column,
        (values) => values.matchingIndices(pattern),
        pattern.hasMatch,
      );

  DataFrameView _filterView(
    String column,
    List<int> Function(StringSeries) matcher,
    bool Function(String) rowTest,
  ) {
    _requireColumn(column);
    _flushPending();
    final rows = <_RowPointer>[];
    for (final chunk in _chunks) {
      for (final row in _matchingRows(chunk, column, matcher, rowTest)) {
        if (!_deletedRows.contains((segment: chunk.id, row: row))) {
          rows.add(_RowPointer(chunk, row));
        }
      }
    }
    return DataFrameView._(this, rows);
  }

  DataFrame _filterString(
    String column,
    List<int> Function(StringSeries) matcher,
    bool Function(String) rowTest,
  ) {
    _requireColumn(column);
    _flushPending();
    final selectedChunks = <_FrameSegment>[];
    for (final chunk in _chunks) {
      final selectedRows = <int>[];
      for (final row in _matchingRows(chunk, column, matcher, rowTest)) {
        if (!_deletedRows.contains((segment: chunk.id, row: row))) {
          selectedRows.add(row);
        }
      }
      _appendSelectedChunk(chunk, selectedRows, selectedChunks);
    }
    return _fromSelectedChunks(selectedChunks);
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
      <({int segment, int row})>{},
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
        for (var row = 0; row < chunk.length; row++) {
          if (_deletedRows.contains((segment: chunk.id, row: row))) continue;
          final value = values[row];
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

  /// Returns the live value count for numeric [column].
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
    final (leftChunks, leftDeletes, leftDeleteCounts) =
        _copySegmentsForConcat(this);
    final (rightChunks, rightDeletes, rightDeleteCounts) =
        _copySegmentsForConcat(other);
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
      {...leftDeleteCounts, ...rightDeleteCounts},
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
    if (_deletedRows.isEmpty && _chunks.length <= 1) return;

    final values = [for (final _ in _columns) <Object?>[]];
    for (final chunk in _chunks) {
      final sourceColumns = [
        for (final column in _columns) chunk.columns[column]!,
      ];
      for (var row = 0; row < chunk.length; row++) {
        if (_deletedRows.contains((segment: chunk.id, row: row))) continue;
        for (var index = 0; index < _columns.length; index++) {
          values[index].add(sourceColumns[index][row]);
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
    _deletedRows.clear();
    _deletedCountsBySegment.clear();
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
        ObjectSeries(:final data) => data.every((value) => value is num),
        StringSeries() => false,
      };

  /// Adds every stored value of numeric [series] to [stats].
  ///
  /// Returns false when a value is not finite and cannot be cached.
  static bool _addSeriesToStats(_NumericStats stats, Series<dynamic> series) {
    for (var row = 0; row < series.length; row++) {
      final value = _cacheableNumber(series[row]);
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
      for (var rowIndex = 0; rowIndex < chunk.length; rowIndex++) {
        if (!_deletedRows.contains((segment: chunk.id, row: rowIndex))) {
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
      final liveCount = chunk.length - (_deletedCountsBySegment[chunk.id] ?? 0);
      if (remaining >= liveCount) {
        remaining -= liveCount;
        continue;
      }
      for (var row = 0; row < chunk.length; row++) {
        if (_deletedRows.contains((segment: chunk.id, row: row))) continue;
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
      for (var row = 0; row < chunk.length; row++) {
        if (_deletedRows.contains((segment: chunk.id, row: row))) continue;
        final value = values[row];
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
      for (var row = 0; row < segment.length; row++) {
        if (_deletedRows.contains((segment: segment.id, row: row))) continue;
        stats.add(values[row] as num);
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

  bool _hasDeletedRows(_FrameSegment chunk) {
    return (_deletedCountsBySegment[chunk.id] ?? 0) != 0;
  }

  bool _sameSchema(DataFrame other) =>
      _columns.length == other._columns.length &&
      _columns.toSet().containsAll(other._columns);

  static int _newSegmentId() => _nextRowId++;

  static (List<_FrameSegment>, Set<({int segment, int row})>, Map<int, int>)
      _copySegmentsForConcat(DataFrame frame) {
    final newSegments = <_FrameSegment>[];
    final segmentIds = <int, int>{};
    for (final segment in frame._chunks) {
      final newId = _newSegmentId();
      segmentIds[segment.id] = newId;
      newSegments.add(_FrameSegment(segment.data, newId));
    }

    final deletes = <({int segment, int row})>{};
    final deleteCounts = <int, int>{};
    for (final deletion in frame._deletedRows) {
      final segmentId = segmentIds[deletion.segment]!;
      deletes.add((
        segment: segmentId,
        row: deletion.row,
      ));
      deleteCounts.update(segmentId, (count) => count + 1, ifAbsent: () => 1);
    }
    return (newSegments, deletes, deleteCounts);
  }

  /// Picks the most compact series type that stores [values] without
  /// silently changing them.
  ///
  /// Ints outside int32 range, or ints too large for float32 in a mixed
  /// numeric column, fall back to an exact [ObjectSeries].
  static Series<dynamic> _seriesFromValues(String name, List<Object?> values) {
    if (values.every((value) => value is int)) {
      if (values.every((value) => _fitsInt32(value as int))) {
        return Series.fromInts(name, values.cast<int>());
      }
      return Series.fromObjects(name, values);
    }
    if (values.every((value) => value is num)) {
      if (values.any((value) => value is int && !_fitsFloat32(value))) {
        return Series.fromObjects(name, values);
      }
      return Series.fromFloats(
        name,
        values.map((value) => (value as num).toDouble()).toList(),
      );
    }
    if (values.every((value) => value is String)) {
      return Series.fromStrings(name, values.cast<String>());
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
        } else if (values is List<String>) {
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
        column: pointer.segment.columns[column]![pointer.row],
    });
  }

  /// Sums a numeric column directly through selected row pointers.
  num sum(String column) {
    _frame._requireColumn(column);
    num total = 0;
    for (final pointer in _rows) {
      final value = pointer.segment.columns[column]![pointer.row];
      if (value is! num) {
        throw ArgumentError.value(column, 'column', 'Column is not numeric');
      }
      total += value;
    }
    return total;
  }

  /// Returns the mean of numeric [column], or `null` when the view is empty.
  double? mean(String column) {
    _frame._requireColumn(column);
    if (_rows.isEmpty) return null;
    return sum(column) / _rows.length;
  }

  /// Filters this view without materializing columns or a mask.
  ///
  /// Null values never match.
  DataFrameView filterContains(String column, String pattern) {
    _frame._requireColumn(column);
    final selected = <_RowPointer>[];
    for (final pointer in _rows) {
      final value = pointer.segment.columns[column]![pointer.row];
      if (value != null && value is! String) {
        throw ArgumentError.value(
          column,
          'column',
          'Column is not a string series',
        );
      }
      if (value?.contains(pattern) ?? false) selected.add(pointer);
    }
    return DataFrameView._(_frame, selected);
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
    return pointer.segment.columns[name]![pointer.row];
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
