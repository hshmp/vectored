import 'dart:math';
import 'dart:io';
import 'dart:typed_data';
// ignore: depend_on_referenced_packages
import 'package:matrix2d/matrix2d.dart' as matrix2d;
import 'package:vectored/vectored.dart';

(double mean, double stdDev) computeStats(List<double> samples) {
  final int n = samples.length;
  if (n <= 1) return (samples.first, 0.0);

  final double mean = samples.reduce((a, b) => a + b) / n;
  final double variance =
      samples.map((x) => pow(x - mean, 2)).reduce((a, b) => a + b) / (n - 1);

  return (mean, sqrt(variance));
}

String formatMilliseconds(double value) =>
    value < 0.001 ? '<0.001 ms' : '${value.toStringAsFixed(4)} ms';

String formatBenchmarkLatency(double milliseconds) => milliseconds < 1
    ? '${(milliseconds * 1000).toStringAsFixed(3)} us'
    : '${milliseconds.toStringAsFixed(2)} ms';

(double mean, double stdDev) benchmark(
  void Function() fn, {
  void Function()? setup,
  int warmup = 3,
  int trials = 15,
}) {
  for (int i = 0; i < warmup; i++) {
    setup?.call();
    fn();
  }

  final samples = <double>[];
  final sw = Stopwatch();

  for (int i = 0; i < trials; i++) {
    setup?.call();
    sw.reset();
    sw.start();
    fn();
    sw.stop();
    samples.add(sw.elapsedMicroseconds / 1000.0);
  }

  return computeStats(samples);
}

((double, double), (double, double)) benchmarkPair(
  void Function() first,
  void Function() second, {
  int warmup = 3,
  int trials = 15,
  int iterations = 1,
  void Function()? setupFirst,
  void Function()? setupSecond,
}) {
  for (int i = 0; i < warmup; i++) {
    setupFirst?.call();
    first();
    setupSecond?.call();
    second();
  }

  final firstSamples = <double>[];
  final secondSamples = <double>[];
  final stopwatch = Stopwatch();

  void measure(
    void Function()? setup,
    void Function() fn,
    List<double> samples,
  ) {
    setup?.call();
    stopwatch
      ..reset()
      ..start();
    for (int i = 0; i < iterations; i++) {
      fn();
    }
    stopwatch.stop();
    samples.add(stopwatch.elapsedMicroseconds / iterations / 1000.0);
  }

  for (int i = 0; i < trials; i++) {
    if (i.isEven) {
      measure(setupFirst, first, firstSamples);
      measure(setupSecond, second, secondSamples);
    } else {
      measure(setupSecond, second, secondSamples);
      measure(setupFirst, first, firstSamples);
    }
  }

  return (computeStats(firstSamples), computeStats(secondSamples));
}

void main(List<String> args) {
  const int size = 1500000;
  final trials = _argumentInt(args, '--trials', 15);
  final rows = _argumentInt(args, '--rows', 50000);
  if (trials <= 0 || rows <= 0) {
    throw ArgumentError('Benchmark rows and trials must be positive');
  }

  if (args.contains('--dataframe-csv')) {
    stderr.writeln(
      'Dart ${Platform.version.split(' ').first}; Matrix2D 1.0.4; '
      'rows=$rows; trials=$trials',
    );
    _benchmarkDataFrame(trials, rowCount: rows, csv: true);
    return;
  }

  print('================================================================');
  print('              VECTORED BENCHMARK SUITE (N = $trials)            ');
  print('================================================================\n');

  final rng = Random(42);

  // float32 arithmetic
  print('>>> 1. Float32 Arithmetic (Dataset: $size elements)...');
  final listA = List<double>.generate(size, (_) => rng.nextDouble());
  final listB = List<double>.generate(size, (_) => rng.nextDouble());
  final f32A = Float32List.fromList(listA);
  final f32B = Float32List.fromList(listB);
  final f32Out = Float32List(size);
  final listDoubleInPlaceOut = List<double>.from(listA);

  final tensorA = FloatTensor.fromList(listA);
  final tensorB = FloatTensor.fromList(listB);
  List<double>? listDoubleOut;
  FloatTensor? tensorFloatAddOut;

  final seriesA = Series.fromFloats('col_a', listA);
  final seriesB = Series.fromFloats('col_b', listB);

  final (stdMean, stdStd) = benchmark(() {
    final out = List<double>.filled(size, 0.0);
    for (int i = 0; i < size; i++) {
      out[i] = listA[i] + listB[i];
    }
    listDoubleOut = out;
  }, trials: trials);

  final (f32Mean, f32Std) = benchmark(() {
    for (int i = 0; i < size; i++) {
      f32Out[i] = f32A[i] + f32B[i];
    }
  }, trials: trials);

  final (listDoubleInPlaceMean, listDoubleInPlaceStd) = benchmark(() {
    for (int i = 0; i < size; i++) {
      listDoubleInPlaceOut[i] += listB[i];
    }
  }, setup: () => listDoubleInPlaceOut.setAll(0, listA), trials: trials);

  final (seriesAllocMean, seriesAllocStd) = benchmark(() {
    final _ = seriesA + seriesB;
  }, trials: trials);

  final (tensorAllocMean, tensorAllocStd) = benchmark(() {
    tensorFloatAddOut = tensorA + tensorB;
  }, trials: trials);

  final (seriesMutMean, seriesMutStd) = benchmark(() {
    seriesA.add_(seriesB);
  }, setup: () => seriesA.tensor.buffer.setAll(0, f32A), trials: trials);

  final (tensorMutMean, tensorMutStd) = benchmark(() {
    tensorA.add_(tensorB);
  }, setup: () => tensorA.buffer.setAll(0, f32A), trials: trials);

  if (listDoubleOut == null || tensorFloatAddOut == null) {
    throw StateError('Float tensor benchmark did not produce results');
  }
  for (int i = 0; i < size; i++) {
    if (listDoubleOut![i] != listA[i] + listB[i] ||
        (tensorFloatAddOut![i] - (listA[i] + listB[i])).abs() > 1e-5) {
      throw StateError('Float add benchmark result mismatch at index $i');
    }
  }

  final allocatingFloatTable = ConsoleTable(
    headers: [
      'Allocating add',
      'Mean latency',
      'Std dev',
      'Speedup vs List<double>'
    ],
  );
  allocatingFloatTable.addRow([
    'List<double> baseline',
    '${stdMean.toStringAsFixed(2)} ms',
    '${stdStd.toStringAsFixed(2)} ms',
    '1.00x',
  ]);
  allocatingFloatTable.addRow([
    'FloatSeries (+)',
    '${seriesAllocMean.toStringAsFixed(2)} ms',
    '${seriesAllocStd.toStringAsFixed(2)} ms',
    '${(stdMean / seriesAllocMean).toStringAsFixed(2)}x',
  ]);
  allocatingFloatTable.addRow([
    'FloatTensor (+)',
    '${tensorAllocMean.toStringAsFixed(2)} ms',
    '${tensorAllocStd.toStringAsFixed(2)} ms',
    '${(stdMean / tensorAllocMean).toStringAsFixed(2)}x',
  ]);

  final inPlaceFloatTable = ConsoleTable(
    headers: [
      'In-place add',
      'Mean latency',
      'Std dev',
      'Speedup vs List<double>'
    ],
  );
  inPlaceFloatTable.addRow([
    'List<double> baseline',
    '${listDoubleInPlaceMean.toStringAsFixed(2)} ms',
    '${listDoubleInPlaceStd.toStringAsFixed(2)} ms',
    '1.00x',
  ]);
  inPlaceFloatTable.addRow([
    'Float32List scalar',
    '${f32Mean.toStringAsFixed(2)} ms',
    '${f32Std.toStringAsFixed(2)} ms',
    '${(listDoubleInPlaceMean / f32Mean).toStringAsFixed(2)}x',
  ]);
  inPlaceFloatTable.addRow([
    'FloatSeries (add_)',
    '${seriesMutMean.toStringAsFixed(2)} ms',
    '${seriesMutStd.toStringAsFixed(2)} ms',
    '${(listDoubleInPlaceMean / seriesMutMean).toStringAsFixed(2)}x',
  ]);
  inPlaceFloatTable.addRow([
    'FloatTensor (add_)',
    '${tensorMutMean.toStringAsFixed(2)} ms',
    '${tensorMutStd.toStringAsFixed(2)} ms',
    '${(listDoubleInPlaceMean / tensorMutMean).toStringAsFixed(2)}x',
  ]);

  print(allocatingFloatTable);
  print('');
  print(inPlaceFloatTable);
  print('');

  // int32 arithmetic
  print('>>> 2. Int32 Arithmetic (Dataset: $size elements)...');
  final intListA = List<int>.generate(size, (i) => i & 1023);
  final intListB = List<int>.generate(size, (i) => (i * 3) & 1023);
  final intA = Int32List.fromList(intListA);
  final intB = Int32List.fromList(intListB);
  final intKernelOut = Int32List(size);
  final tensorIntA = IntTensor.fromList(intListA);
  final tensorIntB = IntTensor.fromList(intListB);
  List<int>? listAddOut;
  List<int>? listSubOut;
  List<int>? listMulOut;
  Int32List? scalarAddOut;
  Int32List? scalarSubOut;
  Int32List? scalarMulOut;
  IntTensor? tensorAddOut;
  IntTensor? tensorSubOut;
  IntTensor? tensorMulOut;

  final (listIntAddMean, listIntAddStd) = benchmark(() {
    final out = List<int>.filled(size, 0);
    for (int i = 0; i < size; i++) {
      out[i] = intListA[i] + intListB[i];
    }
    listAddOut = out;
  }, trials: trials);
  final (listIntSubMean, listIntSubStd) = benchmark(() {
    final out = List<int>.filled(size, 0);
    for (int i = 0; i < size; i++) {
      out[i] = intListA[i] - intListB[i];
    }
    listSubOut = out;
  }, trials: trials);
  final (listIntMulMean, listIntMulStd) = benchmark(() {
    final out = List<int>.filled(size, 0);
    for (int i = 0; i < size; i++) {
      out[i] = intListA[i] * intListB[i];
    }
    listMulOut = out;
  }, trials: trials);
  final (scalarIntAddMean, scalarIntAddStd) = benchmark(() {
    final out = Int32List(size);
    for (int i = 0; i < size; i++) {
      out[i] = intA[i] + intB[i];
    }
    scalarAddOut = out;
  }, trials: trials);
  final (kernelIntAddMean, kernelIntAddStd) = benchmark(() {
    SimdOps.addInt(intA, intB, intKernelOut);
  }, trials: trials);
  final (tensorIntAddMean, tensorIntAddStd) = benchmark(() {
    tensorAddOut = tensorIntA + tensorIntB;
  }, trials: trials);
  final (scalarIntSubMean, scalarIntSubStd) = benchmark(() {
    final out = Int32List(size);
    for (int i = 0; i < size; i++) {
      out[i] = intA[i] - intB[i];
    }
    scalarSubOut = out;
  }, trials: trials);
  final (kernelIntSubMean, kernelIntSubStd) = benchmark(() {
    SimdOps.subInt(intA, intB, intKernelOut);
  }, trials: trials);
  final (tensorIntSubMean, tensorIntSubStd) = benchmark(() {
    tensorSubOut = tensorIntA - tensorIntB;
  }, trials: trials);
  final (scalarIntMulMean, scalarIntMulStd) = benchmark(() {
    final out = Int32List(size);
    for (int i = 0; i < size; i++) {
      out[i] = intA[i] * intB[i];
    }
    scalarMulOut = out;
  }, trials: trials);
  final (kernelIntMulMean, kernelIntMulStd) = benchmark(() {
    SimdOps.mulInt(intA, intB, intKernelOut);
  }, trials: trials);
  final (tensorIntMulMean, tensorIntMulStd) = benchmark(() {
    tensorMulOut = tensorIntA * tensorIntB;
  }, trials: trials);

  if (listAddOut == null ||
      listSubOut == null ||
      listMulOut == null ||
      scalarAddOut == null ||
      scalarSubOut == null ||
      scalarMulOut == null ||
      tensorAddOut == null ||
      tensorSubOut == null ||
      tensorMulOut == null) {
    throw StateError('Integer tensor benchmark did not produce results');
  }
  SimdOps.addInt(intA, intB, intKernelOut);
  for (int i = 0; i < size; i++) {
    if (listAddOut![i] != intListA[i] + intListB[i] ||
        listSubOut![i] != intListA[i] - intListB[i] ||
        listMulOut![i] != intListA[i] * intListB[i] ||
        scalarAddOut![i] != intA[i] + intB[i] ||
        intKernelOut[i] != intA[i] + intB[i] ||
        tensorAddOut![i] != intA[i] + intB[i]) {
      throw StateError('Integer add benchmark result mismatch at index $i');
    }
  }
  SimdOps.subInt(intA, intB, intKernelOut);
  for (int i = 0; i < size; i++) {
    if (scalarSubOut![i] != intA[i] - intB[i] ||
        intKernelOut[i] != intA[i] - intB[i] ||
        tensorSubOut![i] != intA[i] - intB[i]) {
      throw StateError(
          'Integer subtract benchmark result mismatch at index $i');
    }
  }
  SimdOps.mulInt(intA, intB, intKernelOut);
  for (int i = 0; i < size; i++) {
    if (scalarMulOut![i] != intA[i] * intB[i] ||
        intKernelOut[i] != intA[i] * intB[i] ||
        tensorMulOut![i] != intA[i] * intB[i]) {
      throw StateError(
          'Integer multiply benchmark result mismatch at index $i');
    }
  }

  final intTable = ConsoleTable(
    headers: [
      'Operation',
      'List<int> alloc',
      'Int32List alloc',
      'Kernel reused out',
      'IntTensor alloc',
      'Speedup vs List<int>',
    ],
  );
  intTable.addRow([
    'add',
    '${listIntAddMean.toStringAsFixed(2)} ms ± ${listIntAddStd.toStringAsFixed(2)}',
    '${scalarIntAddMean.toStringAsFixed(2)} ms ± ${scalarIntAddStd.toStringAsFixed(2)}',
    '${kernelIntAddMean.toStringAsFixed(2)} ms ± ${kernelIntAddStd.toStringAsFixed(2)}',
    '${tensorIntAddMean.toStringAsFixed(2)} ms ± ${tensorIntAddStd.toStringAsFixed(2)}',
    '${(listIntAddMean / tensorIntAddMean).toStringAsFixed(2)}x',
  ]);
  intTable.addRow([
    'subtract',
    '${listIntSubMean.toStringAsFixed(2)} ms ± ${listIntSubStd.toStringAsFixed(2)}',
    '${scalarIntSubMean.toStringAsFixed(2)} ms ± ${scalarIntSubStd.toStringAsFixed(2)}',
    '${kernelIntSubMean.toStringAsFixed(2)} ms ± ${kernelIntSubStd.toStringAsFixed(2)}',
    '${tensorIntSubMean.toStringAsFixed(2)} ms ± ${tensorIntSubStd.toStringAsFixed(2)}',
    '${(listIntSubMean / tensorIntSubMean).toStringAsFixed(2)}x',
  ]);
  intTable.addRow([
    'multiply',
    '${listIntMulMean.toStringAsFixed(2)} ms ± ${listIntMulStd.toStringAsFixed(2)}',
    '${scalarIntMulMean.toStringAsFixed(2)} ms ± ${scalarIntMulStd.toStringAsFixed(2)}',
    '${kernelIntMulMean.toStringAsFixed(2)} ms ± ${kernelIntMulStd.toStringAsFixed(2)}',
    '${tensorIntMulMean.toStringAsFixed(2)} ms ± ${tensorIntMulStd.toStringAsFixed(2)}',
    '${(listIntMulMean / tensorIntMulMean).toStringAsFixed(2)}x',
  ]);
  print(intTable);
  print('');

  // zero-copy slicing
  print('>>> 3. Slicing & Sub-Array Creation ($size elements)...');

  final (sublistMean, sublistStd) = benchmark(() {
    final _ = listA.sublist(100, size - 100);
  }, trials: trials);

  final (sliceMean, sliceStd) = benchmark(() {
    final _ = tensorA.slice(100, size - 100);
  }, trials: trials);

  final sliceTable = ConsoleTable(
    headers: [
      'Operation',
      'Strategy',
      'Mean Latency',
      'Std Dev (±)',
      'Speedup'
    ],
  );

  sliceTable.addRow([
    'List.sublist()',
    'Deep copy heap allocation',
    '${sublistMean.toStringAsFixed(3)} ms',
    '${sublistStd.toStringAsFixed(3)} ms',
    '1.00x',
  ]);
  sliceTable.addRow([
    'FloatTensor.slice()',
    'Zero-copy O(1) buffer view',
    formatMilliseconds(sliceMean),
    formatMilliseconds(sliceStd),
    sliceMean < 0.001
        ? '>${(sublistMean / 0.001).toStringAsFixed(0)}x'
        : '${(sublistMean / sliceMean).toStringAsFixed(0)}x',
  ]);

  print(sliceTable);
  print('');

  _benchmarkTextSearch(trials);
  _benchmarkDataFrame(trials);
}

void _benchmarkTextSearch(int trials) {
  const int standardRows = 50000;
  final logTags = ['[INFO]', '[WARN]', '[DEBUG]', '[ERROR]', '[FATAL]'];
  final logs = List<String>.generate(standardRows, (i) {
    final tag = logTags[i % logTags.length];
    return '2026-09-22T10:00:00.000Z $tag host_node_${i % 50}: '
        'connection reset by peer in worker thread';
  });

  const int adversarialRows = 512;
  const int adversarialRowLength = 1024;
  final repeatedRows =
      List<String>.filled(adversarialRows, 'A' * adversarialRowLength);

  final cases = <({
    String label,
    List<String> rows,
    String query,
  })>[
    (label: 'short / common hit', rows: logs, query: 'e'),
    (label: 'medium / 20% hit', rows: logs, query: '[FATAL]'),
    (label: 'long / miss', rows: logs, query: 'connection was not reset'),
    (
      label: 'repeated-prefix / miss',
      rows: repeatedRows,
      query: 'AB${'A' * 62}',
    ),
  ];

  final table = ConsoleTable(
    headers: [
      'Case',
      'Engine',
      'Mean latency',
      'Std dev',
      'vs core literal',
    ],
  );

  for (final searchCase in cases) {
    final rows = searchCase.rows;
    final query = searchCase.query;
    final series = TextSeries.fromStrings(searchCase.label, rows);
    final escapedRegex = RegExp(RegExp.escape(query));

    final engines = <String, IntTensor Function()>{
      'String.contains': () {
        final output = IntTensor.vector(rows.length);
        for (int row = 0; row < rows.length; row++) {
          if (rows[row].contains(query)) output[row] = 1;
        }
        return output;
      },
      'precompiled RegExp on rows': () {
        final output = IntTensor.vector(rows.length);
        for (int row = 0; row < rows.length; row++) {
          if (escapedRegex.hasMatch(rows[row])) output[row] = 1;
        }
        return output;
      },
      'TextSeries.contains': () => series.contains(query),
      'TextSeries.match': () => series.match(escapedRegex),
    };
    final samples = {
      for (final engine in engines.keys) engine: <double>[],
    };
    final outputs = <String, IntTensor>{};

    for (final entry in engines.entries) {
      for (int warmup = 0; warmup < 3; warmup++) {
        outputs[entry.key] = entry.value();
      }
    }

    final engineNames = engines.keys.toList();
    final stopwatch = Stopwatch();
    for (int trial = 0; trial < trials; trial++) {
      final start = trial % engineNames.length;
      for (int offset = 0; offset < engineNames.length; offset++) {
        final name = engineNames[(start + offset) % engineNames.length];
        stopwatch
          ..reset()
          ..start();
        outputs[name] = engines[name]!();
        stopwatch.stop();
        samples[name]!.add(stopwatch.elapsedMicroseconds / 1000.0);
      }
    }

    final coreOutput = outputs['String.contains'];
    if (coreOutput == null) {
      throw StateError('Core string search produced no result');
    }
    for (int row = 0; row < rows.length; row++) {
      final expected = coreOutput[row];
      for (final name in engineNames.skip(1)) {
        if (outputs[name]![row] != expected) {
          throw StateError(
            'Text search mismatch: ${searchCase.label}, row $row',
          );
        }
      }
    }

    final stats = <String, (double, double)>{
      for (final name in engineNames) name: computeStats(samples[name]!),
    };
    final coreMean = stats['String.contains']!.$1;
    void addRow(String engine) {
      final (mean, stdDev) = stats[engine]!;
      table.addRow([
        searchCase.label,
        engine,
        '${mean.toStringAsFixed(2)} ms',
        '${stdDev.toStringAsFixed(2)} ms',
        '${(coreMean / mean).toStringAsFixed(2)}x',
      ]);
    }

    for (final name in engineNames) {
      addRow(name);
    }
  }

  print(
      '>>> 4. Literal and RegExp Search (matching outputs; regex precompiled)');
  print(table);
}

void _benchmarkDataFrame(
  int trials, {
  int rowCount = 50000,
  bool csv = false,
}) {
  const pattern = 'category-3';
  final labels = List<String>.generate(
    rowCount,
    (row) => 'event-$row category-${row % 7}',
  );
  final values = List<int>.generate(rowCount, (row) => row % 1000);
  final rows = List<Map<String, Object?>>.generate(
    rowCount,
    (row) => {'label': labels[row], 'value': values[row]},
  );
  final frame = DataFrame.fromColumns(
    {'label': labels, 'value': values},
    chunkSize: 1024,
  );
  final scanFrame = DataFrame.fromSeries([
    Series.fromInts('value', values),
  ]);
  final cachedFrame = DataFrame.fromColumns({'value': values});

  IntTensor? listMask;
  IntTensor? frameMask;
  final (listMaskStats, frameMaskStats) = benchmarkPair(
    () {
      final mask = IntTensor.vector(rowCount);
      for (int row = 0; row < rowCount; row++) {
        if (labels[row].contains(pattern)) mask[row] = 1;
      }
      listMask = mask;
    },
    () => frameMask = frame.contains('label', pattern),
    trials: trials,
  );

  num listSum = 0;
  num frameSum = 0;
  num matrix2dSum = 0;
  final matrix2dApi = matrix2d.Matrix2d();
  void scanListSum() {
    var total = 0;
    for (final value in values) {
      total += value;
    }
    listSum = total;
  }

  final (listSumStats, frameSumStats) = benchmarkPair(
    scanListSum,
    () => frameSum = scanFrame.sum('value'),
    trials: trials,
    iterations: 10000,
  );
  final (_, matrix2dSumStats) = benchmarkPair(
    scanListSum,
    () => matrix2dSum = matrix2dApi.sum(values),
    trials: trials,
    iterations: 10,
  );
  num cachedFrameSum = 0;
  final (listCachedSumStats, frameCachedSumStats) = benchmarkPair(
    scanListSum,
    () => cachedFrameSum = cachedFrame.sum('value'),
    trials: trials,
    iterations: 10000,
  );

  List<Map<String, Object?>>? listFiltered;
  DataFrame? frameFiltered;
  DataFrameView? frameView;
  final (listFilterStats, frameFilterStats) = benchmarkPair(
    () {
      listFiltered = [
        for (final row in rows)
          if ((row['label'] as String).contains(pattern))
            Map<String, Object?>.from(row),
      ];
    },
    () => frameFiltered = frame.filterContains('label', pattern),
    trials: trials,
  );
  final viewFilterStats = benchmark(
    () => frameView = frame.filterViewContains('label', pattern),
    trials: trials,
  );
  List<String>? columnFilteredLabels;
  List<int>? columnFilteredValues;
  final (columnFilterStats, columnFrameFilterStats) = benchmarkPair(
    () {
      final selectedLabels = <String>[];
      final selectedValues = <int>[];
      for (int row = 0; row < rowCount; row++) {
        if (!labels[row].contains(pattern)) continue;
        selectedLabels.add(labels[row]);
        selectedValues.add(values[row]);
      }
      columnFilteredLabels = selectedLabels;
      columnFilteredValues = selectedValues;
    },
    () => frameFiltered = frame.filterContains('label', pattern),
    trials: trials,
  );

  if (listMask == null || frameMask == null) {
    throw StateError('DataFrame contains benchmark produced no result');
  }
  for (int row = 0; row < rowCount; row++) {
    if (listMask![row] != frameMask![row]) {
      throw StateError('DataFrame contains result mismatch at row $row');
    }
  }
  if (listSum != frameSum) {
    throw StateError('DataFrame sum result mismatch');
  }
  if (listSum != matrix2dSum) {
    throw StateError('Matrix2D sum result mismatch');
  }
  if (listSum != cachedFrameSum) {
    throw StateError('Cached DataFrame sum result mismatch');
  }
  final expectedCount = listFiltered!.length;
  if (frameFiltered!.length != expectedCount ||
      frameFiltered!.sum('value') !=
          listFiltered!.fold<num>(
            0,
            (total, row) => total + (row['value'] as int),
          )) {
    throw StateError('DataFrame filter result mismatch');
  }
  if (frameView!.length != expectedCount ||
      frameView!.sum('value') !=
          listFiltered!.fold<num>(
            0,
            (total, row) => total + (row['value'] as int),
          )) {
    throw StateError('DataFrame filter view result mismatch');
  }
  for (int row = 0; row < expectedCount; row++) {
    if (frameFiltered!['label'][row] != listFiltered![row]['label'] ||
        frameFiltered!['value'][row] != listFiltered![row]['value']) {
      throw StateError('DataFrame filtered row mismatch at row $row');
    }
  }
  if (columnFilteredLabels == null ||
      columnFilteredValues == null ||
      columnFilteredLabels!.length != expectedCount ||
      columnFilteredValues!.length != expectedCount) {
    throw StateError('Columnar filter benchmark produced an invalid result');
  }
  for (int row = 0; row < expectedCount; row++) {
    if (columnFilteredLabels![row] != frameFiltered!['label'][row] ||
        columnFilteredValues![row] != frameFiltered!['value'][row]) {
      throw StateError('Columnar filter result mismatch at row $row');
    }
  }

  final table = ConsoleTable(
    headers: [
      'Workload',
      'Implementation',
      'Mean latency',
      'Std dev',
      'vs list'
    ],
  );
  table.addRow([
    'filter view',
    'DataFrameView row pointers',
    formatBenchmarkLatency(viewFilterStats.$1),
    formatBenchmarkLatency(viewFilterStats.$2),
    'not materialized',
  ]);
  table.addSeparator();
  final csvRows = <String>[];
  csvRows.add(_csvBenchmarkRow(
    'dart',
    'vectored-view',
    'filter-view',
    rowCount,
    viewFilterStats.$1,
    viewFilterStats.$2,
  ));

  void addPair(
    String workload,
    String listLabel,
    (double, double) listStats,
    String frameLabel,
    (double, double) frameStats, {
    String implementationLibrary = 'vectored',
    bool separatorAfter = true,
  }) {
    final (listMean, listStdDev) = listStats;
    final (frameMean, frameStdDev) = frameStats;
    table.addRow([
      workload,
      listLabel,
      formatBenchmarkLatency(listMean),
      formatBenchmarkLatency(listStdDev),
      '1.00x',
    ]);
    table.addRow([
      workload,
      frameLabel,
      formatBenchmarkLatency(frameMean),
      formatBenchmarkLatency(frameStdDev),
      '${(listMean / frameMean).toStringAsFixed(2)}x',
    ]);
    if (separatorAfter) table.addSeparator();
    final operation = switch (workload) {
      'contains mask' => 'contains-mask',
      'sum int column' => 'sum',
      'cached sum lookup' => 'sum-cached',
      'materialized filter' => 'filter-materialized',
      'columnar filter' => 'filter-materialized',
      _ => workload,
    };
    final baseline = switch (listLabel) {
      'List<Map> copy' => 'list-map',
      'List columns copy' => 'list-columns',
      _ => 'list',
    };
    csvRows
      ..add(_csvBenchmarkRow(
          'dart', baseline, operation, rowCount, listStats.$1, listStats.$2))
      ..add(_csvBenchmarkRow('dart', implementationLibrary, operation, rowCount,
          frameStats.$1, frameStats.$2));
  }

  addPair(
    'contains mask',
    'List<String>.contains',
    listMaskStats,
    'DataFrame.contains',
    frameMaskStats,
  );
  addPair(
    'sum int column',
    'List<int> loop',
    listSumStats,
    'DataFrame.sum',
    frameSumStats,
    separatorAfter: false,
  );
  addPair(
    'sum int column',
    'List<int> loop',
    listSumStats,
    'Matrix2D.sum',
    matrix2dSumStats,
    implementationLibrary: 'matrix2d',
  );
  addPair(
    'cached sum lookup',
    'List<int> loop',
    listCachedSumStats,
    'DataFrame cached sum',
    frameCachedSumStats,
    separatorAfter: false,
  );
  addPair(
    'materialized filter',
    'List<Map> copy',
    listFilterStats,
    'DataFrame.filterContains',
    frameFilterStats,
    implementationLibrary: 'vectored-list-map',
  );
  addPair(
    'columnar filter',
    'List columns copy',
    columnFilterStats,
    'DataFrame.filterContains',
    columnFrameFilterStats,
    implementationLibrary: 'vectored-list-columns',
    separatorAfter: false,
  );

  final aggregateCsvRows = _benchmarkCachedAggregates(
    trials,
    rowCount: rowCount,
    csv: csv,
  );
  if (csv) {
    print('runtime,library,operation,rows,mean_ms,stddev_ms');
    for (final row in [...csvRows, ...aggregateCsvRows]) {
      print(row);
    }
  } else {
    print('>>> 5. DataFrame Operations ($rowCount rows, chunk size 1024)');
    print(
      'Filter compares row-map copies, column-list copies, and typed chunks.',
    );
    print(table);
  }
}

List<String> _benchmarkCachedAggregates(
  int trials, {
  required int rowCount,
  bool csv = false,
}) {
  const editCount = 40;
  final initial = List<int>.generate(rowCount, (row) => row % 1000);
  DataFrame? frame;
  List<int>? values;
  double frameChecksum = 0;
  double listChecksum = 0;

  void resetFrame() {
    frame = DataFrame.fromColumns(
      {'value': initial},
      chunkSize: 1024,
    );
  }

  void resetList() {
    values = List<int>.of(initial);
  }

  void runFrameWorkload() {
    var checksum = 0.0;
    for (int edit = 0; edit < editCount; edit++) {
      frame!
        ..deleteRow(0)
        ..appendRow({'value': rowCount + edit});
      checksum += frame!.mean('value')!;
    }
    frameChecksum = checksum;
  }

  void runListWorkload() {
    var checksum = 0.0;
    for (int edit = 0; edit < editCount; edit++) {
      values!
        ..removeAt(0)
        ..add(rowCount + edit);
      var sum = 0;
      for (final value in values!) {
        sum += value;
      }
      checksum += sum / values!.length;
    }
    listChecksum = checksum;
  }

  final (listStats, frameStats) = benchmarkPair(
    runListWorkload,
    runFrameWorkload,
    trials: trials,
    setupFirst: resetList,
    setupSecond: resetFrame,
  );
  if ((listChecksum - frameChecksum).abs() > 1e-8) {
    throw StateError('Cached aggregate result differs from list reference');
  }

  final table = ConsoleTable(
    headers: [
      'Workload',
      'Implementation',
      'Mean latency',
      'Std dev',
      'vs list'
    ],
  );
  final (listMean, listStdDev) = listStats;
  final (frameMean, frameStdDev) = frameStats;
  table.addRow([
    '$editCount edits + aggregate queries',
    'List remove/add + scan',
    formatBenchmarkLatency(listMean),
    formatBenchmarkLatency(listStdDev),
    '1.00x',
  ]);
  table.addRow([
    '$editCount edits + aggregate queries',
    'DataFrame lazy delete/append + cache',
    formatBenchmarkLatency(frameMean),
    formatBenchmarkLatency(frameStdDev),
    '${(listMean / frameMean).toStringAsFixed(2)}x',
  ]);
  final csvRows = <String>[
    _csvBenchmarkRow(
        'dart', 'list', 'edit-query-40', rowCount, listMean, listStdDev),
    _csvBenchmarkRow(
        'dart', 'vectored', 'edit-query-40', rowCount, frameMean, frameStdDev),
  ];

  DataFrame? fragmented;
  var rowsBeforeCompaction = 0;
  var physicalRowsBeforeCompaction = 0;
  var chunksBeforeCompaction = 0;
  final (compactMean, compactStdDev) = benchmark(
    () => fragmented!.compact(),
    setup: () {
      fragmented = DataFrame.fromColumns(
        {'value': initial},
        chunkSize: 1024,
      );
      for (int row = 0; row < editCount * 5; row++) {
        fragmented!.deleteRow(0);
        fragmented!.appendRow({'value': rowCount + row});
      }
      rowsBeforeCompaction = fragmented!.length;
      physicalRowsBeforeCompaction = fragmented!.physicalLength;
      chunksBeforeCompaction = fragmented!.physicalChunkCount;
    },
    trials: trials,
  );
  if (fragmented!.physicalLength != rowsBeforeCompaction ||
      fragmented!.length != rowsBeforeCompaction) {
    throw StateError('DataFrame compaction did not remove deleted rows');
  }

  csvRows.add(_csvBenchmarkRow(
    'dart',
    'vectored',
    'compact-200-edits',
    rowCount,
    compactMean,
    compactStdDev,
  ));
  if (!csv) {
    print(
        '>>> 6. Cached Aggregates and Defragmentation ($rowCount rows, chunk size 1024)');
    print(
        'Aggregate query: average(value) - min(value), once after each edit.');
    print(table);
    print(
      'DataFrame.compact: ${compactMean.toStringAsFixed(2)} ms '
      '± ${compactStdDev.toStringAsFixed(2)} ms; '
      '$physicalRowsBeforeCompaction physical rows, $chunksBeforeCompaction chunks '
      '→ ${fragmented!.physicalLength} rows, '
      '${fragmented!.physicalChunkCount} chunks',
    );
  }
  return csvRows;
}

int _argumentInt(List<String> args, String name, int fallback) {
  final prefix = '$name=';
  String? argument;
  for (final value in args) {
    if (value.startsWith(prefix)) {
      argument = value;
      break;
    }
  }
  if (argument == null) return fallback;
  final value = int.tryParse(argument.substring(prefix.length));
  if (value == null) {
    throw FormatException('Invalid benchmark argument: $argument');
  }
  return value;
}

String _csvBenchmarkRow(
  String runtime,
  String library,
  String operation,
  int rows,
  double meanMs,
  double stdDevMs,
) =>
    '$runtime,$library,$operation,$rows,${meanMs.toStringAsPrecision(8)},'
    '${stdDevMs.toStringAsPrecision(8)}';
