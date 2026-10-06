A sample command-line application with an entrypoint in `bin/`, library code
in `lib/`, and example unit test in `test/`.

## DataFrame

`DataFrame` stores named `Series` objects in bounded column chunks. Use
`DataFrame.fromSeries` to assemble existing float, int, and string series, or
`DataFrame.fromColumns` to build the typed series from lists. Appends are
buffered into chunks; concatenation links series chunks without copying values.
Deletes are logical until `compact()` copies live values into fresh series.
Column access returns a lazy view; calling `toList()` materializes it.

```dart
final left = DataFrame.fromColumns({
  'id': [1, 2],
  'name': ['ada', 'grace'],
});

left.appendRow({'id': 3, 'name': 'linus'});
left.deleteRow(1);

final matches = left.contains('name', 'lin');
final filtered = left.filter(matches);
final view = left.filterViewContains('name', 'lin');
final total = left.sum('id');
final names = left['name'].toList();
final matchingIds = view['id'].toList();
final materializedView = view.materialize();
left.compact();
```

`contains` delegates to the packed UTF-8 `StringSeries` matcher; `match` accepts
a precompiled regular expression. Owned numeric columns maintain cached sum and
count plus ordered value frequencies for min/max, so appends and deletes update
aggregates without rescanning live rows. `mean`, `sum`, `count`, `min`, and
`max` expose those numeric aggregates. Frames built with
`fromSeries` remain uncached because their shared series can be mutated
externally. `concat` returns a new frame and shares series chunks.
Later appends or deletes on either input do not change the combined frame.
`compact()` is explicit because it copies live values; use it when deleted rows
or many small chunks consume too much memory. Compaction gathers columns
directly and packs them into bounded chunks.

`filterContains` and `filterMatch` return independent, materialized frames.
`filterViewContains` and `filterViewMatch` instead capture matching source-row
pointers without building a mask or copying columns. Views keep their selected
rows when the source is appended to, deleted from, or compacted; changes to
externally shared Series values remain visible. Use `materialize()` when the
view needs independent typed Series storage.

Run the cross-library benchmark from PowerShell:

```powershell
.\benchmarks\run_benchmarks.ps1
```

It compiles Dart to AOT, runs Vectored and Matrix2D, then runs Pandas and Polars, checks result parity, and writes a wide comparison
CSV with one row per operation and separate mean/stddev columns for each
library and list baseline. The filter rows keep the two Vectored measurements
separate so each can be compared with its corresponding list baseline. The
original long-form records are saved alongside it with a `.raw.csv` suffix.
Both files are written without a UTF-8 BOM.
Install Python dependencies first with
`python -m pip install -r benchmarks/requirements-python.txt`. Override the
defaults with `-Rows 100000 -Trials 25 -Warmups 5 -Python C:\path\to\python.exe`.

For a standalone Dart run, `dart run bin/benchmark.dart` runs the full suite;
`dart run bin/benchmark.dart --dataframe-csv --rows=50000 --trials=15` emits
only the DataFrame CSV.
The Dart suite also benchmarks Matrix2D's numeric sum implementation through a
benchmark-only compatibility copy of Matrix2D 1.0.4. Only the SDK constraint is
relaxed; upstream source and its MIT license are retained. Matrix2D is an
array-math package, not a DataFrame, so comparisons are limited to shared
numeric reductions. Spark is intentionally a separate distributed benchmark:
startup, partitioning, and cluster configuration make local in-process timings
misleading.

Materialized filtering scans the predicate and allocates a new result frame
containing the selected rows. This differs from a mask-only query and from a
lazy view; benchmark each workload separately. The Dart benchmark includes
both a `List<Map>` baseline and a column-oriented list baseline for filtering.
The `filter-view` row measures pointer selection only and does not include
materializing the selected rows; do not compare it as a replacement for
`filter-materialized`.
