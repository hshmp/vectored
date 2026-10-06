# Vectored

Fast data tables for Dart and Flutter. If you've used pandas, you already know
most of it, and you don't need Python or a server to use it.

## Why Vectored

- **Fast:** filtering, searching and totals are much quicker than plain Dart
  lists, and quicker than Pandas in early tests.
- **Familiar:** `sum`, `mean`, `filter` and `concat` work the way they do in
  Pandas.
- **Built for live data:** add and remove rows freely, and totals update
  instantly instead of being recalculated.
- **Pure Dart:** runs anywhere Dart runs, including Flutter apps, with nothing
  extra to install.

## How it compares

| | Vectored | Matrix2D | Pandas | Polars |
|---|---|---|---|---|
| Language | Dart | Dart | Python | Python / Rust |
| Tables with named columns | ✅ | ❌ | ✅ | ✅ |
| Text search and filtering | ✅ | ❌ | ✅ | ✅ |
| Instant totals while editing | ✅ | ❌ | ❌ | ❌ |
| Runs inside a Flutter app | ✅ | ✅ | ❌ | ❌ |
| Group by, joins, CSV import | Planned | ❌ | ✅ | ✅ |

Pick Vectored when your data lives in a Dart or Flutter app. For heavy
analysis on a desktop or server, Polars is still the fastest option.

## Getting started

```dart
import 'package:vectored/vectored.dart';

final people = DataFrame.fromColumns({
  'age': [36, 45, 29],
  'name': ['ada', 'grace', 'linus'],
});
```

## Reading data

```dart
people['name'].toList(); // [ada, grace, linus]
people.rowAt(0);         // {age: 36, name: ada}
```

## Adding and removing rows

```dart
people.appendRow({'age': 52, 'name': 'barbara'});
people.deleteRow(1);
```

Call `people.compact()` now and then after lots of deletes to free up memory.

## Totals and averages

```dart
people.sum('age');
people.mean('age');
people.min('age');
people.max('age');
```

## Searching and filtering

```dart
people.filterContains('name', 'a');         // rows whose name contains "a"
people.filterMatch('name', RegExp(r'^b'));  // rows whose name starts with "b"
```

Use `filterViewContains` instead when you only need to read the results. It
skips copying, so it's faster.

## Stacking tables

```dart
final newcomers = DataFrame.fromColumns({
  'age': [41],
  'name': ['edsger'],
});

final everyone = people.concat(newcomers); // same column names required
```

## Good to know

Decimal numbers are stored with about 7 digits of precision to save memory and
time. Very large whole numbers, such as timestamps, are kept exactly.
