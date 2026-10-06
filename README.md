# Vectored

Fast data tables for Dart and Flutter. If you've used pandas, you already know
most of it, and you don't need Python or a server to use it.

## Why Vectored

- **Fast:** filtering, searching and totals are much quicker than plain Dart
  lists, and quicker than Pandas in early tests.
- **Familiar:** `sum`, `mean`, `filter`, `groupBy` and `concat` work the way
  they do in Pandas.
- **Easy to write:** filters are plain Dart, so your editor autocompletes
  them and catches typos before you run anything.
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
| Group by | ✅ | ❌ | ✅ | ✅ |
| Missing values | ✅ | ❌ | ✅ | ✅ |
| Instant totals while editing | ✅ | ❌ | ❌ | ❌ |
| Runs inside a Flutter app | ✅ | ✅ | ❌ | ❌ |
| Joins, CSV import | Planned | ❌ | ✅ | ✅ |

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

## Filtering

Describe the rows you want, and type `c('column').` to see every option.

```dart
final withA = people.filter((c) => c('name').contains('a'));
withA['name'].toList(); // [ada, barbara]

final over30B = people.filter((c) => c('age').gt(30) & c('name').startsWith('b'));
over30B['name'].toList(); // [barbara]
```

`contains` takes plain text or a `RegExp`, so you can switch between them
without changing your code:

```dart
final Pattern search = RegExp(r'^l'); // or just 'l'
people.filter((c) => c('name').contains(search))['name'].toList(); // [linus]
```

Combine conditions with `&` (and), `|` (or) and `~` (not). Use `view` instead
of `filter` when you only need to read the results. It skips copying, so it's
faster.

```dart
people.view((c) => c('age').gt(30)).sum('age'); // 88
```

## Grouping

```dart
final staff = DataFrame.fromColumns({
  'team': ['web', 'app', 'web', 'app', 'web'],
  'salary': [70, 80, 90, 60, 50],
});

final byTeam = staff.groupBy(['team']).agg((g) => [g.count(), g.mean('salary')]);
byTeam.rowAt(0); // {team: web, count: 3, salary_mean: 70.0}
```

`g.` offers `count`, `sum`, `mean`, `min`, `max`, `first` and `last`.

## Missing values and yes/no columns

Leave a value as `null` when it's unknown. Totals and averages skip it.

```dart
final tasks = DataFrame.fromColumns({
  'done': [true, false, null],
  'hours': [2, null, 5],
});

tasks.mean('hours'); // 3.5
tasks.filter((c) => c('done').isTrue).length; // 1
tasks.filter((c) => c('hours').isNull).length; // 1
```

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
