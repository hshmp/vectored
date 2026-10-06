# Vectored

**The data table that keeps up.** Add and delete rows whenever you like, and
your totals, averages, minimums and maximums are already up to date. You never
rebuild the table and never wait for a recount.

## Why Vectored

- **Truly editable tables:** add and delete rows in place. Other data table
  libraries rebuild the table on every change. Vectored just updates it.
- **Answers that are always ready:** `sum`, `mean`, `min`, `max` and `count`
  are kept current as you edit, so they come back instantly however big the
  table grows.
- **Zero-copy everything:** stacking two tables or viewing filtered rows
  shares the existing data instead of duplicating it.
- **Fast searches:** columns are packed tightly in memory, so filters and text
  search run several times faster than looping over lists, and faster than
  Pandas.
- **Familiar and easy:** if you know Pandas, you know Vectored. Filters are
  plain Dart, so your editor autocompletes them.
- **Runs everywhere:** pure Dart, so the same code runs on servers,
  command-line tools, desktop and mobile with nothing extra to install.

## How it compares

| | Vectored | Pandas | Polars | Matrix2D |
|---|---|---|---|---|
| Add and delete rows without rebuilding | ✅ | ❌ | ❌ | ❌ |
| Totals stay current as you edit | ✅ | ❌ | ❌ | ❌ |
| Stack tables without copying | ✅ | ❌ | ✅ | ❌ |
| Group by | ✅ | ✅ | ✅ | ❌ |
| Text search and filtering | ✅ | ✅ | ✅ | ❌ |
| Missing values | ✅ | ✅ | ✅ | ❌ |
| Same code on server, desktop and mobile | ✅ | ❌ | ❌ | ✅ |

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

## Live editing with instant totals

This is what makes Vectored different. Edit the table in place, and every
total is already correct the moment you ask for it.

```dart
people.appendRow({'age': 52, 'name': 'barbara'});
people.deleteRow(1); // grace leaves

people.count('age'); // 3
people.sum('age');   // 117
people.mean('age');  // 39.0
people.min('age');   // 29
people.max('age');   // 52
```

None of these calls scans the table. Vectored keeps the answers up to date on
every edit, so they're just as fast with a million rows as with three. After
lots of deletes, call `people.compact()` now and then to free up memory.

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
everyone.sum('age'); // 158
```

Stacking is instant: the new table shares the rows of both originals instead
of copying them, and edits to one table never leak into another.

## Good to know

Decimal numbers are stored with about 7 digits of precision to save memory and
time. Very large whole numbers, such as timestamps, are kept exactly.
