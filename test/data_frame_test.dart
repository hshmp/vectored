import 'package:test/test.dart';
import 'package:vectored/vectored.dart';

void main() {
  group('DataFrame', () {
    test('stores existing series and supports fast string filtering and sums',
        () {
      final names = Series.fromStrings(
        'name',
        ['alpha', 'beta', 'alphabet', 'gamma'],
      );
      final values = Series.fromFloats('value', [1, 2, 3, 4]);
      final counts = Series.fromInts('count', [10, 20, 30, 40]);
      final frame = DataFrame.fromSeries([names, values, counts]);

      expect(frame.toSeries('name'), isA<StringSeries>());
      expect(frame.sum('value'), equals(10.0));
      expect(frame.sum('count'), equals(100));

      final filtered = frame.filterContains('name', 'alpha');
      expect(filtered.length, equals(2));
      expect(filtered['name'].toList(), equals(['alpha', 'alphabet']));
      expect(filtered.sum('count'), equals(40));
    });

    test('materialized filters preserve typed columns and update extrema', () {
      final frame = DataFrame.fromColumns({
        'label': ['hit-one', 'miss', 'hit-two', 'hit-three'],
        'integer': [5, 100, 2, 9],
        'decimal': [1.25, 8.5, 3.75, 9.0],
      }, chunkSize: 2);

      final filtered = frame.filterContains('label', 'hit');

      expect(filtered['label'].toList(), ['hit-one', 'hit-two', 'hit-three']);
      expect(filtered.toSeries('integer'), isA<IntSeries>());
      expect(filtered.toSeries('decimal'), isA<FloatSeries>());
      expect(filtered.sum('integer'), 16);
      expect(filtered.min('integer'), 2);
      expect(filtered.max('integer'), 9);

      filtered.deleteRow(1);
      expect(filtered.sum('integer'), 14);
      expect(filtered.min('integer'), 5);
      expect(filtered.max('integer'), 9);
    });

    test('maskless filter views use row pointers and can be materialized', () {
      final frame = DataFrame.fromColumns({
        'label': ['hit-a', 'skip', 'hit-b', 'hit-c'],
        'value': [3, 50, 4, 7],
      }, chunkSize: 2);

      final view = frame
          .filterViewContains('label', 'hit')
          .filterContains('label', '-b');

      expect(view.length, 1);
      expect(view['value'].toList(), [4]);
      expect(view.sum('value'), 4);
      expect(view.mean('value'), 4);
      expect(view.rowAt(0), {'label': 'hit-b', 'value': 4});

      frame.compact();
      expect(view['label'].toList(), ['hit-b']);
      final materialized = view.materialize();
      expect(materialized['label'].toList(), ['hit-b']);
      expect(materialized['value'].toList(), [4]);
      expect(materialized.toSeries('value'), isA<IntSeries>());
    });

    test('shares existing series and observes later series mutations', () {
      final ids = Series.fromInts('id', [1, 2]);
      final frame = DataFrame.fromSeries([ids]);

      ids.tensor[0] = 9;

      expect(frame['id'].toList(), equals([9, 2]));
      expect(frame.sum('id'), equals(11));
      expect(frame.min('id'), equals(2));
    });

    test('aggregates chunk series and excludes deleted rows', () {
      final frame = DataFrame.fromColumns(
        {
          'value': [1, 2, 3, 4, 5],
          'label': ['a', 'b', 'a', 'b', 'a'],
        },
        chunkSize: 2,
      );

      frame.deleteRow(0);
      frame.deleteRow(1);

      expect(frame.sum('value'), equals(11));
      final matchMask = frame.contains('label', 'a');
      expect(
        [for (var i = 0; i < matchMask.length; i++) matchMask[i]],
        equals([0, 0, 1]),
      );
      expect(
          frame.filterContains('label', 'b')['value'].toList(), equals([2, 4]));
    });

    test('caches numeric aggregates across appends, deletes, and compaction',
        () {
      final frame = DataFrame.fromColumns(
        {
          'value': [8, 2, 5]
        },
        chunkSize: 2,
      );

      expect(frame.sum('value'), equals(15));
      expect(frame.count('value'), equals(3));
      expect(frame.min('value'), equals(2));
      expect(frame.max('value'), equals(8));
      expect(frame.mean('value'), closeTo(5, 1e-12));

      frame.appendRow({'value': 1});
      expect(frame.sum('value'), equals(16));
      expect(frame.count('value'), equals(4));
      expect(frame.min('value'), equals(1));
      expect(frame.max('value'), equals(8));
      expect(frame.mean('value'), equals(4));

      frame.deleteRow(3);
      expect(frame.sum('value'), equals(15));
      expect(frame.count('value'), equals(3));
      expect(frame.min('value'), equals(2));
      expect(frame.max('value'), equals(8));
      expect(frame.mean('value'), equals(5));

      frame.deleteRow(0);
      expect(frame.sum('value'), equals(7));
      expect(frame.count('value'), equals(2));
      expect(frame.min('value'), equals(2));
      expect(frame.max('value'), equals(5));
      expect(frame.mean('value'), equals(3.5));

      frame.compact();
      expect(frame.sum('value'), equals(7));
      expect(frame.count('value'), equals(2));
      expect(frame.min('value'), equals(2));
      expect(frame.max('value'), equals(5));
      expect(frame.mean('value'), equals(3.5));
    });

    test('cached aggregates merge on concat and empty extrema are null', () {
      final left = DataFrame.fromColumns({
        'value': [1, 9]
      });
      final right = DataFrame.fromColumns({
        'value': [3, 7]
      });
      right.deleteRow(0);

      final joined = left.concat(right);
      expect(joined.sum('value'), equals(17));
      expect(joined.count('value'), equals(3));
      expect(joined.min('value'), equals(1));
      expect(joined.max('value'), equals(9));
      expect(joined.mean('value'), closeTo(17 / 3, 1e-10));

      final empty = DataFrame.empty(['value']);
      expect(empty.count('value'), isZero);
      expect(empty.min('value'), isNull);
      expect(empty.max('value'), isNull);
      expect(empty.mean('value'), isNull);
    });

    test('extrema frequencies retain duplicate values after deletion', () {
      final frame = DataFrame.fromColumns({
        'value': [2, 2, 5],
      });

      frame.deleteRow(0);
      expect(frame.min('value'), equals(2));
      expect(frame.max('value'), equals(5));
      frame.deleteRow(0);
      expect(frame.min('value'), equals(5));
      expect(frame.max('value'), equals(5));
      expect(frame.mean('value'), equals(5));
    });

    test('recomputes aggregates safely for non-finite numeric values', () {
      final frame = DataFrame.fromColumns({
        'value': [double.infinity, 1.0],
      });

      frame.deleteRow(0);
      expect(frame.sum('value'), equals(1.0));
      expect(frame.count('value'), equals(1));
      expect(frame.min('value'), equals(1.0));
      expect(frame.max('value'), equals(1.0));
      expect(frame.mean('value'), equals(1));
    });

    test('reads columns lazily and appends rows', () {
      final frame = DataFrame.fromColumns(
        {
          'id': [1, 2],
          'name': ['alpha', 'beta'],
        },
        chunkSize: 2,
      );

      expect(frame.length, equals(2));
      expect(frame['name'][1], equals('beta'));
      expect(frame.rowAt(0), equals({'id': 1, 'name': 'alpha'}));

      frame.appendRow({'id': 3, 'name': 'gamma'});

      expect(frame.length, equals(3));
      expect(frame['id'].toList(), equals([1, 2, 3]));
      expect(frame.rowAt(2), equals({'id': 3, 'name': 'gamma'}));
    });

    test('buffers appended rows by column and flushes in order', () {
      final frame = DataFrame.empty(['id', 'name'], chunkSize: 3);
      final first = <String, Object?>{'id': 1, 'name': 'alpha'};
      frame.appendRow(first);
      first['name'] = 'changed';
      frame.appendRows([
        {'id': 2, 'name': 'beta'},
        {'id': 3, 'name': 'gamma'},
      ]);

      expect(frame.physicalLength, 3);
      expect(frame.physicalChunkCount, 1);
      expect(frame['id'].toList(), [1, 2, 3]);
      expect(frame['name'].toList(), ['alpha', 'beta', 'gamma']);
      expect(frame.physicalChunkCount, 1);
    });

    test('appends batches, matches regular expressions, and filters masks', () {
      final frame = DataFrame.fromRows(
        [
          {'id': 1, 'name': 'alpha'},
          {'id': 2, 'name': 'beta'},
          {'id': 3, 'name': 'gamma'},
        ],
        columns: ['id', 'name'],
        chunkSize: 2,
      );
      frame.deleteRow(1);
      frame.appendRows([
        {'id': 4, 'name': 'delta'},
        {'id': 5, 'name': 'omega'},
      ]);

      final mask = frame.match('name', RegExp(r'^(a|g|d)'));
      expect(
        [for (var i = 0; i < mask.length; i++) mask[i]],
        equals([1, 1, 1, 0]),
      );
      expect(
        frame.filter(mask).rowAt(2),
        equals({'id': 4, 'name': 'delta'}),
      );
      expect(
        frame.filterMatch('name', RegExp(r'^o')).rowAt(0),
        equals({'id': 5, 'name': 'omega'}),
      );
    });

    test('rejects invalid masks, rows, and non-numeric sums', () {
      final frame = DataFrame.fromColumns({
        'id': [1, 2],
        'name': ['alpha', 'beta'],
      });

      expect(
        () => frame.filter(IntTensor.fromList([1])),
        throwsArgumentError,
      );
      expect(
        () => frame.appendRows([
          {'id': 3, 'name': 'gamma'},
          {'id': 4},
        ]),
        throwsArgumentError,
      );
      expect(() => frame.sum('name'), throwsArgumentError);
      expect(() => frame.sum('unknown'), throwsArgumentError);
      expect(() => frame.contains('id', '1'), throwsArgumentError);
    });

    test('empty filtered frames do not cache non-numeric columns as sums', () {
      final frame = DataFrame.fromColumns({
        'id': [1, 2],
        'name': ['alpha', 'beta'],
      });

      final empty = frame.filterContains('name', 'missing');

      expect(empty.length, isZero);
      expect(empty.sum('id'), equals(0));
      expect(() => empty.sum('name'), throwsArgumentError);
    });

    test('deletes rows lazily and compacts on request', () {
      final frame = DataFrame.fromColumns(
        {
          'id': [1, 2, 3, 4],
          'value': ['a', 'b', 'c', 'd'],
        },
        chunkSize: 2,
      );

      frame.deleteRow(1);
      expect(frame.length, equals(3));
      expect(frame.physicalLength, equals(4));
      expect(frame['id'].toList(), equals([1, 3, 4]));

      frame.compact();
      expect(frame.length, equals(3));
      expect(frame.physicalLength, equals(3));
      expect(frame['value'].toList(), equals(['a', 'c', 'd']));
    });

    test('concatenates by sharing sealed chunks without aliasing mutations',
        () {
      final left = DataFrame.fromColumns(
        {
          'id': [1, 2],
          'value': ['a', 'b'],
        },
        chunkSize: 4,
      );
      final right = DataFrame.fromColumns(
        {
          'value': ['c'],
          'id': [3],
        },
        chunkSize: 4,
      );

      final joined = left.concat(right);
      left.appendRow({'id': 4, 'value': 'd'});
      right.deleteRow(0);

      expect(joined['id'].toList(), equals([1, 2, 3]));
      expect(left['id'].toList(), equals([1, 2, 4]));
      expect(right['id'].toList(), isEmpty);
    });

    test('self-concatenation keeps repeated rows independently deletable', () {
      final frame = DataFrame.fromColumns({
        'id': [1, 2],
      });
      frame.deleteRow(0);

      final doubled = frame.concat(frame);
      expect(doubled.length, equals(2));
      expect(doubled['id'].toList(), equals([2, 2]));

      doubled.deleteRow(0);
      expect(doubled['id'].toList(), equals([2]));
      expect(frame['id'].toList(), equals([2]));
    });

    test('copies existing delete markers when concatenating', () {
      final left = DataFrame.fromColumns({
        'id': [1, 2],
      });
      final right = DataFrame.fromColumns({
        'id': [3, 4],
      });
      left.deleteRow(0);
      right.deleteRow(1);

      final joined = left.concat(right);
      expect(joined['id'].toList(), equals([2, 3]));

      joined.deleteRow(0);
      expect(joined['id'].toList(), equals([3]));
      expect(left['id'].toList(), equals([2]));
      expect(right['id'].toList(), equals([3]));
    });

    test('validates schemas, row keys, and delete indices', () {
      final frame = DataFrame.empty(['id', 'value']);

      expect(
        () => frame.appendRow({'id': 1}),
        throwsArgumentError,
      );
      expect(
        () => frame.concat(DataFrame.empty(['id', 'other'])),
        throwsArgumentError,
      );
      expect(() => frame.deleteRow(0), throwsRangeError);
      expect(
        () => DataFrame.fromColumns({
          'a': [1],
          'b': [1, 2],
        }),
        throwsArgumentError,
      );
    });
  });
}
