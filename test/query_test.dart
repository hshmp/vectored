import 'package:test/test.dart';
import 'package:vectored/vectored.dart';

void main() {
  group('Bitset', () {
    test('sets, counts and lists flags across word boundaries', () {
      final bits = Bitset(70)
        ..set(0)
        ..set(31)
        ..set(32)
        ..set(69);
      expect(bits.count, equals(4));
      expect(bits.toIndices(), equals([0, 31, 32, 69]));
      expect(bits.indices.toList(), equals([0, 31, 32, 69]));
      expect(bits[31], isTrue);
      expect(bits[33], isFalse);
      bits.clear(31);
      expect(bits.count, equals(3));
    });

    test('invert keeps unused tail bits clear', () {
      final bits = Bitset(5)..set(1);
      expect((~bits).toIndices(), equals([0, 2, 3, 4]));
      expect(Bitset.filled(33, true).count, equals(33));
    });

    test('combines with and, or and andNot', () {
      final a = Bitset.fromBools([true, true, false, false]);
      final b = Bitset.fromBools([true, false, true, false]);
      expect((a & b).toList(), equals([true, false, false, false]));
      expect((a | b).toList(), equals([true, true, true, false]));
      expect(a.copy().andNot(b).toList(), equals([false, true, false, false]));
      expect(() => a.and(Bitset(3)), throwsArgumentError);
    });
  });

  group('nullable series', () {
    test('typed series keep nulls as flags', () {
      final ints = Series.fromInts('x', [1, null, 3]);
      expect(ints, isA<IntSeries>());
      expect(ints.nullCount, equals(1));
      expect(ints.valueAt(1), isNull);
      expect(ints.sum(), equals(4));
      expect(ints.mean(), equals(2.0));

      final floats = Series.fromFloats('y', [null, -2.5, 4.0]);
      expect(floats.bounds, equals((-2.5, 4.0)));
      expect((floats + Series.fromFloats('z', [1, 1, null])).nullCount, 2);

      final text = Series.fromStrings('s', ['a', null]);
      expect(text.data, equals(['a', null]));
      expect(text.take([1, 0]).data, equals([null, 'a']));
    });

    test('bool series pack flags and nulls', () {
      final flags = Series.fromBools('f', [true, null, false, true]);
      expect(flags.trueCount, equals(2));
      expect(flags.valueAt(1), isNull);
      expect(flags.take([3, 1]).valueAt(1), isNull);
    });

    test('frames store nullable columns as typed series', () {
      final frame = DataFrame.fromColumns({
        'n': [1, null, 3],
        'f': [true, null, false],
        's': ['a', 'b', null],
      });
      expect(frame.toSeries('n'), isA<IntSeries>());
      expect(frame.toSeries('f'), isA<BoolSeries>());
      expect(frame.toSeries('s'), isA<StringSeries>());
      expect(frame['n'].toList(), equals([1, null, 3]));
      expect(frame.rowAt(1), equals({'n': null, 'f': null, 's': 'b'}));
      expect(frame.count('n'), equals(2));
      expect(frame.mean('n'), equals(2.0));
      expect(frame.min('n'), equals(1));

      frame.deleteRow(1);
      expect(frame.sum('n'), equals(4));
      frame.compact();
      expect(frame['s'].toList(), equals(['a', null]));
    });

    test('views skip nulls in totals', () {
      final frame = DataFrame.fromColumns({
        'tag': ['x', 'x', 'y'],
        'v': [2, null, 6],
      });
      final view = frame.view((c) => c('tag').eq('x'));
      expect(view.sum('v'), equals(2));
      expect(view.mean('v'), equals(2.0));
      expect(view['v'].toList(), equals([2, null]));
    });
  });

  group('conditions', () {
    late DataFrame people;

    setUp(() {
      people = DataFrame.fromColumns({
        'name': ['ada', 'grace', 'linus', 'barbara', null],
        'age': [36, 45, 29, 52, null],
        'score': [0.1, 0.5, 0.9, 0.3, 0.7],
        'admin': [true, false, false, true, null],
      }, chunkSize: 2);
    });

    List<Object?> names(DataFrame frame) => frame['name'].toList();

    test('text matching by literal, regex and anchors', () {
      expect(names(people.filter((c) => c('name').contains('a'))),
          equals(['ada', 'grace', 'barbara']));
      expect(names(people.filter((c) => c('name').contains(RegExp(r'a$')))),
          equals(['ada', 'barbara']));
      expect(names(people.filter((c) => c('name').startsWith('gr'))),
          equals(['grace']));
      expect(names(people.filter((c) => c('name').endsWith('us'))),
          equals(['linus']));
      expect(names(people.filter((c) => c('name').eq('ada'))), equals(['ada']));
    });

    test('a pattern variable swaps literal and regex search', () {
      List<Object?> search(Pattern pattern) =>
          names(people.filter((c) => c('name').contains(pattern)));
      expect(search('ra'), equals(['grace', 'barbara']));
      expect(search(RegExp(r'^[gl]')), equals(['grace', 'linus']));
    });

    test('numeric comparisons on ints and floats', () {
      expect(names(people.filter((c) => c('age').gt(40))),
          equals(['grace', 'barbara']));
      expect(names(people.filter((c) => c('age').lte(36))),
          equals(['ada', 'linus']));
      expect(names(people.filter((c) => c('age').between(29, 36))),
          equals(['ada', 'linus']));
      // stored float32 0.1 is not greater than 0.1
      expect(people.filter((c) => c('score').gt(0.1)).length, equals(4));
      expect(people.filter((c) => c('score').eq(0.3)).length, equals(1));
      expect(people.filter((c) => c('score').gte(0.5)).length, equals(3));
    });

    test('combining with and, or and not', () {
      expect(
        names(people.filter((c) =>
            c('age').gt(30) & c('name').contains('a') & ~c('admin').isTrue)),
        equals(['grace']),
      );
      expect(
        names(
            people.filter((c) => c('name').eq('ada') | c('name').eq('linus'))),
        equals(['ada', 'linus']),
      );
    });

    test('nulls, booleans, membership and custom tests', () {
      expect(people.filter((c) => c('name').isNull).length, equals(1));
      expect(people.filter((c) => c('age').isNotNull).length, equals(4));
      expect(people.filter((c) => c('age').eq(null)).length, equals(1));
      expect(names(people.filter((c) => c('admin').isTrue)),
          equals(['ada', 'barbara']));
      expect(names(people.filter((c) => c('admin').isFalse)),
          equals(['grace', 'linus']));
      expect(names(people.filter((c) => c('name').neq('ada'))),
          equals(['grace', 'linus', 'barbara']));
      expect(names(people.filter((c) => c('age').isIn([29, 52]))),
          equals(['linus', 'barbara']));
      expect(people.filter((c) => c('name').isIn(['ada', null])).length,
          equals(2));
      expect(
        names(people.filter(
            (c) => c('name').satisfies((v) => (v as String).length == 5))),
        equals(['grace', 'linus']),
      );
    });

    test('mask, view and filter agree and skip deleted rows', () {
      people.deleteRow(0);
      Condition hasA(Columns c) => c('name').contains('a');
      expect(people.mask(hasA).toList(), equals([true, false, true, false]));
      expect(people.view(hasA).length, equals(2));
      expect(names(people.filter(hasA)), equals(['grace', 'barbara']));
      expect(
        people.view(hasA).filter((c) => c('age').gt(50))['name'].toList(),
        equals(['barbara']),
      );
    });

    test('rejects unknown columns and mismatched types', () {
      expect(() => people.filter((c) => c('nope').eq(1)), throwsArgumentError);
      expect(() => people.filter((c) => c('age').contains('3')),
          throwsArgumentError);
      expect(() => people.filter((c) => c('name').gt(3)), throwsArgumentError);
      expect(() => people.filter((c) => c('age').isTrue), throwsArgumentError);
    });

    test('filter results keep typed columns and cached totals', () {
      final adults = people.filter((c) => c('age').gte(36));
      expect(adults.toSeries('age'), isA<IntSeries>());
      expect(adults.toSeries('admin'), isA<BoolSeries>());
      expect(adults.sum('age'), equals(133));
      expect(adults.max('age'), equals(52));
    });
  });

  group('groupBy', () {
    late DataFrame sales;

    setUp(() {
      sales = DataFrame.fromColumns({
        'region': ['north', 'south', 'north', 'east', 'south', null],
        'rep': ['ann', 'bob', 'ann', 'cid', 'dee', 'eve'],
        'units': [10, 4, null, 7, 6, 1],
        'price': [2.5, 3.0, 1.5, 4.0, 2.0, 1.0],
      }, chunkSize: 4);
    });

    test('aggregates per group in first-seen order', () {
      final result = sales.groupBy(['region']).agg((g) => [
            g.count(),
            g.sum('units'),
            g.mean('units'),
            g.min('price'),
            g.max('price'),
          ]);
      expect(result.columns, [
        'region',
        'count',
        'units_sum',
        'units_mean',
        'price_min',
        'price_max',
      ]);
      expect(
          result['region'].toList(), equals(['north', 'south', 'east', null]));
      expect(result['count'].toList(), equals([2, 2, 1, 1]));
      expect(result['units_sum'].toList(), equals([10, 10, 7, 1]));
      expect(result['units_mean'].toList(), equals([10.0, 5.0, 7.0, 1.0]));
      expect(result['price_min'].toList(), equals([1.5, 2.0, 4.0, 1.0]));
      expect(result['price_max'].toList(), equals([2.5, 3.0, 4.0, 1.0]));
    });

    test('groups by several keys with first, last and custom names', () {
      final result = sales.groupBy(['region', 'rep']).agg((g) => [
            g.count(as: 'n'),
            g.first('units', as: 'firstUnits'),
            g.last('price'),
          ]);
      expect(result.length, equals(5));
      expect(
        result.rowAt(0),
        equals({
          'region': 'north',
          'rep': 'ann',
          'n': 2,
          'firstUnits': 10,
          'price_last': 1.5,
        }),
      );
    });

    test('skips deleted rows and works on filtered frames', () {
      sales.deleteRow(0);
      final counts = sales.groupBy(['region']).count();
      // first row was north; south is now seen first
      expect(
          counts['region'].toList(), equals(['south', 'north', 'east', null]));
      expect(counts['count'].toList(), equals([2, 1, 1, 1]));

      final south = sales
          .filter((c) => c('region').eq('south'))
          .groupBy(['rep']).agg((g) => [g.sum('units')]);
      expect(south['units_sum'].toList(), equals([4, 6]));
    });

    test('rejects bad requests', () {
      expect(() => sales.groupBy([]), throwsArgumentError);
      expect(() => sales.groupBy(['nope']), throwsArgumentError);
      expect(
        () => sales.groupBy(['region']).agg((g) => [g.sum('nope')]),
        throwsArgumentError,
      );
      expect(
        () => sales.groupBy(['region']).agg((g) => [g.sum('rep')]),
        throwsArgumentError,
      );
      expect(
        () => sales.groupBy(['region']).agg((g) => [g.count(), g.count()]),
        throwsArgumentError,
      );
    });
  });
}
