import 'package:test/test.dart';
import 'package:vectored/vectored.dart';

void main() {
  group('slice-aware tensor kernels', () {
    test('int slices sum only their window', () {
      final tensor = IntTensor.fromList([1, 2, 3, 4, 5]);
      expect(tensor.slice(0, 2).sum(), equals(3));
      expect(tensor.slice(2, 5).sum(), equals(12));
    });

    test('float slices sum and bound only their window', () {
      final tensor = FloatTensor.fromList([1, 2, 3, 4, 5, 6, 7, 8, 9]);
      expect(tensor.slice(0, 2).sum(), equals(3));
      expect(tensor.slice(1, 7).sum(), equals(27));
      expect(tensor.slice(2, 5).bounds, equals((3.0, 5.0)));
    });

    test('allocating ops on slices match list reference', () {
      final a = FloatTensor.fromList([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      final b = FloatTensor.fromList([10, 20, 30, 40, 50, 60, 70, 80, 90, 1]);
      final left = a.slice(0, 6);
      final right = b.slice(3, 9);
      expect(_floats(left + right), equals([41, 52, 63, 74, 85, 96]));
      expect(_floats(right - left), equals([39, 48, 57, 66, 75, 84]));
      expect(_floats(left * 2.0), equals([2, 4, 6, 8, 10, 12]));
      expect(_floats(left * right), equals([40, 100, 180, 280, 400, 540]));

      final ints = IntTensor.fromList([1, 2, 3, 4, 5, 6, 7]);
      final intLeft = ints.slice(0, 3);
      final intRight = ints.slice(4, 7);
      expect(_ints(intLeft + intRight), equals([6, 8, 10]));
      expect(_ints(intRight - intLeft), equals([4, 4, 4]));
      expect(_ints(intLeft * intRight), equals([5, 12, 21]));
    });

    test('in-place ops on slices leave the rest of the buffer alone', () {
      final floats = FloatTensor.fromList([1, 2, 3, 4, 5, 6]);
      floats.slice(1, 4).sub_(FloatTensor.fromList([1, 1, 1]));
      floats.slice(4, 6).scale_(10);
      expect(_floats(floats), equals([1, 1, 2, 3, 50, 60]));

      final ints = IntTensor.fromList([1, 2, 3, 4, 5]);
      ints.slice(2, 4).sub_(IntTensor.fromList([1, 1]));
      expect(_ints(ints), equals([1, 2, 2, 3, 5]));
    });

    test('mismatched lengths throw in release builds', () {
      expect(
        () => FloatTensor.fromList([1, 2]) + FloatTensor.fromList([1]),
        throwsArgumentError,
      );
      expect(
        () => IntTensor.fromList([1, 2]).add_(IntTensor.fromList([1])),
        throwsArgumentError,
      );
    });
  });

  group('release-mode validation', () {
    test('series reject mismatched index lengths', () {
      expect(
        () => Series.fromInts('x', [1, 2], index: ['a']),
        throwsArgumentError,
      );
      expect(
        () => Series.fromStrings('x', ['a'], index: ['a', 'b']),
        throwsArgumentError,
      );
      expect(
        () => Series.fromObjects('x', [null], index: []),
        throwsArgumentError,
      );
    });

    test('tensor factories reject invalid arguments', () {
      expect(() => FloatTensor.linspace(0, 1, 1), throwsArgumentError);
      expect(() => FloatTensor.arange(0, 1, 0), throwsArgumentError);
    });

    test('large float sums stay close to the float64 reference', () {
      final values = List<double>.filled(1000000, 0.1);
      final reference = values.fold(0.0, (sum, v) => sum + v.toDouble());
      final tensor = FloatTensor.fromList(values);
      expect(tensor.sum(), closeTo(reference, reference * 1e-5));
    });
  });

  group('DataFrame regressions', () {
    test('uncached sum over chunked owned columns is correct', () {
      final left = DataFrame.fromColumns({
        'x': [1, 2, 3],
      }, chunkSize: 2);
      final right = DataFrame.fromSeries([
        Series.fromInts('x', [0]),
      ]);
      expect(left.concat(right).sum('x'), equals(6));
      expect(left.concat(right).mean('x'), equals(1.5));
    });

    test('deleting non-integral float rows keeps stats consistent', () {
      final frame = DataFrame.fromColumns({
        'x': [0.1, 0.2, 0.3],
      });
      frame.deleteRow(0);
      expect(frame.length, equals(2));
      expect(frame.min('x'), equals(frame['x'][0]));
      expect(frame.sum('x'), closeTo(0.5, 1e-6));
    });

    test('appended float rows can be deleted', () {
      final frame = DataFrame.empty(['x']);
      frame.appendRow({'x': 0.1});
      frame.appendRow({'x': 0.7});
      frame.deleteRow(1);
      expect(frame.max('x'), equals(frame['x'][0]));
    });

    test('large integers survive storage and deletion', () {
      final frame = DataFrame.fromColumns({
        'ts': [1700000000000, 1700000000001, 5],
      });
      expect(frame['ts'][0], equals(1700000000000));
      frame.deleteRow(0);
      expect(frame.min('ts'), equals(5));
      expect(frame.max('ts'), equals(1700000000001));
      expect(frame.sum('ts'), equals(1700000000006));
    });

    test('empty string columns are not treated as numeric', () {
      final frame = DataFrame.fromColumns({
        'name': <String>[],
        'value': <int>[],
      });
      frame.appendRow({'name': 'a', 'value': 1});
      expect(() => frame.sum('name'), throwsArgumentError);
      expect(frame.sum('value'), equals(1));
    });

    test('string search skips nulls instead of throwing', () {
      final frame = DataFrame.fromColumns({
        'name': ['alpha', null, 'beta', 'alphabet'],
        'id': [1, 2, 3, 4],
      });
      expect(
        frame.mask((c) => c('name').contains('alpha')).toList(),
        equals([true, false, false, true]),
      );
      expect(
        frame.mask((c) => c('name').contains(RegExp('^b'))).toList(),
        equals([false, false, true, false]),
      );
      expect(frame.filter((c) => c('name').contains('alpha')).sum('id'),
          equals(5));
      expect(
          frame.view((c) => c('name').contains('alpha')).sum('id'), equals(5));
      expect(
        frame
            .view((c) => c('name').contains('a'))
            .filter((c) => c('name').contains('bet')),
        hasLength(2),
      );
      expect(
        () => frame.mask((c) => c('id').contains('1')),
        throwsArgumentError,
      );
    });

    test('filtered stats include every selected chunk', () {
      final frame = DataFrame.fromColumns({
        'tag': ['a', 'a', 'a', 'a'],
        'value': [1, 2, 3, 4],
      }, chunkSize: 2);
      frame.appendRow({'tag': 'a', 'value': 5000000000});
      final filtered = frame.filter((c) => c('tag').contains('a'));
      expect(filtered.sum('value'), equals(5000000010));
      expect(filtered.max('value'), equals(5000000000));
    });

    test('compact preserves column types and values', () {
      final frame = DataFrame.fromColumns({
        'label': ['a', 'b', 'c', 'd', 'e'],
        'value': [1, 2, 3, 4, 5],
        'score': [0.5, 1.5, 2.5, 3.5, 4.5],
      }, chunkSize: 2);
      frame.deleteRow(1);
      frame.appendRow({'label': 'f', 'value': 6, 'score': 5.5});
      frame.compact();
      expect(frame.physicalLength, equals(5));
      expect(frame['label'].toList(), equals(['a', 'c', 'd', 'e', 'f']));
      expect(frame['value'].toList(), equals([1, 3, 4, 5, 6]));
      expect(frame.sum('score'), closeTo(16.5, 1e-9));
      expect(frame.mask((c) => c('label').contains('f')).count, equals(1));
    });
  });
}

List<double> _floats(FloatTensor tensor) =>
    [for (var i = 0; i < tensor.length; i++) tensor[i]];

List<int> _ints(IntTensor tensor) =>
    [for (var i = 0; i < tensor.length; i++) tensor[i]];
