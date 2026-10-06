import 'package:test/test.dart';
import 'package:vectored/vectored.dart';

void main() {
  group('Series Column Abstraction', () {
    test('FloatSeries delegates to SIMD FloatTensor', () {
      final s1 = Series.fromFloats('price', [10.0, 20.0, 30.0]);
      final s2 = Series.fromFloats('tax', [1.5, 2.5, 3.5]);

      final total = s1 + s2;
      expect(total.name, equals('price'));
      expect(total[0], closeTo(11.5, 1e-5));
      expect(total.sum(), closeTo(67.5, 1e-5));
    });

    test('StringSeries handles categorical indexing', () {
      final categories = Series.fromStrings(
        'status',
        ['active', 'pending', 'inactive'],
        index: ['user_1', 'user_2', 'user_3'],
      );

      expect(categories.length, equals(3));
      expect(categories[1], equals('pending'));
      expect(categories.index?[0], equals('user_1'));
    });
  });
}
