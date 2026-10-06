import 'package:test/test.dart';
import 'package:vectored/vectored.dart';

void main() {
  // float tensor tests
  group('FloatTensor (TensorF)', () {
    test('Factories: fromList, vector, and indexing', () {
      final t = FloatTensor.fromList([1.0, 2.0, 3.0, 4.0, 5.0]);
      expect(t.length, equals(5));
      expect(t.count, equals(5));
      expect(t.rank, equals(1));
      expect(t.shape, equals([5]));
      expect(t.strides, equals([1]));
      expect(t.offset, equals(0));

      expect(t[0], closeTo(1.0, 1e-5));
      expect(t[4], closeTo(5.0, 1e-5));

      t[2] = 99.5;
      expect(t[2], closeTo(99.5, 1e-5));
    });

    test('2D Matrix initialization, strides, and coordinates', () {
      final m = FloatTensor.zeros([2, 3]);
      expect(m.rank, equals(2));
      expect(m.length, equals(6));
      expect(m.strides, equals([3, 1]));

      // [1,2] -> idx 5
      m.setAt([1, 2], 42.0);
      expect(m.at([1, 2]), closeTo(42.0, 1e-5));
      expect(m[5], closeTo(42.0, 1e-5));
    });

    test('Generators: linspace and arange', () {
      final lin = FloatTensor.linspace(0.0, 10.0, 5);
      expect(lin.count, equals(5));
      expect(lin[0], closeTo(0.0, 1e-5));
      expect(lin[2], closeTo(5.0, 1e-5));
      expect(lin[4], closeTo(10.0, 1e-5));

      final ar = FloatTensor.arange(0.0, 5.0, 1.0);
      expect(ar.count, equals(5));
      expect(ar[0], closeTo(0.0, 1e-5));
      expect(ar[3], closeTo(3.0, 1e-5));
    });

    test('Pure allocating arithmetic (+, -, *)', () {
      final a = FloatTensor.fromList([10.0, 20.0, 30.0, 40.0]);
      final b = FloatTensor.fromList([1.0, 2.0, 3.0, 4.0]);

      final added = a + b;
      expect(added[0], closeTo(11.0, 1e-5));
      expect(added[3], closeTo(44.0, 1e-5));

      final subtracted = a - b;
      expect(subtracted[0], closeTo(9.0, 1e-5));
      expect(subtracted[3], closeTo(36.0, 1e-5));

      final scaled = a * 2.5;
      expect(scaled[0], closeTo(25.0, 1e-5));
      expect(scaled[3], closeTo(100.0, 1e-5));

      final multiplied = a * b;
      expect(multiplied[0], closeTo(10.0, 1e-5));
      expect(multiplied[3], closeTo(160.0, 1e-5));
    });

    test('In-place mutators (add_, sub_, scale_)', () {
      final a = FloatTensor.fromList([10.0, 20.0, 30.0]);
      final b = FloatTensor.fromList([1.0, 2.0, 3.0]);

      a.add_(b);
      expect(a[0], closeTo(11.0, 1e-5));
      expect(a[2], closeTo(33.0, 1e-5));

      a.sub_(b);
      expect(a[0], closeTo(10.0, 1e-5));
      expect(a[2], closeTo(30.0, 1e-5));

      a.scale_(0.5);
      expect(a[0], closeTo(5.0, 1e-5));
      expect(a[2], closeTo(15.0, 1e-5));
    });

    test('Statistical Reductions: sum, mean, mode, bounds', () {
      final a = FloatTensor.fromList([2.0, 4.0, 4.0, 4.0, 6.0]);
      expect(a.sum(), closeTo(20.0, 1e-5));
      expect(a.mean(), closeTo(4.0, 1e-5));
      expect(a.mode(), closeTo(4.0, 1e-5));

      final (min, max) = a.bounds;
      expect(min, closeTo(2.0, 1e-5));
      expect(max, closeTo(6.0, 1e-5));
    });

    test('Zero-Copy Slicing: Memory buffer is shared with parent', () {
      final parent = FloatTensor.fromList([0.0, 10.0, 20.0, 30.0, 40.0, 50.0]);
      final sub = parent.slice(2, 5); // view [20.0,30.0,40.0]

      expect(sub.length, equals(3));
      expect(sub.offset, equals(2));
      expect(sub[0], closeTo(20.0, 1e-5));
      expect(sub[2], closeTo(40.0, 1e-5));

      sub[1] = 999.0; // mutate view
      expect(parent[3], closeTo(999.0, 1e-5)); // zero-copy proof
    });

    test('Slice bounds checking', () {
      final t = FloatTensor.vector(10);
      expect(() => t.slice(-1, 5), throwsA(isA<AssertionError>()));
      expect(() => t.slice(3, 11), throwsA(isA<AssertionError>()));
      expect(() => t.slice(5, 2), throwsA(isA<AssertionError>()));
    });

    test('Loop-peeled SIMD execution on unaligned slice [1:15]', () {
      final a = FloatTensor.arange(0.0, 20.0, 1.0);
      final b = FloatTensor.arange(0.0, 20.0, 1.0);

      // start=1; unaligned
      final sliceA = a.slice(1, 15); // len=14; idx 1..14
      final sliceB = b.slice(1, 15);

      final result = sliceA + sliceB;
      expect(result.length, equals(14));

      // slice[0] => orig[1]; 1+1=2
      expect(result[0], closeTo(2.0, 1e-5));
      // slice[3] => orig[4]; 4+4=8
      expect(result[3], closeTo(8.0, 1e-5));
      // slice[13] => orig[14]; 14+14=28
      expect(result[13], closeTo(28.0, 1e-5));

      // unaligned sum
      expect(sliceA.sum(), closeTo(105.0, 1e-5)); // 1..14 sum=105
    });

    test('TensorF alias operates identically', () {
      final TensorF t = TensorF.fromList([1.0, 2.0, 3.0]);
      expect(t.sum(), closeTo(6.0, 1e-5));
    });
  });

  // int tensor tests
  group('IntTensor (TensorI)', () {
    test('Factories: fromList, vector, and indexing', () {
      final t = IntTensor.fromList([10, 20, 30, 40, 50]);
      expect(t.length, equals(5));
      expect(t.count, equals(5));
      expect(t.rank, equals(1));
      expect(t.shape, equals([5]));
      expect(t.strides, equals([1]));
      expect(t.offset, equals(0));

      expect(t[0], equals(10));
      expect(t[4], equals(50));

      t[2] = 999;
      expect(t[2], equals(999));
    });

    test('Allocating arithmetic (+, -, *)', () {
      final a = IntTensor.fromList([10, 20, 30, 40, 50]);
      final b = IntTensor.fromList([1, 2, 3, 4, 5]);

      final added = a + b;
      expect(added.buffer, equals([11, 22, 33, 44, 55]));

      final subtracted = a - b;
      expect(subtracted.buffer, equals([9, 18, 27, 36, 45]));

      final multiplied = a * b;
      expect(multiplied.buffer, equals([10, 40, 90, 160, 250]));
    });

    test('In-place mutators (add_, sub_)', () {
      final a = IntTensor.fromList([10, 20, 30, 40]);
      final b = IntTensor.fromList([1, 2, 3, 4]);

      a.add_(b);
      expect(a.buffer, equals([11, 22, 33, 44]));

      a.sub_(b);
      expect(a.buffer, equals([10, 20, 30, 40]));
    });

    test('Zero-Copy Slicing: Memory buffer is shared with parent', () {
      final parent = IntTensor.fromList([0, 10, 20, 30, 40, 50]);
      final sub = parent.slice(2, 5); // view [20,30,40]

      expect(sub.length, equals(3));
      expect(sub.offset, equals(2));
      expect(sub[0], equals(20));
      expect(sub[2], equals(40));

      sub[1] = 999; // mutate view
      expect(parent[3], equals(999)); // zero-copy proof
    });

    test('Slice bounds checking', () {
      final t = IntTensor.vector(10);
      expect(() => t.slice(-1, 5), throwsA(isA<AssertionError>()));
      expect(() => t.slice(3, 11), throwsA(isA<AssertionError>()));
      expect(() => t.slice(5, 2), throwsA(isA<AssertionError>()));
    });

    test('Loop-peeled SIMD execution on unaligned slice [1:15]', () {
      final a = IntTensor.fromList(List.generate(20, (i) => i));
      final b = IntTensor.fromList(List.generate(20, (i) => i * 10));

      final sliceA = a.slice(1, 15); // len=14; idx 1..14
      final sliceB = b.slice(1, 15);

      final result = sliceA + sliceB;
      expect(result.length, equals(14));

      // slice[0] => orig[1]; 1+10=11
      expect(result[0], equals(11));
      // slice[3] => orig[4]; 4+40=44
      expect(result[3], equals(44));
      // slice[13] => orig[14]; 14+140=154
      expect(result[13], equals(154));
    });

    test('Aggregations and O(1) tracking', () {
      // untracked
      final untracked = IntTensor.fromList([10, -5, 15, 20]);
      expect(untracked.sum(), equals(40));
      expect(untracked.mean(), equals(10.0));

 // tracked; o(1)delta
      final tracked = IntTensor.fromList([5, 10, 15], tracked: true);
      expect(tracked.sum(), equals(30));
      expect(tracked.mean(), equals(10.0));

      // mutate; 10 -> 30; delta +20
      tracked[1] = 30;
      expect(tracked.sum(), equals(50));
      expect(tracked.mean(), closeTo(50 / 3, 1e-5));

      // in-place add_ updates running sum
      final addend = IntTensor.fromList([1, 1, 1]);
      tracked.add_(addend); // [6,31,16] -> sum 53
      expect(tracked.sum(), equals(53));
    });

    test('TensorI alias operates identically', () {
      final TensorI t = TensorI.fromList([10, 20, 30]);
      expect(t.sum(), equals(60));
    });
  });
}
