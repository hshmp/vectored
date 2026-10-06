import 'package:matrix2d/matrix2d.dart';
import 'package:test/test.dart';

void main() {
  test('Matrix2D sum supports the 1D numeric column benchmark input', () {
    const matrix = Matrix2d();
    expect(matrix.sum([1, 2, 3, 4]), equals(10));
    expect(
        matrix.sum([
          [1, 2],
          [3, 4],
        ]),
        equals(10));
  });
}
