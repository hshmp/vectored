import 'package:test/test.dart';
import 'package:vectored/vectored.dart';

void main() {
  test('renders an optional separator between related result rows', () {
    final table = ConsoleTable(headers: ['implementation', 'latency']);
    table
      ..addRow(['baseline', '1.00 ms'])
      ..addRow(['optimized', '0.50 ms'])
      ..addSeparator()
      ..addRow(['next baseline', '2.00 ms']);

    final separators = table
        .toString()
        .split('\n')
        .where((line) => line.startsWith('├'))
        .length;
    expect(separators, equals(2));
  });
}
