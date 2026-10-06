import 'package:test/test.dart';
import 'package:vectored/vectored.dart';

void main() {
  group('TextSeries substring matching', () {
    test('standard substrings', () {
      final series = TextSeries.fromStrings('logs', [
        '2026-09-16 [INFO] System boot ok',
        '2026-09-16 [ERROR] Disk space critical',
        '2026-09-16 [WARN] Network latency high',
        '2026-09-16 [ERROR] Out of memory crash',
      ]);

      final matches = series.contains('[ERROR]');
      expect(matches[0], equals(0));
      expect(matches[1], equals(1));
      expect(matches[2], equals(0));
      expect(matches[3], equals(1));
      expect(matches.sum(), equals(2));
    });

    test('empty pattern matches all rows', () {
      final series = TextSeries.fromStrings('data', ['foo', 'bar', '']);
      final matches = series.contains('');
      expect(matches[0], equals(1));
      expect(matches[1], equals(1));
      expect(matches[2], equals(1));
    });

    test('pattern longer than row', () {
      final series = TextSeries.fromStrings('data', ['tiny']);
      final matches = series.contains('very_long_pattern');
      expect(matches[0], equals(0));
    });

    test('periodic pattern', () {
      final series = TextSeries.fromStrings('data', [
        'AAABAA',
        'BAAABAAA',
        'BAAABA',
        'ABABAB',
        'CCCCC',
      ]);

      final m1 = series.contains('AABAA');
      expect(m1[0], equals(1));
      expect(m1[1], equals(1)); // B A [A A B A A] A
      expect(m1[2], equals(0)); // no aabaa

      final m2 = series.contains('ABAB');
      expect(m2[3], equals(1));

      final m3 = series.contains('CCC');
      expect(m3[4], equals(1));
    });

    test('pathological repeated characters', () {
      final series = TextSeries.fromStrings('data', [
        'AAAAAAAAAAAAAAAAAAAA',
        'AAAAAAAABAAAAAAAAAAA',
      ]);

      final matches = series.contains('BAAA');
      expect(matches[0], equals(0));
      expect(matches[1], equals(1));
    });

    test('matches core search on repeated-prefix and utf8 inputs', () {
      final rows = [
        'A' * 4096,
        '${'A' * 2048}B${'A' * 2047}',
        '🙂β🙂β',
        'caf\u00e9',
        '',
      ];
      final patterns = [
        'A' * 64,
        '${'A' * 63}B',
        '${'A'}B${'A' * 62}',
        '🙂β',
        '\u00e9',
        '',
        'missing',
      ];
      final series = TextSeries.fromStrings('data', rows);

      for (final pattern in patterns) {
        final matches = series.contains(pattern);
        final indices = series.containingIndices(pattern);
        for (int row = 0; row < rows.length; row++) {
          final expected = rows[row].contains(pattern);
          expect(matches[row], equals(expected ? 1 : 0));
        }
        expect(indices, [
          for (int row = 0; row < rows.length; row++)
            if (rows[row].contains(pattern)) row,
        ]);
      }
    });

    test('matches core search exhaustively for short binary strings', () {
      final words = <String>[''];
      for (int length = 1; length <= 6; length++) {
        for (int mask = 0; mask < (1 << length); mask++) {
          words.add(String.fromCharCodes([
            for (int bit = 0; bit < length; bit++)
              ((mask >> bit) & 1) == 0 ? 65 : 66,
          ]));
        }
      }
      final series = TextSeries.fromStrings('binary', words);

      for (final pattern in words) {
        final matches = series.contains(pattern);
        final indices = series.containingIndices(pattern);
        for (int row = 0; row < words.length; row++) {
          final expected = words[row].contains(pattern);
          expect(matches[row], equals(expected ? 1 : 0));
        }
        expect(indices, [
          for (int row = 0; row < words.length; row++)
            if (words[row].contains(pattern)) row,
        ]);
      }
    });

    test('RegExp search returns a per-row match mask', () {
      final rows = ['item 42', 'no digits', 'value 7.5', ''];
      final series = TextSeries.fromStrings('data', rows);
      final matches = series.match(RegExp(r'\d+'));

      expect(
        [for (int i = 0; i < matches.length; i++) matches[i]],
        equals([1, 0, 1, 0]),
      );
    });

    test('patterns exceeding 256 bytes', () {
      final target = 'a' * 300;
      final textWith = 'header_${'a' * 300}_tail';
      final textWithout = 'header_${'a' * 299}_b_tail';

      final series = TextSeries.fromStrings('data', [textWith, textWithout]);
      final matches = series.contains(target);

      expect(matches[0], equals(1));
      expect(matches[1], equals(0));
    });

    test('element access via operator []', () {
      final series =
          TextSeries.fromStrings('words', ['alpha', 'beta', 'gamma']);
      expect(series[0], equals('alpha'));
      expect(series[1], equals('beta'));
      expect(series[2], equals('gamma'));
      expect(series.byteLength(0), equals(5));
    });
  });

  test('m = 1 single byte patterns work correctly', () {
    final series =
        TextSeries.fromStrings('data', ['apple', 'banana', 'cherry', '']);

    final matchA = series.contains('a');
    expect(matchA[0], equals(1)); // apple
    expect(matchA[1], equals(1)); // banana
    expect(matchA[2], equals(0)); // cherry
    expect(matchA[3], equals(0)); // empty

    final matchComma = series.contains(',');
    expect(matchComma.sum(), equals(0));
  });

  test('m = 2 two-byte patterns work correctly with 2-byte guard', () {
    final series = TextSeries.fromStrings('data', ['in', 'pin', 'out']);
    final matches = series.contains('in');

    expect(matches[0], equals(1));
    expect(matches[1], equals(1));
    expect(matches[2], equals(0));
  });
}
