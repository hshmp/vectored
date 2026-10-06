import 'dart:io';

import 'package:test/test.dart';

// result comments checked against toString; e.g. `x; // [a, b]`
final _resultLine = RegExp(r'^(\s*)(.+?);\s*//\s*([\[{\d-].*?)\s*$');

void main() {
  test(
    'README dart examples run and print what they claim',
    () async {
      final readme = File('README.md').readAsStringSync();
      final blocks = RegExp(r'```dart\r?\n(.*?)```', dotAll: true)
          .allMatches(readme)
          .map((match) => match.group(1)!)
          .toList();

      expect(blocks, isNotEmpty, reason: 'README has no dart code blocks');

      final source = _buildProgram(blocks);
      final dir = Directory.systemTemp.createTempSync('vectored_readme_');
      addTearDown(() => dir.deleteSync(recursive: true));
      final file = File('${dir.path}/readme_examples.dart')
        ..writeAsStringSync(source);

      final packages = File('.dart_tool/package_config.json').absolute.path;
      final result = await Process.run(
        Platform.resolvedExecutable,
        ['run', '--packages=$packages', file.path],
      );

      expect(
        result.exitCode,
        isZero,
        reason: 'README examples failed:\n${result.stdout}${result.stderr}\n'
            '--- generated program ---\n$source',
      );
    },
    timeout: const Timeout(Duration(minutes: 2)),
  );
}

/// Joins README [blocks] into one runnable program.
///
/// Imports are hoisted to the top, blocks run in order inside `main` so later
/// examples can reuse earlier variables, and result comments become checks.
String _buildProgram(List<String> blocks) {
  final imports = <String>{};
  final body = StringBuffer();

  for (final block in blocks) {
    for (final line in block.split('\n')) {
      final trimmed = line.trimRight();
      if (trimmed.startsWith('import ')) {
        imports.add(trimmed);
        continue;
      }

      final result = _resultLine.firstMatch(trimmed);
      if (result == null) {
        body.writeln('  $trimmed');
      } else {
        final expected = result.group(3)!.replaceAll("'", r"\'");
        body.writeln(
            "  ${result.group(1)}_check(${result.group(2)}, '$expected');");
      }
    }
  }

  return [
    ...imports,
    '',
    'void main() {',
    body.toString(),
    '}',
    '',
    'void _check(Object? actual, String expected) {',
    "  if ('\$actual' != expected) {",
    "    throw StateError('README claims \$expected but got \$actual');",
    '  }',
    '}',
  ].join('\n');
}
