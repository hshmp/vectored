import 'dart:io';

const String _green = '\x1B[32m';
const String _yellow = '\x1B[33m';
const String _reset = '\x1B[0m';

void main(List<String> args) async {
  final root = Directory.current;
  final files = _findDartFiles(root);
  final touched = <String>[];
  final warnings = <String>[];

  for (final file in files) {
    final original = await file.readAsString();
    final updated = _formatFile(original, file.path, warnings);
    if (updated != original) {
      await file.writeAsString(updated);
      touched.add(file.path);
    }
  }

  if (warnings.isNotEmpty) {
    stdout.writeln('');
    stdout.writeln('${_yellow}warnings$_reset');
    for (final warning in warnings) {
      stdout.writeln('$_yellow- $warning$_reset');
    }
    stdout.writeln('');
  }

  final countLabel = touched.length == 1
      ? '1 file touched'
      : '${touched.length} files touched';
  stdout.writeln('$_green$countLabel$_reset');
  if (touched.isNotEmpty) {
    for (final file in touched) {
      stdout.writeln('$_green$file$_reset');
    }
  }
}

List<File> _findDartFiles(Directory root) {
  final candidates = <File>[];
  final ignored = {
    '.git',
    '.dart_tool',
    'build',
    '.idea',
    '.vscode',
  };

  final stack = <Directory>[root];
  while (stack.isNotEmpty) {
    final current = stack.removeLast();
    if (!current.existsSync()) continue;

    for (final entity in current.listSync(followLinks: false)) {
      if (entity is Directory) {
        final name = entity.uri.pathSegments.isNotEmpty
            ? entity.uri.pathSegments.last
            : entity.path;
        if (ignored.contains(name)) continue;
        stack.add(entity);
      } else if (entity is File && entity.path.endsWith('.dart')) {
        candidates.add(entity);
      }
    }
  }

  candidates.sort((a, b) => a.path.compareTo(b.path));
  return candidates;
}

String _formatFile(String source, String path, List<String> warnings) {
  final lines = source.replaceAll('\r\n', '\n').split('\n');
  final output = <String>[];

  for (var index = 0; index < lines.length; index++) {
    final line = lines[index];
    final trimmed = line.trim();

    if (trimmed.isEmpty) {
      if (output.isNotEmpty && output.last.isNotEmpty) {
        output.add('');
      }
      continue;
    }

    final next = _normalizeLine(line, path, index + 1, warnings);
    if (next != null) {
      output.add(next);
    }
  }

  final fixed = _stripEmptyRuns(output);
  if (fixed.isEmpty) return source;
  return '${fixed.join('\n')}\n';
}

String? _normalizeLine(
  String line,
  String path,
  int lineNo,
  List<String> warnings,
) {
  final trimmed = line.trim();

  if (trimmed.startsWith(' // /')){
    return line;
  }

  if (trimmed.startsWith('/*')) {
    warnings.add(
        '${_relativePath(path)}:$lineNo: block comment left unchanged; review manually');
    return line;
  }

  final trailingCommentMatch = RegExp(r'^(.*?)(\s*//.*)$').firstMatch(line);
  if (trailingCommentMatch != null) {
    final prefix = trailingCommentMatch.group(1)!;
    final commentRaw = trailingCommentMatch.group(2)!.trimLeft();
    final body = commentRaw.substring(2).trim();
    if (body.isEmpty) return line;

    if (_isDecorativeComment(body)) {
      warnings
          .add('${_relativePath(path)}:$lineNo: decorative comment removed');
      return prefix.trimRight();
    }

    final normalized = _normalizeComment(body);
    if (normalized != body) {
      return '$prefix // $normalized';
    }

    if (_isNarrativeComment(body)) {
      warnings.add(
          '${_relativePath(path)}:$lineNo: narrative comment may need manual review');
    }

    return line;
  }

  if (trimmed.startsWith(' // ')){
    final body = trimmed.substring(2).trim();
    if (body.isEmpty) return '';

    if (_isDecorativeComment(body)) {
      warnings
          .add('${_relativePath(path)}:$lineNo: decorative comment removed');
      return null;
    }

    final normalized = _normalizeComment(body);
    if (normalized != body) {
      final indent = line.substring(0, line.indexOf('//'));
      return '$indent// $normalized';
    }

    if (_isNarrativeComment(body)) {
      warnings.add(
          '${_relativePath(path)}:$lineNo: narrative comment may need manual review');
    }

    return line;
  }

  return line;
}

bool _isDecorativeComment(String text) {
  final t = text.trim();
  if (t.isEmpty) return true;
  if (RegExp(r'^[\-=_=#*]+$').hasMatch(t)) return true;
  if (RegExp(r'^[\-=_=#*]{3,}\s*$').hasMatch(t)) return true;
  if (RegExp(r'^(?:\d+\.?\s*)?(?:[A-Z0-9\s:/\[\]()\-]+)?$').hasMatch(t) &&
      t.contains(RegExp(r'[A-Z]{2,}')) &&
      (t.contains(':') ||
          t.contains('-') ||
          t.contains('=') ||
          t.contains('*'))) {
    return true;
  }
  return false;
}

String _normalizeComment(String text) {
  var value = text.trim();
  value = value.replaceAll(RegExp(r'\s*[\-_=#*]{3,}\s*'), ' ');
  value = value.replaceAll(RegExp(r'\s*:\s*'), '; ');
  value = value.replaceAll(RegExp(r'\s*\(\s*'), '(');
  value = value.replaceAll(RegExp(r'\s*\)\s*'), ')');
  value = value.replaceAll(RegExp(r'\s+'), ' ');

  value = value.replaceAllMapped(
    RegExp(r'\b[A-Z]{2,}\b'),
    (match) => match.group(0)!.toLowerCase(),
  );

  value = value.replaceAllMapped(
    RegExp(r'\b[A-Z][a-z]+\b'),
    (match) => match.group(0)!.toLowerCase(),
  );

  value = value.replaceAll(RegExp(r'\s*;\s*'), '; ');
  value = value.replaceAll(RegExp(r'\s+,\s*'), ', ');
  value = value.trim();
  return value;
}

bool _isNarrativeComment(String text) {
  final trimmed = text.trim();
  if (trimmed.isEmpty) return false;
  final low = trimmed.toLowerCase();
  if (low.contains(' and ') ||
      low.contains(' or ') ||
      low.contains(' but ') ||
      low.contains(' then ') ||
      low.contains(' this ') ||
      low.contains(' that ') ||
      low.contains(' should ') ||
      low.contains(' must ') ||
      low.contains(' always ') ||
      trimmed.contains('.') ||
      trimmed.contains('!') ||
      trimmed.contains('?')) {
    return true;
  }
  return false;
}

List<String> _stripEmptyRuns(List<String> lines) {
  final output = <String>[];
  var sawEmpty = false;

  for (final line in lines) {
    if (line.trim().isEmpty) {
      if (!sawEmpty && output.isNotEmpty) {
        output.add('');
      }
      sawEmpty = true;
      continue;
    }

    output.add(line);
    sawEmpty = false;
  }

  while (output.isNotEmpty && output.last.trim().isEmpty) {
    output.removeLast();
  }

  return output;
}

String _relativePath(String path) {
  final root = Directory.current.path;
  if (path.startsWith(root)) {
    return path.substring(root.length + 1);
  }
  return path;
}
