enum Alignment { left, right, center }

/// console table formatter
class ConsoleTable {
  final List<String> headers;
  final List<List<String>> _rows = [];
  final Set<int> _separators = {};
  final List<Alignment>? alignments;

  ConsoleTable({required this.headers, this.alignments});

  /// Adds a row of values to the table and converts them to strings.
  void addRow(List<Object?> row) {
    _rows.add(row.map((e) => e?.toString() ?? '').toList());
  }

  /// Adds a horizontal rule after the most recently added row.
  void addSeparator() {
    if (_rows.isNotEmpty) _separators.add(_rows.length);
  }

  /// Computes the maximum width needed for each column.
  List<int> _computeColumnWidths() {
    final widths = List<int>.generate(headers.length, (i) => headers[i].length);

    for (final row in _rows) {
      for (int i = 0; i < headers.length; i++) {
        if (i < row.length && row[i].length > widths[i]) {
          widths[i] = row[i].length;
        }
      }
    }
    return widths;
  }

  /// Infers the alignment for column [colIndex] when no override is set.
  Alignment _getAlignment(int colIndex) {
    if (alignments != null && colIndex < alignments!.length) {
      return alignments![colIndex];
    }
    // numeric -> right
    if (_rows.isNotEmpty && colIndex < _rows[0].length) {
      final sample = _rows[0][colIndex].replaceAll(RegExp(r'[,%x\s]|ms'), '');
      if (num.tryParse(sample) != null) return Alignment.right;
    }
    return Alignment.left;
  }

  String _alignCell(String text, int width, Alignment alignment) {
    final padding = width - text.length;
    if (padding <= 0) return text;

    switch (alignment) {
      case Alignment.left:
        return text.padRight(width);
      case Alignment.right:
        return text.padLeft(width);
      case Alignment.center:
        final leftPad = padding ~/ 2;
        final rightPad = padding - leftPad;
        return ' ' * leftPad + text + ' ' * rightPad;
    }
  }

  /// Renders the table as a boxed string for the console.
  String render() {
    final widths = _computeColumnWidths();
    final buffer = StringBuffer();

    // border helpers
    String buildSeparator(String left, String mid, String right, String fill) {
      final segments = widths.map((w) => fill * (w + 2)); // +2 pad
      return '$left${segments.join(mid)}$right';
    }

    buffer.writeln(buildSeparator('┌', '┬', '┐', '─'));

    buffer.write('│');
    for (int i = 0; i < headers.length; i++) {
      final cell = _alignCell(headers[i], widths[i], Alignment.center);
      buffer.write(' $cell │');
    }
    buffer.writeln();

    buffer.writeln(buildSeparator('├', '┼', '┤', '─'));

    for (var rowIndex = 0; rowIndex < _rows.length; rowIndex++) {
      final row = _rows[rowIndex];
      buffer.write('│');
      for (int i = 0; i < headers.length; i++) {
        final text = i < row.length ? row[i] : '';
        final align = _getAlignment(i);
        final cell = _alignCell(text, widths[i], align);
        buffer.write(' $cell │');
      }
      buffer.writeln();
      if (_separators.contains(rowIndex + 1)) {
        buffer.writeln(buildSeparator('├', '┼', '┤', '─'));
      }
    }

    buffer.write(buildSeparator('└', '┴', '┘', '─'));

    return buffer.toString();
  }

  @override
  String toString() => render();
}
