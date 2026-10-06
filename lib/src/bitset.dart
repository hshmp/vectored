import 'dart:typed_data';

/// A fixed-length set of boolean flags packed 32 to a word.
///
/// Used for boolean columns, null markers, deleted rows and filter results.
/// Word-level operations such as [and], [or] and [count] process 32 rows at
/// a time.
final class Bitset {
  /// The number of flags in this set.
  final int length;

  final Uint32List _words;

  /// Creates a bitset of [length] flags, all cleared.
  Bitset(this.length) : _words = Uint32List(_wordCount(length)) {
    RangeError.checkNotNegative(length, 'length');
  }

  /// Creates a bitset of [length] flags, all set to [value].
  factory Bitset.filled(int length, bool value) {
    final bits = Bitset(length);
    if (value) bits.invert();
    return bits;
  }

  /// Creates a bitset whose flags match [values].
  factory Bitset.fromBools(List<bool> values) {
    final bits = Bitset(values.length);
    for (var i = 0; i < values.length; i++) {
      if (values[i]) bits.set(i);
    }
    return bits;
  }

  Bitset._(this.length, this._words);

  static int _wordCount(int length) => (length + 31) >> 5;

  /// Returns whether flag [index] is set.
  @pragma('vm:prefer-inline')
  bool operator [](int index) =>
      (_words[index >> 5] & (1 << (index & 31))) != 0;

  /// Sets or clears flag [index].
  void operator []=(int index, bool value) => value ? set(index) : clear(index);

  /// Sets flag [index].
  @pragma('vm:prefer-inline')
  void set(int index) => _words[index >> 5] |= 1 << (index & 31);

  /// Clears flag [index].
  @pragma('vm:prefer-inline')
  void clear(int index) => _words[index >> 5] &= ~(1 << (index & 31));

  /// The number of set flags.
  int get count {
    var total = 0;
    for (var i = 0; i < _words.length; i++) {
      total += _popCount(_words[i]);
    }
    return total;
  }

  /// Whether any flag is set.
  bool get any {
    for (var i = 0; i < _words.length; i++) {
      if (_words[i] != 0) return true;
    }
    return false;
  }

  /// Returns an independent copy of this bitset.
  Bitset copy() => Bitset._(length, Uint32List.fromList(_words));

  /// Keeps only flags also set in [other]; returns this bitset.
  Bitset and(Bitset other) {
    _requireSameLength(other);
    for (var i = 0; i < _words.length; i++) {
      _words[i] &= other._words[i];
    }
    return this;
  }

  /// Adds every flag set in [other]; returns this bitset.
  Bitset or(Bitset other) {
    _requireSameLength(other);
    for (var i = 0; i < _words.length; i++) {
      _words[i] |= other._words[i];
    }
    return this;
  }

  /// Clears every flag set in [other]; returns this bitset.
  Bitset andNot(Bitset other) {
    _requireSameLength(other);
    for (var i = 0; i < _words.length; i++) {
      _words[i] &= ~other._words[i];
    }
    return this;
  }

  /// Flips every flag; returns this bitset.
  Bitset invert() {
    for (var i = 0; i < _words.length; i++) {
      _words[i] = ~_words[i];
    }
    _clearTail();
    return this;
  }

  /// Returns a new bitset with flags set in both this and [other].
  Bitset operator &(Bitset other) => copy().and(other);

  /// Returns a new bitset with flags set in either this or [other].
  Bitset operator |(Bitset other) => copy().or(other);

  /// Returns a new bitset with every flag flipped.
  Bitset operator ~() => copy().invert();

  /// The indices of set flags in ascending order.
  Iterable<int> get indices sync* {
    for (var w = 0; w < _words.length; w++) {
      var word = _words[w];
      while (word != 0) {
        final lowest = word & -word;
        yield (w << 5) + lowest.bitLength - 1;
        word ^= lowest;
      }
    }
  }

  /// Returns the indices of set flags in ascending order.
  List<int> toIndices() {
    final result = <int>[];
    for (var w = 0; w < _words.length; w++) {
      var word = _words[w];
      while (word != 0) {
        final lowest = word & -word;
        result.add((w << 5) + lowest.bitLength - 1);
        word ^= lowest;
      }
    }
    return result;
  }

  /// Returns the flags as a list of booleans.
  List<bool> toList() => [for (var i = 0; i < length; i++) this[i]];

  @override
  String toString() => toList().map((value) => value ? 1 : 0).join();

  // unused tail bits; keep zero so count stays exact
  void _clearTail() {
    final used = length & 31;
    if (used != 0) _words[_words.length - 1] &= (1 << used) - 1;
  }

  void _requireSameLength(Bitset other) {
    if (other.length != length) {
      throw ArgumentError.value(other.length, 'other', 'Length mismatch');
    }
  }

  static int _popCount(int word) {
    word = word - ((word >> 1) & 0x55555555);
    word = (word & 0x33333333) + ((word >> 2) & 0x33333333);
    word = (word + (word >> 4)) & 0x0F0F0F0F;
    return ((word * 0x01010101) & 0xFFFFFFFF) >> 24;
  }
}
