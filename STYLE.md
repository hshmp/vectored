# Style Guide

Follow [Effective Dart](https://dart.dev/effective-dart) for base guide.

## Code Blocks

Give breathing room and space between code blocks and other code.

### BAD

```dart
int a() {
  return 0;
}
int b() {
  return 1;
}
int x = 0;
for (int i = 0; i < 1; i++) {
  x += a();
}
if (x) {
  print(x);
}
```

### GOOD

```dart
int a() {
  return 0;
}

int b() {
  return 1;
}

int x = 0;

for (int i = 0; i < 1; i++) {
  x += a();
}

if (x) {
  print(x);
}
```

Group logically similar lines and separate others when order doesn't affect performance.

```dart
var q = Queue();

var dist = List<int>.filled(n, -1);

q.addLast(node);
dist[node] = 0;
```

## Comments

### Single

For single line `//` comments:
- concise
- no decorative elements
- bullet-point text not sentences
- only use capital letters to match identifiers or acronyms
- minimal punctuation
- symbols over conjuctions
- short comments on the end of same line
- before first line if explaining a group of lines

### Functions

Only functions should have detailed sentences describing what they do. Use `///` [documentation comments](https://dart.dev/language/comments) before functions. Parameters and types are referenced like `[param]` in comments.