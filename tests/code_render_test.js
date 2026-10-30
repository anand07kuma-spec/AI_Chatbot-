// Regression test: code-block rendering in static/app.js
// Run: node tests/code_render_test.js   (or via pytest: tests/test_code_render.py)
//
// Guards against the bug where sequential highlight passes re-scanned their own
// <span class="..."> output and produced broken markup like
//   <span <span class="tok-kw">class</span>="tok-num">5</span>
// which browsers rendered as literal HTML-token text inside code blocks.
'use strict';

const fs = require('fs');
const path = require('path');
const assert = require('assert');

const src = fs.readFileSync(path.join(__dirname, '..', 'static', 'app.js'), 'utf8');
const start = src.indexOf('function escapeHtml');
const end = src.indexOf('function buildDocSourcesEl');
assert.ok(start >= 0 && end > start, 'app.js rendering markers not found');

global.window = { location: { origin: 'http://x' } };
const { formatAssistantHtml } = new Function(
  src.slice(start, end) + '\nreturn { escapeHtml, highlightCode, formatAssistantHtml };'
)();

// ---------- helpers ----------
function stripTokens(html) {
  return html
    .replace(/<span class="tok-[a-z]+">/g, '')
    .replace(/<\/span>/g, '')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&amp;/g, '&'); // must be last
}

function codeBlocks(html) {
  const re = /<pre class="code-block">[\s\S]*?<code[^>]*>([\s\S]*?)<\/code><\/pre>/g;
  const out = [];
  let m;
  while ((m = re.exec(html)) !== null) out.push(m[1]);
  return out;
}

function dataCodes(html) {
  const re = /data-code="([^"]*)"/g;
  const out = [];
  let m;
  while ((m = re.exec(html)) !== null) out.push(decodeURIComponent(m[1]));
  return out;
}

function render(text) {
  return formatAssistantHtml(text, [], []);
}

let passed = 0;
function ok(name, fn) {
  fn();
  passed += 1;
  console.log('  ok - ' + name);
}

// ---------- samples ----------
const cSrc =
  '#include <stdio.h>\n' +
  '\n' +
  'int main(void) {\n' +
  '    int arr[5] = {1, 2, 3, 4, 5};\n' +
  '    int n = 5;\n' +
  '    if (n > 3) {\n' +
  '        printf("%d\\n", arr[0]);\n' +
  '    }\n' +
  '    scanf("%d", &n);\n' +
  '    return 0;\n' +
  '}';

const pySrc =
  'def greet(name):\n' +
  '    msg = "class return def"\n' +
  '    # say hello\n' +
  '    print(msg, 42, 3.14)\n' +
  '    return name';

const linkSrc = 'See [docs](https://example.com) for `literal [x](https://y)` text';

const htmlSrc = '<script>alert(1)</script> & <img src=x onerror=alert(1)>';

const twoSrc1 = 'int a = 1;';
const twoSrc2 = 'let b = 2;';

console.log('code_render_test.js');

// 1. corruption patterns must never appear (the original bug)
ok('no corrupted nested span markup in any sample', () => {
  const samples = [
    '```c\n' + cSrc + '\n```',
    '```python\n' + pySrc + '\n```',
    '```c\nint x = 5; /* <span class="tok-num">5</span> */\n```',
    '```\nint x = 5;\n```',
    '```c\n#include <stdio.h>\nint x = 5;',
  ];
  for (const s of samples) {
    const html = render(s);
    assert.ok(!/<span\s+<span/.test(html), 'nested <span <span in: ' + s.slice(0, 40));
    assert.ok(!/<\/span>=/.test(html), '</span>= attribute corruption in: ' + s.slice(0, 40));
    assert.ok(!/=[a-z-]*<span/.test(html), 'tag markup inside attribute in: ' + s.slice(0, 40));
  }
});

// 2. C code round-trips exactly (visible text == source)
ok('C sample: visible text equals original source', () => {
  const html = render('Here is the code:\n\n```c\n' + cSrc + '\n```\n\nDone.');
  const blocks = codeBlocks(html);
  assert.strictEqual(blocks.length, 1, 'expected exactly one code block');
  assert.strictEqual(stripTokens(blocks[0]), cSrc);
});

// 3. Copy button carries the RAW original source (not HTML-escaped)
ok('C sample: data-code decodes to exact original source', () => {
  const html = render('```c\n' + cSrc + '\n```');
  const codes = dataCodes(html);
  assert.strictEqual(codes.length, 1);
  assert.strictEqual(codes[0], cSrc, 'data-code must be raw source');
  assert.ok(!codes[0].includes('&lt;'), 'data-code must not be HTML-escaped');
});

// 4. tokens are actually produced (highlighting still works)
ok('C sample: keyword/number/comment tokens present', () => {
  const html = render('```c\n' + cSrc + '\n```');
  assert.ok(html.includes('<span class="tok-kw">void</span>'), 'kw void');
  assert.ok(html.includes('<span class="tok-kw">return</span>'), 'kw return');
  assert.ok(html.includes('<span class="tok-num">5</span>'), 'num 5');
  assert.ok(html.includes('<span class="tok-com">#include'), 'comment #include');
});

// 5. python: '#' comments, strings swallow keywords, numbers
ok('Python sample: tokens + exact round-trip', () => {
  const html = render('```python\n' + pySrc + '\n```');
  assert.ok(html.includes('<span class="tok-kw">def</span>'), 'kw def');
  assert.ok(html.includes('<span class="tok-com"># say hello</span>'), 'hash comment');
  assert.ok(
    html.includes('<span class="tok-str">&quot;class return def&quot;</span>'),
    'string swallowed keywords'
  );
  assert.ok(html.includes('<span class="tok-num">42</span>'), 'num 42');
  const blocks = codeBlocks(html);
  assert.strictEqual(stripTokens(blocks[0]), pySrc);
});

// 6. XSS stays escaped, inside code and in prose
ok('XSS payloads never become live HTML', () => {
  const html = render('Prose: ' + htmlSrc + '\n\n```html\n' + htmlSrc + '\n```');
  assert.ok(!/<script/i.test(html), 'no raw <script');
  assert.ok(!/<img/i.test(html), 'no raw <img');
  assert.ok(html.includes('&lt;script&gt;'), 'escaped in output');
  const blocks = codeBlocks(html);
  assert.strictEqual(stripTokens(blocks[0]), htmlSrc);
});

// 7. unterminated fence renders as a code block, not a paragraph
ok('unterminated fence becomes a code block', () => {
  const html = render('```c\n#include <stdio.h>\nint x = 5;');
  assert.ok(html.includes('<pre class="code-block">'), 'has code block');
  assert.ok(!html.includes('```'), 'no leftover fence markers');
  assert.ok(!html.includes('<p>'), 'not wrapped in a paragraph');
  assert.strictEqual(codeBlocks(html).length, 1);
});

// 8. no placeholder/stash control characters leak into the output
ok('no \\u0000/\\u0001 control chars in output', () => {
  const samples = [
    '```c\n' + cSrc + '\n```',
    '```python\n' + pySrc + '\n```',
    '```c\nint x = 5;\n```',
    'plain **text** and `code`',
  ];
  for (const s of samples) {
    const html = render(s);
    assert.ok(!html.includes('\u0000'), 'NUL placeholder leaked');
    assert.ok(!html.includes('\u0001'), 'stashed string leaked');
  }
});

// 9. <pre> is never nested inside <p>
ok('no <p><pre invalid nesting', () => {
  const samples = [
    'Intro\n\n```c\nint x = 1;\n```',
    'Intro ```c\nint x = 1;\n``` outro',
    '```c\nint x = 1;\n```\n\nOutro',
    'text ```\nunclosed\n',
  ];
  for (const s of samples) {
    assert.ok(!/<p><pre/.test(render(s)), 'found <p><pre for input: ' + JSON.stringify(s));
  }
});

// 10. markdown outside code still works
ok('headings/lists/links/bold still render', () => {
  const html = render(
    '# Title\n\n- item **bold**\n- second\n\n1. one\n2. two\n\n[click](https://example.com)\n\n> quoted'
  );
  assert.ok(html.includes('<h3>Title</h3>'), 'heading');
  assert.ok(html.includes('<ul><li>item <strong>bold</strong></li>'), 'ul + bold');
  assert.ok(html.includes('<ol><li>one</li><li>two</li>'), 'ol');
  assert.ok(html.includes('<a href="https://example.com"'), 'link');
  assert.ok(html.includes('<blockquote>quoted</blockquote>'), 'blockquote');
});

// 11. markdown links inside code stay literal code
ok('links inside code blocks are not converted', () => {
  const html = render('```text\n' + linkSrc + '\n```');
  const blocks = codeBlocks(html);
  assert.strictEqual(blocks.length, 1);
  assert.ok(!blocks[0].includes('<a '), 'no <a> inside code');
  assert.strictEqual(stripTokens(blocks[0]), linkSrc);
  assert.strictEqual(dataCodes(html)[0], linkSrc, 'link source stays literal');
});

// 12. prose escaping unchanged
ok('prose HTML is escaped', () => {
  const html = render('Use <b>bold</b> & "quotes" carefully.');
  assert.ok(!html.includes('<b>'), 'no raw <b>');
  assert.ok(html.includes('&lt;b&gt;bold&lt;/b&gt;'), 'escaped tags');
  assert.ok(html.includes('&quot;quotes&quot;'), 'escaped quotes');
});

// 13. multiple code blocks keep their own source
ok('two code blocks render independently', () => {
  const html = render('```c\n' + twoSrc1 + '\n```\n\nmid\n\n```js\n' + twoSrc2 + '\n```');
  const blocks = codeBlocks(html);
  const codes = dataCodes(html);
  assert.strictEqual(blocks.length, 2);
  assert.strictEqual(stripTokens(blocks[0]), twoSrc1);
  assert.strictEqual(stripTokens(blocks[1]), twoSrc2);
  assert.deepStrictEqual(codes, [twoSrc1, twoSrc2]);
});

// 14. copy button present in every block
ok('copy button rendered with data-code', () => {
  const html = render('```c\nint x = 1;\n```');
  assert.ok(html.includes('<button class="block-copy"'), 'button');
  assert.ok(/data-code="[^"]+"/.test(html), 'data-code attr');
  assert.ok(html.includes('title="Copy code"'), 'title');
});

// 15. legacy fence (no newline after ```) still renders as code
ok('legacy fence without newline renders as code', () => {
  const html = render('``` \nint x = 5;\n```');
  assert.ok(html.includes('<pre class="code-block">'), 'code block');
  assert.ok(!/<span\s+<span/.test(html), 'no corruption');
});

// 16. plain text with no fence is untouched by code rules
ok('fence-free text renders normally', () => {
  const html = render('Just a message with 42 numbers.');
  assert.ok(html.startsWith('<p>'), 'paragraph');
  assert.ok(!html.includes('<pre'), 'no code block');
});

console.log('\n' + passed + ' assertions groups passed');
