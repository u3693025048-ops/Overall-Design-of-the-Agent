/**
 * check_render.js — 资料库 HTML 运行时质检（无需浏览器）
 *
 * 在 Node 里执行页面内嵌的 JS（用极简 DOM stub 隔离副作用），
 * 对真实内容跑渲染器 md() / py()，捕获静态校验看不见的运行时 bug。
 *
 * 用法：
 *   node check_render.js <library.html>
 *
 * 检查项：
 *   1. JS 语法 / 初始化无异常
 *   2. 每个文件渲染不抛异常
 *   3. 无残留 \u0000 占位符（stash 与 restore 失配）
 *   4. 无 undefined / [object Object] 泄漏（NaN 是合法正文内容，不算）
 *   5. strong 标签栈深度不错位（LaTeX \* 被误当 emphasis 会导致整体错位）
 *   6. strong/em/li/table/tr/td/ol/ul 开闭配平，span 开闭配平
 *   7. \* 占位符已还原
 *   8. 定向抽查（存在才查）：公式块、索引表格、Python 高亮、CSV 转表
 */
const fs = require('fs');

const file = process.argv[2];
if (!file) { console.error('用法：node check_render.js <library.html>'); process.exit(2); }
if (!fs.existsSync(file)) { console.error('文件不存在：' + file); process.exit(2); }

const doc = fs.readFileSync(file, 'utf8');

// ---------- 抽取主脚本（</div><script> 之后那段） ----------
const marker = '</div><script>\n';
const s = doc.indexOf(marker);
if (s < 0) { console.error('未找到主脚本（</div><script>）'); process.exit(1); }
let js = doc.slice(s + marker.length);
const e = js.indexOf('</script>');
if (e < 0) { console.error('主脚本未闭合'); process.exit(1); }
js = js.slice(0, e);

let fail = 0;
// 注意：console.log 只在多参数时才解析 %d，这里统一先拼好字符串
function chk(cond, msg, extra) {
  const line = msg + (extra === undefined ? '' : ' | ' + extra);
  console.log((cond ? '  PASS  ' : '  FAIL  ') + line);
  if (!cond) fail++;
}

console.log('='.repeat(64));
console.log('  运行时质检：' + file.split('/').pop());
console.log('='.repeat(64));

// ---------- 1. 语法 ----------
try {
  new Function(js);
  chk(true, 'JS 语法检查通过', js.length + ' 字符');
} catch (err) {
  chk(false, 'JS 语法错误：' + err.message);
  process.exit(1);
}

// ---------- 2. 沙箱初始化 ----------
function mkNode() {
  return {
    className: '', textContent: '', innerHTML: '', style: {}, dataset: {},
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    appendChild() {}, insertBefore() {}, scrollIntoView() {},
    set onclick(f) {}, get onclick() { return null; },
    set oninput(f) {}, get oninput() { return null; },
  };
}
const sandbox = {
  document: {
    querySelector: () => mkNode(),
    querySelectorAll: () => [],
    createElement: () => mkNode(),
  },
  location: { hash: '' },
  window: { MathJax: null },
  setTimeout: () => 0,
  clearTimeout: () => {},
  console,
};

let api;
try {
  api = new Function(...Object.keys(sandbox),
    js + '\n;return {md, py, esc, FILES, TREE};')(...Object.values(sandbox));
  chk(true, '脚本初始化无运行时错误');
} catch (err) {
  chk(false, '初始化抛出异常：' + err.message);
  console.log(err.stack.split('\n').slice(0, 4).join('\n'));
  process.exit(1);
}

const { md, py, FILES } = api;
const renderable = FILES.filter((f) => f.kind === 'md' || f.kind === 'py');
console.log('  ----  渲染 ' + renderable.length + ' 个可渲染文件'
  + '（共 ' + FILES.length + ' 项，图片/JSON 不渲染）  ----');

// ---------- 3. 逐文件渲染 ----------
const stats = { h1: 0, tables: 0, pre: 0, tex: 0, chars: 0 };
let rendered = 0;

for (const f of renderable) {
  let out = '';
  try {
    out = (f.kind === 'md') ? md(f.text) : '<pre><code>' + py(f.text) + '</code></pre>';
  } catch (err) {
    chk(false, f.path + ' 渲染异常', err.message.slice(0, 100));
    continue;
  }
  rendered++;

  // 3a. stash 占位符必须全部回填
  const leftover = (out.match(/\u0000(\d+)\u0000/g) || []).length;
  if (leftover) chk(false, f.path + ' 残留未回填占位符', leftover + ' 个');

  // 3b. 异常标记（NaN / Inf 是正文中合法的数值术语）
  if (/undefined|\[object Object\]/.test(out)) {
    const m = out.match(/.{0,40}(undefined|\[object Object\]).{0,40}/);
    chk(false, f.path + ' 输出含异常标记',
      '…' + m[0].replace(/\n/g, ' ') + '…');
  }

  // 3c. strong 栈深度：不能用 /<\/strong>[^<]*<strong>/ 判断，
  //     连续的合法加粗（A**…**B**…**）也会命中该模式。
  let depth = 0, misplaced = 0;
  const stre = /<\/?strong>/g;
  let sm;
  while ((sm = stre.exec(out))) {
    if (sm[0] === '<strong>') depth++;
    else if (depth === 0) misplaced++;
    else depth--;
  }
  if (depth !== 0) misplaced++;                  // 存在未闭合
  if (misplaced) {
    chk(false, f.path + ' 强调标签错位',
      misplaced + ' 处（多为 LaTeX \\* 被误当 emphasis）');
  }

  // 3d. 开闭配平
  for (const tag of ['strong', 'em', 'li', 'table', 'tr', 'td', 'ol', 'ul']) {
    const o = (out.match(new RegExp('<' + tag + '[ >]', 'g')) || []).length;
    const c = (out.match(new RegExp('</' + tag + '>', 'g')) || []).length;
    if (o !== c) {
      chk(false, f.path + ' <' + tag + '> 不配平', 'open=' + o + ' close=' + c);
    }
  }
  const so = (out.match(/<span [^>]*>/g) || []).length;
  const sc = (out.match(/<\/span>/g) || []).length;
  if (so !== sc) chk(false, f.path + ' <span> 不配平', 'open=' + so + ' close=' + sc);

  // 3e. LaTeX \* 占位符必须还原
  if (/\u0001/.test(out)) chk(false, f.path + ' 残留未还原的 \\* 占位符');

  stats.h1 += (out.match(/<h1>/g) || []).length;
  stats.tables += (out.match(/<table>/g) || []).length;
  stats.pre += (out.match(/<pre>/g) || []).length;
  stats.tex += (out.match(/class="tex"/g) || []).length;
  stats.chars += out.length;
}

if (fail === 0) {
  chk(true, '全部 ' + rendered + ' 个文件渲染成功',
    '无占位符残留 / 无泄漏 / 无标签错位');
}

console.log('\n  渲染统计：');
console.log('    H1 标题   ' + stats.h1);
console.log('    表格      ' + stats.tables);
console.log('    公式块    ' + stats.tex);
console.log('    代码块    ' + stats.pre);
console.log('    输出合计  ' + (stats.chars / 1024).toFixed(0) + ' KB');

// ---------- 4. 定向抽查（存在才查） ----------
console.log('\n  ----  定向抽查 ----');

const fml = FILES.find((f) => /formulas\.md$/.test(f.path));
if (fml) {
  const o = md(fml.text);
  const n = (o.match(/class="tex"/g) || []).length;
  chk(n > 0, '公式集渲染出 ' + n + ' 个公式块');
  chk(!/&lt;div/.test(o), '公式块未被二次转义');
}

const idx = FILES.find((f) => /00_INDEX\.md$/.test(f.path));
if (idx) {
  const o = md(idx.text);
  const t = (o.match(/<table>/g) || []).length;
  chk(t >= 1, '资产总索引渲染出 ' + t + ' 张表');
}

// 取体积最大的 py 文件做抽查（最有代表性），且只在源码确实含有
// 该语法结构时才断言——否则极简文件（如 def f(): pass）会误报。
const pyFiles = FILES.filter((f) => f.kind === 'py');
if (pyFiles.length) {
  const pyf = pyFiles.slice().sort((a, b) => b.text.length - a.text.length)[0];
  const o = py(pyf.text);
  console.log('  抽查文件：' + pyf.path + '（' + pyf.text.length + ' 字符）');
  if (/def\s/.test(pyf.text)) {
    chk(o.includes('<span class="k">def</span>'), 'Python 关键字高亮生效');
  }
  if (/#[^\n]/.test(pyf.text)) {
    chk(o.includes('<span class="c">'), 'Python 注释高亮生效');
  }
  if (/['"]/.test(pyf.text)) {
    chk(o.includes('<span class="s">'), 'Python 字符串高亮生效');
  }
  // 三引号 docstring 必须整体成为一个字符串 token（不能被拆坏）
  if (/"""|'''/.test(pyf.text)) {
    chk(!/&quot;{3}/.test(o) || o.includes('<span class="s">"""'),
      '三引号 docstring 高亮完整');
  }
}

const csv = FILES.find((f) => /\.csv$/.test(f.path));
if (csv) {
  const o = md(csv.text);
  chk((o.match(/<table>/g) || []).length === 1, 'CSV 渲染为单张表');
}

console.log('\n' + '='.repeat(64));
console.log(fail === 0 ? '  结论：运行时质检全部通过 ✅'
  : '  结论：' + fail + ' 项失败 ❌');
console.log('='.repeat(64));
process.exit(fail === 0 ? 0 : 1);
