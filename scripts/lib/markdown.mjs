// A small Markdown renderer for the project's own docs: headings, paragraphs, nested lists, tables,
// fenced code, block quotes, rules, and inline code, links, images, bold and italic. Text is escaped,
// raw HTML in the source is not passed through, and `rewriteHref` lets the caller map link targets.

export function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

export function slug(text) {
  return text.toLowerCase().replace(/`|\*|_/g, '').replace(/[^\p{L}\p{N}\s-]/gu, '').trim().replace(/\s+/g, '-');
}

function inline(text, opts) {
  const spans = [];
  // code spans first, so their content is left alone
  let s = text.replace(/`([^`]+)`/g, (_, c) => { spans.push(`<code>${escapeHtml(c)}</code>`); return `\u0000${spans.length - 1}\u0000`; });
  s = escapeHtml(s);
  s = s.replace(/!\[([^\]]*)\]\(([^)\s]+)(?:\s+&quot;([^&]*)&quot;)?\)/g, (_, alt, src) => `<img src="${escapeHtml(opts.rewriteSrc(unescapeAmp(src)))}" alt="${alt}" loading="lazy">`);
  s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, label, href) => `<a href="${escapeHtml(opts.rewriteHref(unescapeAmp(href)))}">${label}</a>`);
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/(^|[\s(>])\*([^*\s][^*]*)\*(?=[\s).,;:!?<]|$)/g, '$1<em>$2</em>');
  s = s.replace(/(^|[\s(>])_([^_\s][^_]*)_(?=[\s).,;:!?<]|$)/g, '$1<em>$2</em>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => spans[+i]);
}
const unescapeAmp = (s) => s.replace(/&amp;/g, '&');

const LIST = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;

function blocks(lines, opts) {
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    let m;
    if ((m = /^```(\S*)/.exec(line))) {
      const code = [];
      for (i++; i < lines.length && !/^```/.test(lines[i]); i++) code.push(lines[i]);
      i++;
      out.push(`<pre><code${m[1] ? ` class="language-${escapeHtml(m[1])}"` : ''}>${escapeHtml(code.join('\n'))}</code></pre>`);
    } else if ((m = /^(#{1,6})\s+(.*?)\s*#*\s*$/.exec(line))) {
      const level = m[1].length;
      const id = slug(m[2]);
      opts.headings.push({ level, text: m[2].replace(/`|\*/g, ''), id });
      out.push(`<h${level} id="${id}">${inline(m[2], opts)}</h${level}>`);
      i++;
    } else if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) {
      out.push('<hr>');
      i++;
    } else if (/^\s*>/.test(line)) {
      const quote = [];
      for (; i < lines.length && /^\s*>/.test(lines[i]); i++) quote.push(lines[i].replace(/^\s*>\s?/, ''));
      out.push(`<blockquote>${blocks(quote, opts).join('\n')}</blockquote>`);
    } else if (line.includes('|') && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(lines[i + 1])) {
      const cells = (l) => l.trim().replace(/^\|/, '').replace(/\|$/, '').split(/(?<!\\)\|/).map((c) => c.trim().replace(/\\\|/g, '|'));
      const head = cells(line);
      const align = cells(lines[i + 1]).map((c) => (/^:-+:$/.test(c) ? 'center' : /-:$/.test(c) ? 'right' : ''));
      const rows = [];
      for (i += 2; i < lines.length && lines[i].includes('|') && lines[i].trim(); i++) rows.push(cells(lines[i]));
      const td = (tag, c, k) => `<${tag}${align[k] ? ` class="al-${align[k]}"` : ''}>${inline(c, opts)}</${tag}>`;
      out.push(`<div class="table-wrap" tabindex="0" role="region" aria-label="Table"><table><thead><tr>${head.map((c, k) => td('th', c, k)).join('')}</tr></thead><tbody>${rows.map((r) => `<tr>${head.map((_, k) => td('td', r[k] ?? '', k)).join('')}</tr>`).join('')}</tbody></table></div>`);
    } else if ((m = LIST.exec(line))) {
      const ordered = /\d/.test(m[2]);
      const base = m[1].length;
      const items = [];
      while (i < lines.length) {
        const mm = LIST.exec(lines[i]);
        if (!mm || mm[1].length !== base || /\d/.test(mm[2]) !== ordered) break;
        const body = [mm[3]];
        const width = mm[1].length + mm[2].length + 1;
        i++;
        while (i < lines.length) {
          const l = lines[i];
          if (!l.trim()) {
            // a blank line belongs to the item only when an indented line follows
            let j = i;
            while (j < lines.length && !lines[j].trim()) j++;
            if (j < lines.length && (lines[j].match(/^\s*/)[0].length > base) ) { body.push(''); i++; continue; }
            break;
          }
          if (l.match(/^\s*/)[0].length > base) { body.push(l.slice(Math.min(width, l.match(/^\s*/)[0].length))); i++; continue; }
          break;
        }
        items.push(body);
      }
      const tag = ordered ? 'ol' : 'ul';
      out.push(`<${tag}>${items.map((b) => {
        const inner = blocks(b, opts);
        const loose = inner.length === 1 && inner[0].startsWith('<p>') ? inner[0].slice(3, -4) : inner.join('\n');
        return `<li>${loose}</li>`;
      }).join('')}</${tag}>`);
    } else {
      const para = [];
      for (; i < lines.length && lines[i].trim() && !/^(```|#{1,6}\s|\s*>)/.test(lines[i]) && !LIST.test(lines[i]); i++) para.push(lines[i].trim());
      if (!para.length) { para.push(lines[i++].trim()); }
      out.push(`<p>${inline(para.join(' '), opts)}</p>`);
    }
  }
  return out;
}

/** Renders Markdown to HTML; returns { html, headings, title }. */
export function renderMarkdown(source, options = {}) {
  const opts = { rewriteHref: (h) => h, rewriteSrc: (h) => h, headings: [], ...options };
  const html = blocks(source.replace(/\r\n?/g, '\n').split('\n'), opts).join('\n');
  const title = opts.headings.find((h) => h.level === 1)?.text ?? '';
  return { html, headings: opts.headings, title };
}

/** The text of the section under heading `title` (level 2), up to the next level-2 heading. */
export function section(source, title) {
  const lines = source.replace(/\r\n?/g, '\n').split('\n');
  const start = lines.findIndex((l) => l.trim() === `## ${title}`);
  if (start < 0) return null;
  let end = lines.findIndex((l, k) => k > start && /^##\s/.test(l));
  if (end < 0) end = lines.length;
  return lines.slice(start + 1, end).join('\n').trim();
}
