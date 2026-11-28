// Usage: node html2pdf.js IN.html OUT.pdf
const { chromium } = require('playwright');
(async () => {
  const [inp, out] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('file://' + inp, { waitUntil: 'load' });
  const opts = {
    path: out, format: 'A4', printBackground: true, preferCSSPageSize: true,
    displayHeaderFooter: true,
    headerTemplate: '<div style="font-size:7px;width:100%;text-align:right;padding-right:14mm;color:#777;font-family:DejaVu Sans,sans-serif">WSBench – Build Dossier, Phase 1</div>',
    footerTemplate: '<div style="font-size:7px;width:100%;text-align:center;color:#777;font-family:DejaVu Sans,sans-serif"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',
    outline: true, tagged: true,
  };
  try { await page.pdf(opts); } catch (e) { delete opts.outline; delete opts.tagged; await page.pdf(opts); }
  await browser.close();
})();
