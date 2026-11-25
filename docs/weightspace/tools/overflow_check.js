const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 688, height: 1000 } });  // A4 width minus 2 x 14 mm margins at 96 dpi
  await page.emulateMedia({ media: 'print' });
  await page.goto('file://' + process.argv[2], { waitUntil: 'load' });
  const res = await page.evaluate(() => {
    const W = document.documentElement.clientWidth; const bad = [];
    for (const el of document.querySelectorAll('body *')) {
      const r = el.getBoundingClientRect();
      if (r.right > W + 1) bad.push({ tag: el.tagName, right: Math.round(r.right), text: (el.textContent || '').slice(0, 80) });
    }
    return { W, scrollW: document.documentElement.scrollWidth, n: bad.length, sample: bad.slice(0, 8) };
  });
  console.log(JSON.stringify(res, null, 1));
  await browser.close();
})();
