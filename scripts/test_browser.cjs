// Run with NODE_PATH pointing to an installation of playwright-core.
const { chromium } = require('playwright-core');
const fs = require('node:fs');
const path = require('node:path');
(async () => {
  const out = path.resolve(__dirname, '../captures/browser-test');
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_PATH || '/home/imsel/.cache/ms-playwright/chromium-1223/chrome-linux/chrome',
    headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage']
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const checks = [];
  async function waitStatus(condition) {
    await page.waitForFunction(async (expression) => {
      const s = await (await fetch('/api/status')).json();
      return new Function('s', `return ${expression}`)(s);
    }, condition, { timeout: 30000 });
  }
  function check(value, message) { if (!value) throw new Error(message); checks.push(message); }
  try {
    await page.goto('http://127.0.0.1:5000', { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => document.querySelector('#live-label').textContent === 'LIVE');
    await page.waitForFunction(() => document.querySelector('#stream').naturalWidth > 0);
    check(await page.locator('#stream').evaluate(img => img.naturalWidth > 0), 'Live MJPEG rendered in browser');
    await page.locator('#policy button[data-value="fixed576"]').click();
    await waitStatus("s.latest.policy === 'fixed576' && s.latest.resolution === 576");
    checks.push('Fixed 576 button changes live inference');
    for (const view of ['thermal', 'rgb', 'split']) {
      await page.locator(`#view button[data-value="${view}"]`).click();
      await waitStatus(`s.latest.view === '${view}'`);
      checks.push(`${view} view button works`);
    }
    await page.locator('#confidence').evaluate(el => {
      el.value = '0.43'; el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await waitStatus('s.latest.confidence === 0.43');
    checks.push('Confidence slider updates inference');
    await page.locator('#snapshot').click();
    await page.waitForFunction(() => document.querySelector('#toast').textContent.startsWith('Saved snapshot_'));
    checks.push('Snapshot button saves image and shows success');
    await page.locator('#run-toggle').click();
    await waitStatus('s.running === false');
    await page.waitForFunction(() => document.querySelector('#live-label').textContent === 'STOPPED');
    checks.push('Stop button stops inference and updates UI');
    await page.locator('#run-toggle').click();
    await waitStatus('s.running && s.latest_age_ms < 2000');
    await page.waitForFunction(() => document.querySelector('#live-label').textContent === 'LIVE');
    checks.push('Start button restarts live pipeline');
    await page.locator('#policy button[data-value="sequence"]').click();
    await waitStatus("s.latest.policy === 'sequence'");
    checks.push('Dynamic button restores sequence controller');
    await page.locator('#confidence').evaluate(el => {
      el.value = '0.25'; el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await waitStatus('s.latest.confidence === 0.25');
    await page.waitForFunction(() => Number(document.querySelector('#fps').textContent) > 0);
    await page.waitForFunction(() => document.querySelector('#stream').naturalWidth === 1426);
    await page.waitForFunction(() => !document.querySelector('#toast').classList.contains('show'));
    check(await page.locator('#cpp-workspace').getAttribute('href') === 'http://127.0.0.1:8080/', 'C++ workspace link targets camera server');
    check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Desktop has no horizontal overflow');
    await page.screenshot({ path: path.join(out, 'desktop-dark.png'), fullPage: true });
    await page.locator('#theme-toggle').click();
    check(await page.locator('body').evaluate(el => el.classList.contains('light-theme')), 'Theme toggle switches to light');
    await page.screenshot({ path: path.join(out, 'desktop-light.png'), fullPage: true });
    await page.locator('#theme-toggle').click();
    await page.setViewportSize({ width: 390, height: 844 });
    check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Mobile has no horizontal overflow');
    check(await page.locator('#stream').evaluate(el => {
      const image = el.getBoundingClientRect(), panel = el.closest('.viewer-card').getBoundingClientRect();
      return image.left >= panel.left && image.right <= panel.right && panel.right <= innerWidth;
    }), 'Mobile split video fits entirely inside its panel');
    await page.screenshot({ path: path.join(out, 'mobile-dark.png'), fullPage: true });
    check(errors.length === 0, 'No JavaScript runtime errors');
    fs.writeFileSync(path.join(out, 'result.json'), JSON.stringify({ passed: true, checks, errors, testedAt: new Date().toISOString() }, null, 2));
    console.log(JSON.stringify({ passed: true, checks, screenshots: out }, null, 2));
  } catch (error) {
    await page.screenshot({ path: path.join(out, 'failure.png'), fullPage: true }).catch(() => {});
    fs.writeFileSync(path.join(out, 'result.json'), JSON.stringify({ passed: false, checks, errors, failure: error.message }, null, 2));
    throw error;
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
