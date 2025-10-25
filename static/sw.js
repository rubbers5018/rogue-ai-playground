// Rogue AI - Lean Service Worker (iOS-optimized)
const CACHE_NAME = 'rogue-ai-v1';
const ASSETS = ['/', '/static/manifest.json', '/static/icon-192.png', '/static/icon-512.png'];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys => Promise.all(
      keys.map(k => k !== CACHE_NAME ? caches.delete(k) : null)
    )).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;

  const url = new URL(e.request.url);
  // Skip AI API calls
  if (url.hostname.includes('huggingface') || url.hostname.includes('together') ||
      url.hostname.includes('venice') || url.hostname.includes('openrouter')) {
    return;
  }

  e.respondWith(
    caches.match(e.request).then(r => r || fetch(e.request).then(res => {
      if (res && res.status === 200) {
        caches.open(CACHE_NAME).then(c => c.put(e.request, res.clone()));
      }
      return res;
    })).catch(() => new Response(
      '<h1>🔥 Offline: Reconnect for uncensor grit</h1><p>NeuralDaredevil awaits your return.</p>',
      { headers: { 'Content-Type': 'text/html' } }
    ))
  );
});

// Logout cleanup
self.addEventListener('message', e => {
  if (e.data?.type === 'logout') {
    caches.delete(CACHE_NAME);
  }
});
