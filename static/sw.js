// Define cache names
const CACHE_NAME = 'sleepify-static-v1';
const DYNAMIC_CACHE_NAME = 'sleepify-dynamic-v1';

// Files to cache during install
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/style.css',
  '/manifest.json',

  // Templates (optional if needed offline)
  '/templates/base.html',
  '/templates/index.html',
  '/templates/book.html',
  '/templates/magic_predictor.html',
  '/templates/recccomendations.html',

  // Images
  '/static/images/index_img.jpg',
  '/static/images/sample_img.jpg',
  '/static/images/sleep_banner.avif',

  // Icons
  '/static/icons/icon512_maskable.png',
  '/static/icons/icon512_rounded.png',

  // Favicons
  '/static/favicons/fatFavicon.ico',
  '/static/favicons/favicon.ico',
  '/static/favicons/favicon2.ico',
  '/static/favicons/favicon3.ico',
  '/static/favicons/thinFavicon.ico',

  // Screenshots
  '/static/screenshots/screenshot1.png',
  '/static/screenshots/screenshot2.png',

  // JSON data
  '/static/json/sleep_sessions.json'
];

// Install event: Cache static assets
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      console.log('[Service Worker] Caching static assets');
      return cache.addAll(STATIC_ASSETS);
    })
  );
  self.skipWaiting();
});

// Activate event: Clean up old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys => {
      return Promise.all(
        keys.map(key => {
          if (key !== CACHE_NAME && key !== DYNAMIC_CACHE_NAME) {
            console.log('[Service Worker] Deleting old cache:', key);
            return caches.delete(key);
          }
        })
      );
    })
  );
  self.clients.claim();
});

// Fetch event: Network-first for API/HTML, Cache-first for images/icons
self.addEventListener('fetch', event => {
  const requestUrl = new URL(event.request.url);

  // Use cache-first strategy for static images/icons/screenshots
  if (
    requestUrl.pathname.startsWith('/static/images') ||
    requestUrl.pathname.startsWith('/static/icons') ||
    requestUrl.pathname.startsWith('/static/screenshots') ||
    requestUrl.pathname.startsWith('/static/favicons')
  ) {
    event.respondWith(
      caches.match(event.request).then(cachedResponse => {
        return cachedResponse || fetch(event.request).then(networkResponse => {
          return caches.open(DYNAMIC_CACHE_NAME).then(cache => {
            cache.put(event.request, networkResponse.clone());
            return networkResponse;
          });
        });
      })
    );
    return;
  }

  // Default: Network-first strategy for HTML, CSS, JSON
  event.respondWith(
    fetch(event.request)
      .then(networkResponse => {
        return caches.open(DYNAMIC_CACHE_NAME).then(cache => {
          cache.put(event.request, networkResponse.clone());
          return networkResponse;
        });
      })
      .catch(() => caches.match(event.request))
  );
});
