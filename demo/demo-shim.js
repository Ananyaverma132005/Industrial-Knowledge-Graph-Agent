(function () {
  'use strict';

  var DATA_URL = 'recorded.json';
  var dataPromise = fetch(DATA_URL).then(function (r) { return r.json(); });
  var lastAsked = null;

  var STOP = ['the', 'a', 'an', 'of', 'for', 'to', 'is', 'do', 'i', 'what', 'how', 'on', 'it', 'does', 'are', 'which', 'when', 'was', 'show', 'have', 'has', 'me'];
  function norm(s) { return String(s || '').toLowerCase().replace(/[^a-z0-9\s-]/g, ' ').replace(/\s+/g, ' ').trim(); }
  function words(s) { return norm(s).split(' ').filter(function (w) { return w && STOP.indexOf(w) === -1; }); }
  function similarity(a, b) {
    var A = words(a), B = words(b);
    if (!A.length || !B.length) return 0;
    var hits = A.filter(function (w) { return B.indexOf(w) !== -1; }).length;
    return hits / Math.max(A.length, B.length);
  }

  function findAnswer(data, query) {
    var q = norm(query);
    var prev = lastAsked && norm(lastAsked);
    var chats = data.chat;
    // 1. Exact follow-up recorded after the previous question.
    var hit = chats.find(function (c) { return c.prev && norm(c.prev) === prev && norm(c.q) === q; });
    // 2. Exact standalone question.
    hit = hit || chats.find(function (c) { return !c.prev && norm(c.q) === q; });
    // 3. Closest standalone question by shared keywords.
    if (!hit) {
      var best = null, bestScore = 0;
      chats.forEach(function (c) {
        if (c.prev) return;
        var s = similarity(query, c.q);
        if (s > bestScore) { bestScore = s; best = c; }
      });
      if (bestScore >= 0.5) hit = best;
    }
    return hit;
  }

  function demoReply(data) {
    var examples = data.chat.filter(function (c) { return !c.prev; }).slice(0, 8)
      .map(function (c) { return '- ' + c.q; }).join('\n');
    return {
      answer: 'This hosted demo replays answers recorded from the real agent, and this question isn\'t one of them. ' +
        'Try one of these, or clone the repo to ask anything:\n' + examples,
      route: 'Demo', mode: 'recorded demo', highlight: [], sources: [], cypher: null
    };
  }

  function ok(body) {
    return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } });
  }

  var realFetch = window.fetch.bind(window);
  window.fetch = function (input, init) {
    var url = typeof input === 'string' ? input : input.url;
    if (url === DATA_URL) return realFetch(input, init);
    var path = url.replace(/^https?:\/\/[^/]+/, '');
    var route = ['/health', '/graph-data', '/predict-maintenance', '/chat', '/node/'].find(function (p) { return path.indexOf(p) === 0; });
    if (!route) return realFetch(input, init);

    return dataPromise.then(function (data) {
      if (route === '/health') return ok(data.health);
      if (route === '/graph-data') return ok(data.graph);
      if (route === '/predict-maintenance') return ok(data.risk);
      if (route === '/node/') {
        var id = decodeURIComponent(path.slice('/node/'.length));
        var node = data.nodes[id];
        return node ? ok(node) : new Response(JSON.stringify({ detail: 'node not found' }), { status: 404 });
      }
      var body = {};
      try { body = JSON.parse((init && init.body) || '{}'); } catch (e) { /* ignore */ }
      var hit = findAnswer(data, body.query);
      var response = hit ? Object.assign({}, hit.response) : demoReply(data);
      if (hit) lastAsked = hit.q;
      response.session_id = 'demo';
      // Short pause so the progress indicator is visible, like a real query.
      return new Promise(function (resolve) { setTimeout(function () { resolve(ok(response)); }, 450); });
    });
  };

  // Replays the recorded telemetry stream in place of the /ws/telemetry WebSocket.
  var RealWebSocket = window.WebSocket;
  function TelemetryPlayback() {
    var self = this, i = 0;
    this.readyState = 0;
    dataPromise.then(function (data) {
      self.readyState = 1;
      if (self.onopen) self.onopen({});
      function send() {
        if (self.readyState !== 1) return;
        var frame = data.telemetry[i++ % data.telemetry.length];
        if (self.onmessage) self.onmessage({ data: JSON.stringify(frame) });
      }
      send();
      self._timer = setInterval(send, 3000);
    });
  }
  TelemetryPlayback.prototype.close = function () { this.readyState = 3; clearInterval(this._timer); };
  TelemetryPlayback.prototype.send = function () {};
  window.WebSocket = function (url, protocols) {
    if (String(url).indexOf('/ws/telemetry') !== -1) return new TelemetryPlayback();
    return new RealWebSocket(url, protocols);
  };

  function addBanner() {
    var bar = document.createElement('div');
    bar.setAttribute('role', 'note');
    bar.style.cssText = 'position:fixed;right:16px;bottom:16px;z-index:9999;max-width:min(420px,calc(100% - 32px));' +
      'background:#0f1f3d;color:#fff;font:400 13px/1.45 "IBM Plex Sans",system-ui,sans-serif;padding:10px 14px;' +
      'border-radius:8px;box-shadow:0 6px 20px rgba(0,0,0,.2);';
    bar.innerHTML = '<b>Hosted demo.</b> Answers, graph and telemetry are recorded from the real agent. ' +
      '<a style="color:#9cc3ff" href="https://github.com/Ananyaverma132005/Industrial-Knowledge-Graph-Agent">Run it locally</a> ' +
      'for live LangGraph + Neo4j queries. ' +
      '<button aria-label="Dismiss" style="background:none;border:0;color:#9ca3af;cursor:pointer;font-size:15px;margin-left:4px">×</button>';
    bar.querySelector('button').onclick = function () { bar.remove(); };
    document.body.appendChild(bar);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', addBanner);
  else addBanner();
})();
