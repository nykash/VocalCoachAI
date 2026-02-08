(function () {
  'use strict';

  const CHUNK_DURATION_SEC = 3;
  const API_URL = '/api/score';
  let SAMPLE_RATE = 44100;
  let CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION_SEC;

  const el = {
    score: document.getElementById('score'),
    calibrated: document.getElementById('calibrated'),
    btn: document.getElementById('btn'),
    status: document.getElementById('status'),
    details: document.getElementById('details'),
  };

  let audioContext = null;
  let stream = null;
  let processor = null;
  let source = null;
  let buffer = [];
  let totalSamples = 0;
  let running = false;
  let lastResult = null;

  function setStatus(text, isListening) {
    el.status.textContent = text;
    el.status.classList.toggle('listening', !!isListening);
  }

  function setScore(value) {
    if (value == null || value === '' || isNaN(value)) {
      el.score.textContent = '—';
      el.score.className = 'score-value';
      return;
    }
    const v = Number(value);
    el.score.textContent = Math.round(v);
    el.score.className = 'score-value good';
    if (v < 40) el.score.className = 'score-value low';
    else if (v < 65) el.score.className = 'score-value mid';
  }

  function setCalibrated(value) {
    if (value == null || value === '' || isNaN(value)) {
      el.calibrated.textContent = '—';
      el.calibrated.className = 'score-value calibrated';
      return;
    }
    const v = Number(value);
    el.calibrated.textContent = v.toFixed(2);
    el.calibrated.className = 'score-value calibrated good';
    if (v < 0.4) el.calibrated.className = 'score-value calibrated low';
    else if (v < 0.7) el.calibrated.className = 'score-value calibrated mid';
  }

  function renderDetails(data) {
    if (!data || typeof data !== 'object') {
    el.details.innerHTML = '';
    return;
    }
    const parts = [];
    if (data.hnr_db != null) parts.push(['HNR (dB)', data.hnr_db.toFixed(1)]);
    if (data.rms_cv != null) parts.push(['RMS CV', data.rms_cv.toFixed(2)]);
    el.details.innerHTML = parts.map(([k, v]) => `<dt>${k}:</dt><dd>${v}</dd>`).join('');
  }

  function onAudioProcess(e) {
    if (!running) return;
    const input = e.inputBuffer.getChannelData(0);
    for (let i = 0; i < input.length; i++) {
      buffer.push(input[i]);
      totalSamples++;
      if (totalSamples >= CHUNK_SAMPLES) {
        totalSamples = 0;
        const chunk = buffer.splice(0, CHUNK_SAMPLES);
        sendChunk(chunk, audioContext.sampleRate);
      }
    }
  }

  function sendChunk(float32Array, sampleRate) {
    const sr = sampleRate || SAMPLE_RATE;
    const arr = new Float32Array(float32Array);
    setStatus('Analyzing…', true);
    fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/octet-stream',
        'X-Sample-Rate': String(sr),
      },
      body: arr.buffer,
    })
      .then(function (res) {
        if (!res.ok) return res.json().then(function (j) { throw new Error(j.error || res.statusText); });
        return res.json();
      })
      .then(function (data) {
        lastResult = data;
        setScore(data.breath_support_score);
        setCalibrated(data.calibrated_score != null ? data.calibrated_score : null);
        renderDetails(data);
        setStatus('Listening… (score updates every ' + CHUNK_DURATION_SEC + ' s)', true);
      })
      .catch(function (err) {
        setStatus('Error: ' + err.message, false);
        el.status.classList.add('error');
      });
  }

  function start() {
    setStatus('Requesting microphone…', false);
    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then(function (s) {
        stream = s;
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        SAMPLE_RATE = audioContext.sampleRate;
        CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION_SEC;
        const bufferSize = 4096;
        source = audioContext.createMediaStreamSource(stream);
        processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
        processor.onaudioprocess = onAudioProcess;
        source.connect(processor);
        processor.connect(audioContext.destination);
        buffer = [];
        totalSamples = 0;
        running = true;
        el.btn.textContent = 'Stop';
        el.btn.className = 'btn btn-stop';
        setStatus('Listening… (sing or speak)', true);
      })
      .catch(function (err) {
        setStatus('Microphone error: ' + (err.message || 'denied'), false);
        el.status.classList.add('error');
      });
  }

  function stop() {
    running = false;
    if (processor) {
      try {
        processor.disconnect();
        source.disconnect();
      } catch (_) {}
      processor = null;
      source = null;
    }
    if (stream) {
      stream.getTracks().forEach(function (t) { t.stop(); });
      stream = null;
    }
    if (audioContext) {
      audioContext.close().catch(function () {});
      audioContext = null;
    }
    el.btn.textContent = 'Start microphone';
    el.btn.className = 'btn btn-start';
    setStatus('Stopped. Click to start again.', false);
  }

  el.btn.addEventListener('click', function () {
    if (running) stop();
    else start();
  });
})();
