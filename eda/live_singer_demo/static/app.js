(function () {
  'use strict';

  const API_ANALYZE = '/api/analyze';
  const RECORD_DURATION_SEC = 4;
  const MEL_BINS = 64;
  const FFT_SIZE = 2048;

  const el = {
    melCanvas: document.getElementById('melCanvas'),
    btnMic: document.getElementById('btnMic'),
    btnAnalyze: document.getElementById('btnAnalyze'),
    status: document.getElementById('status'),
    result: document.getElementById('result'),
    topArtist: document.getElementById('topArtist'),
    tags: document.getElementById('tags'),
    artistList: document.getElementById('artistList'),
  };

  let audioContext = null;
  let stream = null;
  let analyser = null;
  let source = null;
  let animationId = null;
  let running = false;
  let recordBuffer = [];
  let recordSamples = 0;
  let sampleRate = 44100;
  let neededSamples = 0;

  function setStatus(text, className) {
    el.status.textContent = text;
    el.status.className = 'status' + (className ? ' ' + className : '');
  }

  // Map linear freq (Hz) to mel: mel = 2595 * log10(1 + f/700)
  function hzToMel(hz) {
    return 2595 * Math.log10(1 + hz / 700);
  }

  function melToHz(mel) {
    return 700 * (Math.pow(10, mel / 2595) - 1);
  }

  // Build FFT bin index -> mel bin index (0 .. MEL_BINS-1)
  function buildMelMap(fftSize, sampleRate) {
    const nyquist = sampleRate / 2;
    const binToHz = i => (i / fftSize) * sampleRate;
    const melMax = hzToMel(nyquist);
    const melMin = hzToMel(0);
    const melRange = melMax - melMin;
    const nBins = fftSize / 2;
    const melBinIndex = [];
    for (let i = 0; i < nBins; i++) {
      const hz = binToHz(i);
      const mel = hzToMel(hz);
      const t = (mel - melMin) / melRange;
      const bin = Math.min(MEL_BINS - 1, Math.floor(t * MEL_BINS));
      melBinIndex.push(bin);
    }
    return melBinIndex;
  }

  let melMap = null;
  let freqData = null;

  function drawLiveMel() {
    if (!running || !analyser || !el.melCanvas) return;
    const ctx = el.melCanvas.getContext('2d');
    const w = el.melCanvas.width;
    const h = el.melCanvas.height;
    if (!freqData) freqData = new Float32Array(analyser.frequencyBinCount);
    if (!melMap) melMap = buildMelMap(FFT_SIZE, audioContext.sampleRate);
    analyser.getFloatFrequencyData(freqData);

    const melSums = new Array(MEL_BINS).fill(0);
    const melCounts = new Array(MEL_BINS).fill(0);
    for (let i = 0; i < freqData.length; i++) {
      const db = freqData[i];
      const linear = Math.pow(10, db / 20);
      const bin = melMap[i];
      melSums[bin] += linear;
      melCounts[bin]++;
    }
    for (let b = 0; b < MEL_BINS; b++) {
      if (melCounts[b] > 0) melSums[b] /= melCounts[b];
    }
    const maxVal = Math.max(...melSums, 1e-6);

    const imgData = ctx.getImageData(1, 0, w - 1, h);
    ctx.putImageData(imgData, 0, 0);
    const col = w - 1;
    const colData = ctx.getImageData(col, 0, 1, h);
    for (let row = 0; row < h; row++) {
      const bin = Math.floor((1 - row / h) * MEL_BINS);
      const v = melSums[bin] / maxVal;
      const intensity = Math.min(255, Math.floor(80 + v * 175));
      const idx = row * 4;
      colData.data[idx] = 124;
      colData.data[idx + 1] = 58;
      colData.data[idx + 2] = 237;
      colData.data[idx + 3] = intensity;
    }
    ctx.putImageData(colData, col, 0);
    animationId = requestAnimationFrame(drawLiveMel);
  }

  function startMelViz() {
    if (el.melCanvas && analyser) {
      el.melCanvas.width = el.melCanvas.offsetWidth;
      el.melCanvas.height = el.melCanvas.offsetHeight;
      const ctx = el.melCanvas.getContext('2d');
      ctx.fillStyle = '#121118';
      ctx.fillRect(0, 0, el.melCanvas.width, el.melCanvas.height);
      drawLiveMel();
    }
  }

  function stopMelViz() {
    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }

  function onAudioProcess(e) {
    if (!running) return;
    const input = e.inputBuffer.getChannelData(0);
    for (let i = 0; i < input.length; i++) {
      recordBuffer.push(input[i]);
      recordSamples++;
    }
    while (recordBuffer.length > neededSamples) {
      recordBuffer.shift();
    }
  }

  function start() {
    setStatus('Requesting microphone…', '');
    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then(function (s) {
        stream = s;
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        sampleRate = audioContext.sampleRate;
        neededSamples = Math.ceil(sampleRate * RECORD_DURATION_SEC);
        const bufferSize = 4096;
        source = audioContext.createMediaStreamSource(stream);
        analyser = audioContext.createAnalyser();
        analyser.fftSize = FFT_SIZE;
        analyser.smoothingTimeConstant = 0.6;
        source.connect(analyser);
        const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
        processor.onaudioprocess = onAudioProcess;
        analyser.connect(processor);
        processor.connect(audioContext.destination);
        recordBuffer = [];
        recordSamples = 0;
        running = true;
        el.btnMic.textContent = 'Stop mic';
        el.btnMic.classList.add('stop');
        el.btnAnalyze.disabled = false;
        setStatus('Listening — sing for a few seconds, then click Analyze.', 'listening');
        startMelViz();
      })
      .catch(function (err) {
        setStatus('Microphone error: ' + (err.message || 'denied'), 'error');
      });
  }

  function stop() {
    running = false;
    stopMelViz();
    if (stream) {
      stream.getTracks().forEach(function (t) { t.stop(); });
      stream = null;
    }
    if (source && analyser) {
      try {
        source.disconnect();
        analyser.disconnect();
      } catch (_) {}
    }
    if (audioContext) {
      audioContext.close().catch(function () {});
      audioContext = null;
    }
    analyser = null;
    source = null;
    el.btnMic.textContent = 'Start mic';
    el.btnMic.classList.remove('stop');
    setStatus('Stopped. Start mic again to record and analyze.', '');
  }

  function analyze() {
    if (recordBuffer.length < sampleRate * 1.5) {
      setStatus('Record at least ~2 s of audio first.', 'error');
      return;
    }
    const samples = recordBuffer.slice(-neededSamples);
    const arr = new Float32Array(samples);
    setStatus('Analyzing…', 'analyzing');
    el.btnAnalyze.disabled = true;
    fetch(API_ANALYZE, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/octet-stream',
        'X-Sample-Rate': String(sampleRate),
      },
      body: arr.buffer,
    })
      .then(function (res) {
        if (!res.ok) {
          return res.json().then(function (j) { throw new Error(j.error || res.statusText); });
        }
        return res.json();
      })
      .then(function (data) {
        el.result.style.display = 'block';
        el.topArtist.textContent = data.top_artist + ' (' + (data.top_prob * 100).toFixed(1) + '%)';
        const top3 = (data.attributes || []).slice(0, 6);
        el.tags.innerHTML = top3.map(function (a) {
          const pct = (a.confidence * 100).toFixed(0);
          return '<span class="tag">' + a.tag + '<span class="pct">' + pct + '%</span></span>';
        }).join('');
        const artistLines = (data.artists || []).slice(0, 10).map(function (a) {
          return a.name + ' ' + (a.prob * 100).toFixed(1) + '%';
        });
        el.artistList.innerHTML = artistLines.map(function (s) {
          return '<span>' + s + '</span>';
        }).join('');
        setStatus('Done. Keep singing and Analyze again for a new result.', 'listening');
        el.btnAnalyze.disabled = false;
      })
      .catch(function (err) {
        setStatus('Error: ' + err.message, 'error');
        el.btnAnalyze.disabled = false;
      });
  }

  el.btnMic.addEventListener('click', function () {
    if (running) stop();
    else start();
  });

  el.btnAnalyze.addEventListener('click', analyze);
})();
