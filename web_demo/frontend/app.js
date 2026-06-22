/* =====================================================================
   Hand Sign Recognition — app.js
   States: waiting | countdown | continuous | paused
   ===================================================================== */

"use strict"

// ── Constants ──────────────────────────────────────────────────────────
const ACTIONS = [
  "A",
  "B",
  "Bye",
  "C",
  "D",
  "Everything",
  "G",
  "Heart",
  "Help",
  "Hi",
  "I",
  "I love you",
  "L",
  "Like",
  "Love",
  "M",
  "N",
  "No",
  "O",
  "Ok",
  "Q",
  "Sorry",
  "Take Photo",
  "Talk",
  "U",
  "Vy",
  "Y",
]
const SEQ_LEN = 15
const ENDING_SIGN = "Talk"
const CD_SECS = 3
const WS_URL = `ws://${location.host}/ws`
const RING_CIRC = 2 * Math.PI * 52
const CONF_DEDUP = true // skip consecutive duplicate labels
const MAX_TRANSCRIPT = 40 // max chips shown

// ── DOM refs ───────────────────────────────────────────────────────────
const video = document.getElementById("video")
const canvas = document.getElementById("canvas")
const ctx = canvas.getContext("2d")
const btnToggle = document.getElementById("btnToggle")
const btnToggleIcon = document.getElementById("btnToggleIcon")
const btnToggleLabel = document.getElementById("btnToggleLabel")
const btnStop = document.getElementById("btnStop")
const stateBadge = document.getElementById("stateBadge")
const stateLabel = document.getElementById("stateLabel")
const connDot = document.getElementById("connDot")
const connLabel = document.getElementById("connLabel")
const cameraCard = document.getElementById("cameraCard")
const overlayCountdown = document.getElementById("overlayCountdown")
const overlayRecording = document.getElementById("overlayRecording")
const overlayNoCam = document.getElementById("overlayNoCam")
const cdNumber = document.getElementById("cdNumber")
const cdRingFill = document.getElementById("cdRingFill")
const recCount = document.getElementById("recCount")
const progressFill = document.getElementById("progressFill")
const resultPlaceholder = document.getElementById("resultPlaceholder")
const resultBody = document.getElementById("resultBody")
const resultLabel = document.getElementById("resultLabel")
const confValue = document.getElementById("confValue")
const confFill = document.getElementById("confFill")
const scoreGrid = document.getElementById("scoreGrid")
const signsGrid = document.getElementById("signsGrid")
const historyList = document.getElementById("historyList")
const fpsBadge = document.getElementById("fpsBadge")
const transcriptChips = document.getElementById("transcriptChips")
const transcriptCount = document.getElementById("transcriptCount")
const statWindows = document.getElementById("statWindows")
const statTime = document.getElementById("statTime")
const threshSlider = document.getElementById("threshSlider")
const threshVal = document.getElementById("threshVal")

// ── App State ──────────────────────────────────────────────────────────
let appState = "waiting" // waiting | countdown | continuous | paused
let ws = null
let wsConnected = false
let cdInterval = null
let cdRemain = CD_SECS
let lastResults = null
let confThreshold = 0.85 // min confidence to add to transcript
let lastLabel = null // for dedup
let transcriptArr = [] // accumulated labels
let windowTotal = 0 // windows predicted this session
let sessionStart = null // Date for session timer
let sessionTimer = null // interval id

// FPS tracking
let fpsLastTime = 0
let fpsFrames = 0

// ── 0.5s cooldown between windows ──────────────────────────────────────
let collectingPaused = false // brief pause after each prediction

// ── Build sign chips ────────────────────────────────────────────────────
ACTIONS.forEach((action) => {
  const chip = document.createElement("span")
  chip.className = "sign-chip"
  chip.id = `chip-${action.replace(/\s+/g, "-")}`
  chip.textContent = action
  signsGrid.appendChild(chip)
})

// ── WebSocket ──────────────────────────────────────────────────────────
function connectWS() {
  try {
    ws = new WebSocket(WS_URL)
  } catch (e) {
    setConnState(false)
    setTimeout(connectWS, 3000)
    return
  }

  ws.onopen = () => setConnState(true)
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data)
    if (msg.type === "progress") updateProgress(msg.frameCount, msg.total)
    else if (msg.type === "prediction") handlePrediction(msg)
  }
  ws.onclose = () => {
    setConnState(false)
    ws = null
    setTimeout(connectWS, 2000)
  }
  ws.onerror = () => ws.close()
}

function setConnState(connected) {
  wsConnected = connected
  connDot.className = "conn-dot " + (connected ? "connected" : "disconnected")
  connLabel.textContent = connected ? "Connected" : "Disconnected"
}

function sendWS(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj))
}

// ── State Machine ───────────────────────────────────────────────────────
function setState(s) {
  appState = s

  // Badge + card state attr
  const badgeState =
    {
      waiting: "waiting",
      countdown: "countdown",
      continuous: "recording",
      paused: "paused",
    }[s] || "waiting"
  stateBadge.setAttribute("data-state", badgeState)
  cameraCard.setAttribute("data-state", badgeState)
  stateLabel.textContent = s.toUpperCase()

  // Overlays
  overlayCountdown.classList.toggle("active", s === "countdown")
  overlayRecording.classList.toggle("active", s === "continuous")

  // Toggle button appearance
  switch (s) {
    case "waiting":
      btnToggle.setAttribute("data-action", "start")
      btnToggleIcon.textContent = "▶"
      btnToggleLabel.textContent = "Start Continuous"
      btnToggle.disabled = false
      btnStop.disabled = true
      break
    case "countdown":
      btnToggle.disabled = true
      btnStop.disabled = true
      break
    case "continuous":
      btnToggle.setAttribute("data-action", "pause")
      btnToggleIcon.textContent = "⏸"
      btnToggleLabel.textContent = "Pause"
      btnToggle.disabled = false
      btnStop.disabled = false
      break
    case "paused":
      btnToggle.setAttribute("data-action", "resume")
      btnToggleIcon.textContent = "▶"
      btnToggleLabel.textContent = "Resume"
      btnToggle.disabled = false
      btnStop.disabled = false
      break
  }
}

// ── Button handlers ─────────────────────────────────────────────────────
function handleToggle() {
  if (!wsConnected) return
  switch (appState) {
    case "waiting":
      startCountdown()
      break
    case "continuous":
      pauseSession()
      break
    case "paused":
      resumeSession()
      break
  }
}

function handleStop() {
  if (appState === "waiting") return
  // Clear countdown if running
  if (cdInterval) {
    clearInterval(cdInterval)
    cdInterval = null
  }
  stopSessionTimer()
  sendWS({ type: "reset" })
  setState("waiting")
}

// ── Countdown ───────────────────────────────────────────────────────────
function startCountdown() {
  setState("countdown")
  cdRemain = CD_SECS
  renderCountdown(cdRemain)
  sendWS({ type: "reset" })

  cdInterval = setInterval(() => {
    cdRemain--
    renderCountdown(cdRemain)
    if (cdRemain <= 0) {
      clearInterval(cdInterval)
      cdInterval = null
      startContinuous()
    }
  }, 1000)
}

function renderCountdown(n) {
  const c = Math.max(0, n)
  cdNumber.textContent = c === 0 ? "!" : c
  cdRingFill.style.strokeDashoffset = RING_CIRC * (1 - c / CD_SECS)
}

// ── Continuous session ──────────────────────────────────────────────────
function startContinuous() {
  // Reset session stats only if coming fresh from waiting
  if (windowTotal === 0) {
    sessionStart = Date.now()
    startSessionTimer()
  }
  setState("continuous")
  updateProgress(0, SEQ_LEN)
}

function pauseSession() {
  setState("paused")
  // Stop sending keypoints by checking appState in MediaPipe callback
}

function resumeSession() {
  setState("continuous")
  updateProgress(0, SEQ_LEN)
}

// ── Session timer ───────────────────────────────────────────────────────
function startSessionTimer() {
  stopSessionTimer()
  sessionTimer = setInterval(updateSessionTimer, 1000)
}

function stopSessionTimer() {
  if (sessionTimer) {
    clearInterval(sessionTimer)
    sessionTimer = null
  }
}

function updateSessionTimer() {
  if (!sessionStart) return
  const elapsed = Math.floor((Date.now() - sessionStart) / 1000)
  const mm = String(Math.floor(elapsed / 60)).padStart(2, "0")
  const ss = String(elapsed % 60).padStart(2, "0")
  statTime.textContent = `${mm}:${ss}`
}

// ── Progress bar ────────────────────────────────────────────────────────
function updateProgress(count, total) {
  const n = Math.min(count, total)
  recCount.innerHTML = `${n} <span>/ ${total}</span>`
  progressFill.style.width = `${(n / total) * 100}%`
}

// ── Keypoint Extraction ─────────────────────────────────────────────────
function extractKeypoints(results) {
  const lh = new Float32Array(63)
  const rh = new Float32Array(63)

  if (results.multiHandLandmarks && results.multiHandedness) {
    for (let i = 0; i < results.multiHandLandmarks.length; i++) {
      const landmarks = results.multiHandLandmarks[i]
      const rawLabel = results.multiHandedness[i].label
      const label = rawLabel === "Left" ? "Right" : "Left" // swap for flipped convention

      const ref_x = landmarks[0].x,
        ref_y = landmarks[0].y,
        ref_z = landmarks[0].z
      const kp = new Float32Array(63)
      for (let j = 0; j < 21; j++) {
        kp[j * 3] = 1.0 - landmarks[j].x - (1.0 - ref_x)
        kp[j * 3 + 1] = landmarks[j].y - ref_y
        kp[j * 3 + 2] = landmarks[j].z - ref_z
      }
      if (label === "Left") lh.set(kp)
      else rh.set(kp)
    }
  }

  const combined = new Float32Array(126)
  combined.set(lh, 0)
  combined.set(rh, 63)
  return Array.from(combined)
}

// ── Prediction Handler ──────────────────────────────────────────────────
function handlePrediction({ label, confidence, allScores, windowIndex }) {
  windowTotal = windowIndex || windowTotal + 1
  statWindows.textContent = windowTotal

  // Update result card (always, even in continuous)
  showResult(label, confidence, allScores)
  highlightChip(label)
  addHistory(label, confidence)

  // Transcript: filter by threshold + dedup
  if (confidence >= confThreshold) {
    if (label === ENDING_SIGN) {
      speakTranscript()
      clearTranscript()
      pauseSession()
    } else if (!CONF_DEDUP || label !== lastLabel) {
      addToTranscript(label, confidence)
      lastLabel = label
    }
  }

  // 0.5s cooldown before next window starts collecting
  if (appState === "continuous") {
    collectingPaused = true
    setTimeout(() => {
      collectingPaused = false
    }, 1250)
  }
}

// ── Result Card ─────────────────────────────────────────────────────────
function showResult(label, confidence, allScores) {
  resultPlaceholder.style.display = "none"
  resultBody.style.display = "flex"

  // Re-trigger animation
  resultLabel.style.animation = "none"
  void resultLabel.offsetWidth // reflow
  resultLabel.style.animation = ""

  resultLabel.textContent = label
  confValue.textContent = (confidence * 100).toFixed(1) + "%"
  confFill.style.width = confidence * 100 + "%"

  // Top-6 scores
  scoreGrid.innerHTML = ""
  const sorted = Object.entries(allScores)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
  const maxScore = sorted[0][1]
  sorted.forEach(([name, score], idx) => {
    const div = document.createElement("div")
    div.className = "score-item" + (idx === 0 ? " top" : "")
    div.innerHTML = `
      <span style="min-width:70px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${name}</span>
      <div class="score-bar-wrap"><div class="score-bar-inner" style="width:${((score / maxScore) * 100).toFixed(1)}%"></div></div>
      <span style="font-family:'JetBrains Mono',monospace;min-width:38px;text-align:right">${(score * 100).toFixed(0)}%</span>`
    scoreGrid.appendChild(div)
  })
}

function highlightChip(label) {
  document
    .querySelectorAll(".sign-chip")
    .forEach((c) => c.classList.remove("active"))
  const chip = document.getElementById(`chip-${label.replace(/\s+/g, "-")}`)
  if (chip) chip.classList.add("active")
}

// ── Transcript ──────────────────────────────────────────────────────────
function addToTranscript(label, confidence) {
  // Remove empty placeholder
  const empty = transcriptChips.querySelector(".transcript-empty")
  if (empty) empty.remove()

  transcriptArr.push({ label, confidence })
  if (transcriptArr.length > MAX_TRANSCRIPT) {
    transcriptArr.shift()
    transcriptChips.removeChild(transcriptChips.firstChild)
  }

  // Plain word chip — no % shown (confidence is in History panel)
  const chip = document.createElement("span")
  chip.className = "t-chip"
  chip.textContent = label
  chip.title = `${label} — ${(confidence * 100).toFixed(1)}% confidence`
  transcriptChips.appendChild(chip)
  transcriptChips.scrollLeft = transcriptChips.scrollWidth // scroll right

  transcriptCount.textContent = `${transcriptArr.length} sign${transcriptArr.length !== 1 ? "s" : ""}`
}

function clearTranscript() {
  transcriptArr = []
  lastLabel = null
  transcriptChips.innerHTML =
    '<span class="transcript-empty">Recognition will appear here\u2026</span>'
  transcriptCount.textContent = "0 signs"
}

// ── Text-to-Speech ──────────────────────────────────────────────────────
function speakTranscript() {
  if (transcriptArr.length === 0) return

  const textToSpeak = transcriptArr.map((item) => item.label).join(" ")
  const utterance = new SpeechSynthesisUtterance(textToSpeak)
  utterance.lang = "en-US"
  utterance.rate = 0.9

  window.speechSynthesis.speak(utterance)
}

// ── Threshold slider ────────────────────────────────────────────────────
function updateThreshold(val) {
  confThreshold = parseInt(val) / 100
  threshVal.textContent = `${val}%`
}

// ── History ─────────────────────────────────────────────────────────────
let historyCount = 0
function addHistory(label, confidence) {
  const empty = historyList.querySelector(".history-empty")
  if (empty) empty.remove()
  historyCount++
  const pct = (confidence * 100).toFixed(1)
  const now = new Date().toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  })
  const low = confidence < 0.7
  const li = document.createElement("li")
  li.className = "history-item"
  li.innerHTML = `
    <div class="hi-left">
      <span class="hi-label">#${historyCount} \u2014 ${label}</span>
      <span class="hi-time">${now}</span>
    </div>
    <span class="hi-conf${low ? " low" : ""}">${pct}%</span>`
  historyList.prepend(li)
}

function clearHistory() {
  historyList.innerHTML =
    '<li class="history-empty">No predictions yet\u2026</li>'
  historyCount = 0
}

// ── MediaPipe Hands ─────────────────────────────────────────────────────
const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
})
hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
})

hands.onResults((results) => {
  lastResults = results

  canvas.width = video.videoWidth || 640
  canvas.height = video.videoHeight || 480
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  if (results.multiHandLandmarks) {
    results.multiHandLandmarks.forEach((landmarks) => {
      drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
        color: "#555",
        lineWidth: 2,
      })
      drawLandmarks(ctx, landmarks, {
        color: "#fff",
        fillColor: "rgba(124,58,237,.85)",
        lineWidth: 1,
        radius: 4,
      })
    })
  }

  // Only send keypoints when actively collecting (and not in 0.5s cooldown)
  if (appState === "continuous" && !collectingPaused) {
    const kp = extractKeypoints(results)
    sendWS({ type: "keypoints", keypoints: kp })
  }

  // FPS counter
  fpsFrames++
  const now = performance.now()
  if (now - fpsLastTime >= 1000) {
    fpsBadge.textContent = `${fpsFrames} fps`
    fpsFrames = 0
    fpsLastTime = now
  }
})

// ── Camera ──────────────────────────────────────────────────────────────
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user",
      },
      audio: false,
    })
    video.srcObject = stream
    await video.play()

    const cam = new Camera(video, {
      onFrame: async () => {
        await hands.send({ image: video })
      },
      width: 640,
      height: 480,
    })
    cam.start()
  } catch (err) {
    console.error("Camera error:", err)
    overlayNoCam.classList.add("active")
  }
}

// ── Keyboard shortcuts ──────────────────────────────────────────────────
document.addEventListener("keydown", (e) => {
  if (e.code === "Space") {
    e.preventDefault()
    handleToggle()
  }
  if (e.code === "Escape") {
    if (appState === "continuous" || appState === "paused") handleStop()
    closeLightbox()
  }
})

// ── Reference Guide: toggle collapse ───────────────────────────────────
function toggleGuide() {
  document.querySelector(".guide-inner").classList.toggle("collapsed")
}

// ── Lightbox ────────────────────────────────────────────────────────────
let lightbox = null
function openLightbox(src, alt) {
  if (!lightbox) {
    lightbox = document.createElement("div")
    lightbox.className = "lightbox-overlay"
    lightbox.innerHTML = '<img src="" alt="" />'
    lightbox.addEventListener("click", closeLightbox)
    document.body.appendChild(lightbox)
  }
  lightbox.querySelector("img").src = src
  lightbox.querySelector("img").alt = alt
  requestAnimationFrame(() => lightbox.classList.add("open"))
}
function closeLightbox() {
  if (lightbox) lightbox.classList.remove("open")
}
window.addEventListener("DOMContentLoaded", () => {
  const img = document.getElementById("guideImg")
  if (img) img.addEventListener("click", () => openLightbox(img.src, img.alt))
})

// ── Init ────────────────────────────────────────────────────────────────
;(function init() {
  setState("waiting")
  startCamera()
  connectWS()
  fpsLastTime = performance.now()
})()
