"use strict";

const elements = {
  canvas: document.querySelector("#scenario-canvas"),
  status: document.querySelector("#load-status"),
  runState: document.querySelector(".run-state"),
  file: document.querySelector("#trace-file"),
  timeline: document.querySelector("#timeline"),
  play: document.querySelector("#play-toggle"),
  previous: document.querySelector("#previous-frame"),
  next: document.querySelector("#next-frame"),
  frameNumber: document.querySelector("#frame-number"),
  simulationTime: document.querySelector("#simulation-time"),
  vehicleId: document.querySelector("#vehicle-id"),
  vehicleAction: document.querySelector("#vehicle-action"),
  vehicleLatency: document.querySelector("#vehicle-latency"),
  vehicleFuel: document.querySelector("#vehicle-fuel"),
  vehicleTrack: document.querySelector("#vehicle-track"),
  deadlineState: document.querySelector("#deadline-state"),
  deadlineFill: document.querySelector("#deadline-fill"),
  deadlineDetail: document.querySelector("#deadline-detail"),
  actionBars: document.querySelector("#action-bars"),
  decisionCount: document.querySelector("#decision-count"),
  traceIdentity: document.querySelector("#trace-identity"),
};

const state = {
  trace: null,
  frameIndex: 0,
  selectedVehicle: 1,
  playing: false,
  lastAnimationTime: 0,
  width: 0,
  height: 0,
};

const actionColors = {
  HOLD: "#5f6f72",
  INVESTIGATE: "#a67528",
  INTERCEPT: "#b5482d",
  EVADE: "#7c3540",
  RETURN: "#356a55",
};

function assertTrace(trace) {
  if (
    !trace ||
    trace.schema_version !== 1 ||
    trace.project !== "market_sim" ||
    !trace.scenario ||
    !trace.metrics ||
    !Array.isArray(trace.frames) ||
    trace.frames.length === 0
  ) {
    throw new Error("This file is not a market_sim trace (schema version 1).");
  }
  for (const frame of trace.frames) {
    if (!Array.isArray(frame.vehicles) || !Array.isArray(frame.decisions)) {
      throw new Error("The trace is missing vehicle or decision frames.");
    }
  }
}

function percentage(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function setText(selector, value) {
  document.querySelector(selector).textContent = value;
}

function loadTrace(trace, label) {
  assertTrace(trace);
  state.trace = trace;
  state.frameIndex = 0;
  state.selectedVehicle = trace.frames[0].vehicles[0]?.id ?? 1;
  state.playing = false;
  elements.timeline.max = String(trace.frames.length - 1);
  elements.timeline.value = "0";
  elements.status.textContent = `${label} · ${trace.frames.length} frames`;
  elements.runState.classList.remove("error");
  elements.runState.classList.add("loaded");
  elements.play.textContent = "Play replay";
  populateSummary();
  populateActionMix();
  render();
}

function populateSummary() {
  const { metrics, scenario } = state.trace;
  setText("#metric-p99", (metrics.p99_latency_microseconds / 1000).toFixed(2));
  setText("#metric-valid", percentage(metrics.grammar_valid_fraction, 0));
  setText("#metric-cache", percentage(metrics.kv_page_reduction_fraction));
  setText("#metric-rate", Math.round(metrics.decisions_per_service_second).toLocaleString());
  setText("#evidence-batches", metrics.scheduler_batches.toLocaleString());
  setText("#evidence-deadlines", percentage(metrics.deadline_met_fraction));
  setText("#evidence-fairness", metrics.jain_completion_fairness.toFixed(3));
  setText("#evidence-realtime", `${metrics.faster_than_realtime_factor.toFixed(1)}×`);
  elements.traceIdentity.textContent =
    `Seed ${scenario.seed} · ${scenario.vehicle_count} vehicles · ` +
    `${scenario.target_count} tracks · batch ${scenario.maximum_batch_size} · ` +
    `${(scenario.decision_deadline_microseconds / 1000).toFixed(1)} ms deadline`;
}

function populateActionMix() {
  const counts = new Map();
  for (const frame of state.trace.frames) {
    for (const decision of frame.decisions) {
      counts.set(decision.action, (counts.get(decision.action) ?? 0) + 1);
    }
  }
  const total = state.trace.metrics.decisions;
  const ordered = ["INTERCEPT", "EVADE", "RETURN", "INVESTIGATE", "HOLD"];
  elements.actionBars.replaceChildren();
  for (const action of ordered) {
    const count = counts.get(action) ?? 0;
    const row = document.createElement("div");
    row.className = "action-row";
    const label = document.createElement("span");
    label.textContent = action;
    const track = document.createElement("span");
    track.className = "bar-track";
    const fill = document.createElement("i");
    fill.style.width = `${total === 0 ? 0 : (count / total) * 100}%`;
    fill.style.background = actionColors[action];
    track.append(fill);
    const value = document.createElement("span");
    value.textContent = String(count);
    row.append(label, track, value);
    elements.actionBars.append(row);
  }
  elements.decisionCount.textContent = `${total.toLocaleString()} decisions`;
}

function worldExtent() {
  return state.trace?.scenario.world_half_extent ?? 220;
}

function worldToCanvas(position) {
  const extent = worldExtent();
  const margin = Math.min(state.width, state.height) * 0.065;
  const size = Math.min(state.width - margin * 2, state.height - margin * 2);
  return {
    x: state.width / 2 + (position.x / extent) * (size / 2),
    y: state.height / 2 - (position.y / extent) * (size / 2),
  };
}

function resizeCanvas() {
  const bounds = elements.canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(bounds.width));
  const height = Math.max(1, Math.round(bounds.height));
  if (elements.canvas.width !== Math.round(width * dpr) || elements.canvas.height !== Math.round(height * dpr)) {
    elements.canvas.width = Math.round(width * dpr);
    elements.canvas.height = Math.round(height * dpr);
  }
  const context = elements.canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  state.width = width;
  state.height = height;
}

function drawGrid(context) {
  context.fillStyle = "#e4dbc8";
  context.fillRect(0, 0, state.width, state.height);
  context.strokeStyle = "#c5baa3";
  context.lineWidth = 1;
  const grid = Math.max(28, Math.min(state.width, state.height) / 11);
  context.beginPath();
  for (let x = state.width / 2 % grid; x < state.width; x += grid) {
    context.moveTo(x, 0);
    context.lineTo(x, state.height);
  }
  for (let y = state.height / 2 % grid; y < state.height; y += grid) {
    context.moveTo(0, y);
    context.lineTo(state.width, y);
  }
  context.stroke();

  const origin = worldToCanvas({ x: 0, y: 0 });
  context.strokeStyle = "#817966";
  context.lineWidth = 1.5;
  context.beginPath();
  context.moveTo(0, origin.y);
  context.lineTo(state.width, origin.y);
  context.moveTo(origin.x, 0);
  context.lineTo(origin.x, state.height);
  context.stroke();
  context.fillStyle = "#18272f";
  context.beginPath();
  context.arc(origin.x, origin.y, 4, 0, Math.PI * 2);
  context.fill();
}

function drawTargetTrails(context) {
  const start = Math.max(0, state.frameIndex - 14);
  const targetIds = state.trace.frames[state.frameIndex].targets.map((target) => target.id);
  context.lineWidth = 1.5;
  for (const id of targetIds) {
    context.strokeStyle = "#a94a34";
    context.beginPath();
    let started = false;
    for (let index = start; index <= state.frameIndex; index += 1) {
      const target = state.trace.frames[index].targets.find((value) => value.id === id);
      if (!target) continue;
      const point = worldToCanvas(target.position);
      if (!started) {
        context.moveTo(point.x, point.y);
        started = true;
      } else {
        context.lineTo(point.x, point.y);
      }
    }
    context.stroke();
  }
}

function drawRadar(context, frame) {
  context.setLineDash([4, 5]);
  for (const value of frame.radar_returns) {
    const observer = frame.vehicles.find((vehicle) => vehicle.id === value.observer_id);
    if (!observer) continue;
    const from = worldToCanvas(observer.position);
    const to = worldToCanvas(value.estimated_position);
    const selected = value.observer_id === state.selectedVehicle;
    context.strokeStyle = selected ? "#9b3d27" : "#938a77";
    context.globalAlpha = selected ? 0.9 : 0.22;
    context.lineWidth = selected ? 1.8 : 1;
    context.beginPath();
    context.moveTo(from.x, from.y);
    context.lineTo(to.x, to.y);
    context.stroke();
    context.beginPath();
    context.arc(to.x, to.y, selected ? 5 : 2.5, 0, Math.PI * 2);
    context.stroke();
  }
  context.globalAlpha = 1;
  context.setLineDash([]);
}

function drawTargets(context, frame) {
  for (const target of frame.targets) {
    const point = worldToCanvas(target.position);
    context.save();
    context.translate(point.x, point.y);
    context.rotate(Math.PI / 4);
    context.fillStyle = "#b5482d";
    context.fillRect(-6, -6, 12, 12);
    context.restore();
    context.fillStyle = "#18272f";
    context.font = "700 11px 'Avenir Next Condensed', sans-serif";
    context.fillText(`T${String(target.id).padStart(2, "0")}`, point.x + 10, point.y - 8);
  }
}

function drawVehicles(context, frame) {
  for (const vehicle of frame.vehicles) {
    const point = worldToCanvas(vehicle.position);
    const heading = Math.atan2(vehicle.velocity.y, vehicle.velocity.x);
    const selected = vehicle.id === state.selectedVehicle;
    const radius = selected ? 9 : 5.5;
    context.save();
    context.translate(point.x, point.y);
    context.rotate(-heading);
    context.fillStyle = actionColors[vehicle.action] ?? "#18272f";
    context.strokeStyle = selected ? "#18272f" : "#f4ecdc";
    context.lineWidth = selected ? 3 : 1;
    context.beginPath();
    context.moveTo(radius * 1.45, 0);
    context.lineTo(-radius, radius * 0.8);
    context.lineTo(-radius * 0.55, 0);
    context.lineTo(-radius, -radius * 0.8);
    context.closePath();
    context.fill();
    context.stroke();
    context.restore();
    if (selected) {
      context.strokeStyle = "#18272f";
      context.lineWidth = 1;
      context.beginPath();
      context.arc(point.x, point.y, 17, 0, Math.PI * 2);
      context.stroke();
    }
  }
}

function updateTelemetry(frame) {
  const vehicle = frame.vehicles.find((value) => value.id === state.selectedVehicle) ?? frame.vehicles[0];
  if (!vehicle) return;
  state.selectedVehicle = vehicle.id;
  const decision = frame.decisions.find((value) => value.vehicle_id === vehicle.id);
  const radar = frame.radar_returns.find((value) => value.observer_id === vehicle.id);
  elements.vehicleId.textContent = String(vehicle.id).padStart(2, "0");
  elements.vehicleAction.textContent = decision?.action ?? vehicle.action;
  elements.vehicleLatency.textContent = decision ? `${(decision.latency_microseconds / 1000).toFixed(3)} ms` : "—";
  elements.vehicleFuel.textContent = percentage(vehicle.fuel_fraction, 0);
  elements.vehicleTrack.textContent = radar ? `T${String(radar.target_id).padStart(2, "0")} / ${radar.measured_range.toFixed(1)} m` : "No return";
  if (decision) {
    const budget = state.trace.scenario.decision_deadline_microseconds;
    const ratio = Math.min(1, decision.latency_microseconds / budget);
    elements.deadlineState.textContent = decision.deadline_met ? "MET" : "MISSED";
    elements.deadlineState.style.color = decision.deadline_met ? "#75a98c" : "#e2a44a";
    elements.deadlineFill.style.width = `${ratio * 100}%`;
    elements.deadlineFill.style.background = decision.deadline_met ? "#477c63" : "#b5482d";
    const delta = Math.abs(decision.latency_microseconds - budget) / 1000;
    elements.deadlineDetail.textContent = `${delta.toFixed(3)} ms ${decision.deadline_met ? "inside" : "over"} the ${(budget / 1000).toFixed(1)} ms envelope.`;
  }
}

function render() {
  resizeCanvas();
  const context = elements.canvas.getContext("2d");
  drawGrid(context);
  if (!state.trace) return;
  const frame = state.trace.frames[state.frameIndex];
  drawTargetTrails(context);
  drawRadar(context, frame);
  drawTargets(context, frame);
  drawVehicles(context, frame);
  updateTelemetry(frame);
  elements.timeline.value = String(state.frameIndex);
  elements.frameNumber.textContent = `${String(state.frameIndex + 1).padStart(2, "0")} / ${String(state.trace.frames.length).padStart(2, "0")}`;
  elements.simulationTime.textContent = `T+${frame.simulation_seconds.toFixed(2).padStart(5, "0")} s`;
}

function setFrame(index) {
  if (!state.trace) return;
  state.frameIndex = Math.max(0, Math.min(state.trace.frames.length - 1, index));
  render();
}

function togglePlay() {
  if (!state.trace) return;
  state.playing = !state.playing;
  elements.play.textContent = state.playing ? "Pause replay" : "Play replay";
  if (state.playing) {
    if (state.frameIndex === state.trace.frames.length - 1) setFrame(0);
    state.lastAnimationTime = performance.now();
    requestAnimationFrame(animate);
  }
}

function animate(now) {
  if (!state.playing || !state.trace) return;
  if (now - state.lastAnimationTime >= 140) {
    if (state.frameIndex >= state.trace.frames.length - 1) {
      state.playing = false;
      elements.play.textContent = "Play replay";
      return;
    }
    setFrame(state.frameIndex + 1);
    state.lastAnimationTime = now;
  }
  requestAnimationFrame(animate);
}

function showError(message) {
  elements.status.textContent = message;
  elements.runState.classList.remove("loaded");
  elements.runState.classList.add("error");
}

elements.file.addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) return;
  try {
    loadTrace(JSON.parse(await file.text()), file.name);
  } catch (error) {
    showError(error instanceof Error ? error.message : "The trace could not be read.");
  }
});

elements.timeline.addEventListener("input", () => setFrame(Number(elements.timeline.value)));
elements.play.addEventListener("click", togglePlay);
elements.previous.addEventListener("click", () => setFrame(state.frameIndex - 1));
elements.next.addEventListener("click", () => setFrame(state.frameIndex + 1));

elements.canvas.addEventListener("click", (event) => {
  if (!state.trace) return;
  const bounds = elements.canvas.getBoundingClientRect();
  const click = { x: event.clientX - bounds.left, y: event.clientY - bounds.top };
  const frame = state.trace.frames[state.frameIndex];
  let closest = null;
  let closestDistance = 22;
  for (const vehicle of frame.vehicles) {
    const point = worldToCanvas(vehicle.position);
    const distance = Math.hypot(point.x - click.x, point.y - click.y);
    if (distance < closestDistance) {
      closest = vehicle;
      closestDistance = distance;
    }
  }
  if (closest) {
    state.selectedVehicle = closest.id;
    render();
  }
});

window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement && event.target !== elements.timeline) return;
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    setFrame(state.frameIndex - 1);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    setFrame(state.frameIndex + 1);
  } else if (event.code === "Space") {
    event.preventDefault();
    togglePlay();
  }
});

new ResizeObserver(render).observe(elements.canvas.parentElement);

fetch("./radar-trace.json")
  .then((response) => {
    if (!response.ok) throw new Error(`Trace request failed (${response.status}).`);
    return response.json();
  })
  .then((trace) => loadTrace(trace, "Reference trace"))
  .catch(() => showError("Reference trace unavailable. Choose a generated JSON trace."));
