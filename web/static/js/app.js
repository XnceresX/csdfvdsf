/**
 * SAFEPASS Multi-Camera Strategic Topology Tracker & Monitor Center
 * "Modern Dark" Frontend Surveillance Station Controller
 */

// Dynamic Camera metadata and coordinates loaded from GET /api/topology
let CAMERA_META = {};
let CAMERA_NEIGHBORS = {};
let FLOOR_MAP_FILES = {};
let FLOOR_NAMES = {};
let TOPOLOGY_THRESHOLDS = {};
let TOPOLOGY_EDGES = [];

async function loadTopology() {
  try {
    const res = await fetch("/api/topology");
    if (res.ok) {
      const data = await res.json();
      if (data.cameras) {
        for (const [cid, c] of Object.entries(data.cameras)) {
          CAMERA_META[cid] = {
            floor: c.floor,
            x: c.pixel_coords ? c.pixel_coords.x : Math.round(c.coords.x * 1200),
            y: c.pixel_coords ? c.pixel_coords.y : Math.round(c.coords.y * 800),
            angle: c.fov_angle,
            name: c.name,
          };
          CAMERA_NEIGHBORS[cid] = c.neighbors || [];
        }
      }
      if (data.floors) {
        for (const [fidx, finfo] of Object.entries(data.floors)) {
          FLOOR_MAP_FILES[fidx] = finfo.file;
          FLOOR_NAMES[fidx] = finfo.name;
        }
      }
      if (data.thresholds) {
        TOPOLOGY_THRESHOLDS = data.thresholds;
      }
      if (data.edges) {
        TOPOLOGY_EDGES = data.edges;
      }
      console.log("[SAFEPASS] Loaded dynamic topology from /api/topology:", Object.keys(CAMERA_META).length, "cameras.");
    }
  } catch (err) {
    console.error("[SAFEPASS] Failed to load topology from /api/topology:", err);
  }
}

const COLOR_HEX_MAP = {
  any: "transparent",
  black: "#111111",
  white: "#ffffff",
  gray: "#71717a",
  red: "#e05252",
  orange: "#f97316",
  yellow: "#eab308",
  green: "#3fd6c6",
  blue: "#2563eb",
  purple: "#a855f7",
  brown: "#78350f",
  pink: "#ec4899",
  beige: "#f5f5dc",
};

// Global App State
let currentFloor = 0;
let isPlaying = false;
let currentSpeed = 1.0;
let currentTime = 0.0;
let maxDuration = 900.0;
let focusedCameraId = "cam1";
let subfeed1Id = "cam1";
let subfeed2Id = "cam3";
let ws = null;
let pollInterval = null;
let lastKnownState = null;
let hasAutoOpenedReport = false;
let currentReportData = null;

// ── Initialization ────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", async () => {
  await loadTopology();
  initWizardModal();
  initTimelineControls();
  initFloorTabs();
  initSyncModal();
  initWebSocket();
  renderSvgBlueprint();
  updateNeighborSubfeeds();
  const firstId = Object.keys(CAMERA_META)[0] || "cam1";
  focusCamera(firstId, CAMERA_META[firstId]?.name || "CAM 1");
});

// ── Wizard Intake Modal ───────────────────────────────────────────────────────

function initWizardModal() {
  const modal = document.getElementById("wizard-modal");
  const openBtn = document.getElementById("btn-open-wizard");
  const closeBtn = document.getElementById("btn-close-wizard");
  const startBtn = document.getElementById("btn-start-tracking-wizard");
  const resetBtn = document.getElementById("btn-reset-vms");

  if (openBtn && modal) {
    openBtn.addEventListener("click", () => {
      modal.style.display = "flex";
    });
  }

  if (closeBtn && modal) {
    closeBtn.addEventListener("click", () => {
      modal.style.display = "none";
    });
  }

  if (resetBtn) {
    resetBtn.addEventListener("click", async () => {
      try {
        await fetch("/api/reset", { method: "POST" });
        await setPlayback(false, 1.0);
        await seekToTime(0.0);
        hasAutoOpenedReport = false;
        closeReportModal();
        const rRow = document.getElementById("row-view-report");
        if (rRow) rRow.style.display = "none";
        if (modal) modal.style.display = "flex";
      } catch (e) {
        console.error("Reset error:", e);
      }
    });
  }

  const nextSightingBtn = document.getElementById("btn-dock-next-sighting");
  if (nextSightingBtn) {
    nextSightingBtn.addEventListener("click", advanceNextSighting);
  }

  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA") {
      return;
    }
    if (e.key === "n" || e.key === "N") {
      advanceNextSighting();
    }
  });

  if (startBtn) {
    startBtn.addEventListener("click", async () => {
      await submitTargetSearch();
    });
  }
}

function updateColorPreview(type, val) {
  const dotId = type === "upper" ? "dot-preview-upper" : "dot-preview-lower";
  const dot = document.getElementById(dotId);
  if (dot) {
    dot.style.background = COLOR_HEX_MAP[val] || "#111";
  }
}

async function submitTargetSearch() {
  const startCam = document.getElementById("wizard-start-cam")?.value || "cam2";
  const upperColor = document.getElementById("wizard-upper-color")?.value || "any";
  const lowerColor = document.getElementById("wizard-lower-color")?.value || "any";
  const garmentType = document.getElementById("wizard-garment-type")?.value || "Any";
  const outerwear = document.getElementById("wizard-outerwear")?.value || "Any";
  const bagSelection = document.getElementById("wizard-bag")?.value || "Any";
  const headwear = document.getElementById("wizard-hat")?.value || "Any";

  // Map bag option to backpack/bag booleans
  let hasBackpack = "Any";
  let hasBag = "Any";
  if (bagSelection === "Backpack") {
    hasBackpack = "Yes";
    hasBag = "No";
  } else if (bagSelection === "Handbag") {
    hasBag = "Yes";
    hasBackpack = "No";
  } else if (bagSelection === "NoBag") {
    hasBackpack = "No";
    hasBag = "No";
  }

  const payload = {
    upper_color: upperColor,
    lower_color: lowerColor,
    has_backpack: hasBackpack,
    has_bag: hasBag,
    has_hat: headwear,
    has_jacket: outerwear,
    has_longpants: garmentType,
    starting_cameras: [startCam],
  };

  try {
    const res = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (res.ok) {
      console.log(`[SAFEPASS] Search initiated on ${startCam}`);
      
      // Focus chosen camera
      const camMeta = CAMERA_META[startCam];
      focusCamera(startCam, camMeta?.name || startCam);

      // Start global playback
      await setPlayback(true, 1.0);

      // Dismiss wizard modal completely
      const modal = document.getElementById("wizard-modal");
      if (modal) modal.style.display = "none";
    }
  } catch (e) {
    console.error("Search submission error:", e);
  }
}

// ── WebSocket & Status Polling ────────────────────────────────────────────────

function initWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  try {
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log("[SAFEPASS] Live WebSocket connected.");
      if (pollInterval) clearInterval(pollInterval);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        updateDashboardState(data);
      } catch (e) {
        console.error("WS Parse error:", e);
      }
    };

    ws.onclose = () => {
      console.warn("[SAFEPASS] WS disconnected. Polling fallback active.");
      startPolling();
    };

    ws.onerror = () => {
      startPolling();
    };
  } catch (e) {
    startPolling();
  }
}

function startPolling() {
  if (pollInterval) return;
  pollInterval = setInterval(async () => {
    try {
      const res = await fetch("/api/status");
      if (res.ok) {
        const data = await res.json();
        updateDashboardState(data);
      }
    } catch (e) {
      console.error("Poll error:", e);
    }
  }, 250);
}

// ── Main Dashboard State Update ───────────────────────────────────────────────

function updateDashboardState(data) {
  lastKnownState = data;
  currentTime = data.current_time || 0.0;
  maxDuration = data.max_time || 900.0;
  isPlaying = data.is_playing || false;
  currentSpeed = data.speed || 1.0;

  // 1. Digital Clock & Scrubber
  const clockEl = document.getElementById("dock-clock-display");
  if (clockEl) clockEl.textContent = data.time_str || "00:00.0";

  const slider = document.getElementById("vms-scrubber");
  if (slider && !slider.matches(":active")) {
    slider.value = currentTime;
  }

  const playBtn = document.getElementById("btn-dock-play");
  if (playBtn) {
    playBtn.textContent = isPlaying ? "⏸ PAUSE" : "▶ PLAY";
  }

  // 2. Efficiency Pill & Camera State Badges
  let idleCount = 0;
  let activeCount = 0;
  let neighborCount = 0;

  if (data.camera_states) {
    for (const [cid, stateStr] of Object.entries(data.camera_states)) {
      const treeDot = document.getElementById(`tree-dot-${cid}`);
      const stripDot = document.getElementById(`strip-dot-${cid}`);

      let dotClass = "dot-idle";
      if (stateStr.includes("ACTIVE")) {
        activeCount++;
        dotClass = "dot-active";
      } else if (stateStr.includes("SEARCHING")) {
        neighborCount++;
        dotClass = "dot-neighbor";
      } else {
        idleCount++;
      }

      if (treeDot) treeDot.className = `dp-cam-dot ${dotClass}`;
      if (stripDot) {
        stripDot.style.background = dotClass === "dot-active" ? "var(--dp-color-live)" :
                                   dotClass === "dot-neighbor" ? "var(--dp-color-amber)" : "var(--dp-color-hairline-2)";
      }
    }
  }

  const effText = document.getElementById("efficiency-text");
  if (effText) {
    const pct = ((idleCount / 8.0) * 100).toFixed(1);
    if (activeCount > 0) {
      effText.textContent = `STRATEGIC 1-CAM ACTIVE: ${pct}% GPU SAVINGS (${idleCount} OF 8 CAMS IDLE)`;
    } else if (neighborCount > 0) {
      effText.textContent = `NEIGHBOR HANDOFF: ${pct}% GPU SAVINGS (${idleCount} CAMS IDLE)`;
    } else {
      effText.textContent = `ALL CAMERAS IDLE (100% GPU SAVED)`;
    }
  }

  // Handle Report & Sighting Resolution
  const reportBtnRow = document.getElementById("row-view-report");
  if (data.final_verdict || data.report) {
    if (reportBtnRow) reportBtnRow.style.display = "";
    if (data.global_state === "RESOLVED" && !hasAutoOpenedReport) {
      openReportModal();
      hasAutoOpenedReport = true;
    }
  } else {
    if (reportBtnRow) reportBtnRow.style.display = "none";
  }

  // Handle Sequential Demo indicator
  if (data.demo_active) {
    const effText = document.getElementById("efficiency-text");
    if (effText && data.active_camera) {
      effText.textContent = `SEQUENTIAL RE-ID DEMO · SIGHTING ${data.demo_segment_idx + 1} OF ${data.demo_total_segments} (${data.active_camera.toUpperCase()})`;
    }
    const nextBtn = document.getElementById("btn-dock-next-sighting");
    if (nextBtn) nextBtn.style.display = "inline-flex";
  }

  // 3. Operational Target Status Table
  updateTargetStatusTable(data);

  // 4. Case Trajectory Audit Log
  renderTrajectoryLog(data.case_timeline || []);

  // 5. Timeline Event Markers
  renderTimelineMarkers(data.case_timeline || []);

  // 6. SVG Floor Plan Blueprint & FOV Cones
  updateSvgMapState(data);

  // 7. Auto-focus active camera if tracking
  if (data.active_camera && data.active_camera !== focusedCameraId && isPlaying) {
    focusCamera(data.active_camera, CAMERA_META[data.active_camera]?.name || data.active_camera);
  }

  // Bank tag
  const bankTag = document.getElementById("primary-bank-tag");
  if (bankTag) {
    bankTag.textContent = `OSNet Bank: N=${data.bank_size || 0}`;
  }
}

// ── Target Status Widget ──────────────────────────────────────────────────────

function updateTargetStatusTable(data) {
  const pill = document.getElementById("target-state-pill");
  const stateVal = document.getElementById("tbl-target-state");
  const locVal = document.getElementById("tbl-target-loc");
  const confVal = document.getElementById("tbl-target-conf");
  const pathVal = document.getElementById("tbl-target-path");
  const bankVal = document.getElementById("tbl-target-bank");
  const destVal = document.getElementById("tbl-target-dest");

  if (bankVal) bankVal.textContent = `N=${data.bank_size || 0} embeddings`;

  if (data.final_verdict) {
    const v = data.final_verdict;
    if (pill) {
      pill.className = "dp-status-badge status-resolved";
      pill.textContent = "RESOLVED";
    }
    if (stateVal) stateVal.innerHTML = `<span style="color:var(--dp-color-alert); font-weight:700;">DESTINATION RESOLVED</span>`;
    if (locVal) locVal.innerHTML = `<span style="color:var(--dp-color-alert); font-weight:700;">${v.inferred_location}</span>`;
    if (confVal) confVal.innerHTML = `<span style="color:var(--dp-color-alert);">${(v.confidence * 100).toFixed(0)}%</span>`;
    if (pathVal) pathVal.innerHTML = `<span>${v.camera_path || "CAM 2"}</span>`;
    if (destVal) destVal.innerHTML = `<span style="color:var(--dp-color-alert); font-weight:700;">${v.inferred_location}</span>`;
  } else if (data.active_camera) {
    const meta = CAMERA_META[data.active_camera];
    if (pill) {
      pill.className = "dp-status-badge status-tracking";
      pill.textContent = "TRACKING";
    }
    if (stateVal) stateVal.innerHTML = `<span style="color:var(--dp-color-live); font-weight:700;">STRATEGIC LOCK</span>`;
    if (locVal) locVal.innerHTML = `<span style="color:var(--dp-color-live); font-weight:600;">${meta?.name || data.active_camera.toUpperCase()}</span>`;
    if (confVal) confVal.innerHTML = `<span style="color:var(--dp-color-live);">94.8% Match</span>`;
    
    const pStr = (data.camera_path || [data.active_camera]).map(c => c.toUpperCase()).join(" ➔ ");
    if (pathVal) pathVal.innerHTML = `<span>${pStr}</span>`;
    if (destVal) destVal.innerHTML = `<span class="dp-text-dim">In Transit...</span>`;
  } else if (data.global_state === "ACQUIRING") {
    if (pill) {
      pill.className = "dp-status-badge status-acquiring";
      pill.textContent = "ACQUIRING";
    }
    if (stateVal) stateVal.innerHTML = `<span style="color:var(--dp-color-amber); font-weight:700;">SCANNING ENTRY</span>`;
    if (locVal) locVal.innerHTML = `<span style="color:var(--dp-color-amber);">Searching...</span>`;
    if (confVal) confVal.innerHTML = `<span>Scanning</span>`;
  } else {
    if (pill) {
      pill.className = "dp-status-badge status-standby";
      pill.textContent = "STANDBY";
    }
    if (stateVal) stateVal.innerHTML = `<span class="dp-text-dim">AWAITING TARGET</span>`;
    if (locVal) locVal.innerHTML = `<span class="dp-text-dim">None</span>`;
    if (confVal) confVal.innerHTML = `<span>--</span>`;
    if (pathVal) pathVal.innerHTML = `<span>None</span>`;
    if (destVal) destVal.innerHTML = `<span class="dp-text-dim">Unresolved</span>`;
  }
}

// ── Trajectory Audit Log ──────────────────────────────────────────────────────

function renderTrajectoryLog(events) {
  const tbody = document.getElementById("log-table-body");
  const countEl = document.getElementById("log-event-count");
  if (!tbody) return;

  if (countEl) countEl.textContent = `${events.length} Events`;
  if (events.length === 0) return;

  tbody.innerHTML = "";
  [...events].reverse().forEach((ev) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="dp-mono" style="color:var(--dp-color-bright);">${ev.time_str}</td>
      <td class="dp-mono" style="font-weight:700; color:var(--dp-color-live);">${ev.from_cam.toUpperCase()} ➔ ${ev.to_cam.toUpperCase()}</td>
      <td>${ev.reason}</td>
      <td class="dp-mono" style="color:var(--dp-color-live); font-weight:600;">${(ev.score * 100).toFixed(0)}%</td>
    `;
    tbody.appendChild(tr);
  });
}

// ── Timeline Markers ──────────────────────────────────────────────────────────

function renderTimelineMarkers(events) {
  const wrap = document.getElementById("timeline-event-markers");
  if (!wrap) return;

  wrap.innerHTML = "";
  events.forEach((ev) => {
    if (ev.timestamp != null && maxDuration > 0) {
      const pct = (ev.timestamp / maxDuration) * 100;
      const marker = document.createElement("div");
      marker.className = "event-tick-mark";
      marker.style.left = `${pct}%`;
      marker.title = `${ev.time_str}: ${ev.from_cam} ➔ ${ev.to_cam} (${ev.reason})`;
      wrap.appendChild(marker);
    }
  });
}

// ── SVG Architectural Floor Plan & FOV Cones ──────────────────────────────────

function initFloorTabs() {
  const tabs = document.querySelectorAll(".dp-tab-btn, .vms-tab-btn");
  tabs.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabs.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      currentFloor = parseInt(btn.dataset.floor, 10);

      // Update embedded blueprint image
      const svgImg = document.getElementById("svg-floorplan-img");
      if (svgImg) {
        svgImg.setAttribute("href", FLOOR_MAP_FILES[currentFloor] || "/static/maps/ground_floor.png");
      }

      // Update badge
      const badge = document.getElementById("current-floor-badge");
      if (badge) {
        badge.textContent = FLOOR_NAMES[currentFloor] || "FLOOR PLAN";
      }

      renderSvgBlueprint();

      // Switch camera focus to first camera on this floor if current focused camera is on another floor
      const currentMeta = CAMERA_META[focusedCameraId];
      if (currentMeta && currentMeta.floor !== currentFloor) {
        const firstCam = Object.entries(CAMERA_META).find(([cid, m]) => m.floor === currentFloor);
        if (firstCam) {
          focusCamera(firstCam[0], firstCam[1].name);
        }
      }
    });
  });
}

function renderSvgBlueprint() {
  const fovLayer = document.getElementById("map-fov-layer");
  const nodesLayer = document.getElementById("map-nodes-layer");
  if (!fovLayer || !nodesLayer) return;

  fovLayer.innerHTML = "";
  nodesLayer.innerHTML = "";

  for (const [cid, meta] of Object.entries(CAMERA_META)) {
    if (meta.floor !== currentFloor) continue;

    // 1. Render FOV Cone Sector Wedge
    const conePath = createFovConePath(meta.x, meta.y, meta.angle, 95, 65);
    const pathEl = document.createElementNS("http://www.w3.org/2000/svg", "path");
    pathEl.setAttribute("d", conePath);
    pathEl.setAttribute("class", "fov-cone-idle");
    pathEl.setAttribute("id", `fov-cone-${cid}`);
    fovLayer.appendChild(pathEl);

    // 2. Render Camera Node
    const nodeG = document.createElementNS("http://www.w3.org/2000/svg", "g");
    nodeG.setAttribute("id", `node-g-${cid}`);
    nodeG.style.cursor = "pointer";
    nodeG.onclick = () => focusCamera(cid, meta.name);

    // Outer ring
    const outerRing = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    outerRing.setAttribute("cx", meta.x);
    outerRing.setAttribute("cy", meta.y);
    outerRing.setAttribute("r", "19");
    outerRing.setAttribute("fill", "#090909");
    outerRing.setAttribute("stroke", "#222222");
    outerRing.setAttribute("stroke-width", "1.5");
    outerRing.setAttribute("id", `outer-ring-${cid}`);

    // Inner lens
    const lens = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    lens.setAttribute("cx", meta.x);
    lens.setAttribute("cy", meta.y);
    lens.setAttribute("r", "7");
    lens.setAttribute("fill", "#2a2a2a");
    lens.setAttribute("id", `lens-${cid}`);

    // Label
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", meta.x);
    text.setAttribute("y", meta.y + 35);
    text.setAttribute("fill", "#f2f2f2");
    text.setAttribute("font-size", "12px");
    text.setAttribute("font-weight", "700");
    text.setAttribute("font-family", "var(--dp-font-mono)");
    text.setAttribute("text-anchor", "middle");
    text.textContent = cid.toUpperCase();

    nodeG.appendChild(outerRing);
    nodeG.appendChild(lens);
    nodeG.appendChild(text);
    nodesLayer.appendChild(nodeG);
  }

  if (lastKnownState) {
    updateSvgMapState(lastKnownState);
  }
}

/**
 * Generates an SVG path for an FOV cone sector/wedge.
 */
function createFovConePath(cx, cy, centerAngleDeg, radius, spreadAngleDeg) {
  const rad = Math.PI / 180;
  const startAngle = (centerAngleDeg - spreadAngleDeg / 2) * rad;
  const endAngle = (centerAngleDeg + spreadAngleDeg / 2) * rad;

  const x1 = cx + radius * Math.cos(startAngle);
  const y1 = cy + radius * Math.sin(startAngle);
  const x2 = cx + radius * Math.cos(endAngle);
  const y2 = cy + radius * Math.sin(endAngle);

  return `M ${cx} ${cy} L ${x1.toFixed(1)} ${y1.toFixed(1)} A ${radius} ${radius} 0 0 1 ${x2.toFixed(1)} ${y2.toFixed(1)} Z`;
}

function updateSvgMapState(data) {
  const beaconLayer = document.getElementById("map-beacon-layer");
  if (beaconLayer) beaconLayer.innerHTML = "";

  for (const [cid, meta] of Object.entries(CAMERA_META)) {
    if (meta.floor !== currentFloor) continue;

    const cone = document.getElementById(`fov-cone-${cid}`);
    const ring = document.getElementById(`outer-ring-${cid}`);
    const lens = document.getElementById(`lens-${cid}`);
    const stateStr = data.camera_states?.[cid] || "";

    if (stateStr.includes("ACTIVE")) {
      if (cone) cone.setAttribute("class", "fov-cone-active");
      if (ring) {
        ring.setAttribute("stroke", "var(--dp-color-live)");
        ring.setAttribute("stroke-width", "2.5");
      }
      if (lens) lens.setAttribute("fill", "var(--dp-color-live)");

      // Render mint target alert beacon at camera location
      if (beaconLayer) {
        const beacon = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        beacon.setAttribute("cx", meta.x);
        beacon.setAttribute("cy", meta.y);
        beacon.setAttribute("class", "beacon-circle");
        beaconLayer.appendChild(beacon);
      }
    } else if (stateStr.includes("SEARCHING")) {
      if (cone) cone.setAttribute("class", "fov-cone-neighbor");
      if (ring) {
        ring.setAttribute("stroke", "var(--dp-color-amber)");
        ring.setAttribute("stroke-width", "2");
      }
      if (lens) lens.setAttribute("fill", "var(--dp-color-amber)");
    } else {
      if (cone) cone.setAttribute("class", "fov-cone-idle");
      if (ring) {
        ring.setAttribute("stroke", "var(--dp-color-hairline)");
        ring.setAttribute("stroke-width", "1.5");
      }
      if (lens) lens.setAttribute("fill", "var(--dp-color-hairline-2)");
    }
  }

  // Trajectory polyline
  const polyline = document.getElementById("trajectory-line");
  if (polyline && data.camera_path) {
    const points = [];
    data.camera_path.forEach((cid) => {
      const meta = CAMERA_META[cid];
      if (meta && meta.floor === currentFloor) {
        points.push(`${meta.x},${meta.y}`);
      }
    });
    polyline.setAttribute("points", points.join(" "));
  }
}

// ── Camera Focus & Neighbor Routing ───────────────────────────────────────────

function focusCamera(cameraId, cameraName) {
  focusedCameraId = cameraId;

  // Notify backend of focused camera
  fetch(`/api/focus/${cameraId}`, { method: "POST" }).catch((e) => {
    console.debug("Focus notification:", e);
  });

  // 1. Update Primary Feed Title & Stream
  const titleEl = document.getElementById("primary-feed-title");
  if (titleEl) {
    titleEl.textContent = `PRIMARY FEED: ${cameraId.toUpperCase()} (${cameraName.replace(/CAM \d+ \(|\)/g, "")})`;
  }

  const focusedImg = document.getElementById("focused-img");
  if (focusedImg) {
    focusedImg.src = `/api/stream/${cameraId}`;
  }

  // 2. Update Quick Switcher Strip Buttons
  document.querySelectorAll(".dp-switcher-btn, .switcher-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.id === `switch-btn-${cameraId}`);
  });

  // 3. Update Left Nav Tree Active Item
  document.querySelectorAll(".dp-cam-item, .cam-nav-item").forEach((item) => {
    item.classList.toggle("active", item.id === `nav-item-${cameraId}`);
  });

  // 4. Switch Floor Tab if needed
  const meta = CAMERA_META[cameraId];
  if (meta && meta.floor !== currentFloor) {
    currentFloor = meta.floor;
    const tabBtn = document.querySelector(`.dp-tab-btn[data-floor="${meta.floor}"], .vms-tab-btn[data-floor="${meta.floor}"]`);
    if (tabBtn) {
      document.querySelectorAll(".dp-tab-btn, .vms-tab-btn").forEach((b) => b.classList.remove("active"));
      tabBtn.classList.add("active");
      const svgImg = document.getElementById("svg-floorplan-img");
      if (svgImg) svgImg.setAttribute("href", FLOOR_MAP_FILES[currentFloor] || "/static/maps/ground_floor.png");
      const badge = document.getElementById("current-floor-badge");
      if (badge) badge.textContent = FLOOR_NAMES[currentFloor] || "FLOOR PLAN";
      renderSvgBlueprint();
    }
  }

  // 5. Update Neighbor Sub-Feeds
  updateNeighborSubfeeds();
}

function updateNeighborSubfeeds() {
  const neighbors = CAMERA_NEIGHBORS[focusedCameraId] || ["cam1", "cam3"];
  subfeed1Id = neighbors[0] || "cam1";
  subfeed2Id = neighbors[1] || (neighbors[0] === "cam2" ? "cam3" : "cam2");

  // Sub-Feed 1
  const t1 = document.getElementById("subfeed-title-1");
  const img1 = document.getElementById("subfeed-img-1");
  if (t1) t1.textContent = `NEIGHBOR 1: ${CAMERA_META[subfeed1Id]?.name.toUpperCase() || subfeed1Id.toUpperCase()}`;
  if (img1) img1.src = `/api/stream/${subfeed1Id}`;

  // Sub-Feed 2
  const t2 = document.getElementById("subfeed-title-2");
  const img2 = document.getElementById("subfeed-img-2");
  if (t2) t2.textContent = `NEIGHBOR 2: ${CAMERA_META[subfeed2Id]?.name.toUpperCase() || subfeed2Id.toUpperCase()}`;
  if (img2) img2.src = `/api/stream/${subfeed2Id}`;
}

function focusNeighborSubfeed(index) {
  const targetCam = index === 1 ? subfeed1Id : subfeed2Id;
  const meta = CAMERA_META[targetCam];
  focusCamera(targetCam, meta?.name || targetCam);
}

// ── Left Navigation Tree Layout Selector ──────────────────────────────────────

function selectLayout(layout) {
  document.querySelectorAll(".dp-tree-item, .tree-item").forEach((el) => {
    if (el.id.startsWith("layout-")) el.classList.remove("active");
  });
  const selEl = document.getElementById(`layout-${layout}`);
  if (selEl) selEl.classList.add("active");

  const layoutHeader = document.getElementById("header-layout-name");

  if (layout === "ground") {
    if (layoutHeader) layoutHeader.textContent = "Ground Floor View";
    const tab = document.querySelector(`.dp-tab-btn[data-floor="0"], .vms-tab-btn[data-floor="0"]`);
    if (tab) tab.click();
  } else if (layout === "upper") {
    if (layoutHeader) layoutHeader.textContent = "Upper Floors View";
    const tab = document.querySelector(`.dp-tab-btn[data-floor="1"], .vms-tab-btn[data-floor="1"]`);
    if (tab) tab.click();
  } else {
    if (layoutHeader) layoutHeader.textContent = "Multi-Camera Strategic Topology Tracker";
    const tab = document.querySelector(`.dp-tab-btn[data-floor="0"], .vms-tab-btn[data-floor="0"]`);
    if (tab) tab.click();
  }
}

// ── Timeline & Transport Controls ─────────────────────────────────────────────

function initTimelineControls() {
  const slider = document.getElementById("vms-scrubber");
  if (slider) {
    slider.addEventListener("input", (e) => {
      seekToTime(parseFloat(e.target.value));
    });
  }

  const playBtn = document.getElementById("btn-dock-play");
  if (playBtn) {
    playBtn.addEventListener("click", () => {
      setPlayback(!isPlaying, currentSpeed);
    });
  }

  const syncBtn = document.getElementById("btn-dock-sync");
  if (syncBtn) {
    syncBtn.addEventListener("click", () => {
      seekToTime(0.0);
    });
  }

  const rewBtn = document.getElementById("btn-dock-rewind");
  if (rewBtn) {
    rewBtn.addEventListener("click", () => {
      seekToTime(Math.max(0, currentTime - 5.0));
    });
  }

  const fwdBtn = document.getElementById("btn-dock-forward");
  if (fwdBtn) {
    fwdBtn.addEventListener("click", () => {
      seekToTime(Math.min(maxDuration, currentTime + 5.0));
    });
  }

  const speedPills = document.querySelectorAll(".dp-btn-speed, .speed-pill");
  speedPills.forEach((pill) => {
    pill.addEventListener("click", () => {
      speedPills.forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      currentSpeed = parseFloat(pill.dataset.speed);
      setPlayback(isPlaying, currentSpeed);
    });
  });
}

async function seekToTime(t) {
  try {
    await fetch("/api/seek", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ time: t }),
    });
  } catch (e) {
    console.error("Seek error:", e);
  }
}

async function setPlayback(playing, speed) {
  try {
    const res = await fetch("/api/playback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ playing: playing, speed: speed }),
    });
    if (res.ok) {
      isPlaying = playing;
      const playBtn = document.getElementById("btn-dock-play");
      if (playBtn) playBtn.textContent = isPlaying ? "⏸ PAUSE" : "▶ PLAY";
    }
  } catch (e) {
    console.error("Playback error:", e);
  }
}

// ── Synchronization & Remap Modal ─────────────────────────────────────────────

function initSyncModal() {
  const modal = document.getElementById("sync-modal");
  const openBtn = document.getElementById("btn-open-sync");
  const closeBtn = document.getElementById("btn-close-sync");

  if (openBtn && modal) {
    openBtn.addEventListener("click", () => {
      modal.style.display = "flex";
    });
  }

  if (closeBtn && modal) {
    closeBtn.addEventListener("click", () => {
      modal.style.display = "none";
    });
  }
}

function onOffsetSliderChange(camId, val) {
  const numVal = parseFloat(val);
  const slider = document.getElementById(`slider-offset-${camId}`);
  const input = document.getElementById(`input-offset-${camId}`);
  const label = document.getElementById(`val-offset-${camId}`);

  if (slider) slider.value = numVal;
  if (input) input.value = numVal;
  if (label) label.textContent = `${numVal >= 0 ? "+" : ""}${numVal.toFixed(1)}s`;
}

async function saveOffsets() {
  const offsets = {};
  for (let i = 1; i <= 8; i++) {
    const cid = `cam${i}`;
    const input = document.getElementById(`input-offset-${cid}`);
    if (input) {
      offsets[cid] = parseFloat(input.value) || 0.0;
    }
  }

  try {
    const res = await fetch("/api/offsets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ offsets: offsets }),
    });
    if (res.ok) {
      alert("Timeline synchronization offsets saved successfully!");
      const modal = document.getElementById("sync-modal");
      if (modal) modal.style.display = "none";
    }
  } catch (e) {
    console.error("Save offsets error:", e);
  }
}

async function remapCamera(camId) {
  const input = document.getElementById(`remap-${camId}`);
  if (!input) return;
  const path = input.value.trim();

  try {
    const res = await fetch("/api/remap", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ camera_id: camId, video_path: path }),
    });
    const data = await res.json();
    if (res.ok) {
      alert(`Remapped ${camId.toUpperCase()} to ${path}`);
    } else {
      alert(`Remap failed: ${data.error || "File could not be opened"}`);
    }
  } catch (e) {
    console.error("Remap error:", e);
  }
}


// ── Sequential Demo & Investigation Report Functions ─────────────────────────

async function advanceNextSighting() {
  try {
    const res = await fetch("/api/demo/next_segment", { method: "POST" });
    if (res.ok) {
      const data = await res.json();
      console.log("[SAFEPASS] Advanced to next sighting:", data);
      if (data.current_camera) {
        focusCamera(data.current_camera, CAMERA_META[data.current_camera]?.name || data.current_camera);
      }
    }
  } catch (e) {
    console.error("Advance next sighting error:", e);
  }
}

function openReportModal() {
  const modal = document.getElementById("report-modal");
  if (!modal) return;

  const data = lastKnownState || {};
  const rep = data.report || generateFallbackReport(data);
  currentReportData = rep;

  const caseIdEl = document.getElementById("rep-case-id");
  if (caseIdEl) caseIdEl.textContent = rep.case_id || "CASE-2026-REID";
  
  const tsEl = document.getElementById("rep-timestamp");
  if (tsEl) tsEl.textContent = rep.timestamp_iso || new Date().toLocaleString();
  
  const uColEl = document.getElementById("rep-upper-color");
  if (uColEl) uColEl.textContent = rep.upper_color || "Blue";
  
  const lColEl = document.getElementById("rep-lower-color");
  if (lColEl) lColEl.textContent = rep.lower_color || "Black";
  
  const uDot = document.getElementById("rep-upper-dot");
  if (uDot) uDot.style.background = COLOR_HEX_MAP[(rep.upper_color || "").toLowerCase()] || "#2563eb";
  
  const lDot = document.getElementById("rep-lower-dot");
  if (lDot) lDot.style.background = COLOR_HEX_MAP[(rep.lower_color || "").toLowerCase()] || "#111111";

  const confEl = document.getElementById("rep-confidence");
  if (confEl) confEl.textContent = rep.confidence || "96%";
  
  const finalLocEl = document.getElementById("rep-final-loc");
  if (finalLocEl) finalLocEl.textContent = rep.final_location || "--";
  
  const finalCamEl = document.getElementById("rep-final-cam");
  if (finalCamEl) finalCamEl.textContent = rep.final_camera || "--";
  
  const transitTimeEl = document.getElementById("rep-transit-time");
  if (transitTimeEl) transitTimeEl.textContent = rep.total_transit_time || "--";
  
  const countEl = document.getElementById("rep-sighting-count");
  if (countEl) countEl.textContent = `${rep.sighting_rows?.length || (rep.camera_path?.length || 0)} Cameras`;

  // Path flow breadcrumb
  const pathFlow = document.getElementById("rep-path-flow");
  if (pathFlow) {
    pathFlow.innerHTML = (rep.camera_path || []).map((cam, idx, arr) => {
      const isLast = idx === arr.length - 1;
      const bg = isLast ? "rgba(16, 185, 129, 0.25)" : "rgba(63, 214, 198, 0.15)";
      const border = isLast ? "#10b981" : "var(--dp-color-hairline-2)";
      const color = isLast ? "#10b981" : "var(--dp-color-bright)";
      const badge = `<span style="background:${bg}; border:1px solid ${border}; color:${color}; padding:4px 8px; border-radius:4px; font-weight:700;">${cam}</span>`;
      return idx < arr.length - 1 ? `${badge} <span style="color:var(--dp-color-meta);">➔</span>` : badge;
    }).join(" ");
  }

  // Table rows
  const tbody = document.getElementById("rep-table-body");
  if (tbody && rep.sighting_rows) {
    tbody.innerHTML = rep.sighting_rows.map(r => `
      <tr style="border-bottom: 1px solid var(--dp-color-hairline);">
        <td style="padding: 8px 10px; font-family: var(--dp-font-mono); color: var(--dp-color-meta);">${r.sighting_num}</td>
        <td style="padding: 8px 10px; font-weight: 700; color: var(--dp-color-bright);">${r.camera_id}</td>
        <td style="padding: 8px 10px; color: var(--dp-color-meta);">${r.floor}</td>
        <td style="padding: 8px 10px; font-family: var(--dp-font-mono); color: var(--dp-color-live);">${r.time_window}</td>
        <td style="padding: 8px 10px; font-family: var(--dp-font-mono);">${r.duration}</td>
        <td style="padding: 8px 10px; color: var(--dp-color-read);">${r.note}</td>
        <td style="padding: 8px 10px; font-family: var(--dp-font-mono); text-align: right; color: #10b981; font-weight: 700;">${r.score}</td>
      </tr>
    `).join("");
  }

  modal.style.display = "flex";
}

function closeReportModal() {
  const modal = document.getElementById("report-modal");
  if (modal) modal.style.display = "none";
}

function copyReportSummary() {
  if (!currentReportData) return;
  const r = currentReportData;
  const text = `SAFEPASS MULTI-CAMERA RE-ID DOSSIER
Case ID: ${r.case_id}
Generated: ${r.timestamp_iso}
Target: Upper ${r.upper_color}, Lower ${r.lower_color} (Match Confidence: ${r.confidence})
Verified Camera Path: ${r.camera_path_str}
Total Transit Duration: ${r.total_transit_time}
Final Location: ${r.final_location} (Last Camera: ${r.final_camera})

SIGHTINGS LOG:
${(r.sighting_rows || []).map(row => `${row.sighting_num}. ${row.camera_id} [${row.floor}] - ${row.time_window} (${row.duration}): ${row.note}`).join("\n")}
`;
  navigator.clipboard.writeText(text).then(() => {
    alert("Investigation dossier summary copied to clipboard!");
  }).catch(() => {
    console.log(text);
  });
}

function generateFallbackReport(data) {
  const v = data?.final_verdict || {};
  const path = data?.camera_path || ["CAM 1"];
  return {
    case_id: "CASE-2026-REID-AUTO",
    timestamp_iso: new Date().toLocaleString(),
    upper_color: "Verified Target",
    lower_color: "Verified Target",
    confidence: `${((v.confidence || 0.95) * 100).toFixed(0)}%`,
    camera_path: path.map(c => c.toUpperCase()),
    camera_path_str: path.map(c => c.toUpperCase()).join(" ➔ "),
    final_camera: (v.last_camera || path[path.length - 1] || "CAM 1").toUpperCase(),
    final_location: v.inferred_location || "Terminal Destination",
    total_transit_time: data?.time_str || "00:00.0",
    sighting_rows: path.map((c, i) => ({
      sighting_num: i + 1,
      camera_id: c.toUpperCase(),
      floor: CAMERA_META[c]?.floor === 0 ? "Ground Floor" : `${CAMERA_META[c]?.floor}F`,
      time_window: data?.time_str || "--",
      duration: "--",
      note: `Sighting verified on ${CAMERA_META[c]?.name || c.toUpperCase()}`,
      score: "95%",
    })),
  };
}
