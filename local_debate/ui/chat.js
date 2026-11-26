// --------------------------------------------------
// Real viewport height fix (Android Chrome / iOS Safari)
// --------------------------------------------------

function setVH() {
    document.documentElement.style.setProperty('--vh', `${window.innerHeight * 0.01}px`);
}
setVH();
window.addEventListener('resize', setVH);

// --------------------------------------------------
// Backend Status
// --------------------------------------------------

async function checkBackendStatus() {
    try {
        const res = await fetch("http://localhost:5050/debate_state", {
            signal: AbortSignal.timeout(3000)
        });
        if (res.ok) {
            document.getElementById("offline-screen").classList.add("hidden");
        } else { showOffline(); }
    } catch (e) { showOffline(); }
}

function showOffline() {
    document.getElementById("offline-screen").classList.remove("hidden");
}

checkBackendStatus();

// --------------------------------------------------
// Theme Toggle
// --------------------------------------------------

const themeBtn   = document.getElementById("theme-toggle-btn");
const themeIcon  = document.getElementById("theme-icon");
const themeLabel = themeBtn.querySelector(".sidebar-footer-label");

function applyTheme(mode) {
    if (mode === "light") {
        document.body.classList.add("light-mode");
        themeIcon.textContent  = "☀️";
        themeLabel.textContent = "Light Mode";
    } else {
        document.body.classList.remove("light-mode");
        themeIcon.textContent  = "🌙";
        themeLabel.textContent = "Dark Mode";
    }
}

applyTheme(localStorage.getItem("ramdos-theme") || "dark");

themeBtn.addEventListener("click", () => {
    const next = document.body.classList.contains("light-mode") ? "dark" : "light";
    localStorage.setItem("ramdos-theme", next);
    applyTheme(next);
});

// --------------------------------------------------
// Sidebar — desktop collapse + mobile overlay + swipe
// --------------------------------------------------

const sidebar       = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebar-toggle-btn");
const backdrop      = document.getElementById("sidebar-backdrop");

const MOBILE_BP = 768;

function isMobile() { return window.innerWidth <= MOBILE_BP; }

let sidebarOpen = true;

function setSidebar(open, save = true) {
    sidebarOpen = open;

    if (isMobile()) {
        sidebar.classList.toggle("mobile-open", open);
        backdrop.classList.toggle("visible", open);
        sidebar.classList.remove("collapsed");
    } else {
        sidebar.classList.toggle("collapsed", !open);
        sidebar.classList.remove("mobile-open");
        backdrop.classList.remove("visible");
        if (save) localStorage.setItem("ramdos-sidebar", open ? "open" : "closed");
        updateSummaryOffset();
    }
}

// Restore desktop preference on load
if (!isMobile()) {
    const pref = localStorage.getItem("ramdos-sidebar");
    setSidebar(pref !== "closed", false);
}

// Hamburger toggle
sidebarToggle.addEventListener("click", () => setSidebar(!sidebarOpen));

// Close on backdrop tap (mobile)
backdrop.addEventListener("click", () => setSidebar(false));

// Auto-close mobile sidebar after picking mode or AI
function closeMobileIfNeeded() {
    if (isMobile()) setSidebar(false);
}

// Re-evaluate on screen resize / rotation
window.addEventListener("resize", () => {
    setVH();
    if (!isMobile()) {
        sidebar.classList.remove("mobile-open");
        backdrop.classList.remove("visible");
        const pref = localStorage.getItem("ramdos-sidebar");
        sidebar.classList.toggle("collapsed", pref === "closed");
        sidebarOpen = pref !== "closed";
    }
});

// --------------------------------------------------
// Swipe gestures (mobile)
// --------------------------------------------------

let touchStartX = 0;
let touchStartY = 0;
let edgeSwipe   = false;

// Swipe right from left edge → open
document.addEventListener("touchstart", e => {
    touchStartX = e.touches[0].clientX;
    touchStartY = e.touches[0].clientY;
    edgeSwipe   = isMobile() && touchStartX < 24;
}, { passive: true });

document.addEventListener("touchend", e => {
    if (!edgeSwipe) return;
    edgeSwipe = false;
    const dx = e.changedTouches[0].clientX - touchStartX;
    const dy = Math.abs(e.changedTouches[0].clientY - touchStartY);
    if (dx > 48 && dy < 80) setSidebar(true);
}, { passive: true });

// Swipe left on open sidebar → close
sidebar.addEventListener("touchstart", e => {
    touchStartX = e.touches[0].clientX;
    touchStartY = e.touches[0].clientY;
}, { passive: true });

sidebar.addEventListener("touchend", e => {
    if (!isMobile()) return;
    const dx = e.changedTouches[0].clientX - touchStartX;
    const dy = Math.abs(e.changedTouches[0].clientY - touchStartY);
    if (dx < -48 && dy < 80) setSidebar(false);
}, { passive: true });

// --------------------------------------------------
// Backend URL
// --------------------------------------------------

const BACKEND = "http://localhost:5050";

// --------------------------------------------------
// Mode & AI state
// --------------------------------------------------

let currentMode = "debate";
let selectedAI  = "ram";

// --------------------------------------------------
// Debate state
// --------------------------------------------------

let debateActive   = false;
let debatePaused   = false;
let currentTopic   = "";
let selectedTurns  = 10;
let summaryPolling = null;

// --------------------------------------------------
// Elements
// --------------------------------------------------

const messagesEl       = document.getElementById("messages");
const form             = document.getElementById("chat-form");
const input            = document.getElementById("user-input");
const sendBtn          = document.getElementById("send-btn");
const startBtn         = document.getElementById("start-btn");
const pauseBtn         = document.getElementById("pause-btn");
const resumeBtn        = document.getElementById("resume-btn");
const stopBtn          = document.getElementById("stop-btn");
const debateControlsEl = document.getElementById("debate-controls");
const modeStatus       = document.getElementById("mode-status");
const modeTitle        = document.getElementById("mode-title");
const modeBtns         = document.querySelectorAll(".model-btn");
const aiSelector       = document.getElementById("ai-selector");
const aiBtns           = document.querySelectorAll(".ai-btn");

// --------------------------------------------------
// Mode Switching
// --------------------------------------------------

modeBtns.forEach(btn => {
    btn.addEventListener("click", () => {
        modeBtns.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        switchMode(btn.getAttribute("data-mode"));
        closeMobileIfNeeded();
    });
});

function switchMode(mode) {
    currentMode = mode;
    messagesEl.innerHTML = "";
    removeSummaryPanel();

    if (mode === "chat") {
        updateChatModeTitle();
        modeStatus.textContent = "Status: Ready";
        input.placeholder      = "Type your message...";
        sendBtn.textContent    = "Send";
        debateControlsEl.classList.add("hidden");
        aiSelector.classList.remove("hidden");
        addMessage("system", `Chat mode activated. Now chatting with ${selectedAI === "ram" ? "Ram" : "Krishna"}.`);

    } else {
        modeTitle.textContent  = "Ram vs Krishna";
        modeStatus.textContent = "Status: Idle";
        input.placeholder      = "Enter debate topic...";
        sendBtn.textContent    = "Set Topic";
        debateControlsEl.classList.remove("hidden");
        aiSelector.classList.add("hidden");
        debateActive = false;
        debatePaused = false;
        currentTopic = "";
        injectTurnSelector();
        addMessage("system", "Debate mode activated. Set a topic to begin.");
    }
}

// --------------------------------------------------
// User Role (future: set this from your auth system)
// "admin" = all turns unlocked
// "free"  = only 5 turns, rest locked (subscription required)
// --------------------------------------------------

const USER_ROLE = "admin"; // TODO: replace with real auth role from your backend

const TURN_OPTIONS = [
    { value: 5,  label: "5 Turns",  minRole: "free"  },
    { value: 10, label: "10 Turns", minRole: "paid"  },
    { value: 15, label: "15 Turns", minRole: "paid"  },
    { value: 20, label: "20 Turns", minRole: "admin" },
];

function isUnlocked(minRole) {
    const hierarchy = ["free", "paid", "admin"];
    return hierarchy.indexOf(USER_ROLE) >= hierarchy.indexOf(minRole);
}

// --------------------------------------------------
// Turn Selector
// --------------------------------------------------

function injectTurnSelector() {
    document.getElementById("turn-selector-wrap")?.remove();

    const wrap = document.createElement("div");
    wrap.id = "turn-selector-wrap";
    wrap.style.cssText = `
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        color: #94a3b8;
        position: relative;
    `;

    // Build dropdown options HTML
    const optionsHTML = TURN_OPTIONS.map(opt => {
        const locked = !isUnlocked(opt.minRole);
        return `
            <div class="turns-option ${locked ? "turns-locked" : ""} ${opt.value === selectedTurns ? "turns-selected" : ""}"
                 data-value="${opt.value}" data-locked="${locked}">
                <span class="turns-option-label">${opt.label}</span>
                ${locked ? `<span class="turns-lock-badge">🔒 Pro</span>` : ""}
            </div>
        `;
    }).join("");

    wrap.innerHTML = `
        <span style="white-space:nowrap;font-weight:500;">Turns:</span>
        <div id="turns-dropdown" class="turns-dropdown">
            <button id="turns-trigger" class="turns-trigger">
                <span id="turns-trigger-label">${selectedTurns} Turns</span>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
            </button>
            <div id="turns-menu" class="turns-menu hidden">
                ${optionsHTML}
            </div>
        </div>
    `;

    debateControlsEl.appendChild(wrap);

    // Toggle dropdown
    document.getElementById("turns-trigger").addEventListener("click", (e) => {
        e.stopPropagation();
        document.getElementById("turns-menu").classList.toggle("hidden");
        document.getElementById("turns-trigger").classList.toggle("open");
    });

    // Option click
    document.querySelectorAll(".turns-option").forEach(el => {
        el.addEventListener("click", () => {
            if (el.dataset.locked === "true") {
                showUpgradeHint(el);
                return;
            }
            setTurns(parseInt(el.dataset.value));
            document.getElementById("turns-menu").classList.add("hidden");
            document.getElementById("turns-trigger").classList.remove("open");
        });
    });

    // Close on outside click
    document.addEventListener("click", () => {
        document.getElementById("turns-menu")?.classList.add("hidden");
        document.getElementById("turns-trigger")?.classList.remove("open");
    });
}

function setTurns(n) {
    selectedTurns = n;
    const label = document.getElementById("turns-trigger-label");
    if (label) label.textContent = `${n} Turns`;
    document.querySelectorAll(".turns-option").forEach(el => {
        el.classList.toggle("turns-selected", parseInt(el.dataset.value) === n);
    });
}

function showUpgradeHint(el) {
    // Remove any existing hint
    document.getElementById("upgrade-hint")?.remove();
    const hint = document.createElement("div");
    hint.id = "upgrade-hint";
    hint.style.cssText = `
        position: absolute;
        top: calc(100% + 6px);
        left: 0;
        background: #1e293b;
        border: 1px solid #4f46e5;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 11px;
        color: #a78bfa;
        white-space: nowrap;
        z-index: 200;
        animation: fadeIn 0.15s ease;
    `;
    hint.textContent = "🔒 Subscribe to unlock more turns";
    document.getElementById("turns-dropdown").appendChild(hint);
    setTimeout(() => hint.remove(), 2500);
}

// --------------------------------------------------
// AI Selection
// --------------------------------------------------

aiBtns.forEach(btn => {
    btn.addEventListener("click", () => {
        aiBtns.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        selectedAI = btn.getAttribute("data-ai");
        if (currentMode === "chat") {
            updateChatModeTitle();
            addMessage("system", `Now chatting with ${selectedAI === "ram" ? "Ram" : "Krishna"}.`);
        }
        closeMobileIfNeeded();
    });
});

function updateChatModeTitle() {
    modeTitle.textContent = selectedAI === "ram"
        ? "🛡️ Ram (Risk Analyst)"
        : "⚡ Krishna (Optimization Analyst)";
}

// --------------------------------------------------
// Messages
// --------------------------------------------------

function addMessage(role, text) {
    const row    = document.createElement("div");
    const bubble = document.createElement("div");
    row.classList.add("message-row", role);
    bubble.classList.add("message-bubble");
    bubble.textContent = text;
    row.appendChild(bubble);
    messagesEl.appendChild(row);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

// --------------------------------------------------
// Summary Panel
// --------------------------------------------------

function getSummaryOffset() {
    if (isMobile()) return "0px";
    return sidebarOpen ? "260px" : "0px";
}

function updateSummaryOffset() {
    const panel = document.getElementById("summary-panel");
    if (panel) panel.style.left = getSummaryOffset();
}

function showSummaryPanel(summaryText) {
    removeSummaryPanel();

    const panel = document.createElement("div");
    panel.id = "summary-panel";
    panel.style.cssText = `
        position: fixed;
        bottom: -400px;
        left: ${getSummaryOffset()};
        right: 0;
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border-top: 2px solid #4f46e5;
        padding: 20px 28px;
        z-index: 100;
        transition: bottom 0.35s cubic-bezier(0.4,0,0.2,1), left 0.28s cubic-bezier(0.4,0,0.2,1);
        max-height: 320px;
        overflow-y: auto;
    `;

    const points = summaryText.split("\n").map(l => l.trim()).filter(l => l.startsWith("-"));
    const pointsHTML = points.map(p =>
        `<li style="margin-bottom:8px;color:#cbd5e1;line-height:1.5;">${p.slice(1).trim()}</li>`
    ).join("");

    panel.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
            <h3 style="color:#a78bfa;margin:0;font-size:15px;">📋 Debate Summary</h3>
            <button id="close-summary" style="background:transparent;border:1px solid #334155;color:#94a3b8;padding:4px 10px;border-radius:6px;cursor:pointer;font-size:12px;">Close</button>
        </div>
        <ul style="list-style:none;padding:0;margin:0;max-width:680px;">
            ${pointsHTML || `<li style="color:#475569;">Summary not available.</li>`}
        </ul>
    `;

    document.body.appendChild(panel);
    requestAnimationFrame(() => { panel.style.bottom = "0px"; });
    document.getElementById("close-summary").addEventListener("click", removeSummaryPanel);
}

function removeSummaryPanel() {
    const panel = document.getElementById("summary-panel");
    if (panel) {
        panel.style.bottom = "-400px";
        setTimeout(() => panel.remove(), 380);
    }
}

// --------------------------------------------------
// Poll for Summary
// --------------------------------------------------

function startSummaryPolling() {
    if (summaryPolling) clearInterval(summaryPolling);
    summaryPolling = setInterval(async () => {
        try {
            const res  = await fetch(`${BACKEND}/debate_summary`);
            const data = await res.json();
            if (data.ready) {
                clearInterval(summaryPolling);
                summaryPolling = null;
                showSummaryPanel(data.summary);
            }
        } catch (err) { console.error("Summary poll error:", err); }
    }, 2000);
}

// --------------------------------------------------
// Form Submission
// --------------------------------------------------

form.addEventListener("submit", async e => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    input.value = "";

    if (currentMode === "chat") {
        addMessage("user", text);
        modeStatus.textContent = "Status: Thinking…";
        try {
            const res  = await fetch(`${BACKEND}/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text, ai: selectedAI })
            });
            const data = await res.json();
            addMessage(selectedAI, data.response);
            modeStatus.textContent = "Status: Ready";
        } catch {
            addMessage("system", "Connection failed.");
            modeStatus.textContent = "Status: Error";
        }

    } else {
        currentTopic = text;
        addMessage("user", `Topic: ${text}`);
        modeStatus.textContent = "Status: Topic Set";
        await fetch(`${BACKEND}/debate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "topic", topic: text })
        });
    }
});

// --------------------------------------------------
// Debate Controls
// --------------------------------------------------

startBtn.addEventListener("click", async () => {
    if (!currentTopic) { addMessage("system", "Set a topic first."); return; }
    if (!debateActive) {
        debateActive = true;
        debatePaused = false;
        removeSummaryPanel();
        modeStatus.textContent = "Status: Running";
        addMessage("system", `Debate started. (${selectedTurns} turns)`);
        await fetch(`${BACKEND}/debate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "start", turns: selectedTurns })
        });
    }
});

pauseBtn.addEventListener("click", async () => {
    if (debateActive && !debatePaused) {
        debatePaused = true;
        modeStatus.textContent = "Status: Paused";
        addMessage("system", "Debate paused.");
        await sendControl("pause");
    }
});

resumeBtn.addEventListener("click", async () => {
    if (debateActive && debatePaused) {
        debatePaused = false;
        modeStatus.textContent = "Status: Running";
        addMessage("system", "Debate resumed.");
        await sendControl("resume");
    }
});

stopBtn.addEventListener("click", async () => {
    if (debateActive) {
        debateActive = false;
        debatePaused = false;
        modeStatus.textContent = "Status: Stopped";
        addMessage("system", "Debate stopped.");
        await sendControl("stop");
        startSummaryPolling();
    }
});

async function sendControl(action) {
    try {
        await fetch(`${BACKEND}/debate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action })
        });
    } catch (err) {
        addMessage("system", "Backend connection failed.");
        console.error(err);
    }
}

// --------------------------------------------------
// Polling for Debate Messages
// --------------------------------------------------

setInterval(async () => {
    if (currentMode !== "debate" || !debateActive || debatePaused) return;
    try {
        const res  = await fetch(`${BACKEND}/debate_state`);
        if (!res.ok) return;
        const data = await res.json();
        if (data.new_message) addMessage(data.role, data.text);
    } catch (err) { console.error("Poll error:", err); }
}, 1500);

// --------------------------------------------------
// Watch for debate ending naturally
// --------------------------------------------------

let wasActive = false;
setInterval(() => {
    if (wasActive && !debateActive) startSummaryPolling();
    wasActive = debateActive;
}, 1000);

// --------------------------------------------------
// Initialize
// --------------------------------------------------

switchMode("debate");