/**
 * app.js — Frontend Application Logic for Voice-First AI Sales Agent Dashboard
 * Handles Microphone Recording, Web Audio Visualizer, REST & WebSocket API Calls,
 * Intent Intelligence rendering, and Product Knowledge management.
 */

document.addEventListener("DOMContentLoaded", () => {
    // ============================================================
    // STATE & DOM ELEMENTS
    // ============================================================
    let sessionId = localStorage.getItem("sales_session_id") || null;
    let isRecording = false;
    let mediaRecorder = null;
    let audioChunks = [];
    let audioContext = null;
    let analyser = null;
    let animFrameId = null;
    let ws = null;

    // DOM Elements
    const micOrbBtn = document.getElementById("micOrbBtn");
    const voiceStateTitle = document.getElementById("voiceStateTitle");
    const voiceStateSubtitle = document.getElementById("voiceStateSubtitle");
    const waveformCanvas = document.getElementById("waveformCanvas");
    const canvasCtx = waveformCanvas ? waveformCanvas.getContext("2d") : null;
    const conversationFeed = document.getElementById("conversationFeed");
    const textPromptInput = document.getElementById("textPromptInput");
    const btnSendPrompt = document.getElementById("btnSendPrompt");
    const inputMicToggle = document.getElementById("inputMicToggle");
    const btnNewSession = document.getElementById("btnNewSession");
    const sidebarSessionId = document.getElementById("sidebarSessionId");
    const toggleAutoTTS = document.getElementById("toggleAutoTTS");

    // Status & Intel Elements
    const callStatePill = document.getElementById("callStatePill");
    const callStateText = document.getElementById("callStateText");
    const langPill = document.getElementById("langPill");
    const topInterestBadge = document.getElementById("topInterestBadge");
    const topInterestText = document.getElementById("topInterestText");

    const intelInterestBadge = document.getElementById("intelInterestBadge");
    const intelConfidenceVal = document.getElementById("intelConfidenceVal");
    const intelConfidenceFill = document.getElementById("intelConfidenceFill");
    const intelIntentionVal = document.getElementById("intelIntentionVal");
    const intelHotStreak = document.getElementById("intelHotStreak");
    const intelColdStreak = document.getElementById("intelColdStreak");
    const actionScheduledVal = document.getElementById("actionScheduledVal");
    const actionWhatsAppVal = document.getElementById("actionWhatsAppVal");
    const postCallSummaryCard = document.getElementById("postCallSummaryCard");

    // ============================================================
    // INITIALIZATION
    // ============================================================
    initSession();
    initTabs();
    initWebSocket();
    loadProductKB();

    // ============================================================
    // SESSION MANAGEMENT
    // ============================================================
    async function initSession() {
        if (!sessionId) {
            sessionId = Math.random().toString(36).substring(2, 10);
            localStorage.setItem("sales_session_id", sessionId);
        }
        sidebarSessionId.textContent = sessionId;
        fetchSessionDetails(sessionId);
    }

    async function fetchSessionDetails(sId) {
        try {
            const res = await fetch(`/api/session/${sId}`);
            if (res.ok) {
                const data = await res.json();
                renderSessionState(data);
            }
        } catch (err) {
            console.warn("Could not fetch session detail:", err);
        }
    }

    if (btnNewSession) {
        btnNewSession.addEventListener("click", async () => {
            try {
                const formData = new FormData();
                formData.append("session_id", sessionId);
                const res = await fetch("/api/session/reset", { method: "POST", body: formData });
                const data = await res.json();
                sessionId = data.session_id;
                localStorage.setItem("sales_session_id", sessionId);
                sidebarSessionId.textContent = sessionId;

                // Reset UI Feed
                conversationFeed.innerHTML = `
                    <div class="feed-welcome">
                        <div class="welcome-sparkle">✦</div>
                        <h3>New Sales Call Started</h3>
                        <p>Speak into your microphone or type a message to begin.</p>
                    </div>
                `;
                postCallSummaryCard.classList.add("hidden");
                fetchSessionDetails(sessionId);
            } catch (e) {
                console.error("Failed to reset session:", e);
            }
        });
    }

    // ============================================================
    // TAB SWITCHING
    // ============================================================
    function initTabs() {
        const navItems = document.querySelectorAll(".nav-item");
        const tabContents = document.querySelectorAll(".tab-content");

        navItems.forEach(item => {
            item.addEventListener("click", () => {
                const targetTab = item.getAttribute("data-tab");
                navItems.forEach(n => n.classList.remove("active"));
                tabContents.forEach(c => c.classList.remove("active"));

                item.classList.add("active");
                const targetElem = document.getElementById(`tab-${targetTab}`);
                if (targetElem) targetElem.classList.add("active");

                if (targetTab === "pipeline") loadPipelineSessions();
                if (targetTab === "dispatches") loadWhatsAppLogs();
            });
        });
    }

    // ============================================================
    // MICROPHONE & AUDIO VISUALIZER
    // ============================================================
    if (micOrbBtn) {
        micOrbBtn.addEventListener("click", toggleRecording);
    }

    if (inputMicToggle) {
        inputMicToggle.addEventListener("click", toggleRecording);
    }

    async function toggleRecording() {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    }

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioChunks = [];
            
            // Audio Context for Waveform
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(stream);
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 64;
            source.connect(analyser);

            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) audioChunks.push(event.data);
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
                await sendAudioToBackend(audioBlob);
                stream.getTracks().forEach(t => t.stop());
                if (audioContext) audioContext.close();
            };

            mediaRecorder.start();
            isRecording = true;

            // UI State: Recording
            micOrbBtn.className = "mic-orb recording";
            voiceStateTitle.textContent = "Listening to Your Voice...";
            voiceStateSubtitle.textContent = "Speak clearly into your microphone";
            drawWaveform();

        } catch (err) {
            console.error("Microphone Access Denied:", err);
            alert("Could not access microphone. Please allow microphone permissions in your browser.");
        }
    }

    function stopRecording() {
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            isRecording = false;
            if (animFrameId) cancelAnimationFrame(animFrameId);

            // UI State: Processing
            micOrbBtn.className = "mic-orb processing";
            voiceStateTitle.textContent = "Transcribing & Analyzing Intent...";
            voiceStateSubtitle.textContent = "Running LangGraph Speech & Multi-Agent Reasoning";
        }
    }

    function drawWaveform() {
        if (!analyser || !canvasCtx || !isRecording) return;
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        function renderFrame() {
            if (!isRecording) return;
            animFrameId = requestAnimationFrame(renderFrame);
            analyser.getByteFrequencyData(dataArray);

            canvasCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
            const barWidth = (waveformCanvas.width / bufferLength) * 2;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const barHeight = (dataArray[i] / 255) * waveformCanvas.height;
                canvasCtx.fillStyle = "rgba(139, 92, 246, 0.7)";
                canvasCtx.fillRect(x, waveformCanvas.height - barHeight, barWidth - 2, barHeight);
                x += barWidth;
            }
        }
        renderFrame();
    }

    async function sendAudioToBackend(blob) {
        const formData = new FormData();
        formData.append("file", blob, "user_mic.wav");
        formData.append("session_id", sessionId);

        try {
            setPipelineStepActive("step-analyze");
            const res = await fetch("/api/audio-transcribe", { method: "POST", body: formData });
            if (!res.ok) throw new Error("Audio transcription failed");

            const data = await res.json();
            if (data.detected_language) {
                langPill.textContent = `🌐 ${data.detected_language.toUpperCase()}`;
            }

            renderTurnResult(data);
        } catch (err) {
            console.error("Error processing audio:", err);
            voiceStateTitle.textContent = "Speech Processing Error";
            voiceStateSubtitle.textContent = "Please try again or use text input";
            micOrbBtn.className = "mic-orb idle";
        }
    }

    // ============================================================
    // TEXT PROMPT CHAT
    // ============================================================
    if (btnSendPrompt) {
        btnSendPrompt.addEventListener("click", sendTextMessage);
    }

    if (textPromptInput) {
        textPromptInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") sendTextMessage();
        });
    }

    async function sendTextMessage() {
        const msg = textPromptInput.value.trim();
        if (!msg) return;
        textPromptInput.value = "";

        micOrbBtn.className = "mic-orb processing";
        voiceStateTitle.textContent = "Processing Query...";
        voiceStateSubtitle.textContent = "Running LangGraph Multi-Agent Sales Pipeline";
        setPipelineStepActive("step-analyze");

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: sessionId, user_message: msg })
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Error processing prompt");
            }

            const data = await res.json();
            renderTurnResult(data);
        } catch (e) {
            console.error(e);
            alert(e.message);
            micOrbBtn.className = "mic-orb idle";
            voiceStateTitle.textContent = "Tap Microphone to Speak";
        }
    }

    // ============================================================
    // RENDER RESPONSE & INTEL METRICS
    // ============================================================
    function renderTurnResult(data) {
        // Clear welcome screen if present
        const welcome = conversationFeed.querySelector(".feed-welcome");
        if (welcome) welcome.remove();

        // 1. User Bubble
        const userBubble = document.createElement("div");
        userBubble.className = "chat-bubble user";
        userBubble.innerHTML = `
            <div class="bubble-meta">
                <span>Customer</span> • <span>${data.detected_language ? data.detected_language.toUpperCase() : 'EN'}</span>
            </div>
            <div class="bubble-card">
                ${escapeHtml(data.user_message)}
            </div>
        `;
        conversationFeed.appendChild(userBubble);

        // 2. Assistant Bubble
        const assistantBubble = document.createElement("div");
        assistantBubble.className = "chat-bubble assistant";
        const audioBtnId = `btnAudio_${Math.random().toString(36).substr(2, 6)}`;
        assistantBubble.innerHTML = `
            <div class="bubble-meta">
                <span>AI Sales Copilot</span> • <span>Final Response</span>
            </div>
            <div class="bubble-card">
                ${escapeHtml(data.final_response)}
                ${data.audio_url ? `
                    <div>
                        <button class="bubble-audio-btn" id="${audioBtnId}">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                            Play Voice
                        </button>
                    </div>
                ` : ''}
            </div>
        `;
        conversationFeed.appendChild(assistantBubble);
        conversationFeed.scrollTop = conversationFeed.scrollHeight;

        // Attach Audio Playback listener
        if (data.audio_url) {
            const btnAudio = document.getElementById(audioBtnId);
            if (btnAudio) {
                btnAudio.addEventListener("click", () => playAudio(data.audio_url));
            }
            if (toggleAutoTTS.checked) {
                playAudio(data.audio_url);
            }
        }

        // 3. Update Intelligence Panel & State
        renderSessionState(data);

        // Set Orb Speaking / Idle State
        micOrbBtn.className = "mic-orb idle";
        voiceStateTitle.textContent = "Tap Microphone to Speak";
        voiceStateSubtitle.textContent = "Continuous Voice & Intent Intelligence Active";
        setPipelineStepActive("step-actions");
    }

    function playAudio(url) {
        micOrbBtn.className = "mic-orb speaking";
        voiceStateTitle.textContent = "AI Sales Agent Speaking...";
        const audio = new Audio(url);
        audio.play();
        audio.onended = () => {
            micOrbBtn.className = "mic-orb idle";
            voiceStateTitle.textContent = "Tap Microphone to Speak";
        };
    }

    function renderSessionState(state) {
        const interest = (state.interest || "warm").toLowerCase();
        
        // Header & Intelligence Badges
        const interestUpper = interest.toUpperCase();
        topInterestText.textContent = `${interestUpper} LEAD`;
        topInterestBadge.className = `lead-header-badge badge-${interest}`;

        intelInterestBadge.textContent = interestUpper;
        intelInterestBadge.className = `interest-badge-large badge-${interest}`;

        // Confidence
        const conf = state.confidence || 0.0;
        intelConfidenceVal.textContent = conf.toFixed(2);
        intelConfidenceFill.style.width = `${Math.min(100, conf * 100)}%`;

        // Intention & Streaks
        intelIntentionVal.textContent = state.intention || state.user_message || "Active Dialogue";
        intelHotStreak.textContent = state.hot_streak || 0;
        intelColdStreak.textContent = state.cold_streak || 0;

        // Actions: Scheduled Time & WhatsApp
        if (state.scheduled_time) {
            actionScheduledVal.textContent = `Scheduled: ${state.scheduled_time}`;
        } else {
            actionScheduledVal.textContent = "No call scheduled";
        }

        if (state.whatsapp_message) {
            actionWhatsAppVal.textContent = `Dispatched: "${state.whatsapp_message.substring(0, 30)}..."`;
        } else {
            actionWhatsAppVal.textContent = "No message queued";
        }

        // Call State
        if (state.call_active === false) {
            callStatePill.className = "status-pill status-ended";
            callStateText.textContent = "CALL TERMINATED";

            // Post-Call Intelligence Summary
            if (state.summary) {
                postCallSummaryCard.classList.remove("hidden");
                document.getElementById("postSummaryText").textContent = state.summary.summary || "Summary generated.";
                document.getElementById("postProductInterest").textContent = state.summary.product_interest || "N/A";
                document.getElementById("postObjections").textContent = Array.isArray(state.summary.objections) ? state.summary.objections.join(", ") : "None";
                document.getElementById("postFollowUp").textContent = state.summary.recommended_follow_up || "N/A";
            }
        } else {
            callStatePill.className = "status-pill status-active";
            callStateText.textContent = "LIVE CALL IN PROGRESS";
            postCallSummaryCard.classList.add("hidden");
        }
    }

    function setPipelineStepActive(stepId) {
        const steps = ["step-analyze", "step-agents", "step-judge", "step-actions"];
        steps.forEach(s => {
            const el = document.getElementById(s);
            if (el) el.classList.remove("active");
        });
        const activeEl = document.getElementById(stepId);
        if (activeEl) activeEl.classList.add("active");
    }

    // ============================================================
    // PRODUCT KNOWLEDGE BASE EDITOR
    // ============================================================
    async function loadProductKB() {
        try {
            const res = await fetch("/api/product");
            if (res.ok) {
                const prod = await res.json();
                document.getElementById("kbProductName").value = prod.name || "";
                document.getElementById("kbProductDesc").value = prod.description || "";
                document.getElementById("kbFeatures").value = Array.isArray(prod.features) ? prod.features.join(", ") : "";
                document.getElementById("kbBenefits").value = Array.isArray(prod.benefits) ? prod.benefits.join(", ") : "";
            }
        } catch (e) {
            console.warn("Could not load Product KB:", e);
        }
    }

    const btnSaveKB = document.getElementById("btnSaveKB");
    if (btnSaveKB) {
        btnSaveKB.addEventListener("click", async () => {
            const payload = {
                name: document.getElementById("kbProductName").value,
                description: document.getElementById("kbProductDesc").value,
                features: document.getElementById("kbFeatures").value.split(",").map(s => s.trim()),
                benefits: document.getElementById("kbBenefits").value.split(",").map(s => s.trim())
            };
            try {
                const res = await fetch("/api/product", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                if (res.ok) alert("Product Knowledge Base saved successfully!");
            } catch (e) {
                alert("Error saving Knowledge Base");
            }
        });
    }

    // ============================================================
    // LOAD PIPELINE & WHATSAPP LOGS
    // ============================================================
    async function loadPipelineSessions() {
        const grid = document.getElementById("pipelineSessionsGrid");
        if (!grid) return;
        grid.innerHTML = `<div style="color:var(--text-tertiary);">Loading sessions...</div>`;

        try {
            const res = await fetch("/api/sessions");
            const data = await res.json();
            if (data.sessions.length === 0) {
                grid.innerHTML = `<div style="color:var(--text-tertiary);">No sessions recorded yet.</div>`;
                return;
            }

            grid.innerHTML = data.sessions.map(s => `
                <div class="session-card" style="margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <strong>Session #${s.session_id}</strong>
                        <span class="lead-header-badge badge-${s.interest.toLowerCase()}">${s.interest.toUpperCase()}</span>
                    </div>
                    <div style="font-size:12px; color:var(--text-secondary);">
                        Turns: ${s.turn_count} | Status: ${s.call_active ? 'Active' : 'Ended'}
                    </div>
                </div>
            `).join("");
        } catch (e) {
            grid.innerHTML = `<div style="color:red;">Error loading sessions.</div>`;
        }
    }

    async function loadWhatsAppLogs() {
        const container = document.getElementById("whatsappLogsContainer");
        if (!container) return;
        container.innerHTML = `<div style="color:var(--text-tertiary);">Loading dispatches...</div>`;

        try {
            const res = await fetch("/api/whatsapp-logs");
            const data = await res.json();
            if (!data.logs || data.logs.length === 0) {
                container.innerHTML = `<div style="color:var(--text-tertiary);">No WhatsApp messages dispatched yet. Highly interested (HOT) leads or scheduled calls automatically trigger WhatsApp follow-ups here.</div>`;
                return;
            }

            container.innerHTML = data.logs.map(log => `
                <div class="intel-card" style="margin-bottom:10px;">
                    <div style="font-size:11px; color:var(--text-tertiary);">Session #${log.session_id}</div>
                    <div style="font-size:13px; color:var(--text-primary); margin-top:4px;">${escapeHtml(log.message)}</div>
                </div>
            `).join("");
        } catch (e) {
            container.innerHTML = `<div style="color:red;">Error loading dispatches.</div>`;
        }
    }

    // ============================================================
    // WEBSOCKET CLIENT
    // ============================================================
    function initWebSocket() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/ws/call`;
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log("WebSocket connected.");
            ws.send(JSON.stringify({ action: "join_session", session_id: sessionId }));
        };

        ws.onmessage = (evt) => {
            try {
                const msg = JSON.parse(evt.data);
                if (msg.type === "node_executing") {
                    setPipelineStepActive(`step-${msg.data.node}`);
                }
            } catch (e) {
                console.error(e);
            }
        };

        ws.onclose = () => {
            setTimeout(initWebSocket, 3000);
        };
    }

    function escapeHtml(text) {
        if (!text) return "";
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
