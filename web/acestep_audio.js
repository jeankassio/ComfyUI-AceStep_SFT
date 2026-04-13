import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const ACESTEP_AUDIO_NODES = ["AceStepSFTSaveAudio", "AceStepSFTPreviewAudio"];

const DARK_BG = "#353535";
const WAVE_COLOR = "#a6a6a6";
const PROGRESS_COLOR = "#0d1b2a";
const PLAY_BTN_BG = "#ffffff";
const PLAY_BTN_ICON = "#353535";
const TIME_COLOR = "#b0bec5";
const WIDGET_MIN_HEIGHT = 100;

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
}

function drawPlayButton(ctx, cx, cy, r, playing) {
    // White circle
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = PLAY_BTN_BG;
    ctx.fill();

    ctx.fillStyle = PLAY_BTN_ICON;
    if (playing) {
        // Pause icon
        const bw = r * 0.25;
        const bh = r * 0.8;
        ctx.fillRect(cx - r * 0.35, cy - bh / 2, bw, bh);
        ctx.fillRect(cx + r * 0.1, cy - bh / 2, bw, bh);
    } else {
        // Play triangle
        ctx.beginPath();
        const s = r * 0.8;
        ctx.moveTo(cx - s * 0.3, cy - s * 0.5);
        ctx.lineTo(cx - s * 0.3, cy + s * 0.5);
        ctx.lineTo(cx + s * 0.55, cy);
        ctx.closePath();
        ctx.fill();
    }
}

function drawWaveform(ctx, x, y, w, h, peaks, progress) {
    if (!peaks || peaks.length === 0) return;

    const barCount = Math.min(peaks.length, Math.floor(w / 3));
    const barWidth = w / barCount;
    const gap = Math.max(1, barWidth * 0.2);

    for (let i = 0; i < barCount; i++) {
        const peakIdx = Math.floor((i / barCount) * peaks.length);
        const amplitude = peaks[peakIdx];
        const barH = Math.max(2, amplitude * h * 0.9);
        const bx = x + i * barWidth;
        const by = y + (h - barH) / 2;

        const pct = i / barCount;
        ctx.fillStyle = pct <= progress ? PROGRESS_COLOR : WAVE_COLOR;
        ctx.fillRect(bx, by, Math.max(1, barWidth - gap), barH);
    }
}

function computePeaks(audioBuffer, numPeaks) {
    const data = audioBuffer.getChannelData(0);
    const blockSize = Math.floor(data.length / numPeaks);
    const peaks = new Float32Array(numPeaks);
    for (let i = 0; i < numPeaks; i++) {
        let max = 0;
        const start = i * blockSize;
        const end = Math.min(start + blockSize, data.length);
        for (let j = start; j < end; j++) {
            const abs = Math.abs(data[j]);
            if (abs > max) max = abs;
        }
        peaks[i] = max;
    }
    return peaks;
}

class AceStepAudioWidget {
    constructor(node) {
        this.node = node;
        this.audioCtx = null;
        this.audioBuffer = null;
        this.source = null;
        this.peaks = null;
        this.playing = false;
        this.startTime = 0;
        this.pauseOffset = 0;
        this.duration = 0;
        this.progress = 0;
        this.animFrame = null;
        this.audioUrl = null;
    }

    async loadAudio(url) {
        if (this.audioUrl === url && this.audioBuffer) return;
        this.stop();
        this.audioUrl = url;

        if (!this.audioCtx) {
            this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }

        try {
            const resp = await fetch(url);
            const arrayBuf = await resp.arrayBuffer();
            this.audioBuffer = await this.audioCtx.decodeAudioData(arrayBuf);
            this.duration = this.audioBuffer.duration;
            this.peaks = computePeaks(this.audioBuffer, 200);
            this.progress = 0;
            this.pauseOffset = 0;
            this.node.setDirtyCanvas(true);
        } catch (e) {
            console.error("[AceStep] Failed to load audio:", e);
        }
    }

    play() {
        if (!this.audioBuffer || !this.audioCtx) return;
        if (this.playing) return;

        if (this.audioCtx.state === "suspended") {
            this.audioCtx.resume();
        }

        this.source = this.audioCtx.createBufferSource();
        this.source.buffer = this.audioBuffer;
        this.source.connect(this.audioCtx.destination);
        this.source.onended = () => {
            if (this.playing) {
                this.playing = false;
                this.pauseOffset = 0;
                this.progress = 0;
                this.node.setDirtyCanvas(true);
                if (this.animFrame) {
                    cancelAnimationFrame(this.animFrame);
                    this.animFrame = null;
                }
            }
        };

        this.source.start(0, this.pauseOffset);
        this.startTime = this.audioCtx.currentTime - this.pauseOffset;
        this.playing = true;
        this._animate();
    }

    pause() {
        if (!this.playing) return;
        this.source.onended = null;
        this.source.stop();
        this.pauseOffset = this.audioCtx.currentTime - this.startTime;
        this.playing = false;
        if (this.animFrame) {
            cancelAnimationFrame(this.animFrame);
            this.animFrame = null;
        }
        this.node.setDirtyCanvas(true);
    }

    stop() {
        if (this.source) {
            try { this.source.onended = null; this.source.stop(); } catch {}
            this.source = null;
        }
        this.playing = false;
        this.pauseOffset = 0;
        this.progress = 0;
        if (this.animFrame) {
            cancelAnimationFrame(this.animFrame);
            this.animFrame = null;
        }
    }

    togglePlay() {
        if (this.playing) {
            this.pause();
        } else {
            this.play();
        }
    }

    seek(pct) {
        this.pauseOffset = Math.max(0, Math.min(1, pct)) * this.duration;
        this.progress = pct;
        if (this.playing) {
            this.source.onended = null;
            try { this.source.stop(); } catch {}
            this.source = this.audioCtx.createBufferSource();
            this.source.buffer = this.audioBuffer;
            this.source.connect(this.audioCtx.destination);
            this.source.onended = () => {
                if (this.playing) {
                    this.playing = false;
                    this.pauseOffset = 0;
                    this.progress = 0;
                    this.node.setDirtyCanvas(true);
                    if (this.animFrame) {
                        cancelAnimationFrame(this.animFrame);
                        this.animFrame = null;
                    }
                }
            };
            this.source.start(0, this.pauseOffset);
            this.startTime = this.audioCtx.currentTime - this.pauseOffset;
        }
        this.node.setDirtyCanvas(true);
    }

    _animate() {
        if (!this.playing) return;
        const elapsed = this.audioCtx.currentTime - this.startTime;
        this.progress = this.duration > 0 ? Math.min(1, elapsed / this.duration) : 0;
        this.node.setDirtyCanvas(true);
        this.animFrame = requestAnimationFrame(() => this._animate());
    }

    getCurrentTime() {
        if (this.playing && this.audioCtx) {
            return this.audioCtx.currentTime - this.startTime;
        }
        return this.pauseOffset;
    }

    destroy() {
        this.stop();
        if (this.audioCtx) {
            this.audioCtx.close().catch(() => {});
            this.audioCtx = null;
        }
    }
}

app.registerExtension({
    name: "AceStepSFT.AudioVisualizer",

    async nodeCreated(node) {
        if (!ACESTEP_AUDIO_NODES.includes(node.comfyClass)) return;

        const audioWidget = new AceStepAudioWidget(node);
        node._aceAudio = audioWidget;

        // Custom draw on node
        const origDraw = node.onDrawForeground;
        node.onDrawForeground = function (ctx) {
            if (origDraw) origDraw.call(this, ctx);

            const aw = this._aceAudio;
            if (!aw || !aw.peaks) return;

            // Calculate widget area height (above the visualizer)
            let widgetsBottom = 30; // header
            if (this.widgets) {
                for (const w of this.widgets) {
                    if (w.last_y != null) {
                        const wy = w.last_y + (w.computedHeight || LiteGraph.NODE_WIDGET_HEIGHT || 20);
                        if (wy > widgetsBottom) widgetsBottom = wy;
                    }
                }
            }

            const drawH = WIDGET_MIN_HEIGHT;
            const drawW = this.size[0];
            const vizY = widgetsBottom;
            const neededH = vizY + drawH + 5;

            // Only expand if needed, never shrink from user resize
            if (!this._aceVizMinH || neededH !== this._aceVizMinH) {
                this._aceVizMinH = neededH;
            }
            if (this.size[1] < this._aceVizMinH) {
                this.size[1] = this._aceVizMinH;
            }

            const vizH = drawH;
            const vizW = drawW;

            // Dark background
            ctx.fillStyle = DARK_BG;
            const r = 8;
            ctx.beginPath();
            ctx.moveTo(0 + r, vizY);
            ctx.lineTo(vizW - r, vizY);
            ctx.quadraticCurveTo(vizW, vizY, vizW, vizY + r);
            ctx.lineTo(vizW, vizY + vizH - r);
            ctx.quadraticCurveTo(vizW, vizY + vizH, vizW - r, vizY + vizH);
            ctx.lineTo(0 + r, vizY + vizH);
            ctx.quadraticCurveTo(0, vizY + vizH, 0, vizY + vizH - r);
            ctx.lineTo(0, vizY + r);
            ctx.quadraticCurveTo(0, vizY, 0 + r, vizY);
            ctx.closePath();
            ctx.fill();

            // Play button
            const btnR = 22;
            const btnCx = 10 + btnR + 5;
            const btnCy = vizY + vizH / 2;
            drawPlayButton(ctx, btnCx, btnCy, btnR, aw.playing);

            // Waveform area
            const waveX = btnCx + btnR + 15;
            const waveW = vizW - waveX - 10;
            const waveH = vizH - 30;
            const waveY = vizY + 10;

            drawWaveform(ctx, waveX, waveY, waveW, waveH, aw.peaks, aw.progress);

            // Time display
            const current = formatTime(aw.getCurrentTime());
            const total = formatTime(aw.duration);
            ctx.fillStyle = TIME_COLOR;
            ctx.font = "11px monospace";
            ctx.textAlign = "right";
            ctx.fillText(`${current} / ${total}`, vizW - 10, vizY + vizH - 8);
            ctx.textAlign = "left";

            // Store hit areas for click handling
            this._aceVizArea = { x: waveX, y: waveY, w: waveW, h: waveH, vizY, vizH };
            this._aceBtnArea = { cx: btnCx, cy: btnCy, r: btnR };
        };

        // Handle clicks
        const origMouse = node.onMouseDown;
        node.onMouseDown = function (e, localPos) {
            const aw = this._aceAudio;
            if (!aw || !aw.peaks) {
                if (origMouse) return origMouse.call(this, e, localPos);
                return;
            }

            const [mx, my] = localPos;

            // Check play button
            const btn = this._aceBtnArea;
            if (btn) {
                const dx = mx - btn.cx;
                const dy = my - btn.cy;
                if (dx * dx + dy * dy <= btn.r * btn.r) {
                    aw.togglePlay();
                    return true;
                }
            }

            // Check waveform seek
            const viz = this._aceVizArea;
            if (viz && mx >= viz.x && mx <= viz.x + viz.w && my >= viz.y && my <= viz.y + viz.h) {
                const pct = (mx - viz.x) / viz.w;
                aw.seek(pct);
                return true;
            }

            if (origMouse) return origMouse.call(this, e, localPos);
        };

        // Cleanup
        const origRemoved = node.onRemoved;
        node.onRemoved = function () {
            if (this._aceAudio) {
                this._aceAudio.destroy();
                this._aceAudio = null;
            }
            if (origRemoved) origRemoved.call(this);
        };
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!ACESTEP_AUDIO_NODES.includes(nodeData.name)) return;

        const origExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            if (origExecuted) origExecuted.call(this, output);

            if (!output || !output.audio || output.audio.length === 0) return;

            const audioInfo = output.audio[0];
            const url = api.apiURL(
                `/view?filename=${encodeURIComponent(audioInfo.filename)}&subfolder=${encodeURIComponent(audioInfo.subfolder)}&type=${encodeURIComponent(audioInfo.type)}`
            );

            if (this._aceAudio) {
                this._aceAudio.loadAudio(url);
            }
        };
    },
});
