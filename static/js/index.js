const videoImg = document.getElementById('videoFeed');

function reloadStream() {
    videoImg.src = "/video_feed?t=" + new Date().getTime();
}

async function uploadVideo(input) {
    const file = input.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    try {
        const response = await fetch('/upload', { method: 'POST', body: formData });
        if (response.ok) reloadStream();
    } catch (error) { console.error('Upload failed', error); }
}

async function resetSource() {
    try {
        const response = await fetch('/reset_source');
        if (response.ok) reloadStream();
    } catch (error) { console.error('Reset failed', error); }
}

setInterval(async () => {
    try {
        const response = await fetch('/stats');
        const data = await response.json();

        const riskIndicator = document.getElementById('risk-indicator');
        const alertPanel = document.getElementById('alertPanel');
        const enhanceEl = document.getElementById('enhanceStatus');
        riskIndicator.innerText = data.risk_level;
        alertPanel.className = 'alert-panel';

        if (data.risk_level === "CRITICAL") {
            riskIndicator.style.color = 'var(--critical-color)';
            riskIndicator.style.textShadow = '0 0 30px rgba(255, 0, 60, 0.6)';
            alertPanel.classList.add('panel-critical');
        }
        else if (data.risk_level === "CAUTION") {
            riskIndicator.style.color = 'var(--caution-color)';
            riskIndicator.style.textShadow = '0 0 30px rgba(255, 196, 0, 0.6)';
            alertPanel.classList.add('panel-caution');
        }
        else {
            riskIndicator.style.color = 'var(--safe-color)';
            riskIndicator.style.textShadow = '0 0 20px rgba(0, 255, 65, 0.3)';
        }

        if (data.was_enhanced) {
            enhanceEl.innerText = "ACTIVE";
            enhanceEl.style.color = "var(--hud-color)";
            enhanceEl.style.textShadow = "0 0 10px rgba(0, 242, 255, 0.5)";
        } else {
            enhanceEl.innerText = "STANDBY";
            enhanceEl.style.color = "#444";
            enhanceEl.style.textShadow = "none";
        }

    } catch (e) {

    }
}, 500);