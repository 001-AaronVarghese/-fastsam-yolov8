document.addEventListener('DOMContentLoaded', () => {
    const map = L.map('hazard-map').setView([40.768, -73.981], 14);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '© OpenStreetMap'
    }).addTo(map);

    if ('geolocation' in navigator) {
        navigator.geolocation.getCurrentPosition(pos => {
            const lat = pos.coords.latitude;
            const lon = pos.coords.longitude;
            map.setView([lat, lon], 15);
            L.circleMarker([lat, lon], { radius: 10, fillColor: '#00f2ff', color: '#fff', weight: 2, fillOpacity: 0.8 }).addTo(map).bindPopup('You are here');
        });
    }
});