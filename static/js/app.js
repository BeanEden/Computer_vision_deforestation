document.addEventListener('DOMContentLoaded', () => {
    let currentLot = 'lot1';
    let zoneData = [];

    const zoneSelector = document.getElementById('zone-selector');
    const lotButtons = document.querySelectorAll('.lot-btn');

    // Lot Selection
    lotButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            lotButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentLot = btn.dataset.lot;
            loadLotData(currentLot);
        });
    });

    async function loadLotData(lot) {
        try {
            const response = await fetch(`/api/data/${lot}`);
            zoneData = await response.json();
            
            // Populate Selector
            zoneSelector.innerHTML = '';
            zoneData.forEach((zone, index) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = `Zone : ${zone.image}`;
                zoneSelector.appendChild(option);
            });

            if (zoneData.length > 0) {
                updateUI(0);
            }
        } catch (error) {
            console.error("Error loading zone data:", error);
        }
    }

    zoneSelector.addEventListener('change', (e) => {
        updateUI(e.target.value);
    });

    function updateUI(index) {
        const data = zoneData[index];
        const name = data.image;

        // Paths based on current lot
        const basePath = `/outputs/${currentLot}`;

        // Observations
        document.getElementById('img-t0').src = `${basePath}/orig0_${name}`;
        document.getElementById('img-t1').src = `${basePath}/orig1_${name}`;

        // Segmentation
        document.getElementById('mask-t0').src = `${basePath}/mask0_${name}`;
        document.getElementById('mask-t1').src = `${basePath}/mask1_${name}`;

        // Final
        document.getElementById('change-map').src = `${basePath}/expert_change_${name}`;

        // Stats
        document.getElementById('stat-t0').textContent = `${data.surface_t0.toLocaleString()} px`;
        document.getElementById('stat-t1').textContent = `${data.surface_t1.toLocaleString()} px`;
        
        const lossEl = document.getElementById('stat-loss');
        const lossVal = data.loss_percentage;
        lossEl.textContent = `${lossVal.toFixed(2)}%`;
        lossEl.style.color = lossVal > 0 ? '#ff5252' : '#81c784';

        // Verdict Badge
        const verdictEl = document.getElementById('verdict-badge');
        verdictEl.textContent = data.Verdict;
        
        // Remove old classes
        verdictEl.className = 'verdict-badge';
        
        const v = data.Verdict.toLowerCase();
        if (v.includes('rupture')) verdictEl.classList.add('bg-critique');
        else if (v.includes('anthropisation')) verdictEl.classList.add('bg-deforestation');
        else if (v.includes('résilience')) verdictEl.classList.add('bg-stable');
        else if (v.includes('succession')) verdictEl.classList.add('bg-revegetalisation');
    }

    // Initial load
    loadLotData('lot1');
});
