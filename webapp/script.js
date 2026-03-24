// Initialize Lucide icons
lucide.createIcons();

// Elements
const form = document.getElementById('premium-form');
const calcBtn = document.getElementById('calc-btn');
const spinner = document.getElementById('spinner');
const btnText = calcBtn.querySelector('span');
const btnIcon = calcBtn.querySelector('.lucide');
const errorMsg = document.getElementById('error-message');

const loadingInput = document.getElementById('loading');
const loadingVal = document.getElementById('loading-val');

const emptyState = document.getElementById('empty-state');
const resultsContent = document.getElementById('results-content');

// Dynamic slider update
loadingInput.addEventListener('input', (e) => {
    loadingVal.textContent = `${e.target.value}%`;
});

// Form Submission
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // UI Loading State
    errorMsg.classList.add('hidden');
    spinner.classList.remove('hidden');
    btnText.textContent = "Calculating...";
    btnIcon.classList.add('hidden');
    calcBtn.disabled = true;

    // Build Payload
    const payload = {
        birth_date: document.getElementById('birth_date').value,
        exp: parseInt(document.getElementById('exp').value),
        matriculation_year: parseInt(document.getElementById('matriculation_year').value),
        vehicle_value: parseFloat(document.getElementById('vehicle_value').value),
        cc: parseInt(document.getElementById('cc').value),
        loading: parseFloat(document.getElementById('loading').value)
    };

    try {
        // Assume backend runs on localhost:8000
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to fetch prediction');
        }

        const data = await response.json();
        
        displayResults(data, payload.loading);

    } catch (err) {
        errorMsg.textContent = 'Error: ' + err.message;
        errorMsg.classList.remove('hidden');
    } finally {
        spinner.classList.add('hidden');
        btnText.textContent = "Calculate Premium";
        btnIcon.classList.remove('hidden');
        calcBtn.disabled = false;
    }
});

// Display logic
function displayResults(data, loadingPct) {
    emptyState.classList.add('hidden');
    resultsContent.classList.remove('hidden');

    // Format currency
    const formatEur = (v) => new Intl.NumberFormat('en-IE', { style: 'currency', currency: 'EUR' }).format(v);

    document.getElementById('res-pure-premium').textContent = formatEur(data.premiums.pure_premium);
    document.getElementById('res-final-premium').textContent = formatEur(data.premiums.final_premium);
    document.getElementById('res-loading-pct').textContent = loadingPct;
    
    // Risk styling
    const riskBadge = document.getElementById('risk-badge');
    const riskTier = document.getElementById('res-risk-tier');
    const riskIconContainer = document.getElementById('risk-icon-container');
    
    riskBadge.className = 'risk-badge'; // reset
    if (data.risk.tier === 'low') {
        riskBadge.classList.add('tier-low');
        riskIconContainer.innerHTML = '<i data-lucide="check-circle-2"></i>';
    } else if (data.risk.tier === 'medium') {
        riskBadge.classList.add('tier-medium');
        riskIconContainer.innerHTML = '<i data-lucide="alert-triangle"></i>';
    } else {
        riskBadge.classList.add('tier-high');
        riskIconContainer.innerHTML = '<i data-lucide="alert-octagon"></i>';
    }
    lucide.createIcons(); // refresh icon inject
    
    riskTier.textContent = data.risk.label;
    document.getElementById('res-risk-ratio').textContent = data.risk.ratio.toFixed(2);
    
    // Breakdown
    document.getElementById('res-glm-pp').textContent = formatEur(data.breakdown.glm_pp);
    document.getElementById('res-nf-pp').textContent = formatEur(data.breakdown.nf_pp);
    document.getElementById('res-glm-freq').textContent = data.breakdown.glm_freq.toFixed(4) + ' f/yr';
    document.getElementById('res-nf-freq').textContent = data.breakdown.nf_freq.toFixed(4) + ' f/yr';
    document.getElementById('res-portfolio-avg').textContent = formatEur(data.breakdown.portfolio_avg);
    document.getElementById('res-loading-amt').textContent = formatEur(data.premiums.loading_amount);
}
