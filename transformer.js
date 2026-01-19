/**
 * Transformer Architecture Explorer
 * Interactive educational visualization with accessibility features
 */

// ============================================
// CONFIGURATION & CONSTANTS
// ============================================

const CONFIG = {
    // Model configuration (simulated GPT-2 small)
    model: {
        vocabSize: 50257,
        embeddingDim: 768,
        numHeads: 12,
        numLayers: 12,
        mlpDim: 3072,
        maxSeqLen: 1024
    },
    // Visualization settings
    viz: {
        matrixCellSize: 8,
        attentionCellSize: 48,
        maxTokensToShow: 10,
        animationDuration: 300
    },
    // Sampling defaults
    sampling: {
        temperature: 1.0,
        topK: 50,
        topP: 0.9
    }
};

// Simple vocabulary simulation (BPE-style tokens)
const VOCABULARY = {
    tokens: {
        'The': 464, 'the': 262, ' cat': 3797, ' sat': 3332, ' on': 319,
        ' a': 257, ' mat': 2603, ' dog': 3290, ' ran': 4966, ' quick': 2068,
        ' brown': 7586, ' fox': 21831, ' jumped': 11687, ' over': 625,
        ' lazy': 16931, ' In': 554, ' in': 287, ' galaxy': 22920,
        ' far': 1290, ' away': 1497, ' Machine': 10850, ' machine': 4572,
        ' learning': 4673, ' is': 318, ' Once': 7454, ' upon': 2402,
        ' time': 640, ' there': 612, ' was': 373, ' lived': 5615,
        '.': 13, ',': 11, '!': 0, '?': 30, ' ': 220,
        'a': 64, 'b': 65, 'c': 66, 'd': 67, 'e': 68, 'f': 69,
        'g': 70, 'h': 71, 'i': 72, 'j': 73, 'k': 74, 'l': 75,
        'm': 76, 'n': 77, 'o': 78, 'p': 79, 'q': 80, 'r': 81,
        's': 82, 't': 83, 'u': 84, 'v': 85, 'w': 86, 'x': 87,
        'y': 88, 'z': 89
    },
    // Common continuation tokens for simulation
    continuations: {
        ' the': [' mat', ' floor', ' table', ' ground', ' roof', ' bed', ' chair'],
        ' on': [' the', ' a', ' top', ' his', ' her', ' my', ' your'],
        ' cat': [' sat', ' slept', ' jumped', ' ran', ' walked', ' meowed'],
        ' dog': [' ran', ' barked', ' jumped', ' slept', ' played'],
        ' fox': [' jumped', ' ran', ' hid', ' slept'],
        ' far': [' away', ',', ' far'],
        ' is': [' a', ' the', ' an', ' very', ' quite', ' not'],
        ' time': [' there', ',', ' in', ' when', ' a'],
        'default': [' the', ' a', ' is', ' was', ' and', ' to', ' of', ' in', ' that', ' for']
    }
};

// ============================================
// STATE MANAGEMENT
// ============================================

const state = {
    inputText: 'The cat sat on the',
    tokens: [],
    tokenIds: [],
    embeddings: null,
    attentionWeights: null,
    mlpOutput: null,
    outputProbabilities: null,
    generatedTokens: [],

    // UI state
    currentSection: 'intro',
    tourStep: 0,
    isTourActive: false,

    // Sampling parameters
    temperature: CONFIG.sampling.temperature,
    topK: CONFIG.sampling.topK,
    topP: CONFIG.sampling.topP,

    // Accessibility
    reducedMotion: false,
    fontSize: 'normal'
};

// ============================================
// UTILITY FUNCTIONS
// ============================================

/**
 * Announce message to screen readers
 */
function announce(message) {
    const announcer = document.getElementById('sr-announcer');
    if (announcer) {
        announcer.textContent = message;
        // Clear after announcement
        setTimeout(() => { announcer.textContent = ''; }, 1000);
    }
}

/**
 * Generate random float in range
 */
function randomFloat(min, max) {
    return Math.random() * (max - min) + min;
}

/**
 * Generate random integer in range
 */
function randomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

/**
 * Softmax function
 */
function softmax(arr, temperature = 1.0) {
    const maxVal = Math.max(...arr);
    const exps = arr.map(x => Math.exp((x - maxVal) / temperature));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
}

/**
 * Sample from probability distribution
 */
function sampleFromDistribution(probs, topK = null, topP = null) {
    let indices = probs.map((p, i) => ({ prob: p, index: i }));
    indices.sort((a, b) => b.prob - a.prob);

    // Apply top-k filtering
    if (topK !== null && topK < indices.length) {
        indices = indices.slice(0, topK);
    }

    // Apply top-p (nucleus) filtering
    if (topP !== null && topP < 1.0) {
        let cumSum = 0;
        let cutoffIdx = 0;
        for (let i = 0; i < indices.length; i++) {
            cumSum += indices[i].prob;
            if (cumSum >= topP) {
                cutoffIdx = i + 1;
                break;
            }
        }
        indices = indices.slice(0, cutoffIdx);
    }

    // Renormalize
    const sum = indices.reduce((acc, item) => acc + item.prob, 0);
    indices = indices.map(item => ({ ...item, prob: item.prob / sum }));

    // Sample
    const rand = Math.random();
    let cumProb = 0;
    for (const item of indices) {
        cumProb += item.prob;
        if (rand < cumProb) {
            return item.index;
        }
    }
    return indices[indices.length - 1].index;
}

/**
 * Interpolate between two colors
 */
function interpolateColor(color1, color2, factor) {
    const c1 = hexToRgb(color1);
    const c2 = hexToRgb(color2);

    const r = Math.round(c1.r + (c2.r - c1.r) * factor);
    const g = Math.round(c1.g + (c2.g - c1.g) * factor);
    const b = Math.round(c1.b + (c2.b - c1.b) * factor);

    return `rgb(${r}, ${g}, ${b})`;
}

function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : { r: 0, g: 0, b: 0 };
}

// ============================================
// TOKENIZATION
// ============================================

/**
 * Simple BPE-style tokenization
 */
function tokenize(text) {
    const tokens = [];
    const tokenIds = [];
    let remaining = text;

    while (remaining.length > 0) {
        let matched = false;

        // Try to match longest token first
        const sortedTokens = Object.keys(VOCABULARY.tokens)
            .filter(t => remaining.startsWith(t))
            .sort((a, b) => b.length - a.length);

        if (sortedTokens.length > 0) {
            const token = sortedTokens[0];
            tokens.push(token);
            tokenIds.push(VOCABULARY.tokens[token]);
            remaining = remaining.slice(token.length);
            matched = true;
        }

        if (!matched) {
            // Fall back to character-level
            const char = remaining[0];
            tokens.push(char);
            tokenIds.push(char.charCodeAt(0) % CONFIG.model.vocabSize);
            remaining = remaining.slice(1);
        }
    }

    return { tokens, tokenIds };
}

/**
 * Determine token type for styling
 */
function getTokenType(token) {
    if (token.startsWith(' ') && token.length > 1) {
        return 'word-token';
    } else if (token.length === 1) {
        if (/[a-zA-Z]/.test(token)) {
            return 'subword-token';
        }
        return 'special-token';
    }
    return 'word-token';
}

/**
 * Render tokenization visualization
 */
function renderTokenization() {
    const display = document.getElementById('token-display');
    const idsDisplay = document.getElementById('token-ids-display');

    if (!display || !idsDisplay) return;

    display.innerHTML = '';
    idsDisplay.innerHTML = '';

    state.tokens.forEach((token, index) => {
        const tokenType = getTokenType(token);

        // Create token element
        const tokenEl = document.createElement('span');
        tokenEl.className = `token ${tokenType}`;
        tokenEl.textContent = token.replace(/ /g, '\u00B7'); // Show spaces as middle dot
        tokenEl.setAttribute('role', 'listitem');
        tokenEl.setAttribute('tabindex', '0');
        tokenEl.setAttribute('aria-label', `Token ${index + 1}: "${token}", ID: ${state.tokenIds[index]}`);

        tokenEl.addEventListener('focus', () => {
            announce(`Token ${index + 1}: ${token}, ID: ${state.tokenIds[index]}`);
        });

        display.appendChild(tokenEl);

        // Create token ID element
        const idEl = document.createElement('span');
        idEl.className = 'token-id';
        idEl.textContent = state.tokenIds[index];
        idsDisplay.appendChild(idEl);
    });

    announce(`Tokenized into ${state.tokens.length} tokens`);
}

// ============================================
// EMBEDDINGS
// ============================================

/**
 * Generate simulated embeddings
 */
function generateEmbeddings() {
    const numTokens = state.tokens.length;
    const dim = CONFIG.model.embeddingDim;

    // Token embeddings (simulated)
    const tokenEmbeddings = [];
    for (let i = 0; i < numTokens; i++) {
        const embedding = [];
        for (let j = 0; j < dim; j++) {
            embedding.push(randomFloat(-1, 1));
        }
        tokenEmbeddings.push(embedding);
    }

    // Positional encodings (sinusoidal)
    const posEncodings = [];
    for (let pos = 0; pos < numTokens; pos++) {
        const encoding = [];
        for (let i = 0; i < dim; i++) {
            const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / dim);
            if (i % 2 === 0) {
                encoding.push(Math.sin(angle));
            } else {
                encoding.push(Math.cos(angle));
            }
        }
        posEncodings.push(encoding);
    }

    // Combined embeddings
    const finalEmbeddings = tokenEmbeddings.map((emb, i) =>
        emb.map((val, j) => val + posEncodings[i][j])
    );

    state.embeddings = {
        token: tokenEmbeddings,
        position: posEncodings,
        final: finalEmbeddings
    };
}

/**
 * Render embedding visualization
 */
function renderEmbeddings() {
    if (!state.embeddings) return;

    const matrices = ['token-matrix', 'position-matrix', 'final-matrix'];
    const data = [state.embeddings.token, state.embeddings.position, state.embeddings.final];

    matrices.forEach((matrixId, idx) => {
        const container = document.getElementById(matrixId);
        if (!container) return;

        container.innerHTML = '';

        // Show reduced version (first 32 dimensions of first few tokens)
        const numTokensToShow = Math.min(state.tokens.length, 6);
        const dimsToShow = 32;

        for (let t = 0; t < numTokensToShow; t++) {
            const row = document.createElement('div');
            row.className = 'matrix-row';

            for (let d = 0; d < dimsToShow; d++) {
                const cell = document.createElement('div');
                cell.className = 'matrix-cell';

                const value = data[idx][t][d];
                const normalizedValue = (value + 2) / 4; // Normalize to 0-1 range
                const intensity = Math.max(0, Math.min(1, normalizedValue));

                // Color based on value
                const hue = idx === 0 ? 220 : idx === 1 ? 280 : 200; // Blue, purple, cyan
                cell.style.backgroundColor = `hsl(${hue}, 70%, ${30 + intensity * 40}%)`;

                cell.title = `Token ${t + 1}, Dim ${d + 1}: ${value.toFixed(3)}`;
                row.appendChild(cell);
            }

            container.appendChild(row);
        }
    });
}

// ============================================
// ATTENTION
// ============================================

/**
 * Generate simulated attention weights
 */
function generateAttention() {
    const numTokens = state.tokens.length;
    const numHeads = CONFIG.model.numHeads;
    const numLayers = CONFIG.model.numLayers;

    state.attentionWeights = [];

    for (let layer = 0; layer < Math.min(numLayers, 4); layer++) {
        const layerWeights = [];

        for (let head = 0; head < Math.min(numHeads, 4); head++) {
            const headWeights = [];

            for (let i = 0; i < numTokens; i++) {
                const row = [];
                for (let j = 0; j < numTokens; j++) {
                    // Causal mask: can only attend to previous tokens
                    if (j > i) {
                        row.push(0);
                    } else {
                        // Higher attention to recent tokens and self
                        const recency = 1 - (i - j) / (i + 1);
                        const selfBoost = i === j ? 0.3 : 0;
                        row.push(recency * 0.7 + selfBoost + randomFloat(0, 0.2));
                    }
                }
                // Normalize row to sum to 1
                const sum = row.reduce((a, b) => a + b, 0);
                headWeights.push(row.map(v => v / sum));
            }

            layerWeights.push(headWeights);
        }

        state.attentionWeights.push(layerWeights);
    }
}

/**
 * Render attention visualization
 */
function renderAttention() {
    if (!state.attentionWeights) return;

    const layerSelect = document.getElementById('layer-select');
    const headSelect = document.getElementById('head-select');
    const matrix = document.getElementById('attention-matrix');
    const labelsTop = document.getElementById('attention-labels-top');
    const labelsLeft = document.getElementById('attention-labels-left');
    const infoPanel = document.getElementById('attention-info');

    if (!matrix || !layerSelect || !headSelect) return;

    const layer = parseInt(layerSelect.value);
    const headValue = headSelect.value;

    // Get attention weights
    let weights;
    if (headValue === 'avg') {
        // Average across heads
        weights = state.attentionWeights[layer][0].map((row, i) =>
            row.map((_, j) => {
                const sum = state.attentionWeights[layer].reduce((acc, head) => acc + head[i][j], 0);
                return sum / state.attentionWeights[layer].length;
            })
        );
    } else {
        weights = state.attentionWeights[layer][parseInt(headValue)];
    }

    // Clear and render
    matrix.innerHTML = '';
    labelsTop.innerHTML = '';
    labelsLeft.innerHTML = '';

    const numTokens = Math.min(state.tokens.length, CONFIG.viz.maxTokensToShow);
    matrix.style.gridTemplateColumns = `repeat(${numTokens}, ${CONFIG.viz.attentionCellSize}px)`;

    // Top labels
    for (let j = 0; j < numTokens; j++) {
        const label = document.createElement('div');
        label.className = 'attention-label-top';
        label.textContent = state.tokens[j].trim() || '\u00B7';
        labelsTop.appendChild(label);
    }

    // Left labels and matrix
    for (let i = 0; i < numTokens; i++) {
        const label = document.createElement('div');
        label.className = 'attention-label-left';
        label.textContent = state.tokens[i].trim() || '\u00B7';
        labelsLeft.appendChild(label);

        for (let j = 0; j < numTokens; j++) {
            const cell = document.createElement('div');
            cell.className = 'attention-cell';
            cell.setAttribute('role', 'gridcell');
            cell.setAttribute('tabindex', '0');

            const weight = weights[i][j];
            const color = interpolateColor('#f0f9ff', '#1e40af', weight);
            cell.style.backgroundColor = color;
            cell.style.color = weight > 0.5 ? 'white' : 'black';
            cell.textContent = weight.toFixed(2);

            const sourceToken = state.tokens[i];
            const targetToken = state.tokens[j];

            cell.setAttribute('aria-label',
                `Attention from "${sourceToken}" to "${targetToken}": ${(weight * 100).toFixed(1)}%`);

            cell.addEventListener('mouseenter', () => {
                infoPanel.innerHTML = `
                    <p><strong>Source:</strong> "${sourceToken}"</p>
                    <p><strong>Target:</strong> "${targetToken}"</p>
                    <p><strong>Attention:</strong> ${(weight * 100).toFixed(1)}%</p>
                `;
            });

            cell.addEventListener('focus', () => {
                announce(`Attention from ${sourceToken} to ${targetToken}: ${(weight * 100).toFixed(1)} percent`);
            });

            matrix.appendChild(cell);
        }
    }
}

// ============================================
// MLP VISUALIZATION
// ============================================

/**
 * Render MLP visualization
 */
function renderMLP() {
    const inputNeurons = document.getElementById('mlp-input-neurons');
    const hiddenNeurons = document.getElementById('mlp-hidden-neurons');
    const outputNeurons = document.getElementById('mlp-output-neurons');

    if (!inputNeurons || !hiddenNeurons || !outputNeurons) return;

    // Clear
    inputNeurons.innerHTML = '';
    hiddenNeurons.innerHTML = '';
    outputNeurons.innerHTML = '';

    // Input layer (show 16 neurons representing 768 dims)
    for (let i = 0; i < 16; i++) {
        const neuron = document.createElement('div');
        neuron.className = 'neuron';
        neuron.style.opacity = 0.3 + Math.random() * 0.7;
        neuron.title = `Input neuron ${i + 1} (representing dims ${i * 48 + 1}-${(i + 1) * 48})`;
        inputNeurons.appendChild(neuron);
    }

    // Hidden layer (show 24 neurons representing 3072 dims)
    for (let i = 0; i < 24; i++) {
        const neuron = document.createElement('div');
        neuron.className = 'neuron';
        neuron.style.opacity = 0.3 + Math.random() * 0.7;
        neuron.title = `Hidden neuron ${i + 1} (representing dims ${i * 128 + 1}-${(i + 1) * 128})`;
        hiddenNeurons.appendChild(neuron);
    }

    // Output layer
    for (let i = 0; i < 16; i++) {
        const neuron = document.createElement('div');
        neuron.className = 'neuron';
        neuron.style.opacity = 0.3 + Math.random() * 0.7;
        neuron.title = `Output neuron ${i + 1} (representing dims ${i * 48 + 1}-${(i + 1) * 48})`;
        outputNeurons.appendChild(neuron);
    }

    // Draw GELU chart
    drawGELU();
}

/**
 * Draw GELU activation function
 */
function drawGELU() {
    const canvas = document.getElementById('gelu-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear
    ctx.clearRect(0, 0, width, height);

    // Get computed styles for theming
    const computedStyle = getComputedStyle(document.documentElement);
    const textColor = computedStyle.getPropertyValue('--color-text-muted').trim() || '#64748b';
    const primaryColor = computedStyle.getPropertyValue('--color-primary').trim() || '#2563eb';
    const borderColor = computedStyle.getPropertyValue('--color-border').trim() || '#e2e8f0';

    // Draw axes
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 1;

    // X axis
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // Y axis
    ctx.beginPath();
    ctx.moveTo(width / 2, 0);
    ctx.lineTo(width / 2, height);
    ctx.stroke();

    // Draw GELU function
    ctx.strokeStyle = primaryColor;
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (let px = 0; px < width; px++) {
        const x = (px - width / 2) / (width / 8); // Map to -4 to 4
        const gelu = x * 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
        const py = height / 2 - gelu * (height / 8);

        if (px === 0) {
            ctx.moveTo(px, py);
        } else {
            ctx.lineTo(px, py);
        }
    }
    ctx.stroke();

    // Labels
    ctx.fillStyle = textColor;
    ctx.font = '10px system-ui';
    ctx.fillText('x', width - 15, height / 2 - 5);
    ctx.fillText('y', width / 2 + 5, 12);
    ctx.fillText('0', width / 2 + 3, height / 2 + 12);
}

// ============================================
// OUTPUT & GENERATION
// ============================================

/**
 * Generate output probabilities
 */
function generateOutputProbabilities() {
    // Get last token for context
    const lastToken = state.tokens[state.tokens.length - 1] || 'default';

    // Get possible continuations
    let continuations = VOCABULARY.continuations[lastToken] || VOCABULARY.continuations['default'];

    // Generate pseudo-random probabilities
    const logits = continuations.map(() => randomFloat(-2, 2));
    const probs = softmax(logits, state.temperature);

    state.outputProbabilities = continuations.map((token, i) => ({
        token,
        probability: probs[i],
        logit: logits[i]
    })).sort((a, b) => b.probability - a.probability);
}

/**
 * Render output visualization
 */
function renderOutput() {
    generateOutputProbabilities();

    const probChart = document.getElementById('prob-chart');
    const predictedToken = document.getElementById('predicted-token');
    const predictedProb = document.getElementById('predicted-prob');
    const originalText = document.getElementById('original-text');

    if (!probChart) return;

    probChart.innerHTML = '';

    // Show top probabilities
    const topProbs = state.outputProbabilities.slice(0, 8);
    const maxProb = topProbs[0].probability;

    topProbs.forEach((item, index) => {
        const container = document.createElement('div');
        container.className = 'prob-bar-container';

        const tokenLabel = document.createElement('span');
        tokenLabel.className = 'prob-token';
        tokenLabel.textContent = item.token.replace(/ /g, '\u00B7');

        const barWrapper = document.createElement('div');
        barWrapper.className = 'prob-bar-wrapper';

        const bar = document.createElement('div');
        bar.className = 'prob-bar';
        bar.style.width = `${(item.probability / maxProb) * 100}%`;

        // Highlight if in top-k or top-p
        if (index < state.topK) {
            bar.style.opacity = '1';
        } else {
            bar.style.opacity = '0.3';
        }

        barWrapper.appendChild(bar);

        const probLabel = document.createElement('span');
        probLabel.className = 'prob-value';
        probLabel.textContent = `${(item.probability * 100).toFixed(1)}%`;

        container.appendChild(tokenLabel);
        container.appendChild(barWrapper);
        container.appendChild(probLabel);

        probChart.appendChild(container);
    });

    // Show predicted token
    if (predictedToken && predictedProb) {
        const top = state.outputProbabilities[0];
        predictedToken.textContent = top.token;
        predictedProb.textContent = `(${(top.probability * 100).toFixed(1)}%)`;
    }

    // Update original text display
    if (originalText) {
        originalText.textContent = state.inputText;
    }
}

/**
 * Generate next token
 */
function generateNextToken() {
    generateOutputProbabilities();

    // Apply sampling
    const probs = state.outputProbabilities.map(p => p.probability);
    const sampledIdx = sampleFromDistribution(probs, state.topK, state.topP);
    const sampledToken = state.outputProbabilities[sampledIdx].token;

    state.generatedTokens.push(sampledToken);

    // Update display
    const continuation = document.getElementById('continuation');
    if (continuation) {
        continuation.textContent = state.generatedTokens.join('');
    }

    // Re-tokenize with new token for next iteration
    const newText = state.inputText + state.generatedTokens.join('');
    const { tokens, tokenIds } = tokenize(newText);
    state.tokens = tokens;
    state.tokenIds = tokenIds;

    // Regenerate visualizations
    generateEmbeddings();
    generateAttention();
    renderOutput();

    announce(`Generated token: ${sampledToken}`);
}

/**
 * Reset generation
 */
function resetGeneration() {
    state.generatedTokens = [];

    const continuation = document.getElementById('continuation');
    if (continuation) {
        continuation.textContent = '';
    }

    // Re-tokenize original
    const { tokens, tokenIds } = tokenize(state.inputText);
    state.tokens = tokens;
    state.tokenIds = tokenIds;

    generateEmbeddings();
    generateAttention();
    renderOutput();

    announce('Generation reset');
}

// ============================================
// SAMPLING CONTROLS
// ============================================

function setupSamplingControls() {
    const tempSlider = document.getElementById('temperature-slider');
    const tempValue = document.getElementById('temp-value');
    const topkSlider = document.getElementById('topk-slider');
    const topkValue = document.getElementById('topk-value');
    const toppSlider = document.getElementById('topp-slider');
    const toppValue = document.getElementById('topp-value');

    if (tempSlider) {
        tempSlider.addEventListener('input', (e) => {
            state.temperature = parseFloat(e.target.value);
            if (tempValue) tempValue.textContent = state.temperature.toFixed(1);
            renderOutput();
        });
    }

    if (topkSlider) {
        topkSlider.addEventListener('input', (e) => {
            state.topK = parseInt(e.target.value);
            if (topkValue) topkValue.textContent = state.topK;
            renderOutput();
        });
    }

    if (toppSlider) {
        toppSlider.addEventListener('input', (e) => {
            state.topP = parseFloat(e.target.value);
            if (toppValue) toppValue.textContent = state.topP.toFixed(2);
            renderOutput();
        });
    }
}

// ============================================
// ACCESSIBILITY CONTROLS
// ============================================

function setupAccessibilityControls() {
    // Theme toggle
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', newTheme);
            themeToggle.setAttribute('aria-pressed', newTheme === 'dark');
            localStorage.setItem('theme', newTheme);
            announce(`${newTheme} mode enabled`);

            // Redraw GELU chart for theme
            setTimeout(drawGELU, 100);
        });
    }

    // High contrast toggle
    const contrastToggle = document.getElementById('contrast-toggle');
    if (contrastToggle) {
        contrastToggle.addEventListener('click', () => {
            const html = document.documentElement;
            const isHigh = html.getAttribute('data-contrast') === 'high';
            html.setAttribute('data-contrast', isHigh ? 'normal' : 'high');
            contrastToggle.setAttribute('aria-pressed', !isHigh);
            localStorage.setItem('contrast', isHigh ? 'normal' : 'high');
            announce(`High contrast ${isHigh ? 'disabled' : 'enabled'}`);
        });
    }

    // Reduce motion toggle
    const motionToggle = document.getElementById('reduce-motion');
    if (motionToggle) {
        motionToggle.addEventListener('click', () => {
            const html = document.documentElement;
            const isReduced = html.getAttribute('data-reduce-motion') === 'true';
            html.setAttribute('data-reduce-motion', !isReduced);
            motionToggle.setAttribute('aria-pressed', !isReduced);
            state.reducedMotion = !isReduced;
            localStorage.setItem('reduceMotion', !isReduced);
            announce(`Animations ${isReduced ? 'enabled' : 'reduced'}`);
        });
    }

    // Font size controls
    const fontDecrease = document.getElementById('font-size-decrease');
    const fontIncrease = document.getElementById('font-size-increase');

    const fontSizes = ['small', 'normal', 'large', 'xlarge'];
    let currentFontIndex = 1;

    if (fontDecrease) {
        fontDecrease.addEventListener('click', () => {
            if (currentFontIndex > 0) {
                currentFontIndex--;
                updateFontSize();
            }
        });
    }

    if (fontIncrease) {
        fontIncrease.addEventListener('click', () => {
            if (currentFontIndex < fontSizes.length - 1) {
                currentFontIndex++;
                updateFontSize();
            }
        });
    }

    function updateFontSize() {
        const size = fontSizes[currentFontIndex];
        document.documentElement.setAttribute('data-font-size', size === 'normal' ? '' : size);
        localStorage.setItem('fontSize', size);
        announce(`Font size: ${size}`);
    }

    // Load saved preferences
    loadSavedPreferences();
}

function loadSavedPreferences() {
    const theme = localStorage.getItem('theme');
    const contrast = localStorage.getItem('contrast');
    const reduceMotion = localStorage.getItem('reduceMotion');
    const fontSize = localStorage.getItem('fontSize');

    if (theme) {
        document.documentElement.setAttribute('data-theme', theme);
        const btn = document.getElementById('theme-toggle');
        if (btn) btn.setAttribute('aria-pressed', theme === 'dark');
    }

    if (contrast) {
        document.documentElement.setAttribute('data-contrast', contrast);
        const btn = document.getElementById('contrast-toggle');
        if (btn) btn.setAttribute('aria-pressed', contrast === 'high');
    }

    if (reduceMotion === 'true') {
        document.documentElement.setAttribute('data-reduce-motion', 'true');
        state.reducedMotion = true;
        const btn = document.getElementById('reduce-motion');
        if (btn) btn.setAttribute('aria-pressed', true);
    }

    if (fontSize && fontSize !== 'normal') {
        document.documentElement.setAttribute('data-font-size', fontSize);
    }

    // Check system preference for dark mode
    if (!theme && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        document.documentElement.setAttribute('data-theme', 'dark');
    }

    // Check system preference for reduced motion
    if (!reduceMotion && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        document.documentElement.setAttribute('data-reduce-motion', 'true');
        state.reducedMotion = true;
    }
}

// ============================================
// KEYBOARD NAVIGATION
// ============================================

function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Alt + key shortcuts
        if (e.altKey) {
            switch (e.key.toLowerCase()) {
                case 'd':
                    e.preventDefault();
                    document.getElementById('theme-toggle')?.click();
                    break;
                case 'c':
                    e.preventDefault();
                    document.getElementById('contrast-toggle')?.click();
                    break;
                case 'm':
                    e.preventDefault();
                    document.getElementById('reduce-motion')?.click();
                    break;
                case 't':
                    e.preventDefault();
                    startTour();
                    break;
                case '=':
                case '+':
                    e.preventDefault();
                    document.getElementById('font-size-increase')?.click();
                    break;
                case '-':
                    e.preventDefault();
                    document.getElementById('font-size-decrease')?.click();
                    break;
            }
        }

        // Ctrl + Enter to process
        if (e.ctrlKey && e.key === 'Enter') {
            const textInput = document.getElementById('text-input');
            if (document.activeElement === textInput) {
                e.preventDefault();
                processInput();
            }
        }

        // Escape to close tour
        if (e.key === 'Escape' && state.isTourActive) {
            endTour();
        }

        // Arrow keys for tour navigation
        if (state.isTourActive) {
            if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
                e.preventDefault();
                nextTourStep();
            } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
                e.preventDefault();
                prevTourStep();
            }
        }
    });
}

// ============================================
// GUIDED TOUR
// ============================================

const TOUR_STEPS = [
    {
        target: null,
        title: 'Welcome to the Transformer Explorer!',
        content: 'This interactive guide will help you understand how transformer models like GPT work. Use arrow keys or buttons to navigate. Press Escape to exit at any time.'
    },
    {
        target: '#input-section',
        title: 'Start with Input',
        content: 'Enter any text here. The transformer will process it and predict what comes next. Try the example prompts or write your own!'
    },
    {
        target: '#tokenization-section',
        title: 'Tokenization',
        content: 'Your text is split into smaller pieces called tokens. These can be whole words, parts of words, or individual characters. Hover over tokens to see their IDs.'
    },
    {
        target: '#embedding-section',
        title: 'Embeddings',
        content: 'Each token is converted into a vector of numbers (embedding) that captures its meaning. Position information is added so the model knows word order.'
    },
    {
        target: '#attention-section',
        title: 'Self-Attention',
        content: 'The heart of the transformer! Each token looks at all previous tokens to gather context. The heatmap shows attention weights - darker means stronger attention.'
    },
    {
        target: '#mlp-section',
        title: 'Feed-Forward Network',
        content: 'After attention, information passes through a neural network that processes and transforms it. The network expands then compresses the data.'
    },
    {
        target: '#output-section',
        title: 'Output Generation',
        content: 'Finally, the model predicts probabilities for the next token. Adjust temperature, top-k, and top-p to control how creative or focused the output is!'
    }
];

function startTour() {
    state.isTourActive = true;
    state.tourStep = 0;

    const overlay = document.getElementById('tour-overlay');
    if (overlay) {
        overlay.hidden = false;
        showTourStep();
    }

    announce('Guided tour started. Use arrow keys to navigate.');
}

function endTour() {
    state.isTourActive = false;

    const overlay = document.getElementById('tour-overlay');
    if (overlay) {
        overlay.hidden = true;
    }

    announce('Tour ended');
}

function showTourStep() {
    const step = TOUR_STEPS[state.tourStep];
    const spotlight = document.getElementById('tour-spotlight');
    const tooltip = document.getElementById('tour-tooltip');
    const title = document.getElementById('tour-title');
    const content = document.getElementById('tour-content');
    const indicator = document.getElementById('tour-step-indicator');
    const prevBtn = document.getElementById('tour-prev');
    const nextBtn = document.getElementById('tour-next');

    if (!tooltip || !title || !content) return;

    title.textContent = step.title;
    content.textContent = step.content;
    indicator.textContent = `${state.tourStep + 1}/${TOUR_STEPS.length}`;

    // Update button states
    if (prevBtn) {
        prevBtn.disabled = state.tourStep === 0;
        prevBtn.style.visibility = state.tourStep === 0 ? 'hidden' : 'visible';
    }

    if (nextBtn) {
        nextBtn.textContent = state.tourStep === TOUR_STEPS.length - 1 ? 'Finish' : 'Next';
    }

    // Position spotlight and tooltip
    if (step.target) {
        const targetEl = document.querySelector(step.target);
        if (targetEl) {
            const rect = targetEl.getBoundingClientRect();
            const padding = 10;

            spotlight.style.top = `${rect.top + window.scrollY - padding}px`;
            spotlight.style.left = `${rect.left - padding}px`;
            spotlight.style.width = `${rect.width + padding * 2}px`;
            spotlight.style.height = `${rect.height + padding * 2}px`;
            spotlight.style.display = 'block';

            // Position tooltip below or above target
            const tooltipRect = tooltip.getBoundingClientRect();
            if (rect.bottom + tooltipRect.height + 20 < window.innerHeight) {
                tooltip.style.top = `${rect.bottom + window.scrollY + 20}px`;
            } else {
                tooltip.style.top = `${rect.top + window.scrollY - tooltipRect.height - 20}px`;
            }
            tooltip.style.left = `${Math.max(20, rect.left)}px`;

            // Scroll into view
            targetEl.scrollIntoView({ behavior: state.reducedMotion ? 'auto' : 'smooth', block: 'center' });
        }
    } else {
        // Center tooltip for intro
        spotlight.style.display = 'none';
        tooltip.style.top = '50%';
        tooltip.style.left = '50%';
        tooltip.style.transform = 'translate(-50%, -50%)';
    }

    announce(`Step ${state.tourStep + 1}: ${step.title}`);
}

function nextTourStep() {
    if (state.tourStep < TOUR_STEPS.length - 1) {
        state.tourStep++;
        showTourStep();
    } else {
        endTour();
    }
}

function prevTourStep() {
    if (state.tourStep > 0) {
        state.tourStep--;
        showTourStep();
    }
}

function setupTour() {
    const tourBtn = document.getElementById('tour-btn');
    const closeBtn = document.getElementById('tour-close');
    const prevBtn = document.getElementById('tour-prev');
    const nextBtn = document.getElementById('tour-next');

    if (tourBtn) tourBtn.addEventListener('click', startTour);
    if (closeBtn) closeBtn.addEventListener('click', endTour);
    if (prevBtn) prevBtn.addEventListener('click', prevTourStep);
    if (nextBtn) nextBtn.addEventListener('click', nextTourStep);
}

// ============================================
// SECTION NAVIGATION
// ============================================

function setupNavigation() {
    // Architecture block clicks
    const archBlocks = document.querySelectorAll('.arch-block');
    archBlocks.forEach(block => {
        block.addEventListener('click', () => {
            const step = block.dataset.step;
            const section = document.getElementById(`${step}-section`);
            if (section) {
                section.scrollIntoView({ behavior: state.reducedMotion ? 'auto' : 'smooth' });
            }
        });

        block.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                block.click();
            }
        });
    });

    // Nav link active state
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.content-section');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const id = entry.target.id;
                navLinks.forEach(link => {
                    link.classList.toggle('active', link.getAttribute('href') === `#${id}`);
                });
            }
        });
    }, { threshold: 0.3 });

    sections.forEach(section => observer.observe(section));

    // Attention control listeners
    const layerSelect = document.getElementById('layer-select');
    const headSelect = document.getElementById('head-select');

    if (layerSelect) layerSelect.addEventListener('change', renderAttention);
    if (headSelect) headSelect.addEventListener('change', renderAttention);
}

// ============================================
// INPUT PROCESSING
// ============================================

function processInput() {
    const textInput = document.getElementById('text-input');
    if (!textInput) return;

    state.inputText = textInput.value.trim();
    state.generatedTokens = [];

    if (!state.inputText) {
        announce('Please enter some text');
        return;
    }

    // Update continuation display
    const continuation = document.getElementById('continuation');
    if (continuation) continuation.textContent = '';

    // Tokenize
    const { tokens, tokenIds } = tokenize(state.inputText);
    state.tokens = tokens;
    state.tokenIds = tokenIds;

    // Generate all visualizations
    generateEmbeddings();
    generateAttention();

    // Render all
    renderTokenization();
    renderEmbeddings();
    renderAttention();
    renderMLP();
    renderOutput();

    announce(`Processed ${tokens.length} tokens`);

    // Scroll to tokenization section
    const tokenSection = document.getElementById('tokenization-section');
    if (tokenSection) {
        tokenSection.scrollIntoView({ behavior: state.reducedMotion ? 'auto' : 'smooth' });
    }
}

function setupInputHandlers() {
    const processBtn = document.getElementById('process-btn');
    const textInput = document.getElementById('text-input');
    const generateBtn = document.getElementById('generate-btn');
    const generateMoreBtn = document.getElementById('generate-more-btn');
    const resetBtn = document.getElementById('reset-btn');

    if (processBtn) {
        processBtn.addEventListener('click', processInput);
    }

    if (textInput) {
        textInput.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                processInput();
            }
        });
    }

    // Example prompts
    const exampleBtns = document.querySelectorAll('.example-btn');
    exampleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const text = btn.dataset.text;
            if (textInput) {
                textInput.value = text;
                processInput();
            }
        });
    });

    // Generation buttons
    if (generateBtn) {
        generateBtn.addEventListener('click', generateNextToken);
    }

    if (generateMoreBtn) {
        generateMoreBtn.addEventListener('click', () => {
            for (let i = 0; i < 10; i++) {
                setTimeout(() => generateNextToken(), i * 100);
            }
        });
    }

    if (resetBtn) {
        resetBtn.addEventListener('click', resetGeneration);
    }
}

// ============================================
// INITIALIZATION
// ============================================

function init() {
    // Setup all handlers
    setupAccessibilityControls();
    setupKeyboardShortcuts();
    setupNavigation();
    setupTour();
    setupInputHandlers();
    setupSamplingControls();

    // Initial processing
    processInput();

    // Re-render on window resize (for responsive charts)
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            drawGELU();
        }, 250);
    });

    console.log('Transformer Explorer initialized');
}

// Start when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
