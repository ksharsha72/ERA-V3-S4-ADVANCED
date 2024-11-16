let modelData = {
    config1: {
        loss: { epoch: [], value: [], train: [] },
        accuracy: { epoch: [], value: [], test: [] },
        isTraining: false
    },
    config2: {
        loss: { epoch: [], value: [], train: [] },
        accuracy: { epoch: [], value: [], test: [] },
        isTraining: false
    }
};

let trainingHistory = [];

let stopTrainingFlags = {
    config1: false,
    config2: false
};

function initCharts() {
    Plotly.newPlot('lossChart', [
        {
            x: [],
            y: [],
            name: 'Config 1 Training Loss',
            type: 'scatter',
            line: { color: '#1f77b4' }
        },
        {
            x: [],
            y: [],
            name: 'Config 1 Test Loss',
            type: 'scatter',
            line: { color: '#1f77b4', dash: 'dot' }
        },
        {
            x: [],
            y: [],
            name: 'Config 2 Training Loss',
            type: 'scatter',
            line: { color: '#ff7f0e' }
        },
        {
            x: [],
            y: [],
            name: 'Config 2 Test Loss',
            type: 'scatter',
            line: { color: '#ff7f0e', dash: 'dot' }
        }
    ], {
        title: 'Loss Comparison',
        xaxis: { title: 'Epoch' },
        yaxis: { title: 'Loss' },
        showlegend: true,
        legend: { x: 1, xanchor: 'right', y: 1 }
    });

    Plotly.newPlot('accuracyChart', [
        {
            x: [],
            y: [],
            name: 'Config 1 Training Accuracy',
            type: 'scatter',
            line: { color: '#1f77b4' }
        },
        {
            x: [],
            y: [],
            name: 'Config 1 Test Accuracy',
            type: 'scatter',
            line: { color: '#1f77b4', dash: 'dot' }
        },
        {
            x: [],
            y: [],
            name: 'Config 2 Training Accuracy',
            type: 'scatter',
            line: { color: '#ff7f0e' }
        },
        {
            x: [],
            y: [],
            name: 'Config 2 Test Accuracy',
            type: 'scatter',
            line: { color: '#ff7f0e', dash: 'dot' }
        }
    ], {
        title: 'Accuracy Comparison',
        xaxis: { title: 'Epoch' },
        yaxis: { title: 'Accuracy (%)' },
        showlegend: true,
        legend: { x: 1, xanchor: 'right', y: 1 }
    });
}

function updateMetrics(modelId, data) {
    try {
        if ('loss' in data) {
            // Update training loss
            document.getElementById(`${modelId}_loss`).textContent =
                `${data.loss.toFixed(4)} (Epoch ${data.epoch}, Batch ${data.batch})`;
        }

        if ('accuracy' in data) {
            // Update all metrics for epoch completion
            document.getElementById(`${modelId}_test_loss`).textContent =
                `${data.test_loss.toFixed(4)} (Epoch ${data.epoch})`;
            document.getElementById(`${modelId}_train_acc`).textContent =
                `${data.train_accuracy.toFixed(2)}% (Epoch ${data.epoch})`;
            document.getElementById(`${modelId}_test_acc`).textContent =
                `${data.accuracy.toFixed(2)}% (Epoch ${data.epoch})`;
        }
    } catch (error) {
        console.error('Error updating metrics:', error);
    }
}

function updateCharts(modelId, data) {
    try {
        if ('avg_train_loss' in data) {
            // For epoch completion data
            modelData[modelId].loss.epoch.push(data.epoch);
            modelData[modelId].loss.value.push(data.test_loss);
            modelData[modelId].loss.train.push(data.avg_train_loss);
            modelData[modelId].accuracy.epoch.push(data.epoch);
            modelData[modelId].accuracy.value.push(data.train_accuracy);
            modelData[modelId].accuracy.test.push(data.accuracy);  // This is test accuracy

            // Update metrics display
            updateMetrics(modelId, {
                loss: data.avg_train_loss,
                test_loss: data.test_loss,
                train_accuracy: data.train_accuracy,
                accuracy: data.accuracy,
                epoch: data.epoch
            });

            // Update loss chart with all data points
            Plotly.update('lossChart', {
                x: [
                    modelData.config1.loss.epoch, modelData.config1.loss.epoch,
                    modelData.config2.loss.epoch, modelData.config2.loss.epoch
                ],
                y: [
                    modelData.config1.loss.train, modelData.config1.loss.value,
                    modelData.config2.loss.train, modelData.config2.loss.value
                ]
            }, {
                'xaxis.title': 'Epoch',
                'yaxis.title': 'Loss'
            });

            // Update accuracy chart with all data points
            Plotly.update('accuracyChart', {
                x: [
                    modelData.config1.accuracy.epoch, modelData.config1.accuracy.epoch,
                    modelData.config2.accuracy.epoch, modelData.config2.accuracy.epoch
                ],
                y: [
                    modelData.config1.accuracy.value, modelData.config1.accuracy.test,
                    modelData.config2.accuracy.value, modelData.config2.accuracy.test
                ]
            }, {
                'xaxis.title': 'Epoch',
                'yaxis.title': 'Accuracy (%)'
            });

        } else if ('loss' in data) {
            // Update just the current training loss
            updateMetrics(modelId, data);
        }
    } catch (error) {
        console.error('Error updating charts:', error);
    }
}

function addLog(configId, message) {
    const logContainer = document.getElementById(`logContainer${configId.slice(-1)}`);
    const timestamp = new Date().toLocaleTimeString();
    logContainer.innerHTML += `<div>[${timestamp}] ${message}</div>`;
    logContainer.scrollTop = logContainer.scrollHeight;
}

function resetMetrics(modelId) {
    document.getElementById(`${modelId}_loss`).textContent = '-';
    document.getElementById(`${modelId}_test_loss`).textContent = '-';
    document.getElementById(`${modelId}_train_acc`).textContent = '-';
    document.getElementById(`${modelId}_test_acc`).textContent = '-';
}

function calculateSummary(modelData) {
    const finalEpoch = modelData.loss.epoch.length - 1;
    return {
        finalTrainLoss: modelData.loss.train[finalEpoch]?.toFixed(4) || 'N/A',
        finalTestLoss: modelData.loss.value[finalEpoch]?.toFixed(4) || 'N/A',
        finalTrainAccuracy: modelData.accuracy.value[finalEpoch]?.toFixed(2) || 'N/A',
        finalTestAccuracy: modelData.accuracy.test[finalEpoch]?.toFixed(2) || 'N/A',
        bestTestAccuracy: Math.max(...modelData.accuracy.test).toFixed(2),
        lowestTrainLoss: Math.min(...modelData.loss.train).toFixed(4)
    };
}

async function saveTrainingState(configId, kernels, batchSize, optimizer, learningRate) {
    try {
        // Get the chart divs
        const lossChart = document.getElementById('lossChart');
        const accuracyChart = document.getElementById('accuracyChart');

        // Create images of the current charts
        const lossImage = await Plotly.toImage(lossChart, { format: 'png', height: 300, width: 400 });
        const accuracyImage = await Plotly.toImage(accuracyChart, { format: 'png', height: 300, width: 400 });

        // Calculate summary statistics
        const summary = calculateSummary(modelData[configId]);

        const historyItem = {
            timestamp: new Date().toISOString(),
            configId: configId,
            config: {
                kernels: kernels,
                batchSize: batchSize,
                optimizer: optimizer,
                learningRate: learningRate
            },
            summary: summary,
            data: modelData[configId],
            lossImage: lossImage,
            accuracyImage: accuracyImage
        };

        trainingHistory.push(historyItem);
        updateHistorySidebar();
    } catch (error) {
        console.error('Error saving training state:', error);
    }
}

function updateHistorySidebar() {
    const container = document.getElementById('history-container');
    container.innerHTML = trainingHistory.map((item, index) => `
        <div class="history-item" onclick="showHistoricalGraphs(${index})">
            <div class="history-config">
                <strong>${new Date(item.timestamp).toLocaleString()}</strong>
                <div class="config-details">
                    <div>Configuration ${item.configId.slice(-1)}</div>
                    <div>Optimizer: ${item.config.optimizer}</div>
                    <div>Batch Size: ${item.config.batchSize}</div>
                    <div>Learning Rate: ${item.config.learningRate}</div>
                    <div>Kernels: ${item.config.kernels.join(', ')}</div>
                </div>
                <div class="summary-details">
                    <div>Final Train Loss: ${item.summary.finalTrainLoss}</div>
                    <div>Final Test Loss: ${item.summary.finalTestLoss}</div>
                    <div>Final Train Acc: ${item.summary.finalTrainAccuracy}%</div>
                    <div>Final Test Acc: ${item.summary.finalTestAccuracy}%</div>
                    <div>Best Test Acc: ${item.summary.bestTestAccuracy}%</div>
                    <div>Best Train Loss: ${item.summary.lowestTrainLoss}</div>
                </div>
            </div>
        </div>
    `).join('');
}

function showHistoricalGraphs(index) {
    const item = trainingHistory[index];
    const modal = document.getElementById('history-modal');
    const modalGraphs = document.getElementById('modal-graphs');
    const modalConfig = document.getElementById('modal-config');

    // Add class to body to prevent scrolling
    document.body.classList.add('modal-open');

    modalGraphs.innerHTML = `
        <div class="historical-graphs">
            <div class="graph-container">
                <h4>Loss Curves</h4>
                <img src="${item.lossImage}" alt="Loss Graph">
            </div>
            <div class="graph-container">
                <h4>Accuracy Curves</h4>
                <img src="${item.accuracyImage}" alt="Accuracy Graph">
            </div>
        </div>
    `;

    modalConfig.innerHTML = `
        <div class="config-details">
            <h3>Configuration Details</h3>
            <table class="config-table">
                <tr><td><strong>Model:</strong></td><td>Configuration ${item.configId.slice(-1)}</td></tr>
                <tr><td><strong>Kernels:</strong></td><td>${item.config.kernels.join(', ')}</td></tr>
                <tr><td><strong>Batch Size:</strong></td><td>${item.config.batchSize}</td></tr>
                <tr><td><strong>Optimizer:</strong></td><td>${item.config.optimizer}</td></tr>
                <tr><td><strong>Learning Rate:</strong></td><td>${item.config.learningRate}</td></tr>
                <tr><td><strong>Final Train Loss:</strong></td><td>${item.summary.finalTrainLoss}</td></tr>
                <tr><td><strong>Final Test Loss:</strong></td><td>${item.summary.finalTestLoss}</td></tr>
                <tr><td><strong>Final Train Accuracy:</strong></td><td>${item.summary.finalTrainAccuracy}%</td></tr>
                <tr><td><strong>Final Test Accuracy:</strong></td><td>${item.summary.finalTestAccuracy}%</td></tr>
                <tr><td><strong>Best Test Accuracy:</strong></td><td>${item.summary.bestTestAccuracy}%</td></tr>
                <tr><td><strong>Best Train Loss:</strong></td><td>${item.summary.lowestTrainLoss}</td></tr>
                <tr><td><strong>Timestamp:</strong></td><td>${new Date(item.timestamp).toLocaleString()}</td></tr>
            </table>
        </div>
    `;

    modal.style.display = 'block';
}

function closeModal() {
    const modal = document.getElementById('history-modal');
    modal.style.display = 'none';
    // Remove class from body to restore scrolling
    document.body.classList.remove('modal-open');
}

function stopTraining(configId) {
    stopTrainingFlags[configId] = true;
    document.getElementById(`stop_${configId}`).disabled = true;
    addLog(configId, 'Stopping training after current epoch completes...');

    // Only update the training status without clearing the data
    modelData[configId].isTraining = false;

    // No need to clear the data or update charts
    addLog(configId, 'Training stopped. Graph preserved.');
}

function updateChartsAfterStop() {
    Plotly.update('lossChart', {
        x: [
            modelData.config1.loss.epoch, modelData.config1.loss.epoch,
            modelData.config2.loss.epoch, modelData.config2.loss.epoch
        ],
        y: [
            modelData.config1.loss.train, modelData.config1.loss.value,
            modelData.config2.loss.train, modelData.config2.loss.value
        ]
    });

    Plotly.update('accuracyChart', {
        x: [
            modelData.config1.accuracy.epoch, modelData.config1.accuracy.epoch,
            modelData.config2.accuracy.epoch, modelData.config2.accuracy.epoch
        ],
        y: [
            modelData.config1.accuracy.value, modelData.config1.accuracy.test,
            modelData.config2.accuracy.value, modelData.config2.accuracy.test
        ]
    });
}

async function startTraining(configId) {
    if (modelData[configId].isTraining) {
        alert('This configuration is already training!');
        return;
    }

    try {
        // Save current state if there's data
        if (modelData[configId].loss.epoch.length > 0) {
            const kernels = [
                parseInt(document.getElementById(`${configId}_kernel1`).value),
                parseInt(document.getElementById(`${configId}_kernel2`).value),
                parseInt(document.getElementById(`${configId}_kernel3`).value),
                parseInt(document.getElementById(`${configId}_kernel4`).value)
            ];
            const batchSize = document.getElementById(`${configId}_batch_size`).value;
            const optimizer = document.getElementById(`${configId}_optimizer`).value;
            const learningRate = document.getElementById(`${configId}_learning_rate`).value;

            await saveTrainingState(configId, kernels, batchSize, optimizer, learningRate);
        }

        // Reset stop flag
        stopTrainingFlags[configId] = false;

        // Get all configuration values
        const kernels = [
            parseInt(document.getElementById(`${configId}_kernel1`).value),
            parseInt(document.getElementById(`${configId}_kernel2`).value),
            parseInt(document.getElementById(`${configId}_kernel3`).value),
            parseInt(document.getElementById(`${configId}_kernel4`).value)
        ];
        const batchSize = document.getElementById(`${configId}_batch_size`).value;
        const optimizer = document.getElementById(`${configId}_optimizer`).value;
        const learningRate = document.getElementById(`${configId}_learning_rate`).value;
        const epochs = parseInt(document.getElementById(`${configId}_epochs`).value);

        // Clear only the specific model's logs
        document.getElementById(`logContainer${configId.slice(-1)}`).innerHTML = '';

        // Reset only this model's data
        modelData[configId] = {
            loss: { epoch: [], value: [], train: [] },
            accuracy: { epoch: [], value: [], test: [] },
            isTraining: true
        };

        // Update button states
        const trainButton = document.querySelector(`button[onclick="startTraining('${configId}')"]`);
        const stopButton = document.getElementById(`stop_${configId}`);
        trainButton.disabled = true;
        stopButton.disabled = false;

        // Reset metrics display for only this configuration
        resetMetrics(configId);

        addLog(configId, 'Starting training...');
        addLog(configId, `Epochs: ${epochs}`);
        addLog(configId, `Kernels: ${kernels.join(', ')}`);
        addLog(configId, `Learning Rate: ${learningRate}`);

        const response = await fetch('/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_id: configId,
                kernels: kernels,
                batch_size: batchSize,
                optimizer: optimizer,
                learning_rate: learningRate,
                epochs: epochs
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const lines = decoder.decode(value).split('\n');
            for (const line of lines) {
                if (!line.trim()) continue;

                try {
                    const data = JSON.parse(line);

                    // Check if training should stop
                    if (stopTrainingFlags[configId]) {
                        addLog(configId, 'Training stopped by user');
                        return;
                    }

                    updateCharts(configId, data);

                    if ('accuracy' in data) {
                        addLog(configId,
                            `Epoch ${data.epoch}: Accuracy = ${data.accuracy.toFixed(2)}%, ` +
                            `Train Loss = ${data.avg_train_loss.toFixed(4)}, ` +
                            `Test Loss = ${data.test_loss.toFixed(4)}`);
                    }
                } catch (error) {
                    console.error('Error processing line:', line, error);
                }
            }
        }
    } catch (error) {
        console.error('Training error:', error);
        addLog(configId, `Error: ${error.message}`);
    } finally {
        // Reset button states
        const trainButton = document.querySelector(`button[onclick="startTraining('${configId}')"]`);
        const stopButton = document.getElementById(`stop_${configId}`);
        trainButton.disabled = false;
        stopButton.disabled = true;
        modelData[configId].isTraining = false;
        stopTrainingFlags[configId] = false;
    }
}

function toggleHistory() {
    const sidebar = document.getElementById('history-sidebar');
    const contentArea = document.querySelector('.content-area');
    const toggle = document.getElementById('history-toggle');

    sidebar.classList.toggle('active');
    contentArea.classList.toggle('shifted');

    if (sidebar.classList.contains('active')) {
        toggle.style.right = '300px';
    } else {
        toggle.style.right = '0';
    }
}

document.addEventListener('DOMContentLoaded', function () {
    initCharts();

    document.getElementById('history-toggle').addEventListener('click', toggleHistory);

    document.addEventListener('click', function (event) {
        const sidebar = document.getElementById('history-sidebar');
        const toggle = document.getElementById('history-toggle');

        if (!sidebar.contains(event.target) &&
            !toggle.contains(event.target) &&
            sidebar.classList.contains('active')) {
            toggleHistory();
        }
    });

    // Add event listener for clicking outside modal
    const modal = document.getElementById('history-modal');

    // Close modal when clicking outside
    window.onclick = function (event) {
        if (event.target == modal) {
            closeModal();
        }
    }

    // Close modal with escape key
    document.addEventListener('keydown', function (event) {
        if (event.key === 'Escape') {
            closeModal();
        }
    });
});

function resetGraph(configId) {
    if (modelData[configId].isTraining) {
        alert('Cannot reset while training is in progress');
        return;
    }

    // Reset the specific model's data
    modelData[configId] = {
        loss: { epoch: [], value: [], train: [] },
        accuracy: { epoch: [], value: [], test: [] },
        isTraining: false
    };

    // Reset metrics display
    document.getElementById(`${configId}_loss`).textContent = '-';
    document.getElementById(`${configId}_test_loss`).textContent = '-';
    document.getElementById(`${configId}_train_acc`).textContent = '-';
    document.getElementById(`${configId}_test_acc`).textContent = '-';

    // Clear logs
    document.getElementById(`logContainer${configId.slice(-1)}`).innerHTML = '';

    // Update charts to remove the reset model's data while keeping the other model's data
    const otherConfigId = configId === 'config1' ? 'config2' : 'config1';

    Plotly.update('lossChart', {
        x: [
            modelData[configId].loss.epoch, modelData[configId].loss.epoch,
            modelData[otherConfigId].loss.epoch, modelData[otherConfigId].loss.epoch
        ],
        y: [
            modelData[configId].loss.train, modelData[configId].loss.value,
            modelData[otherConfigId].loss.train, modelData[otherConfigId].loss.value
        ]
    });

    Plotly.update('accuracyChart', {
        x: [
            modelData[configId].accuracy.epoch, modelData[configId].accuracy.epoch,
            modelData[otherConfigId].accuracy.epoch, modelData[otherConfigId].accuracy.epoch
        ],
        y: [
            modelData[configId].accuracy.value, modelData[configId].accuracy.test,
            modelData[otherConfigId].accuracy.value, modelData[otherConfigId].accuracy.test
        ]
    });

    addLog(configId, 'Graph and metrics reset complete');
}

async function saveGraph(configId) {
    try {
        // Get the chart divs
        const lossChart = document.getElementById('lossChart');
        const accuracyChart = document.getElementById('accuracyChart');

        // Get current configuration
        const kernels = [
            parseInt(document.getElementById(`${configId}_kernel1`).value),
            parseInt(document.getElementById(`${configId}_kernel2`).value),
            parseInt(document.getElementById(`${configId}_kernel3`).value),
            parseInt(document.getElementById(`${configId}_kernel4`).value)
        ];
        const batchSize = document.getElementById(`${configId}_batch_size`).value;
        const optimizer = document.getElementById(`${configId}_optimizer`).value;
        const learningRate = document.getElementById(`${configId}_learning_rate`).value;

        // Create a zip file containing the graphs and configuration
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `training_results_${configId}_${timestamp}`;

        // Save loss chart
        const lossImage = await Plotly.toImage(lossChart, {
            format: 'png',
            width: 1200,
            height: 800
        });
        downloadImage(lossImage, `${filename}_loss.png`);

        // Save accuracy chart
        const accuracyImage = await Plotly.toImage(accuracyChart, {
            format: 'png',
            width: 1200,
            height: 800
        });
        downloadImage(accuracyImage, `${filename}_accuracy.png`);

        // Save configuration and results as JSON
        const configData = {
            timestamp: timestamp,
            model_id: configId,
            configuration: {
                kernels: kernels,
                batch_size: batchSize,
                optimizer: optimizer,
                learning_rate: learningRate
            },
            results: {
                loss: modelData[configId].loss,
                accuracy: modelData[configId].accuracy
            }
        };

        downloadJSON(configData, `${filename}_config.json`);

        addLog(configId, 'Graphs and configuration saved successfully');
    } catch (error) {
        console.error('Error saving graphs:', error);
        addLog(configId, 'Error saving graphs: ' + error.message);
    }
}

function downloadImage(dataUrl, filename) {
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function downloadJSON(data, filename) {
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
} 