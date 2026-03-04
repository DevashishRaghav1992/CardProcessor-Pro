document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileListSection = document.getElementById('file-list');
    const filesPreview = document.getElementById('files-preview');
    const fileCount = document.getElementById('file-count');
    const processBtn = document.getElementById('process-btn');
    const loadingState = document.getElementById('loading-state');
    const resultState = document.getElementById('result-state');
    const downloadBtn = document.getElementById('download-btn');
    const resetBtn = document.getElementById('reset-btn');

    let selectedFiles = [];

    // Trigger file input dialog on click
    dropZone.addEventListener('click', () => fileInput.click());

    // Handle File Input Change
    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    // Drag & Drop Handlers
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        handleFiles(dt.files);
    }, false);

    function handleFiles(files) {
        selectedFiles = Array.from(files).filter(file => file.type.startsWith('image/'));
        if (selectedFiles.length > 0) {
            updateUI(true);
            renderFilePreview();
        }
    }

    function updateUI(hasFiles) {
        if (hasFiles) {
            dropZone.classList.add('hidden');
            fileListSection.classList.remove('hidden');
        } else {
            dropZone.classList.remove('hidden');
            fileListSection.classList.add('hidden');
            resultState.classList.add('hidden');
            loadingState.classList.add('hidden');
            selectedFiles = [];
        }
    }

    function renderFilePreview() {
        fileCount.textContent = selectedFiles.length;
        filesPreview.innerHTML = '';
        selectedFiles.forEach(file => {
            const li = document.createElement('li');
            li.textContent = file.name;
            li.title = file.name;
            filesPreview.appendChild(li);
        });
    }

    // Processing Logic
    processBtn.addEventListener('click', async () => {
        if (selectedFiles.length === 0) return;

        fileListSection.classList.add('hidden');
        loadingState.classList.remove('hidden');

        const formData = new FormData();
        selectedFiles.forEach(file => {
            formData.append('files', file);
        });

        try {
            const response = await fetch('/api/process', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);

                loadingState.classList.add('hidden');
                resultState.classList.remove('hidden');

                downloadBtn.href = url;
                downloadBtn.download = `processed_cards_${new Date().getTime()}.zip`;
            } else {
                const err = await response.json();
                alert(`Error: ${err.error || 'Server error'}`);
                updateUI(true);
            }
        } catch (error) {
            console.error(error);
            alert('Failed to connect to the server.');
            updateUI(true);
        }
    });

    resetBtn.addEventListener('click', () => {
        fileInput.value = '';
        updateUI(false);
    });
});
