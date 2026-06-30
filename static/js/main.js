// ========================================
// FILE UPLOAD HANDLING
// ========================================

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const fileName = document.getElementById('fileName');
const verifyBtn = document.getElementById('verifyBtn');

// Click to upload
if (uploadArea) {
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });
}

// Drag and drop
if (uploadArea) {
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
}

// File input change
if (fileInput) {
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleFile(fileInput.files[0]);
        }
    });
}

function handleFile(file) {
    const validTypes = ['image/png', 'image/jpeg', 'image/webp', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
        alert('Please upload a valid image (PNG, JPG, JPEG, WEBP)');
        fileInput.value = '';
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        alert('File too large. Maximum size is 10MB.');
        fileInput.value = '';
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.style.display = 'block';
        if (fileName) {
            fileName.textContent = file.name;
            fileName.style.display = 'block';
        }
        if (verifyBtn) {
            verifyBtn.disabled = false;
        }
        const errorMsg = document.querySelector('.error-msg');
        if (errorMsg) errorMsg.remove();
    };
    reader.readAsDataURL(file);
}

// Verify button
if (verifyBtn) {
    verifyBtn.addEventListener('click', () => {
        if (!fileInput.files.length) {
            alert('Please select a screenshot first.');
            return;
        }
        verifyBtn.classList.add('loading');
        verifyBtn.disabled = true;
        const form = document.getElementById('uploadForm');
        if (form) {
            form.submit();
        }
    });
}

console.log('🛡️ PayVerify Nepal loaded successfully!');