// scripts.js
document.getElementById('profile-picture').addEventListener('change', function() {
    const fileList = this.files;
    const fileListContainer = document.getElementById('file-list');
    fileListContainer.innerHTML = '';

    for (let i = 0; i < fileList.length; i++) {
        const listItem = document.createElement('div');
        listItem.textContent = fileList[i].name;
        fileListContainer.appendChild(listItem);
    }
});

document.getElementById('registration-form').addEventListener('submit', function(event) {
    event.preventDefault(); // ป้องกันการส่งฟอร์มจริง
    document.getElementById('success-modal').style.display = 'block';
});

document.getElementById('modal-ok-btn').addEventListener('click', function() {
    document.getElementById('success-modal').style.display = 'none';
});