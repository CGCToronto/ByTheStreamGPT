// Create and inject the GPT window
function createGPTWindow() {
    const gptWindow = document.createElement('div');
    gptWindow.id = 'bythestream-gpt-window';
    gptWindow.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 350px;
        height: 500px;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        z-index: 10000;
        display: none;
    `;
    
    // Create header
    const header = document.createElement('div');
    header.style.cssText = `
        padding: 12px;
        background: #4CAF50;
        color: white;
        border-radius: 8px 8px 0 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    `;
    
    const title = document.createElement('span');
    title.textContent = '溪水旁 GPT Assistant';
    
    const closeButton = document.createElement('button');
    closeButton.textContent = '×';
    closeButton.style.cssText = `
        background: none;
        border: none;
        color: white;
        font-size: 24px;
        cursor: pointer;
        padding: 0 8px;
    `;
    
    header.appendChild(title);
    header.appendChild(closeButton);
    
    // Create iframe for popup content
    const iframe = document.createElement('iframe');
    iframe.src = chrome.runtime.getURL('popup.html');
    iframe.style.cssText = `
        width: 100%;
        height: calc(100% - 48px);
        border: none;
        border-radius: 0 0 8px 8px;
    `;
    
    gptWindow.appendChild(header);
    gptWindow.appendChild(iframe);
    document.body.appendChild(gptWindow);
    
    // Add toggle button
    const toggleButton = document.createElement('button');
    toggleButton.id = 'bythestream-gpt-toggle';
    toggleButton.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 50px;
        height: 50px;
        border-radius: 25px;
        background: #4CAF50;
        color: white;
        border: none;
        cursor: pointer;
        z-index: 10000;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    `;
    toggleButton.textContent = '💬';
    document.body.appendChild(toggleButton);
    
    // Event listeners
    toggleButton.addEventListener('click', () => {
        const isVisible = gptWindow.style.display === 'block';
        gptWindow.style.display = isVisible ? 'none' : 'block';
        toggleButton.textContent = isVisible ? '💬' : '✕';
    });
    
    closeButton.addEventListener('click', () => {
        gptWindow.style.display = 'none';
        toggleButton.textContent = '💬';
    });
}

// Initialize when the page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', createGPTWindow);
} else {
    createGPTWindow();
} 