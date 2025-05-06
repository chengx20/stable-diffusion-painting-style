document.addEventListener("DOMContentLoaded", () => {
    const promptInput = document.getElementById("promptInput");
    const wordCount = document.getElementById("wordCount");
    const generateArt = document.getElementById("generateArt");
    const statusMessage = document.getElementById("statusMessage");
    const artworkContainer = document.getElementById("artworkContainer");

    document.getElementById('exampleBlueFlower').addEventListener('click', () => promptInput.value = "A beautiful blue flower");
    document.getElementById('exampleYellowFox').addEventListener('click', () => promptInput.value = "A cunning yellow fox");
    document.getElementById('exampleDog').addEventListener('click', () => promptInput.value = "A happy dog playing in the park");

    // Initialize word count
    promptInput.dispatchEvent(new Event('input'));
    promptInput.addEventListener("input", () => {
        const words = promptInput.value.match(/\b[-?(\w+)?]+\b/gi);
        const count = words ? words.length : 0;
        wordCount.textContent = `Word Count: ${count}/50`;
        wordCount.style.color = count > 50 ? 'red' : 'black';
        generateArt.disabled = count > 50;
    });

    generateArt.addEventListener("click", async () => {
        const prompt = promptInput.value.trim();
        const wordCountCheck = prompt.split(/\s+/).filter(w => w).length;
        if (prompt === "" || wordCountCheck > 50) {
            alert("Please enter a prompt with fewer than 50 words.");
            return;
        }

        statusMessage.textContent = "Generating your artwork...";
        let counter = 0;
        const intervalId = setInterval(() => {
            const messages = [
                "Still trying to generate...",
                "It's more art than science...",
                "Almost there..."
            ];
            statusMessage.textContent = messages[counter % messages.length];
            counter++;
        }, 15000);

        try {
            const response = await fetch('http://localhost:4777/generate-images', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ prompt })
            });

            clearInterval(intervalId);

            if (response.ok) {
                const { images } = await response.json();
                statusMessage.textContent = "Artwork generated successfully!";
                artworkContainer.innerHTML = "";  // clear previous
                images.forEach(base64Image => {
                    const img = new Image();
                    img.src = `data:image/png;base64,${base64Image}`;
                    img.classList.add("generated-art");
                    artworkContainer.appendChild(img);
                });
            } else {
                statusMessage.textContent = "Failed to generate artwork. Please try again.";
            }
        } catch (error) {
            clearInterval(intervalId);
            statusMessage.textContent = `Error: ${error.message}`;
        }
    });
});
