// The interface is responsive, supports suggested questions,
// displays retrieval metadata, and safely renders answers as text.
const form = document.getElementById("chat-form");
const input = document.getElementById("question");
const messages = document.getElementById("messages");
const sendButton = document.getElementById("send-button");

function addMessage(text, sender, meta = "") {
    const wrapper = document.createElement("div");
    wrapper.className = `message ${sender}`;

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    // textContent prevents user or model text from being treated as HTML.
    bubble.textContent = text;

    if (meta) {
        const details = document.createElement("span");
        details.className = "meta";
        details.textContent = meta;
        bubble.appendChild(document.createElement("br"));
        bubble.appendChild(details);
    }

    wrapper.appendChild(bubble);
    messages.appendChild(wrapper);
    messages.scrollTop = messages.scrollHeight;
    return wrapper;
}

async function askQuestion(question) {
    if (!question.trim()) return;

    addMessage(question, "user");
    input.value = "";
    input.focus();
    sendButton.disabled = true;
    sendButton.textContent = "...";

    const loading = addMessage("Searching the agriculture knowledge base...",
                                "bot");

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question })
        });

        const data = await response.json();
        loading.remove();

        if (!response.ok) {
            addMessage(data.error || "The request failed.", "bot");
            return;
        }

        let meta = "";

        if (data.accepted) {
            meta = `Topic: ${data.topic} · Similarity: ${data.similarity}`;
        } else {
            meta = "No sufficiently relevant answer found.";
        }

        addMessage(data.answer, "bot", meta);

    } catch (error) {
        loading.remove();
        addMessage(
            "I could not connect to the server. Please try again.",
            "bot"
        );
    } finally {
        sendButton.disabled = false;
        sendButton.textContent = "Send ↑";
    }
}

form.addEventListener("submit", (event) => {
    event.preventDefault();
    askQuestion(input.value.trim());
});

document.querySelectorAll("[data-question]").forEach((button) => {
    button.addEventListener("click", (event) => {
        event.preventDefault();
        askQuestion(button.dataset.question);
    });
});