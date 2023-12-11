

    Document.addEventListener('DOMContentLoaded', function () {
    const sendBtn = document.getElementById("send-btn");
    const userInput = document.getElementById("userInput");
    const chatbox = document.querySelector(".chat-body-inner"); // Define chatbox element

    sendBtn.addEventListener("click", async () => {
        const user_message = userInput.value.trim();
        if (user_message.length) {
            await postUserMessage(user_message);
            userInput.value = ""; // Clear the input field after sending
        }
    });

    const postUserMessage = async (user_message) => {
        const API_URL = "{% url 'chat' %}"; // Replace with your API endpoint

        try {
            const response = await fetch(API_URL, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    // Add any other headers if needed
                },
                body: JSON.stringify({ user_message: user_message }),
            });

            if (!response.ok) {
                throw new Error("Failed to post user message to the API");
            }

            const responseData = await response.json();

            const incomingMessage = createChatDiv(responseData.message, "incoming");
            chatbox.appendChild(incomingMessage);
        } catch (error) {
            console.error("Error:", error.message);
            // Handle the error if needed
        }
    };

    const createChatDiv = (message, className) => {
        const chatDiv = document.createElement("div");
        chatDiv.classList.add("message", `${className}`);
        let chatContent;
        if (className === "outgoing") {
            chatContent = "<p></p>";
        } else {
            chatContent = "<span class='material-symbols-outlined'>message here</span><p></p>";
        }
        chatDiv.innerHTML = chatContent;
        chatDiv.querySelector("div").textContent = message;
        return chatDiv;
    };

    const handleChat = (user_message) => {
        const outgoingMessage = createChatDiv(user_message, "outgoing");
        chatbox.appendChild(outgoingMessage);
        chatbox.scrollTo(0, chatbox.scrollHeight);

        setTimeout(async () => {
            const incomingMessage = createChatDiv("Thinking...", "incoming");
            chatbox.appendChild(incomingMessage);

            try {
                const responseData = await postUserMessage(user_message);
                incomingMessage.querySelector("div").textContent = responseData.message;
                chatbox.scrollTo(0, chatbox.scrollHeight);
            } catch (error) {
                console.error("Error:", error.message);
                // Handle the error if needed
            }
        }, 600);
    };
});

});