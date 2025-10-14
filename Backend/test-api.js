require('dotenv').config();
const OpenAI = require('openai');

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
});

async function testAPI() {
    try {
        const response = await openai.chat.completions.create({
            model: "gpt-3.5-turbo",
            messages: [
                {
                    role: "user",
                    content: "Hello, this is a test message."
                }
            ],
            max_tokens: 5
        });
        console.log("API Key is working! Response:", response.choices[0].message);
    } catch (error) {
        console.error("Error testing API:", error.message);
        if (error.response) {
            console.error("Full error details:", error.response.data);
        }
    }
}

testAPI(); 