import { SarvamAIClient } from "sarvamai";

const SARVAM_API_KEY = "sk_4va96p7v_jZZcpQjF9GcybnBtBsXnORiT"

const client = new SarvamAIClient({
    apiSubscriptionKey: SARVAM_API_KEY,
});

// Changing text
const respone4 = await client.text.translate({
    input: "Hi my name is veer",
    source_language_code: "auto",
    target_language_code: "hi-IN",
    speaker_gender: "Male"
});
console.log(respone4)

// Talking to LLM    
const response3 = await client.chat.completions({
    messages: [
        {role: "user", content: "give me some unknown facts about our country"}
    ]
});
console.log(response3.choices[0].message.content);

// text to speech
const response2 = await client.textToSpeech.convert({
    text: "hey my name is veer",
    target_language_code: "hi-IN",
});
console.log(response2)

// Text Translation
const response = await client.text.translate({
    input: "Hello, how are you?",
    source_language_code: "auto",
    target_language_code: "hi-IN",
    speaker_gender: "Male"
});
console.log(response);