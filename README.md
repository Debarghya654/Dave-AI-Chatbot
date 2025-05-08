# Dave-AI-Chatbot

Core Functionality:
1. AI Chatbot Interface: Dave interacts conversationally using typed or spoken input from users.
2. Natural Language Processing (NLP): Uses NLTK and spaCy to preprocess user input and detect intent (code request vs. general chat).
3. Context Management: Maintains a history of the last n conversations to generate context-aware responses.

Code Generation Capability:
1. Code Generator via Transformers: Integrates HuggingFace’s pipeline() to generate programming code from natural language descriptions.
2. Multi-Language Code Support: Automatically detects and generates code in: Python (default), JavaScript, C++, Java.
3. Syntax Highlighting: Uses pygments to highlight the generated code in the terminal for better readability.

Voice Interaction:
1. Speech Recognition: Uses speech_recognition to allow users to speak their queries.
2. Text-to-Speech Output: Replies are spoken aloud using pyttsx3, simulating a real assistant.

Conversation Management:
1. Conversation Saving and Loading: Chats can be saved to or loaded from a JSON file for future reference.
2. Conversation Summary and Analytics: summary command prints the full chat log, intents command shows frequency of detected intents.

Error Handling & Logging:
1. Robust Logging: Logs are written for model loading, response generation, file I/O, and errors.
2. Graceful Fallbacks: When errors occur (e.g., API failure or empty voice input), Dave replies with helpful messages.

Extensible Design:
1. Modular Structure: Code is split into functions (intent detection, code generation, etc.) for easy maintenance.
2. Custom Prompt Templates: Code-generation prompts are customizable for different language output or user style.
