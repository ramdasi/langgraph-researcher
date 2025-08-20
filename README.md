# AI Agent Project

This is an AI Agent project built with Python, LangChain, and OpenAI/Ollama models.  
It supports conversational AI, tool usage, and knowledge storage with ChromaDB.

---

## 🚀 Features
- Conversational AI with LangChain
- Tool calling and execution
- Knowledge persistence using **ChromaDB**
- Environment variables managed via `.env`
- REST API with **Flask**

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ai-agent.git
   cd ai-agent
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your `.env` file:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

---

## ▶️ Usage

Start the Flask server:
```bash
python app.py
```

Then open `http://localhost:5000` in your browser or send requests using curl/postman.

---

## 📂 Project Structure
```
.
├── app.py              # Main Flask server
├── agent.py            # AI Agent logic
├── requirements.txt    # Dependencies
├── README.md           # Documentation
└── .env                # API keys & configs
```

---

## 🛠 Tech Stack
- **Python**
- **LangChain**
- **OpenAI / Ollama**
- **ChromaDB**
- **Flask**

---

## 📜 License
This project is licensed under the MIT License.
