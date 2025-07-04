# Youtube-chatbot
🎥 YouTube Video Question Answering App
An AI-powered Streamlit app that allows users to ask questions based on a YouTube video's transcript. Perfect for summarizing lecture videos, diving into interviews, or extracting insights from talks—all powered by LangChain and OpenAI.
🚀 Features
🔍 Automatic transcript retrieval using youtube_transcript_api

🧠 Contextual Q&A powered by LangChain & GPT-4o-mini

📌 Semantic search with FAISS vector store

🎯 Streamlit UI for a clean and interactive experience

![image](https://github.com/user-attachments/assets/a8dcff29-b936-422f-98cb-1d21adec8ffa)




🛠️ How It Works
User pastes a YouTube URL

App extracts video ID and fetches transcript

Transcript is split into chunks using RecursiveCharacterTextSplitter

Chunks embedded using OpenAIEmbeddings and stored in FAISS

LangChain retrieves relevant chunks based on the user’s question

GPT-4o-mini answers using only the retrieved context



![image](https://github.com/user-attachments/assets/d29a6594-d6c7-4ba2-9f95-469dfcebf5c5)


⚠️ Additional Notes
You’ll also need to set your OpenAI API key as an environment variable and store it as .env file.
OPENAI_API_KEY="" place your key here.




