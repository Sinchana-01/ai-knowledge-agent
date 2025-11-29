AI Knowledge Base and Resume Screening Agent
1. Overview

This project implements two independent AI agents inside a single Streamlit interface:

 1.1 Knowledge Base Agent

Allows users to upload PDF or TXT documents and ask questions.
The agent performs:

* Document ingestion
* Text extraction
* TF-IDF vectorization
* Context retrieval using cosine similarity
* Context-aware answering using Google Gemini

1.2 Resume Screening Agent

Allows users to upload one job description and multiple resumes.
The agent:

* Extracts text from documents
* Compares each resume with the job description
* Generates a match score (1–10)
* Provides justification
* Ranks candidates based on score

The complete system runs on Streamlit Cloud and uses the Google Gemini API.

---

2. Key Features

2.1 Knowledge Base Agent

* Accepts PDF and TXT documents
* Extracts text automatically
* Builds TF-IDF index (scikit-learn)
* Retrieves relevant content using cosine similarity
* Generates context-aware answers using Gemini
* Provides fallback answers when no context is found
* Limits responses to user-provided content (reduces hallucinations)

2.2 Resume Screening Agent

* Accepts a job description and multiple resumes
* Extracts and processes documents
* Uses a structured prompt to evaluate each resume
* Produces:

  * Match score (1–10)
  * Explanation summary
* Ranks resumes in descending order of relevance

2.3 General Features

* Clean, minimal UI
* Stateless, fast runtime
* Secure API key handling using Streamlit Secrets
* Deployable on Streamlit Cloud

---

3. System Architecture

The architecture diagram is available at:

```
docs/Architecture.png
```

3.1 Architecture Components

1. **User Interface (Streamlit)**

   * Mode selection
   * File uploads
   * Query input
   * Results display

2. **Knowledge Base Pipeline**

   * PDF/TXT text extraction
   * TF-IDF vectorizer
   * Context retrieval
   * Gemini inference

3. **Resume Screening Pipeline**

   * JD extraction
   * Resume extraction
   * Scoring and ranking via Gemini

4. **External Services**

   * Google Gemini API (gemini-flash-latest)

5. **Deployment Layer**

   * GitHub repository
   * Streamlit Cloud hosting

---

4. Technology Stack

* Python 3.9+
* Streamlit
* Google Gemini Flash API
* scikit-learn (TF-IDF Vectorizer)
* pypdf
* python-dotenv
* GitHub
* Streamlit Cloud

---

5. Project Directory Structure

```
ai-knowledge-agent/
│
├── app.py                      Main Streamlit application
├── requirements.txt            Dependency list
├── README.md                   Documentation
│
├── docs/
│   └── Architecture.png        Architecture diagram
│
├── samples/
│   ├── JD_Backend_Engineer.pdf
│   ├── Resume_Aisha_StrongMatch.pdf
│   ├── Resume_Rohit_MediumMatch.pdf
│   └── Resume_Priya_WeakMatch.pdf
│
└── data/                       Runtime storage for uploaded files
```

---

6. Local Development Setup

6.1 Clone the Repository

```
git clone https://github.com/<your-username>/ai-knowledge-agent.git
cd ai-knowledge-agent
```

6.2 Create and Activate a Virtual Environment

**Windows**

```
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux**

```
python3 -m venv venv
source venv/bin/activate
```

6.3 Install Dependencies

```
pip install -r requirements.txt
```

6.4 Configure API Key

Create a `.env` file:

```
GEMINI_API_KEY=apikey

```

6.5 Run the Application

```
streamlit run app.py
```

Open the local URL displayed in the terminal.

---

7. Streamlit Cloud Deployment

7.1 Push Code to GitHub

Ensure that the repository contains:

* app.py
* requirements.txt
* docs/
* samples/
* README.md

7.2 Deploy

1. Go to: [https://share.streamlit.io](https://share.streamlit.io)
2. Create a new app
3. Select:

   * Repository: ai-knowledge-agent
   * Branch: main
   * Main file: app.py

7.3 Set API Secrets

In Streamlit Cloud:

```
GEMINI_API_KEY = apikey
"
```

Streamlit will automatically deploy and host the application.

---

8. Functional Workflow

8.1 Knowledge Base Agent Workflow

1. User uploads one or more documents
2. Files stored in `data/`
3. System extracts text using:

   * PdfReader (PDF)
   * Native reader (TXT)
4. TF-IDF vectorizer indexes document content
5. User enters a query
6. Query vectorized and compared using cosine similarity
7. Top-K segments selected as context
8. Prompt created:

   * Context
   * Query
   * Instructions
9. Gemini generates final answer
10. Answer returned with source references

8.2 Resume Screening Workflow

1. User uploads a job description
2. User uploads multiple resumes
3. Text extracted from all documents
4. For each resume:

   * JD + resume → scoring prompt
   * Gemini returns:

     * Score (1–10)
     * Justification summary
5. Scores collected and sorted
6. Final ranked list produced

---

9. License

This project is for academic and evaluation purposes only.

---