# Media Bias Detector — Political News Classification

This project presents a fine-tuned **RoBERTa-based** text classification model that predicts the political bias of news headlines — whether they lean **Left**, **Right**, or are **Neutral**.  

Using **Python** and **Streamlit**, the app provides an intuitive interface where you can input a headline and receive real-time bias predictions. Backend model weights are pulled directly from the Hugging Face repository.

---

## Tools & Technologies

- **Python**: Backend scripting and inference with `transformers`, `torch`, and `streamlit`
- **Hugging Face Hub**: Hosted model weights & tokenization
- **Streamlit**: Interactive UI for real-time inference
- **Data Skills Applied**:
  - Text preprocessing & tokenization
  - Confidence scoring and model predictions
  - Model fine-tuning on political datasets
  - Web deployment and user interface design

---

## Project Components

| File / Folder | Description |
|---------------|-------------|
| `app/streamlit_app.py` | Main Streamlit app for user interaction & model inference |
| `requirements.txt` | List of dependencies for running the app |
| `models/` | Directory for cached model files and configurations |
| `README.md` | Documentation for project overview and usage |

---

## Key Features Implemented

- Real-time news bias detection for user-supplied headlines
- Interactive UI with dynamic visual cues and loader
- Backend fine-tuned RoBERTa model with political bias labels
- Model and tokenizer fetched from Hugging Face repository
- Confidence indicator as a color bar for predicted class

---

## Test with political headlines like:
- “Supreme Court to hear petitions on electoral bond transparency next week”
- “PM Modi’s leadership praised globally as India rises to global prominence”
- “Farmers continue protest at Delhi borders demanding policy change”

---

## Objective

This project was undertaken to:
- Demonstrate text classification techniques on real-world data
- Communicate model predictions through a polished UI
- Highlight proficiency with NLP and frontend integration for production-ready data tools

---

## Contact

For inquiries or collaboration opportunities:  
📧 **Email**: [cvchahat25@gmail.com](mailto:cvchahat25@gmail.com)  
🔗 **LinkedIn**: [Chahat Verma](https://www.linkedin.com/in/chahatverma-v777)
