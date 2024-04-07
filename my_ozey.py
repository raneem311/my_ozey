from flask import Flask, render_template, request
import vertexai
from vertexai.language_models import ChatModel
import vertexai.preview.generative_models as generative_models
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "GOOGLE_APPLICATION_CREDENTIALS.json"

vertexai.init(project="wth-418421", location="us-central1")
chat_model = ChatModel.from_pretrained("chat-bison")

parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "temperature": 0.9,
    "top_p": 1
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}


app = Flask(__name__)

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        user_input = request.form["user_input"]

        if user_input.lower() == "exit":
            return render_template("chat.html", response="AI Bot: Goodbye!")

        chat = chat_model.start_chat(
            context="""You are an AI friend called Ozey. You act as a personal cognitive behavioural therapist  to support users on their journey to emotional well-being.
                 make them Feel free to share their thoughts and feelings with you, and together  can work towards positive changes. If the user asks anything outside the domain, you should respond with \"My main focus is in mental health, your mental health. Sorry, I can't answer this.\"
                 When a session starts you should introduce yourself, ask the user how they feel and make them feel free to talk with you.
                 make your answers neither too long nor too short""",
        )
        response = chat.send_message(user_input, **parameters)
        return response.text

    return  "error"

if __name__ == "__main__":
    app.run(debug=True)

