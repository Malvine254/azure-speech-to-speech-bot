from flask import Flask, request, jsonify, render_template
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI

app = Flask(__name__)

# Your Azure OpenAI and Speech SDK configurations
api_key = "bf960d750ff946e8a8908e7f5ed53b71"
client = AzureOpenAI(
    azure_endpoint="https://armelyopenai.openai.azure.com/",
    api_key=api_key,
    api_version="2024-02-15-preview"
)
deployment_id = 'gpt-4-model'
speech_config = speechsdk.SpeechConfig(subscription='adff6f8e12d24ecf8f12cacb35b9ed12', region='eastus')
speech_config.speech_recognition_language = "en-US"
speech_config.speech_synthesis_voice_name = 'en-US-JennyMultilingualNeural'
audio_output_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output_config)
tts_sentence_end = [".", "!", "?", ";", "。", "！", "？", "；", "\n"]

# Function to ask Azure OpenAI and return synthesized response
def ask_openai(prompt):
    response = client.chat.completions.create(model=deployment_id, max_tokens=200, stream=True, messages=[
        {"role": "user", "content": prompt}
    ])
    collected_messages = []
    synthesized_response = ""

    for chunk in response:
        if len(chunk.choices) > 0:
            chunk_message = chunk.choices[0].delta.content
            if chunk_message is not None:
                collected_messages.append(chunk_message)
                if chunk_message in tts_sentence_end:
                    text = ''.join(collected_messages).strip()
                    if text != '':
                        synthesized_response += text
                        collected_messages.clear()
    return synthesized_response

# Route to handle incoming requests
@app.route('/process_request', methods=['POST'])
def process_request():
    if request.method == 'POST':
        try:
            data = request.json
            user_input = data['user_input']
            openai_response = ask_openai(user_input)
            return jsonify({"user_response": user_input, "openai_response": openai_response})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

# Route to render the HTML template
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
