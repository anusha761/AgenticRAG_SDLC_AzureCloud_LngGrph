from pyngrok import ngrok
import os
from pyngrok import conf



os.environ["OPENAI_API_KEY"] = "<api_key>"


conf.get_default().auth_token = "<ngrok_token>"

%cd agenticrag

# Kill any existing tunnels
ngrok.kill()

# Run Streamlit in background
os.system("streamlit run main.py &")

# Open Ngrok tunnel
public_url = ngrok.connect(8501)
print(f"Open your chatbot here: {public_url}")
