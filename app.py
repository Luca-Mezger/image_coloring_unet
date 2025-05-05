from flask import Flask, request, send_file, render_template, abort
import io
from inference import load_model_state, colorize_bytes

app = Flask(__name__)

# Load the model once on start‑up (takes a few hundred ms on CPU)
STATE = load_model_state()

@app.route("/", methods=["GET"])
def index():
    # templates/index.html
    return render_template("index.html")

@app.route("/colorize", methods=["POST"])
def colorize():
    if "image" not in request.files:
        abort(400, "no file field named 'image'")
    img_file = request.files["image"]
    if img_file.filename == "":
        abort(400, "empty filename")

    try:
        png_bytes = colorize_bytes(STATE, img_file.read())
    except Exception as e:
        abort(500, f"colorization failed: {e}")

    return send_file(
        io.BytesIO(png_bytes),
        mimetype="image/png",
        download_name="colorized.png",
    )

if __name__ == "__main__":
    # Debug / hot‑reload
    app.run(debug=True)
