from distutils.log import debug
from flask import Flask, Response, request, redirect
from detection import VideoCamera, gen

app = Flask(__name__, static_folder= "static")
@app.route('/')
def index():
    return redirect("/static/index.html")

@app.route('/video_feed', methods=["POST"])
def video_feed():
    if request.method == "POST":
        url = request.form["url"]
        return Response(gen(VideoCamera(url)),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return "INVALID ACESS"

if __name__ == '__main__':
    app.run(debug=True)