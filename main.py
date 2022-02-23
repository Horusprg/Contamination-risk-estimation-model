from flask import Flask, Response, request, render_template, make_response
from detection import VideoCamera, gen

app = Flask(__name__, template_folder="templates")


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/detect', methods=["POST"])
def detect():
    resp = make_response(render_template("detect.html"))
    url = request.form['url']
    resp.set_cookie("url", url)
    return resp


@app.route('/video_feed', methods=["GET"])
def video_feed():
    if request.cookies.get("url") != 'none':
        return Response(gen(VideoCamera(request.cookies.get("url"))),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
