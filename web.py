#Imports do projeto
from flask import Flask, Response, render_template
from detection import VideoCamera, dado, timer, cont_hist, risk, labels, gen

#URL do vídeo de stream
url = "https://youtu.be/IVLD-t_vzeM"

app = Flask(__name__)

@app.route('/')
def home():
     return render_template("home.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera(url)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False)