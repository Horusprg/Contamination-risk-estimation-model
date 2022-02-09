#Imports do projeto
import dash
import dash_core_components as dcc
import dash_html_components as html
from flask import Flask, Response
from datetime import timedelta
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Output, Input
from detection import VideoCamera, dado, timer, cont_hist, risk, labels


#URL do vídeo de stream
url = "https://youtu.be/IVLD-t_vzeM"

#Stream da detecção de vídeo
def gen(camera):
    while True:
        frame = camera.get_frame(timer, cont_hist, dado)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

@server.route('/video_feed')

def video_feed():
    return Response(gen(VideoCamera(url)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.callback(
            Output('live-update-graph', 'figure'),
            [Input('interval-component', "n_intervals")]
            )

def update_graph(n_intervals):
    tempo = []
    for sec in timer:
        sec = str(timedelta(seconds = sec))
        tempo.append(sec)

    fig = go.Figure(layout={"template":"plotly_dark"})
    fig.add_trace(go.Bar(x=tempo, y=cont_hist))
    fig.update_layout(
        paper_bgcolor="#242424",
        plot_bgcolor="#242424",
        autosize=True,
        margin=dict(l=10, r=10, b=30, t=10),
        )

    return  fig

@app.callback(
            Output('live-update-3d', 'figure'),
            [Input('interval-component', "n_intervals")]
            )

def update_3d(n_intervals):
    xAxes = []
    yAxes = []
    zAxes = []
    for i in range(0, len(dado)):
        for j in range(0, len(dado[i])):
            xAxes.append(dado[i][j][0])
            yAxes.append(-dado[i][j][1])
            zAxes.append(cont_hist)
    fig = go.Figure(go.Histogram2d(
                    x=(xAxes),
                    y=(yAxes),
                    z=(zAxes)
    ))
    return fig

@app.callback(
            Output('live-velocimeter', 'figure'),
            [Input('interval-component', "n_intervals")]
            )
def velocimeter(n_intervals):
    fig = go.Figure()
    fig.add_trace(go.Indicator(
    value = risk[-1],
    delta = {'reference': risk[-2]},
    gauge = {
        'axis': {'visible': False}},
    domain = {'row': 0, 'column': 0}))
    fig.update_layout(
    grid = {'rows': 1, 'columns': 1, 'pattern': "independent"},
    template = {'data' : {'indicator': [{
        'number':{'font_color':"white", 'suffix': "%"},
        'gauge':{'axis_range': (0,100)},
        'title': {'text': "Risco de contaminação", 'font_color':"white", 'font_size': 48},
        'mode' : "number+delta+gauge",
        'delta' : {'reference': risk[-2]}}]
                         }})
    fig.update_layout(paper_bgcolor = "rgb(3, 7, 15)")
    return fig

@app.callback(
            Output('live-pie', 'figure'),
            [Input('interval-component', "n_intervals")]
            )

def pie(n_intervals):
    fig = px.pie(values = labels[1], names = labels[0])
    fig.update_layout(legend_font_size = 32,paper_bgcolor = "rgb(3, 7, 15)")
    
    return fig

app.layout = html.Div(
    className= "layout",
    children=[
        html.H5("MONITORAMENTO DE LOCAL", className="anim-typewriter"),
        html.Img(className="button",src="assets/Group 3.png"),
        html.H4("AMBIENTE"),
        html.Img(className= "video",src="/video_feed"),
        html.Div("CLASSES", className="classes"),
        html.H3("PESSOAS AO LONGO DO DIA", className="contPess"),
        dcc.Graph(id='live-update-graph', className='contagem'),
        html.H3("MAPA DE OCUPAÇÃO", className="contPess2d"),
        dcc.Graph(id='live-update-3d', className='contagem3d'),
        dcc.Graph(id='live-velocimeter', className='velocimeter'),
        dcc.Graph(id='live-pie', className='pie'),
        dcc.Interval(
            id='interval-component',
            interval=1*1000,
            n_intervals=0
            )
    ]
)

if __name__ == '__main__':
    app.run_server(debug=False)