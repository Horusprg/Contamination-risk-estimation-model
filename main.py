#Imports do projeto
from flask import Flask, Response, render_template, request
import plotly.graph_objects as go
import plotly.express as px
from detection import VideoCamera, gen, df, risk, labels, videoconfig
import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input
import flask

#variáveis de entrada
qgrate = 0
area = 0

#Iniciando o servidor
server = Flask(__name__)

#Página inicial
@server.route('/')
def index():
    return render_template('index.html')

#Rota de transmissão de vídeo
@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera(qgrate, area)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#Página de detecção
app = dash.Dash(server=server, url_base_pathname='/detect/')
@app.server.route('/post', methods=['POST'])
def on_post():
    global area
    global qgrate
    area = request.form["area"]
    qgrate = request.form["qgrate"]
    #Dicionário dos quantas disponíveis
    quantas = {'sars-cov-2':2.3666, }
    qgrate = quantas[qgrate]
    return flask.redirect('/detect/')

#Layout do web app de detecção
app.layout = html.Div(
    className= "layout",
    children=[
        html.Div(className="head",
        children=[
            html.H5("MONITORAMENTO DE LOCAL", className="anim-typewriter"),
            html.H4("AMBIENTE"),
            html.Img(className= "video",src="/video_feed"),
        ]),
        html.Div("CLASSES", className="classes"),
        html.H3("PESSOAS AO LONGO DO DIA", className="contPess"),
        dcc.Graph(id='live-peopleCount', className='contagem'),
        html.H3("MAPA DE OCUPAÇÃO", className="contPess2d"),
        dcc.Graph(id='live-peopleHeatmap', className='contagem3d'),
        dcc.Graph(id='live-velocimeter', className='velocimeter'),
        dcc.Graph(id='live-pie', className='pie'),
        html.Div([html.Button("Download CSV", id="btn_csv"),
                    dcc.Download(id="download-dataframe-csv")]),
        dcc.Interval(
            id='interval-component',
            interval=1*1000,
            n_intervals=0
            )
    ]
)

@app.callback(
            Output('live-peopleCount', 'figure'),
            [Input('interval-component', "n_intervals")]
            )
def peopleCount(n_intervals):
    fig = go.Figure(layout={"template":"plotly_dark"})
    fig.add_trace(go.Bar(x=df["timer"], y=df["count"]))
    fig.update_layout(
        paper_bgcolor="#242424",
        plot_bgcolor="#242424",
        autosize=True,
        margin=dict(l=10, r=10, b=30, t=10),
        )
    return fig


@app.callback(
            Output('live-peopleHeatmap', 'figure'),
            [Input('interval-component', "n_intervals")]
            )

def peopleHeatmap(n_intervals):
    x0 = list(filter(None, df["bbox_x0"]))
    xi = list(filter(None, df["bbox_xi"]))
    y0 = list(filter(None, df["bbox_y0"]))
    yi = list(filter(None, df["bbox_yi"]))
    fig = go.Figure(go.Histogram2d(
                    x=(list(map(lambda x,y: (x[0]+y[0])/2, x0, xi))),
                    y=(list(map(lambda x,y: (x[0]+y[0])/2, y0, yi))),
                    histnorm='percent',
                    autobinx=False,
                    xbins=dict(start=0, size=5),
                    autobiny=False,
                    ybins=dict(start=0, size=5)
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
    fig = px.pie(values = labels.values(), names = labels.keys())
    fig.update_layout(legend_font_size = 32,paper_bgcolor = "rgb(3, 7, 15)")
    
    return fig

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    import pandas as pd
    dataframe = pd.DataFrame.from_dict(df)
    return dcc.send_data_frame(dataframe.to_csv, "data.csv")

if __name__ == '__main__':
    app.run_server(debug=True)