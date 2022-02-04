# <p align="center"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="25" height="25"/> Contamination Risk Estimation Model using YOLOv5 - v1.0 </p>
<h3 align="center">Com os eventos propiciados pelas epidemias ocorridas desde a eclosão do vírus SARS-CoV-2, surgiu a ideia de produzir esta ferramenta destinada ao monitoramento e controle de ambientes fechados contra doenças transmissíveis via aérea.
Portanto, são utilizadas equações com base nas propostas por Wells-Riley para estimar a qualidade do ar no local e o risco de contaminação.</>

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# <p align="center">Módulos</p>
<h3>Essa ferramenta está dividida em 3 módulos, sendo 2 deles pertencentes a esse repositório, são eles:</>

1. Aplicação principal, contendo o modelo de estimação do risco de contaminação, detecção de objetos utilizando YOLOv5 e página WEB com informações acerca do local monitorado.

2. Redes neurais treinadas, armazenadas na pasta "wheight", contendo diversas subversões do YOLOv5 de acordo com o uso desejado.

3. Estrutura do YOLOv5, que fornece os arquivos e ferramentas necessárias para a detecção e classificação de pessoas por vídeo.

*Observação: para acessar o 3º módulo, é necesário baixá-la no github dos desenvolvedores do YOLO, disponível em:*

<h2 align="center"><img src="https://img.icons8.com/glyph-neue/256/ffffff/github.png" width="20" height="20"/> https://github.com/ultralytics/yolov5</>

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# <p align="center">Requisitos</p>
<p align="left">
1. Python (https://www.python.org)<br><br>
2. Torch (https://pytorch.org)<br><br>
3. Dash (https://dash.plotly.com)<br><br>
4. Plotly (https://plotly.com/python/)<br><br>
5. Streamlink (https://streamlink.github.io/install.html)<br><br>
6. Numpy (https://numpy.org)<br><br>
7. Matplotlib (matplotlib)<br><br>
8. OpenCV-python (https://opencv.org)<br><br>
9. Pillow (https://pillow.readthedocs.io/en/stable/)<br><br>
10. PyYAML (https://pyyaml.org)<br><br>
11. Torchvision (https://pytorch.org/vision/stable/index.html)<br><br>
12. Cuda Toolkit (https://developer.nvidia.com/cuda-toolkit)
</p>

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
<h1 align="center">Módulo 1: main.py</>
<h3>A aplicação principal apresenta diversas ferramentas em sua estrutura para a apresentação do dashboard final, disposta na seguinte forma:</>

- Imports do projeto
- Variáveis de dados
- Seleção do modelo do yolov5
- Carregamento do vídeo a ser monitorado
- Modelo para estimar o risco de contaminação
- Detecção de vídeo
- Stream da detecção de vídeo
- Gráficos do dashboard
- Site em servidor local
- Servidor local

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Alguns métodos já testados e abandonados:
- k-means subtractor
- KNN subtractor
- haar cascade
- MOG
- Meanshift
- Camshift
- HOG
- YoloV3 - tiny
- YoloV3 - 320
