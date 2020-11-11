import matplotlib.pyplot as plt
import pandas as pd
import chart_studio.plotly as py

import py.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)


data = dict(type='choropetch', locations=[
            'AZ', 'CA', 'NY'], locationmode='USA-states', colorscale='Portland', text=['texto1', 'texto2', 'texto3'], z=[1.0, 2.0, 3.0], colorbar={'title': 'titulo barra de cores'})


layout = dict(geo={'scope': 'usa'})

choromap = go.figure(data=[data], layout=layout)
plot(choromap)
