import plotly.express as px


def plot_line_plotly(data):
    fig = px.line(data[:, 1:], labels=[''])
    fig.show()
