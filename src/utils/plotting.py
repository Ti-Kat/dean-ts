import plotly.express as px


def plot_line_plotly(data, config, dataset_name):
    fig = px.line(data[:, 1:], labels=[''], title=dataset_name + "<br>" + str(config))
    fig.show()
