import plotly.express as px


def plot_line_plotly(data, dataset_name, score, config):
    title = f"""
    Dataset: {str(dataset_name)} <br>
    Score:  {str(score)} <br>
    Config: {str(config)}
    """
    fig = px.line(data[:, 1:], labels=[''], title=title)
    fig.show()
