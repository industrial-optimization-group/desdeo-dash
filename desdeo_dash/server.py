import dash

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
assets_folder = "./assets"

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, assets_folder=assets_folder)
app.title = "Multiobjective optimization with NAUTILUS"
app.config.suppress_callback_exceptions = True
server = app.server
