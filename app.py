# Minimal test app to verify Gunicorn deployment
import dash
from dash import html

# Create the app
app = dash.Dash(__name__)

# CRITICAL: Expose the server for Gunicorn
server = app.server

# Simple layout
app.layout = html.Div([
    html.H1("Test Dashboard - Server is Running!"),
    html.P("If you can see this, the server variable is properly exposed.")
])

# This allows running locally with python test_app.py
if __name__ == '__main__':
    app.run_server(debug=True)

# The server variable should be accessible to Gunicorn
print("Server created successfully")
