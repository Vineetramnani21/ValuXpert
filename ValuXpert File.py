pip install Flask 
import os
from flask import Flask, render_template
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go
import base64
import io

# Initialize the Flask app
app = Flask(__name__)

# Create the 'uploads' folder if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize the Dash app
dash_app = Dash(
    __name__,
    server=app,
    url_base_pathname='/dashboard/',
    external_stylesheets=['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css']
)

# Define the layout for the Dash app
dash_app.layout = html.Div(
    style={'backgroundColor': '#121212', 'color': '#e0e0e0', 'padding': '20px'},
    children=[
        html.H1("ValuXpert Financial Analysis", style={"color": "#bb86fc", "textAlign": "center"}),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select CSV File')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px',
                'color': '#e0e0e0'
            },
            multiple=False
        ),
        html.Div(id='output-data-upload')
    ]
)

# Callback function to handle file upload and display a simple graph
@dash_app.callback(
    Output('output-data-upload', 'children'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    if contents is not None:
        # Decode the uploaded file contents
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            # Read the CSV data
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            
            # Generate a sample plot (assuming CSV has 'Date' and 'Value' columns)
            fig = go.Figure(
                data=[
                    go.Scatter(x=df['Date'], y=df['Value'], mode='lines+markers', name='Financial Data')
                ],
                layout=go.Layout(
                    title="Financial Data Visualization",
                    plot_bgcolor="#121212",
                    paper_bgcolor="#121212",
                    font={'color': '#e0e0e0'}
                )
            )

            return html.Div([
                html.H2(f'Analysis for {filename}', style={"color": "#bb86fc"}),
                dcc.Graph(figure=fig)
            ])
        except Exception as e:
            return html.Div([
                'There was an error processing this file.',
                str(e)
            ])

    return html.Div("Please upload a CSV file to view data.")

# Flask route to render the main layout
@app.route('/')
def index():
    return render_template('layout.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
