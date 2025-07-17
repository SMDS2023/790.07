import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from collections import Counter

# Initialize the Dash app
app = dash.Dash(__name__)

# Load and prepare the data
def load_and_prepare_data(filepath):
    """
    Load the CSV data and prepare it for analysis.
    Returns the processed dataframe.
    """
    # Load the data
    df = pd.read_csv(filepath)
    
    # Convert date columns to datetime for proper handling
    date_columns = ['OffenseDate', 'ArrestDate', 'FileDate', 'DispositionDate']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Clean statute values - remove extra spaces and standardize
    df['Statute'] = df['Statute'].str.strip()
    
    # Identify 790 charges
    df['Is_790_Charge'] = df['Statute'].str.startswith('790', na=False)
    
    # Create a defendant identifier (combining name and DOB for uniqueness)
    df['Defendant_ID'] = df['Defendant'] + '_' + df['DOB'].astype(str)
    
    return df

def get_related_charges(df, selected_statute):
    """
    For a given 790 statute, find all related charges for defendants
    who have that statute.
    """
    # Get all defendants who have the selected statute
    defendants_with_statute = df[df['Statute'] == selected_statute]['Defendant_ID'].unique()
    
    # Get all charges for these defendants
    related_charges_df = df[df['Defendant_ID'].isin(defendants_with_statute)]
    
    # Filter out the primary statute to focus on secondary offenses
    secondary_charges = related_charges_df[related_charges_df['Statute'] != selected_statute]
    
    # Count occurrences of each secondary charge
    charge_counts = secondary_charges['Statute'].value_counts()
    
    # Also get the descriptions for better labeling
    charge_descriptions = secondary_charges.groupby('Statute')['Statute_Description'].first()
    
    # Create a dataframe for plotting
    plot_df = pd.DataFrame({
        'Statute': charge_counts.index,
        'Count': charge_counts.values,
        'Description': [charge_descriptions.get(s, 'No Description') for s in charge_counts.index]
    })
    
    # Add percentage of defendants with this charge
    total_defendants = len(defendants_with_statute)
    plot_df['Percentage'] = (plot_df['Count'] / total_defendants * 100).round(1)
    
    # Create hover text
    plot_df['Hover_Text'] = (
        plot_df['Statute'] + '<br>' +
        plot_df['Description'] + '<br>' +
        'Count: ' + plot_df['Count'].astype(str) + '<br>' +
        'Percentage of Defendants: ' + plot_df['Percentage'].astype(str) + '%'
    )
    
    return plot_df, total_defendants

# Load the data (update the filepath as needed)
# df = load_and_prepare_data('your_file_path.csv')

# For demonstration, create sample data structure
# In production, replace this with actual data loading
np.random.seed(42)
sample_data = {
    'Defendant': ['John Doe'] * 5 + ['Jane Smith'] * 4 + ['Bob Johnson'] * 6,
    'DOB': ['1990-01-01'] * 5 + ['1985-05-15'] * 4 + ['1975-12-20'] * 6,
    'Statute': ['790.07', '843.02', '893.13', '812.014', '784.03'] + 
               ['790.07(1)', '810.02', '893.13', '322.34'] + 
               ['790.07(2)', '812.014', '893.13', '843.02', '784.045', '322.34'],
    'Statute_Description': [
        'Possession of weapon during felony', 'Resisting arrest', 'Drug possession', 
        'Theft', 'Aggravated battery'
    ] * 3,
    'OffenseDate': pd.date_range('2024-01-01', periods=15, freq='D').tolist()
}
df = pd.DataFrame(sample_data)
df['Defendant_ID'] = df['Defendant'] + '_' + df['DOB'].astype(str)
df['Is_790_Charge'] = df['Statute'].str.startswith('790')

# Get unique 790 statutes for dropdown
statute_790_list = sorted(df[df['Is_790_Charge']]['Statute'].unique())

# Define the app layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("790 Charges Analysis Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.H3("Exploring Related Offenses for Firearm Possession Charges", 
                style={'textAlign': 'center', 'color': '#7f8c8d', 'marginTop': '0px'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '30px'}),
    
    # Control Panel
    html.Div([
        html.Div([
            html.Label("Select 790 Statute:", 
                      style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
            dcc.Dropdown(
                id='statute-dropdown',
                options=[{'label': statute, 'value': statute} for statute in statute_790_list],
                value=statute_790_list[0] if statute_790_list else None,
                placeholder="Select a 790 statute...",
                style={'width': '300px'}
            )
        ], style={'marginBottom': '20px'}),
        
        # Summary statistics
        html.Div(id='summary-stats', style={'marginBottom': '20px'})
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 
              'marginBottom': '30px', 'marginLeft': '20px', 'marginRight': '20px'}),
    
    # Main chart
    html.Div([
        dcc.Graph(id='related-charges-bar-chart', style={'height': '600px'})
    ], style={'padding': '20px'}),
    
    # Additional insights
    html.Div([
        html.H4("Key Insights", style={'color': '#2c3e50', 'marginBottom': '15px'}),
        html.Div(id='insights-text')
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px',
              'margin': '20px'})
])

# Callback for updating the chart and insights
@app.callback(
    [Output('related-charges-bar-chart', 'figure'),
     Output('summary-stats', 'children'),
     Output('insights-text', 'children')],
    [Input('statute-dropdown', 'value')]
)
def update_chart(selected_statute):
    if not selected_statute:
        # Return empty chart if no statute selected
        fig = go.Figure()
        fig.update_layout(title="Please select a statute from the dropdown")
        return fig, "No statute selected", "Select a statute to see insights"
    
    # Get related charges data
    plot_df, total_defendants = get_related_charges(df, selected_statute)
    
    # Limit to top 20 for readability
    plot_df = plot_df.head(20)
    
    # Create the bar chart
    fig = go.Figure()
    
    # Add bars with custom colors based on percentage
    colors = ['#e74c3c' if pct > 50 else '#f39c12' if pct > 25 else '#3498db' 
              for pct in plot_df['Percentage']]
    
    fig.add_trace(go.Bar(
        x=plot_df['Statute'],
        y=plot_df['Count'],
        text=plot_df['Percentage'].astype(str) + '%',
        textposition='outside',
        hovertext=plot_df['Hover_Text'],
        hoverinfo='text',
        marker_color=colors,
        name='Co-occurring Charges'
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"Secondary Offenses for Defendants with {selected_statute} Charges<br>" +
                   f"<span style='font-size: 14px; color: gray;'>Total Defendants: {total_defendants}</span>",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Related Statute",
        yaxis_title="Number of Occurrences",
        showlegend=False,
        hovermode='x unified',
        plot_bgcolor='white',
        xaxis=dict(
            tickangle=-45,
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        yaxis=dict(
            gridcolor='lightgray',
            gridwidth=0.5
        )
    )
    
    # Create summary statistics
    summary_html = html.Div([
        html.Div([
            html.Span(f"Total Defendants with {selected_statute}: ", 
                     style={'fontWeight': 'bold'}),
            html.Span(f"{total_defendants}")
        ]),
        html.Div([
            html.Span("Total Unique Secondary Charges: ", 
                     style={'fontWeight': 'bold'}),
            html.Span(f"{len(plot_df)}")
        ]),
        html.Div([
            html.Span("Most Common Secondary Charge: ", 
                     style={'fontWeight': 'bold'}),
            html.Span(f"{plot_df.iloc[0]['Statute']} ({plot_df.iloc[0]['Percentage']}% of defendants)")
            if len(plot_df) > 0 else html.Span("None")
        ])
    ])
    
    # Generate insights
    insights = []
    
    # High correlation charges (>50% of defendants)
    high_correlation = plot_df[plot_df['Percentage'] > 50]
    if len(high_correlation) > 0:
        insights.append(html.Li([
            html.Strong("High Correlation: "),
            f"The following charges appear in over 50% of {selected_statute} cases: ",
            ', '.join(high_correlation['Statute'].tolist())
        ]))
    
    # Drug-related charges
    drug_charges = plot_df[plot_df['Statute'].str.startswith('893', na=False)]
    if len(drug_charges) > 0:
        total_drug_percentage = drug_charges['Percentage'].iloc[0] if len(drug_charges) > 0 else 0
        insights.append(html.Li([
            html.Strong("Drug Connection: "),
            f"Drug charges (893.xx) appear in {total_drug_percentage}% of {selected_statute} cases"
        ]))
    
    # Violence-related charges
    violence_prefixes = ['784', '782', '787']  # Battery, murder, kidnapping
    violence_charges = plot_df[plot_df['Statute'].str[:3].isin(violence_prefixes)]
    if len(violence_charges) > 0:
        insights.append(html.Li([
            html.Strong("Violence Pattern: "),
            f"Violence-related charges found in these cases include: ",
            ', '.join(violence_charges['Statute'].tolist())
        ]))
    
    # Property crimes
    property_prefixes = ['810', '812']  # Burglary, theft
    property_charges = plot_df[plot_df['Statute'].str[:3].isin(property_prefixes)]
    if len(property_charges) > 0:
        insights.append(html.Li([
            html.Strong("Property Crimes: "),
            f"Property-related charges found: ",
            ', '.join(property_charges['Statute'].tolist())
        ]))
    
    insights_div = html.Ul(insights) if insights else html.P("No significant patterns detected")
    
    return fig, summary_html, insights_div

# Run the app
if __name__ == '__main__':
    # When running locally, update the filepath here:
    # df = load_and_prepare_data('path/to/your/790.07 CSV.csv')
    app.run_server(debug=True)
