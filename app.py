import dash
from dash import dcc, html, Input, Output, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from collections import Counter
import hashlib
import os

# Initialize the Dash app
app = dash.Dash(__name__)

# CRITICAL: This exposes the Flask server for Gunicorn
server = app.server

# Function to obscure PII data
def obscure_pii(value, salt="790-analysis"):
    """
    Create a consistent hash of PII data to maintain relationships 
    while protecting privacy
    """
    if pd.isna(value):
        return "Unknown"
    # Create a hash that's consistent but irreversible
    hash_input = f"{value}{salt}".encode()
    hash_output = hashlib.sha256(hash_input).hexdigest()
    # Return first 8 characters for readability
    return f"ID_{hash_output[:8]}"

# Load and prepare the data
def load_and_prepare_data(filepath):
    """
    Load the CSV data and prepare it for analysis.
    Obscures defendant names and DOB for privacy.
    Returns the processed dataframe.
    """
    # Load the data
    df = pd.read_csv(filepath)
    
    # Obscure PII data - create consistent IDs that maintain relationships
    df['Defendant_Obscured'] = df['Defendant'].apply(lambda x: obscure_pii(x, "defendant"))
    df['DOB_Obscured'] = df['DOB'].apply(lambda x: obscure_pii(x, "dob"))
    
    # Create a unique defendant identifier using obscured data
    df['Defendant_ID'] = df['Defendant_Obscured'] + '_' + df['DOB_Obscured']
    
    # Remove original PII columns to ensure they're not accidentally used
    df = df.drop(columns=['Defendant', 'DOB'], errors='ignore')
    
    # Convert date columns to datetime for proper handling
    date_columns = ['OffenseDate', 'ArrestDate', 'FileDate', 'DispositionDate']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Clean statute values - remove extra spaces and standardize
    df['Statute'] = df['Statute'].str.strip()
    
    # Ensure Statute_Description exists and handle missing values
    if 'Statute_Description' not in df.columns:
        df['Statute_Description'] = 'No Description Available'
    else:
        df['Statute_Description'] = df['Statute_Description'].fillna('No Description Available')
    
    # Ensure ChargeOffenseDescription exists and handle missing values
    if 'ChargeOffenseDescription' not in df.columns:
        df['ChargeOffenseDescription'] = 'No Charge Description Available'
    else:
        df['ChargeOffenseDescription'] = df['ChargeOffenseDescription'].fillna('No Charge Description Available')
    
    # Clean and standardize agency names
    if 'Lead_Agency' in df.columns:
        df['Lead_Agency'] = df['Lead_Agency'].str.strip()
        df['Lead_Agency'] = df['Lead_Agency'].fillna('Unknown Agency')
    else:
        df['Lead_Agency'] = 'Unknown Agency'
    
    # Identify 790 charges
    df['Is_790_Charge'] = df['Statute'].str.startswith('790', na=False)
    
    return df

def get_related_charges(df, selected_statute, selected_agency=None):
    """
    For a given 790 statute, find all related charges for defendants
    who have that statute. Optionally filter by agency.
    """
    # Start with base dataframe
    filtered_df = df.copy()
    
    # Apply agency filter if selected
    if selected_agency and selected_agency != 'All Agencies':
        filtered_df = filtered_df[filtered_df['Lead_Agency'] == selected_agency]
    
    # Get all defendants who have the selected statute
    defendants_with_statute = filtered_df[filtered_df['Statute'] == selected_statute]['Defendant_ID'].unique()
    
    # Get all charges for these defendants
    related_charges_df = filtered_df[filtered_df['Defendant_ID'].isin(defendants_with_statute)]
    
    # Filter out the primary statute to focus on secondary offenses
    secondary_charges = related_charges_df[related_charges_df['Statute'] != selected_statute]
    
    # Count occurrences of each secondary charge
    charge_counts = secondary_charges['Statute'].value_counts()
    
    # Get the descriptions for better labeling - using Statute_Description for hover
    charge_descriptions = secondary_charges.groupby('Statute')['Statute_Description'].first()
    
    # Get ChargeOffenseDescription for the table
    charge_offense_descriptions = secondary_charges.groupby('Statute')['ChargeOffenseDescription'].first()
    
    # Create a dataframe for plotting
    plot_df = pd.DataFrame({
        'Statute': charge_counts.index,
        'Count': charge_counts.values,
        'Description': [charge_descriptions.get(s, 'No Description Available') for s in charge_counts.index],
        'ChargeOffenseDescription': [charge_offense_descriptions.get(s, 'No Charge Description Available') for s in charge_counts.index]
    })
    
    # Add percentage of defendants with this charge
    total_defendants = len(defendants_with_statute)
    plot_df['Percentage'] = (plot_df['Count'] / total_defendants * 100).round(1)
    
    # Create hover text with proper descriptions
    plot_df['Hover_Text'] = (
        '<b>' + plot_df['Statute'] + '</b><br>' +
        '<i>' + plot_df['Description'] + '</i><br>' +
        'Count: ' + plot_df['Count'].astype(str) + '<br>' +
        'Percentage of Defendants: ' + plot_df['Percentage'].astype(str) + '%'
    )
    
    return plot_df, total_defendants

# Load the data - checking multiple possible locations
try:
    # Try loading from the same directory first
    if os.path.exists('790.07.csv'):
        df = load_and_prepare_data('790.07.csv')
    elif os.path.exists('./790.07.csv'):
        df = load_and_prepare_data('./790.07.csv')
    elif os.path.exists('../790.07.csv'):
        df = load_and_prepare_data('../790.07.csv')
    else:
        # If file not found, create sample data for demonstration
        print("Warning: CSV file not found. Using sample data for demonstration.")
        np.random.seed(42)
        sample_data = {
            'Defendant': ['John Doe'] * 5 + ['Jane Smith'] * 4 + ['Bob Johnson'] * 6,
            'DOB': ['1990-01-01'] * 5 + ['1985-05-15'] * 4 + ['1975-12-20'] * 6,
            'Statute': ['790.07', '843.02', '893.13', '812.014', '784.03'] + 
                       ['790.07(1)', '810.02', '893.13', '322.34'] + 
                       ['790.07(2)', '812.014', '893.13', '843.02', '784.045', '322.34'],
            'Statute_Description': [
                'Weapons; Possession during commission of felony', 
                'Obstruction; Resisting officer without violence', 
                'Drug Abuse Prevention; Possession of controlled substance',
                'Theft; Petit theft 2nd degree', 
                'Battery; Aggravated battery'
            ] * 3,
            'ChargeOffenseDescription': [
                'WEAPONS-POSSESSION-COMMISSION OF FELONY', 
                'RESIST OFFICER-OBSTRUCT WO VIOLENCE', 
                'COCAINE-POSSESS-POSSESS COCAINE',
                'PETIT THEFT $100 LESS THAN $750', 
                'AGGRAVATED BATTERY-DEADLY WEAPON'
            ] * 3,
            'Lead_Agency': ['SHERIFF OFFICE'] * 5 + ['POLICE DEPT'] * 4 + ['STATE PATROL'] * 6,
            'OffenseDate': pd.date_range('2024-01-01', periods=15, freq='D').tolist()
        }
        df = pd.DataFrame(sample_data)
        # Apply PII obscuration to sample data
        df['Defendant_Obscured'] = df['Defendant'].apply(lambda x: obscure_pii(x, "defendant"))
        df['DOB_Obscured'] = df['DOB'].apply(lambda x: obscure_pii(x, "dob"))
        df['Defendant_ID'] = df['Defendant_Obscured'] + '_' + df['DOB_Obscured']
        df = df.drop(columns=['Defendant', 'DOB'])
        df['Is_790_Charge'] = df['Statute'].str.startswith('790')
except Exception as e:
    print(f"Error loading data: {e}")
    # Create minimal sample data if all else fails
    df = pd.DataFrame({
        'Defendant_ID': ['ID_001', 'ID_002'],
        'Statute': ['790.07', '843.02'],
        'Statute_Description': ['Weapons charge', 'Resisting arrest'],
        'ChargeOffenseDescription': ['WEAPONS CHARGE', 'RESISTING ARREST'],
        'Lead_Agency': ['SHERIFF OFFICE', 'POLICE DEPT'],
        'Is_790_Charge': [True, False]
    })

# Get unique 790 statutes for dropdown
statute_790_list = sorted(df[df['Is_790_Charge']]['Statute'].unique())

# Get unique agencies for dropdown
agency_list = ['All Agencies'] + sorted(df['Lead_Agency'].unique())

# Define the app layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("790 Charges Analysis Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.H3("Analyzing Related Offenses for Firearm Possession Charges", 
                style={'textAlign': 'center', 'color': '#7f8c8d', 'marginTop': '0px'}),
        html.P("Note: All defendant information has been anonymized for privacy protection", 
               style={'textAlign': 'center', 'color': '#95a5a6', 'fontSize': '12px', 'fontStyle': 'italic'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '30px'}),
    
    # Control Panel
    html.Div([
        # Dropdowns in a row
        html.Div([
            # Statute dropdown
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
            ], style={'flex': '1', 'marginRight': '20px'}),
            
            # Agency dropdown
            html.Div([
                html.Label("Filter by Agency:", 
                          style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
                dcc.Dropdown(
                    id='agency-dropdown',
                    options=[{'label': agency, 'value': agency} for agency in agency_list],
                    value='All Agencies',
                    placeholder="Select an agency...",
                    style={'width': '300px'}
                )
            ], style={'flex': '1'})
        ], style={'display': 'flex', 'marginBottom': '20px'}),
        
        # Summary statistics
        html.Div(id='summary-stats', style={'marginBottom': '20px'})
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 
              'marginBottom': '30px', 'marginLeft': '20px', 'marginRight': '20px'}),
    
    # Main chart
    html.Div([
        dcc.Graph(id='related-charges-bar-chart', style={'height': '600px'})
    ], style={'padding': '20px'}),
    
    # Secondary charges table section
    html.Div([
        html.H4("Secondary Charges Detail Table", 
                style={'color': '#2c3e50', 'marginBottom': '15px'}),
        html.Div(id='charges-table-container')
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px',
              'margin': '20px'}),
    
    # Additional insights
    html.Div([
        html.H4("Key Insights", style={'color': '#2c3e50', 'marginBottom': '15px'}),
        html.Div(id='insights-text')
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px',
              'margin': '20px'}),
    
    # Footer with data info
    html.Div([
        html.P(f"Data contains {len(df)} total charges from {len(df['Defendant_ID'].unique())} unique defendants", 
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '12px'})
    ], style={'padding': '10px', 'marginTop': '20px'})
])

# Callback for updating the chart, table, and insights
@app.callback(
    [Output('related-charges-bar-chart', 'figure'),
     Output('charges-table-container', 'children'),
     Output('summary-stats', 'children'),
     Output('insights-text', 'children')],
    [Input('statute-dropdown', 'value'),
     Input('agency-dropdown', 'value')]
)
def update_dashboard(selected_statute, selected_agency):
    if not selected_statute:
        # Return empty chart if no statute selected
        fig = go.Figure()
        fig.update_layout(
            title="Please select a statute from the dropdown",
            plot_bgcolor='white'
        )
        return fig, html.P("No data to display"), "No statute selected", "Select a statute to see insights"
    
    # Get related charges data
    plot_df, total_defendants = get_related_charges(df, selected_statute, selected_agency)
    
    if len(plot_df) == 0:
        fig = go.Figure()
        fig.update_layout(
            title=f"No secondary charges found for {selected_statute}",
            plot_bgcolor='white'
        )
        return fig, html.P("No secondary charges to display"), f"Total defendants with {selected_statute}: {total_defendants}", "No secondary charges to analyze"
    
    # Limit bar chart to top 20 for readability
    plot_df_chart = plot_df.head(20)
    
    # Create the bar chart
    fig = go.Figure()
    
    # Add bars with custom colors based on percentage
    colors = ['#e74c3c' if pct > 50 else '#f39c12' if pct > 25 else '#3498db' 
              for pct in plot_df_chart['Percentage']]
    
    fig.add_trace(go.Bar(
        x=plot_df_chart['Statute'],
        y=plot_df_chart['Count'],
        text=plot_df_chart['Percentage'].astype(str) + '%',
        textposition='outside',
        hovertext=plot_df_chart['Hover_Text'],
        hoverinfo='text',
        marker_color=colors,
        name='Co-occurring Charges'
    ))
    
    # Update layout
    agency_filter_text = f" (Agency: {selected_agency})" if selected_agency != 'All Agencies' else ""
    fig.update_layout(
        title={
            'text': f"Secondary Offenses for Defendants with {selected_statute} Charges{agency_filter_text}<br>" +
                   f"<span style='font-size: 14px; color: gray;'>Analyzing {total_defendants} anonymized defendants</span>",
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
        ),
        height=600
    )
    
    # Create the table for all secondary charges
    # Prepare table data - show all charges, not just top 20
    table_df = plot_df.copy()
    
    # Truncate long descriptions for better display
    table_df['ChargeOffenseDescription_Display'] = table_df['ChargeOffenseDescription'].apply(
        lambda x: x[:100] + '...' if len(str(x)) > 100 else x
    )
    
    # Create the table
    charges_table = dash_table.DataTable(
        id='charges-detail-table',
        columns=[
            {'name': 'Statute', 'id': 'Statute'},
            {'name': 'Charge Description', 'id': 'ChargeOffenseDescription_Display'},
            {'name': 'Count', 'id': 'Count', 'type': 'numeric'},
            {'name': '%', 'id': 'Percentage', 'type': 'numeric', 'format': {'specifier': '.1f'}}
        ],
        data=table_df.to_dict('records'),
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'whiteSpace': 'normal',
            'height': 'auto',
            'minWidth': '50px',
            'maxWidth': '500px'
        },
        style_cell_conditional=[
            {
                'if': {'column_id': 'Statute'},
                'width': '15%'
            },
            {
                'if': {'column_id': 'ChargeOffenseDescription_Display'},
                'width': '60%'
            },
            {
                'if': {'column_id': 'Count'},
                'width': '12.5%',
                'textAlign': 'center'
            },
            {
                'if': {'column_id': 'Percentage'},
                'width': '12.5%',
                'textAlign': 'center'
            }
        ],
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            },
            {
                'if': {
                    'filter_query': '{Percentage} > 50',
                    'column_id': 'Percentage'
                },
                'backgroundColor': '#ffcccc',
                'fontWeight': 'bold'
            },
            {
                'if': {
                    'filter_query': '{Percentage} > 25 && {Percentage} <= 50',
                    'column_id': 'Percentage'
                },
                'backgroundColor': '#ffe6cc'
            }
        ],
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        page_size=15,
        style_table={'height': '450px', 'overflowY': 'auto'}
    )
    
    # Create summary statistics
    summary_html = html.Div([
        html.Div([
            html.Span(f"Total Defendants with {selected_statute}: ", 
                     style={'fontWeight': 'bold'}),
            html.Span(f"{total_defendants}")
        ], style={'marginBottom': '5px'}),
        html.Div([
            html.Span("Total Unique Secondary Charges: ", 
                     style={'fontWeight': 'bold'}),
            html.Span(f"{len(plot_df)}")
        ], style={'marginBottom': '5px'}),
        html.Div([
            html.Span("Agency Filter: ", 
                     style={'fontWeight': 'bold'}),
            html.Span(selected_agency)
        ], style={'marginBottom': '5px'}),
        html.Div([
            html.Span("Most Common Secondary Charge: ", 
                     style={'fontWeight': 'bold'}),
            html.Span(f"{plot_df.iloc[0]['Statute']} - {plot_df.iloc[0]['ChargeOffenseDescription'][:50]}... ({plot_df.iloc[0]['Percentage']}%)")
            if len(plot_df) > 0 else html.Span("None")
        ])
    ])
    
    # Generate insights
    insights = []
    
    # High correlation charges (>50% of defendants)
    high_correlation = plot_df[plot_df['Percentage'] > 50]
    if len(high_correlation) > 0:
        insights.append(html.Li([
            html.Strong("High Correlation Pattern: "),
            f"The following charges appear in over 50% of {selected_statute} cases: ",
            html.Ul([html.Li(f"{row['Statute']} - {row['ChargeOffenseDescription'][:50]}... ({row['Percentage']}%)") 
                    for _, row in high_correlation.iterrows()])
        ]))
    
    # Drug-related charges
    drug_charges = plot_df[plot_df['Statute'].str.startswith('893', na=False)]
    if len(drug_charges) > 0:
        total_drug_cases = drug_charges['Count'].sum()
        drug_percentage = round(total_drug_cases / total_defendants * 100, 1)
        insights.append(html.Li([
            html.Strong("Drug-Firearm Nexus: "),
            f"Drug charges (Chapter 893) appear in {drug_percentage}% of {selected_statute} cases, ",
            f"including: {', '.join(drug_charges['Statute'].head(3).tolist())}"
        ]))
    
    # Violence-related charges
    violence_prefixes = ['784', '782', '787']  # Battery, murder, kidnapping
    violence_charges = plot_df[plot_df['Statute'].str[:3].isin(violence_prefixes)]
    if len(violence_charges) > 0:
        insights.append(html.Li([
            html.Strong("Violence Co-occurrence: "),
            f"Violence-related charges found: ",
            html.Ul([html.Li(f"{row['Statute']} - {row['ChargeOffenseDescription'][:50]}...") 
                    for _, row in violence_charges.iterrows()])
        ]))
    
    # Property crimes
    property_prefixes = ['810', '812']  # Burglary, theft
    property_charges = plot_df[plot_df['Statute'].str[:3].isin(property_prefixes)]
    if len(property_charges) > 0:
        property_percentage = round(property_charges['Count'].sum() / total_defendants * 100, 1)
        insights.append(html.Li([
            html.Strong("Property Crime Connection: "),
            f"Property crimes appear in {property_percentage}% of cases, including: ",
            ', '.join([f"{row['Statute']} ({row['ChargeOffenseDescription'][:30]}...)" 
                      for _, row in property_charges.head(3).iterrows()])
        ]))
    
    # Traffic-related charges
    traffic_charges = plot_df[plot_df['Statute'].str.startswith('322', na=False)]
    if len(traffic_charges) > 0:
        insights.append(html.Li([
            html.Strong("Traffic Violations: "),
            f"Traffic-related charges (Chapter 322) co-occur in {traffic_charges.iloc[0]['Percentage']}% of cases"
        ]))
    
    # Red flag analysis
    if len(high_correlation) > 2:
        insights.append(html.Li([
            html.Strong("⚠️ Potential Red Flag: "),
            html.Span(f"Multiple charges appear in >50% of {selected_statute} cases, suggesting potential charge stacking patterns", 
                     style={'color': '#e74c3c'})
        ]))
    
    # Agency-specific insights
    if selected_agency != 'All Agencies':
        agency_charges = df[df['Lead_Agency'] == selected_agency]
        agency_790_rate = round(len(agency_charges[agency_charges['Is_790_Charge']]) / len(agency_charges) * 100, 1)
        insights.append(html.Li([
            html.Strong("Agency Pattern: "),
            f"{selected_agency} has {agency_790_rate}% of their charges as 790 statutes"
        ]))
    
    insights_div = html.Ul(insights) if insights else html.P("Insufficient data for detailed pattern analysis")
    
    return fig, charges_table, summary_html, insights_div

# Run the app
if __name__ == '__main__':
    # For local development
    app.run_server(debug=True)
    
# For production deployment with Gunicorn
# The 'server' variable is what Gunicorn needs
