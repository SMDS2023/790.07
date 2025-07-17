import dash
from dash import dcc, html, Input, Output, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    
    # Clean demographic data
    if 'Race_Tier_1' in df.columns:
        df['Race_Tier_1'] = df['Race_Tier_1'].str.upper().str.strip()
        # Map race codes to full names for display
        race_mapping = {'B': 'Black', 'W': 'White', 'O': 'Other', 'H': 'Hispanic', 'A': 'Asian'}
        df['Race_Display'] = df['Race_Tier_1'].map(race_mapping).fillna('Unknown')
    else:
        df['Race_Display'] = 'Unknown'
    
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].str.upper().str.strip()
        gender_mapping = {'M': 'Male', 'F': 'Female'}
        df['Gender_Display'] = df['Gender'].map(gender_mapping).fillna('Unknown')
    else:
        df['Gender_Display'] = 'Unknown'
    
    # Create age groups
    if 'Age_At_Offense' in df.columns:
        df['Age_Group'] = pd.cut(df['Age_At_Offense'], 
                                 bins=[0, 18, 25, 35, 45, 55, 100],
                                 labels=['Under 18', '18-25', '26-35', '36-45', '46-55', '56+'],
                                 right=False)
        # Convert to string to handle missing values
        df['Age_Group'] = df['Age_Group'].astype(str)
        df['Age_Group'] = df['Age_Group'].replace('nan', 'Unknown')
    else:
        df['Age_Group'] = 'Unknown'
    
    # Identify 790 charges
    df['Is_790_Charge'] = df['Statute'].str.startswith('790', na=False)
    
    return df

def get_related_charges(df, selected_statute, selected_agency=None):
    """
    For a given 790 statute, find all related charges for defendants
    who have that statute. Counts unique defendants per secondary statute.
    """
    # Start with base dataframe
    filtered_df = df.copy()
    
    # Apply agency filter if selected
    if selected_agency and selected_agency != 'All Agencies':
        filtered_df = filtered_df[filtered_df['Lead_Agency'] == selected_agency]
    
    # Handle combined 790.07 analysis
    if selected_statute == 'All 790.07':
        statute_filter = filtered_df['Statute'].str.match(r'^790\.07(\(|$)')
        primary_statute_display = "All 790.07"
    else:
        statute_filter = filtered_df['Statute'] == selected_statute
        primary_statute_display = selected_statute
    
    # Get all defendants who have the selected statute(s)
    defendants_with_statute = filtered_df[statute_filter]['Defendant_ID'].unique()
    total_defendants = len(defendants_with_statute)
    
    # Get all charges for these defendants
    related_charges_df = filtered_df[filtered_df['Defendant_ID'].isin(defendants_with_statute)]
    
    # Filter out the primary statute(s) to focus on secondary offenses
    if selected_statute == 'All 790.07':
        secondary_charges = related_charges_df[~related_charges_df['Statute'].str.match(r'^790\.07(\(|$)')]
    else:
        secondary_charges = related_charges_df[related_charges_df['Statute'] != selected_statute]
    
    # For bar chart: Count unique defendants per statute
    unique_defendants_per_statute = (
        secondary_charges.groupby('Statute')['Defendant_ID']
        .nunique()
        .reset_index()
        .rename(columns={'Defendant_ID': 'Unique_Defendants'})
    )
    
    # Get statute descriptions for the bar chart
    statute_descriptions = secondary_charges.groupby('Statute')['Statute_Description'].first()
    
    # Create bar chart dataframe
    plot_df = unique_defendants_per_statute.copy()
    plot_df['Description'] = plot_df['Statute'].map(statute_descriptions).fillna('No Description Available')
    plot_df = plot_df.sort_values('Unique_Defendants', ascending=False)
    
    # Create hover text for bar chart
    plot_df['Hover_Text'] = (
        '<b>' + plot_df['Statute'] + '</b><br>' +
        '<i>' + plot_df['Description'] + '</i><br>' +
        'Unique Defendants: ' + plot_df['Unique_Defendants'].astype(str)
    )
    
    # For table: Count occurrences of statute + charge description combinations
    combination_counts = (
        secondary_charges.groupby(['Statute', 'ChargeOffenseDescription'])
        .size()
        .reset_index(name='Occurrence_Count')
    )
    
    # Also get unique defendant count per combination
    combination_defendants = (
        secondary_charges.groupby(['Statute', 'ChargeOffenseDescription'])['Defendant_ID']
        .nunique()
        .reset_index(name='Unique_Defendants')
    )
    
    # Merge the counts
    table_df = combination_counts.merge(
        combination_defendants, 
        on=['Statute', 'ChargeOffenseDescription'],
        how='left'
    )
    table_df = table_df.sort_values(['Statute', 'Occurrence_Count'], ascending=[True, False])
    
    # Get demographic data for primary charges only
    primary_charges_df = filtered_df[statute_filter]
    demographics = {
        'race': primary_charges_df.groupby('Race_Display')['Defendant_ID'].nunique().to_dict(),
        'gender': primary_charges_df.groupby('Gender_Display')['Defendant_ID'].nunique().to_dict(),
        'age': primary_charges_df.groupby('Age_Group')['Defendant_ID'].nunique().to_dict()
    }
    
    return plot_df, table_df, total_defendants, primary_statute_display, demographics

# Load the data - checking multiple possible locations
try:
    # Try loading from the same directory first
    if os.path.exists('790.07.csv'):
        print("Loading data from 790.07.csv...")
        df = load_and_prepare_data('790.07.csv')
        print(f"Data loaded successfully: {len(df)} rows, {len(df.columns)} columns")
    elif os.path.exists('./790.07.csv'):
        print("Loading data from ./790.07.csv...")
        df = load_and_prepare_data('./790.07.csv')
        print(f"Data loaded successfully: {len(df)} rows, {len(df.columns)} columns")
    elif os.path.exists('../790.07.csv'):
        print("Loading data from ../790.07.csv...")
        df = load_and_prepare_data('../790.07.csv')
        print(f"Data loaded successfully: {len(df)} rows, {len(df.columns)} columns")
    else:
        # If file not found, create sample data for demonstration
        print("Warning: CSV file not found. Using sample data for demonstration.")
        np.random.seed(42)
        # Create more realistic sample data with demographics
        sample_records = []
        defendants = ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Davis']
        races = ['B', 'W', 'B', 'W', 'O']
        genders = ['M', 'F', 'M', 'F', 'M']
        ages = [23, 34, 19, 45, 28]
        
        for i, defendant in enumerate(defendants):
            dob = f"19{70+i}-01-01"
            # Primary charge
            sample_records.append({
                'Defendant': defendant,
                'DOB': dob,
                'Statute': '790.07(2)' if i < 3 else '790.07(1)',
                'Statute_Description': 'Weapons; Possession during commission of felony',
                'ChargeOffenseDescription': 'WEAPONS-POSSESSION-COMMISSION OF FELONY',
                'Lead_Agency': 'SHERIFF OFFICE' if i < 3 else 'POLICE DEPT',
                'OffenseDate': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i),
                'Race_Tier_1': races[i],
                'Gender': genders[i],
                'Age_At_Offense': ages[i]
            })
            # Add secondary charges
            if i < 3:  # First 3 defendants get drug charges
                sample_records.append({
                    'Defendant': defendant,
                    'DOB': dob,
                    'Statute': '893.13(6)(A)',
                    'Statute_Description': 'Drug Abuse Prevention; Possession of controlled substance',
                    'ChargeOffenseDescription': 'POSS. OF CANNABIS >20 GMS (WITH A WEAPON)',
                    'Lead_Agency': 'SHERIFF OFFICE',
                    'OffenseDate': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i),
                    'Race_Tier_1': races[i],
                    'Gender': genders[i],
                    'Age_At_Offense': ages[i]
                })
        
        df = pd.DataFrame(sample_records)
        # Apply preparation
        df = load_and_prepare_data(df)
        print(f"Sample data created: {len(df)} rows")
except Exception as e:
    print(f"Error loading data: {e}")
    import traceback
    traceback.print_exc()
    # Create minimal sample data if all else fails
    df = pd.DataFrame({
        'Defendant_ID': ['ID_001', 'ID_002'],
        'Statute': ['790.07', '843.02'],
        'Statute_Description': ['Weapons charge', 'Resisting arrest'],
        'ChargeOffenseDescription': ['WEAPONS CHARGE', 'RESISTING ARREST'],
        'Lead_Agency': ['SHERIFF OFFICE', 'POLICE DEPT'],
        'Is_790_Charge': [True, False],
        'Race_Display': ['Black', 'White'],
        'Gender_Display': ['Male', 'Female'],
        'Age_Group': ['18-25', '26-35']
    })
    print("Using minimal fallback data")

# Get unique 790 statutes for dropdown - add combined option
statute_790_list = ['All 790.07', '790.07(1)', '790.07(2)'] + sorted([s for s in df[df['Is_790_Charge']]['Statute'].unique() 
                                                                     if s not in ['790.07(1)', '790.07(2)']])

# Get unique agencies for dropdown
agency_list = ['All Agencies'] + sorted(df['Lead_Agency'].unique())

# Define the app layout
app.layout = html.Div([
    # Header with Logo
    html.Div([
        html.Div([
            # Logo placeholder - replace src with actual logo path when uploaded
            html.Img(src='/assets/logo.png', 
                    style={'height': '60px', 'marginRight': '20px', 'verticalAlign': 'middle'},
                    alt='Lotter Law Logo'),
            html.Div([
                html.H1("790 Charges Analysis Dashboard", 
                        style={'margin': '0', 'color': '#2c3e50'}),
                html.P("Call or Text: 407-500-7000", 
                      style={'margin': '0', 'color': '#7f8c8d', 'fontSize': '14px'})
            ], style={'display': 'inline-block', 'verticalAlign': 'middle'})
        ], style={'textAlign': 'center'}),
        html.H3("Analyzing Related Offenses for Firearm Possession Charges", 
                style={'textAlign': 'center', 'color': '#7f8c8d', 'marginTop': '10px'}),
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '30px'}),
    
    # Disclaimer Section
    html.Div([
        html.Details([
            html.Summary("Data Disclaimer & Methodology", 
                        style={'fontWeight': 'bold', 'cursor': 'pointer', 'color': '#2c3e50'}),
            html.Div([
                html.P([
                    html.Strong("Data Source: "),
                    "This dashboard analyzes data obtained from government sources. While generally accurate, "
                    "the data may contain errors or omissions inherent in the original records."
                ]),
                html.P([
                    html.Strong("Privacy Protection: "),
                    "All defendant names and dates of birth have been anonymized using cryptographic hashing "
                    "to protect individual privacy while maintaining analytical integrity."
                ]),
                html.P([
                    html.Strong("Methodology: "),
                    "This analysis examines defendants charged under Florida Statute 790.07 (possession of a weapon "
                    "during commission of a felony) and identifies patterns in co-occurring charges. Each defendant "
                    "is counted once per unique statute to avoid inflation of statistics."
                ]),
                html.P([
                    html.Strong("Purpose: "),
                    "This dashboard is designed to identify enforcement trends and potential disparities in the "
                    "application of firearm possession charges, supporting data-driven criminal justice reform efforts."
                ])
            ], style={'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
        ], style={'marginBottom': '20px'})
    ], style={'margin': '20px'}),
    
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
                    value='790.07(2)',  # Default to 790.07(2) as requested
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
    
    # Demographics Section
    html.Div([
        html.H4("Demographic Analysis", style={'color': '#2c3e50', 'marginBottom': '15px'}),
        dcc.Graph(id='demographics-charts', style={'height': '400px'})
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px',
              'margin': '20px'}),
    
    # Main chart
    html.Div([
        dcc.Graph(id='related-charges-bar-chart', style={'height': '600px'})
    ], style={'padding': '20px'}),
    
    # Secondary charges table section
    html.Div([
        html.H4("Secondary Charges Detail - Statute & Charge Description Combinations", 
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
    
    # Footer
    html.Div([
        html.Hr(),
        html.P([
            "© 2024 Lotter Law | ",
            html.A("407-500-7000", href="tel:407-500-7000"),
            " | Data contains ", f"{len(df)}", " total charges from ", 
            f"{len(df['Defendant_ID'].unique())}", " unique defendants"
        ], style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '12px'})
    ], style={'padding': '20px'})
])

# Callback for updating all components
@app.callback(
    [Output('related-charges-bar-chart', 'figure'),
     Output('charges-table-container', 'children'),
     Output('summary-stats', 'children'),
     Output('insights-text', 'children'),
     Output('demographics-charts', 'figure')],
    [Input('statute-dropdown', 'value'),
     Input('agency-dropdown', 'value')]
)
def update_dashboard(selected_statute, selected_agency):
    if not selected_statute:
        # Return empty chart if no statute selected
        fig = go.Figure()
        fig.update_layout(title="Please select a statute from the dropdown", plot_bgcolor='white')
        demo_fig = go.Figure()
        demo_fig.update_layout(title="Select a statute to see demographics", plot_bgcolor='white')
        return fig, html.P("No data to display"), "No statute selected", "Select a statute to see insights", demo_fig
    
    # Get related charges data
    plot_df, table_df, total_defendants, primary_statute_display, demographics = get_related_charges(
        df, selected_statute, selected_agency
    )
    
    # Create demographics charts
    demo_fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Race', 'Gender', 'Age Group'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Race chart
    race_data = pd.DataFrame(list(demographics['race'].items()), columns=['Race', 'Count'])
    race_data = race_data.sort_values('Count', ascending=False)
    demo_fig.add_trace(
        go.Bar(x=race_data['Race'], y=race_data['Count'], name='Race',
               marker_color='#3498db', showlegend=False),
        row=1, col=1
    )
    
    # Gender chart
    gender_data = pd.DataFrame(list(demographics['gender'].items()), columns=['Gender', 'Count'])
    gender_data = gender_data.sort_values('Count', ascending=False)
    demo_fig.add_trace(
        go.Bar(x=gender_data['Gender'], y=gender_data['Count'], name='Gender',
               marker_color='#e74c3c', showlegend=False),
        row=1, col=2
    )
    
    # Age chart
    age_data = pd.DataFrame(list(demographics['age'].items()), columns=['Age Group', 'Count'])
    age_order = ['Under 18', '18-25', '26-35', '36-45', '46-55', '56+', 'Unknown']
    age_data['Age Group'] = pd.Categorical(age_data['Age Group'], categories=age_order, ordered=True)
    age_data = age_data.sort_values('Age Group')
    demo_fig.add_trace(
        go.Bar(x=age_data['Age Group'], y=age_data['Count'], name='Age',
               marker_color='#2ecc71', showlegend=False),
        row=1, col=3
    )
    
    demo_fig.update_layout(
        height=400,
        title_text=f"Demographics of Defendants with {primary_statute_display} Charges",
        showlegend=False
    )
    demo_fig.update_xaxes(tickangle=-45)
    
    if len(plot_df) == 0:
        fig = go.Figure()
        fig.update_layout(
            title=f"No secondary charges found for {primary_statute_display}",
            plot_bgcolor='white'
        )
        return fig, html.P("No secondary charges to display"), f"Total defendants with {primary_statute_display}: {total_defendants}", "No secondary charges to analyze", demo_fig
    
    # Create the bar chart showing unique defendant counts
    fig = go.Figure()
    
    # Limit to top 20 for readability
    plot_df_chart = plot_df.head(20)
    
    # Create bars
    fig.add_trace(go.Bar(
        x=plot_df_chart['Statute'],
        y=plot_df_chart['Unique_Defendants'],
        text=plot_df_chart['Unique_Defendants'].astype(str),
        textposition='outside',
        hovertext=plot_df_chart['Hover_Text'],
        hoverinfo='text',
        marker_color='#3498db',
        name='Unique Defendants'
    ))
    
    # Update layout
    agency_filter_text = f" (Agency: {selected_agency})" if selected_agency != 'All Agencies' else ""
    fig.update_layout(
        title={
            'text': f"Secondary Offenses for Defendants with {primary_statute_display} Charges{agency_filter_text}<br>" +
                   f"<span style='font-size: 14px; color: gray;'>Total Defendants with {primary_statute_display}: {total_defendants}</span>",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Related Statute",
        yaxis_title="Number of Unique Defendants",
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
    
    # Create the table showing statute + charge description combinations
    table_df['ChargeOffenseDescription_Display'] = table_df['ChargeOffenseDescription'].apply(
        lambda x: x[:100] + '...' if len(str(x)) > 100 else x
    )
    
    # Create the table
    charges_table = dash_table.DataTable(
        id='charges-detail-table',
        columns=[
            {'name': 'Statute', 'id': 'Statute'},
            {'name': 'Charge Description', 'id': 'ChargeOffenseDescription_Display'},
            {'name': 'Total Count', 'id': 'Occurrence_Count', 'type': 'numeric'},
            {'name': 'Unique Defendants', 'id': 'Unique_Defendants', 'type': 'numeric'}
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
                'width': '55%'
            },
            {
                'if': {'column_id': 'Occurrence_Count'},
                'width': '15%',
                'textAlign': 'center'
            },
            {
                'if': {'column_id': 'Unique_Defendants'},
                'width': '15%',
                'textAlign': 'center'
            }
        ],
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ],
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        page_size=15,
        style_table={'height': '450px', 'overflowY': 'auto'},
        sort_action="native"
    )
    
    # Create summary statistics
    summary_html = html.Div([
        html.Div([
            html.Span(f"Total Unique Defendants with {primary_statute_display}: ", 
                     style={'fontWeight': 'bold'}),
            html.Span(f"{total_defendants}", style={'fontSize': '18px', 'color': '#e74c3c'})
        ], style={'marginBottom': '5px'}),
        html.Div([
            html.Span("Total Unique Secondary Statutes: ", 
                     style={'fontWeight': 'bold'}),
            html.Span(f"{len(plot_df)}")
        ], style={'marginBottom': '5px'}),
        html.Div([
            html.Span("Agency Filter: ", 
                     style={'fontWeight': 'bold'}),
            html.Span(selected_agency)
        ], style={'marginBottom': '5px'})
    ])
    
    # Generate insights including demographic analysis
    insights = []
    
    # Demographic disparities
    if demographics['race']:
        race_df = pd.DataFrame(list(demographics['race'].items()), columns=['Race', 'Count'])
        race_df['Percentage'] = (race_df['Count'] / race_df['Count'].sum() * 100).round(1)
        race_df = race_df.sort_values('Percentage', ascending=False)
        
        insights.append(html.Li([
            html.Strong("Racial Distribution: "),
            ', '.join([f"{row['Race']}: {row['Count']} ({row['Percentage']}%)" 
                      for _, row in race_df.head(3).iterrows()])
        ]))
        
        # Check for potential disparities
        if race_df.iloc[0]['Percentage'] > 60:
            insights.append(html.Li([
                html.Strong("⚠️ Potential Disparate Impact: "),
                html.Span(f"{race_df.iloc[0]['Race']} defendants represent {race_df.iloc[0]['Percentage']}% "
                         f"of {primary_statute_display} charges", style={'color': '#e74c3c'})
            ]))
    
    # Age patterns
    if demographics['age']:
        age_df = pd.DataFrame(list(demographics['age'].items()), columns=['Age', 'Count'])
        young_adult_count = age_df[age_df['Age'].isin(['18-25', '26-35'])]['Count'].sum()
        young_adult_pct = round(young_adult_count / total_defendants * 100, 1)
        if young_adult_pct > 60:
            insights.append(html.Li([
                html.Strong("Age Pattern: "),
                f"Young adults (18-35) account for {young_adult_pct}% of defendants"
            ]))
    
    # Top secondary charges (existing)
    if len(plot_df) >= 3:
        top_3 = plot_df.head(3)
        insights.append(html.Li([
            html.Strong("Top 3 Secondary Charges: "),
            html.Ol([
                html.Li(f"{row['Statute']} - {row['Unique_Defendants']} defendants") 
                for _, row in top_3.iterrows()
            ])
        ]))
    
    # Drug-related charges analysis (existing)
    drug_charges = plot_df[plot_df['Statute'].str.startswith('893', na=False)]
    if len(drug_charges) > 0:
        total_drug_defendants = drug_charges['Unique_Defendants'].sum()
        insights.append(html.Li([
            html.Strong("Drug-Firearm Nexus: "),
            f"{total_drug_defendants} defendants have drug charges"
        ]))
    
    insights_div = html.Ul(insights) if insights else html.P("Insufficient data for detailed pattern analysis")
    
    return fig, charges_table, summary_html, insights_div, demo_fig

# Run the app
if __name__ == '__main__':
    # For local development
    app.run_server(debug=True)
    
# For production deployment with Gunicorn
# The 'server' variable is what Gunicorn needs
