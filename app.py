"""
Marijuana-Firearm 790.07 Analysis Dashboard
Analyzes the nexus between marijuana possession charges and firearm possession during felony charges
Designed for deployment to GitHub/Render
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import base64
import io

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "790.07 Enforcement Analysis"
server = app.server  # For deployment

# Color scheme for consistent styling
COLORS = {
    'primary': '#1f77b4',
    'marijuana': '#2ca02c',
    'firearm': '#d62728',
    'warning': '#ff7f0e',
    'dark': '#343a40',
    'light': '#f8f9fa',
    'background': '#f5f5f5'
}

def process_uploaded_data(contents, filename):
    """
    Process uploaded CSV file and add analysis columns
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        # Read CSV
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        # Convert date columns
        date_cols = ['FileDate', 'OffenseDate', 'ArrestDate']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Add analysis flags if not present
        if 'Is_790_07' not in df.columns:
            df['Is_790_07'] = df['Statute'].str.contains('790\.07', na=False, regex=True)
        
        if 'Is_Marijuana_Related' not in df.columns:
            # Comprehensive marijuana detection
            marijuana_patterns = [
                'MARIJUANA', 'CANNABIS', 'THC', 'WEED', 'POT',
                '893.13'  # Florida statute for drug possession
            ]
            pattern = '|'.join(marijuana_patterns)
            
            df['Is_Marijuana_Related'] = (
                df['ChargeOffenseDescription'].str.contains(pattern, case=False, na=False) |
                df['Statute_Description'].str.contains(pattern, case=False, na=False) |
                df['Statute'].str.contains('893\.13', na=False, regex=True)
            )
            
            # Detect felony amounts (over 20 grams)
            felony_patterns = [
                '20.*GRAM', 'OVER 20', 'MORE THAN 20',
                '2[1-9].*GRAM', '[3-9][0-9].*GRAM',  # 21+ grams
                'FELONY.*POSSESS', 'OUNCE'
            ]
            felony_pattern = '|'.join(felony_patterns)
            
            df['Is_Felony_Amount'] = df['ChargeOffenseDescription'].str.contains(
                felony_pattern, case=False, na=False
            )
        
        # Create charge categories
        df['Charge_Category'] = df.apply(categorize_charge, axis=1)
        
        # Add officer analysis columns
        officer_790_counts = df[df['Is_790_07']].groupby(['Lead_Officer', 'Lead_Agency']).size()
        df['Officer_790_Count'] = df.apply(
            lambda x: officer_790_counts.get((x['Lead_Officer'], x['Lead_Agency']), 0), axis=1
        )
        
        return df, None
        
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

def categorize_charge(row):
    """
    Categorize charges into analytical groups
    """
    statute = str(row.get('Statute', '')).upper()
    desc = str(row.get('ChargeOffenseDescription', '')).upper()
    stat_desc = str(row.get('Statute_Description', '')).upper()
    
    # Combine all text for better matching
    combined = f"{statute} {desc} {stat_desc}"
    
    # Specific categorization logic
    if '790.07' in statute:
        return '790.07 - Firearm During Felony'
    elif row.get('Is_Marijuana_Related', False):
        if row.get('Is_Felony_Amount', False):
            return 'Marijuana - Felony Amount (20g+)'
        elif 'MISDEMEANOR' in combined:
            return 'Marijuana - Misdemeanor'
        else:
            return 'Marijuana - Other'
    elif '790.23' in statute:
        return 'Felon in Possession'
    elif '790.01' in statute:
        return 'Unlicensed Concealed Carry'
    elif '790.06' in statute:
        return 'CCW License Violation'
    elif '790' in statute[:3]:
        return 'Other Firearm Offenses'
    elif '893' in statute[:3]:
        return 'Other Drug Offenses'
    elif any(x in statute for x in ['784', '787', '812.13']):
        return 'Violent Crimes'
    elif '316' in statute[:3]:
        return 'Traffic Offenses'
    else:
        return 'Other Offenses'

def create_officer_ranking_table(df):
    """
    Create officer ranking based on 790.07 enforcement
    """
    # Calculate officer statistics
    officer_stats = df.groupby(['Lead_Officer', 'Lead_Agency']).agg({
        'Is_790_07': 'sum',
        'Is_Marijuana_Related': 'sum',
        'Defendant': 'nunique',
        'CaseNumber': 'nunique'
    }).reset_index()
    
    officer_stats.columns = ['Officer', 'Agency', '790.07 Charges', 'Marijuana Charges', 
                            'Unique Defendants', 'Total Cases']
    
    # Calculate marijuana-firearm combo rate
    combo_stats = df[df['Is_790_07'] & df['Is_Marijuana_Related']].groupby(
        ['Lead_Officer', 'Lead_Agency']
    ).size().reset_index(name='Marijuana-Firearm Combos')
    
    officer_stats = officer_stats.merge(
        combo_stats, 
        left_on=['Officer', 'Agency'], 
        right_on=['Lead_Officer', 'Lead_Agency'], 
        how='left'
    ).fillna(0)
    
    # Calculate combo percentage
    officer_stats['Combo Rate %'] = (
        officer_stats['Marijuana-Firearm Combos'] / officer_stats['790.07 Charges'] * 100
    ).round(1)
    
    # Sort by 790.07 charges
    officer_stats = officer_stats.sort_values('790.07 Charges', ascending=False)
    
    return officer_stats[['Officer', 'Agency', '790.07 Charges', 'Marijuana Charges', 
                         'Marijuana-Firearm Combos', 'Combo Rate %', 'Unique Defendants']]

# Layout
app.layout = dbc.Container([
    # Store component for data
    dcc.Store(id='stored-data'),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("790.07 Enforcement Analysis Dashboard", 
                       className="text-white mb-2"),
                html.P("Analyzing the nexus between marijuana possession and firearm charges", 
                      className="text-white-50 lead"),
                html.P("Focus: Simple possession (20-25g) leading to felony firearm charges", 
                      className="text-white-50")
            ], style={
                'background': 'linear-gradient(135deg, #d62728 0%, #ff7f0e 100%)',
                'padding': '2rem',
                'borderRadius': '10px',
                'marginBottom': '2rem',
                'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
            })
        ])
    ]),
    
    # Data upload
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Data Upload", className="card-title mb-3"),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            html.I(className="bi bi-cloud-upload", style={'fontSize': '2rem'}),
                            html.Br(),
                            'Drag and Drop or ',
                            html.A('Select CSV File', style={'color': COLORS['primary']})
                        ]),
                        style={
                            'width': '100%',
                            'height': '120px',
                            'lineHeight': '40px',
                            'borderWidth': '2px',
                            'borderStyle': 'dashed',
                            'borderRadius': '10px',
                            'textAlign': 'center',
                            'margin': '10px',
                            'cursor': 'pointer',
                            'borderColor': COLORS['primary']
                        }
                    ),
                    html.Div(id='upload-status', className="mt-3")
                ])
            ], className="shadow")
        ])
    ], className="mb-4"),
    
    # Key Metrics
    html.Div(id='metrics-container', children=[
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Total 790.07 Defendants", className="text-muted"),
                        html.H2("--", id="total-defendants", className="text-danger mb-0"),
                        html.Small("Unique individuals", className="text-muted")
                    ])
                ], className="shadow-sm border-0")
            ], md=2),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Marijuana + 790.07 Combo", className="text-muted"),
                        html.H2("--", id="marijuana-combo", className="text-success mb-0"),
                        html.Small("Co-occurring charges", className="text-muted")
                    ])
                ], className="shadow-sm border-0", style={'borderLeft': f'4px solid {COLORS["marijuana"]}'}),
            ], md=2),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Felony Amount Cases", className="text-muted"),
                        html.H2("--", id="felony-amount", className="text-warning mb-0"),
                        html.Small("20+ grams", className="text-muted")
                    ])
                ], className="shadow-sm border-0")
            ], md=2),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Combo Rate", className="text-muted"),
                        html.H2("--", id="combo-rate", className="text-info mb-0"),
                        html.Small("Of 790.07 charges", className="text-muted")
                    ])
                ], className="shadow-sm border-0")
            ], md=2),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Active Officers", className="text-muted"),
                        html.H2("--", id="active-officers", className="text-primary mb-0"),
                        html.Small("With 790.07 charges", className="text-muted")
                    ])
                ], className="shadow-sm border-0")
            ], md=2),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Agencies Involved", className="text-muted"),
                        html.H2("--", id="active-agencies", className="text-dark mb-0"),
                        html.Small("Unique agencies", className="text-muted")
                    ])
                ], className="shadow-sm border-0")
            ], md=2),
        ], className="mb-4")
    ]),
    
    # Filters
    dbc.Row([
        dbc.Col([
            html.Label("Select Agency", className="fw-bold mb-2"),
            dcc.Dropdown(
                id='agency-filter',
                options=[],
                multi=True,
                placeholder="All Agencies"
            )
        ], md=4),
        
        dbc.Col([
            html.Label("Date Range", className="fw-bold mb-2"),
            dcc.DatePickerRange(
                id='date-range',
                display_format='YYYY-MM-DD',
                style={'width': '100%'}
            )
        ], md=4),
        
        dbc.Col([
            html.Label("Charge Type Filter", className="fw-bold mb-2"),
            dcc.Dropdown(
                id='charge-filter',
                options=[
                    {'label': 'All Charges', 'value': 'all'},
                    {'label': 'Marijuana + 790.07 Combo Only', 'value': 'combo'},
                    {'label': 'Felony Amount Only', 'value': 'felony'},
                    {'label': '790.07 Only', 'value': '790only'}
                ],
                value='all'
            )
        ], md=4),
    ], className="mb-4"),
    
    # Main Analysis Tabs
    dbc.Tabs([
        dbc.Tab(label="Officer Analysis", tab_id="officer-tab"),
        dbc.Tab(label="Charge Patterns", tab_id="pattern-tab"),
        dbc.Tab(label="Timeline Analysis", tab_id="timeline-tab"),
        dbc.Tab(label="Red Flags", tab_id="redflags-tab"),
    ], id="tabs", active_tab="officer-tab", className="mb-4"),
    
    # Tab content
    html.Div(id='tab-content'),
    
    # Footer
    html.Hr(className="mt-5"),
    dbc.Row([
        dbc.Col([
            html.P([
                "Dashboard analyzing Florida Statute 790.07 enforcement patterns. ",
                "Focus on marijuana possession amounts and concealed carry interactions."
            ], className="text-muted text-center small")
        ])
    ])
    
], fluid=True, style={'backgroundColor': COLORS['background'], 'minHeight': '100vh'})

# Callbacks
@app.callback(
    [Output('stored-data', 'data'),
     Output('upload-status', 'children'),
     Output('agency-filter', 'options'),
     Output('date-range', 'start_date'),
     Output('date-range', 'end_date')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def upload_and_process_data(contents, filename):
    """
    Handle file upload and initial processing
    """
    if contents is None:
        return None, dbc.Alert("Please upload a CSV file to begin analysis", color="info"), [], None, None
    
    df, error = process_uploaded_data(contents, filename)
    
    if error:
        return None, dbc.Alert(error, color="danger"), [], None, None
    
    # Get agency options
    agencies = sorted(df['Lead_Agency'].dropna().unique())
    agency_options = [{'label': agency, 'value': agency} for agency in agencies]
    
    # Get date range
    start_date = df['FileDate'].min()
    end_date = df['FileDate'].max()
    
    # Success message
    success_msg = dbc.Alert([
        html.I(className="bi bi-check-circle-fill me-2"),
        f"Successfully loaded {len(df):,} charges from {df['Defendant'].nunique():,} defendants"
    ], color="success", dismissable=True)
    
    return df.to_dict('records'), success_msg, agency_options, start_date, end_date

@app.callback(
    [Output('total-defendants', 'children'),
     Output('marijuana-combo', 'children'),
     Output('felony-amount', 'children'),
     Output('combo-rate', 'children'),
     Output('active-officers', 'children'),
     Output('active-agencies', 'children')],
    [Input('stored-data', 'data'),
     Input('agency-filter', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('charge-filter', 'value')]
)
def update_metrics(data, agencies, start_date, end_date, charge_filter):
    """
    Update key metric cards
    """
    if not data:
        return ["--"] * 6
    
    df = pd.DataFrame(data)
    df['FileDate'] = pd.to_datetime(df['FileDate'])
    
    # Apply filters
    if agencies:
        df = df[df['Lead_Agency'].isin(agencies)]
    
    if start_date and end_date:
        df = df[(df['FileDate'] >= start_date) & (df['FileDate'] <= end_date)]
    
    # Calculate metrics
    total_defendants = df[df['Is_790_07']]['Defendant'].nunique()
    
    # Marijuana combo analysis
    combo_defendants = df[df['Is_790_07'] & df['Is_Marijuana_Related']]['Defendant'].nunique()
    
    # Felony amount cases
    felony_cases = df[df['Is_790_07'] & df['Is_Felony_Amount']]['CaseNumber'].nunique()
    
    # Combo rate
    combo_rate = (combo_defendants / total_defendants * 100) if total_defendants > 0 else 0
    
    # Officers and agencies
    active_officers = df[df['Is_790_07']]['Lead_Officer'].nunique()
    active_agencies = df[df['Is_790_07']]['Lead_Agency'].nunique()
    
    return (
        f"{total_defendants:,}",
        f"{combo_defendants:,}",
        f"{felony_cases:,}",
        f"{combo_rate:.1f}%",
        f"{active_officers:,}",
        f"{active_agencies}"
    )

@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'active_tab'),
     Input('stored-data', 'data'),
     Input('agency-filter', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('charge-filter', 'value')]
)
def render_tab_content(active_tab, data, agencies, start_date, end_date, charge_filter):
    """
    Render content based on selected tab
    """
    if not data:
        return html.Div("Please upload data to view analysis", className="text-center text-muted p-5")
    
    df = pd.DataFrame(data)
    df['FileDate'] = pd.to_datetime(df['FileDate'])
    
    # Apply filters
    if agencies:
        df = df[df['Lead_Agency'].isin(agencies)]
    
    if start_date and end_date:
        df = df[(df['FileDate'] >= start_date) & (df['FileDate'] <= end_date)]
    
    if charge_filter == 'combo':
        # Get defendants with both charges
        combo_defendants = df[df['Is_790_07'] & df['Is_Marijuana_Related']]['Defendant'].unique()
        df = df[df['Defendant'].isin(combo_defendants)]
    elif charge_filter == 'felony':
        df = df[df['Is_Felony_Amount']]
    elif charge_filter == '790only':
        df = df[df['Is_790_07']]
    
    # Render based on active tab
    if active_tab == 'officer-tab':
        return render_officer_analysis(df)
    elif active_tab == 'pattern-tab':
        return render_charge_patterns(df)
    elif active_tab == 'timeline-tab':
        return render_timeline_analysis(df)
    elif active_tab == 'redflags-tab':
        return render_red_flags(df)
    
    return html.Div("Select a tab")

def render_officer_analysis(df):
    """
    Render officer ranking and analysis
    """
    officer_stats = create_officer_ranking_table(df)
    
    # Create bar chart of top officers
    top_officers = officer_stats.head(20)
    
    fig = go.Figure()
    
    # Add 790.07 charges
    fig.add_trace(go.Bar(
        x=top_officers['Officer'],
        y=top_officers['790.07 Charges'],
        name='790.07 Charges',
        marker_color=COLORS['firearm'],
        text=top_officers['790.07 Charges'],
        textposition='auto'
    ))
    
    # Add marijuana charges
    fig.add_trace(go.Bar(
        x=top_officers['Officer'],
        y=top_officers['Marijuana Charges'],
        name='Marijuana Charges',
        marker_color=COLORS['marijuana'],
        text=top_officers['Marijuana Charges'],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Top 20 Officers by 790.07 Enforcement',
        xaxis_title='Officer',
        yaxis_title='Number of Charges',
        barmode='group',
        height=500,
        xaxis_tickangle=-45
    )
    
    # Create data table
    table = dash_table.DataTable(
        id='officer-table',
        columns=[
            {'name': col, 'id': col, 'type': 'numeric' if col != 'Officer' and col != 'Agency' else 'text'}
            for col in officer_stats.columns
        ],
        data=officer_stats.to_dict('records'),
        sort_action='native',
        filter_action='native',
        page_size=20,
        style_cell={'textAlign': 'left'},
        style_data_conditional=[
            {
                'if': {'column_id': 'Combo Rate %'},
                'backgroundColor': COLORS['warning'],
                'color': 'white',
                'fontWeight': 'bold'
            }
        ]
    )
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig)
            ])
        ]),
        dbc.Row([
            dbc.Col([
                html.H4("Officer Ranking Table", className="mb-3"),
                html.P("Click column headers to sort. Use filters to search.", className="text-muted"),
                table
            ])
        ], className="mt-4")
    ])

def render_charge_patterns(df):
    """
    Render charge pattern analysis
    """
    # Create Sankey diagram showing flow from marijuana to 790.07
    marijuana_defs = df[df['Is_Marijuana_Related']]['Defendant'].unique()
    firearm_defs = df[df['Is_790_07']]['Defendant'].unique()
    both_defs = set(marijuana_defs) & set(firearm_defs)
    
    # Calculate flows
    only_marijuana = len(set(marijuana_defs) - set(firearm_defs))
    only_firearm = len(set(firearm_defs) - set(marijuana_defs))
    both = len(both_defs)
    
    # Sankey diagram
    fig_sankey = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["All Defendants", "Marijuana Only", "Both Charges", "790.07 Only"],
            color=[COLORS['dark'], COLORS['marijuana'], COLORS['warning'], COLORS['firearm']]
        ),
        link=dict(
            source=[0, 0, 0],
            target=[1, 2, 3],
            value=[only_marijuana, both, only_firearm],
            color=[COLORS['marijuana'], COLORS['warning'], COLORS['firearm']]
        )
    ))
    
    fig_sankey.update_layout(
        title="Defendant Charge Flow Analysis",
        height=400
    )
    
    # Charge category distribution for combo cases
    combo_charges = df[df['Defendant'].isin(both_defs)]
    charge_dist = combo_charges['Charge_Category'].value_counts()
    
    fig_dist = px.bar(
        x=charge_dist.values,
        y=charge_dist.index,
        orientation='h',
        title='Charge Distribution for Marijuana + 790.07 Defendants',
        labels={'x': 'Count', 'y': 'Charge Category'},
        color=charge_dist.values,
        color_continuous_scale='Reds'
    )
    
    fig_dist.update_layout(height=500)
    
    # Specific marijuana amount analysis
    marijuana_charges = df[df['Is_Marijuana_Related']]
    
    # Extract gram amounts where possible
    gram_pattern = r'(\d+)\s*(?:GRAM|G)'
    marijuana_charges['Gram_Amount'] = marijuana_charges['ChargeOffenseDescription'].str.extract(gram_pattern)[0]
    marijuana_charges['Gram_Amount'] = pd.to_numeric(marijuana_charges['Gram_Amount'], errors='coerce')
    
    # Create histogram of amounts
    amount_data = marijuana_charges.dropna(subset=['Gram_Amount'])
    
    if len(amount_data) > 0:
        fig_amounts = px.histogram(
            amount_data,
            x='Gram_Amount',
            nbins=20,
            title='Distribution of Marijuana Amounts (grams)',
            labels={'Gram_Amount': 'Grams', 'count': 'Number of Charges'},
            color_discrete_sequence=[COLORS['marijuana']]
        )
        
        # Add vertical line at 20 grams (felony threshold)
        fig_amounts.add_vline(x=20, line_dash="dash", line_color="red", 
                            annotation_text="Felony Threshold (20g)")
        
        fig_amounts.update_layout(height=400)
    else:
        fig_amounts = None
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_sankey)
            ], md=6),
            dbc.Col([
                dcc.Graph(figure=fig_dist)
            ], md=6)
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_amounts) if fig_amounts else 
                html.Div("No specific gram amounts found in data", className="text-center p-5 text-muted")
            ])
        ], className="mt-4")
    ])

def render_timeline_analysis(df):
    """
    Render timeline analysis
    """
    # Monthly trend of charges
    df['YearMonth'] = df['FileDate'].dt.to_period('M').astype(str)
    
    monthly_data = df.groupby(['YearMonth', 'Is_790_07', 'Is_Marijuana_Related']).size().reset_index(name='Count')
    
    # Create charge type column
    monthly_data['Charge_Type'] = monthly_data.apply(
        lambda x: '790.07 + Marijuana' if x['Is_790_07'] and x['Is_Marijuana_Related']
        else '790.07 Only' if x['Is_790_07']
        else 'Marijuana Only' if x['Is_Marijuana_Related']
        else 'Other', axis=1
    )
    
    # Group by month and charge type
    timeline = monthly_data.groupby(['YearMonth', 'Charge_Type'])['Count'].sum().reset_index()
    
    fig_timeline = px.line(
        timeline,
        x='YearMonth',
        y='Count',
        color='Charge_Type',
        title='Charge Trends Over Time',
        markers=True,
        color_discrete_map={
            '790.07 + Marijuana': COLORS['warning'],
            '790.07 Only': COLORS['firearm'],
            'Marijuana Only': COLORS['marijuana'],
            'Other': COLORS['dark']
        }
    )
    
    fig_timeline.update_layout(
        xaxis_title='Month',
        yaxis_title='Number of Charges',
        height=500
    )
    
    # Agency comparison over time
    agency_monthly = df[df['Is_790_07']].groupby(
        ['YearMonth', 'Lead_Agency']
    ).size().reset_index(name='Count')
    
    # Get top 5 agencies
    top_agencies = df[df['Is_790_07']]['Lead_Agency'].value_counts().head(5).index
    agency_monthly_top = agency_monthly[agency_monthly['Lead_Agency'].isin(top_agencies)]
    
    fig_agency = px.line(
        agency_monthly_top,
        x='YearMonth',
        y='Count',
        color='Lead_Agency',
        title='790.07 Charges by Top 5 Agencies Over Time',
        markers=True
    )
    
    fig_agency.update_layout(
        xaxis_title='Month',
        yaxis_title='Number of 790.07 Charges',
        height=400
    )
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_timeline)
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_agency)
            ])
        ], className="mt-4")
    ])

def render_red_flags(df):
    """
    Identify and display potential red flags
    """
    red_flags = []
    
    # 1. High marijuana-firearm correlation
    marijuana_defs = set(df[df['Is_Marijuana_Related']]['Defendant'].unique())
    firearm_defs = set(df[df['Is_790_07']]['Defendant'].unique())
    overlap = len(marijuana_defs & firearm_defs)
    overlap_rate = overlap / len(firearm_defs) * 100 if len(firearm_defs) > 0 else 0
    
    if overlap_rate > 30:
        red_flags.append({
            'type': 'warning',
            'title': 'High Marijuana-Firearm Correlation',
            'message': f'{overlap_rate:.1f}% of 790.07 defendants also have marijuana charges, suggesting systematic charge stacking.'
        })
    
    # 2. Felony threshold clustering
    marijuana_charges = df[df['Is_Marijuana_Related']]
    gram_pattern = r'(\d+)\s*(?:GRAM|G)'
    marijuana_charges['Gram_Amount'] = marijuana_charges['ChargeOffenseDescription'].str.extract(gram_pattern)[0]
    marijuana_charges['Gram_Amount'] = pd.to_numeric(marijuana_charges['Gram_Amount'], errors='coerce')
    
    amounts = marijuana_charges['Gram_Amount'].dropna()
    if len(amounts) > 10:
        threshold_cases = amounts[(amounts >= 20) & (amounts <= 25)].count()
        threshold_rate = threshold_cases / len(amounts) * 100
        
        if threshold_rate > 20:
            red_flags.append({
                'type': 'danger',
                'title': 'Suspicious Amount Clustering',
                'message': f'{threshold_rate:.1f}% of marijuana cases fall between 20-25 grams, just over the felony threshold.'
            })
    
    # 3. Officer outliers
    officer_stats = df[df['Is_790_07']].groupby('Lead_Officer').agg({
        'CaseNumber': 'nunique',
        'Is_Marijuana_Related': 'sum'
    }).reset_index()
    
    officer_stats['Marijuana_Rate'] = officer_stats['Is_Marijuana_Related'] / officer_stats['CaseNumber']
    
    # Find officers with unusually high rates
    mean_rate = officer_stats['Marijuana_Rate'].mean()
    std_rate = officer_stats['Marijuana_Rate'].std()
    
    outlier_officers = officer_stats[
        officer_stats['Marijuana_Rate'] > mean_rate + 2 * std_rate
    ]
    
    if len(outlier_officers) > 0:
        red_flags.append({
            'type': 'info',
            'title': 'Officer Enforcement Patterns',
            'message': f'{len(outlier_officers)} officers show significantly higher marijuana-firearm combination rates than average.'
        })
    
    # 4. Demographic disparities
    demo_stats = df[df['Is_790_07']].groupby('Race').size()
    if len(demo_stats) > 1:
        ratio = demo_stats.max() / demo_stats.min()
        if ratio > 3:
            red_flags.append({
                'type': 'warning',
                'title': 'Demographic Disparities',
                'message': f'Significant racial disparities detected with a {ratio:.1f}x difference between groups.'
            })
    
    # 5. CCW holder targeting
    ccw_keywords = ['CONCEALED', 'CCW', 'PERMIT', 'LICENSE']
    ccw_pattern = '|'.join(ccw_keywords)
    
    ccw_cases = df[
        df['Is_790_07'] & 
        df['ChargeOffenseDescription'].str.contains(ccw_pattern, case=False, na=False)
    ]
    
    if len(ccw_cases) > 0:
        ccw_rate = len(ccw_cases) / len(df[df['Is_790_07']]) * 100
        if ccw_rate > 10:
            red_flags.append({
                'type': 'danger',
                'title': 'Concealed Carry Permit Holders',
                'message': f'{ccw_rate:.1f}% of 790.07 cases involve concealed carry references, suggesting lawful gun owners are being targeted.'
            })
    
    # Create alert cards for each red flag
    alerts = []
    for flag in red_flags:
        alerts.append(
            dbc.Alert([
                html.H5(flag['title'], className="alert-heading"),
                html.P(flag['message'])
            ], color=flag['type'], className="mb-3")
        )
    
    # Summary statistics
    summary_stats = dbc.Card([
        dbc.CardBody([
            html.H4("Analysis Summary", className="card-title"),
            html.Hr(),
            html.P([
                html.Strong("Total 790.07 Defendants: "),
                f"{df[df['Is_790_07']]['Defendant'].nunique():,}"
            ]),
            html.P([
                html.Strong("Marijuana Co-occurrence Rate: "),
                f"{overlap_rate:.1f}%"
            ]),
            html.P([
                html.Strong("Average Charges per Defendant: "),
                f"{df.groupby('Defendant').size().mean():.1f}"
            ]),
            html.P([
                html.Strong("Date Range: "),
                f"{df['FileDate'].min().strftime('%Y-%m-%d')} to {df['FileDate'].max().strftime('%Y-%m-%d')}"
            ])
        ])
    ], className="mb-4")
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H3("Red Flag Analysis", className="mb-4"),
                summary_stats,
                *alerts if alerts else [
                    dbc.Alert("No significant red flags detected in the current data selection.", 
                             color="success")
                ]
            ])
        ])
    ])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
