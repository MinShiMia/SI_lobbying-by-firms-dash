'''
 # @ Create Time: 2025-02-14 11:22:20.564782
'''

import dash
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import io
import os
import requests


app = Dash(__name__, title="lobbying-by-firms-dash")

# Declare server for Heroku deployment. Needed for Procfile.
server = app.server


# Load data from GitHub with error handling
def load_csv_from_github(url):
    try:
        response = requests.get(url, timeout=10)  # Set a timeout to avoid hanging requests
        response.raise_for_status()  # Raise an error for HTTP errors (404, 403, etc.)
        csv_data = response.content.decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_data))
        print(df.head())
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure


# URLs for CSV files
df_frequency_url = "https://raw.githubusercontent.com/MinShiMia/SI_lobbying-by-firms-dash/main/data/lda_quarterly_frequency_by_registrant_organization_over_time.csv"
df_expenses_url = "https://raw.githubusercontent.com/MinShiMia/SI_lobbying-by-firms-dash/main/data/lda_quarterly_total_lobbying_expenses_by_registrant_organization_over_time.csv"
df_avg_expenses_url = "https://raw.githubusercontent.com/MinShiMia/SI_lobbying-by-firms-dash/main/data/lda_quarterly_avg_lobbying_expenses_by_registrant_organization_over_time.csv"

# Load the data
df_frequency = load_csv_from_github(df_frequency_url)
df_expenses = load_csv_from_github(df_expenses_url)
df_avg_expenses = load_csv_from_github(df_avg_expenses_url)

# Check if data loaded successfully
if df_frequency.empty:
    print("Failed to load df_frequency.")
if df_expenses.empty:
    print("Failed to load df_expenses.")
if df_avg_expenses.empty:
    print("Failed to load df_expenses.")

# Convert lobbying_expenses to millions for readability
df_expenses['LDA_lobbying_expenses'] = pd.to_numeric(df_expenses['LDA_lobbying_expenses'], errors='coerce').fillna(0)
df_expenses['LDA_lobbying_expenses'] = df_expenses['LDA_lobbying_expenses'] / 1_000_000  # Convert to millions
df_avg_expenses['LDA_lobbying_expenses'] = pd.to_numeric(df_avg_expenses['LDA_lobbying_expenses'], errors='coerce').fillna(0)

# Aggregate Data by issue_name
agg_frequency = (
    df_frequency.groupby('registrant_name')['LDA_frequency']
    .sum()
    .reset_index(name='Total Frequency')
)

agg_expenses = (
    df_expenses.groupby('registrant_name')['LDA_lobbying_expenses']
    .sum()
    .reset_index(name='Total Expenses ($ Million)')
)

agg_avg_expenses = (
    df_avg_expenses.groupby('registrant_name')['LDA_lobbying_expenses']
    .mean()
    .reset_index(name='Average Expenses')
)


# Initialize Dash App
app = dash.Dash(__name__)
app.title = "LDA Frequency and Expenses Dashboard"

# App Layout
app.layout = html.Div([
# App Title
    html.H1(
        "LDA Filing Frequency and Lobbying Expenses by Lobbying Firm Dashboard",
        style={
            'text-align': 'center',
            'font-size': '22px',
            'margin-bottom': '20px',  # Reduce space below the title
            'margin-top': '20px'       # Reduce space above the title
        }
    ),
    # Filter Section at the Top
    html.Div([
        # Select Year filter
        html.Div([
            html.Label("Select Year:", style={'font-size': '14px', 'font-weight': 'bold'}),
            dcc.Dropdown(
                id='year-dropdown',
                options=[{'label': str(year), 'value': year} for year in sorted(df_frequency['Year'].unique())],
                multi=True,
                placeholder="Select Year(s)"
            ),
        ], style={'width': '25%', 'display': 'inline-block', 'padding-right': '10px'}),

        # Select Quarter filter
        html.Div([
            html.Label("Select Quarter:", style={'font-size': '14px', 'font-weight': 'bold'}),
            dcc.Dropdown(
                id='quarter-dropdown',
                options=[{'label': f"Q{q}", 'value': q} for q in sorted(df_frequency['Quarter'].unique())],
                multi=True,
                placeholder="Select Quarter(s)"
            ),
        ], style={'width': '25%', 'display': 'inline-block', 'padding-right': '10px'}),

        # Sort By Dropdown filter
        html.Div([
            html.Label("Sort By:", style={'font-size': '14px', 'font-weight': 'bold'}),
            dcc.Dropdown(
                id='order-dropdown',
                options=[
                    {'label': 'Total Frequency', 'value': 'Total Frequency'},
                    {'label': 'Total Expenses', 'value': 'Total Expenses'}
                ],
                value='Total Frequency',  # Default sorting
                clearable=False
            ),
        ], style={'width': '25%', 'display': 'inline-block', 'padding-right': '10px'}),

        # Select Top Firms Dropdown
        html.Div([
            html.Label("Top Firms to Show:", style={'font-size': '14px', 'font-weight': 'bold'}),
            dcc.Dropdown(
                id='top-firms-dropdown',
                options=[{'label': f"Top {i}", 'value': i} for i in range(10, 110, 10)],  # 10 to 100, step 10
                value=30,  # Default value
                clearable=False
            ),
        ], style={'width': '24%', 'display': 'inline-block'})
    ], style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '0px'}),

    # Dual-Axis Bar Chart
    dcc.Graph(id='dual-axis-bar-chart', style={"height": "650px"}),

    # Dynamic Title Section
    html.Div(id='dynamic-title', style={'font-size': '20px', 'font-weight': 'bold', 'margin-bottom': '0px'}),

    # Time Series Plots (Three Columns with Reduced Space and Y-axis Titles)
    html.Div([
        html.Div([
            dcc.Graph(id='frequency-time-series-plot', style={"height": "400px"}),
            html.Div("LDA Frequency Over Time",
                     style={'text-align': 'center', 'font-size': '14px', 'font-weight': 'bold'})
        ], style={'width': '33%', 'display': 'inline-block', 'padding-right': '10px'}),

        html.Div([
            dcc.Graph(id='expenses-time-series-plot', style={"height": "400px"}),
            html.Div("Total Lobbying Expenses (in Millions)",
                     style={'text-align': 'center', 'font-size': '14px', 'font-weight': 'bold'})
        ], style={'width': '33%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(id='avg-expenses-time-series-plot', style={"height": "400px"}),
            html.Div("Average Lobbying Expenses",
                     style={'text-align': 'center', 'font-size': '14px', 'font-weight': 'bold'})
        ], style={'width': '33%', 'display': 'inline-block'})
    ], style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between', 'margin-top': '0px'}),

])


# Callback for Dynamic Title
@app.callback(
    Output('dynamic-title', 'children'),
    [Input('dual-axis-bar-chart', 'clickData')]
)

def update_dynamic_title(clickData):
    # Default issue's external link
    default_lobbying_firm_url = "https://app.legis1.com/lobbying-firm/detail?organizationId=44966&actorId=2963#summary"

    if not clickData:
        return html.Span([
            f"LDA filing frequency and lobbying expenses for default organization: ",
            html.A(f"Brownstein Hyatt Farber Schreck LLP", href=default_lobbying_firm_url, target="_blank",
                   style={'color': 'blue', 'text-decoration': 'underline'}),
            ". Click on another top organization's bar to see its trend."
        ])

    # Extract selected issue name
    selected_firm = clickData['points'][0]['x']

    # Get the corresponding registrant_lobby_actor_id and organization_id for the selected lobbying firm
    selected_lobby_actor_id = df_frequency.loc[df_frequency['registrant_name'] == selected_firm, 'registrant_lobby_actor_id'].values[0]
    selected_firm_id = df_frequency.loc[df_frequency['registrant_name'] == selected_firm, 'organization_id'].values[0]

    # Construct the external URL
    external_url = f"https://app.legis1.com/lobbying-firm/detail?organizationId={selected_firm_id}&actorId={selected_lobby_actor_id}#summary"

    # Return the dynamic title with a clickable link
    return html.Span([
        f"LDA filing frequency and lobbying expenses for selected lobbying firm: ",
        html.A(f"{selected_firm}", href=external_url, target="_blank",
               style={'color': 'blue', 'text-decoration': 'underline'}),
        "."
    ])


# Callback for the Dual-Axis Bar Chart
@app.callback(
    Output('dual-axis-bar-chart', 'figure'),
    [Input('year-dropdown', 'value'),
     Input('quarter-dropdown', 'value'),
     Input('order-dropdown', 'value'),
     Input('top-firms-dropdown', 'value'),]
)


def update_dual_axis_bar_chart(selected_years, selected_quarters, selected_order, top_n):
    # Filter the data based on the selected years and quarters
    filtered_frequency = df_frequency
    filtered_expenses = df_expenses

    if selected_years:
        filtered_frequency = filtered_frequency[filtered_frequency['Year'].isin(selected_years)]
        filtered_expenses = filtered_expenses[filtered_expenses['Year'].isin(selected_years)]

    if selected_quarters:
        filtered_frequency = filtered_frequency[filtered_frequency['Quarter'].isin(selected_quarters)]
        filtered_expenses = filtered_expenses[filtered_expenses['Quarter'].isin(selected_quarters)]

    # Aggregate data for bar chart
    agg_frequency = (
        filtered_frequency.groupby('registrant_name')['LDA_frequency']
        .sum()
        .reset_index(name='Total Frequency')
    )

    agg_expenses = (
        filtered_expenses.groupby('registrant_name')['LDA_lobbying_expenses']
        .sum()
        .reset_index(name='Total Expenses')
    )
    # agg_expenses['Total Expenses'] = agg_expenses['Total Expenses'] / 1_000_000  # Convert to millions

    # Merge the frequency and expenses data for plotting
    merged_data = pd.merge(agg_frequency, agg_expenses, on='registrant_name', how='outer').fillna(0)

    # Sort the data dynamically based on user selection
    if selected_order and top_n:
        # Ensure selected_order is valid
        if selected_order in ["Total Frequency", "Total Expenses"]:
            merged_data = merged_data.sort_values(by=selected_order, ascending=False)

        # Limit to top N firms after sorting
        merged_data = merged_data.head(top_n)

    # Create the bar chart with two traces (Frequency and Expenses)
    fig = go.Figure()

    # Add Total Frequency bars on the left y-axis
    fig.add_trace(go.Bar(
        x=merged_data['registrant_name'],
        y=merged_data['Total Frequency'],
        name='Total Frequency',
        marker_color='rgb(46, 89, 142)',
        yaxis='y1',
        offsetgroup=0
    ))

    # Add Total Expenses bars on the right y-axis
    fig.add_trace(go.Bar(
        x=merged_data['registrant_name'],
        y=merged_data['Total Expenses'],
        name='Total Expenses ($)',
        marker_color='rgb(77, 0, 0)',
        yaxis='y2',
        offsetgroup=1
    ))

    # Layout for dual y-axis
    fig.update_layout(
        xaxis=dict(tickangle=-45, categoryorder='array', categoryarray=merged_data['registrant_name']),
        yaxis=dict(title="Total Frequency"),
        yaxis2=dict(title="Total Expenses ($)", overlaying='y', side='right'),
        barmode='group',
        bargap=0.2,
        height=600,
        legend=dict(x=0.01, y=1.15, orientation="h")
    )

    return fig


# Callback for Time Series Plots
@app.callback(
    [Output('frequency-time-series-plot', 'figure'),
     Output('expenses-time-series-plot', 'figure'),
     Output('avg-expenses-time-series-plot', 'figure')],
    [Input('dual-axis-bar-chart', 'clickData')]
)


def update_time_series_plot(clickData):
    # Default to "Affordable Care Act (ACA)" if no bar is clicked
    selected_firm = "Brownstein Hyatt Farber Schreck LLP"
    if clickData:
        selected_firm = clickData['points'][0]['x']

    # Frequency Time Series Data
    firm_data_freq = (
        df_frequency[df_frequency['registrant_name'] == selected_firm]
        .groupby(['Year', 'Quarter'])['LDA_frequency']
        .sum()
        .reset_index()
    )
    firm_data_freq['Year_Quarter'] = firm_data_freq['Year'].astype(str) + " Q" + firm_data_freq['Quarter'].astype(str)

    # Frequency Time Series Plot
    freq_fig = go.Figure()
    freq_fig.add_trace(go.Scatter(
        x=firm_data_freq['Year_Quarter'],
        y=firm_data_freq['LDA_frequency'],
        mode='lines+markers',
        line=dict(color='teal'),
        name='Frequency'
    ))
    freq_fig.update_layout(
        title="",
        yaxis_title="Frequency",
        xaxis_title="Year-Quarter",
        height=400,
        margin=dict(t=20, b=40)
    )

    # Expenses Time Series Data
    firm_data_exp = (
        df_expenses[df_expenses['registrant_name'] == selected_firm]
        .groupby(['Year', 'Quarter'])['LDA_lobbying_expenses']
        .sum()
        .reset_index()
    )
    firm_data_exp['Year_Quarter'] = firm_data_exp['Year'].astype(str) + " Q" + firm_data_exp['Quarter'].astype(str)

    # Expenses Time Series Plot
    exp_fig = go.Figure()
    exp_fig.add_trace(go.Scatter(
        x=firm_data_exp['Year_Quarter'],
        y=firm_data_exp['LDA_lobbying_expenses'],
        mode='lines+markers',
        line=dict(color='orange'),
        name='Lobbying Expenses'
    ))
    exp_fig.update_layout(
        title="",
        yaxis_title="Lobbying Expenses (Millions)",
        # yaxis_tickformat=",.2f",
        xaxis_title="Year-Quarter",
        height=400,
        margin=dict(t=20, b=40)
    )

    # Expenses Time Series Data
    firm_data_avg_exp = (
        df_avg_expenses[df_avg_expenses['registrant_name'] == selected_firm]
        .groupby(['Year', 'Quarter'])['LDA_lobbying_expenses']
        .mean()
        .reset_index()
    )
    firm_data_avg_exp['Year_Quarter'] = firm_data_avg_exp['Year'].astype(str) + " Q" + firm_data_avg_exp['Quarter'].astype(str)

    # Average Expenses Time Series Plot
    avg_exp_fig = go.Figure()
    avg_exp_fig.add_trace(go.Scatter(
        x=firm_data_avg_exp['Year_Quarter'],
        y=firm_data_avg_exp['LDA_lobbying_expenses'],
        mode='lines+markers',
        line=dict(color='orange'),
        name='Lobbying Expenses'
    ))
    avg_exp_fig.update_layout(
        title="",
        yaxis_title="Average Lobbying Expenses per LDA",
        # yaxis_tickformat=",.2f",
        xaxis_title="Year-Quarter",
        height=400,
        margin=dict(t=20, b=40)
    )

    return freq_fig, exp_fig, avg_exp_fig

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=True, host="0.0.0.0", port=port)