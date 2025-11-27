import os
import pandas as pd
import numpy as np
import colorsys

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# ======================
# 1. DATA LOADING & PREP
# ======================
df = pd.read_csv("Motor_Vehicle_Collisions_Crashes.csv", low_memory=False)

df["CRASH_HOUR"] = pd.to_datetime(df["CRASH TIME"], format="%H:%M", errors="coerce").dt.hour
df = df.dropna(subset=["LATITUDE", "LONGITUDE"])

df["TOTAL_INJURED"] = df["NUMBER OF PERSONS INJURED"].fillna(0)
df["TOTAL_KILLED"] = df["NUMBER OF PERSONS KILLED"].fillna(0)

factor_cols = [
    "CONTRIBUTING FACTOR VEHICLE 1",
    "CONTRIBUTING FACTOR VEHICLE 2",
    "CONTRIBUTING FACTOR VEHICLE 3",
    "CONTRIBUTING FACTOR VEHICLE 4",
    "CONTRIBUTING FACTOR VEHICLE 5",
]

# Short label mapping for contributing factors
factor_mapping = {
    "Accelerator Defective": "Accel Defect",
    "Aggressive Driving/Road Rage": "Aggressive/Road Rage",
    "Alcohol Involvement": "Alcohol",
    "Animals Action": "Animals",
    "Backing Unsafely": "Backing Unsafe",
    "Brakes Defective": "Brake Defect",
    "Cell Phone (hand-Held)": "Phone (Handheld)",
    "Cell Phone (hands-free)": "Phone (Hands-free)",
    "Driver Inattention/Distraction": "Driver Distracted",
    "Driver Inexperience": "Inexperienced",
    "Driverless/Runaway Vehicle": "Runaway Vehicle",
    "Drugs (illegal)": "Drugs",
    "Eating or Drinking": "Eating/Drinking",
    "Failure to Keep Right": "Didn't Keep Right",
    "Failure to Yield Right-of-Way": "Didn't Yield",
    "Fatigued/Drowsy": "Fatigued",
    "Fell Asleep": "Fell Asleep",
    "Following Too Closely": "Tailgating",
    "Glare": "Glare",
    "Headlights Defective": "Headlight Defect",
    "Illnes": "Illness",
    "Lane Marking Improper/Inadequate": "Bad Lane Markings",
    "Listening/Using Headphones": "Using Headphones",
    "Lost Consciousness": "Lost Consciousness",
    "Obstruction/Debris": "Obstruction",
    "Other Electronic Device": "Other Device",
    "Other Lighting Defects": "Lighting Defect",
    "Other Vehicular": "Other Vehicular",
    "Outside Car Distraction": "Outside Distract.",
    "Oversized Vehicle": "Oversized",
    "Passenger Distraction": "Passenger Distract.",
    "Passing Too Closely": "Close Passing",
    "Passing or Lane Usage Improper": "Bad Lane Use",
    "Pavement Defective": "Pavement Defect",
    "Pavement Slippery": "Slippery Pavement",
    "Pedestrian/Bicyclist/Other Pedestrian Error/Confusion": "Ped/Bike Conf.",
    "Physical Disability": "Disability",
    "Prescription Medication": "Medication",
    "Reaction to Uninvolved Vehicle": "Reacted to Other",
    "Shoulders Defective/Improper": "Bad Shoulder",
    "Steering Failure": "Steering Failure",
    "Texting": "Texting",
    "Tinted Windows": "Tinted",
    "Tire Failure/Inadequate": "Tire Failure",
    "Tow Hitch Defective": "Tow Hitch",
    "Traffic Control Device Improper/Non-Working": "Bad Signal",
    "Traffic Control Disregarded": "Ignored Signal",
    "Turning Improperly": "Bad Turn",
    "Unsafe Lane Changing": "Unsafe Lane Change",
    "Unsafe Speed": "Unsafe Speed",
    "Using On Board Navigation Device": "Using Nav",
    "Vehicle Vandalism": "Vandalism",
    "View Obstructed/Limited": "View Blocked",
    "Windshield Inadequate": "Bad Windshield",
}

df["TOP_FACTOR"] = df[factor_cols].bfill(axis=1).iloc[:, 0]
df["TOP_FACTOR_SHORT"] = df["TOP_FACTOR"].map(factor_mapping).fillna(df["TOP_FACTOR"])

vehicle_cols = [
    "VEHICLE TYPE CODE 1",
    "VEHICLE TYPE CODE 2",
    "VEHICLE TYPE CODE 3",
    "VEHICLE TYPE CODE 4",
    "VEHICLE TYPE CODE 5",
]
vehicle_types = sorted(list({v for col in vehicle_cols for v in df[col].dropna().unique()}))
borough_options = sorted(df["BOROUGH"].dropna().unique())

def classify_vehicle(v):
    if pd.isna(v):
        return "other"
    
    v_low = str(v).lower().strip()

    # MOTORCYCLE CATEGORY
    motorcycle_keywords = [
        "motorcycle", "motorbike", "scooter", "moped", "dirt", "bike",
        "bicycle", "citibike", "mini", "hover", "skate", "unic", "one wheel",
        "e-bike", "ebike", "e bike", "e-scooter", "escooter", "e scooter",
        "kick", "stand", "razor"
    ]
    if any(k in v_low for k in motorcycle_keywords):
        return "motorcycle"

    # TRUCK CATEGORY
    truck_keywords = [
        "truck", "van", "bus", "ambul", "fire", "fdny", "usps", "box",
        "freight", "dump", "tractor", "semi", "delivery", "tow",
        "sweep", "cement", "mixer", "fork", "lift", "backhoe",
        "construction", "cargo", "commercial", "flat", "pick", "pickup",
        "uhaul", "sanitation", "loader", "bobcat", "plow", "snow",
        "armored", "atv", "toolcat"
    ]
    if any(k in v_low for k in truck_keywords):
        return "truck"

    # TRUE CAR CATEGORY (strict definitions)
    car_keywords = [
        "sedan", "station wagon", "sport utility", "suv", "suburban",
        "passenger", "4 dr", "2 dr", "coupe", "convertible", "hatch",
        "minivan", "wagon"
    ]
    if any(k in v_low for k in car_keywords):
        return "car"

    # EVERYTHING ELSE → OTHER
    return "other"

for col in vehicle_cols:
    df[col + "_CATEGORY"] = df[col].apply(classify_vehicle)

# ======================
# 2. APP & LAYOUT
# ======================
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
        /* ================================
           GLOBAL FILTER THEME — #fee2e2
           ================================ */

        /* Dropdown background + border */
        .Select-menu-outer,
        .Select-value-label,
        .Select--multi .Select-value {
            background-color: #fee2e2 !important;
            border-color: #b91c1c !important;
            color: #333 !important;
        }

        .Select-control {
            background-color: #f8f9fa !important;
        }

        /* Multi-select value "pills" */
        .Select-value {
            background-color: #f8f9fa !important;
            border: 1px solid #b91c1c !important;
            color: #333 !important;
        }

        /* Dropdown arrow + remove button */
        .Select-arrow-zone,
        .Select-clear-zone {
            color: #b91c1c !important;
        }

        /* Focus styling */
        .is-focused .Select-control {
            border-color: #b91c1c !important;
            box-shadow: 0 0 0 2px rgba(185, 28, 28, 0.3) !important;
        }

        /* Menu body */
        .Select-menu-outer {
            background-color: #fee2e2 !important;
        }

        /* Dropdown options */
        .VirtualizedSelectOption {
            background-color: #fee2e2 !important;
            color: #333 !important;
        }

        .VirtualizedSelectOption:hover {
            background-color: #fca5a5 !important;
        }

        /* ================================
           CHECKLIST SWITCHES
           ================================ */

        .custom-control-input:checked ~ .custom-control-label::before {
            background-color: #b91c1c !important;
            border-color: #b91c1c !important;
        }

        .custom-switch .custom-control-label::before {
            background-color: #fee2e2 !important;
            border: 1px solid #b91c1c !important;
        }

        .custom-switch .custom-control-label:hover::before {
            background-color: #fca5a5 !important;
        }

        /* ================================
           RANGE SLIDER (dcc.RangeSlider)
           ================================ */

        .rc-slider-track {
            background-color: #fee2e2 !important;
        }

        .rc-slider-handle {
            border: solid 2px #b91c1c !important;
        }

        .rc-slider-handle:active {
            box-shadow: 0 0 5px #fee2e2;
        }

        .rc-slider-dot-active {
            border-color: #b91c1c !important;
            background: #b91c1c !important;
        }

        .rc-slider-rail {
            background-color: #fee2e2 !important;
        }
        .form-check-input:checked {
            background-color: #b91c1c;
            border-color: #fff;
        }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

gradient_bg = {
    "background": "linear-gradient(135deg, #f3f4f6, #e5e7eb)",
    "minHeight": "100vh",
    "padding": "24px",
    "fontFamily": "Poppins, sans-serif",
}

card_style = {
    "borderRadius": "16px",
    "padding": "18px",
    "backgroundColor": "#f8f9fa",
    "border": "1px solid #d1d5db",
}

kpi_card = {
    "borderRadius": "16px",
    "padding": "18px",
    "backgroundColor": "white",
    "textAlign": "center",
    "border": "1px solid #d1d5db",
    "height": "80px",
    "display": "flex",
    "flexDirection": "column",
    "justifyContent": "center",
}

title_style = {"fontSize": "22px", "fontWeight": "600"}

app.layout = dbc.Container(
    fluid=True,
    style=gradient_bg,
    children=[

        # ======================
        # HEADER
        # ======================
        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.Div(
                            "NYC Motor Vehicle Crashes Dashboard",
                            style={
                                "fontSize": "34px",
                                "fontWeight": "800",
                                "letterSpacing": "1px",
                                "background": "linear-gradient(90deg, #b91c1c, #f87171)",
                                "-webkit-background-clip": "text",
                                "color": "transparent",
                                "textAlign": "center",
                                "marginBottom": "4px",
                            }
                        ),
                        html.Div(
                            "INSIGHTS ON CRASHES, INJURIES & RISK FACTORS",
                            style={
                                "fontSize": "14px",
                                "fontWeight": "700",
                                "color": "#555",
                                "textAlign": "center",
                                "marginTop": "-6px",
                                "letterSpacing": "2px",
                            }
                        ),
                    ]
                ),
                width=12
            ),
            className="mb-4"
        ),

        # ======================
        # FILTERS (Horizontal)
        # ======================
        dbc.Card(
            html.Div(
                [
                    # Borough
                    html.Div(
                        [
                            html.Label("Borough", style={"fontWeight": "600", "fontSize": "12px"}),
                            dcc.Dropdown(
                                id="borough-filter",
                                options=[{"label": b, "value": b} for b in borough_options],
                                multi=True,
                                placeholder="Select borough(s)",
                                style={"minWidth": "180px"}
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "column", "gap": "4px"}
                    ),
        
                    # Hour Slider
                    html.Div(
                        [
                            html.Label("Hour of Day", style={"fontWeight": "600", "fontSize": "12px"}),
                            dcc.RangeSlider(
                                id="hour-filter",
                                min=0,
                                max=23,
                                step=1,
                                value=[0, 23],
                                marks={
                                    0: "12am",
                                    6: "6am",
                                    12: "12pm",
                                    18: "6pm",
                                    23: "11pm",
                                },
                                tooltip={"placement": "bottom", "always_visible": False},
                                allowCross=False,
                            ),
                        ],
                        style={
                            "flexGrow": "1",
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "4px",
                            "padding": "0 20px"
                        }
                    ),
        
                    # Vehicle Category
                    html.Div(
                        [
                            html.Label("Vehicle Category", style={"fontWeight": "600", "fontSize": "12px"}),
                            dbc.Checklist(
                                id="vehicle-filter",
                                options=[
                                    {"label": "Cars", "value": "car"},
                                    {"label": "Motorcycles", "value": "motorcycle"},
                                    {"label": "Trucks / Vans", "value": "truck"},
                                    {"label": "Other", "value": "other"},
                                ],
                                value=[],
                                inline=True,
                                switch=True,
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "column", "gap": "4px"}
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "width": "100%",
                    "gap": "40px"
                }
            ),
            style={**card_style, "marginBottom": "25px"},
        ),


        # ======================
        # KPI ROW (Full Width)
        # ======================
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            html.Div("TOTAL COLLISIONS", 
                                     style={"fontWeight":"600","fontSize":"10px",
                                            "letterSpacing":"3px","textAlign":"center","color":"#919191"}),
                            html.Div(id="ban-total-collisions", 
                                     style={"fontSize":"28px","fontWeight":"700","textAlign":"center"}),
                        ],
                        style=kpi_card,
                    ),
                    md=3
                ),

                dbc.Col(
                    dbc.Card(
                        [
                            html.Div("TOTAL INJURIES", 
                                     style={"fontWeight":"600","fontSize":"10px",
                                            "letterSpacing":"3px","textAlign":"center","color":"#919191"}),
                            html.Div(id="ban-total-injuries", 
                                     style={"fontSize":"28px","fontWeight":"700","textAlign":"center"}),
                        ],
                        style=kpi_card,
                    ),
                    md=3
                ),

                dbc.Col(
                    dbc.Card(
                        [
                            html.Div("TOTAL FATALITIES", 
                                     style={"fontWeight":"600","fontSize":"10px",
                                            "letterSpacing":"3px","textAlign":"center","color":"#919191"}),
                            html.Div(id="ban-total-fatalities", 
                                     style={"fontSize":"28px","fontWeight":"700","textAlign":"center"}),
                        ],
                        style=kpi_card,
                    ),
                    md=3
                ),

                dbc.Col(
                    dbc.Card(
                        [
                            html.Div("TOP 5 CONTRIBUTING FACTORS", 
                                     style={"fontWeight":"600","fontSize":"10px",
                                            "letterSpacing":"3px","textAlign":"center","color":"#919191"}),
                            html.Div(id="ban-top-factor", 
                                     style={"fontSize":"22px","fontWeight":"700","textAlign":"center"}),
                        ],
                        style=kpi_card,
                    ),
                    md=3
                ),
            ],
            className="g-3 mb-4",
        ),

        # ======================
        # ROW 1 — MAPS
        # ======================
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                            [
                                html.Div(
                                    "Crashes by Hour of Day",
                                    style={**title_style, "marginBottom": "5px"}
                                ),
                                dcc.Graph(
                                    id="map-fig-hour",
                                    config={"displayModeBar": False},
                                    style={"height": "380px"}
                                ),
                            ],
                            style={**card_style, "padding": "20px"}
                        ),
                    md=6
                ),

                dbc.Col(
                    dbc.Card(
                        [
                            html.Div(
                                "Worst Crash Hotspots", 
                                style={**title_style, "marginBottom": "5px"}
                            ),
                            dcc.Graph(
                                    id="map-fig-hotspots",
                                    config={"displayModeBar": False},
                                    style={"height": "385px", "overflow": "hidden"}
                                ),
                        ],
                        style=card_style,
                    ),
                    md=6
                ),
            ],
            className="g-3 mb-4",
        ),

        # ======================
        # ROW 2 — FACTORS & USER INJURIES
        # ======================
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            html.Div("Top 5 Contributing Factors", style=title_style),
                            dcc.Graph(id="factor-bar-fig", config={"displayModeBar": False}),
                        ],
                        style={**card_style,},
                    ),
                    md=6
                ),

                dbc.Col(
                    dbc.Card(
                        [
                            html.Div("Injuries by Boroughs", style=title_style),
                            dcc.Graph(id="user-type-fig", config={"displayModeBar": False}),
                        ],
                        style=card_style,
                    ),
                    md=6
                ),
            ],
            className="g-3"
        ),
    ]
)

# ======================
# 3. CALLBACK
# ======================
@app.callback(
    [
        Output("ban-total-collisions", "children"),
        Output("ban-total-injuries", "children"),
        Output("ban-total-fatalities", "children"),
        Output("ban-top-factor", "children"),
        Output("map-fig-hotspots", "figure"),
        Output("map-fig-hour", "figure"),
        Output("factor-bar-fig", "figure"),
        Output("user-type-fig", "figure"),
    ],
    [
        Input("borough-filter", "value"),
        Input("hour-filter", "value"),
        Input("vehicle-filter", "value"),
    ],
)
def update_dashboard(selected_boroughs, selected_hours, selected_vehicles):
    dff = df.copy()

    # Filters
    if selected_boroughs:
        dff = dff[dff["BOROUGH"].isin(selected_boroughs)]

    if selected_hours:
        hmin, hmax = selected_hours
        dff = dff[dff["CRASH_HOUR"].between(hmin, hmax, inclusive="both")]

    # Vehicle filter (multi-select checklist)
    if selected_vehicles:
    
        category_cols = [col + "_CATEGORY" for col in vehicle_cols]
    
        # Row matches if ANY selected category appears in ANY vehicle column
        mask = dff[category_cols].apply(
            lambda row: any(cat in row.values for cat in selected_vehicles),
            axis=1
        )
    
        dff = dff[mask]


    if dff.empty:
        empty = go.Figure()
        empty.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis={"visible": False}, yaxis={"visible": False},
            annotations=[dict(text="No data for selected filters", x=0.5, y=0.5, showarrow=False)]
        )
        return ("0","0","0","N/A",empty,empty,empty,empty)

    # KPIs
    total_collisions = f"{len(dff):,}"
    total_injuries = f"{dff['TOTAL_INJURED'].sum():,}"
    total_fatalities = f"{dff['TOTAL_KILLED'].sum():,}"
    top_factor = dff["TOP_FACTOR_SHORT"].value_counts().idxmax()

    # ======================
    # Hotspots Heatmap
    # ======================
    dff_map = dff.sample(min(5000, len(dff)), random_state=42)
    
    fig_hotspots = px.density_mapbox(
        dff_map,
        lat="LATITUDE",
        lon="LONGITUDE",
        z="TOTAL_INJURED",
        radius=20,
        center={"lat": 40.7050, "lon": -73.9700},
        zoom=10,
        height=450,
        color_continuous_scale=[(0, "#fee2e2"), (1, "#b91c1c")],
        range_color=(0, 10),
    )
    
    # Invisible scatter layer for custom tooltips
    fig_hotspots.add_scattermapbox(
        lat=dff_map["LATITUDE"],
        lon=dff_map["LONGITUDE"],
        mode="markers",
        marker=dict(size=8, color="rgba(0,0,0,0)"),
        hovertemplate=(
            "<b>Street:</b> %{customdata[0]}<br>"
            "<b>Borough:</b> %{customdata[1]}<br>"
            "<b>Total Injured:</b> %{customdata[2]}<extra></extra>"
        ),
        customdata=np.stack([
            dff_map["ON STREET NAME"].fillna("Unknown"),
            dff_map["BOROUGH"].fillna("Unknown"),
            dff_map["TOTAL_INJURED"].fillna(0),
        ], axis=-1)
    )
    
    fig_hotspots.update_layout(
        mapbox_style="open-street-map",
        autosize=False,
        height=450,
        uirevision="constant",
        margin=dict(l=0, r=0, t=0, b=0),
        mapbox=dict(
            center={"lat": 40.7050, "lon": -73.9700},
            zoom=10,
        ),
    
        # Hide injured scale bar
        coloraxis_showscale=False,
    
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="#f8f9fa",
            bordercolor="#333333",
            font=dict(color="#333333", size=14),
            align="left",
            namelength=-1,
        ),
    )

    # ======================
    # Hourly Line Graph
    # ======================
    def hour_to_label(h):
        if h == 0:
            return "12am"
        elif h < 12:
            return f"{h}am"
        elif h == 12:
            return "12pm"
        else:
            return f"{h-12}pm"
    
    hour_group = (
        dff.groupby("CRASH_HOUR")
           .size()
           .reindex(range(24), fill_value=0)
           .reset_index()
    )
    hour_group.columns = ["CRASH_HOUR", "COUNT"]
    hour_group["LABEL"] = hour_group["CRASH_HOUR"].apply(hour_to_label)
    
    # Build Y-axis ticks: 1K, 2K, 3K…
    max_y = hour_group["COUNT"].max()
    yticks = list(range(0, int(max_y) + 1000, 1000))
    yticklabels = [f"{int(v/1000)}K" if v != 0 else "0" for v in yticks]
    
    # Curve line
    fig_hour = px.line(
        hour_group,
        x="CRASH_HOUR",
        y="COUNT",
        markers=False,
        line_shape="spline",
    )
    
    # Hover text
    fig_hour.update_traces(
        hovertemplate="<b>Crash Hour:</b> %{customdata}<br>"
                      "<b>Number of Crashes:</b> %{y}<extra></extra>",
        customdata=hour_group["LABEL"],
        line=dict(width=3, color="#b91c1c"),
    )
    
    # Styling
    fig_hour.update_layout(
        xaxis=dict(
            title="Hour of Day",
            tickmode="array",
            tickvals=[0, 4, 8, 12, 16, 20, 23],
            ticktext=["12am", "4am", "8am", "12pm", "4pm", "8pm", "12am"],
            showline=True,
            linecolor="#333",
            gridcolor="rgba(0,0,0,0.07)",
            linewidth=1,
            range=[-0.5, 23.5],
        ),
    
        yaxis=dict(
            title="Number of Crashes",
            showline=True,
            linecolor="#333",
            gridcolor="rgba(0,0,0,0.07)",
            zeroline=False,
            tickvals=yticks,
            ticktext=yticklabels,
        ),
    
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#333", size=12),
    
        margin=dict(l=70, r=30, t=30, b=30),

        hoverlabel=dict(
            bgcolor="#f8f9fa",
            bordercolor="#333333",
            font=dict(color="#333333", size=14),
            align="left",
            namelength=-1,
        ),
    )

    # ======================
    # Factor Treemap (fixed + sorted)
    # ======================
    factor_counts = (
        dff["TOP_FACTOR_SHORT"]
        .value_counts()
        .head()
        .reset_index(name="Count")
        .rename(columns={"index": "TOP_FACTOR_SHORT"})
    )
    
    # Ensure descending order so rank 1 = darkest
    factor_counts = factor_counts.sort_values("Count", ascending=False).reset_index(drop=True)
    
    # Normalize
    max_c = factor_counts["Count"].max()
    min_c = factor_counts["Count"].min()
    factor_counts["NORM"] = (factor_counts["Count"] - min_c) / (max_c - min_c + 1e-9)
    
    # Gradient function
    def gradient_color(v):
        import colorsys
        h = 0.0
        s = 0.75
        l = 0.85 - 0.45 * v    # v=1 => dark, v=0 => light
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
    
    factor_counts["COLOR"] = factor_counts["NORM"].apply(gradient_color)
    
    fig_factor = px.treemap(
        factor_counts,
        path=["TOP_FACTOR_SHORT"],
        values="Count",
        color="COLOR",
    )
    
    fig_factor.update_traces(
        marker=dict(
            colors=factor_counts["COLOR"],
            line=dict(width=0)
        ),
        texttemplate="%{label}<br>%{value:,}",
        hovertemplate=(
            "<b>Reason:</b> %{label}<br>"
            "<b>Number of Injuries:</b> %{value:,}<extra></extra>"
        ),
    )
    
    fig_factor.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#f8f9fa",
        plot_bgcolor="#f8f9fa",
        coloraxis_showscale=False,
        hoverlabel=dict(
            bgcolor="#f8f9fa",
            bordercolor="#333",
            font=dict(color="#333", size=13),
            align="left",
            namelength=-1,
        ),
    )

    # ======================
    # Injuries by Borough
    # ======================
    user_group = dff.groupby("BOROUGH", dropna=False)[
        [
            "NUMBER OF PEDESTRIANS INJURED",
            "NUMBER OF CYCLIST INJURED",
            "NUMBER OF MOTORIST INJURED",
            "TOTAL_INJURED",
        ]
    ].sum().reset_index()
    
    user_group["BOROUGH"] = user_group["BOROUGH"].fillna("Unknown")
    
    # Percentage metric
    user_group["PEDESTRIAN_SHARE"] = np.where(
        user_group["TOTAL_INJURED"] > 0,
        (user_group["NUMBER OF PEDESTRIANS INJURED"] / user_group["TOTAL_INJURED"]) * 100,
        0,
    )
    
    fig_combined = go.Figure()
    
    # ======================
    # Bars (base rectangles)
    # ======================
    fig_combined.add_bar(
        x=user_group["BOROUGH"],
        y=user_group["TOTAL_INJURED"],
        name="Total Injuries",
        marker=dict(color="#fee2e2", line=dict(width=0)),
        hovertemplate="<b>Borough:</b> %{x}<br><b>Total Injuries:</b> %{y:,}<extra></extra>",
    )
    
    # ======================
    # Add rounded top corners
    # ======================
    bar_width = 0.6
    radius = 0.25
    
    for i, row in user_group.iterrows():
        x_center = i
        height = row["TOTAL_INJURED"]
    
        x0 = x_center - bar_width/2
        x1 = x_center + bar_width/2
    
        # Rounded top as a shape
        fig_combined.add_shape(
            type="rect",
            x0=x0, y0=height - radius,
            x1=x1, y1=height,
            xref="x", yref="y",
            fillcolor="#fee2e2",
            line=dict(width=0),
            layer="above",
            # This creates the curved top using path
            path=f"M {x0} {height-radius} "
                 f"L {x1} {height-radius} "
                 f"Q {x_center} {height} {x0} {height-radius} Z"
        )
    
    # ======================
    # Line (Pedestrian Share)
    # ======================
    fig_combined.add_trace(
        go.Scatter(
            x=user_group["BOROUGH"],
            y=user_group["PEDESTRIAN_SHARE"],
            name="% Pedestrian Injuries",
            mode="lines+markers",
            marker=dict(size=7, color="#b91c1c"),
            line=dict(width=3, color="#b91c1c"),
            yaxis="y2",
            hovertemplate="<b>Borough:</b> %{x}<br><b>% Pedestrians Injured:</b> %{y:.2f}%<extra></extra>",
        )
    )
    
    # Safe max
    max_share = user_group["PEDESTRIAN_SHARE"].max()
    if not np.isfinite(max_share) or max_share <= 0:
        max_share = 100
    
    # ======================
    # Layout
    # ======================
    fig_combined.update_layout(
        paper_bgcolor="#f8f9fa",
        plot_bgcolor="#f8f9fa",
        margin=dict(l=30, r=30, t=10, b=30),
    
        xaxis=dict(showgrid=False),
    
        yaxis=dict(
            title="Total Injuries",
            gridcolor="rgba(0,0,0,0.05)",
            zeroline=False,
        ),
    
        yaxis2=dict(
            title="% Pedestrian Injuries",
            overlaying="y",
            side="right",
            range=[0, max_share * 1.25],
            showgrid=False,
        ),
    
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.05
        ),

        hoverlabel=dict(
            bgcolor="#f8f9fa",
            bordercolor="#333",
            font=dict(color="#333"),
        ),
    )

    return (
        total_collisions,
        total_injuries,
        total_fatalities,
        top_factor,
        fig_hotspots,
        fig_hour,
        fig_factor,
        fig_combined,
    )

server = app.server 

if __name__ == "__main__":
    app.run_server(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        debug=False
    )