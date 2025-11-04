"""
Plotly Dash Dashboard for NQ Trading Bot
Properly integrated with Flask server
"""
from dash import Dash, html, dcc, Input, Output, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc

from config import DASHBOARD_UPDATE_INTERVAL
from utils.data_handler import data_handler
from utils.indicators import TechnicalIndicators
from utils.logger import trading_logger

class TradingDashboard:
    """Dash-based trading dashboard"""
    
    def __init__(self, server=None):
        """Initialize dashboard with Flask server"""
        self.logger = trading_logger.system_logger
        self.server = server
        
        # Initialize Dash app
        if server is not None:
            self.app = Dash(
                __name__,
                server=server,
                url_base_pathname='/dashboard/',
                external_stylesheets=[dbc.themes.DARKLY],
                suppress_callback_exceptions=True
            )
            self.logger.info("Dashboard attached to Flask server")
        else:
            self.app = Dash(
                __name__,
                url_base_pathname='/dashboard/',
                external_stylesheets=[dbc.themes.DARKLY],
                suppress_callback_exceptions=True
            )
            self.logger.info("Dashboard running standalone")
        
        # Setup layout and callbacks
        self._setup_layout()
        self._setup_callbacks()
        
        self.logger.info("Trading Dashboard initialized successfully")
    
    def _setup_layout(self):
        """Define dashboard layout"""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("NQ Futures Trading Bot", className="text-center mb-4"),
                    html.H5(id='current-time', className="text-center text-muted")
                ])
            ]),
            
            # Performance Metrics Cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Total P&L", className="card-title"),
                            html.H2(id='total-pnl', className="text-success")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Win Rate", className="card-title"),
                            html.H2(id='win-rate', className="text-info")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Profit Factor", className="card-title"),
                            html.H2(id='profit-factor', className="text-warning")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Max Drawdown", className="card-title"),
                            html.H2(id='max-drawdown', className="text-danger")
                        ])
                    ])
                ], width=3),
            ], className="mb-4"),
            
            # Main Chart
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='price-chart', style={'height': '600px'})
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Signal Indicators
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Current Signals")),
                        dbc.CardBody([
                            html.Div(id='signal-indicators')
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("System Status")),
                        dbc.CardBody([
                            html.Div(id='system-status')
                        ])
                    ])
                ], width=6),
            ], className="mb-4"),
            
            # Recent Trades Table
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Recent Trades")),
                        dbc.CardBody([
                            html.Div(id='trades-table')
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Update Interval Component
            dcc.Interval(
                id='interval-component',
                interval=DASHBOARD_UPDATE_INTERVAL,
                n_intervals=0
            )
            
        ], fluid=True, style={'backgroundColor': '#1e1e1e'})
    
    def _setup_callbacks(self):
        """Setup all dashboard callbacks"""
        
        @self.app.callback(
            Output('current-time', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_time(n):
            return f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        @self.app.callback(
            [Output('total-pnl', 'children'),
             Output('win-rate', 'children'),
             Output('profit-factor', 'children'),
             Output('max-drawdown', 'children')],
            Input('interval-component', 'n_intervals')
        )
        def update_metrics(n):
            """Update performance metrics"""
            try:
                metrics = data_handler.calculate_performance_metrics()
                
                total_pnl = f"${metrics.get('total_pnl', 0):,.2f}"
                win_rate = f"{metrics.get('win_rate', 0)*100:.1f}%"
                profit_factor = f"{metrics.get('profit_factor', 0):.2f}"
                max_dd = f"{metrics.get('max_drawdown', 0)*100:.1f}%"
                
                return total_pnl, win_rate, profit_factor, max_dd
            except Exception as e:
                self.logger.error(f"Error updating metrics: {e}")
                return "$0.00", "0%", "0.00", "0%"
        
        @self.app.callback(
            Output('price-chart', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_chart(n):
            """Update main price chart"""
            try:
                df = data_handler.get_latest_bars(200)
                
                if df.empty:
                    return self._create_empty_chart()
                
                df = TechnicalIndicators.add_all_indicators(df)
                
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.6, 0.2, 0.2],
                    subplot_titles=('NQ Futures Price', 'RSI', 'Volume')
                )
                
                # Candlestick
                fig.add_trace(
                    go.Candlestick(
                        x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name='NQ',
                        increasing_line_color='#26a69a',
                        decreasing_line_color='#ef5350'
                    ),
                    row=1, col=1
                )
                
                # VWAP
                if 'vwap' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index, y=df['vwap'],
                            name='VWAP', line=dict(color='yellow', width=2)
                        ),
                        row=1, col=1
                    )
                
                # EMAs
                if 'ema_21' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index, y=df['ema_21'],
                            name='EMA 21', line=dict(color='orange', width=1.5)
                        ),
                        row=1, col=1
                    )
                
                # Bollinger Bands
                if 'bb_upper' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index, y=df['bb_upper'],
                            name='BB Upper', line=dict(color='gray', width=1, dash='dash')
                        ),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=df.index, y=df['bb_lower'],
                            name='BB Lower', line=dict(color='gray', width=1, dash='dash'),
                            fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
                        ),
                        row=1, col=1
                    )
                
                # RSI
                if 'rsi' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index, y=df['rsi'],
                            name='RSI', line=dict(color='purple', width=2)
                        ),
                        row=2, col=1
                    )
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # Volume
                colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' 
                         for i in range(len(df))]
                fig.add_trace(
                    go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=colors),
                    row=3, col=1
                )
                
                fig.update_layout(
                    title='NQ Futures - 1 Minute Chart',
                    xaxis_rangeslider_visible=False,
                    template='plotly_dark',
                    height=600,
                    showlegend=True,
                    legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0.5)')
                )
                
                fig.update_xaxes(title_text="Time", row=3, col=1)
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
                fig.update_yaxes(title_text="Volume", row=3, col=1)
                
                return fig
                
            except Exception as e:
                self.logger.error(f"Error updating chart: {e}")
                return self._create_empty_chart()
        
        @self.app.callback(
            Output('signal-indicators', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_signals(n):
            """Update current signal indicators"""
            try:
                signals = data_handler.get_all_signals()
                
                if signals.empty:
                    return html.Div("No signals generated yet", className="text-muted")
                
                latest_signals = signals.tail(5)
                signal_cards = []
                
                for _, signal in latest_signals.iterrows():
                    color = "success" if signal['signal'] == 'LONG' else "danger"
                    card = dbc.Card([
                        dbc.CardBody([
                            html.H6(f"{signal['strategy'].upper()}: {signal['signal']}", 
                                   className=f"text-{color}"),
                            html.P([
                                f"Price: {signal['price']:.2f} | ",
                                f"Confidence: {signal['confidence']*100:.0f}%"
                            ], className="mb-0 small"),
                            html.P(f"{signal['timestamp'].strftime('%H:%M:%S')}", 
                                  className="text-muted small mb-0")
                        ])
                    ], className="mb-2")
                    signal_cards.append(card)
                
                return html.Div(signal_cards)
                
            except Exception as e:
                self.logger.error(f"Error updating signals: {e}")
                return html.Div("Error loading signals", className="text-danger")
        
        @self.app.callback(
            Output('system-status', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_system_status(n):
            """Update system status"""
            try:
                df = data_handler.get_latest_bars(1)
                
                if df.empty:
                    status_color = "warning"
                    status_text = "Waiting for data..."
                    last_update = "N/A"
                else:
                    status_color = "success"
                    status_text = "Receiving data"
                    last_update = df.index[-1].strftime('%H:%M:%S')
                
                return html.Div([
                    dbc.Badge(status_text, color=status_color, className="mb-2"),
                    html.P(f"Last data: {last_update}", className="mb-0")
                ])
            except Exception as e:
                return html.Div("Error", className="text-danger")
        
        @self.app.callback(
            Output('trades-table', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_trades_table(n):
            """Update recent trades table"""
            try:
                trades = data_handler.get_all_trades()
                
                if trades.empty:
                    return html.Div("No trades executed yet", className="text-muted")
                
                recent_trades = trades.tail(10).sort_values('timestamp', ascending=False)
                
                return dash_table.DataTable(
                    data=recent_trades.to_dict('records'),
                    columns=[
                        {'name': 'Time', 'id': 'timestamp'},
                        {'name': 'Action', 'id': 'action'},
                        {'name': 'Price', 'id': 'price'},
                        {'name': 'Size', 'id': 'size'},
                        {'name': 'Signal', 'id': 'signal'},
                        {'name': 'P&L', 'id': 'pnl'},
                        {'name': 'Status', 'id': 'status'}
                    ],
                    style_cell={
                        'backgroundColor': '#2e2e2e',
                        'color': 'white',
                        'textAlign': 'center'
                    },
                    style_header={
                        'backgroundColor': '#1e1e1e',
                        'fontWeight': 'bold'
                    },
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{pnl} > 0'},
                            'color': '#26a69a'
                        },
                        {
                            'if': {'filter_query': '{pnl} < 0'},
                            'color': '#ef5350'
                        }
                    ]
                )
                
            except Exception as e:
                self.logger.error(f"Error updating trades table: {e}")
                return html.Div("Error loading trades", className="text-danger")
    
    def _create_empty_chart(self):
        """Create empty chart placeholder"""
        fig = go.Figure()
        fig.add_annotation(
            text="Waiting for data...<br>Run 'python quick_test.py' to send test data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            template='plotly_dark',
            height=600,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
