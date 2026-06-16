# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026, Sony AI Inc.
"""
Rally Analysis Plot Widget

Contains the rally analysis scatter plot matrix visualization.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PySide6.QtCore import Signal, Qt
from PySide6.QtWebEngineWidgets import QWebEngineView

import plotly.express as px


class RallyAnalysisPlot(QWidget):
    """Widget for displaying rally analysis scatter plot matrix"""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Info label
        self.info_label = QLabel("Rally Analysis: 10D Scatter Plot Matrix")
        self.info_label.setStyleSheet("QLabel { padding: 5px; background-color: #f0f0f0; border: 1px solid #ccc; font-weight: bold; }")
        layout.addWidget(self.info_label, stretch=0)

        # Web view for plotly interactive plot
        self.web_view = QWebEngineView()
        self.web_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.web_view, stretch=1)

        self.setLayout(layout)

    def plot_data(self, data_rally: pd.DataFrame):
        """
        Create rally analysis scatter plot matrix

        Args:
            data_rally: DataFrame with rally-level data containing:
                - time_from_last: array of time values
                - pos: array of positions (Nx3)
                - vel: array of velocities (Nx3)
                - spin: array of spins (Nx3)
                - rally_last_shot: 'robot' or 'human'
                - rally_winner: 'robot' or 'human'
        """
        if len(data_rally) == 0:
            self.info_label.setText("No rally data available")
            return

        # Prepare data for plotting
        plot_data = []
        for idx, rally in data_rally.iterrows():
            # Extract the 10 quantities from the last shot in the rally
            time_from_last = rally['time_from_last'][-1] if len(rally['time_from_last']) > 0 else 0
            pos = rally['pos'].reshape(-1, 3)
            vel = rally['vel'].reshape(-1, 3)
            spin = rally['spin'].reshape(-1, 3)

            # Create a combined label for serve-winner combination
            lastshot_winner_label = f"{rally['rally_last_shot']}_last_shot_{rally['rally_winner']}_win"

            plot_data.append({
                'time_from_last': time_from_last,
                'pos_x': pos[-1, 0],
                'pos_y': pos[-1, 1],
                'pos_z': pos[-1, 2],
                'vel_x': vel[-1, 0],
                'vel_y': vel[-1, 1],
                'vel_z': vel[-1, 2],
                'spin_x': spin[-1, 0],
                'spin_y': spin[-1, 1],
                'spin_z': spin[-1, 2],
                'rally_winner': rally['rally_winner'],
                'rally_last_shot': rally['rally_last_shot'],
                'lastshot_winner': lastshot_winner_label
            })

        df_plot = pd.DataFrame(plot_data)

        # Create scatter plot matrix using plotly express
        dimensions = ['time_from_last', 'pos_x', 'pos_y', 'pos_z',
             'vel_x', 'vel_y', 'vel_z', 'spin_x', 'spin_y', 'spin_z']

        # Define color map for each combination
        color_map = {
            'robot_last_shot_robot_win': '#0173B2',  # Blue
            'robot_last_shot_human_win': '#DE8F05',  # Orange
            'human_last_shot_robot_win': '#029E73',  # Green
            'human_last_shot_human_win': '#CC78BC'   # Purple
        }

        fig = px.scatter_matrix(
            df_plot,
            dimensions=dimensions,
            color='lastshot_winner',
            color_discrete_map=color_map,
            title='Rally Analysis: 10D Scatter Plot Matrix'
        )

        fig.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(size=6))

        # Convert plotly figure to HTML and display in web view
        html_string = fig.to_html(include_plotlyjs='cdn')
        self.web_view.setHtml(html_string)

        self.info_label.setText(f"Rally Analysis: {len(data_rally)} rallies visualized")
