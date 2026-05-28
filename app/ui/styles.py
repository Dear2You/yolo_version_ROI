LIGHT_THEME = {
    "window_bg": "#dceaf7",
    "panel_bg": "#f4f9fe",
    "panel_border": "#1f4f7a",
    "soft_border": "#7aa2c5",
    "title_fg": "#123653",
    "text_fg": "#17324a",
    "muted_fg": "#526f87",
    "accent": "#2f6f9f",
    "button_bg": "#d4e6f6",
    "button_green": "#d4e6f6",
    "button_hover": "#bed8ef",
    "button_pressed": "#9fc2df",
    "status_bg": "#ffffff",
    "status_border": "#1f4f7a",
    "table_header": "#c3dcf2",
}

DARK_THEME = {
    "window_bg": "#112131",
    "panel_bg": "#172d42",
    "panel_border": "#9dc6ea",
    "soft_border": "#5b85a8",
    "title_fg": "#e5f3ff",
    "text_fg": "#e8f2fb",
    "muted_fg": "#aac4da",
    "accent": "#8cc8ff",
    "button_bg": "#24445f",
    "button_green": "#24445f",
    "button_hover": "#305a7e",
    "button_pressed": "#1b344a",
    "status_bg": "#0e1b29",
    "status_border": "#9dc6ea",
    "table_header": "#203c55",
}


def theme_palette(name: str):
    return DARK_THEME if name == "dark" else LIGHT_THEME


def main_window_styles(theme: dict) -> str:
    return f"""
    QMainWindow {{
        background: {theme['window_bg']};
    }}

    QWidget#centralWidget {{
        background: {theme['window_bg']};
        color: {theme['text_fg']};
        selection-background-color: {theme['accent']};
        selection-color: {theme['panel_bg']};
        font-family: "SimSun", "Times New Roman";
        font-size: 11px;
    }}

    QLabel {{
        color: {theme['text_fg']};
        background: transparent;
    }}

    QGroupBox QLabel {{
        background: transparent;
    }}

    QGroupBox {{
        border: 2px solid {theme['panel_border']};
        border-radius: 0;
        margin-top: 0;
        font-weight: 600;
        font-size: 12px;
        padding-top: 0;
        background: {theme['panel_bg']};
    }}

    QLabel#sectionLabel {{
        background: {theme['table_header']};
        border: 1px solid {theme['panel_border']};
        color: {theme['title_fg']};
        font-weight: 700;
        font-size: 11px;
        padding: 2px 4px;
    }}

    QFrame#actionFrame {{
        border: 2px solid {theme['panel_border']};
        border-radius: 0;
        background: {theme['panel_bg']};
    }}

    QStatusBar {{
        background: {theme['panel_bg']};
        color: {theme['muted_fg']};
        border-top: 1px solid {theme['soft_border']};
    }}

    QTableWidget {{
        border: 2px solid {theme['panel_border']};
        border-radius: 0;
        background: {theme['panel_bg']};
        gridline-color: {theme['soft_border']};
        color: {theme['text_fg']};
    }}

    QHeaderView::section {{
        background: {theme['table_header']};
        color: {theme['text_fg']};
        padding: 4px;
        border: 0;
        border-bottom: 1px solid {theme['soft_border']};
        font-weight: bold;
    }}

    QProgressBar {{
        border: 2px solid {theme['panel_border']};
        border-radius: 0;
        text-align: center;
        background: {theme['panel_bg']};
        color: {theme['text_fg']};
        min-height: 14px;
    }}

    QProgressBar::chunk {{
        background-color: {theme['accent']};
        border-radius: 0;
    }}

    QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox {{
        background: {theme['panel_bg']};
        border: 2px solid {theme['soft_border']};
        border-radius: 0;
        padding: 4px 6px;
        min-height: 16px;
        color: {theme['text_fg']};
    }}

    QCheckBox {{
        spacing: 8px;
        color: {theme['text_fg']};
    }}
    """


def button_style(theme: dict, background_color: str = None, border_color: str = None) -> str:
    bg = background_color or theme["button_bg"]
    border = border_color or theme["panel_border"]
    return f"""
    QPushButton {{
        background-color: {bg};
        border: 2px solid {border};
        border-radius: 0;
        padding: 4px 8px;
        font-size: 10px;
        font-weight: 600;
        min-height: 14px;
        color: {theme['text_fg']};
        font-family: "SimSun", "Times New Roman";
    }}
    QPushButton:hover {{
        background-color: {theme['button_hover']};
    }}
    QPushButton:pressed {{
        background-color: {theme['button_pressed']};
        padding: 4px 7px;
    }}
    QPushButton:disabled {{
        background-color: #cfcfcf;
        color: #7f7f7f;
        border-color: #b5b5b5;
    }}
    """


def card_style(theme: dict) -> str:
    return f"background: {theme['status_bg']}; border: 2px solid {theme['status_border']}; border-radius: 0;"


def path_label_style(theme: dict) -> str:
    return f"border:1px solid {theme['panel_border']}; background:{theme['panel_bg']}; padding: 6px; border-radius: 0;"
