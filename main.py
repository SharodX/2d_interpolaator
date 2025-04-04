# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 02:48:00 2023

@author: villu
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import pandas as pd
import io
import matplotlib.ticker as ticker
import matplotlib as mpl
from matplotlib import colors
import json

# ----- Default configuration template (adjust as needed) -----
DEFAULT_CONFIG = {
    "interpolation_type": "linear",
    "smoothing_factor": 0.0,
    "dens": 300,
    "ip_bounds_from_data": False,
    "ip_x_lower": -10.0,
    "ip_x_upper": 10.0,
    "ip_y_lower": -10.0,
    "ip_y_upper": 10.0,
    "cbar_build": "Eelkonstrueeritud",
    "colorbar_visible": True,
    "colorbar_float_format": 2,
    "colorbar_type": "viridis",
    "colorbar_divisions": 10,
    "colorbar_bounds_from_data": False,
    "colorbar_x_lower": 0.0,
    "colorbar_x_upper": 1.0,
    "cbar_color_count": 2,
    "x_axis_visible": True,
    "y_axis_visible": True,
    "axis_float_format": 1,
    "axis_step": 1.0,
    "x_axis_title": "",
    "y_axis_title": "",
    "colorbar_title": "",
    "chart_title": "",
    "draw_dpi": 300,
    "save_dpi": 300,
    "save_extension": "png",
    "file_to_draw": "Näidisfail",
}

# A simple helper for discrete colormaps
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    if isinstance(base_cmap, str) or base_cmap is None:
        return plt.cm.get_cmap(base_cmap, N)
    base = plt.colormaps.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

# Ensure session_state has all default values at startup
for key, val in DEFAULT_CONFIG.items():
    if key not in st.session_state:
        st.session_state[key] = val

min_cbars, max_cbars = 2, 10

# Create placeholders in session_state for user-defined color scales
for n in range(max_cbars + 1):
    if f"number_{n}" not in st.session_state:
        st.session_state[f"number_{n}"] = n
    if f"color_{n}" not in st.session_state:
        cmap_proxy = discrete_cmap(max_cbars)(n / max_cbars)
        st.session_state[f"color_{n}"] = mpl.colors.rgb2hex(cmap_proxy)

# --- SIDEBAR CONFIG ---
sidebar = st.sidebar

# --- Config Import/Export + Reset ---
with sidebar:
    with st.expander("Seadistuste eksport/import"):
        # Button to reset to default config
        if st.button("Lähtesta vaikeväärtustele"):
            for key, val in DEFAULT_CONFIG.items():
                st.session_state[key] = val
            st.rerun()

        # Upload config
        config_file = st.file_uploader("Lae konfiguratsioon", type="json", key="config_upload")
        if config_file:
            config = json.load(config_file)
        # Fully override existing keys
        if st.button("Lähtesta üleslaetud konfiguratsioonile"):
            for key, value in config.items():
                st.session_state[key] = value
            st.rerun()

        # Export config
        if st.button("Salvesta konfiguratsioon"):
            export_config = {
                k: v
                for k, v in st.session_state.items()
                if isinstance(v, (int, float, str, bool, list))
            }
            config_bytes = io.BytesIO(json.dumps(export_config, indent=2).encode("utf-8"))
            st.download_button(
                "Lae konfiguratsioon alla",
                data=config_bytes,
                file_name="konfiguratsioon.json",
                mime="application/json",
            )

# Which file to draw
with sidebar:
    st.radio(
        "Kuvatud fail", ["Näidisfail", "Üleslaetud"], key="file_to_draw"
    )

uploaded_file = st.file_uploader("Lae oma fail üles", type="csv")
error_placeholder = st.empty()

@st.cache_data
def load_data(sheets_url):
    csv_url = sheets_url.replace("/edit#gid=", "/export?format=csv&gid=")
    return pd.read_csv(csv_url, header=0)

# Example data
df_example = load_data(st.secrets["public_gsheets_url"])

# Choose data source
if st.session_state["file_to_draw"] == "Näidisfail":
    df = df_example
else:
    if uploaded_file is None:
        st.warning("Faili pole üles laetud")
        st.stop()
    df = pd.read_csv(uploaded_file)

# Prepare x, y, v
x = np.array(df[df.columns[0]])
y = np.array(df[df.columns[1]])
v = np.array(df[df.columns[2]])

# Fix the y_min, y_max (previously had a typo referencing x.max())
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()
v_min, v_max = v.min(), v.max()

# --- READ SIDEBAR SETTINGS ---
with sidebar:
    with st.expander("Interpoleerimise seadistus"):
        interpolation_type = st.selectbox(
            "rbf-funktsiooni tüüp",
            [
                "multiquadric",
                "inverse",
                "gaussian",
                "linear",
                "cubic",
                "quintic",
                "thin_plate",
            ],
            key="interpolation_type",
        )
        smoothing_factor = st.number_input(
            "Silumise parameeter",
            0.0,
            value=st.session_state["smoothing_factor"],
            key="smoothing_factor",
            format="%.6f",
            step=1e-6,
        )
        dens = st.slider(
            "Võrgustiku tihedus",
            10,
            1000,
            value=st.session_state["dens"],
            key="dens",
        )
        ip_bounds_from_data = st.checkbox(
            "Graafiku piirid käsitsi",
            value=st.session_state["ip_bounds_from_data"],
            key="ip_bounds_from_data",
        )
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            ip_x_lower = st.number_input(
                "x_min",
                -100.0,
                100.0,
                value=x_min,
                disabled=not ip_bounds_from_data,
                key="ip_x_lower",
            )
            ip_y_lower = st.number_input(
                "y_min",
                -100.0,
                100.0,
                value=y_min,
                disabled=not ip_bounds_from_data,
                key="ip_y_lower",
            )
        with subcol2:
            ip_x_upper = st.number_input(
                "x_max",
                ip_x_lower + 0.1,
                100.0,
                value=x_max,
                disabled=not ip_bounds_from_data,
                key="ip_x_upper",
            )
            ip_y_upper = st.number_input(
                "y_max",
                ip_y_lower + 0.1,
                100.0,
                value=y_max,
                disabled=not ip_bounds_from_data,
                key="ip_y_upper",
            )

    cbar_build = st.radio(
        "test",
        options=["Eelkonstrueeritud", "Ehitan ise"],
        label_visibility="collapsed",
        key="cbar_build",
    )
    colorbar_visible = st.checkbox("Värviskaala", key="colorbar_visible")
    colorbar_float_format = st.slider(
        "Värviskaala komakohad",
        0,
        3,
        value=st.session_state["colorbar_float_format"],
        key="colorbar_float_format",
    )

    if cbar_build == "Eelkonstrueeritud":
        freeze_default_colorbars = False
        freeze_selfbuilt_colorbars = True
    else:
        freeze_default_colorbars = True
        freeze_selfbuilt_colorbars = False

    with st.expander("Värviskaala eelkonstrueeritud"):
        colorbar_type = st.selectbox(
            "Värviskaala tüüp",
            plt.colormaps(),
            disabled=freeze_default_colorbars,
            key="colorbar_type",
        )
        colorbar_divisions = st.slider(
            "Värvijaotuste arv",
            2,
            20,
            value=st.session_state["colorbar_divisions"],
            disabled=freeze_default_colorbars,
            key="colorbar_divisions",
        )
        colorbar_bounds_from_data = st.checkbox(
            "Värviskaala piirid käsitsi",
            value=False,
            disabled=freeze_default_colorbars,
            key="colorbar_bounds_from_data",
        )
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            colorbar_x_lower = st.number_input(
                "cbar_min",
                0.0,
                10.0,
                value=v_min,
                disabled=not colorbar_bounds_from_data or freeze_default_colorbars,
                key="colorbar_x_lower",
            )
        with subcol2:
            colorbar_x_upper = st.number_input(
                "cbar_max",
                0.0,
                10.0,
                value=v_max,
                disabled=not colorbar_bounds_from_data or freeze_default_colorbars,
                key="colorbar_x_upper",
            )

    with st.expander("Värviskaala ehitan ise"):
        cbar_color_count = st.slider(
            "Värvide arv",
            min_cbars,
            max_cbars,
            disabled=freeze_selfbuilt_colorbars,
            key="cbar_color_count",
        )
        cbar_colors = []
        cbar_breakpoints = []
        for color in range(max_cbars):
            freeze_color_input = color >= cbar_color_count
            freeze_number_input = color > cbar_color_count
            a = st.number_input(
                "n",
                label_visibility="collapsed",
                key=f"number_{color}",
                disabled=freeze_number_input or freeze_selfbuilt_colorbars,
            )
            b = st.color_picker(
                "c1",
                label_visibility="collapsed",
                key=f"color_{color}",
                disabled=freeze_color_input or freeze_selfbuilt_colorbars,
            )
            if not freeze_color_input:
                cbar_colors.append(st.session_state[f"color_{color}"])
            if not freeze_number_input:
                cbar_breakpoints.append(st.session_state[f"number_{color}"])

        cmap = colors.ListedColormap(cbar_colors)
        levels = cbar_breakpoints
        if not np.all(np.diff(levels) > 0) and not freeze_selfbuilt_colorbars:
            with error_placeholder:
                st.error("Värviskaala väärtused peavad olema kasvavas järjekorras")
                st.stop()
        norm = colors.BoundaryNorm(levels, cmap.N)

    with st.expander("Telgede seadistus"):
        x_axis_visible = st.checkbox("x-telg", key="x_axis_visible")
        y_axis_visible = st.checkbox("y-telg", key="y_axis_visible")
        axis_float_format = st.slider(
            "Telgede komakohad", 0, 3, value=st.session_state["axis_float_format"], key="axis_float_format"
        )
        axis_step = st.number_input("Telje samm", 0.1, 10.0, value=st.session_state["axis_step"], key="axis_step")

    with st.expander("Graafiku tähised"):
        x_axis_title = st.text_input("x-telje pealkiri", key="x_axis_title")
        y_axis_title = st.text_input("y-telje pealkiri", key="y_axis_title")
        colorbar_title = st.text_input("Värviskaala pealkiri", key="colorbar_title")
        chart_title = st.text_input("Joonise pealkiri", key="chart_title")

    with st.expander("Salvestamise parameetrid"):
        draw_dpi = st.number_input("Kuva dpi", 60, value=st.session_state["draw_dpi"], key="draw_dpi")
        save_dpi = st.number_input("Salvestuse dpi", 60, value=st.session_state["save_dpi"], key="save_dpi")
        save_extension = st.radio("Salvestuse formaat", ["png", "svg"], key="save_extension")

# ----- Interpolation bounds -----
if st.session_state["ip_bounds_from_data"]:
    x_min, x_max = st.session_state["ip_x_lower"], st.session_state["ip_x_upper"]
    y_min, y_max = st.session_state["ip_y_lower"], st.session_state["ip_y_upper"]

# ----- Colorbar bounds -----
if st.session_state["cbar_build"] == "Eelkonstrueeritud" and st.session_state["colorbar_bounds_from_data"]:
    v_min, v_max = st.session_state["colorbar_x_lower"], st.session_state["colorbar_x_upper"]

xi, yi = np.linspace(x_min, x_max, st.session_state["dens"]), np.linspace(y_min, y_max, st.session_state["dens"])
xi, yi = np.meshgrid(xi, yi)

rbf = scipy.interpolate.Rbf(x, y, v, function=interpolation_type, smooth=smoothing_factor)
zi = rbf(xi, yi)

fig, ax = plt.subplots()

if st.session_state["cbar_build"] == "Eelkonstrueeritud":
    cmap_obj = discrete_cmap(st.session_state["colorbar_divisions"], st.session_state["colorbar_type"])
    im = ax.imshow(
        zi,
        vmin=v_min,
        vmax=v_max,
        origin="lower",
        cmap=cmap_obj,
        extent=[x_min, x_max, y_min, y_max],
        aspect="equal",
    )
    if st.session_state["colorbar_visible"]:
        fig.colorbar(
            im,
            label=st.session_state["colorbar_title"],
            ticks=np.linspace(v_min, v_max, st.session_state["colorbar_divisions"] + 1),
            format=f"%.{st.session_state['colorbar_float_format']}f",
        )
else:  # "Ehitan ise"
    im = ax.imshow(
        zi,
        origin="lower",
        cmap=cmap,
        norm=norm,
        extent=[x_min, x_max, y_min, y_max],
        aspect="equal",
    )
    if st.session_state["colorbar_visible"]:
        fig.colorbar(
            mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
            ax=ax,
            spacing="proportional",
            label=st.session_state["colorbar_title"],
            format=f"%.{st.session_state['colorbar_float_format']}f",
        )

fig.suptitle(st.session_state["chart_title"])
ax.set_xlabel(st.session_state["x_axis_title"])
ax.set_ylabel(st.session_state["y_axis_title"])

ax.xaxis.set_visible(st.session_state["x_axis_visible"])
ax.yaxis.set_visible(st.session_state["y_axis_visible"])

ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(f"%.{st.session_state['axis_float_format']}f"))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(f"%.{st.session_state['axis_float_format']}f"))

start_x, end_x = ax.get_xlim()
start_y, end_y = ax.get_ylim()

ax.xaxis.set_ticks(np.arange(start_x, end_x, st.session_state["axis_step"]))
ax.yaxis.set_ticks(np.arange(start_y, end_y, st.session_state["axis_step"]))

st.pyplot(fig, dpi=st.session_state["draw_dpi"])

# ----- Download the figure -----
fn = f"interpolated.{st.session_state['save_extension']}"
img = io.BytesIO()
plt.savefig(img, format=st.session_state["save_extension"], dpi=st.session_state["save_dpi"], bbox_inches="tight")
st.download_button("Lae pilt alla", data=img, file_name=fn, mime="image/png")

# Optionally display dataframe
st.write(df)
