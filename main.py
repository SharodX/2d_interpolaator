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

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    if type(base_cmap) == str or base_cmap == None:
       return plt.cm.get_cmap(base_cmap, N)        

    base = plt.colormaps.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

# st.set_page_config(layout="wide")
k = True

min_cbars, max_cbars = 2, 10
for n in range(max_cbars + 1):
    if f"number_{n}" not in st.session_state:
        st.session_state[f"number_{n}"] = n
    if f"color_{n}" not in st.session_state:
        if n == max_cbars + 1:
            pass
        cmap_proxy = discrete_cmap(max_cbars)(n/(max_cbars))
        st.session_state[f"color_{n}"] = mpl.colors.rgb2hex(cmap_proxy)

#%%

# @st.cache_data
# def plot_all_cmaps(k):
#     N_ROWS, N_COLS = 13, 13 # 13, 13 <-- for all in one figure 
#     HEIGHT, WIDTH = 7, 14

#     cmap_ids = plt.colormaps()
#     n_cmaps = len(cmap_ids)
    
#     print(f'mpl version: {mpl.__version__},\nnumber of cmaps: {n_cmaps}')
    
#     index = 0
#     while index < n_cmaps:
#         fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(WIDTH, HEIGHT))
#         for row in range(N_ROWS):
#             for col in range(N_COLS):
#                 ax = axes[row, col]
#                 cmap_id = cmap_ids[index]
#                 cmap = plt.get_cmap(cmap_id)
#                 mpl.colorbar.ColorbarBase(ax, cmap=cmap,
#                                           orientation='horizontal')
#                 ax.set_title(f"'{cmap_id}', {index}", fontsize=8)
#                 ax.tick_params(left=False, right=False, labelleft=False,
#                                labelbottom=False, bottom=False)
                
#                 last_iteration = index == n_cmaps-1
#                 if (row==N_ROWS-1 and col==N_COLS-1) or last_iteration:
#                     plt.tight_layout()
#                     #plt.savefig('colormaps'+str(index)+'.png')
#                     # plt.show()
#                     if last_iteration: return fig
#                 index += 1

# col1, col2 = st.columns(2)

# with col2:
#     st.pyplot(plot_all_cmaps(k))
#%% upload file

sidebar = st.sidebar

with sidebar:
    file_to_draw = st.radio("Kuvatud fail", ["Näidisfail", "Üleslaetud"])
    
uploaded_file = st.file_uploader("Lae oma fail üles", type ="csv")
error_placeholder = st.empty()

@st.cache_data
def load_data(sheets_url):
    csv_url = sheets_url.replace("/edit#gid=", "/export?format=csv&gid=")
    print(csv_url)
    return pd.read_csv(csv_url, header = 0)

df_example = load_data(st.secrets["public_gsheets_url"])

#%% CRE U03_214 M1

if file_to_draw == "Näidisfail":
    df = df_example
else:
    if uploaded_file == None:
        st.warning('Faili pole üles laetud')
        st.stop()
    df = pd.read_csv(uploaded_file)

#%%

x = np.array(df[df.columns[0]])
y = np.array(df[df.columns[1]])
v = np.array(df[df.columns[2]])

x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), x.max()
v_min, v_max = v.min(), v.max()

#%% Configure sidebar

with sidebar:
    with st.expander("Interpoleerimise seadistus"):
        interpolation_type = st.selectbox("rbf-funktsiooni tüüp", ['multiquadric',
                                                                   'inverse',
                                                                   'gaussian',
                                                                   'linear',
                                                                   'cubic',
                                                                   'quintic',
                                                                   'thin_plate']
                                      )
        smoothing_factor = st.number_input("Silumise parameeter", 0.0, value = 0.0, format = "%.6f", step = 1e-6  )
        dens = st.slider("Võrgustiku tihedus", 10, 1000, value = 300)
        ip_bounds_from_data = st.checkbox("Graafiku piirid käsitsi", value = False)
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            ip_x_lower = st.number_input("x_min", -100.0, 100.0, value = x_min, disabled = not ip_bounds_from_data)
            ip_y_lower = st.number_input("y_min",-100.0, 100.0, value = y_min, disabled = not ip_bounds_from_data)
        with subcol2:
            ip_x_upper = st.number_input("x_max", ip_x_lower + 0.1, 100.0, value = x_max, disabled = not ip_bounds_from_data)
            ip_y_upper = st.number_input("y_max", ip_y_lower + 0.1, 100.0, value = y_max, disabled = not ip_bounds_from_data)
                
    cbar_build = st.radio("test", options = ["Eelkonstrueeritud", "Ehitan ise"], label_visibility = "collapsed")
    colorbar_visible = st.checkbox("Värviskaala")
    colorbar_float_format = st.slider("Värviskaala komakohad", 0, 3, value = 2)
    if cbar_build == "Eelkonstrueeritud":
        freeze_default_colorbars = False
        freeze_selfbuilt_colorbars = True
    else:
        freeze_default_colorbars = True
        freeze_selfbuilt_colorbars = False
    with st.expander("Värviskaala eelkonstrueeritud"):
        colorbar_type = st.selectbox("Värviskaala tüüp", plt.colormaps(), disabled = freeze_default_colorbars)
        colorbar_divisions = st.slider("Värvijaotuste arv", 2, 20, value = 10, disabled = freeze_default_colorbars)
        colorbar_bounds_from_data = st.checkbox("Värviskaala piirid käsitsi", value = False, disabled = freeze_default_colorbars)
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            colorbar_x_lower = st.number_input("cbar_min", 0.0, 10.0, value = v_min, disabled = not colorbar_bounds_from_data or freeze_default_colorbars)
        with subcol2:
            colorbar_x_upper = st.number_input("cbar_max", 0.0, 10.0, value = v_max, disabled = not colorbar_bounds_from_data or freeze_default_colorbars)           
    with st.expander("Värviskaala ehitan ise"):
        cbar_color_count = st.slider("Värvide arv", min_cbars, max_cbars, disabled = freeze_selfbuilt_colorbars)
        cbar_colors = []
        cbar_breakpoints = []
        for color in range(max_cbars):
            freeze_color_input = False
            freeze_number_input = False
            if color >= cbar_color_count:
                freeze_color_input = True
                if color > cbar_color_count:
                    freeze_number_input = True
            a = st.number_input("n", label_visibility = "collapsed", key = f"number_{color}", disabled = freeze_number_input or freeze_selfbuilt_colorbars)
            b = st.color_picker("c1", label_visibility = "collapsed", key = f"color_{color}", disabled = freeze_color_input or freeze_selfbuilt_colorbars)
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
        x_axis_visible = st.checkbox("x-telg")
        y_axis_visible = st.checkbox("y-telg")
        axis_float_format = st.slider("Telgede komakohad", 0, 3, value = 1)
        axis_step = st.number_input("Telje samm", 0.1, 10.0, value = 1.0)
    
    with st.expander("Graafiku tähised"):
        x_axis_title = st.text_input("x-telje pealkiri")
        y_axis_title = st.text_input("y-telje pealkiri")
        colorbar_title = st.text_input("Värviskaala pealkiri")
        chart_title = st.text_input("Joonise pealkiri")
    with st.expander("Salvestamise parameetrid"):
        draw_dpi = st.number_input("Kuva dpi", 60, value = 300)
        save_dpi = st.number_input("Salvestuse dpi", 60, value = 300)
        save_extension = st.radio("Salvestuse formaat", ["png", "svg"])

#%%

if ip_bounds_from_data:
    x_min, x_max = ip_x_lower, ip_x_upper
    y_min, y_max = ip_y_lower, ip_y_upper
    
if cbar_build == "Eelkonstrueeritud" and colorbar_bounds_from_data:
    v_min, v_max = colorbar_x_lower, colorbar_x_upper

xi, yi = np.linspace(x_min, x_max, dens), np.linspace(y_min, y_max, dens)
xi, yi = np.meshgrid(xi, yi)

rbf = scipy.interpolate.Rbf(x, y, v, function = interpolation_type, smooth = smoothing_factor)
zi = rbf(xi, yi)

fig, ax = plt.subplots()

if cbar_build == "Eelkonstrueeritud":
    im = ax.imshow(zi, vmin = v_min, vmax = v_max, origin='lower', cmap=discrete_cmap(colorbar_divisions, colorbar_type),
                   extent=[x_min, x_max, y_min, y_max], aspect="equal")
    if colorbar_visible:
        fig.colorbar(im, label = colorbar_title, ticks = np.linspace(v_min, v_max, colorbar_divisions + 1), format=f"%.{colorbar_float_format}f")
elif cbar_build == "Ehitan ise":
    im = ax.imshow(zi, origin='lower', cmap=cmap, norm = norm,
                   extent=[x_min, x_max, y_min, y_max], aspect="equal")
    if colorbar_visible:
        fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm = norm), spacing = "proportional", label = colorbar_title, format=f"%.{colorbar_float_format}f")

# ax.set_title("%s" % ('some_text'))
# start, end = ax.get_ylim()
# ax.yaxis.set_ticks(np.arange(start, end, 2))
# ax.axes.xaxis.set_ticklabels([])
# ax.axes.yaxis.set_ticklabels([])

# fig.subplots_adjust(right=0.8)
# fig.colorbar(im)
# cbar_ax = fig.add_axes([0.85, 0.3, 0.02, 0.4])
fig.suptitle(chart_title)

ax.set_xlabel(x_axis_title)
ax.set_ylabel(y_axis_title)

# fig.text(0.075, 0.5, "some_text", ha='center', va='center', rotation="vertical")
# fig.text(0.45, 0.075, "some_text", ha='center', va='center')

# fig.tight_layout()

ax.xaxis.set_visible(x_axis_visible)
ax.yaxis.set_visible(y_axis_visible)

ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(f"%.{axis_float_format}f"))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(f"%.{axis_float_format}f"))


start_x, end_x = ax.get_xlim()
start_y, end_y = ax.get_ylim()


ax.xaxis.set_ticks(np.arange(start_x, end_x, axis_step))
ax.yaxis.set_ticks(np.arange(start_y, end_y, axis_step))

st.pyplot(fig, dpi = draw_dpi)
    
fn = f"interpolated.{save_extension}"
img = io.BytesIO()
plt.savefig(img, format=save_extension, dpi = save_dpi, bbox_inches='tight')



btn = st.download_button(
   label="Lae pilt alla",
   data=img,
   file_name=fn,
   mime="image/png"
)

df
# plt.savefig("test")
