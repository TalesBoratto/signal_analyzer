from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd

import streamlit as st
from scipy.io import loadmat

from data_processing import (
    process,
    amplitude_spectrum,
)
from aux_functions import (
    extract_mat_sensor_names,
    line_figure,
    error_figure,
    reconstruction_figure,
)

st.set_page_config(
    page_title="Vibration Signal Analyzer",
    layout="wide",
)

PLOT_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
}

# =============================================================================
# INTERFACE: INPUT
# =============================================================================

st.title("Vibration Signal Analyzer")
st.caption(
    "Acceleration signal integration analysis and comparisons."
)

st.sidebar.header("Input")

demo = st.sidebar.checkbox(
    "Use synthetic signal -- demonstration only",
    value=True,
    disabled=True,
)

if demo:
    fs = st.sidebar.number_input(
        "Sampling rate [Hz]",
        min_value=50.0,
        value=100.0,
        step=10.0,
    )

    time = np.arange(int(30 * fs)) / fs
    data = (
        0.3
        + 2.0 * np.sin(2.0 * np.pi * 5.0 * time)
        + 0.4 * np.sin(2.0 * np.pi * 12.0 * time)
    )

    name = "Demonstration Signal"
    source = "synthetic_demo"

else:
    uploaded = st.sidebar.file_uploader(
        "Carregar arquivo MATLAB",
        type=["mat"],
    )

    if uploaded is None:
        st.info(
            "Upload a MATLAB (.mat) file using the sidebar to begin."
        )
        st.stop()

    try:
        mat = loadmat(
            io.BytesIO(uploaded.getvalue()),
        )

    except NotImplementedError:
        st.error(
            "This file could be on MATLABv7.3 format,"
            "not yet supported by this app version."
        )
        st.stop()

    except Exception as exc:
        st.error(f"It was not possible to read the MAT file: {exc}")
        st.stop()

    data_keys = [
        key
        for key in ("data", "Data","X")
        if key in mat
    ]

    if not data_keys:
        available = [
            key
            for key in mat
            if not key.startswith("__")
        ]

        st.error(
            "File must contains a 'data' or 'Data' variable."
        )
        st.write("Variables Found:", available)
        st.stop()

    if len(data_keys) == 1:
        selected_variable = data_keys[0]
    else:
        selected_variable = st.sidebar.selectbox(
            "The file contains both data and Data. Select one:",
            options=data_keys,
        )

    matrix = np.asarray(mat[selected_variable])

    if matrix.dtype.kind not in {"i", "u", "f"}:
        st.error(
            f"Varible '{selected_variable}' should be numeric and real."
        )
        st.stop()

    # Um vetor 1D é interpretado como um único sensor.
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)

    if matrix.ndim != 2:
        st.error(
            "The data should contains 2 dimentions: "
            "lines = samples e columns = sensors."
        )
        st.stop()

    n_samples, n_sensors = matrix.shape

    if n_samples < 3 or n_sensors < 1:
        st.error(
            f"Format Found: {matrix.shape}. "
            "At least 3 samples and 1 sensor are required. "
        )
        st.stop()

    st.sidebar.caption(
        f"{n_samples:,} samples × {n_sensors} sensors"
    )


    sensor_labels = None

    if "labelshulp" in mat:
        sensor_labels = extract_mat_sensor_names(
            mat["labelshulp"],
            n_sensors=n_sensors,
        )

        if sensor_labels is None:
            st.sidebar.warning(
                "'labelshulp' do not contains a valid label "
                "per column. Sensor IDs will be used."
            )

    labels_used = sensor_labels is not None

    sensor_names = [
        (
            f"Sensor {index + 1} — {sensor_labels[index]}"
            if labels_used
            else f"Sensor {index + 1}"
        )
        for index in range(n_sensors)
    ]

    st.subheader("Sensor selection")

    selected_sensor = st.selectbox(
        "Select the sensor to analyze",
        options=list(range(n_sensors)),
        format_func=lambda index: sensor_names[index],
        key=f"sensor_selection_{uploaded.name}_{selected_variable}",
    )

    data = np.asarray(
        matrix[:, selected_sensor],
        dtype=float,
    )

    if not np.all(np.isfinite(data)):
        st.error(
            f"{sensor_names[selected_sensor]} contains NaN ou infinity. "
            "Fix the signal before analysis or select another sensor."
        )
        st.stop()

    name = sensor_names[selected_sensor]
    source = uploaded.name


    fs = st.sidebar.number_input(
        "Sampling Rate [Hz]",
        min_value=0.001,
        value=100.0,
        step=1.0,
        format="%.3f",
    )


unit = st.sidebar.text_input(
    "Acceleration unit",
    value="m/s²",
)

st.sidebar.caption(
    "The unit is a label only. No automatic conversion is performed."
)

st.sidebar.header("High-pass filter")

cutoff = st.sidebar.number_input(
    "Cutoff frequency [Hz]",
    min_value=0.000001,
    value=1.0,
    step=0.1,
    format="%.6f",
)

order = st.sidebar.number_input(
    "Butterworth order",
    min_value=1,
    max_value=10,
    value=4,
    step=1,
)

if not 0 < cutoff < fs / 2:
    st.error("Cutoff must satisfy 0 < cutoff < fs/2.")
    st.stop()

st.sidebar.caption(
    "HPF applied only to acceleration. Full record, without trimming."
)


try:
    with st.spinner("Processing signal..."):
        result = process(
            data=data,
            fs=fs,
            cutoff=cutoff,
            order=int(order),
            name=name,
            unit=unit,
        )

except (ValueError, TypeError) as exc:
    st.error(f"Processing error: {exc}")
    st.stop()



original = result["original"]
acc_dc = result["acc_dc"]
velocity = result["velocity"]
points = result["points"]

st.subheader(name)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Removed DC",
    f"{np.mean(original.data):.6g}",
    help=f"Unit: {unit}",
)
c2.metric("Samples", f"{original.size:,}")
c3.metric("Sampling rate", f"{fs:g} Hz")
c4.metric("Nominal duration", f"{original.size / fs:.3f} s")


# =============================================================================
# ACELERAÇÃO E FFT ANTES DO HPF
# =============================================================================

st.subheader("Acceleration and spectrum before high-pass filtering")

left, right = st.columns(2)

with left:
    fig = line_figure(
        x=acc_dc.time,
        y=acc_dc.data,
        title="Acceleration after DC removal — no HPF",
        x_label="Time [s]",
        y_label=f"Acceleration [{unit}]",
        color="black",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config=PLOT_CONFIG,
        key="acceleration",
    )

with right:
    frequency, amplitude = amplitude_spectrum(acc_dc)

    fig = line_figure(
        x=frequency,
        y=amplitude,
        title="Single-sided amplitude spectrum — no HPF",
        x_label="Frequency [Hz]",
        y_label=f"Amplitude [{unit}]",
        color="black",
    )

    fig.add_vline(
        x=cutoff,
        line_color="crimson",
        line_dash="dash",
        annotation_text=f"Cutoff: {cutoff:g} Hz",
        annotation_position="top right",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config=PLOT_CONFIG,
        key="spectrum",
    )

st.caption(
    "Both plots use acceleration after DC removal and before HPF. "
    "The red line indicates the selected cutoff. "
    "The displayed FFT is not high-pass filtered and uses no window."
)

# =============================================================================
# VELOCIDADE: WAVEFORM E FFT
# =============================================================================

st.subheader("Estimated velocity")

left, right = st.columns(2)

with left:
    fig = line_figure(
        x=velocity.time,
        y=velocity.data,
        title="Velocity estimated by FFT integration",
        x_label="Time [s]",
        y_label=f"Velocity [{velocity.unity}]",
        color="seagreen",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config=PLOT_CONFIG,
        key="velocity_waveform",
    )

with right:
    frequency_vel, amplitude_vel = amplitude_spectrum(velocity)

    fig = line_figure(
        x=frequency_vel,
        y=amplitude_vel,
        title="Velocity — single-sided amplitude spectrum",
        x_label="Frequency [Hz]",
        y_label=f"Amplitude [{velocity.unity}]",
        color="seagreen",
    )

    fig.add_vline(
        x=cutoff,
        line_color="crimson",
        line_dash="dash",
        annotation_text=f"Cutoff: {cutoff:g} Hz",
        annotation_position="top right",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config=PLOT_CONFIG,
        key="velocity_spectrum",
    )

st.caption(
    "The velocity spectrum is calculated from the estimated velocity, "
    "after acceleration high-pass filtering and FFT integration. "
    "No additional filter or window is applied to display this spectrum."
)

# =============================================================================
# COMPARAÇÃO DAS WAVEFORMS DE ACELERAÇÃO
# =============================================================================

st.subheader("Reference and reconstructed acceleration")

fig = reconstruction_figure(
    points=points,
    unit=unit,
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config=PLOT_CONFIG,
    key="acceleration_reconstruction_comparison",
)

st.caption(
    "The reference is acceleration after DC removal, before HPF. "
    "The reconstructed signals are obtained by differentiating "
    "the estimated velocity using FFT and finite differences."
)

st.caption(
    "Click a legend item to hide or show a curve. "
    "Double-click a legend item to isolate it. "
    "All three curves share the same acceleration axis."
)

# =============================================================================
# ACELERAÇÃO E ERROS DE RECONSTRUÇÃO
# =============================================================================

st.subheader("Acceleration and reconstruction errors")

error_options = {
    "Signed error — same acceleration axis": "signed",
    "Absolute error — same acceleration axis": "absolute",
    "Relative error — secondary axis": "relative",
}

selected_error = st.selectbox(
    "Error measure shown in the chart",
    options=list(error_options),
    index=0,
    key="error_measure_selection_v2",
)

error_type = error_options[selected_error]

if error_type == "absolute":
    st.latex(
        r"e_{\mathrm{abs}}[n] = "
        r"\left|a_{\mathrm{DC}}[n]-\hat{a}[n]\right|"
    )

    st.caption(
        "Absolute errors are nonnegative and share the acceleration "
        "axis and physical unit."
    )

elif error_type == "signed":
    st.latex(
        r"e[n] = a_{\mathrm{DC}}[n]-\hat{a}[n]"
    )

    st.caption(
        "Signed errors share the acceleration axis and physical unit. "
        "The sign follows reference minus estimate."
    )

else:
    st.latex(
        r"e_{\mathrm{rel}}[n] = "
        r"\frac{a_{\mathrm{DC}}[n]-\hat{a}[n]}"
        r"{a_{\mathrm{DC}}[n]}"
    )

    st.caption(
        "Relative errors use the secondary axis on the right. "
        "Values are dimensionless, without percentage conversion."
    )

    st.warning(
        "Relative errors can become very large near reference zero "
        "crossings. Exactly zero reference values produce NaN. "
        "No threshold, clipping or RMS normalization is applied."
    )

    zero_count = int(np.count_nonzero(acc_dc.data == 0.0))

    st.caption(
        f"Points with reference exactly equal to zero: {zero_count}"
    )

fig = error_figure(
    points=points,
    unit=unit,
    error_type=error_type,
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config=PLOT_CONFIG,
    key="combined_reconstruction_errors",
)

st.caption(
    "Click a legend item to hide or show a curve. "
    "Double-click a legend item to isolate it."
)

st.caption(
    "Changing the error measure affects only this chart. "
    "Metrics and exported pointwise results remain unchanged. "
    "The reference is acceleration after DC removal, before HPF."
)

# =============================================================================
# MÉTRICAS
# =============================================================================

st.subheader("Metric comparison")

st.write(f"Acceleration metrics — unit: {unit}")
st.dataframe(result["acceleration_metrics"])

st.write(f"Velocity metrics — unit: {velocity.unity}")
st.dataframe(result["velocity_metrics"])

st.write("Reconstruction errors relative to acceleration after DC removal")
st.dataframe(result["reconstruction_metrics"])

st.caption(
    "NRMSE is a global metric normalized by reference RMS. "
    "It is not the pointwise relative error denominator."
)

# =============================================================================
# EXPORTAÇÃO
# =============================================================================

settings = {
    "source": source,
    "signal": name,
    "fs": float(fs),
    "acceleration_unit": unit,
    "cutoff_hz": float(cutoff),
    "filter_order": int(order),
    "filter_stages": "acceleration_only",
    "trim_applied": False,
    "dc_removed": float(np.mean(original.data)),
    "relative_error": "(acc_dc - estimate) / acc_dc",
    "zero_reference_policy": "NaN",
    "fft_display": "acc_dc, no HPF, no window",
}

if not demo:
    settings.update({
        "mat_data_variable": selected_variable,
        "sensor_id": int(selected_sensor + 1),
        "sensor_index_zero_based": int(selected_sensor),
        "sensor_label": (
            sensor_labels[selected_sensor]
            if labels_used
            else None
        ),
        "labels_used": bool(labels_used),
        "n_samples": int(n_samples),
        "n_sensors": int(n_sensors),
        "matrix_orientation": "samples_by_sensors",
    })

settings["displayed_error_measure"] = error_type
settings["displayed_error_axis"] = (
    "secondary"
    if error_type == "relative"
    else "primary"
)
settings["displayed_relative_error_as_percent"] = False

buffer = io.BytesIO()

with zipfile.ZipFile(
    buffer,
    mode="w",
    compression=zipfile.ZIP_DEFLATED,
) as archive:
    archive.writestr(
        "pointwise_results.csv",
        points.to_csv(index=False),
    )

    for filename, key in (
        ("acceleration_metrics.csv", "acceleration_metrics"),
        ("velocity_metrics.csv", "velocity_metrics"),
        ("reconstruction_metrics.csv", "reconstruction_metrics"),
    ):
        archive.writestr(
            filename,
            result[key].to_csv(),
        )

    archive.writestr(
        "config.json",
        json.dumps(settings, indent=2, ensure_ascii=False),
    )

st.download_button(
    label="Download signals, errors, metrics and settings",
    data=buffer.getvalue(),
    file_name="signal_analysis_results.zip",
    mime="application/zip",
)

# =============================================================================
# INFORMAÇÕES COMPLETAS — FINAL DA PÁGINA
# =============================================================================

st.divider()

with st.expander(
    "File, signal and processing details",
    expanded=False,
):
    details = {
        "Source": source,
        "Selected signal": name,
        "Number of samples": int(original.size),
        "Sampling rate [Hz]": float(fs),
        "Sampling interval [s]": float(original.dt),
        "Nominal duration [s]": float(original.size / fs),
        "Acceleration unit": unit,
        "Removed DC": float(np.mean(original.data)),
        "High-pass cutoff [Hz]": float(cutoff),
        "Butterworth design order": int(order),
        "Filter application": "Acceleration only",
        "DC removal": "Mean subtraction",
        "Integration": "Frequency domain — FFT",
        "Differentiation methods": "FFT and finite differences",
        "Error reference": "Acceleration after DC removal, before HPF",
        "Signed error": "Reference - estimate",
        "Relative error": "(Reference - estimate) / reference",
        "Zero-reference policy": "Relative error = NaN",
        "Edge trimming": "Not applied",
        "Metrics interval": "Full record",
        "Acceleration FFT input": "After DC removal, before HPF",
        "Velocity FFT input": "Estimated velocity",
        "FFT window": "None",
    }

    if not demo:
        details.update({
            "MAT data variable": selected_variable,
            "Sensor ID / column": int(selected_sensor + 1),
            "Number of sensors in file": int(n_sensors),
            "Sensor labels": (
                "labelshulp"
                if labels_used
                else "Column IDs"
            ),
            "Matrix orientation": "Rows = samples; columns = sensors",
        })

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Property": key,
                    "Value": str(value),
                }
                for key, value in details.items()
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("#### First samples of the selected signal")

    preview_size = min(10, original.size)

    st.dataframe(
        pd.DataFrame({
            "Sample index": np.arange(preview_size),
            "Time [s]": original.time[:preview_size],
            "Original acceleration": original.data[:preview_size],
            "Acceleration after DC": acc_dc.data[:preview_size],
        }),
        hide_index=True,
        use_container_width=True,
    )