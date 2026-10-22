
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

st.set_page_config(
    page_title="PhysicsLab Analyzer",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 PhysicsLab Analyzer")
st.write(
    "Experimental Physics Data Analysis "
    "using Python"
)

st.markdown(
    "Upload experimental measurements, "
    "perform linear regression, and "
    "visualize your results."
)

uploaded_file = st.file_uploader(
    "Upload CSV or Excel data",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("Experimental Data")
        st.dataframe(df, use_container_width=True)

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

        if len(numeric_columns) < 2:
            st.error(
                "Please provide at least two "
                "numeric columns."
            )
            st.stop()

        x_column = st.selectbox(
            "Select X-axis",
            numeric_columns
        )

        y_column = st.selectbox(
            "Select Y-axis",
            numeric_columns,
            index=1
        )

        if x_column == y_column:
            st.warning(
                "Select two different columns."
            )
            st.stop()

        clean = df[[x_column, y_column]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        x = clean[x_column].to_numpy(dtype=float)
        y = clean[y_column].to_numpy(dtype=float)

        if len(x) < 3 or np.unique(x).size < 2:
            st.error(
                "At least three valid measurements "
                "and two distinct X values are required."
            )
            st.stop()

        result = linregress(x, y)
        predictions = result.slope * x + result.intercept
        residuals = y - predictions
        rmse = np.sqrt(np.mean(residuals**2))

        st.subheader("Analysis Results")

        c1, c2, c3 = st.columns(3)
        c1.metric("Slope", f"{result.slope:.4f}")
        c2.metric("R²", f"{result.rvalue**2:.4f}")
        c3.metric("RMSE", f"{rmse:.4f}")

        st.write(
            f"Intercept: {result.intercept:.4f}"
        )
        st.write(
            f"Slope Standard Error: "
            f"{result.stderr:.4f}"
        )

        st.caption(
            "Standard error reflects statistical "
            "fit uncertainty, not instrument uncertainty."
        )

        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = result.slope * x_line + result.intercept

        fig, ax = plt.subplots()
        ax.scatter(x, y, label="Experimental Data")
        ax.plot(
            x_line, y_line,
            color="red", label="Linear Fit"
        )
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title("Experimental Data Analysis")
        ax.legend()
        ax.grid(alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Residual Analysis")

        fig2, ax2 = plt.subplots()
        ax2.scatter(x, residuals)
        ax2.axhline(
            0, color="red", linestyle="--"
        )
        ax2.set_xlabel(x_column)
        ax2.set_ylabel("Residual")
        ax2.set_title("Residual Plot")
        ax2.grid(alpha=0.3)

        st.pyplot(fig2)
        plt.close(fig2)

        results_csv = clean.copy()
        results_csv["Predicted"] = predictions
        results_csv["Residual"] = residuals

        st.download_button(
            "Download Analysis CSV",
            results_csv.to_csv(index=False),
            file_name="physicslab_results.csv",
            mime="text/csv"
        )

    except Exception as error:
        st.error(f"Analysis failed: {error}")

else:
    st.info(
        "Upload a dataset to begin analysis."
    )

st.divider()
st.caption("PhysicsLab Analyzer | Built with Python")
