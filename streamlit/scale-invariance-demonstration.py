# Standard Packages
from typing import List, Tuple

# Third-Party Packages
import numpy as np
from plotly.graph_objects import Contour, Figure, Scatter
import streamlit as st
from sympy.core.expr import Expr
from sympy.core.symbol import symbols
from sympy.matrices import Matrix
from sympy.matrices.dense import MutableDenseMatrix
from sympy.printing.latex import latex
from sympy.utilities import lambdify


def get_expressions(normalized: bool = True) -> Tuple[Expr, MutableDenseMatrix]:
    theta0, theta1 = symbols("theta0 theta1")
    hypothesis: Expr = theta0 ** 2 + theta1 ** 2 if normalized else theta0 ** 2 + 2 * (theta1) ** 2
    gradient: MutableDenseMatrix = (Matrix([hypothesis.diff(theta0), hypothesis.diff(theta1)]))
    return hypothesis, gradient


def get_figure(hypothesis: Expr, gradient: MutableDenseMatrix, eta0: float, origin: np.ndarray, style: str, showlabels: bool, showgradient: bool, showcomponents: bool) -> Figure:

    # Function Lambdifying
    theta0, theta1 = symbols("theta0 theta1")
    f, grad = lambdify([theta0, theta1], hypothesis), lambdify([theta0, theta1], gradient)

    # Parameter and Contour Generation
    x = y = np.arange(start=-12, stop=12, step=0.25)
    meshgrid: List[np.ndarray] = np.meshgrid(x, y)
    Z = f(meshgrid[0], meshgrid[1])
    point: np.ndarray = origin - eta0 * grad(origin[0], origin[1]).reshape(origin.shape)

    # Figure Meta Data
    markercolor = labelcolor = arrowcolor = "white" if style == "fill" else "black"

    # Figure Generation
    fig: Figure = Figure()
    fig.add_trace(Contour(x=x, y=y, z=Z, showscale=False, autocontour=False, contours={"showlabels": showlabels, "labelfont": {"color": labelcolor}, "start": 0, "end": np.max(Z), "size": 20, "coloring": style}))
    fig.add_trace(Scatter(x=[*origin[0:1]], y=[*origin[1:2]], mode="markers", marker=dict(symbol="x", color=markercolor, size=10)))
    if showgradient:
        fig.add_trace(Scatter(x=[*point[0:1]], y=[*point[1:2]], mode="markers", marker=dict(symbol="x", color=markercolor, size=10)))
        fig.add_annotation(ax=origin[0], ay=origin[1], x=point[0], y=point[1], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=4, arrowsize=1.5, arrowwidth=2, arrowcolor=arrowcolor)
    if showcomponents:
        fig.add_annotation(ax=origin[0], ay=origin[1] - 0.5, x=origin[0], y=point[1], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=4, arrowsize=1.5, arrowwidth=2, arrowcolor=arrowcolor, opacity=0.5)
        fig.add_annotation(ax=origin[0], ay=origin[1], x=point[0], y=origin[1], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=4, arrowsize=1.5, arrowwidth=2, arrowcolor=arrowcolor, opacity=0.5)
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
    fig.update_xaxes(title="Parameter 0")
    fig.update_yaxes(title="Parameter 1")

    return fig


if __name__ == "__main__":

    # Initialization
    st.beta_set_page_config(page_title="Scale Invariance Demonstration", layout="wide", initial_sidebar_state="expanded")
    theta0, theta1 = symbols("theta0 theta1")
    origin: np.ndarray = np.array([-5, -5])

    # Sidebar
    st.sidebar.title("Scale Invariance of the Gradient Descent")
    st.sidebar.subheader("")
    eta0_left: float = st.sidebar.number_input(label="Learning Rate (Left Plot)", min_value=0.05, max_value=1.0, value=1.0, step=0.05)
    eta0_right: float = st.sidebar.number_input(label="Learning Rate (Right Plot)", min_value=0.05, max_value=0.5, value=0.5, step=0.05)
    st.sidebar.subheader("Plot Configuration")
    style: str = st.sidebar.selectbox(label="Contour Coloring", options=["fill", "lines"], format_func=lambda x: {"fill": "Coloring between Contour Lines", "lines": "Coloring on Contour Lines"}[x])
    showlabels: bool = st.sidebar.selectbox(label="Contour Levels", options=[True, False], format_func=lambda x: {True: "Display Contour Labels", False: "Hide Contour Labels"}[x])
    showgradient: bool = st.sidebar.selectbox(label="Gradient Vector", options=[True, False], format_func=lambda x: {True: "Display Gradient Vector", False: "Hide Gradient Vector"}[x])
    showcomponents: bool = st.sidebar.selectbox(label="Vector Components", options=[True, False], format_func=lambda x: {True: "Display Vector Components", False: "Hide Vector Components"}[x])
    showcalcs: bool = st.sidebar.selectbox(label="Calculations", options=[True, False], format_func=lambda x: {True: "Display Calculations", False: "Hide Calculations"}[x])

    # Page Layout
    st.header("Visualizations")
    left, right = st.beta_columns(spec=[0.5, 0.5])

    hypothesis, gradient = get_expressions(normalized=True)
    fig: Figure = get_figure(hypothesis=hypothesis, gradient=gradient, eta0=eta0_left, origin=origin, style=style, showlabels=showlabels, showgradient=showgradient, showcomponents=showcomponents)
    left.plotly_chart(fig, use_container_width=True)

    hypothesis, gradient = get_expressions(normalized=False)
    fig: Figure = get_figure(hypothesis=hypothesis, gradient=gradient, eta0=eta0_right, origin=origin, style=style, showlabels=showlabels, showgradient=showgradient, showcomponents=showcomponents)
    right.plotly_chart(fig, use_container_width=True)

    if showcalcs:
        st.header("Calculations")
        left, right = st.beta_columns(spec=[0.5, 0.5])
        hypothesis, gradient = get_expressions(normalized=True)
        left.markdown(f"$f(\\theta) = {latex(hypothesis)}, \\nabla f(\\theta) = {latex(gradient)}, \\theta = {latex(Matrix([origin[0], origin[1]]))}$")
        left.markdown(f"$\\theta_{{t+1}} = \\theta_t - \eta\\nabla f(\\theta_{{t}}) = {latex(Matrix(origin))} - {eta0_left:.2f} \\times {latex(gradient.subs([(theta0, origin[0]), (theta1, origin[1])]))} = {latex(Matrix(origin) - np.round(eta0_left, 2) * gradient.subs([(theta0, origin[0]), (theta1, origin[1])]))}$")
        hypothesis, gradient = get_expressions(normalized=False)
        right.markdown(f"$f(\\theta) = {latex(hypothesis)}, \\nabla f(\\theta) = {latex(gradient)}, \\theta = {latex(Matrix([origin[0], origin[1]]))}$")
        right.markdown(f"$\\theta_{{t+1}} = \\theta_t - \eta\\nabla f(\\theta_{{t}}) = {latex(Matrix(origin))} - {eta0_right:.2f} \\times {latex(gradient.subs([(theta0, origin[0]), (theta1, origin[1])]))} = {latex(Matrix(origin) - np.round(eta0_right, 2) * gradient.subs([(theta0, origin[0]), (theta1, origin[1])]))}$")


