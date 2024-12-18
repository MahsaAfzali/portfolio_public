import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import plotly.express as px
from sklearn.metrics import roc_curve, auc


def plot_data_distribution(df: pd.DataFrame, features_to_plot: list):

    n_rows = len(features_to_plot) 
    _, ax = plt.subplots(nrows=n_rows, ncols=2, figsize=(16, n_rows * 4))

    # Flatten the ax array for easier indexing
    ax = ax.ravel()

    for i, feature in enumerate(features_to_plot):
        # First column: Histogram
        sns.histplot(df[feature], kde=True, ax=ax[i * 2])
        ax[i * 2].set_title(f'Histogram of {feature}')
        
        # Second column: Q-Q Plot
        stats.probplot(df[feature], dist="norm", plot=ax[i * 2 + 1])
        ax[i * 2 + 1].set_title(f'Q-Q Plot of {feature}')

    plt.tight_layout()
    plt.show()


def plot_auc_roc_curve(y_true, y_probability, title: str=""):
    fpr, tpr, _ = roc_curve(y_true, y_probability)

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f}) {title}',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.show(renderer="png")