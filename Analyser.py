import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from scipy.signal import periodogram, detrend
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error
from sklearn.feature_selection import mutual_info_regression
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
import warnings
warnings.filterwarnings("ignore")

start=time.time()

BASE_DIR=os.getcwd()
INPUT_FILE=os.path.join(BASE_DIR,"avocado.csv")
OUTPUT_DIR=os.path.join(BASE_DIR,"Output")
CHART_DIR=os.path.join(OUTPUT_DIR,"charts")
MODEL_DIR=os.path.join(OUTPUT_DIR,"models")

os.makedirs(CHART_DIR,exist_ok=True)
os.makedirs(MODEL_DIR,exist_ok=True)

DARK="#0b1220"
PANEL="#111827"
TEXT="#e5e7eb"
ACCENT="#22d3ee"
ACCENT2="#a78bfa"
ACCENT3="#34d399"
ACCENT4="#f59e0b"

plt.rcParams.update({
"figure.facecolor":DARK,
"axes.facecolor":PANEL,
"text.color":TEXT,
"axes.labelcolor":TEXT,
"xtick.color":TEXT,
"ytick.color":TEXT,
"font.size":13
})

df=pd.read_csv(INPUT_FILE)
df.columns=[c.lower().replace(" ","_") for c in df.columns]

df["date"]=pd.to_datetime(df["date"])
df=df.sort_values("date")
df["year"]=df["date"].dt.year
df["month"]=df["date"].dt.month
df["week"]=df["date"].dt.isocalendar().week.astype(int)

target="averageprice"
numeric=df.select_dtypes(include=np.number).columns.tolist()
df[numeric]=df[numeric].fillna(df[numeric].median())

scaler=StandardScaler()
scaled=scaler.fit_transform(df[numeric])
pickle.dump(scaler,open(os.path.join(MODEL_DIR,"scaler.pkl"),"wb"))

pca=PCA(n_components=min(6,len(numeric)))
pca_data=pca.fit_transform(scaled)
pickle.dump(pca,open(os.path.join(MODEL_DIR,"pca.pkl"),"wb"))

fig,ax=plt.subplots(figsize=(14,8))
ax.plot(range(1,len(pca.explained_variance_ratio_)+1),np.cumsum(pca.explained_variance_ratio_),color=ACCENT,linewidth=3)
ax.set_title("Principal Component Variance Explained")
ax.set_xlabel("Principal Component Index")
ax.set_ylabel("Cumulative Explained Variance Ratio")
ax.set_xticks(range(1,len(pca.explained_variance_ratio_)+1))
fig.savefig(os.path.join(CHART_DIR,"pca_variance.png"),dpi=300,bbox_inches="tight")
plt.close()

fig,ax=plt.subplots(figsize=(14,8))
ax.scatter(pca_data[:,0],pca_data[:,1],s=8,color=ACCENT3)
ax.set_title("PCA Projection")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
fig.savefig(os.path.join(CHART_DIR,"pca_projection.png"),dpi=300,bbox_inches="tight")
plt.close()

iso=IsolationForest(contamination=0.02)
anom=iso.fit_predict(scaled)

fig,ax=plt.subplots(figsize=(14,8))
ax.scatter(df["date"],df[target],c=anom,cmap="coolwarm",s=6)
ax.set_title("Anomaly Detection Timeline")
ax.set_xlabel("Date")
ax.set_ylabel("Average Price (USD)")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.savefig(os.path.join(CHART_DIR,"anomaly.png"),dpi=300,bbox_inches="tight")
plt.close()

kmeans=KMeans(n_clusters=6,n_init=20)
clusters=kmeans.fit_predict(scaled)
sil=silhouette_score(scaled,clusters)

fig,ax=plt.subplots(figsize=(14,8))
ax.scatter(pca_data[:,0],pca_data[:,1],c=clusters,cmap="viridis",s=8)
ax.set_title("Structural Regime Clustering")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
fig.savefig(os.path.join(CHART_DIR,"cluster.png"),dpi=300,bbox_inches="tight")
plt.close()

series=df[target].values
series_detrended=detrend(series)
series_centered=series_detrended-np.mean(series_detrended)

freq,power=periodogram(series_centered,fs=52,scaling='spectrum',window='hann')
mask=(freq>0)&(freq<=26)
freq=freq[mask]
power=power[mask]

fig,ax=plt.subplots(figsize=(14,8))
ax.plot(freq,power,color=ACCENT2,linewidth=2)
ax.set_title("Frequency Spectrum Analysis")
ax.set_xlabel("Frequency (cycles per year)")
ax.set_ylabel("Spectral Power")
ax.set_yscale("log")
ax.set_xlim(0,26)
fig.savefig(os.path.join(CHART_DIR,"fourier.png"),dpi=300,bbox_inches="tight")
plt.close()

rolling_mean=df[target].rolling(52).mean()
rolling_std=df[target].rolling(52).std()

fig,ax=plt.subplots(figsize=(14,8))
ax.plot(df["date"],df[target],alpha=0.3,label="Observed")
ax.plot(df["date"],rolling_mean,color=ACCENT,label="Rolling Mean",linewidth=3)
ax.set_title("Rolling Mean Trend")
ax.set_xlabel("Date")
ax.set_ylabel("Average Price (USD)")
ax.legend()
fig.savefig(os.path.join(CHART_DIR,"rolling_mean.png"),dpi=300,bbox_inches="tight")
plt.close()

fig,ax=plt.subplots(figsize=(14,8))
ax.plot(df["date"],rolling_std,color=ACCENT4,linewidth=3)
ax.set_title("Rolling Volatility")
ax.set_xlabel("Date")
ax.set_ylabel("Standard Deviation (USD)")
fig.savefig(os.path.join(CHART_DIR,"rolling_volatility.png"),dpi=300,bbox_inches="tight")
plt.close()

monthly=df.groupby(["year","month"])[target].mean().unstack()

fig,ax=plt.subplots(figsize=(14,8))
sns.heatmap(monthly,cmap="viridis",ax=ax)
ax.set_title("Monthly Heatmap")
ax.set_xlabel("Month")
ax.set_ylabel("Year")
fig.savefig(os.path.join(CHART_DIR,"monthly_heatmap.png"),dpi=300,bbox_inches="tight")
plt.close()

fig,ax=plt.subplots(figsize=(14,8))
sns.boxplot(x="month",y=target,data=df,ax=ax,color=ACCENT3)
ax.set_title("Seasonal Boxplot")
ax.set_xlabel("Month")
ax.set_ylabel("Average Price (USD)")
fig.savefig(os.path.join(CHART_DIR,"seasonal_boxplot.png"),dpi=300,bbox_inches="tight")
plt.close()

returns=df[target].pct_change().fillna(0)
cum_returns=(1+returns).cumprod()

fig,ax=plt.subplots(figsize=(14,8))
ax.plot(df["date"],cum_returns,color=ACCENT)
ax.set_title("Cumulative Return")
ax.set_xlabel("Date")
ax.set_ylabel("Growth Multiple")
fig.savefig(os.path.join(CHART_DIR,"cumulative_return.png"),dpi=300,bbox_inches="tight")
plt.close()

corr=df[numeric].corr()

fig,ax=plt.subplots(figsize=(14,10))
sns.heatmap(corr,cmap="viridis",ax=ax)
ax.set_title("Correlation Heatmap")
fig.savefig(os.path.join(CHART_DIR,"correlation.png"),dpi=300,bbox_inches="tight")
plt.close()

fig,ax=plt.subplots(figsize=(14,8))
sns.histplot(df[target],bins=60,color=ACCENT,ax=ax)
ax.set_title("Distribution Histogram")
ax.set_xlabel("Average Price (USD)")
ax.set_ylabel("Frequency")
fig.savefig(os.path.join(CHART_DIR,"distribution.png"),dpi=300,bbox_inches="tight")
plt.close()

year_group=df.groupby("year")[target].mean()

fig,ax=plt.subplots(figsize=(14,8))
ax.plot(year_group.index,year_group.values,color=ACCENT3,linewidth=3)
ax.set_title("Yearly Trend")
ax.set_xlabel("Year")
ax.set_ylabel("Average Price (USD)")
fig.savefig(os.path.join(CHART_DIR,"yearly_trend.png"),dpi=300,bbox_inches="tight")
plt.close()

mi=mutual_info_regression(df[numeric],df[target])
imp=pd.Series(mi,index=numeric).sort_values()

fig,ax=plt.subplots(figsize=(14,8))
imp.plot(kind="barh",color=ACCENT2,ax=ax)
ax.set_title("Feature Importance")
ax.set_xlabel("Information Gain")
fig.savefig(os.path.join(CHART_DIR,"feature_importance.png"),dpi=300,bbox_inches="tight")
plt.close()

execution=round(time.time()-start,2)

styles=getSampleStyleSheet()

title_style=ParagraphStyle(name="title",fontSize=34,leading=42,alignment=1,textColor=HexColor("#22d3ee"),spaceAfter=50,spaceBefore=50)
heading_style=ParagraphStyle(name="heading",fontSize=22,leading=28,alignment=1,textColor=HexColor("#a78bfa"),spaceBefore=40,spaceAfter=30)
body_style=ParagraphStyle(name="body",fontSize=12,leading=20,spaceAfter=40)

doc=SimpleDocTemplate(os.path.join(OUTPUT_DIR,"Institutional_Intelligence_Report.pdf"),leftMargin=72,rightMargin=72,topMargin=72,bottomMargin=72)

elements=[]
elements.append(Paragraph("Avocado Market Intelligence Report",title_style))

summary=f"""
Executive Summary<br/><br/>
Dataset Size: {len(df)} observations<br/>
Years Covered: {df.year.min()} to {df.year.max()}<br/>
Cluster Quality Score: {round(sil,3)}<br/>
Execution Time: {execution} seconds<br/><br/>
This report provides structural, statistical, temporal, and predictive intelligence analysis.
"""

elements.append(Paragraph(summary,body_style))
elements.append(PageBreak())

charts=sorted(os.listdir(CHART_DIR))

for chart in charts:
    elements.append(Paragraph(chart.replace("_"," ").replace(".png","").title(),heading_style))
    elements.append(Image(os.path.join(CHART_DIR,chart),width=6.5*inch,height=4.5*inch))
    elements.append(Spacer(1,40))
    explanation=f"""
This chart presents structural intelligence extracted from the dataset using statistical, machine learning, and signal processing methods.
"""
    elements.append(Paragraph(explanation,body_style))
    elements.append(PageBreak())

doc.build(elements)

print("Complete")
print("Execution Time:",execution)