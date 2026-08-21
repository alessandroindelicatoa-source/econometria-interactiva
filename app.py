import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from econometrics_lab.data_manager import load_bytes, apply_transform, missing_summary, handle_missing
from econometrics_lab.utils import dataframe_profile, numeric_columns, categorical_columns, build_formula
from econometrics_lab.model_engine import (
    fit_cross_section, fit_ordered, fit_zero_inflated_poisson, fit_panel,
    fit_iv, fit_did, fit_arima, fit_var
)
from econometrics_lab.diagnostics import ols_diagnostics, vif_table, influence_table
from econometrics_lab.plot_factory import (
    histogram, box_violin, scatter, correlation_heatmap, scatter_matrix,
    line_plot, group_means, missingness, coefficient_plot, actual_predicted,
    residual_fitted, residual_hist, qq_plot, roc_plot, marginal_effects_plot,
    model_comparison, did_trends, style
)
from econometrics_lab.fuzzy import fuzzy_index, topsis
from econometrics_lab.exporting import models_excel, docx_report, pdf_report
from econometrics_lab.codegen import generate_code
from econometrics_lab.interpretation import interpret_model
from econometrics_lab.ui_v2 import (
    inject_css, hero, dataset_fingerprint, recommend_models, model_header,
    comparison_table, assistant_recommendation, candidate_panel_pairs
)
from econometrics_lab.publication import apply_publication_style, figure_bytes, figure_html
from econometrics_lab.research_v2 import health_check, robustness_ols, event_study

APP_DIR=Path(__file__).parent
st.set_page_config(
    page_title="Econometrics Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
inject_css()

NAV=[
    "🏠 Workspace",
    "📁 Data",
    "📊 Explore",
    "🧮 Model Studio",
    "🧪 Research Lab",
    "🎮 Simulator",
    "📑 Report",
]

DEFAULTS={
    "df":None,"source_name":None,"column_map":{},"models":[],
    "current_model":None,"nav":"🏠 Workspace",
    "pub_style":"Minimal","pub_size":"Double column",
}
for k,v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k]=v

def go_to(page_name):
    st.session_state["nav"] = page_name

def set_df(df,name="dataset",mapping=None):
    st.session_state.df=df
    st.session_state.source_name=name
    st.session_state.column_map=mapping or {}
    st.session_state.current_model=None

def need_data():
    if st.session_state.df is None:
        st.warning("Load a dataset first in **Data**.")
        st.button("Go to Data", on_click=go_to, args=("📁 Data",))
        st.stop()
    return st.session_state.df

def save_model(m):
    existing=[x.name for x in st.session_state.models]
    if m.name in existing:
        i=existing.index(m.name)
        st.session_state.models[i]=m
    else:
        st.session_state.models.append(m)
    st.session_state.current_model=m
    st.success(f"Model saved: **{m.name}**")

def pub_fig(fig,title=None):
    return apply_publication_style(fig,st.session_state.pub_style,st.session_state.pub_size,title)

def download_figure(fig,key):
    c1,c2,c3,c4=st.columns(4)
    c1.download_button(
        "HTML",figure_html(fig),file_name=f"{key}.html",mime="text/html",
        key=f"{key}_html"
    )
    for col,fmt,label in [(c2,"png","PNG"),(c3,"svg","SVG"),(c4,"pdf","PDF")]:
        try:
            data=figure_bytes(fig,fmt,scale=2 if fmt=="png" else 1)
            col.download_button(label,data,file_name=f"{key}.{fmt}",
                                mime={"png":"image/png","svg":"image/svg+xml","pdf":"application/pdf"}[fmt],
                                key=f"{key}_{fmt}")
        except Exception:
            col.caption(f"{label}: available when Kaleido is active")

def model_graph(model,df,chart):
    if chart=="Coefficients":
        return coefficient_plot(model)
    if chart=="Actual vs predicted":
        return actual_predicted(df,model)
    if chart=="Residuals vs fitted":
        return residual_fitted(model)
    if chart=="Residual distribution":
        return residual_hist(model)
    if chart=="Q–Q":
        return qq_plot(model)
    if chart=="ROC / AUC":
        return roc_plot(df,model)
    if chart=="Marginal effects":
        return marginal_effects_plot(model)
    raise ValueError(chart)

def model_graph_choices(m):
    choices=["Coefficients"]
    if m.fitted is not None and m.residuals is not None:
        choices += ["Actual vs predicted","Residuals vs fitted","Residual distribution","Q–Q"]
    if m.family in ("Logit","Probit","Cloglog"):
        choices += ["ROC / AUC"]
    if m.marginal_effects is not None:
        choices += ["Marginal effects"]
    return choices

def render_health_table(model):
    h=health_check(model)
    for _,r in h.iterrows():
        status=str(r["status"])
        icon={"Good":"🟢","Warning":"🟠","High":"🔴","Check":"🔵","N/A":"⚪"}.get(status,"⚪")
        st.markdown(f"**{icon} {r['dimension']} — {status}**  \n{r['detail']}")
    return h

def render_model_result(model,df):
    model_header(model)
    tabs=st.tabs(["Results","Interpretation","Diagnostics","Graphics"])
    with tabs[0]:
        st.dataframe(
            model.coef_table.style.format({
                "coef":"{:.5g}","std_err":"{:.5g}","stat":"{:.4g}",
                "p_value":"{:.4g}","ci_low":"{:.5g}","ci_high":"{:.5g}"
            }),
            use_container_width=True,hide_index=True
        )
        if model.marginal_effects is not None:
            with st.expander("Average marginal effects"):
                st.dataframe(model.marginal_effects,use_container_width=True,hide_index=True)
        if getattr(model,"notes",None):
            st.caption(" · ".join(model.notes))
    with tabs[1]:
        nonconst=[t for t in model.coef_table["term"].astype(str).tolist()
                  if t.lower() not in ("intercept","const")]
        focal=st.selectbox("Focal coefficient",nonconst or model.coef_table["term"].astype(str).tolist(),
                           key=f"result_focal_{model.name}")
        st.markdown(interpret_model(model,focal))
    with tabs[2]:
        render_health_table(model)
        if model.family in ("OLS","Linear Probability Model"):
            try:
                with st.expander("Detailed test table"):
                    st.dataframe(ols_diagnostics(model),use_container_width=True,hide_index=True)
                with st.expander("Variance Inflation Factors"):
                    st.dataframe(vif_table(model),use_container_width=True,hide_index=True)
            except Exception as e:
                st.caption(str(e))
    with tabs[3]:
        chart=st.selectbox("Graph",model_graph_choices(model),key=f"result_graph_{model.name}")
        try:
            fig=pub_fig(model_graph(model,df,chart))
            st.plotly_chart(fig,use_container_width=True)
            download_figure(fig,f"{model.name}_{chart}".replace(" ","_").replace("/","_"))
        except Exception as e:
            st.error(str(e))

# ---------- sidebar ----------
with st.sidebar:
    st.markdown("## ECONOMETRICS LAB")
    st.caption("Interactive Econometric Research Environment")
    st.radio("Navigation",NAV,key="nav",label_visibility="collapsed")
    st.divider()
    if st.session_state.df is not None:
        d=st.session_state.df
        fp=dataset_fingerprint(d)
        st.markdown(f"**{st.session_state.source_name}**")
        st.caption(f"{fp['rows']:,} rows · {fp['cols']} variables")
        st.caption(f"Missing: {fp['missing_pct']:.2f}% · Models: {len(st.session_state.models)}")
    else:
        st.caption("No dataset loaded")
    with st.expander("Publication graphics"):
        st.selectbox("Style",["Minimal","Economics journal","APA","Presentation","Dark presentation"],key="pub_style")
        st.selectbox("Size",["Single column","Double column","Presentation 16:9","Square"],key="pub_size")

page=st.session_state.nav

# ============================================================
# WORKSPACE
# ============================================================
if page=="🏠 Workspace":
    hero(
        "Econometrics Lab",
        "From raw data to econometric evidence: explore, estimate, diagnose, stress-test, simulate and export."
    )
    if st.session_state.df is None:
        c1,c2=st.columns([1.35,1])
        with c1:
            st.markdown("### Start a new analysis")
            st.markdown("Upload your own research data or open a synthetic project designed to test the full workflow.")
            st.button(
                "Open Data workspace",
                type="primary",
                use_container_width=True,
                on_click=go_to,
                args=("📁 Data",),
            )
        with c2:
            st.markdown('<div class="card"><strong>Designed for research</strong><br><span class="small">Econometric models, causal workflows, diagnostics, specification robustness, fuzzy indicators, publication graphics and reproducible exports.</span></div>',unsafe_allow_html=True)
    else:
        df=st.session_state.df;fp=dataset_fingerprint(df)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Observations",f"{fp['rows']:,}")
        c2.metric("Variables",fp["cols"])
        c3.metric("Missingness",f"{fp['missing_pct']:.2f}%")
        c4.metric("Saved models",len(st.session_state.models))
        st.markdown("### Continue your analysis")
        b1,b2,b3=st.columns(3)
        b1.button("📊 Explore data",use_container_width=True,on_click=go_to,args=("📊 Explore",))
        b2.button("🧮 Build a model",type="primary",use_container_width=True,on_click=go_to,args=("🧮 Model Studio",))
        b3.button("🧪 Open Research Lab",use_container_width=True,on_click=go_to,args=("🧪 Research Lab",))

        st.markdown("### Dataset intelligence")
        st.markdown(assistant_recommendation(df))
        if st.session_state.models:
            st.markdown("### Recent models")
            recent=st.session_state.models[-5:][::-1]
            tbl=pd.DataFrame([{"Model":m.name,"Family":m.family,"Formula":m.formula,"Created":m.created} for m in recent])
            st.dataframe(tbl,use_container_width=True,hide_index=True)

# ============================================================
# DATA
# ============================================================
elif page=="📁 Data":
    hero("Data","Import, inspect, clean and transform your research dataset.","Data workspace")
    tabs=st.tabs(["Import","Profile","Missing data","Transform","Filter"])
    with tabs[0]:
        c1,c2=st.columns(2)
        with c1:
            st.markdown("### Upload data")
            up=st.file_uploader("CSV, Excel, Stata, SPSS or Parquet",
                                type=["csv","xlsx","xls","dta","sav","parquet"])
            if up is not None:
                try:
                    df,mapping=load_bytes(up.getvalue(),up.name)
                    set_df(df,up.name,mapping)
                    st.success(f"Loaded {len(df):,} observations and {df.shape[1]} variables.")
                    if any(k!=v for k,v in mapping.items()):
                        with st.expander("Renamed variables"):
                            st.dataframe(pd.DataFrame({"original":mapping.keys(),"econometric_name":mapping.values()}),
                                         use_container_width=True,hide_index=True)
                except Exception as e:
                    st.exception(e)
        with c2:
            st.markdown("### Demo projects")
            demo=st.selectbox("Dataset",["Panel / causal research demo","Time-series research demo"])
            st.caption("The panel demo supports OLS, binary/count models, FE/RE, IV and DiD.")
            if st.button("Load demo project",type="primary",use_container_width=True):
                fn="panel_causal_demo.csv" if demo.startswith("Panel") else "time_series_demo.csv"
                df=pd.read_csv(APP_DIR/"data"/fn)
                set_df(df,fn,{c:c for c in df.columns})
                st.rerun()
    with tabs[1]:
        df=need_data();fp=dataset_fingerprint(df)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Rows",f"{fp['rows']:,}");c2.metric("Columns",fp["cols"])
        c3.metric("Numeric",fp["numeric"]);c4.metric("Duplicates",f"{fp['duplicates']:,}")
        st.dataframe(df.head(250),use_container_width=True)
        st.markdown("### Variable profile")
        st.dataframe(dataframe_profile(df),use_container_width=True,hide_index=True)
        st.download_button("Download current dataset",df.to_csv(index=False).encode(),
                           file_name="econometrics_lab_data.csv",mime="text/csv")
    with tabs[2]:
        df=need_data()
        st.dataframe(missing_summary(df),use_container_width=True,hide_index=True)
        c1,c2=st.columns([1,2])
        mode=c1.selectbox("Treatment",["Drop rows","Mean","Median","Mode","Forward fill","Backward fill"])
        cols=c2.multiselect("Variables",df.columns.tolist(),default=df.columns.tolist())
        if st.button("Apply missing-data treatment"):
            set_df(handle_missing(df,mode,cols),st.session_state.source_name,st.session_state.column_map)
            st.success("Treatment applied.")
    with tabs[3]:
        df=need_data();nums=numeric_columns(df)
        c1,c2=st.columns(2)
        op=c1.selectbox("Transformation",["Log","Log(1+x)","Square","Standardize","Difference","Lag","Winsorize","Interaction","Dummy threshold"])
        cols=c2.multiselect("Variable(s)",nums)
        new=st.text_input("New variable name (optional)")
        val=None
        if op=="Lag": val=st.number_input("Lag",1,50,1)
        elif op=="Winsorize": val=st.number_input("Tail proportion",.001,.20,.01,.001)
        elif op=="Dummy threshold": val=st.number_input("Threshold",value=0.0)
        if st.button("Create transformation",type="primary"):
            try:
                set_df(apply_transform(df,op,cols,new or None,val),st.session_state.source_name,st.session_state.column_map)
                st.success("Transformation created.")
            except Exception as e: st.exception(e)
    with tabs[4]:
        df=need_data()
        st.caption("Pandas query syntax, e.g. `age >= 30 and immigrant == 1`.")
        q=st.text_input("Filter expression")
        if st.button("Apply filter") and q:
            try:
                set_df(df.query(q),st.session_state.source_name,st.session_state.column_map)
                st.success("Filter applied.")
            except Exception as e: st.exception(e)

# ============================================================
# EXPLORE
# ============================================================
elif page=="📊 Explore":
    df=need_data();nums=numeric_columns(df)
    hero("Explore","Understand distributions, relationships, groups, dependence and data quality before modelling.","Visual analytics")
    tabs=st.tabs(["Distributions","Relationships","Groups","Correlation","Time / Panel","Data quality","Publication"])
    with tabs[0]:
        c1,c2,c3=st.columns(3)
        kind=c1.selectbox("Chart",["Histogram","Box","Violin"])
        y=c2.selectbox("Variable",nums)
        group=c3.selectbox("Group / colour",["None"]+df.columns.tolist())
        group=None if group=="None" else group
        if kind=="Histogram":
            bins=st.slider("Bins",5,100,30)
            fig=histogram(df,y,group,bins)
        else:
            x=st.selectbox("X category",["None"]+df.columns.tolist())
            x=None if x=="None" else x
            fig=box_violin(df,y,x,group,kind)
        fig=pub_fig(fig);st.plotly_chart(fig,use_container_width=True);download_figure(fig,"distribution")
    with tabs[1]:
        if len(nums)<2:
            st.info("At least two numeric variables are required.")
        else:
            c1,c2,c3=st.columns(3)
            x=c1.selectbox("X",nums,key="ex_rel_x")
            y=c2.selectbox("Y",nums,index=min(1,len(nums)-1),key="ex_rel_y")
            trend=c3.selectbox("Trend",["None","OLS","LOWESS"])
            color=st.selectbox("Colour",["None"]+df.columns.tolist(),key="ex_rel_c")
            color=None if color=="None" else color
            fig=scatter(df,x,y,color,None,None if trend=="None" else trend)
            fig=pub_fig(fig);st.plotly_chart(fig,use_container_width=True);download_figure(fig,"relationship")
    with tabs[2]:
        c1,c2=st.columns(2)
        y=c1.selectbox("Outcome",nums,key="ex_grp_y")
        grp=c2.selectbox("Grouping variable",df.columns.tolist(),key="ex_grp_x")
        fig=pub_fig(group_means(df,y,grp));st.plotly_chart(fig,use_container_width=True);download_figure(fig,"group_means")
    with tabs[3]:
        variables=st.multiselect("Variables",nums,default=nums[:min(8,len(nums))])
        method=st.radio("Method",["pearson","spearman","kendall"],horizontal=True)
        if len(variables)>=2:
            fig=pub_fig(correlation_heatmap(df,variables,method))
            st.plotly_chart(fig,use_container_width=True);download_figure(fig,"correlation")
            if len(variables)<=8 and st.checkbox("Show scatter matrix"):
                sm=pub_fig(scatter_matrix(df,variables));st.plotly_chart(sm,use_container_width=True)
    with tabs[4]:
        c1,c2,c3=st.columns(3)
        tx=c1.selectbox("Time/index",df.columns.tolist(),key="ex_time_x")
        ty=c2.selectbox("Outcome",nums,key="ex_time_y")
        tg=c3.selectbox("Panel/group",["None"]+df.columns.tolist(),key="ex_time_g")
        tg=None if tg=="None" else tg
        fig=pub_fig(line_plot(df,tx,ty,tg))
        st.plotly_chart(fig,use_container_width=True);download_figure(fig,"time_panel")
    with tabs[5]:
        fp=dataset_fingerprint(df)
        c1,c2,c3=st.columns(3)
        c1.metric("Average missing",f"{fp['missing_pct']:.2f}%")
        c2.metric("Duplicated rows",fp["duplicates"])
        c3.metric("Binary variables",len(fp["binary"]))
        fig=pub_fig(missingness(df))
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(missing_summary(df),use_container_width=True,hide_index=True)
    with tabs[6]:
        st.markdown("### Publication mode")
        st.markdown(f"**Current style:** {st.session_state.pub_style}  \n**Current size:** {st.session_state.pub_size}")
        st.info("Change publication style and dimensions from the sidebar. Every Explore and model graph inherits those settings.")
        st.markdown("""
**Export formats:** interactive HTML plus PNG, SVG and PDF when Kaleido is available.  
**Suggested use:** Single column for journals, Double column for wide figures, Presentation 16:9 for talks.
""")

# ============================================================
# MODEL STUDIO
# ============================================================
elif page=="🧮 Model Studio":
    df=need_data();nums=numeric_columns(df)
    hero("Model Studio","Specify, estimate and inspect econometric models without leaving the workflow.","Unified model builder")

    mode=st.segmented_control("Model family",["Cross-sectional","Panel","Causal","Time series"],default="Cross-sectional")
    estimated=None

    if mode=="Cross-sectional":
        c1,c2=st.columns([1,1])
        y=c1.selectbox("Dependent variable",df.columns.tolist(),key="ms_y")
        family=c2.selectbox("Estimator",[
            "OLS","WLS","Linear Probability Model","Logit","Probit","Cloglog",
            "Poisson","Negative Binomial","Zero-Inflated Poisson",
            "Ordered Logit","Ordered Probit","Quantile Regression"
        ],key="ms_family")
        rec=recommend_models(df,y)
        if rec:
            st.caption("Suggested from outcome structure: " + " · ".join([r[0] for r in rec[:3]]))
        x=st.multiselect("Explanatory variables",[c for c in df.columns if c!=y],key="ms_x")
        cats=[];interactions=[];cov="Classical";cluster=None;weights=None;q=.5
        if family not in ("Ordered Logit","Ordered Probit","Zero-Inflated Poisson","Quantile Regression"):
            cats=st.multiselect("Treat as categorical",[c for c in x if df[c].nunique(dropna=True)<30],key="ms_cats")
            with st.expander("Interactions and covariance"):
                a=st.selectbox("Interaction A",["None"]+x,key="ms_ia")
                b=st.selectbox("Interaction B",["None"]+x,key="ms_ib")
                if a!="None" and b!="None" and a!=b: interactions=[(a,b)]
                cov=st.selectbox("Standard errors",["Classical","HC0","HC1","HC2","HC3","Cluster"],
                                 index=4 if family in ("OLS","WLS","Linear Probability Model") else 0,key="ms_cov")
                if cov=="Cluster":
                    cluster=st.selectbox("Cluster variable",df.columns.tolist(),key="ms_cluster")
                if family=="WLS":
                    weights=st.selectbox("Weight variable",[c for c in nums if c!=y],key="ms_weights")
        if family=="Quantile Regression":
            q=st.slider("Quantile",.05,.95,.50,.05,key="ms_q")
        name=st.text_input("Model name",value=f"{family} — {y}",key="ms_name")
        try:
            preview_family="OLS" if family=="Linear Probability Model" else family
            if x:
                preview=build_formula(y,x,cats,interactions)
                st.markdown(f'<div class="formula">{preview}</div>',unsafe_allow_html=True)
        except Exception:
            pass
        if st.button("RUN MODEL",type="primary",use_container_width=True,disabled=not bool(x)):
            try:
                if family=="Zero-Inflated Poisson":
                    estimated=fit_zero_inflated_poisson(df,name,y,x)
                elif family in ("Ordered Logit","Ordered Probit"):
                    estimated=fit_ordered(df,name,y,x,"logit" if family.endswith("Logit") else "probit")
                else:
                    engine_family="OLS" if family=="Linear Probability Model" else family
                    estimated=fit_cross_section(df,name,engine_family,y,x,cats,interactions,cov,q,weights,cluster)
                    if family=="Linear Probability Model":
                        estimated.family="Linear Probability Model"
                save_model(estimated)
            except Exception as e:
                st.exception(e)

    elif mode=="Panel":
        ids,times=candidate_panel_pairs(df)
        c1,c2,c3=st.columns(3)
        entity=c1.selectbox("Entity ID",df.columns.tolist(),index=df.columns.get_loc(ids[0]) if ids and ids[0] in df.columns else 0)
        time=c2.selectbox("Time",df.columns.tolist(),index=df.columns.get_loc(times[0]) if times and times[0] in df.columns else min(1,len(df.columns)-1))
        family=c3.selectbox("Estimator",["Fixed Effects","Random Effects","Pooled OLS","First Differences"])
        y=st.selectbox("Dependent variable",nums,key="panel_y")
        x=st.multiselect("Regressors",[c for c in nums if c!=y],key="panel_x")
        c1,c2=st.columns(2)
        time_fe=c1.checkbox("Time fixed effects",value=True,disabled=family!="Fixed Effects")
        cov=c2.selectbox("Covariance",["robust","clustered","unadjusted"])
        name=st.text_input("Model name",f"{family} — {y}",key="panel_name")
        if st.button("RUN PANEL MODEL",type="primary",use_container_width=True,disabled=not bool(x)):
            try:
                estimated=fit_panel(df,name,family,y,x,entity,time,time_fe,cov);save_model(estimated)
            except Exception as e: st.exception(e)

    elif mode=="Causal":
        causal=st.segmented_control("Design",["IV / 2SLS","Difference-in-Differences"],default="IV / 2SLS")
        if causal=="IV / 2SLS":
            y=st.selectbox("Outcome",nums,key="iv_y")
            endog=st.multiselect("Endogenous regressor(s)",[c for c in nums if c!=y],key="iv_endog")
            inst=st.multiselect("Instrument(s)",[c for c in nums if c!=y and c not in endog],key="iv_inst")
            exog=st.multiselect("Exogenous controls",[c for c in nums if c!=y and c not in endog and c not in inst],key="iv_exog")
            name=st.text_input("Model name",f"2SLS — {y}",key="iv_name")
            if st.button("RUN 2SLS",type="primary",use_container_width=True,disabled=not(endog and inst)):
                try:
                    estimated=fit_iv(df,name,y,exog,endog,inst);save_model(estimated)
                except Exception as e: st.exception(e)
        else:
            y=st.selectbox("Outcome",nums,key="did_y")
            c1,c2=st.columns(2)
            treat=c1.selectbox("Treatment indicator",nums,key="did_t")
            post=c2.selectbox("Post indicator",nums,key="did_p")
            controls=st.multiselect("Controls",[c for c in nums if c not in [y,treat,post]],key="did_c")
            add_fe=st.checkbox("Unit and time fixed effects")
            c1,c2=st.columns(2)
            unit=c1.selectbox("Unit",["None"]+df.columns.tolist(),key="did_unit");unit=None if unit=="None" else unit
            time=c2.selectbox("Time",["None"]+df.columns.tolist(),key="did_time");time=None if time=="None" else time
            cov=st.selectbox("Inference",["HC3","HC2","HC1","HC0","Cluster unit"],key="did_cov")
            name=st.text_input("Model name",f"DiD — {y}",key="did_name")
            if st.button("RUN DiD",type="primary",use_container_width=True):
                try:
                    estimated=fit_did(df,name,y,treat,post,controls,unit,time,add_fe,cov);save_model(estimated)
                except Exception as e: st.exception(e)

    else:
        ts=st.segmented_control("Time-series model",["Stationarity","ARIMA","VAR"],default="ARIMA")
        if ts=="Stationarity":
            from statsmodels.tsa.stattools import adfuller,kpss
            y=st.selectbox("Series",nums,key="ts_stationary_y")
            if st.button("RUN ADF + KPSS",type="primary"):
                try:
                    s=df[y].dropna()
                    adf=adfuller(s,autolag="AIC");kp=kpss(s,regression="c",nlags="auto")
                    st.dataframe(pd.DataFrame([
                        ["ADF",adf[0],adf[1],"Unit root"],
                        ["KPSS",kp[0],kp[1],"Stationary"]
                    ],columns=["test","statistic","p_value","null_hypothesis"]),use_container_width=True,hide_index=True)
                except Exception as e: st.exception(e)
        elif ts=="ARIMA":
            y=st.selectbox("Series",nums,key="ts_arima_y")
            c1,c2,c3=st.columns(3)
            p=c1.number_input("p",0,10,1);d=c2.number_input("d",0,3,0);q=c3.number_input("q",0,10,1)
            exog=st.multiselect("Exogenous variables",[c for c in nums if c!=y],key="ts_arima_x")
            name=st.text_input("Model name",f"ARIMA({p},{d},{q}) — {y}",key="ts_arima_name")
            if st.button("RUN ARIMA",type="primary",use_container_width=True):
                try:
                    estimated=fit_arima(df,name,y,p,d,q,exog);save_model(estimated)
                except Exception as e: st.exception(e)
        else:
            vars_=st.multiselect("Endogenous variables",nums,default=nums[:min(2,len(nums))],key="ts_var_vars")
            lags=st.number_input("Lags",1,12,1)
            name=st.text_input("Model name",f"VAR({lags})",key="ts_var_name")
            if st.button("RUN VAR",type="primary",use_container_width=True,disabled=len(vars_)<2):
                try:
                    estimated=fit_var(df,name,vars_,lags);save_model(estimated)
                except Exception as e: st.exception(e)

    if st.session_state.current_model is not None:
        st.divider()
        render_model_result(st.session_state.current_model,df)

# ============================================================
# RESEARCH LAB
# ============================================================
elif page=="🧪 Research Lab":
    df=need_data();nums=numeric_columns(df)
    hero("Research Lab","Diagnose assumptions, stress-test specifications and interrogate identification.","Robustness & causal diagnostics")
    tabs=st.tabs(["Model health","Robustness","Specification curve","Event study","Compare models","Fuzzy","Econometric assistant"])

    with tabs[0]:
        if not st.session_state.models:
            st.info("Estimate at least one model in Model Studio.")
        else:
            mn=st.selectbox("Model",[m.name for m in st.session_state.models],key="health_model")
            m=next(x for x in st.session_state.models if x.name==mn)
            model_header(m)
            render_health_table(m)
            if m.residuals is not None:
                c1,c2=st.columns(2)
                c1.plotly_chart(pub_fig(residual_fitted(m)),use_container_width=True)
                c2.plotly_chart(pub_fig(qq_plot(m)),use_container_width=True)

    with tabs[1]:
        st.markdown("### OLS robustness battery")
        if not nums:
            st.info("Numeric variables required.")
        else:
            y=st.selectbox("Outcome",nums,key="rob_y")
            focal=st.selectbox("Main coefficient",[c for c in nums if c!=y],key="rob_focal")
            controls=st.multiselect("Baseline controls",[c for c in nums if c not in [y,focal]],key="rob_ctrl")
            c1,c2=st.columns(2)
            cluster=c1.selectbox("Cluster variable",["None"]+df.columns.tolist(),key="rob_cluster")
            cluster=None if cluster=="None" else cluster
            fe_vars=c2.multiselect("Fixed-effect controls",[c for c in df.columns if c not in [y,focal] and c not in controls],
                                   key="rob_fe")
            winsor=st.multiselect("Winsorize at 1% / 99%",[c for c in nums if c!=y],key="rob_win")
            infl=st.checkbox("Exclude influential observations using Cook's D > 4/N",value=True)
            if st.button("RUN ROBUSTNESS BATTERY",type="primary",use_container_width=True):
                try:
                    res,rob_models=robustness_ols(df,y,focal,controls,cluster,fe_vars,winsor,infl)
                    st.session_state["robustness_result"]=res
                    st.dataframe(res,use_container_width=True,hide_index=True)
                    if len(res):
                        fig=go.Figure(go.Scatter(
                            x=res["coef"],y=res["specification"],mode="markers",
                            error_x=dict(type="data",symmetric=False,
                                         array=res["ci_high"]-res["coef"],
                                         arrayminus=res["coef"]-res["ci_low"])
                        ))
                        fig.add_vline(x=0,line_dash="dash")
                        fig.update_xaxes(title=f"Coefficient: {focal} (95% CI)")
                        fig=pub_fig(style(fig,"Robustness of focal coefficient"))
                        st.plotly_chart(fig,use_container_width=True);download_figure(fig,"robustness")
                        sig=(res["p_value"]<.05).sum()
                        sign_consistency=max((res["coef"]>0).sum(),(res["coef"]<0).sum())
                        st.info(f"Sign is consistent in **{sign_consistency}/{len(res)}** specifications; p < .05 in **{sig}/{len(res)}**.")
                except Exception as e: st.exception(e)

    with tabs[2]:
        y=st.selectbox("Outcome",nums,key="sc_y")
        focal=st.selectbox("Focal variable",[c for c in nums if c!=y],key="sc_focal")
        candidate=st.multiselect("Candidate controls",[c for c in nums if c not in [y,focal]],key="sc_controls")
        max_controls=st.slider("Maximum controls per model",0,min(5,len(candidate)),min(3,len(candidate)))
        max_specs=st.slider("Maximum specifications",10,200,100,10)
        if st.button("RUN SPECIFICATION CURVE",type="primary",use_container_width=True):
            rows=[]
            combos=[]
            for k in range(max_controls+1):
                combos.extend(list(combinations(candidate,k)))
            for i,ctrl in enumerate(combos[:max_specs],1):
                try:
                    mm=fit_cross_section(df,f"Spec {i}","OLS",y,[focal]+list(ctrl),cov="HC3")
                    rr=mm.coef_table[mm.coef_table.term==focal].iloc[0]
                    rows.append({"spec":i,"controls":", ".join(ctrl) or "None","coef":rr.coef,
                                 "low":rr.ci_low,"high":rr.ci_high,"p":rr.p_value})
                except Exception:
                    pass
            res=pd.DataFrame(rows)
            st.dataframe(res,use_container_width=True,hide_index=True)
            if len(res):
                fig=go.Figure(go.Scatter(
                    x=res["spec"],y=res["coef"],mode="markers",
                    error_y=dict(type="data",symmetric=False,array=res["high"]-res["coef"],arrayminus=res["coef"]-res["low"]),
                    marker=dict(size=7)
                ))
                fig.add_hline(y=0,line_dash="dash")
                fig.update_xaxes(title="Specification");fig.update_yaxes(title=f"Estimate: {focal}")
                fig=pub_fig(style(fig,"Specification curve"))
                st.plotly_chart(fig,use_container_width=True);download_figure(fig,"specification_curve")

    with tabs[3]:
        st.markdown("### Event Study")
        y=st.selectbox("Outcome",nums,key="ev_y")
        treat=st.selectbox("Treatment indicator",nums,key="ev_treat")
        rel=st.selectbox("Relative-time variable",df.columns.tolist(),key="ev_rel")
        controls=st.multiselect("Controls",[c for c in nums if c not in [y,treat] and c!=rel],key="ev_controls")
        c1,c2,c3=st.columns(3)
        unit=c1.selectbox("Unit FE / cluster",["None"]+df.columns.tolist(),key="ev_unit");unit=None if unit=="None" else unit
        time=c2.selectbox("Calendar-time FE",["None"]+df.columns.tolist(),key="ev_time");time=None if time=="None" else time
        ref=c3.number_input("Reference period",value=-1)
        if st.button("RUN EVENT STUDY",type="primary",use_container_width=True):
            try:
                es,formula,_=event_study(df,y,treat,rel,controls,unit,time,ref,cluster=True)
                st.code(formula,language="text")
                st.dataframe(es,use_container_width=True,hide_index=True)
                if len(es):
                    fig=go.Figure(go.Scatter(
                        x=es["period"],y=es["coef"],mode="markers+lines",
                        error_y=dict(type="data",symmetric=False,array=es["ci_high"]-es["coef"],arrayminus=es["coef"]-es["ci_low"])
                    ))
                    fig.add_hline(y=0,line_dash="dash");fig.add_vline(x=0,line_dash="dot")
                    fig.update_xaxes(title="Relative time");fig.update_yaxes(title="Treatment effect (95% CI)")
                    fig=pub_fig(style(fig,"Event-study estimates"))
                    st.plotly_chart(fig,use_container_width=True);download_figure(fig,"event_study")
            except Exception as e: st.exception(e)

    with tabs[4]:
        models=st.session_state.models
        if not models:
            st.info("No models saved.")
        else:
            chosen_names=st.multiselect("Models",[m.name for m in models],default=[m.name for m in models[-min(4,len(models)):]],key="cmp_models")
            chosen=[m for m in models if m.name in chosen_names]
            if chosen:
                st.dataframe(comparison_table(chosen),use_container_width=True)
                common=set(chosen[0].coef_table.term)
                for mm in chosen[1:]: common &= set(mm.coef_table.term)
                common=sorted(common-set(["Intercept","const"]))
                if common:
                    term=st.selectbox("Graph common coefficient",common,key="cmp_term")
                    fig=pub_fig(model_comparison(chosen,term))
                    st.plotly_chart(fig,use_container_width=True);download_figure(fig,"model_comparison")

    with tabs[5]:
        subt=st.segmented_control("Fuzzy method",["Likert fuzzy index","TOPSIS"],default="Likert fuzzy index")
        if subt=="Likert fuzzy index":
            items=st.multiselect("Likert items (1–5)",nums,key="fz_items")
            method=st.radio("Defuzzification",["centroid","weighted"],horizontal=True,key="fz_method")
            new=st.text_input("New variable","fuzzy_index",key="fz_new")
            if st.button("CREATE FUZZY INDEX",disabled=not items):
                try:
                    out=df.copy();out[new]=fuzzy_index(out,items,method=method)
                    set_df(out,st.session_state.source_name,st.session_state.column_map)
                    st.success(f"Created `{new}`.")
                except Exception as e: st.exception(e)
        else:
            crit=st.multiselect("Criteria",nums,key="tp_crit")
            benefits=[]
            for c in crit:
                benefits.append(st.checkbox(f"{c}: higher is better",value=True,key=f"tp_{c}"))
            new=st.text_input("New score","topsis_score",key="tp_new")
            if st.button("COMPUTE TOPSIS",disabled=not crit):
                try:
                    out=df.copy();out[new]=topsis(out,crit,benefit=benefits)
                    set_df(out,st.session_state.source_name,st.session_state.column_map)
                    st.success(f"Created `{new}`.")
                except Exception as e: st.exception(e)

    with tabs[6]:
        y=st.selectbox("Outcome to analyse",["None"]+df.columns.tolist(),key="assist_y")
        st.markdown(assistant_recommendation(df,None if y=="None" else y))
        st.caption("This assistant is rule-based and transparent: it does not send your dataset to an external AI service.")

# ============================================================
# SIMULATOR
# ============================================================
elif page=="🎮 Simulator":
    df=need_data()
    hero("Simulator","Turn an estimated model into profiles, predictions and counterfactual comparisons.","Prediction playground")
    usable=[m for m in st.session_state.models
            if m.result is not None and hasattr(m.result,"predict")
            and m.family in ("OLS","WLS","Linear Probability Model","Logit","Probit","Cloglog","Poisson","Negative Binomial")]
    if not usable:
        st.info("Estimate a formula-based cross-sectional model first.")
        st.stop()
    mn=st.selectbox("Model",[m.name for m in usable])
    model=next(m for m in usable if m.name==mn)
    model_header(model)

    vars_in=[c for c in df.columns if c in model.formula and c!=model.y_name]
    if not vars_in:
        st.info("No editable regressors detected in the formula.")
        st.stop()

    st.markdown("### Compare two profiles")
    profile_a={};profile_b={}
    ca,cb=st.columns(2)
    for c in vars_in:
        if pd.api.types.is_numeric_dtype(df[c]):
            lo=float(df[c].quantile(.01));hi=float(df[c].quantile(.99));med=float(df[c].median())
            if np.isfinite(lo) and np.isfinite(hi) and lo<hi:
                with ca: profile_a[c]=st.slider(c+" · A",lo,hi,med,key=f"sim_a_{c}")
                with cb: profile_b[c]=st.slider(c+" · B",lo,hi,med,key=f"sim_b_{c}")
            else:
                profile_a[c]=profile_b[c]=med
        else:
            vals=df[c].dropna().astype(str).unique().tolist()
            with ca: profile_a[c]=st.selectbox(c+" · A",vals,key=f"sim_a_{c}")
            with cb: profile_b[c]=st.selectbox(c+" · B",vals,key=f"sim_b_{c}")
    try:
        pa=float(np.asarray(model.result.predict(pd.DataFrame([profile_a]))).ravel()[0])
        pb=float(np.asarray(model.result.predict(pd.DataFrame([profile_b]))).ravel()[0])
        c1,c2,c3=st.columns(3)
        c1.metric("Profile A",f"{pa:.4f}")
        c2.metric("Profile B",f"{pb:.4f}")
        if model.family in ("Logit","Probit","Cloglog","Linear Probability Model"):
            c3.metric("Difference",f"{(pb-pa)*100:+.2f} pp")
        else:
            c3.metric("Difference",f"{pb-pa:+.4f}")
        comp=pd.DataFrame({"Variable":vars_in,"Profile A":[profile_a[v] for v in vars_in],"Profile B":[profile_b[v] for v in vars_in]})
        st.dataframe(comp,use_container_width=True,hide_index=True)
    except Exception as e:
        st.exception(e)

# ============================================================
# REPORT
# ============================================================
elif page=="📑 Report":
    df=need_data()
    hero("Report","Compare specifications, export results and recover reproducible code.","Results & reproducibility")
    tabs=st.tabs(["Model table","Export","Code","Manage"])
    models=st.session_state.models
    with tabs[0]:
        if not models:
            st.info("No saved models.")
        else:
            chosen_names=st.multiselect("Models",[m.name for m in models],default=[m.name for m in models],key="rep_models")
            chosen=[m for m in models if m.name in chosen_names]
            if chosen:
                table=comparison_table(chosen)
                st.dataframe(table,use_container_width=True)
                st.caption("* p<0.10, ** p<0.05, *** p<0.01. Standard errors in parentheses.")
    with tabs[1]:
        if not models:
            st.info("No saved models.")
        else:
            chosen_names=st.multiselect("Models to export",[m.name for m in models],default=[m.name for m in models],key="export_models")
            chosen=[m for m in models if m.name in chosen_names]
            c1,c2,c3=st.columns(3)
            if chosen:
                c1.download_button("Excel",models_excel(chosen),"econometrics_lab_results.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)
                c2.download_button("Word",docx_report(chosen),"econometrics_lab_report.docx","application/vnd.openxmlformats-officedocument.wordprocessingml.document",use_container_width=True)
                c3.download_button("PDF",pdf_report(chosen),"econometrics_lab_report.pdf","application/pdf",use_container_width=True)
            st.download_button("Current dataset (CSV)",df.to_csv(index=False).encode(),"analysis_dataset.csv","text/csv")
    with tabs[2]:
        family=st.selectbox("Model",["OLS","Logit","Probit","Poisson"],key="code_family")
        y=st.selectbox("Dependent variable",df.columns.tolist(),key="code_y")
        x=st.multiselect("Regressors",[c for c in df.columns if c!=y],key="code_x")
        lang=st.radio("Language",["Python","R","Stata","Gretl"],horizontal=True,key="code_lang")
        code=generate_code(lang,family,y,x)
        st.code(code,language={"Python":"python","R":"r","Stata":"stata","Gretl":"text"}[lang])
    with tabs[3]:
        if not models:
            st.info("No saved models.")
        else:
            st.dataframe(pd.DataFrame([{"Model":m.name,"Family":m.family,"Created":m.created,"Formula":m.formula} for m in models]),use_container_width=True,hide_index=True)
            delete=st.selectbox("Delete model",["None"]+[m.name for m in models])
            c1,c2=st.columns(2)
            if c1.button("Delete selected",disabled=delete=="None",use_container_width=True):
                st.session_state.models=[m for m in models if m.name!=delete]
                if st.session_state.current_model and st.session_state.current_model.name==delete:
                    st.session_state.current_model=None
                st.rerun()
            if c2.button("Clear all models",use_container_width=True):
                st.session_state.models=[];st.session_state.current_model=None;st.rerun()
