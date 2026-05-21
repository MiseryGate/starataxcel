# 🏙️ Melbourne's Strata Spectrum — StrataXcel

> Insights on small-scale strata, rental property occupancy, and suburb characteristics across Melbourne's Local Government Areas.

---

## 📋 Table of Contents

- [Background](#background)
- [Problem Definition](#problem-definition)
- [Research Questions](#research-questions)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Demo Prototype](#demo-prototype)
- [Recommendations](#recommendations)
- [Tech Stack](#tech-stack)
- [Data Sources](#data-sources)
- [Team](#team)

---

## 🏢 Background

Victoria's strata sector is significant and growing:

| Metric | Value |
|--------|-------|
| Victorians living in strata | 19% |
| Total strata schemes | 123,180 |
| Full-time strata managers | 1,423 |
| Schemes with 10 lots or fewer | 87% |

> Source: Australasian Strata Highlights 2022, City Futures Research Centre, UNSW Sydney, June 2023.

**StrataXcel** is a start-up offering a building automation system for strata buildings. Renters in Melbourne demand decent living standards by choosing quality rental properties and paying reasonable rent.

---

## 🎯 Problem Definition

### Problem Identification
- Market intelligence gap
- Unstructured and non-actionable information

### Business Needs
- Data summarization
- Geospatial visualization

### Business Values
- Better market understanding
- Improved decision-making for strata stakeholders

### Stakeholders
- Companies & Owners Corporations
- Rental Providers, Agents, and Renters
- Government Bodies

### SWOT Analysis

| | Strengths | Weaknesses |
|---|---|---|
| | Advanced Strata Management Platform | Data privacy concerns |
| | Cost reduction in utilities | Integration challenges |
| | Enhanced efficiency & sustainability | High initial implementation cost |
| | Data-driven decision-making | Limited product capability |
| | Improved security | Needs comprehensive location visualization |

| | Opportunities | Threats |
|---|---|---|
| | Establishing smart buildings | Resistance to change |
| | Creating new potential customers | Cybersecurity risk |
| | Increasing property value | Lack of regulation and SOP |
| | Mapping priority development areas | Bias and ethical concerns |

---

## ❓ Research Questions

**RQ1** — What are the characteristics of small-scale strata and how are they distributed?

**RQ2** — What are the factors affecting the occupancy rate of rental residential properties in suburbs?

---

## 🔬 Methodology

This project applies the **Knowledge Discovery from Databases (KDD)** framework:

```
Data → Selection → Preprocessing → Transformation → Data Mining → Interpretation/Evaluation → Knowledge
```

### Data Sources
- **CoreLogic** — Property data (298,709 rows across 10 LGAs; accessed in batches of 50,000 rows/month)
- **OpenStreetMap** — Geospatial and public facility data via web scraping
- **Crime Statistics Agency** — Offence count data by suburb

### Data Transformation

| Research Question | Approach |
|---|---|
| RQ1 | Rule-based transformation: identifying small-scale strata by total number of units |
| RQ2 | Cleaning, normalising, encoding, and converting targets to binary matrix format for ML |

### Data Mining (RQ2)
- **Objective:** Predict multiple occupancy labels per data instance
- **Method:** Multi-label classification
- **Best Model:** Random Forest (accuracy: 65%)
- **Interpretability:** SHAP (SHapley Additive exPlanations) for feature importance

### Visualisations
- Map visualisation (interactive & heatmap)
- Sankey diagram
- Stacked bar charts

---

## 📊 Key Findings

### RQ1 — Small-Scale Strata Distribution

Top 5 suburbs with the most small-scale strata buildings span **3 council areas**:

| Suburb | Small-Scale Buildings | Council |
|---|---|---|
| Noble Park | 618 | Greater Dandenong |
| Glenroy | 538 | Merri-bek |
| Dandenong | 517 | Greater Dandenong |
| Pascoe Vale | 425 | Merri-bek |
| Elwood | 408 | Port Phillip |

- Most small-scale strata are concentrated within the city centre.
- **Merri-bek** has the highest spatial dispersion index.
- **Port Phillip** has the lowest — making it the most efficient area for targeted outreach.

### RQ1 — Medium-Scale Strata Distribution

Top 5 suburbs span **3 council areas** — Melbourne (1), Boroondara (1), Port Phillip (3):

| Suburb | Medium-Scale Buildings | Council |
|---|---|---|
| South Yarra | 220 | Melbourne |
| St Kilda | 161 | Port Phillip |
| Hawthorn | 134 | Boroondara |
| Elwood | 124 | Port Phillip |
| St Kilda East | 91 | Port Phillip |

- **Melbourne** council has the highest spatial dispersion.
- **Port Phillip** again has the lowest, reinforcing it as the most operationally efficient target area.

---

### RQ2 — Factors Affecting Rental Occupancy

The top 3 variables influencing occupancy rate (identified via SHAP):

| Factor | Impact |
|---|---|
| Average Building Age | **−20%** occupancy vs. average (older buildings = lower occupancy) |
| Number of Hospitals nearby | **+14%** occupancy vs. average |
| Offence Count | **−13%** occupancy vs. average |

**Top suburbs by factor:**

| Hospital Access | Building Age | Offence Count |
|---|---|---|
| Glen Iris | Middle Park | Essendon North |
| Hawthorn East | Armadale | Cremorne |
| South Yarra | Albert Park | Templestowe Lower |

> Most top suburbs are located within **10 km of the CBD**, except Templestowe Lower (> 10 km).

---

## 🖥️ Demo Prototype

The prototype is a multi-page Streamlit application featuring:

- **Dashboard** — Summary tables with council area and suburb analysis (total properties, average/median price, unique buildings)
- **Strata Properties Map** — Interactive map, density heatmap, and scale comparison view
- **Building Detail View** — Unit-level data, price distribution, and property type breakdown
- **Chatbot** — AI-powered Q&A for querying strata insights (e.g. top suburbs for small-scale strata, promotion strategies)

---

## ✅ Final Recommendations

1. **Target Port Phillip first** — lowest spatial dispersion for both small and medium-scale strata, maximising outreach efficiency.
2. **Prioritise newer buildings** — building age is the single largest negative driver of occupancy; StrataXcel's automation system is most compelling for properties at risk of age-related decline.
3. **Leverage hospital proximity** — suburbs near hospitals show significantly higher occupancy; these are high-value acquisition targets.
4. **Avoid high-offence suburbs for initial rollout** — crime rate negatively impacts occupancy and may affect product perception.
5. **Focus on Greater Dandenong and Merri-bek** for volume (highest count of small-scale strata buildings).

---

## 🛠️ Tech Stack

- **Python** — Data processing, ML modelling
- **Scikit-learn** — Random Forest multi-label classification
- **SHAP** — Feature importance & model interpretability
- **Streamlit** — Interactive web prototype
- **Leaflet / OpenStreetMap** — Map visualisation
- **Plotly** — Sankey diagrams and charts
- **Web Scraping** — Public facility data collection

---

## 📁 Data Sources

| Source | Usage |
|---|---|
| [CoreLogic](https://www.corelogic.com.au/) | Property price, building, and strata data |
| [OpenStreetMap](https://www.openstreetmap.org/) | Public facilities (hospitals, schools, parks, etc.) |
| [Crime Statistics Agency Victoria](https://www.crimestatistics.vic.gov.au/) | Offence count by suburb |

---

## 👥 Team

Built with ❤️ by **Group 15**