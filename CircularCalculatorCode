import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import math

# Set page configuration to make it more compact
st.set_page_config(layout="wide")

# Custom CSS to reduce padding and make the layout more compact
st.markdown("""
    <style>
        .reportview-container .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            max-width: 1200px;
        }
        .stPlotly, .stplot {
            height: 400px !important;
        }
        h1, h2, h3 {
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit Webpage
st.title("Circular Economy Calculator - Rubber Industry")

# Four columns layout
col1, col2, col3, col4 = st.columns(4)

# Footwear Company Inputs
with col1:
    st.subheader("Footwear Company")

    st.markdown("*Variable Costs - F*")
    vMC_Foot = st.number_input("Virgin Material Costs - F ($/Kg)", value=2.28, step=0.01, format="%.2f")
    vMPC_Foot = st.number_input("Virgin Material Procurement Costs - F ($/Kg)", value=1.00, step=0.01, format="%.2f")
    rMC_Foot = st.number_input("Recycled Material Costs - F ($/Kg)", value=0.5, step=0.01, format="%.2f")
    rMPC_Foot = st.number_input("Recycled Material Procurement Costs - F ($/Kg)", value=0.5, step=0.01, format="%.2f")
    oVC_Foot = st.number_input("Other Variable Costs - F ($/unit)", value=35.00, step=0.01, format="%.2f")

    st.markdown("*Fixed Costs - F*")
    oFC_Foot = st.number_input("Other Fixed Costs - F (million $)", value=100.00, step=0.01, format="%.2f")
    CI_Foot = st.number_input("Circular Investments - F (million $)", value=55.0, step=0.01, format="%.2f")

with col2:
    st.subheader("Footwear Company (cont.)")

    st.markdown("*Total Revenue - F*")
    SP_Foot = st.number_input("Selling Price - F ($)", value=70.00, step=0.01, format="%.2f")
    D_Foot = st.number_input("Demand - F (million units)", value=220.00, step=0.01, format="%.2f")
    MS_Foot = st.number_input("Market Share - F (%)", value=10.00, step=0.01, format="%.2f") / 100

    st.markdown("*Design - F*")
    RC_Foot = st.number_input("Recycled Content - F (%)", value=70.00, step=0.01, format="%.2f") / 100
    Q_Foot = st.number_input("Material Use - F (Kg)", value=1.00, step=0.01, format="%.2f")

# Tyre Company Inputs
with col3:
    st.subheader("Tyre Company")

    st.markdown("*Variable Costs - T*")
    vMC_Tyre = st.number_input("Virgin Material Costs - T ($/Kg)", value=2.28, step=0.01, format="%.2f")
    vMPC_Tyre = st.number_input("Virgin Material Procurement Costs - T ($/Kg)", value=1.00, step=0.01, format="%.2f")
    rMC_Tyre = st.number_input("Recycled Material Costs - T ($/Kg)", value=0.3, step=0.01, format="%.2f")
    rMPC_Tyre = st.number_input("Recycled Material Procurement Costs - T ($/Kg)", value=0.5, step=0.01, format="%.2f")
    oVC_Tyre = st.number_input("Other Variable Costs - T ($/unit)", value=55.00, step=0.01, format="%.2f")

    st.markdown("*Fixed Costs -T*")
    oFC_Tyre = st.number_input("Other Fixed Costs - T (million $)", value=2500.00, step=0.01, format="%.2f")
    CI_Tyre = st.number_input("Circular Investments - T (million $)", value=780.00, step=0.01, format="%.2f")

with col4:
    st.subheader("Tyre Company (cont.)")

    st.markdown("*Total Revenue - T*")
    SP_Tyre= st.number_input("Selling Price - T ($)", value=150.00, step=0.01, format="%.2f")
    D_Tyre = st.number_input("Demand - T (million units)", value=450.00, step=0.01, format="%.2f")
    MS_Tyre = st.number_input("Market Share - T (%)", value=25.00, step=0.01, format="%.2f") / 100

    st.markdown("*Design - T*")
    RC_Tyre = st.number_input("Recycled Content - T (%)", value=10.00, step=0.01, format="%.2f") / 100
    Q_Tyre = st.number_input("Material Use - T (Kg)", value=10.00, step=0.01, format="%.2f")

# Function to calculate and plot break-even points
def calculate_and_plot_breakeven(company_name, FC, CI, SP, VC, Num, RC, Q, vMC, vMPC, rMC, rMPC, oVC):
    num_range = np.arange(1, Num * 2, step=10)

    total_circular_costs = (FC + CI) + (VC / Num * num_range)
    total_linear_costs = FC + (((vMC * vMPC * Q) + oVC) * num_range)
    revenue = SP * num_range

    breakeven_revenue_vs_linear = FC / (SP - ((vMC * vMPC * Q) + oVC))
    breakeven_revenue_vs_circular = (FC + CI) / (SP - (VC / Num))
    circular_benefit_point = CI / (((vMC * vMPC * Q) + oVC) - (VC / Num))

    plt.figure(figsize=(6, 4))
    plt.plot(num_range, revenue, label="Revenue")
    plt.plot(num_range, total_circular_costs, label="Costs (Circular)")
    plt.plot(num_range, total_linear_costs, label="Costs (Linear)", linestyle='--')

    plt.axvline(x=breakeven_revenue_vs_linear, color='r', linestyle='--',
                label=f"BEP: Rev vs Lin ({breakeven_revenue_vs_linear:.2f})")
    plt.axvline(x=breakeven_revenue_vs_circular, color='g', linestyle='--',
                label=f"BEP: Rev vs Cir ({breakeven_revenue_vs_circular:.2f})")
    plt.axvline(x=circular_benefit_point, color='b', linestyle='--',
                label=f"Lin vs Cir ({circular_benefit_point:.2f})")

    plt.title(f"{company_name} - Breakeven Point")
    plt.xlabel(f"{company_name} Demand (million units)")
    plt.ylabel("Cost, Revenue (million $)")
    plt.legend(fontsize='x-small')
    plt.grid(True)

    return plt, {
        'Revenue vs Linear Costs': breakeven_revenue_vs_linear,
        'Revenue vs Circular Costs': breakeven_revenue_vs_circular,
        'Circular Benefit Point': circular_benefit_point
    }


# Calculate values for Footwear Company
Num_Foot = D_Foot * MS_Foot
VC_Foot = (((vMC_Foot * vMPC_Foot * (1 - RC_Foot) + (rMC_Foot * rMPC_Foot * RC_Foot)) * Q_Foot) + oVC_Foot) * Num_Foot
FC_Foot = oFC_Foot + CI_Foot
Rev_Foot = SP_Foot * D_Foot * MS_Foot
P_Foot = Rev_Foot - VC_Foot - FC_Foot

Lin_VC_Foot = (((vMC_Foot * vMPC_Foot * Q_Foot) + oVC_Foot) * Num_Foot)

# Calculate values for Tyre Company
Num_Tyre = D_Tyre * MS_Tyre
VC_Tyre = (((vMC_Tyre * vMPC_Tyre * (1 - RC_Tyre) + (rMC_Tyre * rMPC_Tyre * RC_Tyre)) * Q_Tyre) + oVC_Tyre) * Num_Tyre
FC_Tyre = oFC_Tyre + CI_Tyre
Rev_Tyre = SP_Tyre * D_Tyre * MS_Tyre
P_Tyre = Rev_Tyre - VC_Tyre - FC_Tyre

Lin_VC_Tyre = ((vMC_Tyre * vMPC_Tyre * Q_Tyre) + oVC_Tyre) * Num_Tyre

# Plot break-even graphs side by side
col1, col2 = st.columns(2)

with col1:
    plt_foot, bep_points_foot = calculate_and_plot_breakeven("Footwear", oFC_Foot, CI_Foot, SP_Foot, VC_Foot, Num_Foot,
                                                            RC_Foot, Q_Foot, vMC_Foot, vMPC_Foot, rMC_Foot, rMPC_Foot,
                                                            oVC_Foot)
    st.pyplot(plt_foot)

with col2:
    plt_tyre, bep_points_tyre = calculate_and_plot_breakeven("Tyre", oFC_Tyre, CI_Tyre, SP_Tyre, VC_Tyre, Num_Tyre,
                                                            RC_Tyre, Q_Tyre, vMC_Tyre, vMPC_Tyre, rMC_Tyre, rMPC_Tyre,
                                                            oVC_Tyre)
    st.pyplot(plt_tyre)

# Create a table for break-even results
bep_results = pd.DataFrame({
    'Break-even Point': ['Revenue vs Linear Costs', 'Revenue vs Circular Costs', 'Circular Benefit Point'],
    'Footwear': [f"{bep_points_foot['Revenue vs Linear Costs']:.2f}", f"{bep_points_foot['Revenue vs Circular Costs']:.2f}",
                 f"{bep_points_foot['Circular Benefit Point']:.2f}"],
    'Tyre': [f"{bep_points_tyre['Revenue vs Linear Costs']:.2f}", f"{bep_points_tyre['Revenue vs Circular Costs']:.2f}",
             f"{bep_points_tyre['Circular Benefit Point']:.2f}"]
})

st.write("### Breakeven Point")
st.table(bep_results)

# Profit Calculation
P_Foot = Rev_Foot - VC_Foot - FC_Foot
P_Foot_Margin = (P_Foot / Rev_Foot) * 100
P_Foot_Lin = Rev_Foot - Lin_VC_Foot - FC_Foot
P_Foot_Margin_Lin = (P_Foot_Lin / Rev_Foot) * 100
#Cir_Profit_Difference_Foot = P_Foot - P_Foot_Lin

P_Tyre = Rev_Tyre - VC_Tyre - FC_Tyre
P_Tyre_Margin = (P_Tyre / Rev_Tyre) * 100
P_Tyre_Lin = Rev_Tyre - Lin_VC_Tyre - FC_Tyre
P_Tyre_Margin_Lin = (P_Tyre_Lin / Rev_Tyre) * 100
#Cir_Profit_Difference_Tyre = P_Tyre - P_Tyre_Lin

# Virgin Material Saved
Material_Saved_Foot = (Num_Foot * vMC_Foot * vMPC_Foot * Q_Foot) - (((vMC_Foot * vMPC_Foot * (1 - RC_Foot)) + (rMC_Foot * rMPC_Foot * RC_Foot)) * Q_Foot * Num_Foot)

Material_Saved_Tyre = (Num_Tyre * vMC_Tyre * vMPC_Tyre * Q_Tyre) - (((vMC_Tyre * vMPC_Tyre * (1 - RC_Tyre)) + (rMC_Tyre * rMPC_Tyre * RC_Tyre)) * Q_Tyre * Num_Tyre)

# Display the results
col1, col2 = st.columns(2)

with col1:
    st.markdown("***Results - Footwear***")
    st.write(f"Total Costs (million $): {VC_Foot + FC_Foot}")
    st.write(f"Total Revenue (million $): {Rev_Foot}")
    st.write(f"Profit (million $): {P_Foot}")
    st.write(f"Profit Margin (%): {P_Foot_Margin:.2f}%")
    st.write(f"Linear Profit (million $): {P_Foot_Lin:.2f}")
    st.write(f"Linear Profit Margin (%): {P_Foot_Margin_Lin:.2f}%")
    st.write(f"Material Cost Savings (million $): {Material_Saved_Foot:.2f}")

with col2:
    st.markdown("***Results - Tyre***")
    st.write(f"Total Costs (million $): {VC_Tyre + FC_Tyre}")
    st.write(f"Total Revenue (million $): {Rev_Tyre}")
    st.write(f"Profit (million $): {P_Tyre}")
    st.write(f"Profit Margin (%): {P_Tyre_Margin:.2f}%")
    st.write(f"Linear Profit (million $): {P_Tyre_Lin}")
    st.write(f"Linear Profit Margin (%): {P_Tyre_Margin_Lin:.2f}%")
    st.write(f"Material Cost Savings (million $): {Material_Saved_Tyre:.2f}")

# Manually inputting the Date and updated Price (GBP/kg) data as a DataFrame
data = {
    'Date': [
        'Jan-20', 'Feb-20', 'Mar-20', 'Apr-20', 'May-20', 'Jun-20', 'Jul-20', 'Aug-20', 'Sep-20', 'Oct-20', 'Nov-20', 'Dec-20',
        'Jan-21', 'Feb-21', 'Mar-21', 'Apr-21', 'May-21', 'Jun-21', 'Jul-21', 'Aug-21', 'Sep-21', 'Oct-21', 'Nov-21', 'Dec-21',
        'Jan-22', 'Feb-22', 'Mar-22', 'Apr-22', 'May-22', 'Jun-22', 'Jul-22', 'Aug-22', 'Sep-22', 'Oct-22', 'Nov-22', 'Dec-22',
        'Jan-23', 'Feb-23', 'Mar-23', 'Apr-23', 'May-23', 'Jun-23', 'Jul-23', 'Aug-23', 'Sep-23', 'Oct-23', 'Nov-23', 'Dec-23',
        'Jan-24', 'Feb-24', 'Mar-24', 'Apr-24'
    ],
    'Price ($/kg)': [
    1.68, 1.61, 1.50, 1.33, 1.35, 1.40, 1.48, 1.70, 1.86, 2.19, 2.30, 2.33, 2.30, 2.35, 2.37,
    2.15, 2.29, 2.12, 1.87, 1.90, 1.79, 1.87, 1.93, 1.92, 1.97, 2.11, 2.12, 2.09, 2.06, 2.03,
    1.78, 1.61, 1.48, 1.50, 1.43, 1.54, 1.63, 1.62, 1.58, 1.54, 1.56, 1.53, 1.49, 1.47, 1.55,
    1.61, 1.67, 1.66, 1.80, 2.02, 2.39, 2.28
]
}

# Convert the data into a DataFrame
df_rubber_clean = pd.DataFrame(data)

# Convert the Date column to a proper datetime format
df_rubber_clean['Date'] = pd.to_datetime(df_rubber_clean['Date'], format='%b-%y')

# Assuming stable recycled material prices for both Footwear and Tyre companies
stable_recycled_footwear = (1 - RC_Foot) * df_rubber_clean['Price ($/kg)'] + (RC_Foot * rMC_Foot)
stable_recycled_tyre = (1 - RC_Tyre) * df_rubber_clean['Price ($/kg)'] + (RC_Tyre * rMC_Tyre)

# Calculate mean and standard deviation for all three price series
mean_price = df_rubber_clean['Price ($/kg)'].mean()
std_price = df_rubber_clean['Price ($/kg)'].std()

mean_footwear = stable_recycled_footwear.mean()
std_footwear = stable_recycled_footwear.std()

mean_tyre = stable_recycled_tyre.mean()
std_tyre = stable_recycled_tyre.std()

# Create a range of prices for the normal distributions
price_range = np.linspace(
    min(df_rubber_clean['Price ($/kg)'].min(), stable_recycled_footwear.min(), stable_recycled_tyre.min()),
    max(df_rubber_clean['Price ($/kg)'].max(), stable_recycled_footwear.max(), stable_recycled_tyre.max()),
    100
)

# Calculate the normal distributions
price_distribution = stats.norm.pdf(price_range, mean_price, std_price)
footwear_distribution = stats.norm.pdf(price_range, mean_footwear, std_footwear)
tyre_distribution = stats.norm.pdf(price_range, mean_tyre, std_tyre)

st.write("### Price Variation")

# Create two columns for the plots
col1, col2 = st.columns(2)

with col1:
    # Plot for rubber price fluctuation
    plt.figure(figsize=(10, 6))
    plt.plot(df_rubber_clean['Date'], df_rubber_clean['Price ($/kg)'], label='Virgin Rubber Price')
    plt.plot(df_rubber_clean['Date'], stable_recycled_footwear, label='Resultant Price (Footwear)')
    plt.plot(df_rubber_clean['Date'], stable_recycled_tyre, label='Resultant Price (Tyre)')
    plt.title("Rubber Price Variation")
    plt.xlabel("Date")
    plt.ylabel("Price ($/kg)")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

with col2:
    # Plot for normal distributions
    plt.figure(figsize=(10, 6))
    plt.hist(df_rubber_clean['Price ($/kg)'], bins=20, density=True, alpha=0.3, color='red', label='Virgin Rubber Prices')
    plt.hist(stable_recycled_footwear, bins=20, density=True, alpha=0.3, color='blue', label='Resultant Price (Footwear)')
    plt.hist(stable_recycled_tyre, bins=20, density=True, alpha=0.3, color='grey', label='Resultant Price (Tyre)')
    plt.plot(price_range, price_distribution, 'red', lw=2, label='Normal Dist. (Virgin Rubber)')
    plt.plot(price_range, footwear_distribution, 'blue', lw=2, label='Normal Dist. (Footwear)')
    plt.plot(price_range, tyre_distribution, 'grey', lw=2, label='Normal Dist. (Tyre)')
    plt.axvline(mean_price, color='red', linestyle='dashed', linewidth=2)
    plt.axvline(mean_footwear, color='blue', linestyle='dashed', linewidth=2)
    plt.axvline(mean_tyre, color='grey', linestyle='dashed', linewidth=2)
    plt.title("Normal Distributions of Rubber Prices")
    plt.xlabel("Price ($/kg)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Create a dataframe for the price statistics
price_stats = pd.DataFrame({
    'Metric': ['Mean ($/kg)', 'Standard Deviation ($/kg)', 'Lower 68% Range ($/kg)', 'Upper 68% Range ($/kg)'],
    'Virgin Rubber': [
        f"{mean_price:.4f}",
        f"{std_price:.4f}",
        f"{mean_price - std_price:.4f}",
        f"{mean_price + std_price:.4f}"
    ],
    'Resultant (Footwear)': [
        f"{mean_footwear:.4f}",
        f"{std_footwear:.4f}",
        f"{mean_footwear - std_footwear:.4f}",
        f"{mean_footwear + std_footwear:.4f}"
    ],
    'Resultant (Tyre)': [
        f"{mean_tyre:.4f}",
        f"{std_tyre:.4f}",
        f"{mean_tyre - std_tyre:.4f}",
        f"{mean_tyre + std_tyre:.4f}"
    ]
})

# Display the table
st.table(price_stats)

# Calculate and display price variation percentages
actual_variation = (std_price / mean_price) * 100
footwear_variation = (std_footwear / mean_footwear) * 100
tyre_variation = (std_tyre / mean_tyre) * 100

Fluc_Decrease_Foot = (footwear_variation - actual_variation) * 100 / actual_variation
Fluc_Decrease_Tyre = (tyre_variation - actual_variation) * 100 / actual_variation

st.markdown("***Decrease in Price Variation***")
variation_stats = pd.DataFrame({
    'Metric': ['Price Variation (%)', 'Decrease in Price Variation (%)'],
    'Actual Rubber': [f"{actual_variation:.2f}%", "-"],
    'Resultant (Footwear)': [f"{footwear_variation:.2f}%", f"{Fluc_Decrease_Foot:.2f}%"],
    'Resultant (Tyre)': [f"{tyre_variation:.2f}%", f"{Fluc_Decrease_Tyre:.2f}%"]
})
st.table(variation_stats)

#New

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("Footwear Growth Rates")
    DG_Foot = st.number_input("Demand Growth Rate - F (%/year)", value=4.00, step=0.01, format="%.2f") / 100
    SPG_Foot = st.number_input("Selling Price Growth Rate - F (%/year)", value=3.00, step=0.01, format="%.2f") / 100
    vMCG_Foot = st.number_input("Virgin Material Costs Growth Rate - F (%/year)", value=3.00, step=0.01, format="%.2f") / 100
    rMCG_Foot = st.number_input("Recycled Material Costs Growth Rate - F (%/year)", value=1.00, step=0.01, format="%.2f") / 100
    YI_Foot = st.number_input("Yearly Investments - F (million $/year)", value=300.00, step=0.01, format="%.2f")
    YIG_Foot = st.number_input("Yearly Investment Growth Rate - F (%/year)", value=2.00, step=0.01, format="%.2f") / 100

with col2:
    st.subheader("Tyre Growth Rates")
    DG_Tyre = st.number_input("Demand Growth Rate - T (%/year)", value=6.15, step=0.01, format="%.2f") / 100
    SPG_Tyre = st.number_input("Selling Price Growth Rate - T (%/year)", value=3.00, step=0.01, format="%.2f") / 100
    vMCG_Tyre = st.number_input("Virgin Material Costs Growth Rate - T (%/year)", value=3.00, step=0.01, format="%.2f") / 100
    rMCG_Tyre = st.number_input("Recycled Material Costs Growth Rate - T (%/year)", value=1.00, step=0.01, format="%.2f") / 100
    YI_Tyre = st.number_input("Yearly Investments - T (million $/year)", value=800.00, step=0.01, format="%.2f")
    YIG_Tyre = st.number_input("Yearly Investment Growth Rate - T (%/year)", value=5.00, step=0.01, format="%.2f") / 100

with col3:
    st.subheader("End-of-Life Routes")
    st.markdown("Footwear")
    W_Foot = st.number_input("Waste - F (%)", value=30.00, step=0.01, format="%.2f") / 100
    DC_Foot = st.number_input("Downcycled - F (%)", value=10.00, step=0.01, format="%.2f") / 100

    st.markdown("Tyre")
    W_Tyre = st.number_input("Waste - T (%)", value=30.00, step=0.01, format="%.2f") / 100
    DC_Tyre = st.number_input("Downcycled - T (%)", value=10.00, step=0.01, format="%.2f") / 100
    Ret_Tyre = st.number_input("Retreaded - T (%)", value=10.00, step=0.01, format="%.2f") / 100

with col4:
    st.subheader("Resource Recovery Company")
    Resource_Recovery_Capacity = st.number_input("Resource Recovery Company Capacity (tonnes/year)", value=70000.00, step=0.01, format="%.2f")

# Create arrays for growing values over 5 years
years = 5

D_Foot_array = [D_Foot * (1 + DG_Foot)**i for i in range(years)]
D_Tyre_array = [D_Tyre * (1 + DG_Tyre)**i for i in range(years)]

vMC_Foot_array = [vMC_Foot * (1 + vMCG_Foot)**i for i in range(years)]
vMC_Tyre_array = [vMC_Tyre * (1 + vMCG_Tyre)**i for i in range(years)]

rMC_Foot_array = [rMC_Foot * (1 + rMCG_Foot)**i for i in range(years)]
rMC_Tyre_array = [rMC_Tyre * (1 + rMCG_Tyre)**i for i in range(years)]

YI_Foot_array = [YI_Foot * (1 + YIG_Foot)**i for i in range(years)]
YI_Tyre_array = [YI_Tyre * (1 + YIG_Tyre)**i for i in range(years)]

# Calculate revenue, costs, and profit for each year
results = []
for year in range(years):
    # Footwear calculations
    Rev_Foot = SP_Foot * MS_Foot * D_Foot_array[year]
    VC_Foot = (((vMC_Foot_array[year] * vMPC_Foot * (1 - RC_Foot) + (rMC_Foot_array[year] * rMPC_Foot * RC_Foot)) * Q_Foot) + oVC_Foot) * D_Foot_array[year] * MS_Foot
    FC_Foot = YI_Foot_array[year]
    P_Foot = Rev_Foot - VC_Foot - FC_Foot
    Lin_VC_Foot = ((vMC_Foot_array[year] * vMPC_Foot * Q_Foot) + oVC_Foot) * D_Foot_array[year] * MS_Foot
    P_Foot_Lin = Rev_Foot - Lin_VC_Foot - FC_Foot

    # Tyre calculations
    Rev_Tyre = SP_Tyre * MS_Tyre * D_Tyre_array[year]
    VC_Tyre = (((vMC_Tyre_array[year] * vMPC_Tyre * (1 - RC_Tyre) + (rMC_Tyre_array[year] * rMPC_Tyre * RC_Tyre)) * Q_Tyre) + oVC_Tyre) * D_Tyre_array[year] * MS_Tyre
    FC_Tyre = YI_Tyre_array[year]
    P_Tyre = Rev_Tyre - VC_Tyre - FC_Tyre
    Lin_VC_Tyre = ((vMC_Tyre_array[year] * vMPC_Tyre * Q_Tyre) + oVC_Tyre) * D_Tyre_array[year] * MS_Tyre
    P_Tyre_Lin = Rev_Tyre - Lin_VC_Tyre - FC_Tyre

    results.append({
        'Year': year + 1,
        'Footwear Revenue': Rev_Foot,
        'Footwear Profit': P_Foot,
        'Footwear Profit Linear': P_Foot_Lin,
        'Tyre Revenue': Rev_Tyre,
        'Tyre Profit': P_Tyre,
        'Tyre Profit Linear': P_Tyre_Lin
    })

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

st.write("### Revenue and Profit")
col1, col2 = st.columns(2)

with col1:
    st.markdown("***Footwear Company Projections***")
    footwear_projections = results_df[['Year', 'Footwear Revenue', 'Footwear Profit', 'Footwear Profit Linear']]
    footwear_projections = footwear_projections.rename(columns={
        'Footwear Revenue': 'Revenue',
        'Footwear Profit': 'Circular Profit',
        'Footwear Profit Linear': 'Linear Profit'
    })
    st.table(footwear_projections)

with col2:
    st.markdown("***Tyre Company Projections***")
    tyre_projections = results_df[['Year', 'Tyre Revenue', 'Tyre Profit', 'Tyre Profit Linear']]
    tyre_projections = tyre_projections.rename(columns={
        'Tyre Revenue': 'Revenue',
        'Tyre Profit': 'Circular Profit',
        'Tyre Profit Linear': 'Linear Profit'
    })
    st.table(tyre_projections)

# Calculate growth rates
footwear_profit_growth = (results_df['Footwear Profit'].pct_change() * 100).mean()
footwear_profit_growth_linear = (results_df['Footwear Profit Linear'].pct_change() * 100).mean()
tyre_profit_growth = (results_df['Tyre Profit'].pct_change() * 100).mean()
tyre_profit_growth_linear = (results_df['Tyre Profit Linear'].pct_change() * 100).mean()

# Create a summary table
summary_df = pd.DataFrame({
    'Metric': ['Recycled Content (%)', 'Profit Growth Rate (%)', 'Profit Growth Rate (Linear)'],
    'Footwear': [RC_Foot * 100, footwear_profit_growth, footwear_profit_growth_linear],
    'Tyre': [RC_Tyre * 100, tyre_profit_growth, tyre_profit_growth_linear]
})


# Display the summary
st.markdown("***Profit Growth Rate***")
st.table(summary_df)

# Function to calculate circularity
def calculate_circularity(RC, W, DC):
    M_virgin = (1 - RC) * 100
    M_wasted = W * 100
    M_downcycled = DC * 100
    return 100 - (M_virgin + M_wasted + M_downcycled) / 2

# Calculate circularity for Footwear and Tyre
circularity_foot = calculate_circularity(RC_Foot, W_Foot, DC_Foot)
circularity_tyre = calculate_circularity(RC_Tyre, W_Tyre, DC_Tyre)

average_circularity = (circularity_foot + circularity_tyre) / 2

# Display the circularity scores
st.write("### Circularity and Capacity")

# Function to create SVG for circular gauge
def create_gauge_svg(value, color):
    percentage = min(max(value, 0), 100)  # Ensure value is between 0 and 100
    angle = 2 * math.pi * (percentage / 100)
    x = 50 + 40 * math.sin(angle)
    y = 50 - 40 * math.cos(angle)
    large_arc_flag = 1 if percentage > 50 else 0

    svg = f'''
    <svg width="150" height="150" viewBox="0 0 100 100">
      <circle cx="50" cy="50" r="45" fill="none" stroke="#e0e0e0" stroke-width="5"/>
      <path d="M50,50 L50,10 A40,40 0 {large_arc_flag},1 {x},{y} Z" fill="{color}"/>
      <text x="50" y="50" font-family="Arial" font-size="15" fill="black" text-anchor="middle" dy=".3em">{percentage:.1f}%</text>
    </svg>
    '''
    return svg

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("*Footwear Company Circularity*")
    st.markdown(create_gauge_svg(circularity_foot, "#38B6FF"), unsafe_allow_html=True)

with col2:
    st.markdown("*Tyre Company Circularity*")
    st.markdown(create_gauge_svg(circularity_tyre, "#8C8C8C"), unsafe_allow_html=True)

with col3:
    st.markdown("*Average Circularity*")
    st.markdown(create_gauge_svg(average_circularity, "#C1FF72"), unsafe_allow_html=True)

#Calculate Capacity
Recycled_Foot = 1 - W_Foot - DC_Foot
Recycled_Tyre = 1 - W_Tyre - DC_Tyre - Ret_Tyre
RRC_Total = ((Recycled_Foot * Q_Foot * Num_Foot) + (Recycled_Tyre * Q_Tyre * Num_Tyre)) * 1000

RRC_Req = RRC_Total / Resource_Recovery_Capacity

# Create a Capacity table
capacity_df = pd.DataFrame({
    'Metric': ['Quantity of EOL Footwear Recycled (tonnes/year)', 'Quantity of EOL Tyres Recycled (tonnes/year)', 'Total Quantity Recycled (tonnes/year)', 'Number of Resource Recovery Companies Required'],
    'Footwear': [(Recycled_Foot * Q_Foot * Num_Foot * 1000), (Recycled_Tyre * Q_Tyre * Num_Tyre * 1000), RRC_Total, RRC_Req],
})
st.text("")
st.text("")
# Display the summary
st.markdown("***Capacity Summary***")
st.table(capacity_df)
