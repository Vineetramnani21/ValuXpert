import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from Forecasting import sale_analysis
from RatioAnalysis import run
from WACC import run_wacc_app
from FCFF_final import run_fcff_analysis
from Intrinsic_Growth import run_fcff_roic

# Helper Functions
EBIT_margins = None
def convert_to_full_year(year_str):
    if pd.isna(year_str):
        return None
    if isinstance(year_str, (int, float)):
        return int(year_str)
    try:
        date = datetime.strptime(str(year_str).strip(), "%b-%y")
        full_year = date.year
        return full_year if full_year <= datetime.now().year else full_year - 100
    except ValueError:
        try:
            return int(float(year_str))
        except ValueError:
            return None


def is_valid_year(s):
    result = convert_to_full_year(s)
    return result is not None and 1900 <= result <= 2100



def analyze_historical_statement(df):
    global historical_data, EBIT_margins
    try:
        year_row = None
        for i, row in df.iterrows():
            if row.iloc[1:].apply(is_valid_year).any():
                year_row = i
                break

        if year_row is None:
            raise ValueError("Could not find a row with valid year data")

        years = df.iloc[year_row, 1:].values
        valid_years = [convert_to_full_year(y) for y in years if is_valid_year(y)]
        years = np.array(valid_years)

        def to_numeric_array(arr):
            return np.array([float(str(x).replace(',', '').strip()) if pd.notna(x) else 0 for x in arr])

        raw_material = to_numeric_array(df.iloc[year_row + 2, 1:len(valid_years) + 1].values)
        inventory = to_numeric_array(df.iloc[year_row + 3, 1:len(valid_years) + 1].values)
        power_fuel = to_numeric_array(df.iloc[year_row + 4, 1:len(valid_years) + 1].values)
        other_mfr_exp = to_numeric_array(df.iloc[year_row + 5, 1:len(valid_years) + 1].values)
        employee_cost = to_numeric_array(df.iloc[year_row + 6, 1:len(valid_years) + 1].values)
        selling_and_gen_exp1 = to_numeric_array(df.iloc[year_row + 7, 1:len(valid_years) + 1].values)
        selling_and_gen_exp2 = to_numeric_array(df.iloc[year_row + 8, 1:len(valid_years) + 1].values)
        selling_and_gen_exp = selling_and_gen_exp1 + selling_and_gen_exp2
        depreciation = to_numeric_array(df.iloc[year_row+10,1:len(valid_years)+1].values)
        Interest = to_numeric_array(df.iloc[year_row+11,1:len(valid_years)+1].values)
        Tax = to_numeric_array(df.iloc[year_row+13,1:len(valid_years)+1].values)
        sales = to_numeric_array(df.iloc[year_row + 1, 1:len(valid_years) + 1].values)
        sales_growth = np.zeros(len(sales))
        sales_growth[1:] = ((sales[1:] / sales[:-1]) - 1)*100
        NoofEquityShares = to_numeric_array(df.iloc[year_row + 77, 1:len(valid_years)+1].values)
        Dividendamount = to_numeric_array(df.iloc[year_row + 15, 1:len(valid_years)+1].values)
        EquitySCap = to_numeric_array(df.iloc[year_row + 41, 1:len(valid_years)+1].values)
        Reserves =to_numeric_array( df.iloc[year_row + 42, 1:len(valid_years)+1].values)
        Borrowings = to_numeric_array(df.iloc[year_row + 43, 1:len(valid_years)+1].values)
        OtherL = to_numeric_array(df.iloc[year_row + 44, 1:len(valid_years)+1].values)
        TotalL= to_numeric_array(df.iloc[year_row + 45, 1:len(valid_years)+1].values)
        NetBlock =to_numeric_array( df.iloc[year_row + 46, 1:len(valid_years)+1].values)
        CWP= to_numeric_array( df.iloc[year_row + 47, 1:len(valid_years)+1].values)
        Investment =to_numeric_array( df.iloc[year_row + 48, 1:len(valid_years)+1].values)
        Otherasset =to_numeric_array(df.iloc[year_row + 49, 1:len(valid_years)+1].values)
        TNCA =to_numeric_array(df.iloc[year_row + 50, 1:len(valid_years)+1].values)
        Receivables =to_numeric_array( df.iloc[year_row + 51, 1:len(valid_years)+1].values) 
        Inventory =to_numeric_array( df.iloc[year_row + 52, 1:len(valid_years)+1].values)
        CB =to_numeric_array(df.iloc[year_row + 53, 1:len(valid_years)+1].values)
        COGS = (raw_material + power_fuel + other_mfr_exp + employee_cost) - inventory
        selling_and_gen_expsales = (selling_and_gen_exp/sales)*100
        gross_profit = sales - COGS
        gross_margin = (gross_profit/sales)*100
        EBITDA= gross_profit - selling_and_gen_exp
        EBITDA_growth = np.zeros(len(EBITDA))
        EBITDA_growth[1:] = ((EBITDA[1:] / EBITDA[:-1]) - 1)*100
        ebitda_margin = (EBITDA/sales)*100
        EBT= EBITDA - depreciation - Interest
        DepreciationSales= (depreciation/sales)*100
        EBT_growth = np.zeros(len(EBT))
        EBT_growth[1:] = ((EBT[1:] / EBT[:-1]) - 1)*100
        EBT_margins= (EBT/sales)*100
        EBIT_margins=(EBITDA-depreciation/sales)*100
        #st.write(EBIT_margins)
        ReturnonCap =((EBITDA-depreciation)/(EquitySCap+Reserves+Borrowings))*100
        OperatingSales= EBIT_margins
        InterestCovR = (EBITDA-depreciation)/Interest
        DebtTurnR= sales/Receivables
        CredTurnR = sales/OtherL
        InventoryTurnR =sales/Inventory
        FixedassetTurnR= sales/NetBlock
        CapiTurnR= sales/ EquitySCap+Reserves
        DebtorDays= 365/DebtTurnR
        PayDays =365/CredTurnR
        InventoryDay= 365/InventoryTurnR
        CashConv = ((DebtorDays+InventoryDay)-PayDays)
        Net_Profit= EBT - Tax
        Net_Profit_growth = np.zeros(len(Net_Profit))
        Net_Profit_growth[1:] = ((Net_Profit[1:] / Net_Profit[:-1]) - 1)*100
        RetEquiP = (Net_Profit/EquitySCap+Reserves)*100
        Net_Margins= Net_Profit/sales*100
        EPS= Net_Profit/ NoofEquityShares
        DivPShare= Dividendamount/NoofEquityShares
        DivPayR= (DivPShare/EPS)*100
        Dividend_growth = np.zeros(len(DivPShare))
        Dividend_growth[1:] = ((DivPShare[1:] / DivPShare[:-1]) - 1)*100
        RetaEarn = 100 -DivPayR
        RetaEarnP = RetaEarn+0
        SelfSustainG = RetaEarnP*RetEquiP
        Total_Current_Assets= Receivables + Inventory + CB
        Total_assets = TNCA + Total_Current_Assets
        Other_assets = Otherasset-Receivables-Inventory-CB


        # Create a DataFrame with results
        data = {
            'Year': years,
            'Sales': sales,  # Add Sales to the DataFrame
            'Sales Growth': sales_growth,
            'COGS': COGS,
            'Gross Profit': gross_profit, # Add Gross Profit
            'Gross Margin': gross_margin,
            'Selling and general expense':selling_and_gen_exp,
            'S&G Exp % Sales':selling_and_gen_expsales,
            'EBITDA' : EBITDA,
            'EBITDA Growth':EBITDA_growth,
            'EBITDA Margin':ebitda_margin,
            'Depreciation' : depreciation,
            'Depreciation % Sales': DepreciationSales,
            'Interest': Interest,
            'EBT': EBT,
            'EBT Growth': EBT_growth,
            'EBT Margin':EBT_margins,
            'Tax':Tax,
            'Net Profit':Net_Profit,
            'Net Profit Growth':Net_Profit_growth,
            'Net Margins': Net_Margins,
            'No. of Equity Shares': NoofEquityShares,
            'Earning Per Share':EPS,
            'Dividend Per Share':DivPShare,
            'Dividend Payout Ratio':DivPayR,
            'Dividend Growth': Dividend_growth,
            'Retained Earnings':RetaEarn
        }
        
        result_df = pd.DataFrame(data)
        data1 = {
            'Equity Share Capital': EquitySCap,
            'Reserves':Reserves,
            'Borrowings': Borrowings,
           'Other Liabilities':OtherL,
            'Total Liabilities':TotalL,
            'Fixed Assets Net Block':NetBlock,
            'Capital Work in Progress':CWP,
            'Investments':Investment,
            'Other Assets':Other_assets,
           'Total Non-Current Assets':TNCA,
            'Receivables':Receivables,
            'Inventory':Inventory,
           'Cash & Bank':CB,
            'Total Current Assets':Total_Current_Assets,
           'Total Assets': Total_assets,
          }
        result_df1 = pd.DataFrame(data1)
        st.subheader("📄 Income Statement Analysis")
        st.dataframe(result_df, use_container_width=True)
        st.subheader("📄 Balance Sheet Analysis")
        st.dataframe(result_df1, use_container_width=True)
        csv1 = result_df.to_csv(index=False)
        st.download_button("Download Income Statement Analysis", csv1, "income_statement_analysis.csv", "text/csv")
        csv2 = result_df1.to_csv(index=False)
        st.download_button("Download Balance Sheet Analysis", csv2, "balance_sheet_analysis.csv", "text/csv")
        # Visualization
        st.subheader("📊 Historical Analysis Graphs")
        fig, ax = plt.subplots()
        ax.plot(result_df['Year'], result_df['Sales'], label='Sales', marker='o')
        ax.plot(result_df['Year'], result_df['COGS'], label='COGS', marker='o')
        ax.plot(result_df['Year'], result_df['Gross Profit'], label='Gross Profit', marker='o')
        ax.set_title("Sales, COGS, and Gross Profit Over Years")
        ax.set_xlabel("Year")
        ax.set_ylabel("Amount")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error in historical statement analysis: {str(e)}")
        return None



def calculate_wacc():
    st.subheader("🔮 Weighted Average Cost of Capital (WACC)")
    # Add WACC logic and display


def intrinsic_growth_analysis():
    st.subheader("📈 Intrinsic Growth Analysis")
    # Add intrinsic growth logic and display






def fcff_analysis():
    st.subheader("📉 Free Cash Flow to Firm (FCFF) Analysis")
    # Add FCFF logic and display

         


# Main App Function
def demo():
    st.title("📊ValuXpert : The Financial Modeling Tool")
   

    st.markdown(
        """
        <style>
        .central-content {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            margin: auto;
        }
        .button-container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            justify-items: center;
            margin: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.sidebar.file_uploader("📂 Upload CSV File", type=['csv'])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, skiprows=2)
            st.sidebar.success("File uploaded successfully!")

            st.markdown('<div class="central-content">', unsafe_allow_html=True)
            st.subheader("🔍 Select an Analysis Option")

            st.markdown('<div class="button-container">', unsafe_allow_html=True)
            if st.button("📜 Historical Analysis"):
                with st.spinner("Processing historical statements..."):
                    result_df = analyze_historical_statement(df)
                    #st.write(f"hlo:{EBIT_margins}")
                    
                    #desired_var = result_df["EBIT_margins"]
                    #st.write(desired_var)
                    #st.write(result_df. EBIT_margins)
                    if result_df is not None:
                        st.success("Analysis complete!")
                        
                        st.dataframe(result_df, use_container_width=True)
                        csv = result_df.to_csv(index=False)
                        st.download_button("Download Historical Analysis", csv, "historical_analysis.csv", "text/csv")

            if st.button("📊 Ratio Analysis"):
                run(df)
                
            if st.button("🔮 WACC"):
                run_wacc_app()

            if st.button("🌟 Intrinsic Growth"):
                run_fcff_roic()

            if st.button("📉 FCFF"):
                run_fcff_analysis()

            if st.button("🔮 Forecasting"):
                sale_analysis(df)
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.info("Upload a CSV file to begin.")

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    demo()
