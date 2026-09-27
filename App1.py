import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Retail Sales Data Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Retail Sales Data Analysis & Business Insights")
st.write(
    "Analyze retail sales data, clean the dataset, create visualizations "
    "and generate useful business insights."
)

# -----------------------------
# Demo Dataset
# -----------------------------
@st.cache_data
def generate_demo_data(n=600):

    np.random.seed(42)

    dates = pd.date_range(
        "2024-01-01",
        "2025-06-30",
        freq="D"
    )

    categories = [
        "Technology",
        "Furniture",
        "Office Supplies"
    ]

    regions = [
        "North",
        "South",
        "East",
        "West"
    ]

    customers = [
        f"CUST-{i:04d}"
        for i in range(1, 101)
    ]

    products = [
        "Laptop",
        "Monitor",
        "Keyboard",
        "Office Chair",
        "Desk",
        "Printer",
        "Notebook",
        "Pen Set",
        "Headphones",
        "Mobile"
    ]

    df = pd.DataFrame({

        "Order Date": np.random.choice(
            dates,
            n
        ),

        "Customer ID": np.random.choice(
            customers,
            n
        ),

        "Product Name": np.random.choice(
            products,
            n
        ),

        "Category": np.random.choice(
            categories,
            n
        ),

        "Region": np.random.choice(
            regions,
            n
        ),

        "Quantity": np.random.randint(
            1,
            10,
            n
        )
    })

    prices = {

        "Laptop": 850,
        "Monitor": 250,
        "Keyboard": 45,
        "Office Chair": 180,
        "Desk": 300,
        "Printer": 220,
        "Notebook": 12,
        "Pen Set": 10,
        "Headphones": 80,
        "Mobile": 500
    }

    df["Sales"] = [
        round(
            prices[product] *
            quantity *
            np.random.uniform(0.80, 1.20),
            2
        )

        for product, quantity
        in zip(
            df["Product Name"],
            df["Quantity"]
        )
    ]

    df["Profit"] = [

        round(
            sales *
            np.random.uniform(0.05, 0.25),
            2
        )

        for sales in df["Sales"]
    ]

    # Add some data-quality issues
    df.loc[5, "Category"] = " technology "
    df.loc[12, "Region"] = "west "
    df.loc[20, "Quantity"] = np.nan
    df.loc[25, "Profit"] = np.nan

    # Add duplicate rows
    df = pd.concat(
        [
            df,
            df.iloc[[2, 7, 15]]
        ],
        ignore_index=True
    )

    return df


# -----------------------------
# Column Cleaning
# -----------------------------
def clean_column_name(column):

    column = str(column).strip()

    column = column.lower()

    column = re.sub(
        r"[^a-z0-9]+",
        "_",
        column
    )

    return column.strip("_")


def standardize_columns(df):

    df = df.copy()

    df.columns = [
        clean_column_name(column)
        for column in df.columns
    ]

    aliases = {

        "order_date": [
            "order_date",
            "date",
            "orderdate",
            "sales_date"
        ],

        "customer_id": [
            "customer_id",
            "customerid",
            "customer",
            "customer_code"
        ],

        "product_name": [
            "product_name",
            "product",
            "productname",
            "item"
        ],

        "category": [
            "category",
            "product_category",
            "category_name"
        ],

        "region": [
            "region",
            "area",
            "territory"
        ],

        "quantity": [
            "quantity",
            "qty",
            "units",
            "units_sold"
        ],

        "sales": [
            "sales",
            "revenue",
            "amount",
            "total_sales"
        ],

        "profit": [
            "profit",
            "net_profit",
            "profit_amount"
        ]
    }

    rename_map = {}

    for standard_name, options in aliases.items():

        for option in options:

            if option in df.columns:

                rename_map[option] = standard_name

                break

    df = df.rename(
        columns=rename_map
    )

    return df


# -----------------------------
# Data Cleaning
# -----------------------------
def clean_data(df):

    df = standardize_columns(df)

    # Remove duplicates
    duplicates_removed = int(
        df.duplicated().sum()
    )

    df = df.drop_duplicates().copy()

    # Convert date
    if "order_date" in df.columns:

        df["order_date"] = pd.to_datetime(
            df["order_date"],
            errors="coerce"
        )

    # Convert numeric columns
    for column in [
        "quantity",
        "sales",
        "profit"
    ]:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Clean text columns
    for column in [
        "customer_id",
        "product_name",
        "category",
        "region"
    ]:

        if column in df.columns:

            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
            )

    # Standardize category
    if "category" in df.columns:

        df["category"] = (
            df["category"]
            .str.title()
        )

    # Standardize region
    if "region" in df.columns:

        df["region"] = (
            df["region"]
            .str.title()
        )

    # Fill missing numeric values
    for column in [
        "quantity",
        "sales",
        "profit"
    ]:

        if column in df.columns:

            median_value = df[column].median()

            df[column] = df[column].fillna(
                median_value
            )

    # Fill missing text
    for column in [
        "customer_id",
        "product_name",
        "category",
        "region"
    ]:

        if column in df.columns:

            df[column] = df[column].fillna(
                "Unknown"
            )

    # Remove invalid dates
    if "order_date" in df.columns:

        df = df.dropna(
            subset=["order_date"]
        )

    return (
        df,
        duplicates_removed
    )


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=[
        "csv",
        "xlsx",
        "xls"
    ]
)


# -----------------------------
# Load Dataset
# -----------------------------
if uploaded_file is not None:

    try:

        if uploaded_file.name.endswith(".csv"):

            raw_df = pd.read_csv(
                uploaded_file
            )

        else:

            raw_df = pd.read_excel(
                uploaded_file
            )

        source = uploaded_file.name

    except Exception as error:

        st.error(
            f"Error reading file: {error}"
        )

        st.stop()

else:

    raw_df = generate_demo_data()

    source = "Built-in Demo Dataset"


# -----------------------------
# Clean Dataset
# -----------------------------
cleaned_df, duplicates_removed = clean_data(
    raw_df
)


st.info(
    f"Data Source: **{source}**  |  "
    f"Raw Rows: **{len(raw_df):,}**  |  "
    f"Cleaned Rows: **{len(cleaned_df):,}**"
)


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📋 Overview",
        "🧹 Data Cleaning",
        "📈 EDA & Visualizations",
        "💡 Business Insights",
        "⬇️ Downloads"
    ]
)


# =====================================================
# TAB 1
# =====================================================

with tab1:

    st.subheader(
        "📋 Dataset Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    # Total Records
    col1.metric(
        "Total Records",
        f"{len(cleaned_df):,}"
    )

    # Total Sales
    if "sales" in cleaned_df.columns:

        col2.metric(
            "Total Sales",
            f"₹{cleaned_df['sales'].sum():,.2f}"
        )

    # Total Profit
    if "profit" in cleaned_df.columns:

        col3.metric(
            "Total Profit",
            f"₹{cleaned_df['profit'].sum():,.2f}"
        )

    # Total Quantity
    if "quantity" in cleaned_df.columns:

        col4.metric(
            "Total Quantity",
            f"{cleaned_df['quantity'].sum():,.0f}"
        )

    st.write(
        "### 👀 Data Preview"
    )

    st.dataframe(
        cleaned_df.head(20),
        use_container_width=True
    )

    st.write(
        "### 📊 Statistical Summary"
    )

    st.dataframe(
        cleaned_df.describe(
            include="all"
        ).transpose(),
        use_container_width=True
    )


# =====================================================
# TAB 2
# =====================================================

with tab2:

    st.subheader(
        "🧹 Data Cleaning & Quality"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Duplicate Rows Removed",
        duplicates_removed
    )

    col2.metric(
        "Missing Values Before",
        int(
            raw_df.isna()
            .sum()
            .sum()
        )
    )

    col3.metric(
        "Missing Values After",
        int(
            cleaned_df.isna()
            .sum()
            .sum()
        )
    )

    st.write(
        "### Missing Values Comparison"
    )

    missing_table = pd.DataFrame({

        "Before Cleaning":
            raw_df.isna().sum(),

        "After Cleaning":
            cleaned_df.isna().sum()
    })

    st.dataframe(
        missing_table,
        use_container_width=True
    )

    st.write(
        "### Cleaning Operations Performed"
    )

    st.markdown(
        """
        ✅ Removed duplicate records  
        
        ✅ Standardized column names  
        
        ✅ Converted dates to datetime format  
        
        ✅ Converted numeric columns correctly  
        
        ✅ Removed extra spaces from text  
        
        ✅ Standardized category names  
        
        ✅ Standardized region names  
        
        ✅ Filled missing numeric values using median  
        
        ✅ Filled missing text values with `Unknown`  
        
        ✅ Removed invalid dates
        """
    )


# =====================================================
# TAB 3
# =====================================================

with tab3:

    st.subheader(
        "📈 Exploratory Data Analysis"
    )

    required_columns = [
        "sales",
        "profit",
        "quantity"
    ]

    if all(
        column in cleaned_df.columns
        for column in required_columns
    ):

        # -------------------------------------
        # Category Analysis
        # -------------------------------------

        if "category" in cleaned_df.columns:

            category_summary = (

                cleaned_df
                .groupby("category")
                [["sales", "profit", "quantity"]]
                .sum()
                .sort_values(
                    "sales",
                    ascending=False
                )
            )

            st.write(
                "### 📊 Sales by Category"
            )

            fig, ax = plt.subplots(
                figsize=(9, 5)
            )

            sns.barplot(
                x=category_summary.index,
                y=category_summary["sales"],
                ax=ax
            )

            ax.set_xlabel(
                "Category"
            )

            ax.set_ylabel(
                "Sales"
            )

            ax.set_title(
                "Total Sales by Category"
            )

            plt.xticks(
                rotation=20
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

        # -------------------------------------
        # Region Analysis
        # -------------------------------------

        if "region" in cleaned_df.columns:

            region_summary = (

                cleaned_df
                .groupby("region")
                [["sales", "profit", "quantity"]]
                .sum()
                .sort_values(
                    "sales",
                    ascending=False
                )
            )

            st.write(
                "### 🌍 Sales by Region"
            )

            fig, ax = plt.subplots(
                figsize=(9, 5)
            )

            sns.barplot(
                x=region_summary.index,
                y=region_summary["sales"],
                ax=ax
            )

            ax.set_xlabel(
                "Region"
            )

            ax.set_ylabel(
                "Sales"
            )

            ax.set_title(
                "Sales by Region"
            )

            plt.xticks(
                rotation=20
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

        # -------------------------------------
        # Monthly Sales Trend
        # -------------------------------------

        if "order_date" in cleaned_df.columns:

            monthly = (

                cleaned_df
                .set_index("order_date")
                .resample("ME")
                [["sales", "profit"]]
                .sum()
                .reset_index()
            )

            st.write(
                "### 📅 Monthly Sales Trend"
            )

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            ax.plot(
                monthly["order_date"],
                monthly["sales"],
                marker="o"
            )

            ax.set_xlabel(
                "Month"
            )

            ax.set_ylabel(
                "Sales"
            )

            ax.set_title(
                "Monthly Sales Trend"
            )

            ax.grid(
                True,
                alpha=0.3
            )

            plt.xticks(
                rotation=45
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

        # -------------------------------------
        # Sales vs Profit
        # -------------------------------------

        st.write(
            "### 💰 Sales vs Profit"
        )

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        sns.scatterplot(
            data=cleaned_df,
            x="sales",
            y="profit",
            hue=(
                "category"
                if "category"
                in cleaned_df.columns
                else None
            ),
            ax=ax
        )

        ax.set_title(
            "Sales vs Profit Relationship"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        # -------------------------------------
        # Correlation Heatmap
        # -------------------------------------

        st.write(
            "### 🔥 Correlation Heatmap"
        )

        numeric_columns = (
            cleaned_df
            .select_dtypes(
                include=np.number
            )
            .columns
        )

        correlation = (
            cleaned_df[
                numeric_columns
            ].corr()
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.heatmap(
            correlation,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax
        )

        ax.set_title(
            "Correlation Between Numeric Variables"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        # -------------------------------------
        # Top Products
        # -------------------------------------

        if "product_name" in cleaned_df.columns:

            top_products = (

                cleaned_df
                .groupby("product_name")
                ["sales"]
                .sum()
                .sort_values(
                    ascending=False
                )
                .head(10)
            )

            st.write(
                "### 🏆 Top 10 Products by Sales"
            )

            st.dataframe(
                top_products
                .rename(
                    "Total Sales"
                )
                .reset_index(),
                use_container_width=True
            )

    else:

        st.warning(
            "The dataset must contain "
            "Sales, Profit and Quantity "
            "columns for complete EDA."
        )


# =====================================================
# TAB 4
# =====================================================

with tab4:

    st.subheader(
        "💡 Business Insights"
    )

    if all(
        column in cleaned_df.columns
        for column in [
            "sales",
            "profit",
            "quantity"
        ]
    ):

        total_sales = (
            cleaned_df["sales"].sum()
        )

        total_profit = (
            cleaned_df["profit"].sum()
        )

        total_quantity = (
            cleaned_df["quantity"].sum()
        )

        profit_margin = (

            total_profit /
            total_sales *
            100

            if total_sales != 0
            else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Average Sales",
            f"₹{cleaned_df['sales'].mean():,.2f}"
        )

        col2.metric(
            "Average Profit",
            f"₹{cleaned_df['profit'].mean():,.2f}"
        )

        col3.metric(
            "Average Quantity",
            f"{cleaned_df['quantity'].mean():.2f}"
        )

        col4.metric(
            "Profit Margin",
            f"{profit_margin:.2f}%"
        )

        # Category
        if "category" in cleaned_df.columns:

            category_sales = (
                cleaned_df
                .groupby("category")
                ["sales"]
                .sum()
            )

            category_profit = (
                cleaned_df
                .groupby("category")
                ["profit"]
                .sum()
            )

            top_category = (
                category_sales.idxmax()
            )

            top_profit_category = (
                category_profit.idxmax()
            )

        else:

            top_category = "N/A"
            top_profit_category = "N/A"

        # Region
        if "region" in cleaned_df.columns:

            region_sales = (
                cleaned_df
                .groupby("region")
                ["sales"]
                .sum()
            )

            region_profit = (
                cleaned_df
                .groupby("region")
                ["profit"]
                .sum()
            )

            top_region = (
                region_sales.idxmax()
            )

            top_profit_region = (
                region_profit.idxmax()
            )

        else:

            top_region = "N/A"
            top_profit_region = "N/A"

        # Product
        if "product_name" in cleaned_df.columns:

            product_sales = (
                cleaned_df
                .groupby("product_name")
                ["sales"]
                .sum()
            )

            top_product = (
                product_sales.idxmax()
            )

        else:

            top_product = "N/A"

        st.write(
            "### 🔎 Key Observations"
        )

        st.markdown(
            f"""
            **1. Top Sales Category:**  
            {top_category}

            **2. Top Profit Category:**  
            {top_profit_category}

            **3. Top Sales Region:**  
            {top_region}

            **4. Top Profit Region:**  
            {top_profit_region}

            **5. Top Product by Sales:**  
            {top_product}

            **6. Overall Profit Margin:**  
            {profit_margin:.2f}%

            **7. Sales Trend:**  
            The monthly sales chart can be used to identify
            increasing and decreasing sales periods.

            **8. Sales and Profit Relationship:**  
            The scatter plot helps understand the relationship
            between sales and profit.

            **9. Correlation Analysis:**  
            The heatmap shows the relationship between
            numerical variables.
            """
        )

        st.info(
            "Business insights are automatically calculated "
            "from the dataset being analyzed."
        )

    else:

        st.warning(
            "Upload a complete retail sales dataset "
            "to generate business insights."
        )


# =====================================================
# TAB 5
# =====================================================

with tab5:

    st.subheader(
        "⬇️ Download Results"
    )

    # Cleaned dataset
    csv_data = (
        cleaned_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label="⬇️ Download Cleaned Dataset",
        data=csv_data,
        file_name="cleaned_retail_sales.csv",
        mime="text/csv"
    )

    # Analysis report
    if all(
        column in cleaned_df.columns
        for column in [
            "sales",
            "profit",
            "quantity"
        ]
    ):

        report = f"""
RETAIL SALES DATA ANALYSIS REPORT
=================================

Total Records:
{len(cleaned_df)}

Total Sales:
{cleaned_df['sales'].sum():.2f}

Total Profit:
{cleaned_df['profit'].sum():.2f}

Total Quantity:
{cleaned_df['quantity'].sum():.0f}

Average Sales:
{cleaned_df['sales'].mean():.2f}

Average Profit:
{cleaned_df['profit'].mean():.2f}

Profit Margin:
{(cleaned_df['profit'].sum() / cleaned_df['sales'].sum() * 100):.2f}%

Duplicate Rows Removed:
{duplicates_removed}

Missing Values Before Cleaning:
{int(raw_df.isna().sum().sum())}

Missing Values After Cleaning:
{int(cleaned_df.isna().sum().sum())}
"""

        st.download_button(
            label="⬇️ Download Analysis Report",
            data=report,
            file_name="retail_sales_analysis_report.txt",
            mime="text/plain"
        )


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.caption(
    "Retail Sales Data Analysis & Business Insights | Internship Project"
)