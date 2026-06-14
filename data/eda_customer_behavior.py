# ============================================================
# Exploratory Data Analysis on Customer Behavior & Churn
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data"
SNAPSHOT_DATE = "2025-09-30"

sns.set(style="whitegrid")


# ------------------------------------------------------------
# Load all datasets
# ------------------------------------------------------------
def load_data():
    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    churn = pd.read_csv(os.path.join(DATA_DIR, "churn_labels.csv"))
    orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"), parse_dates=["order_date"])
    tickets = pd.read_csv(os.path.join(DATA_DIR, "support_tickets.csv"))
    web = pd.read_csv(os.path.join(DATA_DIR, "web_events_snapshot.csv"))

    # Filter orders by snapshot date
    orders = orders[orders["order_date"] >= SNAPSHOT_DATE]

    return customers, churn, orders, tickets, web


# ------------------------------------------------------------
# Missing Values Summary
# ------------------------------------------------------------
def missing_summary(df, name):
    print(f"\n=== Missing Values: {name} ===")
    print(df.isna().sum())


# ------------------------------------------------------------
# Customer Behavior Analysis
# ------------------------------------------------------------
def analyze_customer_behavior(customers):
    print("\n=== Customer Behavior ===")
    print(customers.describe(include="all"))

    # Loyalty tier distribution
    plt.figure(figsize=(6,4))
    customers["loyalty_tier"].value_counts().plot(kind="bar", color="skyblue")
    plt.title("Loyalty Tier Distribution")
    plt.xlabel("Tier")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Skin type distribution
    plt.figure(figsize=(6,4))
    customers["skin_type"].value_counts().plot(kind="bar", color="salmon")
    plt.title("Skin Type Distribution")
    plt.xlabel("Skin Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Order Patterns
# ------------------------------------------------------------
def analyze_orders(orders):
    print("\n=== Order Patterns ===")
    print(orders.describe())

    # Monthly order volume
    orders["month"] = orders["order_date"].dt.to_period("M")
    monthly_orders = orders.groupby("month")["order_id"].count()

    plt.figure(figsize=(10,4))
    monthly_orders.plot(kind="line", marker="o")
    plt.title("Monthly Order Volume")
    plt.xlabel("Month")
    plt.ylabel("Orders")
    plt.tight_layout()
    plt.show()

    # Gross amount distribution
    plt.figure(figsize=(6,4))
    sns.histplot(orders["gross_amount"], bins=40, kde=True)
    plt.title("Gross Amount Distribution")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Support Ticket Issues
# ------------------------------------------------------------
def analyze_tickets(tickets):
    print("\n=== Support Ticket Issues ===")
    print(tickets["issue_type"].value_counts())

    plt.figure(figsize=(8,4))
    tickets["issue_type"].value_counts().plot(kind="bar", color="purple")
    plt.title("Support Ticket Types")
    plt.xlabel("Issue Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Sentiment distribution
    plt.figure(figsize=(6,4))
    sns.histplot(tickets["sentiment_score"], bins=30, kde=True, color="green")
    plt.title("Ticket Sentiment Distribution")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Web/App Engagement
# ------------------------------------------------------------
def analyze_web(web):
    print("\n=== Web/App Engagement ===")
    engagement_cols = [
        "sessions_30d", "product_views_30d", "cart_adds_30d",
        "wishlist_adds_30d", "email_opens_30d", "campaign_clicks_30d"
    ]

    print(web[engagement_cols].describe())

    # Heatmap of correlations
    plt.figure(figsize=(8,6))
    sns.heatmap(web[engagement_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Engagement Feature Correlation")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Churn Distribution
# ------------------------------------------------------------
def analyze_churn(customers, churn):
    print("\n=== Churn Distribution ===")

    df = customers.merge(churn, on="customer_id", how="left")

    plt.figure(figsize=(6,4))
    df["churn_flag"].value_counts().plot(kind="bar", color=["red", "blue"])
    plt.title("Churn Distribution")
    plt.xlabel("Churn Flag")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Churn by loyalty tier
    plt.figure(figsize=(8,4))
    sns.countplot(data=df, x="loyalty_tier", hue="churn_flag")
    plt.title("Churn by Loyalty Tier")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    customers, churn, orders, tickets, web = load_data()

    # Missing values
    missing_summary(customers, "customers")
    missing_summary(churn, "churn_labels")
    missing_summary(orders, "orders")
    missing_summary(tickets, "support_tickets")
    missing_summary(web, "web_events")

    analyze_customer_behavior(customers)
    analyze_orders(orders)
    analyze_tickets(tickets)
    analyze_web(web)
    analyze_churn(customers, churn)


if __name__ == "__main__":
    main()
