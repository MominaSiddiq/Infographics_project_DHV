#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 19:59:00 2023

@author: macbook
"""

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.dates as mdates



def read_and_clean_data(filename):
    """
    This function reads data into a pandas dataframe take transpose and 
    clean the data.

    Parameters
    ----------
    filename : csv
       take the filenames as an argument.

    Returns
    -------
    two dataframes.

    """

    # read the data from the file
    data = pd.read_csv(filename, skiprows = 4, index_col = False)

    # cleaning the data(removing empty rows etc.)
    data.dropna(axis = 0, how = 'all', inplace = True)

    # remove emty columns
    data.dropna(axis = 1, how = 'all', inplace = True)

    # Drop the unnecessary columns in the data
    data.drop(columns = ['Country Code', 'Indicator Name',
              'Indicator Code'], inplace = True)
    
    # taking the transpose
    years_col_df = data.T

    # setting the header
    years_col_df.columns = years_col_df.iloc[0]
    years_col_df = years_col_df[1:]

    # reset index for making years as columns
    years_col_df = years_col_df.reset_index()
    years_col_df = years_col_df.rename(columns = {'index': 'Year'})

    # setting years as index
    years_col_df.set_index('Year', inplace = True)
    
    # removing empty rows
    years_col_df.dropna(axis = 0, how = 'all', inplace = True)

    # removing empty columns
    years_col_df.dropna(axis = 1, how = 'all', inplace = True)

    # Removeing any unnecessary columns in the transpose of data
    years_col_df = years_col_df.loc[:, ~years_col_df.columns.duplicated()]

    # Removing any duplicated rows
    years_col_df = years_col_df[~years_col_df.index.duplicated(keep='first')]
    
    # taking the transpose again for country column 
    country_col_df = years_col_df.T
    
    # Reset index for making countries as columns
    country_col_df = country_col_df.reset_index().rename(columns={'index': 'Country'})
    
    return country_col_df, years_col_df


# Filtering all the indicators data for selected data 
def filtered_data(df, countries, start_year, end_year):
    """
    filtering data on selective years and countries for all the indicators 

    Parameters
    ----------
    data : python dataframe

    Returns
    -------
    filtered data.

    """
    # Ensure the DataFrame has an index named 'Country' or reset it if necessary
    if df.index.name != 'Country Name':
        if 'Country Name' in df.columns:
            df.set_index('Country Name', inplace=True)
        else:
            print("Country Name column not found.")
            return None
        
    # Filter for the selected countries
    filtered_df = df[df.index.isin(countries)]

    # Convert years to string if necessary
    start_year, end_year = str(start_year), str(end_year)

    # Filter for the range of years, ensuring they are in string format
    filtered_df = filtered_df.loc[:, start_year:end_year]

    return filtered_df


# Obtaining the summary statistics of data by the describe method
def summary_statistics(data):
    """
    applies describe method on different indicators.

    Parameters
    ----------
    data : pandas dataframe
       The numerical data to analyze

    Returns
    -------
    summary_stats
        summary of selected data.

    """
    summary = {}
    for key, df in data.items():
        summary[key] = df.describe()
    
    return summary  


def plot_area_graph_for_emissions(filtered_dfs, ghg_key):
    """
    Plot an area graph for greenhouse gas emissions from the filtered_dfs dictionary.

    Parameters:
    ----------
    filtered_dfs : dict
        Dictionary of filtered dataframes with emissions data.
        
    ghg_key : str
        Key for the greenhouse gas emissions dataframe.
        
     Returns
     -------
     None.

    """
    # Extract the greenhouse gas emissions dataframe
    ghg_df = filtered_dfs[ghg_key]
    
   # Double-check the structure of the DataFrame
    if 'Country Name' in ghg_df.columns:
        # If 'Country Name' is a column, set it as the index
        ghg_df = ghg_df.set_index('Country Name')
        
    elif ghg_df.index.name != 'Country Name' and 'Country Name' not in ghg_df.index:
        # If 'Country Name' is neither a column nor the index name, raise an error
        raise KeyError("The DataFrame does not have 'Country Name' as a column or index name.")
    
    # Transpose the dataframe to have countries as columns and years as rows
    ghg_transposed = ghg_df.transpose()
    
    # Convert the year indices to integer if they are not already
    ghg_transposed.index = ghg_transposed.index.map(int)

    # Sum up all the emissions for each year to get a total
    total_emissions_by_year = ghg_transposed.sum(axis=1)
    
    # Styling parameters
    plt.style.use('seaborn-whitegrid')
    font = {'family': 'sans-serif', 'weight': 'normal', 'size': 10}
    plt.rc('font', **font)

    # Plot the area graph
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(total_emissions_by_year.index, total_emissions_by_year, color='#a9d18e', label='Total Emissions')

    # Annotate peak points
    peaks = total_emissions_by_year[(total_emissions_by_year.shift(1) < total_emissions_by_year) &
                                    (total_emissions_by_year.shift(-1) < total_emissions_by_year)]
    # Annotate with arrows
    for year, value in peaks.items():
        ax.annotate(f'{value:.2f}', xy=(year, value), xytext=(0, 20),
            textcoords='offset points', ha='center',
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            bbox=dict(boxstyle="round,pad=0.3", fc="lightgray", ec="black", alpha=0.5))
        
    # Customize the plot to match the sample report style
    ax.set_title('Total Greenhouse Gas Emissions Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Total Emissions', fontsize=12)
    ax.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust layout to fit labels
    plt.show()
    
    
def plot_renewable_energy_consumption(filtered_dfs, renewable_key):
    """
    Plot a horizontal bar graph for renewable energy consumption with a gap of 5 years.

    Parameters:
    ----------
    filtered_dfs : dict
        Dictionary of filtered dataframes with emissions data.
    renewable_key : str
        Key for the renewable energy consumption dataframe.
        
    Returns
    -------
    None.

    """
   # Extract the renewable energy consumption dataframe
    renewable_df = filtered_dfs[renewable_key]

    # Check if 'Country Name' is a column; if so, set it as the index
    if 'Country Name' in renewable_df.columns:
        renewable_df = renewable_df.set_index('Country Name')
    elif renewable_df.index.name != 'Country Name':
        raise KeyError("The DataFrame does not have 'Country Name' as a column or index name.")
    
    # Transpose the dataframe to have countries as columns and years as rows
    renewable_transposed = renewable_df.transpose()

    # Ensure that the index is of type integer
    renewable_transposed.index = renewable_transposed.index.map(int)

    # Select only the years that are multiples of 5
    years_to_plot = [year for year in renewable_transposed.index if year % 5 == 0]
    renewable_filtered = renewable_transposed[renewable_transposed.index.isin(years_to_plot)]

    # Sum up all the renewable energy consumption for each selected year to get a total
    total_consumption_by_year = renewable_filtered.sum(axis=1)

    # Plot the horizontal bar graph
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(total_consumption_by_year.index, total_consumption_by_year, color='#5b9bd5')

    # Define the style for the annotation boxes
    bbox_props = dict(boxstyle="square,pad=0.1", fc="grey", alpha=0.5, lw=0)

    # Annotate each bar with the year and the value
    for bar in bars:
        year = bar.get_y() + bar.get_height() / 2
        value = bar.get_width()
        ax.text(value, year, f'{int(year)}: {value:.2f} %',
                va='center', ha='left', color='black', bbox=bbox_props)

    # Customize the plot
    ax.set_title('Total Renewable Energy Consumption Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Total Consumption', fontsize=12)
    ax.set_ylabel('Year', fontsize=12)
    plt.grid(axis='x', linestyle='--', linewidth=1.0, color='gray')
    plt.tight_layout()  # Adjust layout to fit labels
    plt.show()


def group_data_into_periods(data, start_year, end_year, period_length):
    """
    Group data into specified periods.

    Parameters:
    ----------
    data : DataFrame
        DataFrame containing the data to group.
    start_year : int
        The starting year of the data.
    end_year : int
        The ending year of the data.
    period_length : int
        The length of each period in years.

    Returns:
    -------
    DataFrame
        DataFrame with grouped data.
    """
    # Transpose the DataFrame to have years as rows
    df_transposed = data.transpose()

    # Ensure the DataFrame contains the specified range of years
    df_transposed = df_transposed.loc[str(start_year):str(end_year)]

    # Convert the index to integer for easier comparison
    df_transposed.index = df_transposed.index.astype(int)

    # Group data by periods
    grouped_data = {}
    for year in range(start_year, end_year + 1, period_length):
        period_end = min(year + period_length - 1, end_year)
        period_label = f'{year}-{period_end}'
        period_data = df_transposed.loc[year:period_end].sum()
        grouped_data[period_label] = period_data

    return pd.DataFrame(grouped_data)


def plot_urban_rural_population(urban_population, rural_population, start_year, end_year, period=10):
    """
    Plots multiple pie charts for urban vs rural population over different periods, each period spanning 10 years.
    Parameters:
    urban_population : dict
        A dictionary with years as keys and urban population as values.
    rural_population : dict
        A dictionary with years as keys and rural population as values.
    start_year : int
        The starting year for the data.
    end_year : int
        The ending year for the data.
    period : int
        The number of years in each period to group the data.
    """
    # Determine the number of periods and create subplot for each pie chart
    num_periods = (end_year - start_year + 1) // period
    fig, axs = plt.subplots(1, num_periods, figsize=(5 * num_periods, 5))
    
    # If only one pie chart is needed, put axs in a list to make it iterable
    if num_periods == 1:
        axs = [axs]

    # Loop through each period and create pie charts
    for i, period_start in enumerate(range(start_year, end_year, period)):
        period_end = period_start + period
        period_urban_pop = sum(urban_population.get(year, 0) for year in range(period_start, period_end))
        period_rural_pop = sum(rural_population.get(year, 0) for year in range(period_start, period_end))
        
        data = [period_urban_pop, period_rural_pop]
        labels = ['Urban Population', 'Rural Population']
        colors = ['#ff9999', '#66b3ff']
        explode = (0.1, 0)  # explode the first slice

        # Plot the pie chart
        axs[i].pie(data, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axs[i].set_title(f'{period_start} - {period_end - 1}')

    plt.suptitle('Urban vs Rural Population Distribution over 10-Year Periods')
    plt.tight_layout()
    plt.show()


def plot_urban_rural_pop(grouped_urban, grouped_rural):
    """
    Creates a horizontal stacked bar chart with grouped urban and rural population data,
    with annotations formatted to display values in millions.
    """
    # Aggregate the data for each period
    periods = grouped_urban.columns
    urban_data = grouped_urban.sum()
    rural_data = grouped_rural.sum()

    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    # Create the horizontal stacked bar chart
    bar_positions = np.arange(len(periods))
    ax.barh(bar_positions, urban_data, color='skyblue', edgecolor='black', label='Urban Population')
    ax.barh(bar_positions, rural_data, left=urban_data, color='lightgreen', edgecolor='black', label='Rural Population')

    # Format the annotations to display values in millions with one decimal place
    for idx, (urban, rural) in enumerate(zip(urban_data, rural_data)):
        ax.text(urban / 2, idx, f'{urban/1e6:.1f}M', va='center', ha='center', color='black')
        ax.text(urban + rural / 2, idx, f'{rural/1e6:.1f}M', va='center', ha='center', color='black')

    # Set the y-ticks to the periods
    ax.set_yticks(bar_positions)
    ax.set_yticklabels(periods)

    # Set labels and title with predefined colors
    ax.set_title('Urban vs Rural Population Worldwide', fontsize=14, fontweight='bold')
    ax.set_xlabel('Population (millions)', color='darkgreen')
    ax.set_ylabel('Period', color='darkgreen')

    # Add gridlines behind the bars
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)

    # Remove the spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add the legend
    ax.legend()

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
    

def main():
    """
    A main function calling other functions.

    Returns
    -------
    None.

    """
    # List of files
    filename = ["CO2_emission.csv", "greenhouse_gas_emission.csv", 
                "renewable_energy_consumption.csv", "urban_population.csv",
                "rural_population.csv", "agriculture_land.csv", "net_migration.csv"]
    
    # Process each file and save the transposed data
    transposed_files = []  # To keep track of the new transposed files
    
    for file in filename:
        # Process the data
        country_col_df, years_col_df = read_and_clean_data(file)

        # Create a new filename for the transposed data
        transposed_filename = file.replace('.csv', '_trans.csv')
        transposed_files.append(transposed_filename)

        # Save the transposed data
        country_col_df.to_csv(transposed_filename, index=False)
   
    # selecting countries
    selected_countries = ['Qatar', 'China', 'United Kingdom', 'Pakistan',
        'Netherlands', 'Portugal', 'United States', 'South Asia', 'Bangladesh',
        'Italy', 'Japan', 'Turkiye', 'Sri Lanka', 'New Zeeland', 'Oman',]
    
    # selecting years
    start_year = 1990
    end_year = 2022

    # List to store filtered DataFrames
    filtered_dfs = {}

    # Read and filter each transposed file
    for transposed_file in transposed_files:
        # Read the transposed data
        df = pd.read_csv(transposed_file)

        # Filter the data
        filtered_df = filtered_data(df, selected_countries, start_year, end_year)

        # Add the filtered DataFrame to the list
        filtered_dfs[transposed_file] = filtered_df
        
        # Add the filtered DataFrame to the dictionary
        if filtered_df is not None:
            filtered_dfs[transposed_file] = filtered_df
            print(f"Filtered data from {transposed_file} added to the list")
        else:
            print(f"Skipped {transposed_file} due to missing 'Country Name' column.")
            
    # Print the filtered data for each file in the dictionary
    for filename, filtered_df in filtered_dfs.items():
        print(f"Filtered data from {filename}:")
        print(filtered_df)
        print("\n")  
        
    # Call the function with the filtered_dfs dictionary to get the summary
    summary_stats = summary_statistics(filtered_dfs)

    # Print the summary statistics for each filtered DataFrame
    for filename, stats in summary_stats.items():
        print(f"Summary Statistics for {filename}:")
        print(stats)
        print("\n")  
        
    # plotting for greenhouse gases emission
    ghg_key = 'greenhouse_gas_emission_trans.csv'  
    plot_area_graph_for_emissions(filtered_dfs, ghg_key)
    
    # plotting for renewable energy consumption 
    renewable_key = 'renewable_energy_consumption_trans.csv'
    plot_renewable_energy_consumption(filtered_dfs, renewable_key)
    
    # grouping the data into periods for the stacked bar graph of urban vs rural population 
    urban_pop = filtered_dfs['urban_population_trans.csv']
    rural_pop = filtered_dfs['rural_population_trans.csv']
    
    #grouping in periods
    group_urban_pop = group_data_into_periods(urban_pop, start_year, end_year, 5)
    group_rural_pop = group_data_into_periods(rural_pop, start_year, end_year, 5)
    
    # plot for urban vs rural population 
    plot_urban_rural_pop(group_urban_pop, group_rural_pop)
    
    # plot pie charts for urban vs rural population 
    plot_urban_rural_population(urban_pop, rural_pop, start_year, end_year)
    
    
    

if __name__ == "__main__":
    main()
    