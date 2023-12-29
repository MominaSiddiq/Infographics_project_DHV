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


def plot_urban_rural_population(urban_data, rural_data):
    """
    Plots multiple pie charts for urban vs rural population over different periods.
    
    Parameters:
    urban_data : DataFrame
        DataFrame with periods as columns and aggregated urban population as values.
    rural_data : DataFrame
        DataFrame with periods as columns and aggregated rural population as values.
    """
   # Determine the number of periods and create subplot for each pie chart
    num_periods = len(urban_data.columns)
    fig, axs = plt.subplots(1, num_periods, figsize=(num_periods * 5, 5))  # Adjust width for each subplot

    # Ensure axs is iterable
    if num_periods == 1:
        axs = [axs]

    # Define the color scheme
    colors = ['#5B9BD5', '#FFC000']  # Blue for urban, Gold for rural

    # Loop through each period and create pie charts
    for i, period in enumerate(urban_data.columns):
        period_urban_pop = urban_data[period].sum()
        period_rural_pop = rural_data[period].sum()

        data = [period_urban_pop, period_rural_pop]
        explode = (0.1, 0)  # Explode the first slice (Urban)

        # Plot the pie chart with the desired styling
        wedges, texts, autotexts = axs[i].pie(data, explode=explode, colors=colors, 
                                              autopct='%1.1f%%', startangle=90, pctdistance=0.75)
        # Increase the font size of the percentages and set their color to black
        plt.setp(autotexts, size=10, weight="bold", color='black')

        # Draw a circle at the center to achieve the donut shape
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        axs[i].add_artist(centre_circle)

        # Set the title for each pie chart
        axs[i].set_title(f"{period}", color='black', size=14)

    # Adjust the layout
    plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust the spacing between the subplots

    # Create a legend for the colors used
    fig.legend(wedges, ['Urban Population', 'Rural Population'],
                loc='lower center', bbox_to_anchor=(0.5, 0), ncol=num_periods)

    plt.suptitle('Urban vs Rural Population Distribution Over Periods', color='black', size=16)
    plt.show()
    
    
def plot_stacked_bar(data, title, x_label, y_label, legend_title):
    """
    Creates a stacked bar chart for grouped data.
    
    Parameters:
    data : DataFrame
        DataFrame containing the data to plot, indexed by period with categories as columns.
    title : str
        The title of the plot.
    x_label, y_label : str
        Labels for the x and y axes.
    legend_title : str
        The title of the legend.
    """
    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    # Initialize the bottom position for the stacked bars
    bottom = np.zeros(len(data))

    # Create the stacked bar chart
    for column in data.columns:
        ax.bar(data.index, data[column], bottom=bottom, label=column)
        bottom += data[column].values

    # Annotate the bars with the total for each period
    for idx, period in enumerate(data.index):
        total = bottom[idx]
        ax.text(idx, total, f'{total:.1f}', va='bottom', ha='center', color='black')

    # Set labels and title
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add the legend
    ax.legend(title=legend_title)

    # Rotate the x-axis labels if necessary
    plt.xticks(rotation=45)

    # Adjust layout and display the plot
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
    
    
def plot_migration_trends(migration_data):
    """
    Creates a time series plot for net migration data for each country.

    Parameters:
    migration_data : DataFrame
        DataFrame containing the migration data with countries as rows and years as columns.
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(15, 10))

    # Plot time series for each country
    years = migration_data.columns.astype(int)
    for country in migration_data.index:
        # Apply a simple rolling mean to smooth the lines
        rolling_mean = migration_data.loc[country].rolling(window=3, min_periods=1).mean()
        ax.plot(years, rolling_mean, label=country)

    # Add a zero-line for reference
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    # Formatting
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Net Migration', fontsize=14)
    ax.set_title('Net Migration Trends Over Time', fontsize=16)
    ax.grid(True)
    
    # Scale y-axis to log for better visualization
    ax.set_yscale('symlog')

    # Improve the legend
    handles, labels = ax.get_legend_handles_labels()
    # Sort both handles and labels based on labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    # Add legend outside of the plot to avoid blocking the view
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), title='Country Name')

    # Show plot with tight layout
    plt.tight_layout(rect=[0, 0, 0.85, 1])
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
    agriculture_data = filtered_dfs['agriculture_land_trans.csv']
    migration_data = filtered_dfs['net_migration_trans.csv']
    
    #grouping in periods
    group_urban_pop = group_data_into_periods(urban_pop, start_year, end_year, 10)
    group_rural_pop = group_data_into_periods(rural_pop, start_year, end_year, 10)
    
    # plot for urban vs rural population 
    plot_urban_rural_pop(group_urban_pop, group_rural_pop)
    
    # plot pie charts for urban vs rural population 
    plot_urban_rural_population(group_urban_pop, group_rural_pop)
    
    # Grouping agricultural land data
    agricultural_land_grouped = group_data_into_periods(agriculture_data, 1990, 2022, 10)

    # Grouping migration data
    #migration_data_grouped = group_data_into_periods(migration_data, 1990, 2022, 10)
    
    # For Agricultural Land Data
    plot_stacked_bar(agricultural_land_grouped, 
                     'Agricultural Land Use Over Time', 
                     'Land Area', 
                     'Period', 
                     'Type of Land Use')

    # For Migration Data
    plot_migration_trends(migration_data)
        


if __name__ == "__main__":
    main()
    